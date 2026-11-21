"""Local Privacy Perimeter: Automatic EN/DE Language Detection + Bidirectional PII/PHI Vault.

Runs 100% locally BEFORE any cloud LLM API call so raw PII/PHI never leaves the network.
"""

from __future__ import annotations

import re
from typing import Any

from lingua import Language, LanguageDetectorBuilder

# Singleton Lingua detector constrained to English and German for high accuracy & low latency
_DETECTOR = (
    LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.GERMAN)
    .with_preloaded_language_models()
    .build()
)

# High-precision regexes ordered from most specific to most general
_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "EMAIL",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ),
    (
        "IBAN",
        re.compile(r"\b(?:DE|AT|CH|GB|FR|NL|ES|IT)[0-9]{2}(?:[\s-]?[0-9A-Z]{4}){3,6}(?:[\s-]?[0-9A-Z]{1,4})?\b"),
    ),
    (
        "IPV4",
        re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"),
    ),
    (
        "IPV6",
        re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){3,7}[0-9a-fA-F]{1,4}\b"),
    ),
    (
        "CARD",
        re.compile(r"\b(?:\d{4}[\s-]){3}\d{4}\b"),
    ),
    (
        "MEDICAL_ID",
        re.compile(r"\b(?:MRN|PAT|PATIENT|EMR|HIPAA)[-_#: ]?\d{4,12}\b", re.IGNORECASE),
    ),
    (
        "ACCOUNT_ID",
        re.compile(r"\b(?:ACC|ACCT|CUST|KND|KD|INV|CONTRACT|SSN)[-_#: ]?(?=[A-Z0-9]*\d)[A-Z0-9]{4,14}\b", re.IGNORECASE),
    ),
    (
        "PHONE",
        re.compile(
            r"(?<!\w)(?:\+(?:1|49|43|41|44)[\s.-]?(?:\(0\)[\s.-]?)?(?:\d[\s.-]?){7,13}\d|\(\d{3}\)[\s.-]?\d{3}[\s.-]?\d{4}|\b\d{3}[-.]\d{3}[-.]\d{4}\b|\b0\d{2,4}[\s/-]\d{5,9}\b)"
        ),
    ),
]


def detect_language(subject: str | None, body: str | None) -> tuple[str, float]:
    """Detect English ('en') or German ('de') from raw ticket text (null-subject safe)."""
    text = f"{(subject or '').strip()} {(body or '').strip()}".strip()
    if not text:
        return "en", 0.50

    confidences = _DETECTOR.compute_language_confidence_values(text)
    if not confidences:
        return "en", 0.50

    top = confidences[0]
    code = "de" if top.language == Language.GERMAN else "en"
    return code, round(float(top.value), 4)


def _mask_preview(entity_type: str, raw: str) -> str:
    raw_s = raw.strip()
    if entity_type == "EMAIL" and "@" in raw_s:
        local, dom = raw_s.split("@", 1)
        return f"{local[:1]}***@{dom}"
    if len(raw_s) <= 4:
        return "***"
    return f"{raw_s[:2]}***{raw_s[-2:]}"


def redact_pii(subject: str | None, body: str | None) -> dict[str, Any]:
    """Bidirectionally scrub PII/PHI into indexed placeholders before external API calls."""
    subj_clean = (subject or "").strip()
    body_clean = (body or "").strip()
    combined_original = f"Subject: {subj_clean}\nBody: {body_clean}" if subj_clean else body_clean

    counters: dict[str, int] = {}
    value_to_placeholder: dict[str, str] = {}
    placeholder_to_value: dict[str, str] = {}
    entities_found: list[dict[str, str]] = []

    def _scrub_string(text: str) -> str:
        if not text:
            return text
        out = text
        for ent_type, pattern in _PII_PATTERNS:
            def _repl(match: re.Match[str], etype: str = ent_type) -> str:
                raw_val = match.group(0)
                # Never double-redact an existing placeholder or ticket ID
                if raw_val.startswith("[") and raw_val.endswith("]"):
                    return raw_val
                if re.fullmatch(r"TKT-\d{5}", raw_val):
                    return raw_val
                if raw_val in value_to_placeholder:
                    return value_to_placeholder[raw_val]
                counters[etype] = counters.get(etype, 0) + 1
                placeholder = f"[{etype}_{counters[etype]}]"
                value_to_placeholder[raw_val] = placeholder
                placeholder_to_value[placeholder] = raw_val
                entities_found.append(
                    {
                        "type": etype,
                        "placeholder": placeholder,
                        "preview": _mask_preview(etype, raw_val),
                    }
                )
                return placeholder

            out = pattern.sub(_repl, out)
        return out

    redacted_subject = _scrub_string(subj_clean)
    redacted_body = _scrub_string(body_clean)
    combined_redacted = (
        f"Subject: {redacted_subject}\nBody: {redacted_body}"
        if subj_clean
        else redacted_body
    )

    return {
        "pii_detected": len(entities_found) > 0,
        "redaction_count": len(entities_found),
        "entities_found": entities_found,
        "original_subject": subj_clean,
        "original_body": body_clean,
        "redacted_subject": redacted_subject,
        "redacted_body": redacted_body,
        "original_text": combined_original,
        "redacted_text": combined_redacted,
        "vault_map": placeholder_to_value,
    }


def restore_pii(text: str, vault_map: dict[str, str]) -> str:
    """Restore placeholders locally if an operator explicitly requests re-hydration."""
    restored = text
    for placeholder, original in vault_map.items():
        restored = restored.replace(placeholder, original)
    return restored


def contains_raw_pii(text: str) -> bool:
    """Verification helper: returns True if any raw PII pattern remains in text."""
    for _, pat in _PII_PATTERNS:
        for m in pat.finditer(text or ""):
            val = m.group(0)
            if not re.fullmatch(r"TKT-\d{5}", val):
                return True
    return False
