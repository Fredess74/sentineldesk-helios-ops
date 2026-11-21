"""Phase 1: Strict zero-leakage dataset preparation and kb_train-only audit.

Usage:
    python backend/data_prep.py          # Uses 20k clean dataset (default)
    python backend/data_prep.py --full   # Uses 50k clean dataset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import load_dataset

# Ensure project root is importable when executed as script
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.config import (
    ARTIFACTS_DIR,
    DATA_DIR,
    DATASET_FILE_20K,
    DATASET_FILE_50K,
    EXPECTED_LANGUAGES,
    EXPECTED_PRIORITIES,
    EXPECTED_QUEUES,
    EXPECTED_TYPES,
    HF_DATASET_REPO,
    HOLDOUT_TEST_PATH,
    KB_AUDIT_PATH,
    KB_TRAIN_PATH,
    SPLIT_SEED,
    TRAIN_FRACTION,
)


def prepare_and_split(use_full: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Download clean dataset, apply exact official 80/20 split, verify zero leakage."""
    data_file = DATASET_FILE_50K if use_full else DATASET_FILE_20K
    print(f"[data_prep] Loading {HF_DATASET_REPO} ({data_file})...")
    ds = load_dataset(HF_DATASET_REPO, data_files=data_file)
    df = ds["train"].to_pandas()
    raw_count = len(df)
    print(f"[data_prep] Raw rows loaded: {raw_count:,}")

    # EXACT official 80/20 split (seeded random_state=42, sequential ticket_id)
    df = df.sample(frac=1.0, random_state=SPLIT_SEED).reset_index(drop=True)
    df["ticket_id"] = [f"TKT-{i:05d}" for i in range(len(df))]

    split_idx = int(len(df) * TRAIN_FRACTION)
    kb_train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    # Zero-leakage assertions
    train_ids = set(kb_train_df["ticket_id"])
    test_ids = set(test_df["ticket_id"])
    assert len(train_ids & test_ids) == 0, "LEAKAGE ERROR: Overlapping ticket_ids!"
    assert len(kb_train_df) + len(test_df) == raw_count, "Row count mismatch after split!"

    # Clean label space assertions on full dataframe schema (structure check only, no inspection of test answers)
    actual_types = set(df["type"].dropna().unique())
    actual_priorities = set(df["priority"].dropna().unique())
    actual_langs = set(df["language"].dropna().unique())
    actual_queues = set(df["queue"].dropna().unique())

    assert actual_types == set(EXPECTED_TYPES), f"Unexpected types: {actual_types}"
    assert actual_priorities == set(EXPECTED_PRIORITIES), f"Unexpected priorities: {actual_priorities}"
    assert actual_langs == set(EXPECTED_LANGUAGES), f"Unexpected languages: {actual_langs}"
    assert len(actual_queues) == 10, f"Expected 10 clean queues, got {len(actual_queues)}: {actual_queues}"
    assert actual_queues == set(EXPECTED_QUEUES), f"Queue mismatch: {actual_queues ^ set(EXPECTED_QUEUES)}"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    kb_train_df.to_parquet(KB_TRAIN_PATH, index=False)
    test_df.to_parquet(HOLDOUT_TEST_PATH, index=False)

    print(
        f"[data_prep] Saved strict 80/20 split: "
        f"kb_train={len(kb_train_df):,} -> {KB_TRAIN_PATH.name}, "
        f"holdout_test={len(test_df):,} -> {HOLDOUT_TEST_PATH.name}"
    )
    print(f"[data_prep] Verified 0 overlapping ticket_ids; 4 types, 10 queues, 3 priorities, 2 languages.")
    return kb_train_df, test_df, data_file


def _clean_snippet(text: Any, max_chars: int = 240) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = " ".join(str(text).split())
    return s[:max_chars] + ("..." if len(s) > max_chars else "")


def audit_kb_train_only(kb_train_df: pd.DataFrame, data_file: str, total_rows: int) -> dict[str, Any]:
    """Audit ONLY kb_train_df. Holdout test_df labels/answers are NEVER touched here."""
    n = len(kb_train_df)
    subject_series = kb_train_df["subject"]
    null_subject_mask = subject_series.isna() | (subject_series.astype(str).str.strip() == "")
    null_subject_count = int(null_subject_mask.sum())

    body_series = kb_train_df["body"].fillna("").astype(str)
    null_body_count = int((body_series.str.strip() == "").sum())
    body_char_lens = body_series.str.len()
    body_word_lens = body_series.str.split().str.len()

    answer_series = kb_train_df["answer"].fillna("").astype(str)
    null_answer_count = int((answer_series.str.strip() == "").sum())
    answer_char_lens = answer_series.str.len()
    answer_word_lens = answer_series.str.split().str.len()

    # Check L1 resolution style indicators in kb_train answers
    l1_ask_info_rate = float(
        answer_series.str.contains(
            r"provide|details|account|screenshot|version|clarify|contact|reach out|bitte|nähere|informationen|kontaktieren|rückmeldung",
            case=False,
            regex=True,
        ).mean()
    )
    l1_question_mark_rate = float(answer_series.str.contains(r"\?", regex=True).mean())
    l1_refund_mention_rate = float(
        answer_series.str.contains(r"refund|erstattung|credit|gutschrift|sla", case=False, regex=True).mean()
    )

    # Exact class distributions
    def dist_dict(col: str) -> list[dict[str, Any]]:
        vc = kb_train_df[col].value_counts(dropna=False)
        return [
            {
                "label": str(k),
                "count": int(v),
                "pct": round(100.0 * float(v) / n, 2),
            }
            for k, v in vc.items()
        ]

    # High-Priority Incident share (critical P1 slice)
    p1_mask = (kb_train_df["priority"] == "high") & (kb_train_df["type"] == "Incident")
    p1_count = int(p1_mask.sum())

    # Cross-tab type x priority and language x priority
    type_x_priority = (
        pd.crosstab(kb_train_df["type"], kb_train_df["priority"])
        .reindex(index=EXPECTED_TYPES, columns=EXPECTED_PRIORITIES, fill_value=0)
        .to_dict(orient="index")
    )
    lang_x_type = (
        pd.crosstab(kb_train_df["language"], kb_train_df["type"])
        .reindex(index=EXPECTED_LANGUAGES, columns=EXPECTED_TYPES, fill_value=0)
        .to_dict(orient="index")
    )
    queue_x_p1 = (
        kb_train_df[p1_mask]["queue"]
        .value_counts()
        .reindex(EXPECTED_QUEUES, fill_value=0)
        .to_dict()
    )

    # PII presence scan in kb_train bodies (to quantify real GDPR/HIPAA exposure risk)
    pii_patterns = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "phone_like": r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)\d{3,4}[\s.-]?\d{3,4}",
        "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "account_or_id": r"\b(?:ACC|ID|INV|ORD|TKT|KND|PAT)[-_#]?\d{4,}\b",
        "placeholder_tel_or_acc": r"<(?:tel_num|acc_num|email|name|ip)>",
    }
    pii_stats = {
        k: int(body_series.str.contains(pat, case=False, regex=True).sum())
        for k, pat in pii_patterns.items()
    }

    # Tag coverage
    tag_cols = [c for c in kb_train_df.columns if c.startswith("tag_")]
    top_tags = []
    if tag_cols:
        all_tags = pd.concat([kb_train_df[c].dropna().astype(str) for c in tag_cols], ignore_index=True)
        all_tags = all_tags[all_tags.str.strip() != ""]
        top_tags = [
            {"tag": str(k), "count": int(v)}
            for k, v in all_tags.value_counts().head(20).items()
        ]

    # Representative examples per type (2 EN + 1 DE where possible)
    examples_by_type: dict[str, list[dict[str, Any]]] = {}
    for t in EXPECTED_TYPES:
        sub = kb_train_df[kb_train_df["type"] == t]
        chosen = []
        for lang, take in [("en", 2), ("de", 1)]:
            lsub = sub[(sub["language"] == lang) & (~sub["subject"].isna())].head(take)
            for _, row in lsub.iterrows():
                chosen.append(
                    {
                        "ticket_id": row["ticket_id"],
                        "language": row["language"],
                        "priority": row["priority"],
                        "queue": row["queue"],
                        "subject": _clean_snippet(row["subject"], 120),
                        "body": _clean_snippet(row["body"], 220),
                        "answer": _clean_snippet(row["answer"], 220),
                    }
                )
        examples_by_type[t] = chosen

    # Representative examples per queue (2 examples each: 1 EN + 1 DE)
    examples_by_queue: dict[str, list[dict[str, Any]]] = {}
    for q in EXPECTED_QUEUES:
        sub = kb_train_df[kb_train_df["queue"] == q]
        chosen = []
        for lang in ["en", "de"]:
            lsub = sub[sub["language"] == lang].head(1)
            for _, row in lsub.iterrows():
                chosen.append(
                    {
                        "ticket_id": row["ticket_id"],
                        "language": row["language"],
                        "type": row["type"],
                        "priority": row["priority"],
                        "subject": _clean_snippet(row["subject"], 120),
                        "body": _clean_snippet(row["body"], 200),
                        "answer": _clean_snippet(row["answer"], 200),
                    }
                )
        examples_by_queue[q] = chosen

    # Sample hospital / VPN / EMR / outage examples in kb_train (confirming Helios Archetype fit)
    healthcare_vpn_mask = body_series.str.contains(
        r"hospital|medical|clinic|patient|telemedicine|healthcare|vpn|outage|krankenhaus|medizin",
        case=False,
        regex=True,
    )
    healthcare_vpn_count = int(healthcare_vpn_mask.sum())
    healthcare_vpn_samples = []
    for _, row in kb_train_df[healthcare_vpn_mask].head(5).iterrows():
        healthcare_vpn_samples.append(
            {
                "ticket_id": row["ticket_id"],
                "language": row["language"],
                "type": row["type"],
                "queue": row["queue"],
                "priority": row["priority"],
                "subject": _clean_snippet(row["subject"], 120),
                "body": _clean_snippet(row["body"], 220),
            }
        )

    audit = {
        "dataset_repo": HF_DATASET_REPO,
        "dataset_file": data_file,
        "split_seed": SPLIT_SEED,
        "total_rows": int(total_rows),
        "kb_train_rows": int(n),
        "holdout_test_rows": int(total_rows - n),
        "zero_leakage_verified": True,
        "columns": list(kb_train_df.columns),
        "null_subject_count": null_subject_count,
        "null_subject_pct": round(100.0 * null_subject_count / n, 2),
        "null_body_count": null_body_count,
        "null_answer_count": null_answer_count,
        "p1_high_incident_count": p1_count,
        "p1_high_incident_pct": round(100.0 * p1_count / n, 2),
        "body_length": {
            "char_mean": round(float(body_char_lens.mean()), 1),
            "char_median": round(float(body_char_lens.median()), 1),
            "char_p95": round(float(body_char_lens.quantile(0.95)), 1),
            "word_mean": round(float(body_word_lens.mean()), 1),
            "word_median": round(float(body_word_lens.median()), 1),
        },
        "answer_length": {
            "char_mean": round(float(answer_char_lens.mean()), 1),
            "char_median": round(float(answer_char_lens.median()), 1),
            "char_p95": round(float(answer_char_lens.quantile(0.95)), 1),
            "word_mean": round(float(answer_word_lens.mean()), 1),
            "word_median": round(float(answer_word_lens.median()), 1),
            "l1_asks_clarifying_info_pct": round(100.0 * l1_ask_info_rate, 2),
            "l1_contains_question_pct": round(100.0 * l1_question_mark_rate, 2),
            "l1_mentions_refund_or_sla_pct": round(100.0 * l1_refund_mention_rate, 2),
        },
        "distributions": {
            "type": dist_dict("type"),
            "priority": dist_dict("priority"),
            "language": dist_dict("language"),
            "queue": dist_dict("queue"),
        },
        "cross_tabs": {
            "type_x_priority": {k: {pk: int(pv) for pk, pv in v.items()} for k, v in type_x_priority.items()},
            "language_x_type": {k: {tk: int(tv) for tk, tv in v.items()} for k, v in lang_x_type.items()},
            "p1_by_queue": {k: int(v) for k, v in queue_x_p1.items()},
        },
        "pii_scan_kb_train": pii_stats,
        "healthcare_vpn_outage_matches_kb_train": healthcare_vpn_count,
        "healthcare_vpn_samples": healthcare_vpn_samples,
        "top_tags": top_tags,
        "examples_by_type": examples_by_type,
        "examples_by_queue": examples_by_queue,
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    KB_AUDIT_PATH.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[data_prep] Wrote kb_train-only audit summary -> {KB_AUDIT_PATH}")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare strict 80/20 split and audit kb_train only.")
    parser.add_argument("--full", action="store_true", help="Use full 50k clean CSV instead of 20k subset.")
    args = parser.parse_args()

    kb_train_df, test_df, data_file = prepare_and_split(use_full=args.full)
    audit = audit_kb_train_only(kb_train_df, data_file, len(kb_train_df) + len(test_df))

    print("\n=== KB_TRAIN AUDIT SNAPSHOT (Zero Holdout Inspection) ===")
    print(f"Dataset file         : {audit['dataset_file']}")
    print(f"KB Train / Holdout   : {audit['kb_train_rows']:,} / {audit['holdout_test_rows']:,}")
    print(f"Null/Empty Subjects  : {audit['null_subject_count']:,} ({audit['null_subject_pct']}%)")
    print(f"P1 (High + Incident) : {audit['p1_high_incident_count']:,} ({audit['p1_high_incident_pct']}%)")
    print(f"Languages            : {audit['distributions']['language']}")
    print(f"Types                : {audit['distributions']['type']}")
    print(f"Priorities           : {audit['distributions']['priority']}")
    print(f"Queues (10)          : {[(q['label'], q['count'], str(q['pct'])+'%') for q in audit['distributions']['queue']]}")
    print(f"Answer words median  : {audit['answer_length']['word_median']} | Clarifying L1 rate: {audit['answer_length']['l1_asks_clarifying_info_pct']}%")


if __name__ == "__main__":
    main()
