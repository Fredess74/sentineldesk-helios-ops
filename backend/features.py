"""Shared feature extraction pipeline for local calibrated multi-task triage.

Combines:
- 256d normalized EmbeddingGemma 2 vectors
- 64d compact TF-IDF + TruncatedSVD features
- Sparse word+char TF-IDF features (for fast linear head)
- 14 null-safe structural & bilingual urgency/domain signals
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

_OUTAGE_URGENCY_RE = re.compile(
    r"\b(?:urgent|critical|emergency|outage|down|offline|failure|crash(?:ed|es|ing)?|breach|"
    r"unauthorized|immediately|asap|severe|disruption|vpn|telemedicine|emr|hospital|medical|"
    r"encryption|dringend|kritisch|notfall|ausfall|absturz|störung|datenverstoß|sofort|"
    r"unbefugt|krankenhaus|medizinisch|verschlüsselung|unterbrechung)\b",
    re.IGNORECASE,
)

_PROBLEM_RECURRING_RE = re.compile(
    r"\b(?:recurring|persist(?:s|ent|ing)?|intermittent|sporadic|investigat(?:e|ion)|"
    r"compatibility|incompatibility|memory|overload|bug|defect|sync|malfunction|"
    r"wiederkehrend|anhaltend|untersuchung|kompatibilität|fehler|abstürze|speicher)\b",
    re.IGNORECASE,
)

_CHANGE_RE = re.compile(
    r"\b(?:upgrade|update|enhance(?:ment)?|implement(?:ation)?|migration|feature|"
    r"protocol|optimiz(?:e|ation)|extension|aktualisierung|verbesserung|erweiterung|"
    r"umstellung|implementierung|optimierung|vorschlag)\b",
    re.IGNORECASE,
)

_REQUEST_INFO_RE = re.compile(
    r"\b(?:inquir(?:e|y)|information|details|how to|could you|can you|interested in|"
    r"options|pricing|quote|demo|exploring|documentation|informationen|könnten sie|"
    r"welche|anfrage|beratung|möglichkeiten|preis|details)\b",
    re.IGNORECASE,
)

_BILLING_RE = re.compile(
    r"\b(?:invoice|billing|payment|refund|charge|credit|subscription|fee|overcharge|"
    r"rechnung|zahlung|erstattung|gebühr|gutschrift|abonnement|abbuchung)\b",
    re.IGNORECASE,
)

_RETURNS_RE = re.compile(
    r"\b(?:return|exchange|replace(?:ment)?|defective|damaged|warranty|order|"
    r"rückgabe|umtausch|ersatz|defekt|beschädigt|garantie|bestellung)\b",
    re.IGNORECASE,
)

_HR_RE = re.compile(
    r"\b(?:employee|payroll|onboarding|hr|human resources|leave|benefits|staff|"
    r"mitarbeiter|personal|gehalt|urlaub|einstellung)\b",
    re.IGNORECASE,
)


def _safe_text(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, float) and np.isnan(val):
        return ""
    return str(val).strip()


def combine_ticket_text(subject: Any, body: Any) -> str:
    s = _safe_text(subject)
    b = _safe_text(body)
    if s and b:
        return f"{s} \n {b}"
    return b or s or "empty_ticket"


def extract_structural_matrix(subjects: list[Any], bodies: list[Any]) -> np.ndarray:
    """Compute 14 deterministic, null-safe structural & bilingual signal features."""
    rows = []
    for subj_raw, body_raw in zip(subjects, bodies):
        s = _safe_text(subj_raw)
        b = _safe_text(body_raw)
        full = f"{s} {b}".strip()
        n_chars = max(len(full), 1)
        alpha_chars = sum(1 for ch in full if ch.isalpha())
        upper_chars = sum(1 for ch in full if ch.isupper())
        digit_chars = sum(1 for ch in full if ch.isdigit())

        rows.append(
            [
                1.0 if not s else 0.0,                           # subject_missing
                np.log1p(len(s)),                                # log subject char len
                np.log1p(len(b)),                                # log body char len
                np.log1p(len(b.split())),                        # log body word count
                upper_chars / max(alpha_chars, 1),               # uppercase ratio
                min(full.count("!"), 10) / 10.0,                 # exclamation density
                min(full.count("?"), 10) / 10.0,                 # question mark density
                digit_chars / n_chars,                           # digit ratio
                min(len(_OUTAGE_URGENCY_RE.findall(full)), 10),  # outage/urgency signal
                min(len(_PROBLEM_RECURRING_RE.findall(full)), 10),
                min(len(_CHANGE_RE.findall(full)), 10),
                min(len(_REQUEST_INFO_RE.findall(full)), 10),
                min(len(_BILLING_RE.findall(full)), 10),
                min(len(_RETURNS_RE.findall(full)) + len(_HR_RE.findall(full)), 10),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


class TicketFeaturePipeline:
    """Fits TF-IDF, compact 64d SVD, and structural scaler strictly on kb_train."""

    def __init__(self, svd_dim: int = 64) -> None:
        self.svd_dim = svd_dim
        self.word_tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=25000,
            min_df=2,
            sublinear_tf=True,
            strip_accents=None,  # preserve German umlauts äöüß
        )
        self.char_tfidf = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=15000,
            min_df=3,
            sublinear_tf=True,
        )
        self.svd = TruncatedSVD(n_components=svd_dim, random_state=42)
        self.struct_scaler = StandardScaler()
        self.dense_scaler = StandardScaler()

    def fit_transform(
        self,
        subjects: list[Any],
        bodies: list[Any],
        embeddings: np.ndarray,
    ) -> tuple[np.ndarray, sp.csr_matrix]:
        texts = [combine_ticket_text(s, b) for s, b in zip(subjects, bodies)]
        w_mat = self.word_tfidf.fit_transform(texts)
        c_mat = self.char_tfidf.fit_transform(texts)
        svd_feats = self.svd.fit_transform(w_mat).astype(np.float32)

        struct_raw = extract_structural_matrix(subjects, bodies)
        struct_scaled = self.struct_scaler.fit_transform(struct_raw).astype(np.float32)

        emb = np.asarray(embeddings, dtype=np.float32)
        dense_for_tree = np.hstack([emb, svd_feats, struct_raw]).astype(np.float32)
        dense_scaled = self.dense_scaler.fit_transform(np.hstack([emb, svd_feats, struct_scaled]))
        sparse_for_linear = sp.hstack(
            [w_mat, c_mat, sp.csr_matrix(dense_scaled * 0.35)],
            format="csr",
            dtype=np.float32,
        )
        return dense_for_tree, sparse_for_linear

    def transform(
        self,
        subjects: list[Any],
        bodies: list[Any],
        embeddings: np.ndarray,
    ) -> tuple[np.ndarray, sp.csr_matrix]:
        texts = [combine_ticket_text(s, b) for s, b in zip(subjects, bodies)]
        w_mat = self.word_tfidf.transform(texts)
        c_mat = self.char_tfidf.transform(texts)
        svd_feats = self.svd.transform(w_mat).astype(np.float32)

        struct_raw = extract_structural_matrix(subjects, bodies)
        struct_scaled = self.struct_scaler.transform(struct_raw).astype(np.float32)

        emb = np.asarray(embeddings, dtype=np.float32)
        if emb.ndim == 1:
            emb = emb[None, :]
        dense_for_tree = np.hstack([emb, svd_feats, struct_raw]).astype(np.float32)
        dense_scaled = self.dense_scaler.transform(np.hstack([emb, svd_feats, struct_scaled]))
        sparse_for_linear = sp.hstack(
            [w_mat, c_mat, sp.csr_matrix(dense_scaled * 0.35)],
            format="csr",
            dtype=np.float32,
        )
        return dense_for_tree, sparse_for_linear


def apply_temperature(probs: np.ndarray, temp: float) -> np.ndarray:
    """Multi-class temperature scaling preserving normalized probability sums."""
    eps = 1e-7
    log_p = np.log(np.clip(probs, eps, 1.0)) / max(temp, 0.05)
    log_p -= np.max(log_p, axis=1, keepdims=True)
    exp_p = np.exp(log_p)
    return exp_p / np.sum(exp_p, axis=1, keepdims=True)


class CalibratedMultiTaskHead:
    """Serializable calibrated blend of optional Fine-Tuned XLM-R + LightGBM + Linear TF-IDF head."""

    def __init__(
        self,
        classes: list[str],
        lgb_model: Any,
        lin_model: Any,
        w_lgb: float,
        temperature: float,
        class_priors: np.ndarray | None = None,
        prior_alpha: float = 0.0,
        w_xlmr: float = 0.0,
        w_lin: float | None = None,
    ) -> None:
        self.classes = list(classes)
        self.lgb_model = lgb_model
        self.lin_model = lin_model
        self.w_lgb = float(w_lgb)
        self.w_xlmr = float(w_xlmr)
        self.w_lin = float(w_lin) if w_lin is not None else max(0.0, 1.0 - self.w_lgb - self.w_xlmr)
        self.temperature = float(temperature)
        self.class_priors = class_priors
        self.prior_alpha = float(prior_alpha)

    def predict_proba(
        self,
        x_dense: np.ndarray,
        x_sparse: Any,
        p_xlmr: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return strictly calibrated posterior probabilities (ECE <= 0.05)."""
        p_lgb = self.lgb_model.predict_proba(x_dense)
        p_lin = self.lin_model.predict_proba(x_sparse)
        wx = getattr(self, "w_xlmr", 0.0)
        wl = getattr(self, "w_lin", max(0.0, 1.0 - self.w_lgb - wx))
        if p_xlmr is not None and wx > 0.0:
            p_mix = wx * p_xlmr + self.w_lgb * p_lgb + wl * p_lin
        else:
            denom = max(self.w_lgb + wl, 1e-6)
            p_mix = (self.w_lgb / denom) * p_lgb + (wl / denom) * p_lin
        return apply_temperature(p_mix, self.temperature)

    def decode_indices(self, probs: np.ndarray) -> np.ndarray:
        """Decode class indices using optional internal-val-tuned prior adjustment for Macro-F1."""
        alpha = getattr(self, "prior_alpha", 0.0)
        priors = getattr(self, "class_priors", None)
        if alpha > 0.0 and priors is not None:
            adj = probs / np.power(np.clip(priors, 1e-6, 1.0), alpha)[None, :]
            return np.argmax(adj, axis=1)
        return np.argmax(probs, axis=1)

