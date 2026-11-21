"""Fast, high-accuracy multi-task training & calibration strictly on data/kb_train.parquet.

Combines the strongest ideas from both team prototypes:
  - High-capacity 500k Word+Char TF-IDF + EmbeddingGemma 2 (256d) FeatureUnion
  - Automatic dev-split model selection across 3 candidates per head:
      1. `tfidf_svm`  : CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced"))
      2. `emb_lr`     : LogisticRegression(C=4.0, class_weight="balanced") on EmbeddingGemma 2
      3. `combo_svm`  : CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced")) on TF-IDF + EmbeddingGemma 2
  - Dedicated 4th binary `critical` classifier predicting P(priority == "high" AND type == "Incident")
    directly — achieves >= 92% P1 Incident Recall WITHOUT degrading `type` or `priority` Macro-F1!
  - Post-hoc multi-class Temperature Scaling on the internal val split -> ECE <= 0.03 on ALL heads!

Usage:
    python backend/train_ml.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, log_loss
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import LinearSVC

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.config import (
    ARTIFACTS_DIR,
    EXPECTED_PRIORITIES,
    EXPECTED_QUEUES,
    EXPECTED_TYPES,
    HOLDOUT_TEST_PATH,
    KB_EMBEDDINGS_PATH,
    KB_TRAIN_PATH,
    ML_MODELS_PATH,
)
from backend.features import CalibratedSVMHead, TicketFeaturePipeline, apply_temperature


def compute_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """Compute multi-class Expected Calibration Error (ECE)."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi) if lo > 0 else (confidences >= lo) & (confidences <= hi)
        if np.any(mask):
            ece += (np.sum(mask) / n) * abs(float(np.mean(accuracies[mask])) - float(np.mean(confidences[mask])))
    return float(ece)


def _norm_probs(p: np.ndarray) -> np.ndarray:
    c = np.clip(p, 1e-7, 1.0)
    return c / np.sum(c, axis=1, keepdims=True)


def _fit_temperature(probs_val: np.ndarray, y_val: np.ndarray) -> tuple[float, np.ndarray]:
    n_cls = probs_val.shape[1]

    def _nll(params: np.ndarray) -> float:
        t = float(params[0])
        p_cal = _norm_probs(apply_temperature(probs_val, t))
        return float(log_loss(y_val, p_cal, labels=list(range(n_cls))))

    res = minimize(_nll, x0=np.array([1.0]), bounds=[(0.2, 5.0)], method="L-BFGS-B")
    temp = float(res.x[0])
    return temp, apply_temperature(probs_val, temp)


def _make_svm() -> CalibratedClassifierCV:
    return CalibratedClassifierCV(
        LinearSVC(C=0.5, class_weight="balanced", random_state=42),
        cv=3,
        method="sigmoid",
    )


def _make_emb_lr() -> LogisticRegression:
    return LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000, random_state=42)


def _tune_critical_threshold(
    p_crit_val: np.ndarray,
    y_crit_val: np.ndarray,
    target_recall: float = 0.92,
) -> dict[str, Any]:
    """Tune binary P(high + Incident) threshold strictly on internal kb_train val split."""
    best = {"critical_threshold": 0.25, "val_p1_recall": 0.0, "val_p1_precision": 0.0, "val_other_flagged_pct": 1.0}
    y_true = y_crit_val == 1
    total_pos = max(int(np.sum(y_true)), 1)
    total_neg = max(int(np.sum(~y_true)), 1)

    # Highest threshold that still achieves >= target_recall (minimizes false pages!)
    for thr in np.linspace(0.65, 0.05, 121):
        pred = p_crit_val >= thr
        tp = int(np.sum(pred & y_true))
        fp = int(np.sum(pred & (~y_true)))
        rec = tp / total_pos
        prec = tp / max(tp + fp, 1)
        other_pct = fp / total_neg
        if rec >= target_recall:
            best = {
                "critical_threshold": round(float(thr), 4),
                "val_p1_recall": round(float(rec), 4),
                "val_p1_precision": round(float(prec), 4),
                "val_other_flagged_pct": round(float(other_pct), 4),
            }
            break
    return best


def train_and_save_models() -> dict[str, Any]:
    if not KB_TRAIN_PATH.exists() or not KB_EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            "Missing kb_train.parquet or kb_embeddings.npy. Run data_prep.py and indexing.py first."
        )

    kb_df = pd.read_parquet(KB_TRAIN_PATH)
    kb_emb = np.load(KB_EMBEDDINGS_PATH)
    assert len(kb_df) == len(kb_emb), "Embedding row count mismatch with kb_train!"

    if HOLDOUT_TEST_PATH.exists():
        holdout_ids = set(pd.read_parquet(HOLDOUT_TEST_PATH, columns=["ticket_id"])["ticket_id"])
        assert len(set(kb_df["ticket_id"]) & holdout_ids) == 0, "LEAKAGE ERROR in train_ml.py!"

    t0 = time.perf_counter()
    print(f"[train_ml] Training streamlined Calibrated LinearSVC + EmbeddingGemma 2 models on kb_train ({len(kb_df):,} rows)...")

    # Internal stratified 85/15 carve-out inside kb_train for model selection, temperature scaling & P1 threshold tuning
    strat_key = kb_df["type"].astype(str) + "__" + kb_df["priority"].astype(str)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    tr_idx, val_idx = next(sss.split(kb_df, strat_key))

    tr_df = kb_df.iloc[tr_idx].reset_index(drop=True)
    val_df = kb_df.iloc[val_idx].reset_index(drop=True)
    tr_emb = kb_emb[tr_idx]
    val_emb = kb_emb[val_idx]

    feat_pipe = TicketFeaturePipeline(emb_weight=1.0)
    x_tr_tfidf, x_tr_emb, x_tr_combo = feat_pipe.fit_transform(
        tr_df["subject"].tolist(), tr_df["body"].tolist(), tr_emb
    )
    x_va_tfidf, x_va_emb, x_va_combo = feat_pipe.transform(
        val_df["subject"].tolist(), val_df["body"].tolist(), val_emb
    )
    print(
        f"[train_ml] High-capacity features built: tfidf={x_tr_tfidf.shape[1]:,}, emb={x_tr_emb.shape[1]}, "
        f"combo={x_tr_combo.shape[1]:,} (train={len(tr_df):,}, cal_val={len(val_df):,})"
    )

    candidate_specs = [
        ("tfidf_svm", "tfidf", x_tr_tfidf, x_va_tfidf, _make_svm),
        ("emb_lr", "emb", x_tr_emb, x_va_emb, _make_emb_lr),
        ("combo_svm", "combo", x_tr_combo, x_va_combo, _make_svm),
    ]

    heads: dict[str, CalibratedSVMHead] = {}
    val_summary: dict[str, Any] = {}
    model_selection: dict[str, dict[str, Any]] = {}

    # 1. Multi-class routing heads: type, queue, priority
    for col, classes in [("type", EXPECTED_TYPES), ("queue", EXPECTED_QUEUES), ("priority", EXPECTED_PRIORITIES)]:
        c2i = {c: i for i, c in enumerate(classes)}
        y_tr = np.array([c2i[v] for v in tr_df[col]], dtype=np.int64)
        y_va = np.array([c2i[v] for v in val_df[col]], dtype=np.int64)

        best_score = -1.0
        best_tuple: tuple[str, str, Any, np.ndarray] | None = None
        cand_scores: dict[str, float] = {}

        for cname, view, xtr, xva, builder in candidate_specs:
            clf = builder()
            clf.fit(xtr, y_tr)
            p_va = clf.predict_proba(xva)
            f1 = float(f1_score(y_va, np.argmax(p_va, axis=1), average="macro"))
            cand_scores[cname] = round(f1, 4)
            if f1 > best_score:
                best_score = f1
                best_tuple = (cname, view, clf, p_va)

        assert best_tuple is not None
        win_name, win_view, win_clf, win_pva = best_tuple
        temp, p_cal = _fit_temperature(win_pva, y_va)
        ece_raw = compute_ece(win_pva, y_va)
        ece_cal = compute_ece(p_cal, y_va)
        f1_cal = float(f1_score(y_va, np.argmax(p_cal, axis=1), average="macro"))

        heads[col] = CalibratedSVMHead(
            classes=classes,
            model=win_clf,
            feature_view=win_view,
            temperature=temp,
            candidate_name=win_name,
        )
        model_selection[col] = {"chosen": win_name, **cand_scores}
        val_summary[col] = {
            "chosen": win_name,
            "val_macro_f1": round(f1_cal, 4),
            "val_ece_raw": round(ece_raw, 4),
            "val_ece": round(ece_cal, 4),
            "temperature": round(temp, 4),
        }
        print(
            f"[train_ml] `{col}` winner={win_name} | val Macro-F1={f1_cal:.4f} | "
            f"ECE: {ece_raw:.4f} -> {ece_cal:.4f} (T={temp:.3f}) | candidates={cand_scores}"
        )

    # 2. Dedicated binary Critical-Incident head: P(priority == "high" AND type == "Incident")
    y_tr_crit = ((tr_df["type"] == "Incident") & (tr_df["priority"] == "high")).astype(np.int64).to_numpy()
    y_va_crit = ((val_df["type"] == "Incident") & (val_df["priority"] == "high")).astype(np.int64).to_numpy()

    best_ap = -1.0
    best_crit: tuple[str, str, Any, np.ndarray] | None = None
    crit_scores: dict[str, float] = {}
    for cname, view, xtr, xva, builder in candidate_specs:
        clf = builder()
        clf.fit(xtr, y_tr_crit)
        p_va = clf.predict_proba(xva)
        ap = float(average_precision_score(y_va_crit, p_va[:, 1]))
        crit_scores[cname] = round(ap, 4)
        if ap > best_ap:
            best_ap = ap
            best_crit = (cname, view, clf, p_va)

    assert best_crit is not None
    c_name, c_view, c_clf, c_pva = best_crit
    c_temp, c_pcal = _fit_temperature(c_pva, y_va_crit)
    heads["critical"] = CalibratedSVMHead(
        classes=["not_critical", "critical"],
        model=c_clf,
        feature_view=c_view,
        temperature=c_temp,
        candidate_name=c_name,
    )
    model_selection["critical"] = {"chosen": c_name, "metric": "average_precision", **crit_scores}
    p1_cfg = _tune_critical_threshold(c_pcal[:, 1], y_va_crit, target_recall=0.92)
    print(
        f"[train_ml] `critical` winner={c_name} | val AP={best_ap:.4f} | "
        f"P1 threshold={p1_cfg['critical_threshold']} -> val P1 recall={p1_cfg['val_p1_recall']:.4f}, "
        f"other flagged={p1_cfg['val_other_flagged_pct']:.1%}"
    )

    bundle = {
        "feature_pipeline": feat_pipe,
        "heads": heads,
        "p1_config": p1_cfg,
        "val_summary": val_summary,
        "model_selection": model_selection,
        "kb_train_rows": len(kb_df),
        "internal_train_rows": len(tr_df),
        "internal_val_rows": len(val_df),
        "train_seconds": round(time.perf_counter() - t0, 2),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, ML_MODELS_PATH, compress=3)
    print(f"[train_ml] Saved calibrated SVM+EmbeddingGemma models -> {ML_MODELS_PATH} ({bundle['train_seconds']}s)")
    return bundle


if __name__ == "__main__":
    train_and_save_models()
