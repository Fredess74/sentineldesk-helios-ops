"""Train & calibrate local multi-task triage heads strictly on data/kb_train.parquet.

Architecture per target head (`type`, `queue`, `priority`):
  1. LightGBM on [256d EmbeddingGemma 2 | 64d TF-IDF SVD | 14 structural features]
  2. Regularized LogisticRegression on [word+char TF-IDF | scaled dense features]
  3. Validation-optimal probability blend + multi-class Temperature Scaling (ECE <= 0.05)
  4. P1 (`high` + `Incident`) recall threshold tuning on internal 15% val split (>= 90% recall)

Usage:
    python backend/train_ml.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedShuffleSplit

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
from backend.features import CalibratedMultiTaskHead, TicketFeaturePipeline, apply_temperature


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


def _fit_blend_and_temperature(
    p_lgb: np.ndarray,
    p_lin: np.ndarray,
    y_val: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """Find optimal blend weight w_lgb and temperature T minimizing validation NLL."""
    best_loss = float("inf")
    best_w = 0.5
    for w in np.linspace(0.0, 1.0, 21):
        p_mix = _norm_probs(w * p_lgb + (1.0 - w) * p_lin)
        loss = log_loss(y_val, p_mix, labels=list(range(p_lgb.shape[1])))
        if loss < best_loss:
            best_loss = loss
            best_w = float(w)

    p_blend = _norm_probs(best_w * p_lgb + (1.0 - best_w) * p_lin)

    def _nll(params: np.ndarray) -> float:
        t = float(params[0])
        p_cal = _norm_probs(apply_temperature(p_blend, t))
        return float(log_loss(y_val, p_cal, labels=list(range(p_lgb.shape[1]))))

    res = minimize(_nll, x0=np.array([1.0]), bounds=[(0.2, 5.0)], method="L-BFGS-B")
    best_temp = float(res.x[0])
    p_cal = apply_temperature(p_blend, best_temp)
    return best_w, best_temp, p_cal


def _tune_prior_alpha(
    probs_val: np.ndarray,
    y_val: np.ndarray,
    class_priors: np.ndarray,
    max_alpha: float = 0.65,
) -> tuple[float, float]:
    """Tune Menon-style logit/prior adjustment exponent on internal val split for Macro-F1."""
    best_alpha = 0.0
    best_f1 = float(f1_score(y_val, np.argmax(probs_val, axis=1), average="macro"))
    for alpha in np.linspace(0.0, max_alpha, 27):
        adj = probs_val / np.power(np.clip(class_priors, 1e-6, 1.0), alpha)[None, :]
        pred = np.argmax(adj, axis=1)
        f1 = float(f1_score(y_val, pred, average="macro"))
        if f1 > best_f1 + 1e-4:
            best_f1 = f1
            best_alpha = float(alpha)
    return best_alpha, best_f1


def _tune_p1_thresholds(
    p_type_val: np.ndarray,
    p_prio_val: np.ndarray,
    y_type_val: np.ndarray,
    y_prio_val: np.ndarray,
    type_classes: list[str],
    prio_classes: list[str],
    target_recall: float = 0.95,
) -> dict[str, Any]:
    """Tune thresholds on internal val split to guarantee High-Priority Incident Recall >= 90%."""
    inc_idx = type_classes.index("Incident")
    high_idx = prio_classes.index("high")
    true_p1 = (y_type_val == inc_idx) & (y_prio_val == high_idx)
    p_inc = p_type_val[:, inc_idx]
    p_high = p_prio_val[:, high_idx]
    p_joint = p_inc * p_high

    # Search compact thresholds for (inc_thr, high_thr) maximizing Macro-F1 while hitting P1 recall >= target
    best_cfg = {
        "incident_threshold": 0.35,
        "high_threshold": 0.32,
        "p1_joint_threshold": 0.25,
        "val_p1_recall": 0.0,
        "val_p1_precision": 0.0,
        "val_type_f1": 0.0,
        "val_prio_f1": 0.0,
    }
    best_score = -1.0

    for inc_thr in np.linspace(0.20, 0.50, 13):
        for high_thr in np.linspace(0.20, 0.50, 13):
            pred_type = np.argmax(p_type_val, axis=1)
            pred_prio = np.argmax(p_prio_val, axis=1)

            # Promote borderline high-risk incidents when both probabilities clear tuned bars
            promote_inc = (p_inc >= inc_thr) & (p_high >= high_thr)
            pred_type_adj = np.where(promote_inc, inc_idx, pred_type)
            pred_prio_adj = np.where(promote_inc, high_idx, pred_prio)

            pred_p1 = (pred_type_adj == inc_idx) & (pred_prio_adj == high_idx)
            tp = int(np.sum(pred_p1 & true_p1))
            fn = int(np.sum((~pred_p1) & true_p1))
            fp = int(np.sum(pred_p1 & (~true_p1)))
            recall = tp / max(tp + fn, 1)
            precision = tp / max(tp + fp, 1)

            t_f1 = float(f1_score(y_type_val, pred_type_adj, average="macro"))
            p_f1 = float(f1_score(y_prio_val, pred_prio_adj, average="macro"))
            combo = 0.5 * (t_f1 + p_f1) + 0.15 * precision

            if recall >= target_recall and combo > best_score:
                best_score = combo
                best_cfg = {
                    "incident_threshold": round(float(inc_thr), 4),
                    "high_threshold": round(float(high_thr), 4),
                    "p1_joint_threshold": round(float(inc_thr * high_thr), 4),
                    "val_p1_recall": round(float(recall), 4),
                    "val_p1_precision": round(float(precision), 4),
                    "val_type_f1": round(t_f1, 4),
                    "val_prio_f1": round(p_f1, 4),
                }

    # Also compute a clean joint P1 threshold for the Escalation Gate
    if best_cfg["val_p1_recall"] < target_recall:
        # Fallback: sweep joint probability directly
        for jt in np.linspace(0.35, 0.08, 28):
            pred_p1 = p_joint >= jt
            tp = int(np.sum(pred_p1 & true_p1))
            recall = tp / max(int(np.sum(true_p1)), 1)
            if recall >= target_recall:
                best_cfg["p1_joint_threshold"] = round(float(jt), 4)
                best_cfg["val_p1_recall"] = round(float(recall), 4)
                break

    return best_cfg


def train_and_save_models() -> dict[str, Any]:
    if not KB_TRAIN_PATH.exists() or not KB_EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            "Missing kb_train.parquet or kb_embeddings.npy. Run data_prep.py and indexing.py first."
        )

    kb_df = pd.read_parquet(KB_TRAIN_PATH)
    kb_emb = np.load(KB_EMBEDDINGS_PATH)
    assert len(kb_df) == len(kb_emb), "Embedding row count mismatch with kb_train!"

    # Zero-leakage check against holdout
    if HOLDOUT_TEST_PATH.exists():
        holdout_ids = set(pd.read_parquet(HOLDOUT_TEST_PATH, columns=["ticket_id"])["ticket_id"])
        assert len(set(kb_df["ticket_id"]) & holdout_ids) == 0, "LEAKAGE ERROR in train_ml.py!"

    t0 = time.perf_counter()
    print(f"[train_ml] Training calibrated multi-task models on kb_train ({len(kb_df):,} rows)...")

    # Internal stratified 85/15 carve-out inside kb_train for calibration & threshold tuning
    strat_key = kb_df["type"].astype(str) + "__" + kb_df["priority"].astype(str)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    tr_idx, val_idx = next(sss.split(kb_df, strat_key))

    tr_df = kb_df.iloc[tr_idx].reset_index(drop=True)
    val_df = kb_df.iloc[val_idx].reset_index(drop=True)
    tr_emb = kb_emb[tr_idx]
    val_emb = kb_emb[val_idx]

    feat_pipe = TicketFeaturePipeline(svd_dim=64)
    x_tr_dense, x_tr_sparse = feat_pipe.fit_transform(
        tr_df["subject"].tolist(), tr_df["body"].tolist(), tr_emb
    )
    x_val_dense, x_val_sparse = feat_pipe.transform(
        val_df["subject"].tolist(), val_df["body"].tolist(), val_emb
    )
    print(
        f"[train_ml] Features built: dense={x_tr_dense.shape}, sparse={x_tr_sparse.shape} "
        f"(train={len(tr_df):,}, cal_val={len(val_df):,})"
    )

    targets_spec = [
        ("type", EXPECTED_TYPES, 1.5, 240),
        ("queue", EXPECTED_QUEUES, 2.5, 260),
        ("priority", EXPECTED_PRIORITIES, 1.2, 220),
    ]

    heads: dict[str, CalibratedMultiTaskHead] = {}
    val_probs: dict[str, np.ndarray] = {}
    val_y: dict[str, np.ndarray] = {}
    val_summary: dict[str, Any] = {}

    for col, classes, lin_c, n_est in targets_spec:
        c2i = {c: i for i, c in enumerate(classes)}
        y_tr = np.array([c2i[v] for v in tr_df[col]], dtype=np.int64)
        y_va = np.array([c2i[v] for v in val_df[col]], dtype=np.int64)

        print(f"[train_ml] Fitting `{col}` ({len(classes)} classes)...")
        lgb = LGBMClassifier(
            objective="multiclass",
            num_class=len(classes),
            n_estimators=n_est,
            learning_rate=0.06,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        lgb.fit(x_tr_dense, y_tr)

        lin = LogisticRegression(
            C=lin_c,
            max_iter=600,
            solver="lbfgs",
            random_state=42,
        )
        lin.fit(x_tr_sparse, y_tr)

        p_lgb_va = lgb.predict_proba(x_val_dense)
        p_lin_va = lin.predict_proba(x_val_sparse)
        w_lgb, temp, p_cal_va = _fit_blend_and_temperature(p_lgb_va, p_lin_va, y_va)
        priors = np.bincount(y_tr, minlength=len(classes)).astype(np.float64)
        priors /= max(priors.sum(), 1.0)
        # Apply prior adjustment for severely imbalanced queue/priority heads (preserves calibrated probs for ECE)
        prior_alpha, f1_adj = (
            _tune_prior_alpha(p_cal_va, y_va, priors, max_alpha=0.55)
            if col in ("queue", "priority")
            else (0.0, float(f1_score(y_va, np.argmax(p_cal_va, axis=1), average="macro")))
        )

        f1_raw = float(f1_score(y_va, np.argmax(p_cal_va, axis=1), average="macro"))
        ece_val = compute_ece(p_cal_va, y_va)
        print(
            f"[train_ml]   `{col}` val Macro-F1={f1_adj:.4f} (raw={f1_raw:.4f}), ECE={ece_val:.4f}, "
            f"w_lgb={w_lgb:.2f}, T={temp:.3f}, prior_alpha={prior_alpha:.2f}"
        )

        heads[col] = CalibratedMultiTaskHead(classes, lgb, lin, w_lgb, temp, class_priors=priors, prior_alpha=prior_alpha)
        val_probs[col] = p_cal_va
        val_y[col] = y_va
        val_summary[col] = {
            "val_macro_f1": round(f1_adj, 4),
            "val_ece": round(ece_val, 4),
            "w_lgb": round(w_lgb, 3),
            "temperature": round(temp, 4),
            "prior_alpha": round(prior_alpha, 3),
        }

    # Tune P1 thresholds on internal val split for >= 95% internal val recall (safety margin for >= 90% holdout)
    p1_cfg = _tune_p1_thresholds(
        val_probs["type"],
        val_probs["priority"],
        val_y["type"],
        val_y["priority"],
        EXPECTED_TYPES,
        EXPECTED_PRIORITIES,
        target_recall=0.95,
    )
    print(f"[train_ml] P1 calibration config (internal val): {p1_cfg}")

    bundle = {
        "feature_pipeline": feat_pipe,
        "heads": heads,
        "p1_config": p1_cfg,
        "val_summary": val_summary,
        "kb_train_rows": len(kb_df),
        "internal_train_rows": len(tr_df),
        "internal_val_rows": len(val_df),
        "train_seconds": round(time.perf_counter() - t0, 2),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, ML_MODELS_PATH, compress=3)
    print(f"[train_ml] Saved calibrated models -> {ML_MODELS_PATH} ({bundle['train_seconds']}s)")
    return {"val_summary": val_summary, "p1_config": p1_cfg, "train_seconds": bundle["train_seconds"]}


def _fit_3way_blend_and_temperature(
    p_lgb: np.ndarray,
    p_lin: np.ndarray,
    p_xlmr: np.ndarray | None,
    y_val: np.ndarray,
) -> tuple[float, float, float, float, np.ndarray]:
    """Find optimal simplex weights (w_xlmr, w_lgb, w_lin) and temperature T minimizing val NLL."""
    n_cls = p_lgb.shape[1]
    best_loss = float("inf")
    best_weights = (0.0, 0.5, 0.5)

    if p_xlmr is None:
        for w_lgb in np.linspace(0.0, 1.0, 21):
            w_lin = 1.0 - w_lgb
            p_mix = _norm_probs(w_lgb * p_lgb + w_lin * p_lin)
            loss = log_loss(y_val, p_mix, labels=list(range(n_cls)))
            if loss < best_loss:
                best_loss = loss
                best_weights = (0.0, float(w_lgb), float(w_lin))
    else:
        grid = np.linspace(0.0, 1.0, 11)
        for wx in grid:
            for wg in grid:
                wl = round(1.0 - wx - wg, 4)
                if wl < -1e-6:
                    continue
                wl = max(0.0, wl)
                p_mix = _norm_probs(wx * p_xlmr + wg * p_lgb + wl * p_lin)
                loss = log_loss(y_val, p_mix, labels=list(range(n_cls)))
                if loss < best_loss:
                    best_loss = loss
                    best_weights = (float(wx), float(wg), float(wl))

    wx, wg, wl = best_weights
    p_blend = _norm_probs((wx * p_xlmr if p_xlmr is not None else 0.0) + wg * p_lgb + wl * p_lin)

    def _nll(params: np.ndarray) -> float:
        t = float(params[0])
        p_cal = _norm_probs(apply_temperature(p_blend, t))
        return float(log_loss(y_val, p_cal, labels=list(range(n_cls))))

    res = minimize(_nll, x0=np.array([1.0]), bounds=[(0.2, 5.0)], method="L-BFGS-B")
    best_temp = float(res.x[0])
    p_cal = apply_temperature(p_blend, best_temp)
    return wx, wg, wl, best_temp, p_cal


def recalibrate_on_kb_val() -> dict[str, Any]:
    """Re-tune 3-way ensemble weights (including XLM-R if trained), temperature, prior_alpha & P1 thresholds strictly on kb_train internal 15% val split."""
    if not ML_MODELS_PATH.exists():
        return train_and_save_models()

    kb_df = pd.read_parquet(KB_TRAIN_PATH)
    kb_emb = np.load(KB_EMBEDDINGS_PATH)
    bundle = joblib.load(ML_MODELS_PATH)

    xlmr_val_path = ARTIFACTS_DIR / "xlmr_val_probs.npz"
    xlmr_val = np.load(xlmr_val_path) if xlmr_val_path.exists() else None
    if xlmr_val is not None:
        print(f"[train_ml] Loaded Multi-Task XLM-R internal val probs from {xlmr_val_path.name} for 3-way stacking!")

    strat_key = kb_df["type"].astype(str) + "__" + kb_df["priority"].astype(str)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    tr_idx, val_idx = next(sss.split(kb_df, strat_key))

    tr_df = kb_df.iloc[tr_idx].reset_index(drop=True)
    val_df = kb_df.iloc[val_idx].reset_index(drop=True)
    val_emb = kb_emb[val_idx]

    feat_pipe = bundle["feature_pipeline"]
    x_val_dense, x_val_sparse = feat_pipe.transform(
        val_df["subject"].tolist(), val_df["body"].tolist(), val_emb
    )

    val_probs: dict[str, np.ndarray] = {}
    val_y: dict[str, np.ndarray] = {}
    val_summary: dict[str, Any] = {}

    for col, classes in [("type", EXPECTED_TYPES), ("queue", EXPECTED_QUEUES), ("priority", EXPECTED_PRIORITIES)]:
        c2i = {c: i for i, c in enumerate(classes)}
        y_tr = np.array([c2i[v] for v in tr_df[col]], dtype=np.int64)
        y_va = np.array([c2i[v] for v in val_df[col]], dtype=np.int64)
        head = bundle["heads"][col]

        p_lgb_va = head.lgb_model.predict_proba(x_val_dense)
        p_lin_va = head.lin_model.predict_proba(x_val_sparse)
        p_xlmr_va = xlmr_val[col] if xlmr_val is not None else None

        wx, wg, wl, temp, p_cal_va = _fit_3way_blend_and_temperature(p_lgb_va, p_lin_va, p_xlmr_va, y_va)
        priors = np.bincount(y_tr, minlength=len(classes)).astype(np.float64)
        priors /= max(priors.sum(), 1.0)
        prior_alpha, f1_adj = (
            _tune_prior_alpha(p_cal_va, y_va, priors, max_alpha=0.55)
            if col in ("queue", "priority")
            else (0.0, float(f1_score(y_va, np.argmax(p_cal_va, axis=1), average="macro")))
        )
        head.w_xlmr = wx
        head.w_lgb = wg
        head.w_lin = wl
        head.temperature = temp
        head.class_priors = priors
        head.prior_alpha = prior_alpha
        ece_val = compute_ece(p_cal_va, y_va)
        val_probs[col] = p_cal_va
        val_y[col] = y_va
        val_summary[col] = {
            "val_macro_f1": round(f1_adj, 4),
            "val_ece": round(ece_val, 4),
            "w_xlmr": round(wx, 3),
            "w_lgb": round(wg, 3),
            "w_lin": round(wl, 3),
            "temperature": round(temp, 4),
            "prior_alpha": round(prior_alpha, 3),
        }
        print(
            f"[train_ml] Recalibrated `{col}`: val_macro_f1={f1_adj:.4f}, ECE={ece_val:.4f}, "
            f"weights=(xlmr={wx:.2f}, lgb={wg:.2f}, lin={wl:.2f}), T={temp:.3f}, alpha={prior_alpha:.2f}"
        )

    p1_cfg = _tune_p1_thresholds(
        val_probs["type"],
        val_probs["priority"],
        val_y["type"],
        val_y["priority"],
        EXPECTED_TYPES,
        EXPECTED_PRIORITIES,
        target_recall=0.95,
    )
    print(f"[train_ml] Recalibrated P1 config (kb_train internal val): {p1_cfg}")
    bundle["p1_config"] = p1_cfg
    bundle["val_summary"] = val_summary
    bundle["uses_xlmr"] = xlmr_val is not None
    joblib.dump(bundle, ML_MODELS_PATH, compress=3)
    return {"val_summary": val_summary, "p1_config": p1_cfg}


if __name__ == "__main__":
    if "--recalibrate-only" in sys.argv:
        recalibrate_on_kb_val()
    else:
        train_and_save_models()
