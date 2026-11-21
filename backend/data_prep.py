"""Phase 1: Strict zero-leakage dataset preparation (official 28.5k clean CSV default) and kb_train-only audit.

Uses duplicate-text group splitting so identical/near-duplicate ticket texts never span
kb_train and holdout_test, and commits a reproducible splits/split_assignment.csv.

Usage:
    python backend/data_prep.py          # Official 28.5k clean CSV (recommended)
    python backend/data_prep.py --20k    # Faster 20k clean subset
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import StratifiedGroupKFold

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.config import (
    ARTIFACTS_DIR,
    BASE_DIR,
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

SPLITS_DIR = BASE_DIR / "splits"
SPLIT_ASSIGNMENT_PATH = SPLITS_DIR / "split_assignment.csv"


def _norm_dedup_key(subject: Any, body: Any) -> str:
    s = "" if subject is None or (isinstance(subject, float) and np.isnan(subject)) else str(subject)
    b = "" if body is None or (isinstance(body, float) and np.isnan(body)) else str(body)
    norm = re.sub(r"\s+", " ", f"{s} {b}".lower()).strip()
    return hashlib.sha1(norm.encode("utf-8", errors="ignore")).hexdigest()


def prepare_and_split(use_20k: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Download official clean dataset, apply duplicate-grouped stratified 80/20 split, verify zero leakage."""
    data_file = DATASET_FILE_20K if use_20k else DATASET_FILE_50K
    print(f"[data_prep] Loading {HF_DATASET_REPO} ({data_file})...")
    ds = load_dataset(HF_DATASET_REPO, data_files=data_file)
    df = ds["train"].to_pandas()
    raw_count = len(df)
    print(f"[data_prep] Raw rows loaded: {raw_count:,}")

    # Deterministic shuffle + sequential ticket IDs
    df = df.sample(frac=1.0, random_state=SPLIT_SEED).reset_index(drop=True)
    df["ticket_id"] = [f"TKT-{i:05d}" for i in range(len(df))]

    # Duplicate-text grouping prevents identical/near-duplicate tickets from leaking across train/holdout
    groups = [_norm_dedup_key(s, b) for s, b in zip(df["subject"], df["body"])]
    strat_key = df["type"].astype(str) + "__" + df["priority"].astype(str)

    # 5-fold StratifiedGroupKFold -> fold 0 is exact ~20% holdout, folds 1..4 are ~80% kb_train
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED)
    train_idx, test_idx = next(sgkf.split(df, strat_key, groups))

    kb_train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)

    # Zero-leakage assertions (IDs AND duplicate text groups!)
    train_ids = set(kb_train_df["ticket_id"])
    test_ids = set(test_df["ticket_id"])
    train_groups = {groups[i] for i in train_idx}
    test_groups = {groups[i] for i in test_idx}

    assert len(train_ids & test_ids) == 0, "LEAKAGE ERROR: Overlapping ticket_ids!"
    assert len(train_groups & test_groups) == 0, "LEAKAGE ERROR: Duplicate text spans train & holdout!"
    assert len(kb_train_df) + len(test_df) == raw_count, "Row count mismatch after split!"

    # Clean label space assertions
    actual_types = set(df["type"].dropna().unique())
    actual_priorities = set(df["priority"].dropna().unique())
    actual_langs = set(df["language"].dropna().unique())
    actual_queues = set(df["queue"].dropna().unique())

    assert actual_types == set(EXPECTED_TYPES), f"Unexpected types: {actual_types}"
    assert actual_priorities == set(EXPECTED_PRIORITIES), f"Unexpected priorities: {actual_priorities}"
    assert actual_langs == set(EXPECTED_LANGUAGES), f"Unexpected languages: {actual_langs}"
    assert len(actual_queues) == 10, f"Expected 10 clean queues, got {len(actual_queues)}"
    assert actual_queues == set(EXPECTED_QUEUES), f"Queue mismatch: {actual_queues ^ set(EXPECTED_QUEUES)}"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    kb_train_df.to_parquet(KB_TRAIN_PATH, index=False)
    test_df.to_parquet(HOLDOUT_TEST_PATH, index=False)

    # Save lightweight shareable split manifest (allows teammate apples-to-apples comparison)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    split_manifest = pd.DataFrame(
        {
            "ticket_id": df["ticket_id"],
            "split": ["kb_train" if i in set(train_idx) else "holdout_test" for i in range(len(df))],
        }
    )
    split_manifest.to_csv(SPLIT_ASSIGNMENT_PATH, index=False)

    print(
        f"[data_prep] Saved duplicate-grouped 80/20 split: "
        f"kb_train={len(kb_train_df):,} -> {KB_TRAIN_PATH.name}, "
        f"holdout_test={len(test_df):,} -> {HOLDOUT_TEST_PATH.name}"
    )
    print(f"[data_prep] Verified 0 overlapping ticket_ids AND 0 duplicate-text leaks; wrote {SPLIT_ASSIGNMENT_PATH.name}")
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

    def dist_dict(col: str) -> list[dict[str, Any]]:
        vc = kb_train_df[col].value_counts(dropna=False)
        return [
            {"label": str(k), "count": int(v), "pct": round(100.0 * float(v) / n, 2)}
            for k, v in vc.items()
        ]

    p1_mask = (kb_train_df["priority"] == "high") & (kb_train_df["type"] == "Incident")
    p1_count = int(p1_mask.sum())

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

    healthcare_vpn_mask = body_series.str.contains(
        r"hospital|medical|clinic|patient|telemedicine|healthcare|vpn|outage|krankenhaus|medizin",
        case=False,
        regex=True,
    )
    healthcare_vpn_count = int(healthcare_vpn_mask.sum())

    audit = {
        "dataset_repo": HF_DATASET_REPO,
        "dataset_file": data_file,
        "split_seed": SPLIT_SEED,
        "duplicate_grouped_split": True,
        "total_rows": int(total_rows),
        "kb_train_rows": int(n),
        "holdout_test_rows": int(total_rows - n),
        "zero_leakage_verified": True,
        "null_subject_count": null_subject_count,
        "null_subject_pct": round(100.0 * null_subject_count / n, 2),
        "null_body_count": null_body_count,
        "null_answer_count": null_answer_count,
        "p1_high_incident_count": p1_count,
        "p1_high_incident_pct": round(100.0 * p1_count / n, 2),
        "body_length": {
            "char_mean": round(float(body_char_lens.mean()), 1),
            "char_median": round(float(body_char_lens.median()), 1),
            "word_median": round(float(body_word_lens.median()), 1),
        },
        "answer_length": {
            "char_mean": round(float(answer_char_lens.mean()), 1),
            "char_median": round(float(answer_char_lens.median()), 1),
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
        "healthcare_vpn_outage_matches_kb_train": healthcare_vpn_count,
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    KB_AUDIT_PATH.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[data_prep] Wrote kb_train-only audit summary -> {KB_AUDIT_PATH}")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare duplicate-grouped zero-leakage 80/20 split.")
    parser.add_argument("--20k", dest="use_20k", action="store_true", help="Use smaller 20k subset instead of 28.5k.")
    parser.add_argument("--full", action="store_true", help="Explicitly select full 28.5k clean dataset (default).")
    args = parser.parse_args()

    kb_train_df, test_df, data_file = prepare_and_split(use_20k=args.use_20k)
    audit = audit_kb_train_only(kb_train_df, data_file, len(kb_train_df) + len(test_df))
    print(
        f"\n=== KB_TRAIN AUDIT SNAPSHOT ===\n"
        f"File: {audit['dataset_file']} | KB Train: {audit['kb_train_rows']:,} | Holdout: {audit['holdout_test_rows']:,}\n"
        f"Null Subjects: {audit['null_subject_count']:,} ({audit['null_subject_pct']}%) | "
        f"P1 (High+Incident): {audit['p1_high_incident_count']:,} ({audit['p1_high_incident_pct']}%)"
    )


if __name__ == "__main__":
    main()
