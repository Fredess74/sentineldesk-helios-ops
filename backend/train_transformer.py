"""Multi-Task Multilingual Transformer (xlm-roberta-base) fine-tuned strictly on kb_train.

Jointly trains on:
  - type (4 classes)
  - queue (10 classes, inverse-sqrt balanced loss for 21.7x queue skew)
  - priority (3 classes)
  - aux_tags (top 40 tag_1..tag_8 multi-label auxiliary supervision — training ONLY!)

Uses the exact same internal 85/15 stratified carve-out inside kb_train.parquet
as train_ml.py (zero holdout leakage).

Usage:
    python backend/train_transformer.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.config import (
    ARTIFACTS_DIR,
    EXPECTED_PRIORITIES,
    EXPECTED_QUEUES,
    EXPECTED_TYPES,
    HOLDOUT_TEST_PATH,
    KB_TRAIN_PATH,
)
from backend.indexing import safe_str

XLMR_MODEL_NAME = "xlm-roberta-base"
XLMR_CHECKPOINT_PATH = ARTIFACTS_DIR / "multitask_xlmr.pt"
XLMR_VAL_PROBS_PATH = ARTIFACTS_DIR / "xlmr_val_probs.npz"
XLMR_MAX_LEN = 192
AUX_TAG_TOP_K = 40


def format_xlmr_text(subject: Any, body: Any) -> str:
    s = safe_str(subject) or "none"
    b = safe_str(body) or "none"
    return f"Subject: {s} | Body: {b}"


class MultiTaskXLMR(nn.Module):
    """Shared multilingual transformer encoder with 3 triage heads + 1 training-only tag head."""

    def __init__(
        self,
        model_name: str = XLMR_MODEL_NAME,
        n_type: int = 4,
        n_queue: int = 10,
        n_prio: int = 3,
        n_aux_tags: int = AUX_TAG_TOP_K,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if pretrained:
            self.encoder = AutoModel.from_pretrained(model_name)
            hidden = self.encoder.config.hidden_size
        else:
            cfg = AutoConfig.from_pretrained(model_name)
            self.encoder = AutoModel.from_config(cfg)
            hidden = cfg.hidden_size

        self.dropout = nn.Dropout(0.18)
        self.type_head = nn.Linear(hidden, n_type)
        self.queue_head = nn.Linear(hidden, n_queue)
        self.prio_head = nn.Linear(hidden, n_prio)
        self.aux_tag_head = nn.Linear(hidden, n_aux_tags)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Attention-masked mean pooling (more robust than CLS alone on noisy/null-subject tickets)
        mask = attention_mask.unsqueeze(-1).type_as(out.last_hidden_state)
        pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        x = self.dropout(pooled)
        return (
            self.type_head(x),
            self.queue_head(x),
            self.prio_head(x),
            self.aux_tag_head(x),
        )


class TicketTorchDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        y_type: np.ndarray | None = None,
        y_queue: np.ndarray | None = None,
        y_prio: np.ndarray | None = None,
        y_tags: np.ndarray | None = None,
    ) -> None:
        self.texts = texts
        self.y_type = y_type
        self.y_queue = y_queue
        self.y_prio = y_prio
        self.y_tags = y_tags

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item: dict[str, Any] = {"text": self.texts[idx]}
        if self.y_type is not None:
            item["y_type"] = int(self.y_type[idx])
            item["y_queue"] = int(self.y_queue[idx])
            item["y_prio"] = int(self.y_prio[idx])
            item["y_tags"] = self.y_tags[idx]
        return item


class MultiTaskXLMRPredictor:
    """Fast GPU/CPU inference wrapper for the fine-tuned multi-task XLM-R model."""

    def __init__(self, checkpoint_path: Path = XLMR_CHECKPOINT_PATH) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model_name = ckpt.get("model_name", XLMR_MODEL_NAME)
        self.max_len = int(ckpt.get("max_len", XLMR_MAX_LEN))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = MultiTaskXLMR(
            model_name=self.model_name,
            n_type=len(EXPECTED_TYPES),
            n_queue=len(EXPECTED_QUEUES),
            n_prio=len(EXPECTED_PRIORITIES),
            n_aux_tags=int(ckpt.get("n_aux_tags", AUX_TAG_TOP_K)),
            pretrained=False,
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict_proba_batch(
        self,
        subjects: list[Any],
        bodies: list[Any],
        batch_size: int = 64,
    ) -> dict[str, np.ndarray]:
        texts = [format_xlmr_text(s, b) for s, b in zip(subjects, bodies)]
        p_t_list, p_q_list, p_p_list = [], [], []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(self.device)
            attn_mask = enc["attention_mask"].to(self.device)
            if self.device == "cuda":
                with torch.amp.autocast("cuda", dtype=self.dtype):
                    lt, lq, lp, _ = self.model(input_ids, attn_mask)
            else:
                lt, lq, lp, _ = self.model(input_ids, attn_mask)
            p_t_list.append(torch.softmax(lt.float(), dim=-1).cpu().numpy())
            p_q_list.append(torch.softmax(lq.float(), dim=-1).cpu().numpy())
            p_p_list.append(torch.softmax(lp.float(), dim=-1).cpu().numpy())

        return {
            "type": np.vstack(p_t_list),
            "queue": np.vstack(p_q_list),
            "priority": np.vstack(p_p_list),
        }


def _extract_aux_tag_matrix(df: pd.DataFrame, top_tags: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    tag_cols = [c for c in df.columns if c.startswith("tag_")]
    if top_tags is None:
        cnt: Counter[str] = Counter()
        for c in tag_cols:
            for val in df[c].dropna():
                s = str(val).strip()
                if s:
                    cnt[s] += 1
        top_tags = [t for t, _ in cnt.most_common(AUX_TAG_TOP_K)]

    tag2idx = {t: i for i, t in enumerate(top_tags)}
    mat = np.zeros((len(df), len(top_tags)), dtype=np.float32)
    for row_i, vals in enumerate(zip(*(df[c].tolist() for c in tag_cols))):
        for v in vals:
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                idx = tag2idx.get(str(v).strip())
                if idx is not None:
                    mat[row_i, idx] = 1.0
    return mat, top_tags


def train_multitask_transformer(epochs: int = 4, batch_size: int = 32, lr: float = 3e-5) -> dict[str, Any]:
    if not KB_TRAIN_PATH.exists():
        raise FileNotFoundError(f"{KB_TRAIN_PATH} not found.")

    kb_df = pd.read_parquet(KB_TRAIN_PATH)
    if HOLDOUT_TEST_PATH.exists():
        holdout_ids = set(pd.read_parquet(HOLDOUT_TEST_PATH, columns=["ticket_id"])["ticket_id"])
        assert len(set(kb_df["ticket_id"]) & holdout_ids) == 0, "LEAKAGE ERROR!"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_xlmr] Training Multi-Task {XLMR_MODEL_NAME} on {device} (bf16={device == 'cuda'})...")
    t0 = time.perf_counter()

    # EXACT same internal 85/15 split inside kb_train as train_ml.py
    strat_key = kb_df["type"].astype(str) + "__" + kb_df["priority"].astype(str)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    tr_idx, val_idx = next(sss.split(kb_df, strat_key))

    tr_df = kb_df.iloc[tr_idx].reset_index(drop=True)
    val_df = kb_df.iloc[val_idx].reset_index(drop=True)

    t2i = {c: i for i, c in enumerate(EXPECTED_TYPES)}
    q2i = {c: i for i, c in enumerate(EXPECTED_QUEUES)}
    p2i = {c: i for i, c in enumerate(EXPECTED_PRIORITIES)}

    y_tr_t = np.array([t2i[v] for v in tr_df["type"]], dtype=np.int64)
    y_tr_q = np.array([q2i[v] for v in tr_df["queue"]], dtype=np.int64)
    y_tr_p = np.array([p2i[v] for v in tr_df["priority"]], dtype=np.int64)
    y_tr_tags, top_tags = _extract_aux_tag_matrix(tr_df, top_tags=None)

    y_va_t = np.array([t2i[v] for v in val_df["type"]], dtype=np.int64)
    y_va_q = np.array([q2i[v] for v in val_df["queue"]], dtype=np.int64)
    y_va_p = np.array([p2i[v] for v in val_df["priority"]], dtype=np.int64)
    y_va_tags, _ = _extract_aux_tag_matrix(val_df, top_tags=top_tags)

    tr_texts = [format_xlmr_text(s, b) for s, b in zip(tr_df["subject"], tr_df["body"])]
    val_texts = [format_xlmr_text(s, b) for s, b in zip(val_df["subject"], val_df["body"])]

    tokenizer = AutoTokenizer.from_pretrained(XLMR_MODEL_NAME)

    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        enc = tokenizer(
            [b["text"] for b in batch],
            padding=True,
            truncation=True,
            max_length=XLMR_MAX_LEN,
            return_tensors="pt",
        )
        enc["y_type"] = torch.tensor([b["y_type"] for b in batch], dtype=torch.long)
        enc["y_queue"] = torch.tensor([b["y_queue"] for b in batch], dtype=torch.long)
        enc["y_prio"] = torch.tensor([b["y_prio"] for b in batch], dtype=torch.long)
        enc["y_tags"] = torch.tensor(np.stack([b["y_tags"] for b in batch]), dtype=torch.float32)
        return enc

    tr_loader = DataLoader(
        TicketTorchDataset(tr_texts, y_tr_t, y_tr_q, y_tr_p, y_tr_tags),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        TicketTorchDataset(val_texts, y_va_t, y_va_q, y_va_p, y_va_tags),
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn,
    )

    torch.manual_seed(42)
    model = MultiTaskXLMR(
        model_name=XLMR_MODEL_NAME,
        n_type=len(EXPECTED_TYPES),
        n_queue=len(EXPECTED_QUEUES),
        n_prio=len(EXPECTED_PRIORITIES),
        n_aux_tags=len(top_tags),
        pretrained=True,
    ).to(device)

    # Mild inverse-sqrt class weighting for the 21.7x imbalanced 10-queue head
    q_counts = np.bincount(y_tr_q, minlength=len(EXPECTED_QUEUES)).astype(np.float32)
    q_weights = np.power(q_counts.max() / np.clip(q_counts, 1.0, None), 0.35)
    q_weights = q_weights / q_weights.mean()

    loss_type_fn = nn.CrossEntropyLoss(label_smoothing=0.03)
    loss_queue_fn = nn.CrossEntropyLoss(weight=torch.tensor(q_weights, device=device), label_smoothing=0.04)
    loss_prio_fn = nn.CrossEntropyLoss(label_smoothing=0.03)
    loss_aux_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(tr_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.10),
        num_training_steps=total_steps,
    )

    best_score = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_val_probs: dict[str, np.ndarray] | None = None
    best_metrics: dict[str, float] = {}

    for ep in range(1, epochs + 1):
        ep_t0 = time.perf_counter()
        model.train()
        running_loss = 0.0
        for batch in tr_loader:
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            yt = batch["y_type"].to(device)
            yq = batch["y_queue"].to(device)
            yp = batch["y_prio"].to(device)
            ytags = batch["y_tags"].to(device)

            if device == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    lt, lq, lp, laux = model(input_ids, attn_mask)
                    loss = (
                        1.0 * loss_type_fn(lt, yt)
                        + 1.35 * loss_queue_fn(lq, yq)
                        + 1.15 * loss_prio_fn(lp, yp)
                        + 0.35 * loss_aux_fn(laux, ytags)
                    )
            else:
                lt, lq, lp, laux = model(input_ids, attn_mask)
                loss = (
                    1.0 * loss_type_fn(lt, yt)
                    + 1.35 * loss_queue_fn(lq, yq)
                    + 1.15 * loss_prio_fn(lp, yp)
                    + 0.35 * loss_aux_fn(laux, ytags)
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.item())

        # Evaluate on internal 15% val split
        model.eval()
        vt_probs, vq_probs, vp_probs = [], [], []
        with torch.inference_mode():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                if device == "cuda":
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        lt, lq, lp, _ = model(input_ids, attn_mask)
                else:
                    lt, lq, lp, _ = model(input_ids, attn_mask)
                vt_probs.append(torch.softmax(lt.float(), dim=-1).cpu().numpy())
                vq_probs.append(torch.softmax(lq.float(), dim=-1).cpu().numpy())
                vp_probs.append(torch.softmax(lp.float(), dim=-1).cpu().numpy())

        pt = np.vstack(vt_probs)
        pq = np.vstack(vq_probs)
        pp = np.vstack(vp_probs)

        f1_t = float(f1_score(y_va_t, np.argmax(pt, axis=1), average="macro"))
        f1_q = float(f1_score(y_va_q, np.argmax(pq, axis=1), average="macro"))
        f1_p = float(f1_score(y_va_p, np.argmax(pp, axis=1), average="macro"))
        avg_f1 = (f1_t + f1_q + f1_p) / 3.0
        ep_sec = time.perf_counter() - ep_t0

        print(
            f"[train_xlmr] Epoch {ep}/{epochs} ({ep_sec:.1f}s) | loss={running_loss/len(tr_loader):.4f} | "
            f"val F1: type={f1_t:.4f}, queue={f1_q:.4f}, prio={f1_p:.4f} | avg={avg_f1:.4f}"
        )

        if avg_f1 > best_score:
            best_score = avg_f1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_val_probs = {"type": pt, "queue": pq, "priority": pp}
            best_metrics = {
                "type_macro_f1": round(f1_t, 4),
                "queue_macro_f1": round(f1_q, 4),
                "priority_macro_f1": round(f1_p, 4),
                "avg_macro_f1": round(avg_f1, 4),
            }

    assert best_state is not None and best_val_probs is not None
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": XLMR_MODEL_NAME,
            "max_len": XLMR_MAX_LEN,
            "n_aux_tags": len(top_tags),
            "top_tags": top_tags,
            "val_metrics": best_metrics,
            "state_dict": best_state,
        },
        XLMR_CHECKPOINT_PATH,
    )
    np.savez_compressed(
        XLMR_VAL_PROBS_PATH,
        type=best_val_probs["type"],
        queue=best_val_probs["queue"],
        priority=best_val_probs["priority"],
    )
    total_sec = round(time.perf_counter() - t0, 1)
    print(f"[train_xlmr] Saved best checkpoint -> {XLMR_CHECKPOINT_PATH} ({total_sec}s) | {best_metrics}")
    return {"val_metrics": best_metrics, "train_seconds": total_sec}


if __name__ == "__main__":
    train_multitask_transformer()
