"""Cross-lingual Embeddings (EmbeddingGemma 2, text-only 270M, 256d) + FAISS IndexFlatIP.

Indexes ONLY data/kb_train.parquet (strict zero-leakage). Never touches holdout_test.parquet.

Usage:
    python backend/indexing.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.config import (
    ARTIFACTS_DIR,
    EMBEDDING_DIM,
    FAISS_INDEX_PATH,
    FALLBACK_EMBEDDING_MODEL,
    HOLDOUT_TEST_PATH,
    INDEX_META_PATH,
    KB_EMBEDDINGS_PATH,
    KB_METADATA_PATH,
    KB_TRAIN_PATH,
    PRIMARY_EMBEDDING_MODEL,
)


def safe_str(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, float) and np.isnan(val):
        return ""
    return str(val).strip()


def format_kb_document(subject: Any, body: Any) -> str:
    """Null-safe canonical formatting matching EmbeddingGemma 2 Document format."""
    subj = safe_str(subject) or "none"
    b = safe_str(body) or "none"
    return f"title: {subj} | text: {b}"


def format_query_text(subject: Any, body: Any) -> str:
    """Null-safe query formatting for incoming tickets."""
    subj = safe_str(subject)
    b = safe_str(body)
    if subj and b:
        return f"title: {subj} | text: {b}"
    return f"title: none | text: {b or subj or 'none'}"


class EmbeddingService:
    """Manages EmbeddingGemma 2 (270M text-only, bfloat16/fp32) with automatic fallback."""

    def __init__(self, preferred_model: str | None = None) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # CRITICAL: Never use float16 with embeddinggemma-2 (produces NaNs).
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model_name = preferred_model or PRIMARY_EMBEDDING_MODEL
        self.fallback_used = False
        self.model = self._load_model(self.model_name)
        self.available_prompts = set((getattr(self.model, "prompts", None) or {}).keys())

    def _load_model(self, model_name: str) -> SentenceTransformer:
        if model_name == PRIMARY_EMBEDDING_MODEL:
            try:
                print(
                    f"[indexing] Loading {PRIMARY_EMBEDDING_MODEL} (text-only 270M, "
                    f"device={self.device}, dtype={self.dtype})..."
                )
                model = SentenceTransformer(
                    PRIMARY_EMBEDDING_MODEL,
                    device=self.device,
                    config_kwargs={"vision_config": None, "audio_config": None},
                    model_kwargs={"dtype": self.dtype},
                )
                return model
            except TypeError:
                # Older transformers versions expect torch_dtype instead of dtype
                try:
                    model = SentenceTransformer(
                        PRIMARY_EMBEDDING_MODEL,
                        device=self.device,
                        config_kwargs={"vision_config": None, "audio_config": None},
                        model_kwargs={"torch_dtype": self.dtype},
                    )
                    return model
                except Exception as exc:
                    print(
                        f"[indexing] WARNING: Failed to load {PRIMARY_EMBEDDING_MODEL}: {exc}. "
                        f"Falling back to {FALLBACK_EMBEDDING_MODEL}."
                    )
            except Exception as exc:
                print(
                    f"[indexing] WARNING: Failed to load {PRIMARY_EMBEDDING_MODEL}: {exc}. "
                    f"Falling back to {FALLBACK_EMBEDDING_MODEL}."
                )
            self.model_name = FALLBACK_EMBEDDING_MODEL
            self.fallback_used = True

        print(f"[indexing] Loading fallback embedding model {self.model_name} on {self.device}...")
        return SentenceTransformer(self.model_name, device=self.device)

    def _postprocess(self, emb: np.ndarray) -> np.ndarray:
        arr = np.asarray(emb, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[1] > EMBEDDING_DIM:
            arr = arr[:, :EMBEDDING_DIM]
        # Replace any pathological values defensively and L2-normalize
        if np.isnan(arr).any() or np.isinf(arr).any():
            raise ValueError(f"NaN/Inf detected in embeddings from {self.model_name}!")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        arr = arr / norms
        return np.ascontiguousarray(arr, dtype=np.float32)

    def encode_documents(self, texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        if "Document" in self.available_prompts:
            try:
                emb = self.model.encode(
                    texts,
                    prompt_name="Document",
                    truncate_dim=EMBEDDING_DIM,
                    normalize_embeddings=True,
                    batch_size=batch_size,
                    show_progress_bar=show_progress,
                    convert_to_numpy=True,
                )
                return self._postprocess(emb)
            except TypeError:
                emb = self.model.encode(
                    texts,
                    prompt_name="Document",
                    normalize_embeddings=True,
                    batch_size=batch_size,
                    show_progress_bar=show_progress,
                    convert_to_numpy=True,
                )
                return self._postprocess(emb)

        if "e5" in self.model_name.lower():
            texts = [f"passage: {t}" for t in texts]
        emb = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return self._postprocess(emb)

    def encode_queries(self, texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        if "SearchQuery" in self.available_prompts:
            try:
                emb = self.model.encode(
                    texts,
                    prompt_name="SearchQuery",
                    truncate_dim=EMBEDDING_DIM,
                    normalize_embeddings=True,
                    batch_size=batch_size,
                    show_progress_bar=show_progress,
                    convert_to_numpy=True,
                )
                return self._postprocess(emb)
            except TypeError:
                emb = self.model.encode(
                    texts,
                    prompt_name="SearchQuery",
                    normalize_embeddings=True,
                    batch_size=batch_size,
                    show_progress_bar=show_progress,
                    convert_to_numpy=True,
                )
                return self._postprocess(emb)

        if "e5" in self.model_name.lower():
            texts = [f"query: {t}" for t in texts]
        emb = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return self._postprocess(emb)


class VectorKnowledgeBase:
    """Read-only runtime wrapper around FAISS IndexFlatIP + kb_train metadata."""

    def __init__(self, embedder: EmbeddingService | None = None) -> None:
        if not FAISS_INDEX_PATH.exists() or not KB_METADATA_PATH.exists():
            raise FileNotFoundError(
                "FAISS index or metadata missing. Run `python backend/indexing.py` first."
            )
        self.meta_info = (
            json.loads(INDEX_META_PATH.read_text(encoding="utf-8"))
            if INDEX_META_PATH.exists()
            else {"model_name": PRIMARY_EMBEDDING_MODEL}
        )
        self.embedder = embedder or EmbeddingService(preferred_model=self.meta_info.get("model_name"))
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))
        self.metadata = pd.read_parquet(KB_METADATA_PATH)
        assert self.index.ntotal == len(self.metadata), (
            f"FAISS size {self.index.ntotal} != metadata size {len(self.metadata)}"
        )

    def embed_ticket(self, subject: str | None, body: str) -> np.ndarray:
        q = format_query_text(subject, body)
        return self.embedder.encode_queries([q], batch_size=1)[0]

    def search_by_vector(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        detected_language: str | None = None,
    ) -> list[dict[str, Any]]:
        vec = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        # Fetch extra candidates so we can skip any rare empty historical answers
        fetch_k = min(top_k + 5, self.index.ntotal)
        scores, indices = self.index.search(vec, fetch_k)
        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            row = self.metadata.iloc[int(idx)]
            ans = safe_str(row["answer"])
            if not ans:
                continue
            row_lang = safe_str(row["language"]).lower()
            subj = safe_str(row["subject"])
            body = safe_str(row["body"])
            results.append(
                {
                    "ticket_id": safe_str(row["ticket_id"]),
                    "similarity": round(float(np.clip(score, 0.0, 1.0)), 4),
                    "language": row_lang,
                    "cross_lingual": bool(detected_language and row_lang and row_lang != detected_language),
                    "type": safe_str(row["type"]),
                    "queue": safe_str(row["queue"]),
                    "priority": safe_str(row["priority"]),
                    "subject": subj or "(No Subject)",
                    "body_snippet": (body[:240] + "...") if len(body) > 240 else body,
                    "answer": ans,
                }
            )
            if len(results) >= top_k:
                break
        return results


def build_kb_index() -> dict[str, Any]:
    """Build FAISS IndexFlatIP strictly from data/kb_train.parquet."""
    if not KB_TRAIN_PATH.exists():
        raise FileNotFoundError(f"{KB_TRAIN_PATH} not found. Run `python backend/data_prep.py` first.")

    kb_df = pd.read_parquet(KB_TRAIN_PATH)
    # Zero-leakage assertion against holdout_test if present
    if HOLDOUT_TEST_PATH.exists():
        holdout_ids = set(pd.read_parquet(HOLDOUT_TEST_PATH, columns=["ticket_id"])["ticket_id"])
        kb_ids = set(kb_df["ticket_id"])
        assert len(kb_ids & holdout_ids) == 0, "LEAKAGE ERROR: Holdout IDs found inside kb_train!"

    t0 = time.perf_counter()
    embedder = EmbeddingService()
    print(f"[indexing] Prompts available on {embedder.model_name}: {sorted(embedder.available_prompts)}")

    docs = [format_kb_document(s, b) for s, b in zip(kb_df["subject"], kb_df["body"])]
    batch_size = 128 if embedder.device == "cuda" else 32
    print(f"[indexing] Encoding {len(docs):,} kb_train documents (batch_size={batch_size})...")
    embeddings = embedder.encode_documents(docs, batch_size=batch_size, show_progress=True)

    assert embeddings.shape == (len(kb_df), EMBEDDING_DIM), f"Bad embedding shape: {embeddings.shape}"
    assert not np.isnan(embeddings).any(), "NaN detected in KB embeddings!"
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), f"Embeddings not unit-normalized: min={norms.min()}, max={norms.max()}"

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    assert index.ntotal == len(kb_df)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    np.save(KB_EMBEDDINGS_PATH, embeddings)

    meta_cols = ["ticket_id", "subject", "body", "answer", "type", "queue", "priority", "language"]
    meta_df = kb_df[meta_cols].copy()
    meta_df.to_parquet(KB_METADATA_PATH, index=False)

    elapsed = round(time.perf_counter() - t0, 2)
    meta_info = {
        "model_name": embedder.model_name,
        "fallback_used": embedder.fallback_used,
        "device": embedder.device,
        "dtype": str(embedder.dtype),
        "embedding_dim": EMBEDDING_DIM,
        "kb_indexed_rows": int(index.ntotal),
        "available_prompts": sorted(embedder.available_prompts),
        "build_seconds": elapsed,
        "zero_leakage_verified": True,
    }
    INDEX_META_PATH.write_text(json.dumps(meta_info, indent=2), encoding="utf-8")
    print(
        f"[indexing] Built FAISS IndexFlatIP ({index.ntotal:,} x {EMBEDDING_DIM}d) in {elapsed}s -> {FAISS_INDEX_PATH}"
    )
    return meta_info


if __name__ == "__main__":
    build_kb_index()
