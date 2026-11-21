# Constraints, Rules & Verified Environment

**Verified Date:** 2026-11-21 (Local Windows Workstation)

---

## 1. Binding Rules & Non-Negotiable Compliance Boundaries

| Constraint | Source | Binding Implication | Verification Method |
|---|---|---|---|
| **Strict Zero Data Leakage** | Official Brief §3 & §7; Master Mission Phase 1 | Duplicate-text grouped 80/20 split via `StratifiedGroupKFold(n_splits=5)` (`splits/split_assignment.csv`). `kb_train.parquet` (`22,869`) ONLY for audit, training, calibration, threshold tuning, and FAISS indexing. `holdout_test.parquet` (`5,718`) labels and `answer` NEVER indexed, audited, or fed to inference. | Programmatic assertions in `data_prep.py`, `indexing.py`, `train_ml.py`, `evaluate.py`, and `tests/test_pipeline.py` verifying `len(train_ids & test_ids) == 0`, `0` duplicate-text leaks, and `index.ntotal == len(kb_train)`. |
| **Clean 10-Queue Dataset Only** | Master Mission Phase 1 | Load `aa_dataset-tickets-multi-lang-5-2-50-version.csv` (`28,587` clean rows, default) or `--20k` -> `dataset-tickets-multi-lang-4-20k.csv`. Never load merged legacy CSVs with 52 noisy queues. | Assert exact 4 `type` classes, 10 `queue` classes, 3 `priority` classes, 2 `language` classes (`en`/`de`). |
| **Null-Subject Resilience** | Official Brief §3; Master Mission | Missing/empty/whitespace `subject` must never crash embedding, feature extraction, triage, retrieval, or UI rendering; must report `subject_missing: true` and degrade < 3% Macro-F1. | Explicit null-safe formatting (`title: none | text: ...`), dedicated holdout slice evaluation, unit + UI preset test. |
| **Zero Raw PII Egress** | Master Mission Bonus #3; GDPR/HIPAA/EU AI Act Art. 50 | Emails, IPv4/IPv6, US/DACH phones, IBANs, credit/account/patient IDs scrubbed locally into indexed placeholders (`[EMAIL_1]`, `[IP_1]`, etc.) **before** any cloud API call. | Adversarial suite checks outbound cloud payload contains 0 raw PII tokens; UI renders "What Left Your Network" diff drawer. |
| **EmbeddingGemma 2 Precision Guard** | Master Mission §3.1 | Never run `google/embeddinggemma-2` in `float16` (causes NaNs). Must use `bfloat16` on CUDA or `float32` on CPU, with `config_kwargs={"vision_config": None, "audio_config": None}` (text-only 270M). | Verified `np.isnan(embeddings).sum() == 0` and unit norm check (`~1.0`) in `indexing.py`. |
| **No Fabricated Policies / Citations** | Official Brief §4; Master Mission §3.1 | Every `[TKT-XXXXX]` citation must exist in retrieved top-5; no invented refund/SLA/currency/percentage claims absent from retrieved `answer` texts. | Deterministic Policy Guard + Pass-2 Grounding Auditor veto flips unsafe drafts to `ESCALATE_ON_CALL`. |

---

## 2. Measured Hardware & Local Runtime Capabilities

| Component | Measured State | Operational Effect |
|---|---|---|
| **GPU** | NVIDIA GeForce RTX 5070 Laptop GPU (8 GB VRAM, CUDA Driver 13.3) | Enables fast `bfloat16` CUDA inference for `google/embeddinggemma-2` (270M text-only) and GPU-accelerated local Ollama `gemma4:e4b` drafting. |
| **Python Environment** | Isolated `.venv` (Python 3.11.9, PyTorch `cu128`, `sentence-transformers>=6.1.0`, `google-genai>=3.2.0`, `faiss-cpu`, `scikit-learn`, `rapidfuzz`, `lingua-language-detector`, `fastapi`) | Full native support for `google/embeddinggemma-2`, Calibrated LinearSVC, `RapidFuzz` grounding, and `client.decisions.create(model="gemini-decision-1", ...)`. |
| **Node / Frontend** | Node `v24.11.1`, npm `11.6.2` | Clean React + Vite + Tailwind production build served directly by FastAPI and auto-deployed to GitHub Pages (`.github/workflows/deploy-pages.yml`). |
| **Hugging Face Access** | `google/embeddinggemma-2` confirmed `gated: False, private: False` | Direct download without needing HF auth token; automatic fallback to `intfloat/multilingual-e5-small` preserved for portability. |
| **Local Air-Gapped Mode** | Strictly `0` cloud API calls (`cloud_calls_made == 0`); warm-only Ollama check (`/api/ps`) + instant On-Device Precedent Synthesizer + `RapidFuzz` auditor | Sub-200 ms deterministic air-gapped execution without Windows Blackwell cold-start timeouts (`ollama 0.30.8`). |
| **Cloud Gemini API Key** | Active in gitignored `.env` (never committed or logged) | Live `gemini-decision-1` Pass-1/Pass-2 + live `gemini-3.5-flash-lite` (`~655 ms`) PII-scrubbed cloud drafting; graceful fallback to Air-Gapped Local mode if unset. |
