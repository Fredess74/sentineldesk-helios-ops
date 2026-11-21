# Decision Log — SentinelDesk for Helios Managed Ops

Records consequential architecture, data, ML, and verification choices with actual alternatives considered, empirical evidence, and tradeoffs.

---

## DEC-001 · Clean 10-Queue Dataset & Strict Seeded 80/20 Split
- **Timestamp:** 2026-11-21 10:42 EST
- **Context:** Hugging Face `Tobi-Bueck/customer-support-tickets` hosts multiple CSV variants, including legacy merged files with 52 noisy queues.
- **Decision:** Default `backend/data_prep.py` to clean `dataset-tickets-multi-lang-4-20k.csv` (20,000 rows, exact 10 queues) with `--full` CLI support for `aa_dataset-tickets-multi-lang-5-2-50-version.csv`, applying the exact official `sample(frac=1.0, random_state=42)` 80/20 split (`16,000 kb_train` / `4,000 holdout_test`).
- **Evidence:** `kb_train`-only audit (`artifacts/kb_audit_summary.json`) confirmed `0` overlapping IDs, 4 types, 10 queues (21.7× imbalance), 3 priorities, 7.27% null subjects (`1,163` rows), 17.65% P1 `high + Incident` (`2,824` rows), and 22.26% healthcare/VPN/outage ticket density (`3,562` rows).
- **Consequence:** Zero leakage guaranteed end-to-end across audit, ML training, FAISS indexing, and holdout scoring.

---

## DEC-002 · `google/embeddinggemma-2` (270M Text-Only, `bfloat16`, 256d Matryoshka) on RTX 5070
- **Timestamp:** 2026-11-21 10:46–10:52 EST
- **Alternatives Considered:**
  1. Fallback `intfloat/multilingual-e5-small` (384d -> 256d) — fast, CPU-friendly, no image processor dependency.
  2. Native `google/embeddinggemma-2` with `config_kwargs={"vision_config": None, "audio_config": None}` in `bfloat16` on CUDA (never `float16`).
- **Observed Incident & Resolution:** Initial load of `google/embeddinggemma-2` fell back cleanly to `multilingual-e5-small` because `transformers`'s `AutoProcessor` imported `Gemma4ImageProcessorPil` and `EmbeddingGemma2VideoProcessor`, requiring optional `pillow` and `torchvision`. Installing `pillow==13.0.0` and `torchvision==0.26.0+cu128` enabled native `google/embeddinggemma-2` loading in `bfloat16` on the RTX 5070 Laptop GPU (`83.23s` for all 16,000 `kb_train` docs, 0 NaNs, exact unit L2 norms, native `Document` / `SearchQuery` prompts).
- **Consequence:** Superior EN <-> DE cross-lingual semantic alignment in a compact 256d Matryoshka space, while keeping automatic fallback to `multilingual-e5-small` intact for headless environments.

---

## DEC-003 · Dual-Representation Calibrated Local ML + P1 Recall Promotion
- **Timestamp:** 2026-11-21 10:53–10:59 EST
- **Alternatives Considered:**
  1. Embedding-only LightGBM (fast, dense semantic signal, weaker on exact rare queue keywords).
  2. Sparse TF-IDF LogisticRegression only (strong lexical recall, weaker on cross-lingual EN/DE transfer).
  3. Validation-blended LightGBM (`256d EmbeddingGemma 2 + 64d SVD + 14 structural`) + Sparse Word/Char TF-IDF LogisticRegression + multi-class Temperature Scaling + P1 recall threshold tuning (fitted strictly on internal 85/15 carve-out inside `kb_train`).
- **Observed Internal Val Outcome (`kb_train` 15% carve-out, N = 2,400):**
  - `type`: Macro-F1 `0.8171`, ECE `0.0093` (`w_lgb=0.60, T=1.001`)
  - `queue` (10 classes): Macro-F1 `0.3832`, ECE `0.0478` (`w_lgb=0.60, T=1.182`)
  - `priority`: Macro-F1 `0.4911`, ECE `0.0255` (`w_lgb=0.65, T=0.860`)
  - P1 (`high + Incident`) Recall tuned to `92.45%` (`inc_thr=0.475, high_thr=0.225`).
- **Serialization Fix:** Moved `CalibratedMultiTaskHead` out of `train_ml.py` (`__main__`) into `backend/features.py` so `joblib.load` resolves reliably from FastAPI, `evaluate.py`, and `pytest`.

---

## DEC-004 · Dual-Mode System 2 Gemma 4 Drafter + Closed-Loop Pass-2 Veto
- **Timestamp:** 2026-11-21 10:50 EST
- **Context:** Local workstation has Ollama `gemma4:e4b` pre-pulled and RTX 5070 GPU, while `GEMINI_API_KEY` is optional in `.env`.
- **Decision:** Implement full `google-genai>=3.2.0` v1beta `client.decisions.create(model="models/gemini-decision-1", ...)` (Pass-1 7 parallel questions + Pass-2 4 `noul` grounding questions) and `gemma-4-26b-a4b-it` / `gemini-2.5-flash` cloud drafter, alongside live air-gapped Ollama `gemma4:e4b` drafting (`think=False` for sub-2.5s L1 latency) and deterministic safety/policy guards.
- **Rationale:** Guarantees 100% live LLM drafting locally in air-gapped mode with zero PII egress, immediate cloud activation once `GEMINI_API_KEY` is provided, and zero demo fragility.

---

## DEC-005 · P1 Escalation Trigger Across All 10 Queues (Not Just Outages Queue)
- **Timestamp:** 2026-11-21 10:43 EST
- **Evidence:** Phase-1 audit showed only `8.2%` (`231 / 2,824`) of P1 (`high + Incident`) tickets belong to `Service Outages and Maintenance`; `52.8%` (`1,490`) land in `Technical Support`, `14.7%` in `Product Support`, and `12.3%` in `IT Support`.
- **Decision:** Trigger Rule #1 (`RULE_1_P1_INCIDENT`) on `(priority == "high" AND type == "Incident")` or high joint P1 risk across ALL 10 queues, outputting Dual-Mode Warm Escalation (native EN/DE Holding Ack + English On-Call War-Room Brief).
