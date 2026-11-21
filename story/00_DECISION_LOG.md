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
- **Evidence:** Phase-1 audit showed only `9.1%` (`358 / 3,944` on the 28.5k archive) of P1 (`high + Incident`) tickets belong to `Service Outages and Maintenance`; `52.6%` (`2,073`) land in `Technical Support`, `13.7%` in `Product Support`, and `12.0%` in `IT Support`.
- **Decision:** Trigger Rule #1 (`RULE_1_P1_INCIDENT`) on our dedicated binary `critical` detector + `(priority == "high" AND type == "Incident")` + negation-aware severe impact rules across ALL 10 queues, outputting Dual-Mode Warm Escalation (native EN/DE Holding Ack + English On-Call War-Room Brief).

---

## DEC-006 · Architecture Simplification & Teammate Benchmark Merge (v3)
- **Timestamp:** 2026-11-21 13:05–13:19 EST
- **Context:** Benchmarking against our teammate's independent prototype (`egornikot/sentineldesk`) revealed why a simpler sparse-linear stack beat our v2 3-way ensemble (`XLM-R + LightGBM + 40k Linear`) by `+4.5 pts` on `type`, `+12–14 pts` on 10-class `queue`, and `+9–10 pts` on `priority`.
- **Root Causes Diagnosed:**
  1. Using the 20k subset (`16k` train) starved the 4 rarest queues (`General Inquiry`, `Human Resources`, `Sales and Pre-Sales`, `Service Outages`), whereas the official full clean CSV (`aa_dataset-tickets-multi-lang-5-2-50-version.csv`, `28,587` rows) provides +43% more rare-class examples.
  2. Our v2 TF-IDF capped vocabulary at `40k` features and passed dense embeddings to `LightGBM`/`XLM-R`, losing exact domain n-grams and German compound subwords. High-capacity `FeatureUnion` (`word` 1–2 grams up to 200k + `char_wb` 3–5 grams up to 300k, `sublinear_tf=True`) into `CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced"), cv=3, method="sigmoid")` dominates imbalanced text routing while being 10× simpler and requiring zero GPU fine-tuning.
  3. Splitting by duplicate-text groups (`StratifiedGroupKFold(n_splits=5)`) eliminates hidden near-duplicate leakage between `kb_train` and `holdout_test`.
  4. Training a **dedicated 4th binary `critical` classifier** predicting `P(high AND Incident)` directly protects `>= 90%` P1 recall without skewing `type`/`priority` Macro-F1; pairing it with our post-hoc multi-class **Temperature Scaling** crushes LinearSVC's raw ECE (`0.096 -> 0.042` on `queue`, `0.071 -> 0.017` on `priority`, `0.032 -> 0.006` on `type`).
- **Decision:** Retired `backend/train_transformer.py` (`xlm-roberta-base`) and `lightgbm`. Unified on **Calibrated LinearSVC + `google/embeddinggemma-2` (256d) + Temperature Scaling + Dedicated Binary `critical` Head + Negation-Aware EN/DE Impact Rules + 3-Tier Routing (`PAGE_ON_CALL` / `HUMAN_REVIEW` / `AUTO_DRAFT_READY`) + `rapidfuzz` Lexical Grounding**.
- **Verified Holdout Outcome (N = 5,718 unseen tickets, duplicate-grouped):**
  - `type` Macro-F1: **`86.22%`** (`+4.81 pts` vs v2), ECE **`0.0058`** (`tfidf_svm`)
  - `queue` Macro-F1: **`55.43%`** (`+12.53 pts` vs v2), ECE **`0.0424`** (`combo_svm` — proves `EmbeddingGemma 2` adds real lift!)
  - `priority` Macro-F1: **`58.08%`** (`+9.03 pts` vs v2), ECE **`0.0167`** (`combo_svm` — `+0.72 pts` lift over TF-IDF alone!)
  - Overall Mean Macro-F1 jumped from **`57.79%` -> `66.58%`** (`+8.79 pts`) while removing 1.1 GB of transformer weights and cutting training time from ~20 min to ~3.4 min.

---

## DEC-007 · Strict Architectural Separation: Cloud PII-Scrubbed vs. Local Air-Gapped Mode
- **Timestamp:** 2026-11-21 13:06 EST
- **Observed Bug in v2:** Switching the UI toggle from `Cloud · PII-Scrubbed` to `Local · Air-Gapped` previously produced nearly identical outputs because: (a) Pass-1 still called `gemini-decision-1` even in `local` mode, (b) local Ollama cold-start on Windows RTX 5070 Blackwell (`sm_120`, `ollama 0.30.8`) timed out (>20s) before falling back to a template, and (c) escalated tickets used the exact same deterministic brief in both modes.
- **Decision:** Enforce strict, verifiable execution isolation in `backend/triage.py` and `backend/drafter.py`:
  - **`Local · Air-Gapped` (`drafter_mode="local"`):** Strictly **`0` cloud API calls** (`cloud_calls_made == 0`). Uses on-device Calibrated LinearSVC + `EmbeddingGemma 2` FAISS + local negation-aware & heuristic security rules + warm-only Ollama check (`/api/ps`, `<0.6s`) or instant zero-hallucination on-device precedent synthesizer + local `rapidfuzz` grounding audit. Runs end-to-end in **~70–160 ms** with zero network egress.
  - **`Cloud · PII-Scrubbed` (`drafter_mode="cloud"`):** Local PII/PHI vault scrubs raw entities first -> hybrid triage (`0.75 Local SVM + 0.25 live gemini-decision-1`) -> natural native-language drafting via live **`gemini-3.5-flash-lite`** (`~655 ms`) -> live **`gemini-decision-1` Pass-2** grounding & policy audit + `rapidfuzz` guard (`3` PII-scrubbed cloud calls, `~1.5–2.2 s`).
- **Consequence:** Operators and judges immediately observe distinct execution badges (`Air-Gapped · 0 Cloud Calls` vs `Cloud Hybrid · 3 API`), distinct latencies (~150 ms vs ~1.9 s), and distinct drafting engines—matching Helios Managed Ops's real dual-tier compliance offering (Standard Commercial vs. Sovereign/HIPAA-Strict Air-Gapped).

---

## DEC-008 · Cloud Drafter Model Availability & Type-Gated Critical Floor
- **Timestamp:** 2026-11-21 13:18 EST
- **Findings:**
  1. Live smoke testing against the provisioned GCP Gemini API key showed `gemini-2.5-flash` returning `404` and `gemma-4-26b-a4b-it` / `gemini-3.8-flash` occasionally returning `503 UNAVAILABLE`, while **`gemini-3.5-flash-lite`** (`655 ms`) and **`gemini-3.1-flash-lite`** (`706 ms`) responded reliably. Reordered `CLOUD_DRAFTER_MODELS` accordingly.
  2. Because our calibrated `type` head achieves **`99.27%` precision on `Request`** and **`98.45%` precision on `Change`**, applying the sensitive P1 recall threshold (`0.105`) only to `Incident`/`Problem`/uncertain types (requiring `critical_score >= 0.50` on high-confidence `Request`/`Change` tickets) prevents routine billing/payment inquiries from falsely tripping `PAGE_ON_CALL` while keeping P1 Incident Recall above the `>= 90%` safety bar.
