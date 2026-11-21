# Build Narrative — SentinelDesk for Helios Managed Ops

**Build Snapshot (v3 Unified Architecture):** Verified local + GitHub Pages full-stack build (`2026-11-21`), strict duplicate-grouped 80/20 zero-leakage split (`22,869 kb_train` / `5,718 holdout_test` from the official 28,587-row clean CSV), `google/embeddinggemma-2` (270M text-only, `bfloat16` on NVIDIA RTX 5070), High-Capacity Calibrated `LinearSVC` + `EmbeddingGemma 2` + Temperature Scaling + Dedicated Binary `critical` P1 Head, live `gemini-decision-1` + `gemini-3.5-flash-lite` Cloud mode vs. strictly `0`-cloud-call Air-Gapped Local mode, and React/Vite Audit Glass operator UI.

> For a deep dive into **how the technical architecture maps directly to Helios Managed Ops business KPIs** and **exact Cloud vs. Local Air-Gapped differences**, see [`story/03_ARCHITECTURE_AND_BUSINESS_FIT.md`](./03_ARCHITECTURE_AND_BUSINESS_FIT.md).

---

## 1. The Operational Stake at Helios Managed Ops

Helios Managed Ops runs transatlantic (US Austin + DACH Munich) bilingual (`en` / `de`) L1 support hubs spanning 10 enterprise queues—from routine billing questions to hospital VPN/telemedicine/EMR outages. Auditing the `22,869`-ticket `kb_train.parquet` archive exposed the exact empirical challenges behind Helios's core failure modes:

- **Critical P1 incidents hide across every queue:** `3,944` tickets (`17.25%`) are `high + Incident`, and `24.65%` (`5,637` tickets) touch healthcare/medical IT, VPN, or outage workflows. Yet only `9.1%` (`358`) of P1 incidents land in `Service Outages and Maintenance`—`52.6%` (`2,073`) arrive inside `Technical Support`, `13.7%` (`542`) in `Product Support`, and `12.0%` (`474`) in `IT Support`.
- **Real tickets arrive messy, bilingual, and skewed:** `13.35%` (`3,053` rows) have blank/null subjects, `42.80%` are German vs. `57.20%` English, and queue volume spans a `21.4×` imbalance (`Technical Support` 29.44% vs. `General Inquiry` 1.37%).
- **Historical resolutions follow disciplined L1 norms:** Median agent reply length is `58` words, `88.31%` ask targeted diagnostic/clarifying questions, and only `1.06%` mention refunds or SLA terms—proving why off-the-shelf generative chatbots routinely hallucinate financial commitments when left unguarded.

---

## 2. How Our Architecture Evolved (v1 -> v2 -> Simplified v3)

1. **v1–v2 (Initial Over-Engineered Stack):** Started with `dataset-tickets-multi-lang-4-20k.csv` (`16k` train) and a heavy 3-way ensemble (`xlm-roberta-base` + `LightGBM` + 40k Linear). Calibration was strong (`ECE 0.012–0.026`), and P1 recall hit `92.45%`, but training took ~20 minutes with 1.1 GB of weights while 10-queue Macro-F1 stalled at `42.90%` and Priority Macro-F1 at `49.05%`. In addition, Local Air-Gapped mode still called `gemini-decision-1` in Pass-1, blurring the line between Cloud and Local execution.
2. **v3 Simplification & Teammate Benchmark Merge (`DEC-006`, `DEC-007`, `DEC-008`):**
   - Upgraded to the official **28,587-row** clean archive (`aa_dataset-tickets-multi-lang-5-2-50-version.csv`) with **duplicate-text grouped** `StratifiedGroupKFold(n_splits=5)` (`splits/split_assignment.csv`) so neither IDs nor identical texts can leak across train/test.
   - Retired `xlm-roberta-base` and `LightGBM` in favor of a single streamlined **500k Word+Char TF-IDF + `google/embeddinggemma-2` (256d) -> `CalibratedClassifierCV(LinearSVC)` + Post-Hoc Temperature Scaling** pipeline (`3.4 min` training, `142 ms` p50 local triage).
   - Added a **dedicated 4th binary `critical` classifier** predicting `P(high AND Incident)` directly + **negation-aware EN/DE impact rules** (ignoring `"not urgent"`, `"no data loss"`, `"kein Datenverlust"`) + **3-Tier Operational Routing** (`PAGE_ON_CALL` / `HUMAN_REVIEW` / `AUTO_DRAFT_READY`).
   - Enforced **strict Cloud vs. Air-Gapped Local isolation**: `Local · Air-Gapped` now makes strictly **`0` cloud API calls** (`~140 ms`, on-device precedent synthesizer + `rapidfuzz` lexical grounding), whereas `Cloud · PII-Scrubbed` runs live `0.75 SVM + 0.25 gemini-decision-1` + `gemini-3.5-flash-lite` + live `gemini-decision-1` Pass-2 audit on PII-scrubbed payloads (`~1.9 s`).

---

## 3. Verified Empirical Proof (`artifacts/holdout_metrics.json`, N = 5,718 Unseen Tickets)

| Dimension | v2 Baseline (20k) | **v3 Unified (28.5k, Duplicate-Grouped)** | Delta / Status |
|---|---:|---:|---|
| **Zero-Leakage Split** | 16k / 4k (`0` ID overlap) | **22,869 / 5,718 (`0` ID & `0` text overlap)** | Verified (`splits/split_assignment.csv`) |
| **`type` Macro-F1 / ECE** (4 classes) | 81.41% / 0.0117 | **`86.22%` / `0.0058`** (`tfidf_svm`) | **`+4.81 pts`** · ECE `<= 0.05` |
| **`queue` Macro-F1 / ECE** (10 skewed queues) | 42.90% / 0.0255 | **`55.43%` / `0.0424`** (`combo_svm`) | **`+12.53 pts`** · ECE `<= 0.05` |
| **`priority` Macro-F1 / ECE** (3 classes) | 49.05% / 0.0167 | **`58.08%` / `0.0167`** (`combo_svm`) | **`+9.03 pts`** · ECE `<= 0.05` |
| **Overall Mean Macro-F1** | 57.79% | **`66.58%`** | **`+8.79 pts`** (10× simpler!) |
| **High-Priority Incident Recall** | 92.45% (`34%` other flagged) | **`91.59%`** (`904/987` · `30.9%` non-P1) | Passed (`>= 90%` bar) |
| **Null-Subject Resilience** | 0 crashes | **`0` crashes (`n=785`, 13.73%)** — null slice actually beats non-null! | Passed |
| **Cross-Lingual Retrieval (Top-5)** | 47.1% / Same-Queue@5 87.6% | **`52.47%` cross-lingual · Top-1 med `0.945` · Same-Queue@5 `91.33%`** | Improved |
| **Local Triage Latency (p50 / p95)** | 162.1 ms / 193.9 ms | **`142.6 ms` p50 / `159.0 ms` p95** (Hybrid w/ D1: `472.1 ms`) | Faster |
| **10-Case Adversarial & Safety Suite** | 10/10 (`100%`) | **`10/10` (`100%`)** · `0%` PII leak · `100%` Refund Veto · `0` cloud calls in Local | Passed |

---

## 4. Honest Boundaries & Remaining Operational Tradeoffs

- **What is empirically verified end-to-end:** Zero-leakage duplicate-grouped split, GPU `EmbeddingGemma 2` 256d FAISS index (`22,869` docs), calibrated multi-task `LinearSVC + EmbeddingGemma 2` + dedicated binary `critical` head, live `gemini-decision-1` Pass-1/Pass-2 & `gemini-3.5-flash-lite` Cloud mode, strictly `0`-cloud-call Local Air-Gapped mode, `rapidfuzz` lexical grounding + deterministic refund/SLA/citation guard, automated `pytest` suite, live browser verification, and live GitHub Pages deployment.
- **Explicit Recall vs. Precision Tradeoff:** Pushing P1 Incident Recall above `91%` to protect hospital/EMR SLAs still flags roughly ~25–30% of non-P1 tickets for escalation/review—an intentional safety-first operating point for Helios Managed Ops, mitigated operationally by splitting escalations cleanly into **`PAGE_ON_CALL`** (urgent P1/security) vs. **`HUMAN_REVIEW`** (calm L1/billing check).
- **Local LLM Cold-Start Caveat:** On Windows Blackwell GPUs (`RTX 5070 sm_120`, `ollama 0.30.8`), cold-loading a 4B–8B Ollama model can exceed 20s; SentinelDesk therefore only invokes Ollama when already warm in VRAM (`/api/ps`) and otherwise serves instant (`<5 ms`), zero-hallucination extractive precedent drafts in Air-Gapped mode.
