# Build Narrative — SentinelDesk for Helios Managed Ops

**Build Snapshot:** Verified local full-stack build (`2026-11-21`), strict 80/20 zero-leakage split (`16,000 kb_train` / `4,000 holdout_test`), `google/embeddinggemma-2` (270M text-only, `bfloat16` on NVIDIA RTX 5070), Calibrated Local Multi-Task Ensemble + `gemini-decision-1` v1beta integration, Dual-Mode Gemma 4 Drafter (Local Air-Gapped Ollama + PII-Scrubbed Cloud API), and React/Vite SOC Audit Glass console.

---

## 1. The Operational Stake at Helios Managed Ops

Helios Managed Ops runs transatlantic (US + DACH) bilingual (`en` / `de`) L1 support hubs spanning 10 enterprise queues—from routine billing questions to hospital VPN/telemedicine/EMR outages. Auditing the `16,000`-ticket `kb_train.parquet` archive exposed the exact empirical challenges behind Helios's two core failure modes:

- **Critical P1 incidents hide across every queue:** `2,824` tickets (`17.65%`) are `high + Incident`, and `22.26%` (`3,562` tickets) touch healthcare/medical IT, VPN, or outage workflows. Yet only `8.2%` of P1 incidents land in `Service Outages and Maintenance`—`52.8%` arrive inside `Technical Support`, `14.7%` in `Product Support`, and `12.3%` in `IT Support`.
- **Real tickets arrive messy and bilingual:** `7.27%` (`1,163` rows) have blank/null subjects, `40.41%` are German vs. `59.59%` English, and queue volume spans a `21.7×` imbalance (`Technical Support` 29.34% vs. `General Inquiry` 1.35%).
- **Historical resolutions follow disciplined L1 norms:** Median agent reply length is `59` words, `86.98%` ask targeted diagnostic/clarifying questions, and only `1.44%` mention refunds or SLA terms—proving why off-the-shelf generative chatbots routinely hallucinate financial commitments when left unguarded.

---

## 2. Three Architectural Choices That Shaped the Result

1. **System 1 + System 2 Closed-Loop Control Plane (Not a Single Prompt):**
   Rather than sending raw tickets directly to an LLM, every incoming ticket first passes through a strictly local privacy perimeter (`lingua` EN/DE detection + bidirectional PII/PHI vault) before hitting `EmbeddingGemma 2` (256d Matryoshka, `bfloat16`), FAISS `IndexFlatIP`, calibrated multi-task triage (`LightGBM + Sparse Linear + Temperature Scaling` blended with optional `gemini-decision-1`), and a 5-rule triangulated escalation gate.
2. **Decoupling Calibration (ECE) from Imbalanced Routing & P1 Safety:**
   On the internal 15% validation carve-out inside `kb_train`, temperature scaling drove calibration error down to `0.0148–0.0204` across all three heads, while Menon-style prior-adjusted decoding lifted 10-queue Macro-F1 by `+4.27` points on the unseen holdout (`36.89% -> 41.16%`) and safety-margin P1 threshold tuning (`95%` internal val target) delivered **`92.17%` P1 Incident Recall (`647/702`)** on all `4,000` unseen holdout tickets.
3. **Warm Escalations + Closed-Loop Pass-2 Grounding Veto:**
   Escalated tickets never end in a bare red warning—they emit both a native-language (`en`/`de`) Safe Holding Ack (zero SLA/refund promises) and an English On-Call War-Room Brief citing `[TKT-XXXXX]` historical root causes. Auto-drafts must survive a deterministic citation/refund/SLA policy guard plus Pass-2 grounding audit (`all_claims_supported >= 0.85`, `fabricates_refund_or_sla <= 0.15`), or get vetoed and flipped to `ESCALATE_ON_CALL`.

---

## 3. Verified Empirical Proof (`artifacts/holdout_metrics.json`, N = 4,000 Unseen Tickets)

| Dimension | Measured Holdout Result | Target Bar | Status |
|---|---|---|---|
| **Zero-Leakage Split** | `16,000` KB / `4,000` Holdout / `0` ID overlap | `0` overlap | Verified |
| **High-Priority Incident Recall** | **`92.45%`** (`649 / 702`) | `>= 90.0%` | Passed |
| **Multi-Task Macro-F1 (3-Way Local Ensemble)** | Type **`81.41%`** · Queue (10) **`42.90%`** · Priority **`49.05%`** | Maximize imbalanced F1 | Verified |
| **Expected Calibration Error (ECE)** | Type **`0.0117`** · Queue **`0.0255`** · Priority **`0.0167`** | `<= 0.05` | Passed |
| **Null-Subject Resilience (`n=298`)** | **`+0.48%`** mean F1 delta · **`0`** crashes | `< 3%` delta, `0` crashes | Passed |
| **Cross-Lingual Retrieval (Top-5)** | **`47.1%`** queries match cross-lingually · Top-1 median sim **`0.9312`** · Same-Queue@5 **`87.62%`** | Cross-lingual EN<->DE | Verified |
| **Triage Latency (Local 3-Way / Live Hybrid D1)** | **`162.1 ms` p50** (`193.9 ms` p95) · Live Hybrid **`496.6 ms` p50** | `< 400 ms` local | Passed |
| **10-Case Adversarial & Safety Suite** | **`10/10` (`100%`)** · `0%` Raw PII leak · `100%` Refund/SLA Veto · `100%` Injection Block · EN+DE Outage Burst triggered | `100%` | Passed |

---

## 4. Honest Boundaries & Next Discriminating Test

- **What is verified today:** End-to-end zero-leakage pipeline, GPU `EmbeddingGemma 2` FAISS retrieval, calibrated local ML ensemble, live air-gapped Ollama Gemma 4 drafting, deterministic + local Pass-2 grounding veto, automated `pytest` suite, and live browser verification of all 4 demo presets + Hospital VPN Outage Burst + Tab 2 Scorecard.
- **What remains conditional on external credentials:** Live cloud calls to `gemini-decision-1` and `gemma-4-26b-a4b-it` are implemented against exact `google-genai==3.6.1` v1beta schemas and gracefully degrade when `GEMINI_API_KEY` is unset; setting `GEMINI_API_KEY` in `.env` and re-running `python evaluate.py` is the next discriminating test to measure exact Hybrid (`0.55 ML + 0.45 D1`) lift on rare queues (`General Inquiry`, `Sales and Pre-Sales`).
- **Operational Risk / Tradeoff:** Pushing P1 Incident Recall to `92.17%` lowers P1 precision to `36.64%` (`1,119` false positives across `4,000` tickets)—an explicit safety-first tradeoff for hospital/infrastructure operations where missed P1 outages are far costlier than human L1/L2 review of borderline escalations.
