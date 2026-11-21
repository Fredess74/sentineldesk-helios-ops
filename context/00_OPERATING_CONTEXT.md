# Operating Context — SentinelDesk for Helios Managed Ops

**Source Authority:**
- Official Challenge Brief: [SentinelDesk — Autonomous Incident Triage & Grounded Resolution Copilot](https://docs.google.com/document/d/1KeZWHjmfjx00pQIGDGHroOCecdvJzpkU/mobilebasic) (Track: Applied AI, NLP, RAG & Agentic Workflows; 6-hour build mission, live demo + Q&A).
- Target Company Archetype: **Helios Managed Ops** — Transatlantic (US + DACH: Germany, Austria, Switzerland) Multi-Client Managed IT & Enterprise BPO Shared-Services Provider.
- Official Dataset: [`Tobi-Bueck/customer-support-tickets`](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) (`CC-BY-NC-4.0`, clean 10-queue EN/DE dataset).

---

## 1. Stakeholders & Operational Roles

| Role | Persona at Helios Managed Ops | Primary Pain / Need | Adoption Risk / Friction |
|---|---|---|---|
| **Operator (Daily User)** | Bilingual L1 Service Desk Analyst (EN/DE hubs in Austin & Munich) | Drowning in unstructured mixed-language tickets; wastes time manually classifying queues, checking EN vs. DE silos, and verifying ungrounded AI drafts (rework tax). | Rejects black-box AI that requires reading 5 external tabs to trust a single sentence or sends risky promises automatically. |
| **Escalation Owner** | L2/L3 On-Call Infrastructure & Network Engineer | Receives cold L1 handoffs ("VPN broken, pls check") with zero context, missing suspected system, and delayed notification on P1 hospital outages. | Alert fatigue if routine tickets false-escalate; catastrophic SLA breach if P1 Incidents miss escalation. |
| **Decision Maker / Compliance** | VP of Managed Operations & DPO (GDPR / HIPAA / EU AI Act Art. 50) | Exposure of client PII/PHI (patient IDs, emails, IPs, IBANs) to cloud LLMs; legal exposure from fabricated SLA credits or refunds. | Blocks any cloud AI tool lacking auditable PII redaction, deterministic policy vetoes, and zero-hallucination guarantees. |
| **Beneficiary (End Client)** | Enterprise Clients (Hospital Networks, FinTech/Banking, B2B SaaS, Hardware Vendors) | Fast, accurate native-language resolution (`en` or `de`); immediate P1 response during telemedicine/EMR outages. | Loses trust when client-A policies leak into client-B replies or German tickets get English boilerplate. |
| **Evaluator (Hackathon Jury)** | Technical & Product Judges (5-min live demo + 3-min unseen-ticket Q&A) | Inspectable evidence: reproducible 20% holdout metrics (`Macro-F1`, `High-Priority Incident Recall`), zero-leakage proof, live console handling unseen edge cases. | Penalizes slideware, data leakage, missing null-subject handling, or slow/crashing demos. |

---

## 2. Current Baseline Workflow vs. Strongest Alternative

1. **Current Manual Baseline (Status Quo):**
   - Incoming EN/DE tickets land in a shared L1 inbox (many with missing/empty subjects).
   - Analysts manually read, tag `type` / `queue` / `priority`, search separate EN or DE ticket archives, and draft replies or paste bare descriptions into PagerDuty/Jira for L2/L3.
   - **Failure Modes:** P1 hospital outages buried behind routine requests; EN fixes invisible to DE agents; slow handling times.
2. **Strongest Naive Alternative (Off-the-Shelf LLM Chatbot):**
   - Prompts a frontier LLM directly with raw ticket text and top-k keyword hits.
   - **Failure Modes:** Hallucinates refunds/SLA credits/unsupported integrations; leaks raw PII/PHI to cloud endpoints; produces uncalibrated confidence scores ("always 95% confident"); dumps bare red warnings on escalations without engineering handoff briefs.
3. **SentinelDesk Control Plane (System 1 + System 2 Closed-Loop Architecture):**
   - **Local Privacy Perimeter first:** Automatic EN/DE language detection (`lingua`) + bidirectional local PII/PHI vault before any cloud call.
   - **System 1 Calibrated Gatekeeper:** High-capacity 500k word+char TF-IDF + `google/embeddinggemma-2` (256d) `Calibrated LinearSVC` + post-hoc Temperature Scaling + dedicated binary `critical` P1 detector; in **Cloud mode**, blended (`0.75 SVM + 0.25 D1`) with live `gemini-decision-1` single-pass calibrated scoring; in **Local Air-Gapped mode**, strictly **0 cloud calls** (`~70 ms` end-to-end).
   - **5-Rule 3-Tier Escalation Gate + Multilingual Burst Radar:** Explicit, auditable safety rules (`PAGE_ON_CALL` vs `HUMAN_REVIEW` vs `AUTO_DRAFT_READY`) with negation-aware EN/DE impact detection + real-time EN+DE outage clustering.
   - **System 2 Grounded Multilingual Drafter (`gemini-3.5-flash-lite` PII-Scrubbed Cloud OR Air-Gapped Local Precedent Synthesizer / Warm Ollama) + Pass-2 Grounding & Policy Veto:** Enforces strict `[TKT-XXXXX]` precedent grounding, `rapidfuzz` lexical grounding verification, native-language replies even from cross-lingual precedents, deterministic + calibrated refund/SLA veto, and Dual-Mode Warm Escalations (Native Holding Ack + On-Call War-Room Brief).

---

## 3. Working Assumptions & Scope Boundaries

- **Included in Delivery:** End-to-end working FastAPI backend, React/Vite Audit Glass SOC console (deployed to GitHub Pages + local runtime), zero-leakage duplicate-grouped 28.5k data pipeline (`splits/split_assignment.csv`), FAISS 256d cross-lingual index (`22,869 kb_train`), calibrated multi-task LinearSVC + dedicated binary `critical` head, live `gemini-decision-1` Pass-1/Pass-2 integration, strict Cloud vs Air-Gapped Local mode separation, reproducible `evaluate.py` holdout benchmark, automated pytest suite, and live browser verification.
- **Explicitly Excluded (Reverse Anti-Patterns):** Blind auto-send without human sign-off; egress of unredacted PII to cloud APIs; training or indexing on any holdout row or duplicate-text copy; fabricating refund amounts, SLA percentages, or L3 root fixes absent from retrieved precedents.
- **Data Split Standard:** Default build uses the full official 28,587-row clean 10-queue dataset (`aa_dataset-tickets-multi-lang-5-2-50-version.csv`) split via duplicate-grouped `StratifiedGroupKFold(n_splits=5)` (`22,869 kb_train` / `5,718 holdout_test`), with optional `--20k` CLI support.
