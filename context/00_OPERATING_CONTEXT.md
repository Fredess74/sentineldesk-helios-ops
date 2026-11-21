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
   - **System 1 Calibrated Gatekeeper:** Hybrid ensemble combining local calibrated ML (`EmbeddingGemma 2` 256d + TF-IDF SVD + structural signals) with `gemini-decision-1` single-pass calibrated scoring (gracefully degrading when offline/keyless).
   - **5-Rule Triangulated Escalation Gate + Multilingual Burst Radar:** Explicit, auditable safety rules + real-time EN+DE outage clustering.
   - **System 2 Grounded Multilingual Drafter (Gemma 4 — Local Air-Gapped or Scrubbed Cloud) + Pass-2 Grounding & Policy Veto:** Enforces strict `[TKT-XXXXX]` precedent grounding, native-language replies even from cross-lingual precedents, deterministic + calibrated refund/SLA veto, and Dual-Mode Warm Escalations (Native Holding Ack + On-Call War-Room Brief).

---

## 3. Working Assumptions & Scope Boundaries

- **Included in Delivery:** End-to-end working FastAPI backend, React/Vite Audit Glass SOC console, zero-leakage data pipeline, FAISS 256d cross-lingual index, calibrated multi-task ML heads, `gemini-decision-1` Pass-1/Pass-2 integration with transparent fallback, dual-mode Gemma 4 drafter (local Ollama `gemma4:e4b` + cloud Gemini/Gemma API), reproducible `evaluate.py` holdout benchmark, automated pytest suite, and live browser verification.
- **Explicitly Excluded (Reverse Anti-Patterns):** Blind auto-send without human sign-off; egress of unredacted PII to cloud APIs; training or indexing on any holdout row; fabricating refund amounts, SLA percentages, or L3 root fixes absent from retrieved precedents.
- **Reversible Operational Assumption:** Default build uses the clean 20k subset (`dataset-tickets-multi-lang-4-20k.csv`) for rapid iteration with full `--full` CLI support for the 50k clean file (`aa_dataset-tickets-multi-lang-5-2-50-version.csv`).
