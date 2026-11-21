# Brief-to-Implementation Traceability Map

Maps every requirement in the [Official Hackathon Brief](https://docs.google.com/document/d/1KeZWHjmfjx00pQIGDGHroOCecdvJzpkU/mobilebasic) and Master Mission prompt to exact code modules, verifiable artifacts, and automated checks.

| Requirement | Category | Code Module(s) | Output Artifact / UI Surface | Verification Check |
|---|---|---|---|---|
| **Strict 80/20 split (`random_state=42`) & clean 10-queue dataset** | Core / Compliance | `backend/data_prep.py` | `data/kb_train.parquet`, `data/holdout_test.parquet`, `artifacts/kb_audit_summary.json` | Zero ID intersection assertion; `pytest tests/test_pipeline.py` |
| **Phase 1 Research & Data Audit** | Phase 1 | `backend/data_prep.py` | `docs/01_RESEARCH_AND_DATA_AUDIT.md` | All numbers sourced strictly from `kb_train.parquet` |
| **Phase 2 Kano Model & Full PRD** | Phase 2 | — | `docs/02_KANO_AND_PRD.md` | Full Kano matrix + frozen `/api/triage` JSON schema |
| **Multi-Task Triage Engine (`type`, `queue`, `priority` + calibrated confidence + auto EN/DE)** | Core MVP | `backend/indexing.py`, `backend/features.py`, `backend/train_ml.py`, `backend/decision_engine.py`, `backend/privacy.py`, `backend/triage.py` | `artifacts/ml_models.joblib`, UI Audit Glass Panel | `evaluate.py`: Macro-F1, High-Priority Incident Recall >= 90%, ECE, null-subject slice |
| **Confidence & Risk Escalation Gate (5 explicit rules + Triangulated Disagreement)** | Core MVP + Attractive | `backend/escalation.py` | Escalation Status Badge + Triggered Rule Badges | Adversarial & unit tests for P1, low conf, OOD sim, disagreement, injection, policy veto |
| **Evidence-Backed Multilingual RAG (`kb_train` only, top-5, `[TKT-XXXXX]` citations, no fabrication)** | Core MVP | `backend/indexing.py`, `backend/drafter.py` | `artifacts/kb_faiss.index`, Precedent Cards + clickable `[TKT-XXXXX]` chips | 100% citation validity; 0% fabricated refund/SLA on adversarial suite |
| **Pass-2 Grounding & Policy Veto + Dual-Mode Warm Escalation** | Attractive | `backend/decision_engine.py`, `backend/drafter.py` | Pass-2 Grounding Audit Card + Safe Holding Ack / On-Call War-Room Brief | Adversarial refund preset triggers deterministic + audit veto -> flips to `ESCALATE_ON_CALL` |
| **Cross-Lingual EN <-> DE Precedent Matching** | Bonus #1 | `backend/indexing.py` (`EmbeddingGemma 2` 256d), `backend/drafter.py` | Precedent Card "Cross-Lingual Match" badge + native-language reply | Demo Preset #1 (German null-subject) + holdout cross-lingual match rate metric |
| **Live Multilingual Outage Burst Radar** | Bonus #2 | `backend/escalation.py`, `backend/main.py` | Top Major Incident Outage Banner + Simulate / Reset buttons | Simulate 3 EN + 1 DE hospital VPN tickets -> banner activates with Master Incident ID |
| **Bidirectional PII/PHI Vault + Air-Gapped Local Gemma 4 Toggle** | Bonus #3 | `backend/privacy.py`, `backend/drafter.py` | "What Left Your Network" PII Diff Drawer + Drafter Mode Toggle | Adversarial PII preset masks email/IP/phone/IBAN/account before cloud payload |
| **Interactive Live Triage Console + Embedded Holdout Scorecard** | Core MVP | `backend/main.py`, `frontend/src/**` | Live Web App (Tab 1 Live Console + Tab 2 20% Holdout Scorecard) | End-to-end browser verification of all 4 presets, burst sim, and Tab 2 + screenshots |
| **Reproducible 20% Holdout Evaluation** | Core MVP | `evaluate.py` | `artifacts/holdout_metrics.json`, Tab 2 Scorecard, `README.md` | Executed on full 20% holdout split + 10-case adversarial suite |
