"""5-Rule Triangulated Escalation Gate (3-Tier Routing + Negation-Aware Impact Rules) + Outage Burst Radar."""

from __future__ import annotations

import re
import threading
import time
from collections import Counter, deque
from datetime import datetime, timezone
from typing import Any

import numpy as np

from backend.config import (
    BURST_MAX_BUFFER,
    BURST_MIN_TICKETS,
    BURST_SIMILARITY_THRESHOLD,
    BURST_WINDOW_SECONDS,
    MIN_CONFIDENCE_THRESHOLD,
    MIN_RETRIEVAL_SIMILARITY,
    PROMPT_INJECTION_THRESHOLD,
    UNSUPPORTED_COMMITMENT_THRESHOLD,
)

# Negation-aware bilingual EN/DE impact rules (adapted from teammate Egor's rules.py)
_NEGATION_RE = re.compile(
    r"(?i)\b(?:no|not|never|without|isn'?t|wasn'?t|aren'?t|don'?t|doesn'?t|kein\w*|nicht|nie|ohne)\b[\w\s,']{0,22}$"
)

_IMPACT_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "clinical",
        re.compile(
            r"(?i)\b(?:patient\w*|hospital\w*|clinic\w*|clinician\w*|doctor\w*|ärzt\w*|arzt|krankenhaus\w*|klinik\w*|"
            r"emr|ehr|telemedicine|telemedizin|medical records?|patientenakte\w*)\b"
        ),
    ),
    (
        "outage",
        re.compile(
            r"(?i)\b(?:outage|down for (?:all|every)\w*|(?:is|are|went|gone) down|not working for (?:all|any|every)\w*|"
            r"nobody can|no one can|all users|entire (?:company|team|office|site)|company-wide|site-wide|"
            r"vpn (?:tunnel|gateway)?\s*(?:failed|down|dropped|outage)|ausfall|komplett ausgefallen|niemand kann|alle (?:nutzer|mitarbeiter))\b"
        ),
    ),
    (
        "security",
        re.compile(
            r"(?i)\b(?:breach\w*|hack(?:ed|er)?|ransomware|malware|phishing|unauthori[sz]ed access|compromised|"
            r"data leak\w*|sicherheitsvorfall|datenverstoß|angriff|gehackt|datenleck|unbefugt\w* zugriff)\b"
        ),
    ),
    (
        "data_loss",
        re.compile(
            r"(?i)\b(?:data loss|lost (?:all )?(?:our |the )?data|deleted (?:all|our)|corrupt\w*|datenverlust|daten verloren)\b"
        ),
    ),
    (
        "urgency",
        re.compile(
            r"(?i)\b(?:urgent\w*|immediate\w*|asap|emergency|critical|p1|dringend\w*|sofort\w*|notfall|kritisch\w*)\b"
        ),
    ),
]

_SEVERE_IMPACT = {"clinical", "outage", "security", "data_loss"}


def detect_impact_signals(text: str) -> list[str]:
    """Detect non-negated EN/DE severe impact & urgency phrases (ignores 'no data loss', 'not urgent')."""
    raw = text or ""
    found: list[str] = []
    for name, pat in _IMPACT_RULES:
        for m in pat.finditer(raw):
            prefix = raw[max(0, m.start() - 32) : m.start()]
            if _NEGATION_RE.search(prefix):
                continue
            found.append(name)
            break
    return found


def severe_impact_triggers_page(impact_signals: list[str]) -> list[str]:
    """Page on severe impact only when paired with urgency, a second severe signal, or active security breach."""
    severe = [s for s in impact_signals if s in _SEVERE_IMPACT]
    if severe and ("urgency" in impact_signals or len(severe) >= 2 or "security" in severe):
        return severe
    return []


def compute_neighbor_majority(precedents: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute similarity-weighted majority labels across top-5 FAISS precedents."""
    if not precedents:
        return {"type": None, "queue": None, "priority": None, "type_share": 0.0, "queue_share": 0.0}

    def weighted_vote(field: str) -> tuple[str | None, float]:
        weights: dict[str, float] = {}
        total = 0.0
        for p in precedents[:5]:
            val = p.get(field)
            w = max(float(p.get("similarity", 0.0)), 0.01)
            if val:
                weights[val] = weights.get(val, 0.0) + w
                total += w
        if not weights:
            return None, 0.0
        winner = max(weights, key=weights.get)
        return winner, round(weights[winner] / max(total, 1e-6), 4)

    t_lbl, t_share = weighted_vote("type")
    q_lbl, q_share = weighted_vote("queue")
    p_lbl, _ = weighted_vote("priority")
    return {
        "type": t_lbl,
        "queue": q_lbl,
        "priority": p_lbl,
        "type_share": t_share,
        "queue_share": q_share,
    }


def compute_triangulated_agreement(
    local_ml_preds: dict[str, str],
    decision1_preds: dict[str, str] | None,
    neighbor_majority: dict[str, Any],
    top1_similarity: float,
) -> dict[str, Any]:
    """Check agreement across Local ML, gemini-decision-1 (in Cloud mode), and top-5 FAISS neighbors."""
    conflicts: list[str] = []

    if decision1_preds is not None:
        for field in ("type", "queue"):
            votes = [local_ml_preds.get(field), decision1_preds.get(field)]
            if top1_similarity >= 0.65 and neighbor_majority.get(field):
                votes.append(neighbor_majority[field])
            unique_votes = {v for v in votes if v}
            if len(unique_votes) == len(votes) and len(votes) >= 2:
                conflicts.append(
                    f"{field}: ML={local_ml_preds.get(field)} vs D1={decision1_preds.get(field)} vs KB={neighbor_majority.get(field)}"
                )
            elif local_ml_preds.get(field) != decision1_preds.get(field):
                conflicts.append(
                    f"{field}: ML={local_ml_preds.get(field)} disagrees with D1={decision1_preds.get(field)}"
                )
    else:
        for field, share_key in (("type", "type_share"), ("queue", "queue_share")):
            nb_lbl = neighbor_majority.get(field)
            nb_share = float(neighbor_majority.get(share_key, 0.0))
            if top1_similarity >= 0.72 and nb_lbl and nb_share >= 0.65 and nb_lbl != local_ml_preds.get(field):
                conflicts.append(
                    f"{field}: ML={local_ml_preds.get(field)} conflicts with KB consensus={nb_lbl} ({nb_share:.0%})"
                )

    if not conflicts:
        all_match = True
        for field in ("type", "queue"):
            vals = [local_ml_preds.get(field)]
            if decision1_preds:
                vals.append(decision1_preds.get(field))
            if neighbor_majority.get(field):
                vals.append(neighbor_majority.get(field))
            if len({v for v in vals if v}) > 1:
                all_match = False
        status = "UNANIMOUS" if all_match else "MAJORITY"
    else:
        status = "CONFLICT"

    return {
        "status": status,
        "local_ml": local_ml_preds,
        "decision_1": decision1_preds,
        "faiss_neighbors_majority": {
            "type": neighbor_majority.get("type"),
            "queue": neighbor_majority.get("queue"),
            "priority": neighbor_majority.get("priority"),
        },
        "conflict_details": conflicts,
    }


def evaluate_escalation_gate(
    type_pred: str,
    type_conf: float,
    queue_pred: str,
    queue_conf: float,
    priority_pred: str,
    priority_conf: float,
    p1_joint_risk: float,
    p1_joint_threshold: float,
    top1_similarity: float,
    model_agreement: dict[str, Any],
    security_signals: dict[str, float],
    p_high: float | None = None,
    critical_score: float | None = None,
    critical_threshold: float = 0.25,
    impact_signals: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate 5 explicit rules and assign a clean 3-tier route:
      - PAGE_ON_CALL     (urgent P1 incident / severe clinical-outage-security impact / prompt injection)
      - HUMAN_REVIEW     (low confidence / OOD precedent gap / disagreement / unsupported refund-SLA demand)
      - AUTO_DRAFT_READY (safe, confident, precedent-grounded L1 reply)
    """
    reasons: list[dict[str, str]] = []
    page_triggered = False
    impacts = impact_signals or []

    # Rule 1: Critical P1 Incident (dedicated binary critical detector OR confident high+Incident OR severe impact rules)
    crit_val = float(critical_score) if critical_score is not None else float(p1_joint_risk)
    label_is_p1 = priority_pred == "high" and type_pred == "Incident"
    # Type head achieves 98.5-99.3% precision on Request/Change; only apply low P1 floor to Incidents/Problems/uncertain types
    eff_crit_thr = 0.50 if (type_pred in ("Request", "Change") and type_conf >= MIN_CONFIDENCE_THRESHOLD) else float(critical_threshold)
    detector_is_p1 = critical_score is not None and crit_val >= eff_crit_thr
    severe_impacts = severe_impact_triggers_page(impacts)

    if label_is_p1 or detector_is_p1 or severe_impacts:
        page_triggered = True
        triggers = []
        if label_is_p1:
            triggers.append(f"classified high Incident ({type_conf:.0%}/{priority_conf:.0%})")
        if detector_is_p1:
            triggers.append(f"critical detector score {crit_val:.0%} ≥ {critical_threshold:.0%}")
        if severe_impacts:
            triggers.append("negation-checked impact: " + ", ".join(severe_impacts))
        reasons.append(
            {
                "rule_id": "RULE_1_P1_INCIDENT",
                "label": "Critical P1 Incident",
                "detail": f"Immediate on-call escalation in {queue_pred} ({'; '.join(triggers)}).",
            }
        )

    # Rule 2: Low Calibrated Confidence
    effective_prio_conf = priority_conf
    if type_pred != "Incident" and priority_pred != "high" and p_high is not None:
        effective_prio_conf = max(priority_conf, 1.0 - float(p_high))

    low_heads = []
    if type_conf < MIN_CONFIDENCE_THRESHOLD:
        low_heads.append(f"type={type_conf:.0%}")
    if queue_conf < MIN_CONFIDENCE_THRESHOLD:
        low_heads.append(f"queue={queue_conf:.0%}")
    if effective_prio_conf < MIN_CONFIDENCE_THRESHOLD:
        low_heads.append(f"priority={effective_prio_conf:.0%}")
    if low_heads:
        reasons.append(
            {
                "rule_id": "RULE_2_LOW_CONFIDENCE",
                "label": "Low Calibrated Confidence",
                "detail": f"Calibrated confidence below {MIN_CONFIDENCE_THRESHOLD:.0%} ({', '.join(low_heads)}).",
            }
        )

    # Rule 3: Precedent Gap (OOD)
    if top1_similarity < MIN_RETRIEVAL_SIMILARITY:
        reasons.append(
            {
                "rule_id": "RULE_3_PRECEDENT_GAP",
                "label": "Precedent Gap (OOD)",
                "detail": (
                    f"Top-1 historical KB similarity ({top1_similarity:.0%}) is below "
                    f"the {MIN_RETRIEVAL_SIMILARITY:.0%} grounding floor."
                ),
            }
        )

    # Rule 4: Triangulated Disagreement
    if model_agreement.get("status") == "CONFLICT":
        reasons.append(
            {
                "rule_id": "RULE_4_TRIANGULATED_DISAGREEMENT",
                "label": "Triangulated Disagreement",
                "detail": "; ".join(model_agreement.get("conflict_details", [])) or "Voters disagreed on routing.",
            }
        )

    # Rule 5: Security / Policy Violation
    inj = float(security_signals.get("prompt_injection", 0.0))
    commit = float(security_signals.get("demands_unsupported_commitment", 0.0))
    if inj >= PROMPT_INJECTION_THRESHOLD:
        page_triggered = True
        reasons.append(
            {
                "rule_id": "RULE_5_PROMPT_INJECTION",
                "label": "Security Gate: Prompt Injection",
                "detail": f"Adversarial instruction override / jailbreak attempt detected (score {inj:.0%}).",
            }
        )
    if commit >= UNSUPPORTED_COMMITMENT_THRESHOLD:
        reasons.append(
            {
                "rule_id": "RULE_5_UNSUPPORTED_COMMITMENT",
                "label": "Policy Gate: Unsupported SLA / Refund Demand",
                "detail": (
                    f"Ticket demands unverified refund, SLA credit, or binding commitment "
                    f"(score {commit:.0%}); routed to human review."
                ),
            }
        )

    if page_triggered:
        route = "PAGE_ON_CALL"
        status = "ESCALATE_ON_CALL"
    elif reasons:
        route = "HUMAN_REVIEW"
        status = "ESCALATE_ON_CALL"
    else:
        route = "AUTO_DRAFT_READY"
        status = "AUTO_DRAFT_READY"

    return {
        "status": status,
        "route": route,
        "initial_status": status,
        "flipped_by_pass2_veto": False,
        "impact_signals": impacts,
        "reasons": reasons,
    }


class OutageBurstRadar:
    """Rolling cross-lingual semantic burst detector (Bonus #2)."""

    def __init__(
        self,
        window_seconds: int = BURST_WINDOW_SECONDS,
        similarity_threshold: float = BURST_SIMILARITY_THRESHOLD,
        min_tickets: int = BURST_MIN_TICKETS,
        max_buffer: int = BURST_MAX_BUFFER,
    ) -> None:
        self.window_seconds = window_seconds
        self.similarity_threshold = similarity_threshold
        self.min_tickets = min_tickets
        self._buffer: deque[dict[str, Any]] = deque(maxlen=max_buffer)
        self._lock = threading.Lock()
        self._seq = 0

    def reset(self) -> dict[str, Any]:
        with self._lock:
            self._buffer.clear()
        return {"active_outage": False, "incidents": [], "buffer_size": 0}

    def record(
        self,
        request_id: str,
        subject: str,
        body: str,
        language: str,
        queue: str,
        priority: str,
        ticket_type: str,
        embedding: np.ndarray,
    ) -> dict[str, Any]:
        now = time.time()
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm

        with self._lock:
            self._prune(now)
            self._seq += 1
            self._buffer.append(
                {
                    "request_id": request_id,
                    "subject": (subject or "").strip() or "(No Subject)",
                    "body_preview": ((body or "").strip()[:140] + "...") if len((body or "").strip()) > 140 else (body or "").strip(),
                    "language": language,
                    "queue": queue,
                    "priority": priority,
                    "type": ticket_type,
                    "timestamp": now,
                    "iso_time": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
                    "vec": vec,
                }
            )
            return self._compute_clusters_locked()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            self._prune(time.time())
            return self._compute_clusters_locked()

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._buffer and self._buffer[0]["timestamp"] < cutoff:
            self._buffer.popleft()

    def _compute_clusters_locked(self) -> dict[str, Any]:
        items = list(self._buffer)
        n = len(items)
        if n < self.min_tickets:
            return {"active_outage": False, "incidents": [], "buffer_size": n}

        vecs = np.stack([it["vec"] for it in items], axis=0)
        sim_mat = vecs @ vecs.T

        visited = [False] * n
        incidents: list[dict[str, Any]] = []
        date_tag = datetime.now(tz=timezone.utc).strftime("%Y%m%d")

        for i in range(n):
            if visited[i]:
                continue
            comp: list[int] = []
            stack = [i]
            visited[i] = True
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in range(n):
                    if not visited[v] and sim_mat[u, v] >= self.similarity_threshold:
                        visited[v] = True
                        stack.append(v)

            if len(comp) >= self.min_tickets:
                comp.sort()
                sub_items = [items[idx] for idx in comp]
                langs = sorted({it["language"] for it in sub_items})
                queues = Counter(it["queue"] for it in sub_items)
                top_queue = queues.most_common(1)[0][0]
                pair_sims = [
                    float(sim_mat[a, b])
                    for ia, a in enumerate(comp)
                    for b in comp[ia + 1 :]
                ]
                avg_sim = sum(pair_sims) / max(len(pair_sims), 1)
                inc_id = f"MIR-{date_tag}-{len(incidents) + 1:02d}"

                incidents.append(
                    {
                        "master_incident_id": inc_id,
                        "ticket_count": len(sub_items),
                        "affected_languages": langs,
                        "is_cross_lingual": len(langs) > 1,
                        "primary_queue": top_queue,
                        "avg_cluster_similarity": round(avg_sim, 4),
                        "summary": sub_items[0]["subject"] if sub_items[0]["subject"] != "(No Subject)" else sub_items[0]["body_preview"][:90],
                        "tickets": [
                            {
                                "request_id": it["request_id"],
                                "subject": it["subject"],
                                "language": it["language"],
                                "queue": it["queue"],
                                "priority": it["priority"],
                                "timestamp": it["iso_time"],
                            }
                            for it in sub_items
                        ],
                    }
                )

        incidents.sort(key=lambda x: x["ticket_count"], reverse=True)
        return {
            "active_outage": len(incidents) > 0,
            "incidents": incidents,
            "buffer_size": n,
        }
