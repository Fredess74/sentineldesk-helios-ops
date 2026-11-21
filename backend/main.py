"""FastAPI server for SentinelDesk for Helios Managed Ops.

Serves:
  - /api/health
  - /api/triage
  - /api/burst, /api/burst/simulate, /api/burst/reset
  - /api/metrics (reads artifacts/holdout_metrics.json live)
  - Built React/Vite frontend static assets from frontend/dist/
"""

from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.config import FRONTEND_DIST_DIR, HOLDOUT_METRICS_PATH
from backend.triage import SentinelDeskEngine

_ENGINE: SentinelDeskEngine | None = None


def get_engine() -> SentinelDeskEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = SentinelDeskEngine()
    return _ENGINE


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_engine()
    yield


app = FastAPI(
    title="SentinelDesk for Helios Managed Ops",
    version="1.0.0",
    description="Multilingual Incident Triage, Triangulated Escalation & Grounded Resolution Copilot",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TriageRequest(BaseModel):
    subject: str | None = Field(default=None, description="Optional ticket subject (null/blank supported)")
    body: str = Field(..., min_length=1, description="Incoming unstructured ticket body")
    drafter_mode: Literal["cloud", "local"] = Field(
        default="cloud",
        description="cloud = PII-scrubbed Gemini/Gemma API; local = air-gapped Ollama gemma4:e4b",
    )
    record_in_burst_radar: bool = Field(default=True)


# Realistic transatlantic (3 EN + 1 DE) hospital VPN outage burst simulation tickets
HOSPITAL_VPN_BURST_TICKETS = [
    {
        "subject": "CRITICAL: Hospital VPN Gateway Down — Telemedicine & EMR Cut Off",
        "body": (
            "Emergency alert from St. Jude Austin Medical Center: Our primary site-to-site VPN tunnel dropped "
            "at 14:02 UTC. Clinicians cannot reach the EMR database or live telemedicine sessions. "
            "Gateway 10.48.12.4 returns TLS handshake timeout across all hospital wards."
        ),
    },
    {
        "subject": "Urgent VPN Outage Blocking Telemedicine and Patient EMR Access",
        "body": (
            "Second clinic site reporting total VPN gateway failure. All doctors attempting to load EMR patient "
            "records and telemedicine portals get connection refused. Restarting local routers did not restore "
            "the hospital VPN tunnel. Immediate escalation required."
        ),
    },
    {
        "subject": "",
        "body": (
            "Our regional hospital network lost VPN connectivity to the central EMR and telemedicine cluster "
            "five minutes ago. Emergency room intake cannot sync patient charts over the VPN gateway. Please escalate P1."
        ),
    },
    {
        "subject": "DRINGEND: Krankenhaus-VPN-Ausfall blockiert Telemedizin und EMR-Zugriff",
        "body": (
            "Kritische Störung im Klinikum München-Nord: Der VPN-Tunnel zum zentralen EMR-System und zur "
            "Telemedizin-Plattform ist vollständig ausgefallen. Ärzte können keine Patientenakten abrufen "
            "und bekommen sofortige Verbindungsabbrüche am VPN-Gateway."
        ),
    },
]


@app.get("/api/health")
def api_health() -> dict[str, Any]:
    return get_engine().health_info()


@app.post("/api/triage")
def api_triage(req: TriageRequest) -> dict[str, Any]:
    engine = get_engine()
    return engine.triage_ticket(
        subject=req.subject,
        body=req.body,
        drafter_mode=req.drafter_mode,
        record_in_burst_radar=req.record_in_burst_radar,
    )


@app.get("/api/burst")
def api_burst_status() -> dict[str, Any]:
    return get_engine().burst_radar.snapshot()


@app.post("/api/burst/simulate")
def api_burst_simulate() -> dict[str, Any]:
    engine = get_engine()
    results = []
    for item in HOSPITAL_VPN_BURST_TICKETS:
        res = engine.triage_ticket(
            subject=item["subject"],
            body=item["body"],
            drafter_mode="local",
            record_in_burst_radar=True,
            skip_draft=True,
            use_decision_api=False,
        )
        results.append(
            {
                "request_id": res["request_id"],
                "language": res["detected_language"],
                "subject": item["subject"] or "(No Subject)",
                "type": res["triage"]["type"]["label"],
                "queue": res["triage"]["queue"]["label"],
                "priority": res["triage"]["priority"]["label"],
                "escalation_status": res["escalation"]["status"],
            }
        )
    return {
        "simulated_count": len(results),
        "simulated_tickets": results,
        "burst_radar": engine.burst_radar.snapshot(),
    }


@app.post("/api/burst/reset")
def api_burst_reset() -> dict[str, Any]:
    return get_engine().burst_radar.reset()


@app.get("/api/metrics")
def api_metrics() -> dict[str, Any]:
    if not HOLDOUT_METRICS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Holdout evaluation metrics not generated yet. Run `python evaluate.py`.",
        )
    return json.loads(HOLDOUT_METRICS_PATH.read_text(encoding="utf-8"))


# Serve built React frontend if present
if FRONTEND_DIST_DIR.exists():
    assets_dir = FRONTEND_DIST_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str) -> FileResponse:
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API route not found")
        candidate = FRONTEND_DIST_DIR / full_path
        if full_path and candidate.is_file():
            return FileResponse(str(candidate))
        return FileResponse(str(FRONTEND_DIST_DIR / "index.html"))
