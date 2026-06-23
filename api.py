"""
EventFlow AI — FastAPI backend for the React command center.
Run: uvicorn api:app --reload --port 8000
"""
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src import api_services as svc

app = FastAPI(
    title="EventFlow AI",
    description="BTP Command Center API — GridGuard inference & operational intelligence",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EventInput(BaseModel):
    event_cause: str = "vehicle_breakdown"
    corridor: str = "Hebbal - Silk Board"
    zone: str = "Central"
    hour: int = Field(18, ge=0, le=23)
    requires_road_closure: bool = False
    lat: float = 13.035
    lon: float = 77.597


class TriageInput(BaseModel):
    zone: str = "Central"
    capacity: int = Field(14, ge=1, le=100)
    active_events: Optional[List[Dict[str, Any]]] = None
    new_event: Optional[EventInput] = None


def _read_mappls_static_key() -> str:
    cfg_path = Path(__file__).parent / "config.yaml"
    if not cfg_path.exists():
        return ""
    for line in cfg_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("MAPPLS_API_KEY:"):
            return stripped.split(":", 1)[1].strip().strip('"').strip("'")
    return ""


@app.get("/health")
def health():
    return {"status": "ok", "service": "EventFlow AI", "model": "GridGuard V9"}


@app.get("/mappls/map-config")
def mappls_map_config():
    """Map SDK credentials for the React map (OAuth token + static/REST keys)."""
    try:
        from src.mappls_feature_engine import (
            MAPPLS_CLIENT_ID,
            MAPPLS_CLIENT_SECRET,
            MAPPLS_REST_KEY,
            MapplsTokenManager,
        )

        token_mgr = MapplsTokenManager(MAPPLS_CLIENT_ID, MAPPLS_CLIENT_SECRET)
        access_token = token_mgr.get_token()
        static_key = _read_mappls_static_key()

        return {
            "static_key": static_key,
            "rest_key": MAPPLS_REST_KEY,
            "access_token": access_token,
            "center": {"lat": 12.9716, "lng": 77.5946},
            "zoom": 11,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Mappls config error: {exc}") from exc


@app.get("/mappls/traffic-snapshot")
def mappls_traffic_snapshot(lat: Optional[float] = None, lon: Optional[float] = None):
    """Live Bengaluru corridor delays via Mappls Route ADV (cached ~2 min)."""
    try:
        from src.mappls_traffic import get_traffic_snapshot

        return get_traffic_snapshot(lat=lat, lon=lon)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Traffic snapshot error: {exc}") from exc


@app.get("/meta/options")
def meta_options():
    return {
        "event_causes": svc.EVENT_CAUSES,
        "zones": svc.ZONES,
        "planned_types": svc.PLANNED_TYPES,
    }


@app.post("/predict/event")
def predict_event(event: EventInput):
    try:
        result = svc.predict_event(event.model_dump())
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/triage/optimize")
def triage_optimize(body: TriageInput):
    try:
        if body.active_events:
            events = body.active_events
        elif body.new_event:
            pred = svc.predict_event(body.new_event.model_dump())
            ev = body.new_event.model_dump()
            events = svc.default_active_events(
                ev,
                pred["stis"]["stis"],
                pred["stis"]["min_officers"],
            )
        else:
            raise HTTPException(status_code=400, detail="Provide active_events or new_event")
        return svc.run_triage(events, body.zone, body.capacity)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/briefing/risk")
def briefing_risk(hour: int = 8):
    if not 0 <= hour <= 23:
        raise HTTPException(status_code=400, detail="hour must be 0-23")
    return svc.get_risk_briefing(hour)


@app.get("/briefing/trends")
def briefing_trends():
    return svc.get_trends()


@app.get("/audit/stations")
def audit_stations():
    return svc.get_station_efficiency()


@app.get("/audit/metrics")
def audit_metrics():
    return {"metrics": svc.get_validation_metrics()}


@app.get("/audit/cascades")
def audit_cascades():
    return svc.get_cascades()


@app.get("/planned/summary")
def planned_summary():
    return svc.get_planned_summary()


@app.get("/planned/dossier/{event_id}")
def planned_dossier(event_id: int):
    dossier = svc.get_planned_dossier(event_id)
    if not dossier.get("available"):
        raise HTTPException(status_code=404, detail=dossier.get("message", "Not found"))
    return dossier


@app.get("/planned/analytics")
def planned_analytics():
    return svc.get_planned_analytics()


class GraphQueryInput(BaseModel):
    lat: float = 13.035
    lon: float = 77.597
    corridor: str = "Non-corridor"
    k_hops: int = Field(2, ge=1, le=5)


@app.get("/graph/status")
def graph_status():
    return svc.get_graph_status()


@app.post("/graph/features")
def graph_features(body: GraphQueryInput):
    return svc.get_graph_features(body.model_dump())


@app.post("/graph/affected")
def graph_affected(body: GraphQueryInput):
    """Phase 3 — affected subgraph for an event location."""
    return svc.get_affected_subgraph(body.model_dump())


class V12EventInput(BaseModel):
    """V12-style prediction bundle (or V9 + manual fields for demo)."""
    event_id: str = "event_0"
    event_cause: str = "accident"
    corridor: str = "Bellary Road 1"
    zone: str = "Central"
    police_station: str = "Hebbal"
    hour: int = Field(18, ge=0, le=23)
    lat: float = 13.04
    lon: float = 77.59
    requires_road_closure: bool = False
    predicted_hours: Optional[float] = None
    p35_hours: Optional[float] = None
    p50_hours: Optional[float] = None
    p65_hours: Optional[float] = None
    impact_score: Optional[float] = None
    regime: Optional[str] = None
    active_overlap: int = Field(0, ge=0)
    graph_features: Optional[Dict[str, Any]] = None
    affected_subgraph: Optional[Dict[str, Any]] = None


class PrescriptiveBatchInput(BaseModel):
    events: List[V12EventInput]
    zone_capacity: Optional[Dict[str, int]] = None


class PrescriptiveRequest(BaseModel):
    event: V12EventInput
    zone_capacity: Optional[Dict[str, int]] = None
    concurrent_events: Optional[List[V12EventInput]] = None


class WhatIfInput(BaseModel):
    event: V12EventInput
    recommendations: Dict[str, Any]


@app.post("/prescriptive/recommend")
def prescriptive_recommend(body: PrescriptiveRequest):
    try:
        return svc.get_prescriptive_recommendation(
            body.event.model_dump(),
            body.zone_capacity,
            [e.model_dump() for e in body.concurrent_events] if body.concurrent_events else None,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/prescriptive/batch")
def prescriptive_batch(body: PrescriptiveBatchInput):
    try:
        return svc.get_prescriptive_batch(
            [e.model_dump() for e in body.events],
            body.zone_capacity,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/prescriptive/whatif")
def prescriptive_whatif(body: WhatIfInput):
    try:
        return svc.run_prescriptive_whatif(body.event.model_dump(), body.recommendations)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc