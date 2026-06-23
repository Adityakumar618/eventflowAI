"""
EventFlow API service layer — operational intelligence for the React command center.
Does not modify inference or ML training code.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

CENSORING_RATES = {
    "vehicle_breakdown": 0.0,
    "accident": 0.0,
    "congestion": 0.0,
    "water_logging": 42.6,
    "pot_holes": 66.7,
    "tree_fall": 35.9,
    "construction": 15.2,
    "others": 30.6,
    "road_conditions": 43.5,
}

MIN_OFFICERS = {
    "water_logging": 6,
    "tree_fall": 4,
    "accident": 5,
    "vehicle_breakdown": 2,
    "construction": 3,
    "congestion": 2,
    "pot_holes": 1,
}

STIS_BASE = {
    "water_logging": 7.8,
    "tree_fall": 6.5,
    "accident": 7.2,
    "vehicle_breakdown": 3.8,
    "construction": 5.5,
    "congestion": 6.0,
    "pot_holes": 2.5,
    "road_conditions": 4.0,
    "others": 4.5,
    "procession": 7.5,
    "public_event": 6.8,
    "vip_movement": 8.5,
}

PLANNED_TYPES = ["procession", "public_event", "construction", "vip_movement"]

OFFICER_MAP_PLANNED = {
    "procession": 8,
    "public_event": 6,
    "construction": 3,
    "vip_movement": 12,
}

EVENT_CAUSES = [
    "water_logging",
    "tree_fall",
    "accident",
    "vehicle_breakdown",
    "construction",
    "congestion",
    "pot_holes",
    "road_conditions",
    "others",
]

ZONES = ["North", "South", "East", "West", "Central"]


def severity_label(score: float) -> str:
    if score >= 7:
        return "CRITICAL"
    if score >= 4:
        return "HIGH"
    if score >= 2:
        return "MEDIUM"
    return "LOW"


def severity_color(score: float) -> str:
    if score >= 7:
        return "#ef4444"
    if score >= 4:
        return "#f97316"
    if score >= 2:
        return "#fbbf24"
    return "#22c55e"


def _find_quantile(timeline: np.ndarray, survival: np.ndarray, thresh: float) -> Optional[float]:
    idx = np.where(survival <= thresh)[0]
    return round(float(timeline[idx[0]]), 1) if len(idx) > 0 else None


def compute_stis(
    event_cause: str,
    hour: int,
    requires_closure: bool,
    zone: str = "Central",
) -> Dict[str, Any]:
    rush_bonus = 1.2 if (8 <= hour <= 11) or (17 <= hour <= 20) else 1.0
    closure_bonus = 1.1 if requires_closure else 1.0
    stis = min(10.0, round(STIS_BASE.get(event_cause, 5.0) * rush_bonus * closure_bonus * 0.95, 1))
    min_officers = MIN_OFFICERS.get(event_cause, 3)
    if requires_closure:
        min_officers += 2
    return {
        "stis": stis,
        "label": severity_label(stis),
        "color": severity_color(stis),
        "min_officers": min_officers,
        "capacity_reduction": "High" if requires_closure else "Medium",
        "junction_centrality": "High" if zone in ("North", "Central") else "Medium",
        "detour_minutes": round(stis * 2.3, 0),
        "deployment_note": (
            f"{min_officers} officers · "
            f"{'Barricades required' if requires_closure else 'Standard setup'}"
        ),
    }


def compute_economic_impact(deploy_hrs: float, hour: int, centrality: float = 0.4) -> Dict[str, Any]:
    volume_mult = 2.0 if (17 <= hour <= 20) else (1.8 if (8 <= hour <= 11) else 0.9)
    cost_lakh = round(134 * centrality * deploy_hrs * volume_mult * 0.3, 1)
    affected = int(8_000_000 * centrality * volume_mult / 2)
    if cost_lakh < 100:
        cost_display = f"₹{cost_lakh:.1f} Lakh"
    else:
        cost_display = f"₹{cost_lakh / 100:.2f} Crore"
    return {
        "cost_lakh": cost_lakh,
        "cost_display": cost_display,
        "affected_commuters": affected,
        "volume_multiplier": volume_mult,
        "peak_label": "Peak (2×)" if volume_mult >= 1.8 else "Off-peak (0.9×)",
        "duration_hours": deploy_hrs,
    }


@lru_cache(maxsize=1)
def _get_inference_engine():
    from src.inference import get_inference_engine

    return get_inference_engine()


def load_km_curves() -> Dict[str, Any]:
    path = BASE_DIR / "models" / "km_curves.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def survival_analysis(event_cause: str) -> Dict[str, Any]:
    km_data = load_km_curves()
    censor_rate = CENSORING_RATES.get(event_cause, 0.0)
    if event_cause not in km_data:
        return {"available": False, "event_cause": event_cause, "censor_rate": censor_rate}

    timeline = np.array(km_data[event_cause]["timeline"])
    survival = np.array(km_data[event_cause]["survival_function"])
    p25 = _find_quantile(timeline, survival, 0.75)
    p50 = _find_quantile(timeline, survival, 0.50)
    p10 = _find_quantile(timeline, survival, 0.10)

    return {
        "available": True,
        "event_cause": event_cause,
        "timeline": timeline.tolist(),
        "survival": survival.tolist(),
        "p25_hours": p25,
        "p50_hours": p50,
        "p10_hours": p10,
        "censor_rate": censor_rate,
        "high_censor_warning": censor_rate > 40,
        "x_max": float(min(max(timeline), 48)),
    }


def predict_event(event: Dict[str, Any]) -> Dict[str, Any]:
    engine = _get_inference_engine()
    event_dict = {
        "event_cause": event.get("event_cause", "vehicle_breakdown"),
        "zone": event.get("zone", "Central"),
        "corridor": event.get("corridor", "Hebbal - Silk Board"),
        "hour": int(event.get("hour", 12)),
        "requires_road_closure": bool(event.get("requires_road_closure", False)),
        "lat": float(event.get("lat", 13.035)),
        "lon": float(event.get("lon", 77.597)),
    }
    ml = engine.predict(event_dict)
    survival = survival_analysis(event_dict["event_cause"])
    stis_info = compute_stis(
        event_dict["event_cause"],
        event_dict["hour"],
        event_dict["requires_road_closure"],
        event_dict["zone"],
    )
    deploy_hrs = ml["predicted_hours"]
    economic = compute_economic_impact(deploy_hrs, event_dict["hour"])

    chronic_msg = None
    if survival.get("high_censor_warning"):
        chronic_msg = (
            f"{survival['censor_rate']:.0f}% of `{event_dict['event_cause']}` events were never formally closed. "
            "Treat as potentially chronic; schedule a 6-hour follow-up check."
        )

    return {
        "event": event_dict,
        "ml_prediction": ml,
        "survival": survival,
        "stis": stis_info,
        "economic": economic,
        "chronic_warning": chronic_msg,
    }


def default_active_events(new_event: Dict[str, Any], stis: float, min_officers: int) -> List[Dict[str, Any]]:
    zone = new_event.get("zone", "Central")
    cause = new_event.get("event_cause", "accident")
    return [
        {
            "event_id": f"NEW: {cause}",
            "stis": stis,
            "zone": zone,
            "min_officers_needed": min_officers,
            "status": "NEW",
        },
        {"event_id": "Accident · Marathahalli", "stis": 7.1, "zone": zone, "min_officers_needed": 4, "status": "ACTIVE"},
        {"event_id": "Tree Fall · Indiranagar", "stis": 5.5, "zone": zone, "min_officers_needed": 4, "status": "ACTIVE"},
        {"event_id": "VB · Whitefield", "stis": 2.2, "zone": zone, "min_officers_needed": 2, "status": "ACTIVE"},
    ]


def run_triage(active_events: List[Dict[str, Any]], zone: str, capacity: int = 14) -> Dict[str, Any]:
    from src.triage_optimizer import MultiEventTriageOptimizer

    opt = MultiEventTriageOptimizer()
    result = opt.optimize(active_events, {zone: capacity})
    rows = []
    for ev in active_events:
        alloc = result["allocation"].get(ev["event_id"], {})
        assigned = alloc.get("officers_assigned", 0)
        needed = ev["min_officers_needed"]
        rows.append(
            {
                "event": ("🆕 " if ev.get("status") == "NEW" else "▶ ") + ev["event_id"],
                "stis": ev["stis"],
                "min_needed": needed,
                "assigned": assigned,
                "coverage": f"{alloc.get('coverage_ratio', 0) * 100:.0f}%",
                "decision": alloc.get("status", "PENDING"),
            }
        )
    return {
        "triage_active": result["triage_active"],
        "zone": zone,
        "capacity": capacity,
        "rows": rows,
        "allocation": result["allocation"],
    }


def get_risk_briefing(hour: int = 8) -> Dict[str, Any]:
    cluster_path = BASE_DIR / "data" / "precomputed" / "cluster_centers.json"
    tensor_path = BASE_DIR / "models" / "temporal_risk_tensor.pkl"

    if not cluster_path.exists() or not tensor_path.exists():
        return {
            "available": False,
            "hour": hour,
            "hotspots": [],
            "top_risks": _sample_risks(),
            "message": "Run precompute pipeline to generate hotspot data.",
        }

    with open(cluster_path, encoding="utf-8") as f:
        centers = json.load(f)
    risk_tensor = joblib.load(tensor_path)

    map_rows = []
    for cid_str, info in centers.items():
        cid = int(cid_str)
        if cid < risk_tensor.shape[0]:
            risk = float(risk_tensor[cid, hour])
            if risk > 0.05:
                map_rows.append(
                    {
                        "lat": info["lat"],
                        "lon": info["lon"],
                        "risk": risk,
                        "corridor": info.get("corridor", "Unknown"),
                        "cause": info.get("top_cause", "unknown"),
                        "n_events": info.get("n_events", 0),
                    }
                )

    top_risks = []
    try:
        from src.hotspot_forecaster import SpatioTemporalForecaster

        forecaster = SpatioTemporalForecaster(str(BASE_DIR / "models"), str(BASE_DIR / "data" / "precomputed"))
        forecaster.cluster_centers = {int(k): v for k, v in centers.items()}
        forecaster.risk_tensor = risk_tensor
        top_risks = forecaster.predict_risk(hour)[:8]
    except Exception as exc:
        top_risks = [{"error": str(exc)}]

    return {"available": True, "hour": hour, "hotspots": map_rows, "top_risks": top_risks}


def _sample_risks() -> List[Dict[str, Any]]:
    sample = [
        ("Hebbal - Bellary Road", 0.87, "water_logging"),
        ("Silk Board Junction", 0.81, "congestion"),
        ("Mekhri Circle", 0.74, "accident"),
        ("Marathahalli Bridge", 0.68, "vehicle_breakdown"),
        ("KR Puram", 0.55, "tree_fall"),
    ]
    return [
        {"corridor": c, "risk_score": r, "top_cause": cause, "n_events": 0}
        for c, r, cause in sample
    ]


def get_trends() -> Dict[str, Any]:
    trend_path = BASE_DIR / "data" / "precomputed" / "trend_analysis.json"
    if not trend_path.exists():
        return {"available": False}
    with open(trend_path, encoding="utf-8") as f:
        return {"available": True, **json.load(f)}


def get_station_efficiency() -> Dict[str, Any]:
    path = BASE_DIR / "data" / "precomputed" / "station_efficiency.parquet"
    if not path.exists():
        return {"available": False, "stations": [], "band_counts": {}}
    eff_df = pd.read_parquet(path).reset_index()
    eff_df.columns = [c.replace("_", " ").title() for c in eff_df.columns]
    stations = eff_df.head(20).to_dict(orient="records")
    band_counts = eff_df["Performance Band"].value_counts().to_dict() if "Performance Band" in eff_df.columns else {}
    return {"available": True, "stations": stations, "band_counts": band_counts}


def get_validation_metrics() -> List[Dict[str, str]]:
    return [
        {"title": "Road Closure AUC", "value": "0.714", "subtitle": "3-fold CV"},
        {"title": "KM Curves Fitted", "value": "15", "subtitle": "Event cause strata"},
        {"title": "Cascade Chains", "value": "1,954", "subtitle": "Detected in 8,173 events"},
        {"title": "Spatial Clusters", "value": "66", "subtitle": "DBSCAN @ 500m radius"},
    ]


def get_cascades() -> Dict[str, Any]:
    path = BASE_DIR / "data" / "precomputed" / "cascade_probabilities.json"
    if not path.exists():
        return {"available": False, "rows": []}
    with open(path, encoding="utf-8") as f:
        cascade_data = json.load(f)

    rows = []
    for cause, stats in cascade_data.items():
        p = min(stats["probability"], 1.0)
        rows.append(
            {
                "primary_cause": cause,
                "cascade_prob": round(p, 3),
                "cascade_pct": f"{p * 100:.1f}%",
                "triggered": stats["triggered_count"],
                "total_events": stats["total_events"],
                "typical_secondary": ", ".join(stats.get("typical_secondary", [])[:2]) or "—",
            }
        )
    rows.sort(key=lambda x: x["cascade_prob"], reverse=True)
    top = rows[0] if rows else None
    insight = None
    if top:
        insight = (
            f"`{top['primary_cause']}` events trigger the highest cascade rate "
            f"({top['cascade_pct']}). Pre-position patrols at adjacent junctions."
        )
    return {"available": True, "rows": rows, "top_insight": insight}


def get_planned_summary() -> Dict[str, Any]:
    survival_path = BASE_DIR / "data" / "processed" / "survival_ready.parquet"
    if not survival_path.exists():
        return {"available": False, "events": [], "summary": {}}

    df = pd.read_parquet(survival_path)
    planned_df = df[df["event_cause"].isin(PLANNED_TYPES)].copy()
    avg_duration = planned_df[planned_df["event_observed"] == 1]["duration_hrs"].median()
    summary = {
        "total_planned": int(len(planned_df)),
        "need_closure": int(planned_df["requires_road_closure"].sum()),
        "median_duration_hrs": float(avg_duration) if pd.notna(avg_duration) else None,
        "night_events": int((planned_df["hour"] >= 22).sum() + (planned_df["hour"] <= 6).sum()),
    }

    sample = planned_df.dropna(subset=["address"]).head(50) if "address" in planned_df.columns else planned_df.head(50)
    events = []
    for idx, row in sample.iterrows():
        start = row["start_datetime"]
        events.append(
            {
                "id": int(idx),
                "label": (
                    f"{str(row['event_cause']).upper()} · "
                    f"{str(row.get('address', ''))[:50]} · "
                    f"{start.strftime('%d %b %H:%M') if pd.notna(start) else 'N/A'}"
                ),
                "event_cause": str(row["event_cause"]),
                "requires_road_closure": bool(row.get("requires_road_closure", 0)),
                "hour": int(row.get("hour", 12)),
            }
        )
    return {"available": True, "summary": summary, "events": events, "types": PLANNED_TYPES}


def get_planned_dossier(event_id: int) -> Dict[str, Any]:
    survival_path = BASE_DIR / "data" / "processed" / "survival_ready.parquet"
    cascade_path = BASE_DIR / "data" / "precomputed" / "cascade_probabilities.json"
    if not survival_path.exists():
        return {"available": False}

    df = pd.read_parquet(survival_path)
    if event_id not in df.index:
        return {"available": False, "message": "Event not found"}

    row = df.loc[event_id]
    cause = str(row["event_cause"])
    cascade_data = {}
    if cascade_path.exists():
        with open(cascade_path, encoding="utf-8") as f:
            cascade_data = json.load(f)

    cascade_info = cascade_data.get(cause, {})
    cascade_prob = min(cascade_info.get("probability", 0), 1.0)
    secondary = cascade_info.get("typical_secondary", [])
    needs_closure = int(row.get("requires_road_closure", 0))

    planned_df = df[df["event_cause"].isin(PLANNED_TYPES)]
    cause_durations = planned_df[(planned_df["event_cause"] == cause) & (planned_df["event_observed"] == 1)][
        "duration_hrs"
    ]
    median_hrs = float(cause_durations.median()) if len(cause_durations) > 0 else 4.0
    p90_hrs = float(cause_durations.quantile(0.90)) if len(cause_durations) > 0 else 8.0
    p10_hrs = float(cause_durations.quantile(0.10)) if len(cause_durations) > 0 else 1.0

    officers = OFFICER_MAP_PLANNED.get(cause, 5)
    if needs_closure:
        officers += 4

    stis = STIS_BASE.get(cause, 6.0)
    exec_parts = [
        f"This {cause.replace('_', ' ')} event requires {officers} officers "
        f"and is expected to last {median_hrs:.1f} hours (worst case: {p90_hrs:.1f}h)."
    ]
    if needs_closure:
        exec_parts.append("Road closure is required — pre-position barricades 1 hour before start.")
    if cascade_prob > 0.1:
        exec_parts.append(
            f"{int(cascade_prob * 100)}% historical cascade rate — alert nearby patrols for "
            f"{', '.join(secondary[:2])} events."
        )
    else:
        exec_parts.append("Cascade risk is low.")
    exec_parts.append("Pre-position units by 1 hour before event start.")

    return {
        "available": True,
        "event_id": event_id,
        "cause": cause,
        "median_hrs": median_hrs,
        "p10_hrs": p10_hrs,
        "p90_hrs": p90_hrs,
        "stis": stis,
        "stis_label": severity_label(stis),
        "officers": officers,
        "needs_closure": bool(needs_closure),
        "cascade_prob": cascade_prob,
        "secondary_causes": secondary[:2],
        "executive_summary": " ".join(exec_parts),
    }


def get_planned_analytics() -> Dict[str, Any]:
    survival_path = BASE_DIR / "data" / "processed" / "survival_ready.parquet"
    if not survival_path.exists():
        return {"available": False}
    planned_df = pd.read_parquet(survival_path)
    planned_df = planned_df[planned_df["event_cause"].isin(PLANNED_TYPES)]
    dur_data = (
        planned_df[planned_df["event_observed"] == 1]
        .groupby("event_cause")["duration_hrs"]
        .median()
        .reset_index()
    )
    hour_data = planned_df.groupby("hour").size().reset_index(name="count")
    return {
        "available": True,
        "duration_by_cause": dur_data.to_dict(orient="records"),
        "events_by_hour": hour_data.to_dict(orient="records"),
    }


# ── Phase 2 graph features (additive — no ML model changes) ─────────────────

@lru_cache(maxsize=1)
def _graph_enricher():
    from src.graph_feature_bridge import GraphFeatureEnricher

    return GraphFeatureEnricher.from_precomputed()


def get_graph_status() -> Dict[str, Any]:
    enricher = _graph_enricher()
    features_path = BASE_DIR / "data" / "precomputed" / "corridor_graph_features.json"
    meta = {}
    if features_path.exists():
        with open(features_path, encoding="utf-8") as f:
            meta = json.load(f)
    return {
        "ready": enricher.ready,
        "feature_names": enricher.feature_names,
        "n_nodes": meta.get("n_nodes"),
        "n_edges": meta.get("n_edges"),
        "artifacts_path": str(BASE_DIR / "data" / "precomputed"),
    }


def get_graph_features(event: Dict[str, Any]) -> Dict[str, Any]:
    enricher = _graph_enricher()
    k_hops = int(event.get("k_hops", 2))
    feats = enricher.features_for_dict(event, k_hops=k_hops)
    return {"available": enricher.ready, "features": feats, "event": event}


def get_affected_subgraph(event: Dict[str, Any]) -> Dict[str, Any]:
    enricher = _graph_enricher()
    k_hops = int(event.get("k_hops", 2))
    return enricher.affected_subgraph(event, k_hops=k_hops)


# ── Phase 3 prescriptive engine (additive — consumes V12 outputs, no ML calls) ─

@lru_cache(maxsize=1)
def _prescriptive_engine():
    from src.prescriptive import PrescriptiveEngine

    return PrescriptiveEngine()


def get_prescriptive_recommendation(
    event: Dict[str, Any],
    zone_capacity: Optional[Dict[str, int]] = None,
    concurrent_events: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    return _prescriptive_engine().recommend_one(event, zone_capacity, concurrent_events)


def get_prescriptive_batch(
    events: List[Dict[str, Any]],
    zone_capacity: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    return _prescriptive_engine().recommend_batch(events, zone_capacity)


def run_prescriptive_whatif(
    event: Dict[str, Any],
    recommendations: Dict[str, Any],
) -> Dict[str, Any]:
    from src.prescriptive import WhatIfSimulator, normalize_v12_event

    ev = normalize_v12_event(event)
    mp = {"allocation": recommendations.get("allocation", {})}
    bd = recommendations.get("barricading") or recommendations.get("barricade_diversion") or {}
    if "barricade_level" not in bd and "level" in bd:
        bd = {**bd, "barricade_level": bd["level"]}
    return WhatIfSimulator().simulate(ev, mp, bd)