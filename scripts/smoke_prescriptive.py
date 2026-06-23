#!/usr/bin/env python
"""Smoke test for Phase 3 prescriptive engine (no ML inference)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.prescriptive import PrescriptiveEngine, build_prescriptive_response

V12_SAMPLE = {
    "event_id": "EVT-001",
    "event_cause": "accident",
    "corridor": "Bellary Road 1",
    "zone": "Central",
    "police_station": "Hebbal",
    "hour": 18,
    "lat": 13.04,
    "lon": 77.59,
    "requires_road_closure": True,
    "p35_hours": 0.6,
    "p50_hours": 1.2,
    "p65_hours": 2.8,
    "impact_score": 7.4,
    "regime": "medium",
    "active_overlap": 2,
}

CONCURRENT = [
    V12_SAMPLE,
    {
        "event_id": "EVT-002",
        "event_cause": "tree_fall",
        "corridor": "ORR East 1",
        "zone": "Central",
        "lat": 12.95,
        "lon": 77.70,
        "p50_hours": 3.5,
        "impact_score": 6.1,
        "regime": "long",
        "active_overlap": 1,
    },
]


def main():
    engine = PrescriptiveEngine()
    single = engine.recommend_one(V12_SAMPLE, zone_capacity={"Central": 14}, concurrent_events=CONCURRENT)
    batch = engine.recommend_batch(CONCURRENT, zone_capacity={"Central": 14})

    print("=== Single recommendation ===")
    print(json.dumps({
        "solver": single["manpower_allocation"]["solver"],
        "officers": single["recommendations"]["manpower"],
        "barricade": single["recommendations"]["barricading"]["level"],
        "diversions": len(single["recommendations"]["diversions"]),
        "improvement_pct": single["what_if"]["improvement_pct"],
        "confidence": single["confidence"],
    }, indent=2))

    print("\n=== Batch triage_active ===", batch["manpower_allocation"]["triage_active"])
    print("SMOKE OK")


if __name__ == "__main__":
    main()