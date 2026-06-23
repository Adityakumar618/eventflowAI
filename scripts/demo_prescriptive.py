import os
import sys
import json
from pathlib import Path

# Ensure src is in the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prescriptive import PrescriptiveEngine

def get_demo_events():
    return [
        {
            "event_id": "demo_1",
            "event_cause": "water_logging",
            "corridor": "Hebbal - Silk Board",
            "zone": "Central",
            "police_station": "Hebbal",
            "hour": 18,
            "lat": 13.035,
            "lon": 77.597,
            "requires_road_closure": False,
            "predicted_hours": 3.5,
            "p35_hours": 2.1,
            "p50_hours": 3.0,
            "p65_hours": 4.5,
            "impact_score": 8.5,
            "regime": "medium",
            "active_overlap": 2,
            "graph_features": {},
            "affected_subgraph": {}
        },
        {
            "event_id": "demo_2",
            "event_cause": "accident",
            "corridor": "Silk Board Junction",
            "zone": "South",
            "police_station": "Madiwala",
            "hour": 8,
            "lat": 12.9177,
            "lon": 77.6228,
            "requires_road_closure": True,
            "predicted_hours": 5.2,
            "p35_hours": 4.0,
            "p50_hours": 5.0,
            "p65_hours": 6.5,
            "impact_score": 9.2,
            "regime": "long",
            "active_overlap": 4,
            "graph_features": {},
            "affected_subgraph": {}
        },
        {
            "event_id": "demo_3",
            "event_cause": "tree_fall",
            "corridor": "Mekhri Circle",
            "zone": "North",
            "police_station": "Yeshwanthpur",
            "hour": 14,
            "lat": 13.0090,
            "lon": 77.5770,
            "requires_road_closure": False,
            "predicted_hours": 1.5,
            "p35_hours": 1.0,
            "p50_hours": 1.4,
            "p65_hours": 2.2,
            "impact_score": 3.4,
            "regime": "short",
            "active_overlap": 0,
            "graph_features": {},
            "affected_subgraph": {}
        }
    ]

def main():
    engine = PrescriptiveEngine()
    events = get_demo_events()
    
    results = []
    print("="*60)
    print("GridGuard V12 Prescriptive Engine - Live Demo")
    print("="*60)
    for ev in events:
        print(f"\nProcessing Event: {ev['event_id']} ({ev['event_cause']} @ {ev['corridor']})")
        print(f"Predicted Duration: {ev['predicted_hours']}h | Impact Score: {ev['impact_score']}")
        
        try:
            res = engine.recommend_one(ev)
            results.append({"event": ev, "recommendation": res})
            
            mp = res.get('manpower', {})
            bd = res.get('barricade_diversion', {})
            wi = res.get('what_if', {})
            
            print(f"  -> Manpower: {mp.get('total_deployed', 'N/A')} officers from {mp.get('station', 'Nearest')}")
            print(f"  -> Barricade Level: {bd.get('barricade_level', 'N/A')}")
            if bd.get('diversion_routes'):
                print(f"  -> Diversion Routes: {', '.join([r if isinstance(r, str) else r.get('route', '') for r in bd.get('diversion_routes')[:2]])}")
            
            if wi.get('delay_reduction_pct') is not None:
                print(f"  -> Est. Delay Reduction: -{wi.get('delay_reduction_pct'):.1f}%")
            print(f"  -> Rationale: {res.get('rationale', '')}")
            
        except Exception as e:
            print(f"  -> ERROR generating recommendation: {e}")
            
    out_dir = Path("experiments/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "demo_prescriptive_output.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Full output saved to {out_file}")

if __name__ == '__main__':
    main()
