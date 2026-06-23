"""
Production Inference + Recommendation Engine for Event-Driven Congestion
========================================================================
This is the heart of the solution to the operational challenge.

Given an incoming (or planned) event:
  - Predicts duration (P50 + interval)
  - Predicts closure probability
  - Predicts composite congestion impact
  - Outputs concrete recommendations:
      * Recommended manpower (officers)
      * Barricading decision + intensity
      * Diversion / watch corridors
      * Cascade watch list
      * Priority / urgency

Combines:
  - Advanced GM feature engineering (planned density, text signals, spatial centrality, interactions)
  - Multi-output models
  - Simple but effective decision rules (easy to explain to BTP)
"""
import json
import joblib
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from typing import Dict, Any, List
import logging

# allow direct run
sys.path.insert(0, str(Path(__file__).resolve().parent))
from advanced_event_fe import EventImpactFeatureEngineer, PLANNED_CAUSES, load_or_fit_fe

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent

class EventImpactPredictor:
    def __init__(self):
        self.fe: EventImpactFeatureEngineer = load_or_fit_fe()
        self.dur_model = joblib.load(BASE_DIR / "models" / "lgb_event_dur_quantile.pkl")
        self.closure_model = joblib.load(BASE_DIR / "models" / "lgb_event_closure.pkl")
        self.impact_model = joblib.load(BASE_DIR / "models" / "lgb_event_impact.pkl")

        with open(BASE_DIR / "models" / "event_impact_feature_list.json") as f:
            meta = json.load(f)
            self.feature_cols = meta["features"]

        # Load corridor density for explainability
        try:
            with open(BASE_DIR / "models" / "event_corridor_density.json") as f:
                d = json.load(f)
                self.corridor_density = d.get("density", {})
        except:
            self.corridor_density = {}

    def _safe_predict(self, model, X: pd.DataFrame) -> np.ndarray:
        Xf = X.reindex(columns=self.feature_cols, fill_value=0.0)
        return model.predict(Xf)

    def predict(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        event: dict with at minimum:
            event_cause, latitude, longitude, corridor, zone, hour, requires_road_closure (optional),
            description, priority, event_type ('planned'/'unplanned')
        """
        row = pd.DataFrame([event])
        # ensure minimal columns for FE (robust to partial input)
        defaults = {
            'event_type': 'unplanned',
            'priority': 'Low',
            'description': '',
            'reason_breakdown': '',
            'police_station': 'unknown',
            'requires_road_closure': 0
        }
        for col, val in defaults.items():
            if col not in row.columns:
                row[col] = val

        row['start_datetime'] = pd.Timestamp.now().normalize() + pd.Timedelta(hours=int(event.get('hour', 12)))

        # Advanced features
        row_fe = self.fe.transform(row)
        X = row_fe

        # Predictions
        dur_log = float(self._safe_predict(self.dur_model, X)[0])
        dur_p50 = max(0.2, float(np.expm1(dur_log)))

        closure_proba = float(self._safe_predict(self.closure_model, X)[0])
        # clip to sane range
        closure_proba = float(np.clip(closure_proba, 0.01, 0.99))

        impact_log = float(self._safe_predict(self.impact_model, X)[0])
        impact = max(0.3, float(np.expm1(impact_log)))

        # Intervals (simple heuristic calibrated from old project + impact)
        p10 = max(0.15, dur_p50 * 0.35)
        p90 = max(dur_p50 + 0.3, dur_p50 * 2.6)

        is_planned = event.get('event_type', 'unplanned') == 'planned' or event.get('event_cause') in PLANNED_CAUSES

        # ========== RECOMMENDATION ENGINE ==========
        recs = self._generate_recommendations(
            event=event,
            dur_p50=dur_p50,
            p90=p90,
            closure_proba=closure_proba,
            impact=impact,
            is_planned=is_planned,
            row_fe=row_fe.iloc[0]
        )

        return {
            "predicted_duration_hrs": round(dur_p50, 2),
            "p10_hrs": round(p10, 2),
            "p90_hrs": round(p90, 2),
            "closure_probability": round(closure_proba, 3),
            "congestion_impact_score": round(impact, 2),
            "is_planned_event": bool(is_planned),
            "recommendations": recs,
            "model": "EventImpact_v1_GrandmasterFE"
        }

    def _generate_recommendations(
        self,
        event: Dict,
        dur_p50: float,
        p90: float,
        closure_proba: float,
        impact: float,
        is_planned: bool,
        row_fe: pd.Series
    ) -> Dict[str, Any]:
        cause = event.get('event_cause', 'vehicle_breakdown')
        corridor = event.get('corridor', 'unknown')
        zone = event.get('zone', 'Central')
        requires_closure = bool(event.get('requires_road_closure', False))
        hour = int(event.get('hour', 12))

        # 1. Manpower (officers)
        base = 2
        if cause in ['accident', 'water_logging', 'tree_fall']:
            base = 5
        if cause in ['construction', 'public_event', 'procession']:
            base = 4

        # scale by impact + closure
        scale = 1.0 + (impact / 12.0) + (closure_proba * 1.8)
        if requires_closure:
            scale *= 1.35
        if is_planned and hour >= 22 or hour <= 6:
            scale *= 0.9   # night works sometimes need fewer on street

        officers = int(np.clip(round(base * scale), 2, 14))

        # 2. Barricading
        barricade = "NONE"
        if closure_proba > 0.55 or requires_closure or cause in ['construction', 'tree_fall', 'water_logging']:
            barricade = "FULL"
        elif closure_proba > 0.30 or cause in ['public_event', 'procession', 'accident']:
            barricade = "PARTIAL"

        # 3. Diversion / positioning
        diversion = []
        if row_fe.get('corridor_planned_density', 0) > 0.15:
            diversion.append(f"High historical planned activity on {corridor}. Pre-position patrols on parallel routes.")
        if row_fe.get('corridor_centrality', 0) > 0.25:
            diversion.append("Major corridor — prepare variable message signs and adjacent junction monitoring.")
        if is_planned:
            diversion.append("Event known in advance: coordinate with organizers for timed lane closures if possible.")
        else:
            diversion.append("Reactive incident — monitor for secondary breakdowns within 2km / 45min window.")

        # 4. Cascade watch
        watch_for = []
        if cause in ['accident', 'tree_fall']:
            watch_for = ['vehicle_breakdown', 'congestion']
        elif cause in ['water_logging', 'construction']:
            watch_for = ['accident', 'vehicle_breakdown']
        elif cause == 'public_event':
            watch_for = ['congestion', 'accident']

        # 5. Urgency / briefing text
        if impact > 8 or (closure_proba > 0.6 and dur_p50 > 3):
            urgency = "CRITICAL - Escalate to Command immediately"
        elif impact > 4 or closure_proba > 0.4:
            urgency = "HIGH"
        else:
            urgency = "STANDARD"

        return {
            "manpower_officers": officers,
            "barricade": barricade,
            "diversion_guidance": diversion,
            "cascade_watch": watch_for,
            "urgency": urgency,
            "plain_english": (
                f"{'PLANNED ' if is_planned else ''}{cause} on {corridor}. "
                f"Expect ~{dur_p50:.1f}h (worst {p90:.1f}h). "
                f"Closure risk {int(closure_proba*100)}%. "
                f"Deploy {officers} officers + {barricade} barricades. "
                f"Impact score {impact:.1f}."
            )
        }

# Quick singleton
_predictor = None
def get_event_impact_predictor() -> EventImpactPredictor:
    global _predictor
    if _predictor is None:
        _predictor = EventImpactPredictor()
    return _predictor

if __name__ == "__main__":
    predictor = get_event_impact_predictor()

    # Example planned event (construction on a high-density corridor)
    test_planned = {
        "event_cause": "construction",
        "event_type": "planned",
        "latitude": 13.01,
        "longitude": 77.65,
        "corridor": "ORR East 2",
        "zone": "East",
        "hour": 23,
        "requires_road_closure": True,
        "description": "kride work and metro station construction",
        "priority": "High"
    }

    result = predictor.predict(test_planned)
    print(json.dumps(result, indent=2, default=str))