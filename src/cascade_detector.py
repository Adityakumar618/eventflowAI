import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1-a))


class CascadeDetector:
    """
    Mines historical cascades: secondary incidents triggered within
    DIST_KM and TIME_MIN of a primary event.
    """
    DIST_KM  = 2.0
    TIME_MIN = 45

    CASCADE_RELATIONSHIPS = {
        'accident':         ['vehicle_breakdown', 'congestion'],
        'tree_fall':        ['accident', 'congestion', 'road_conditions'],
        'water_logging':    ['vehicle_breakdown', 'accident', 'road_conditions'],
        'construction':     ['congestion', 'road_conditions'],
        'vip_movement':     ['congestion'],
        'public_event':     ['congestion', 'accident'],
        'vehicle_breakdown':['congestion'],
    }

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.cascade_probs = {}
        self.cascade_counts = {}

    def detect_historical_cascades(self, df: pd.DataFrame) -> dict:
        logging.info("Detecting historical cascade chains...")

        df_closed = df[df['event_observed'] == 1].copy()
        df_closed = df_closed.dropna(subset=['latitude', 'longitude', 'start_datetime'])
        df_sorted = df_closed.sort_values('start_datetime').reset_index(drop=True)

        cascade_pairs = []

        for i, primary in df_sorted.iterrows():
            # Window: events starting within TIME_MIN after primary
            t_start = primary['start_datetime']
            t_end   = t_start + pd.Timedelta(minutes=self.TIME_MIN)
            nearby  = df_sorted[
                (df_sorted['start_datetime'] > t_start) &
                (df_sorted['start_datetime'] <= t_end)
            ]

            for j, secondary in nearby.iterrows():
                if i == j:
                    continue
                try:
                    dist = haversine(
                        primary['latitude'], primary['longitude'],
                        secondary['latitude'], secondary['longitude']
                    )
                    if dist <= self.DIST_KM:
                        cascade_pairs.append({
                            'primary_cause':   primary['event_cause'],
                            'secondary_cause': secondary['event_cause'],
                            'dist_km':         round(dist, 2),
                            'delay_min':       round(
                                (secondary['start_datetime'] - t_start).total_seconds() / 60, 1)
                        })
                except Exception:
                    continue

        logging.info(f"Found {len(cascade_pairs)} cascade pairs.")

        # Compute cascade probability per primary cause
        cause_counts   = df_sorted['event_cause'].value_counts().to_dict()
        cascade_df     = pd.DataFrame(cascade_pairs) if cascade_pairs else pd.DataFrame(columns=['primary_cause','secondary_cause'])

        result = {}
        for cause, total in cause_counts.items():
            triggered = len(cascade_df[cascade_df['primary_cause'] == cause]) if len(cascade_df) else 0
            result[cause] = {
                'probability':     round(triggered / max(total, 1), 4),
                'triggered_count': triggered,
                'total_events':    total,
                'typical_secondary': (
                    cascade_df[cascade_df['primary_cause'] == cause]['secondary_cause']
                    .value_counts().head(3).index.tolist()
                    if len(cascade_df) else self.CASCADE_RELATIONSHIPS.get(cause, [])
                )
            }

        self.cascade_probs  = {k: v['probability'] for k, v in result.items()}
        self.cascade_counts = result

        # Save
        out_path = self.models_dir / 'cascade_probabilities.json'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)
        logging.info(f"Cascade probabilities saved to {out_path}")
        return result

    def warn(self, event: dict) -> dict:
        cause = event.get('event_cause', 'unknown')
        prob  = self.cascade_probs.get(cause, 0.0)
        data  = self.cascade_counts.get(cause, {})
        watch = data.get('typical_secondary',
                         self.CASCADE_RELATIONSHIPS.get(cause, []))

        if prob < 0.03:
            return {'cascade_risk': 'LOW', 'probability': prob, 'watch_for': []}

        return {
            'cascade_risk':      'HIGH' if prob > 0.15 else 'MEDIUM',
            'probability':        round(prob, 3),
            'warning': (
                f"⚠️ {int(prob*100)}% of {cause} events on this corridor "
                f"triggered a secondary incident within {self.TIME_MIN} minutes."
            ),
            'watch_for':          watch,
            'watch_radius_km':    self.DIST_KM,
            'watch_window_min':   self.TIME_MIN,
            'pre_position_rec': (
                f"Alert nearby patrols to watch for "
                f"{' or '.join(watch)} events within {self.DIST_KM}km."
                if watch else "Monitor surrounding junctions."
            )
        }


if __name__ == "__main__":
    df = pd.read_parquet("data/processed/survival_ready.parquet")
    detector = CascadeDetector("data/precomputed")
    result = detector.detect_historical_cascades(df)
    for cause, stats in sorted(result.items(), key=lambda x: -x[1]['probability'])[:8]:
        print(f"{cause:25s}  cascade_prob={stats['probability']:.3f}  n={stats['triggered_count']}/{stats['total_events']}")
