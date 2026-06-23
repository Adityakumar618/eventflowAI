import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from sklearn.cluster import DBSCAN
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


class SpatioTemporalForecaster:
    """
    DBSCAN clusters events spatially, then builds a temporal risk tensor
    (cluster × hour_of_day) to forecast where and when events are most likely.
    """
    # DBSCAN params: ~500m radius in radian space (500m / 6371km)
    EARTH_RADIUS_KM = 6371.0
    EPS_KM = 0.5
    MIN_SAMPLES = 5

    def __init__(self, models_dir: str = "models", precomputed_dir: str = "data/precomputed"):
        self.models_dir      = Path(models_dir)
        self.precomputed_dir = Path(precomputed_dir)
        self.precomputed_dir.mkdir(parents=True, exist_ok=True)
        self.dbscan          = None
        self.cluster_centers = {}
        self.risk_tensor     = None   # shape: (n_clusters, 24)

    def fit_dbscan(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Fitting DBSCAN spatial clustering...")
        valid = df.dropna(subset=['latitude', 'longitude']).copy()

        coords_rad = np.radians(valid[['latitude', 'longitude']].values)
        eps_rad    = self.EPS_KM / self.EARTH_RADIUS_KM

        self.dbscan = DBSCAN(
            eps=eps_rad, min_samples=self.MIN_SAMPLES,
            algorithm='ball_tree', metric='haversine'
        )
        valid['cluster'] = self.dbscan.fit_predict(coords_rad)

        n_clusters = len(set(valid['cluster'])) - (1 if -1 in valid['cluster'].values else 0)
        logging.info(f"Found {n_clusters} spatial clusters ({(valid['cluster']==-1).sum()} noise points)")

        # Cluster centers
        for cid in sorted(valid['cluster'].unique()):
            if cid == -1:
                continue
            sub = valid[valid['cluster'] == cid]
            self.cluster_centers[int(cid)] = {
                'lat':      round(sub['latitude'].mean(), 5),
                'lon':      round(sub['longitude'].mean(), 5),
                'n_events': len(sub),
                'top_cause': sub['event_cause'].value_counts().index[0],
                'corridor': sub['corridor'].value_counts().index[0] if 'corridor' in sub else 'Unknown'
            }

        joblib.dump(self.dbscan, self.models_dir / 'dbscan_clusters.pkl')
        return valid

    def compute_temporal_risk_tensor(self, df_clustered: pd.DataFrame):
        logging.info("Computing temporal risk tensor (cluster × hour)...")
        valid = df_clustered[df_clustered['cluster'] >= 0].copy()

        clusters = sorted(valid['cluster'].unique())
        n_clusters = max(clusters) + 1
        tensor = np.zeros((n_clusters, 24))

        for cid in clusters:
            sub = valid[valid['cluster'] == cid]
            for hour, count in sub['hour'].value_counts().items():
                tensor[int(cid), int(hour)] = count

        # Normalize each cluster row to [0, 1]
        row_max = tensor.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1
        self.risk_tensor = tensor / row_max

        joblib.dump(self.risk_tensor, self.models_dir / 'temporal_risk_tensor.pkl')
        logging.info("Temporal risk tensor saved.")

    def predict_risk(self, target_hour: int) -> list:
        """Return top-5 hotspot clusters for a given hour."""
        if self.risk_tensor is None:
            self.risk_tensor = joblib.load(self.models_dir / 'temporal_risk_tensor.pkl')

        hour_risk = self.risk_tensor[:, target_hour]
        top_idx   = np.argsort(hour_risk)[::-1][:10]

        results = []
        for cid in top_idx:
            if hour_risk[cid] < 0.1:
                continue
            center = self.cluster_centers.get(int(cid), {})
            if not center:
                continue
            results.append({
                'cluster_id':  int(cid),
                'risk_score':  round(float(hour_risk[cid]), 3),
                'lat':         center.get('lat'),
                'lon':         center.get('lon'),
                'n_events':    center.get('n_events', 0),
                'top_cause':   center.get('top_cause', 'unknown'),
                'corridor':    center.get('corridor', 'Unknown'),
                'recommendation': f"Pre-position 1 patrol unit near {center.get('corridor','this junction')} before {target_hour:02d}:00"
            })
        return results

    def run_full_pipeline(self, df: pd.DataFrame):
        df_clustered = self.fit_dbscan(df)
        self.compute_temporal_risk_tensor(df_clustered)

        # Save cluster centers
        with open(self.precomputed_dir / 'cluster_centers.json', 'w') as f:
            import json
            json.dump(self.cluster_centers, f, indent=2)
        logging.info("Full spatiotemporal pipeline complete.")
        return df_clustered


if __name__ == "__main__":
    df = pd.read_parquet("data/processed/survival_ready.parquet")
    forecaster = SpatioTemporalForecaster()
    forecaster.run_full_pipeline(df)

    # Quick test
    risks = forecaster.predict_risk(target_hour=18)
    print(f"\nTop hotspots for 18:00:")
    for r in risks[:5]:
        print(f"  Cluster {r['cluster_id']:3d} | risk={r['risk_score']:.2f} | {r['corridor']} | {r['top_cause']}")
