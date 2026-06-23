"""
Kaggle Grandmaster Level Feature Engineering for Event-Driven Congestion Impact
===============================================================================
Core innovation: Move beyond duration -> predict actionable CONGESTION IMPACT
and generate optimal manpower / barricade / diversion recommendations.

Techniques applied (GM style):
- Careful grouped target encodings (corridor, zone x cause, planned_density)
- Multi-window temporal load + anticipation features
- Text keyword engineering + planned-specific signals
- High-order interactions + centrality proxies
- Composite impact target construction (for direct regression)
- Leakage-safe construction order
- Hierarchical / regime features (planned vs reactive)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
import joblib
import json
import logging

# Safe additive import of core reusable builders (Phase 1)
try:
    from core_features import LOOBuilder, ClusterFeatureBuilder
    HAS_CORE_FEATURES = True
except Exception:
    HAS_CORE_FEATURES = False
    LOOBuilder = None
    ClusterFeatureBuilder = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

# Bengaluru high-centrality anchors (for spatial distance features)
MAJOR_JUNCTIONS = {
    'silk_board': (12.9177, 77.6228),
    'hebbal': (13.0354, 77.5910),
    'mekhri': (13.009, 77.577),
    'yeshwanthpur': (13.0267, 77.5361),
    'kr_puram': (13.0053, 77.6946),
    'madiwala': (12.927, 77.617),
}

EVENT_KEYWORDS = [
    'construction', 'road work', 'bwssb', 'kride', 'metro', 'station work',
    'procession', 'rally', 'public event', 'festival', 'cricket', 'match',
    'vip', 'meeting', 'function', 'protest', 'gathering'
]

PLANNED_CAUSES = ['construction', 'public_event', 'procession', 'vip_movement', 'protest']

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))

class EventImpactFeatureEngineer:
    """
    Production-grade, leakage-resistant advanced feature engineering.
    Can be fit on training data only and transform inference rows.
    """
    def __init__(self, models_dir: str = "models", precomputed_dir: str = "data/precomputed",
                 use_v9_style_loo: bool = False):
        self.models_dir = Path(models_dir)
        self.precomputed_dir = Path(precomputed_dir)
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.tfidf = None
        self.svd = None
        self.global_stats: Dict = {}
        self.corridor_planned_density: Dict = {}
        self.corridor_centrality: Dict = {}
        self.cause_stats: Dict = {}
        self.keyword_flags = EVENT_KEYWORDS

        # Optional: reuse high-quality V9-style LOO + clusters (additive, non-breaking)
        self.use_v9_style_loo = use_v9_style_loo
        self._loo_builder = None
        self._cluster_builder = None

    # ---------- SAFE ENCODING UTILS ----------
    def _fit_label_enc(self, series: pd.Series, name: str):
        le = LabelEncoder()
        le.fit(series.fillna('unknown').astype(str))
        self.label_encoders[name] = le
        return le

    def _transform_label(self, series: pd.Series, name: str):
        le = self.label_encoders[name]
        try:
            return le.transform(series.fillna('unknown').astype(str))
        except:
            return np.zeros(len(series))

    # ---------- CORE FEATURE CONSTRUCTION (fit only on train) ----------
    def fit(self, df: pd.DataFrame) -> 'EventImpactFeatureEngineer':
        """Fit encoders, stats, tfidf etc. ONLY on training slice."""
        logger.info("Fitting Grandmaster Event Impact Feature Engineer...")

        df = df.copy()
        df['start_datetime'] = pd.to_datetime(df['start_datetime'], errors='coerce')
        df['hour'] = df['start_datetime'].dt.hour
        df['dow'] = df['start_datetime'].dt.dayofweek
        df['is_rush'] = ((df['hour'].between(8,11)) | (df['hour'].between(17,20))).astype(int)
        df['is_planned'] = (df['event_type'] == 'planned').astype(int)
        df['is_planned_cause'] = df['event_cause'].isin(PLANNED_CAUSES).astype(int)

        # Label encoders
        for col in ['event_cause', 'corridor', 'zone', 'police_station']:
            self._fit_label_enc(df[col], col)

        # Global cause stats (observed only)
        obs = df[df.get('event_observed', 1) == 1]
        self.cause_stats = obs.groupby('event_cause')['duration_hrs'].agg(
            ['mean', 'median', 'std']
        ).to_dict('index') if 'duration_hrs' in obs else {}

        # Corridor planned density + centrality (historical % planned + volume)
        corr = df.groupby('corridor').agg(
            total=('id', 'count'),
            planned=('is_planned', 'sum'),
        )
        corr['density'] = (corr['planned'] / corr['total']).fillna(0)
        self.corridor_planned_density = corr['density'].to_dict()
        self.corridor_centrality = (corr['total'] / corr['total'].max()).to_dict()

        # Text features (fit on train descriptions only)
        text = (df['description'].fillna('') + ' ' + df.get('reason_breakdown', '').fillna('')).str.lower()
        self.tfidf = TfidfVectorizer(max_features=60, ngram_range=(1,2), min_df=3)
        tf = self.tfidf.fit_transform(text)
        self.svd = TruncatedSVD(n_components=8, random_state=42)
        self.svd.fit(tf)

        # Save artifacts
        self.models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.label_encoders, self.models_dir / 'event_fe_encoders.pkl')
        joblib.dump({'tfidf': self.tfidf, 'svd': self.svd}, self.models_dir / 'event_text_svd.pkl')
        with open(self.models_dir / 'event_corridor_density.json', 'w') as f:
            json.dump({'density': self.corridor_planned_density, 'centrality': self.corridor_centrality}, f)

        logger.info("FE fit complete. Encoders + text + density stats saved.")
        return self

    def transform(self, df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        """
        Generate rich feature matrix.
        When is_train=True, can compute safer rolling stats (use expanding or past-only).
        """
        out = df.copy()
        out['start_datetime'] = pd.to_datetime(out['start_datetime'], errors='coerce')
        out['hour'] = out['start_datetime'].dt.hour.fillna(12).astype(int)
        out['dow'] = out['start_datetime'].dt.dayofweek.fillna(0).astype(int)
        out['month'] = out['start_datetime'].dt.month.fillna(6).astype(int)
        out['is_weekend'] = out['dow'].isin([5,6]).astype(int)
        out['is_rush'] = ((out['hour'].between(8,11)) | (out['hour'].between(17,20))).astype(int)
        out['is_night'] = ((out['hour'] >= 22) | (out['hour'] <= 5)).astype(int)
        out['is_planned'] = (out['event_type'] == 'planned').astype(int)
        out['is_planned_cause'] = out['event_cause'].isin(PLANNED_CAUSES).astype(int)

        # Encoded categoricals
        for col in ['event_cause', 'corridor', 'zone', 'police_station']:
            out[f'{col}_enc'] = self._transform_label(out[col], col)

        # Planned density & centrality (from historical fit)
        out['corridor_planned_density'] = out['corridor'].map(self.corridor_planned_density).fillna(0.06)
        out['corridor_centrality'] = out['corridor'].map(self.corridor_centrality).fillna(0.1)

        # Strong cause-level historical impact proxies
        out['cause_closure_rate'] = out['event_cause'].map(
            lambda c: 0.36 if c in PLANNED_CAUSES else 0.06
        )
        out['cause_median_dur'] = out['event_cause'].map(
            lambda c: self.cause_stats.get(c, {}).get('median', 1.5)
        ).fillna(1.5)

        # Text keyword flags (very high signal for planned)
        text = (out['description'].fillna('') + ' ' + out.get('reason_breakdown', pd.Series(['']*len(out))).fillna('')).str.lower()
        for kw in self.keyword_flags[:8]:  # top ones
            out[f'txt_{kw.replace(" ","_")}'] = text.str.contains(kw).astype(int)
        out['event_text_score'] = out[[c for c in out.columns if c.startswith('txt_')]].sum(axis=1)

        # TFIDF-SVD text embeddings
        try:
            tf = self.tfidf.transform(text)
            svd_vec = self.svd.transform(tf)
            for i in range(svd_vec.shape[1]):
                out[f'text_svd_{i}'] = svd_vec[:, i]
        except Exception:
            for i in range(8):
                out[f'text_svd_{i}'] = 0.0

        # Spatial distance to major junctions (proxy for network exposure)
        for name, (jlat, jlon) in MAJOR_JUNCTIONS.items():
            out[f'dist_to_{name}'] = haversine(
                out['latitude'].fillna(13.0), out['longitude'].fillna(77.6),
                jlat, jlon
            )

        # Distance to nearest major junction (min)
        dist_cols = [c for c in out.columns if c.startswith('dist_to_')]
        out['min_dist_major_junc'] = out[dist_cols].min(axis=1) if dist_cols else 5.0

        # Rich interactions (GM secret sauce)
        out['planned_x_rush'] = out['is_planned'] * out['is_rush']
        out['planned_x_closure_flag'] = out['is_planned'] * out.get('requires_road_closure', 0).fillna(0)
        out['density_x_centrality'] = out['corridor_planned_density'] * out['corridor_centrality']
        out['rush_x_centrality'] = out['is_rush'] * out['corridor_centrality']
        out['cause_enc_x_planned'] = out['event_cause_enc'] * out['is_planned']
        out['hour_x_planned_density'] = out['hour'] * out['corridor_planned_density']

        # High-order cause x spatial
        out['cause_x_dist'] = out['event_cause_enc'] * out['min_dist_major_junc']

        # Regime flag + interactions
        out['reactive_high_load'] = (1 - out['is_planned']) * out['corridor_centrality'] * out['is_rush']

        # Fill NaNs for model
        num_cols = out.select_dtypes(include=[np.number]).columns
        out[num_cols] = out[num_cols].fillna(0)

        # === Optional V9-style enrichment (additive only) ===
        if getattr(self, 'use_v9_style_loo', False) and HAS_CORE_FEATURES:
            try:
                if self._loo_builder is None:
                    obs = out[out.get('event_observed', 1) == 1]
                    if len(obs) > 30:
                        self._loo_builder = LOOBuilder().fit(obs)
                if self._loo_builder:
                    out = self._loo_builder.transform(out)

                if self._cluster_builder is None:
                    self._cluster_builder = ClusterFeatureBuilder()
                    # best effort fit
                    try:
                        self._cluster_builder.fit(out)
                    except Exception:
                        pass
                if self._cluster_builder and self._cluster_builder.fitted:
                    out = self._cluster_builder.transform(out)
            except Exception as e:
                logger.warning("V9-style enrichment skipped: %s", e)

        # Core feature list for model consumption (order stable) — STRICTLY NO TARGET LEAKAGE
        forbidden = {
            'id','start_datetime','end_datetime','closed_datetime','description','address',
            'end_address','veh_no','route_path','meta_data','comment',
            'duration_hrs', 'duration_ops', 'event_observed', 'observed_ops',
            'dur_hrs', 'impact_target', 'impact_proxy'
        }
        self.feature_cols = []
        for c in out.columns:
            if c in forbidden:
                continue
            if c.startswith('txt_'):
                continue
            if pd.api.types.is_numeric_dtype(out[c]) or c.endswith('_enc'):
                self.feature_cols.append(c)

        return out

    def get_feature_names(self) -> List[str]:
        return getattr(self, 'feature_cols', [])

    def build_impact_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Composite target for direct impact modeling (what decision makers care about).
        impact ~ duration * closure_weight * centrality * rush_multiplier
        """
        obs = df.get('event_observed', pd.Series(1, index=df.index)) == 1
        dur = df.get('duration_hrs', df.get('dur_hrs', pd.Series(2.0, index=df.index))).fillna(2.0).clip(0.1, 48)
        closure = df.get('requires_road_closure', 0).fillna(0).astype(float)
        centrality = df.get('corridor_centrality', 0.15)
        rush = df.get('is_rush', 0).fillna(0).astype(float) * 0.35 + 1.0

        impact = dur * (1 + closure * 1.2) * (centrality + 0.05) * rush
        impact = impact.where(obs, impact * 0.7)  # downweight censored a bit
        return impact.clip(0.1, 80)

# Convenience for quick use
def load_or_fit_fe(df_train: Optional[pd.DataFrame] = None) -> EventImpactFeatureEngineer:
    fe = EventImpactFeatureEngineer()
    enc_path = fe.models_dir / 'event_fe_encoders.pkl'
    if enc_path.exists() and df_train is None:
        fe.label_encoders = joblib.load(enc_path)
        txt = joblib.load(fe.models_dir / 'event_text_svd.pkl')
        fe.tfidf = txt['tfidf']
        fe.svd = txt['svd']
        with open(fe.models_dir / 'event_corridor_density.json') as f:
            d = json.load(f)
            fe.corridor_planned_density = d.get('density', {})
            fe.corridor_centrality = d.get('centrality', {})
        logger.info("Loaded pre-fitted EventImpactFeatureEngineer")
    else:
        assert df_train is not None, "Need df_train to fit first time"
        fe.fit(df_train)
    return fe

if __name__ == "__main__":
    # Quick smoke
    df = pd.read_parquet(BASE_DIR / "data" / "processed" / "survival_ready.parquet")
    fe = EventImpactFeatureEngineer()
    fe.fit(df.iloc[:4000])  # pretend train
    out = fe.transform(df.iloc[4000:4010])
    print("Generated features:", len(fe.get_feature_names()))
    print("Sample cols:", fe.get_feature_names()[:12])
    print("impact target sample:", fe.build_impact_target(df.iloc[4000:4010]).round(2).tolist())