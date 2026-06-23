"""
Master precompute script — run this ONCE to generate all cached artefacts.
Usage:  python src/precompute.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

from src.data_pipeline      import ASTraMCleaner, ASTraMFeatureEngine
from src.survival_model     import SurvivalEnsemble
from src.closure_predictor  import RoadClosurePredictor
from src.cascade_detector   import CascadeDetector
from src.hotspot_forecaster import SpatioTemporalForecaster
from src.response_analyzer  import ResponseEfficiencyAnalyzer, TrendIntelligenceEngine


def run_all():
    # ── 1. Data Pipeline ────────────────────────────────────────────────────
    logging.info("=" * 60)
    logging.info("STEP 1: Data Pipeline")
    logging.info("=" * 60)
    cleaner = ASTraMCleaner("data/raw/astram_events.csv", "data/processed")
    df_raw  = cleaner.run()
    engine  = ASTraMFeatureEngine("data/processed")
    df      = engine.fit_transform(df_raw)

    # ── 2. Survival Ensemble ────────────────────────────────────────────────
    logging.info("=" * 60)
    logging.info("STEP 2: Survival Ensemble")
    logging.info("=" * 60)
    survival = SurvivalEnsemble("models")
    survival.fit(df)

    # ── 3. Road Closure Predictor ───────────────────────────────────────────
    logging.info("=" * 60)
    logging.info("STEP 3: Road Closure Predictor")
    logging.info("=" * 60)
    closure_pred = RoadClosurePredictor("models")
    auc = closure_pred.fit(df)
    logging.info(f"Closure predictor AUC: {auc:.4f}")

    # ── 4. Cascade Detector ─────────────────────────────────────────────────
    logging.info("=" * 60)
    logging.info("STEP 4: Cascade Detector")
    logging.info("=" * 60)
    cascade = CascadeDetector("data/precomputed")
    cascade_result = cascade.detect_historical_cascades(df)
    for cause, stats in sorted(cascade_result.items(), key=lambda x: -x[1]['probability'])[:5]:
        logging.info(f"  {cause:25s}  cascade_prob={stats['probability']:.3f}")

    # ── 5. Hotspot Forecaster ───────────────────────────────────────────────
    logging.info("=" * 60)
    logging.info("STEP 5: DBSCAN Hotspot Forecaster")
    logging.info("=" * 60)
    forecaster = SpatioTemporalForecaster("models", "data/precomputed")
    forecaster.run_full_pipeline(df)
    top_risks = forecaster.predict_risk(18)
    logging.info(f"Top risk at 18:00: {top_risks[0]['corridor'] if top_risks else 'none'}")

    # ── 6. Response Efficiency ──────────────────────────────────────────────
    logging.info("=" * 60)
    logging.info("STEP 6: Response Efficiency + Trend Intelligence")
    logging.info("=" * 60)
    resp  = ResponseEfficiencyAnalyzer()
    stats = resp.compute_station_scores(df)
    logging.info(f"Station efficiency computed for {len(stats)} stations.")

    trend    = TrendIntelligenceEngine()
    trend_r  = trend.compute_trends(df)
    logging.info(trend_r['frequency_trend']['insight'])
    logging.info(trend_r['duration_trend']['insight'])

    logging.info("=" * 60)
    logging.info("✅  ALL PRECOMPUTE STEPS COMPLETE")
    logging.info("=" * 60)


if __name__ == "__main__":
    run_all()
