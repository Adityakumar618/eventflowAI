import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import logging
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class RoadClosurePredictor:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.model = None

    FEATURES = [
        'hour', 'day_of_week', 'month', 'is_weekend', 'is_rush_hour',
        'hour_sin', 'hour_cos', 'zone_encoded', 'corridor_encoded',
        'police_station_encoded', 'event_cause_encoded', 'is_weather_related',
    ]

    def fit(self, df: pd.DataFrame):
        logging.info("Fitting Road Closure Predictor...")
        
        # Use requires_road_closure as target — confirmed safe from leakage audit
        target = 'requires_road_closure'
        df_model = df.dropna(subset=[target] + self.FEATURES).copy()
        df_model[target] = df_model[target].astype(int)

        X = df_model[self.FEATURES]
        y = df_model[target]

        base = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05,
            num_leaves=31, random_state=42, verbosity=-1
        )
        # Isotonic calibration for reliable probabilities
        self.model = CalibratedClassifierCV(base, cv=3, method='isotonic')
        self.model.fit(X, y)

        # Quick AUC
        auc = cross_val_score(self.model, X, y, cv=3, scoring='roc_auc').mean()
        logging.info(f"Road Closure Predictor AUC (3-fold CV): {auc:.4f}")

        joblib.dump(self.model, self.models_dir / 'lgb_closure.pkl')
        logging.info("Road Closure model saved.")
        return auc

    def predict_proba_with_explanation(self, event: dict) -> dict:
        if self.model is None:
            self.model = joblib.load(self.models_dir / 'lgb_closure.pkl')

        row = pd.DataFrame([{f: event.get(f, 0) for f in self.FEATURES}])
        prob = float(self.model.predict_proba(row)[0][1])

        # Human-readable factors
        factors = []
        if event.get('is_weather_related', 0):
            factors.append("Weather-related events frequently require closures")
        if event.get('hour', 12) in range(8, 11) or event.get('hour', 12) in range(17, 21):
            factors.append("Peak hour increases closure likelihood")
        if event.get('requires_road_closure', 0):
            factors.append("Operator marked closure required at intake")

        return {
            'probability': round(prob, 3),
            'recommendation': 'CLOSE ROAD' if prob > 0.5 else 'MONITOR',
            'confidence': 'HIGH' if abs(prob - 0.5) > 0.3 else 'MEDIUM',
            'factors': factors,
            'display': f"{int(prob*100)}% probability of road closure required"
        }


if __name__ == "__main__":
    df = pd.read_parquet("data/processed/survival_ready.parquet")
    predictor = RoadClosurePredictor()
    predictor.fit(df)
