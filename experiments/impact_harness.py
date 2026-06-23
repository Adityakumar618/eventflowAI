"""
Lightweight Experiment Harness for Impact / Event-Driven Work
=============================================================
Safe, additive only. Used for new impact experiments (v2+).

Provides:
- Purged temporal splits
- Regime-aware evaluation
- Consistent artifact saving + logging
- Reusable baseline comparison helpers

Does NOT touch production V9 inference or models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Any
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, roc_auc_score
import json
import time
import joblib
import logging

logger = logging.getLogger(__name__)
RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def purged_temporal_splits(df: pd.DataFrame, n_splits: int = 5, purge_days: int = 2) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    TimeSeriesSplit + simple purge.
    df must be sorted by start_datetime.
    """
    df = df.sort_values("start_datetime").reset_index(drop=True)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = []
    for train_idx, test_idx in tscv.split(df):
        if purge_days > 0 and len(train_idx) > 0:
            train_end_time = df.iloc[train_idx[-1]]["start_datetime"]
            purge_cutoff = train_end_time - pd.Timedelta(days=purge_days)
            train_idx = df[(df.index.isin(train_idx)) & (df["start_datetime"] <= purge_cutoff)].index.values
        splits.append((train_idx, test_idx))
    return splits


def evaluate_impact(y_true: np.ndarray, y_pred: np.ndarray, regime: pd.Series = None) -> Dict[str, float]:
    res = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "n": len(y_true)
    }
    if regime is not None:
        for r in pd.unique(regime):
            mask = (regime == r)
            if mask.sum() > 5:
                res[f"mae_{r}"] = float(mean_absolute_error(y_true[mask], y_pred[mask]))
    return res


def log_experiment(run_name: str, metrics: Dict, params: Dict, feature_list: List[str],
                   model_artifacts: Dict[str, Any] = None) -> Path:
    ts = int(time.time())
    run_dir = RESULTS_DIR / f"{run_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(run_dir / "params.json", "w") as f:
        json.dump(params, f, indent=2)
    with open(run_dir / "features.json", "w") as f:
        json.dump({"n": len(feature_list), "features": feature_list}, f, indent=2)

    if model_artifacts:
        for name, obj in model_artifacts.items():
            if hasattr(obj, "save"):
                obj.save(run_dir / f"{name}.pkl")
            else:
                joblib.dump(obj, run_dir / f"{name}.pkl")

    logger.info(f"Experiment {run_name} logged to {run_dir}")
    return run_dir


class ImpactExperimentRunner:
    def __init__(self, name: str):
        self.name = name
        self.results = []

    def run_cv(self, df: pd.DataFrame, feature_fn: Callable, model_fn: Callable,
               target_col: str, regime_col: str = "event_type", n_splits: int = 4):
        splits = purged_temporal_splits(df, n_splits)
        all_metrics = []

        for i, (tr, te) in enumerate(splits):
            X_train = feature_fn(df.iloc[tr])
            X_test = feature_fn(df.iloc[te])
            y_train = df.iloc[tr][target_col].values
            y_test = df.iloc[te][target_col].values
            regime_test = df.iloc[te][regime_col] if regime_col in df else None

            model = model_fn()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            m = evaluate_impact(y_test, preds, regime_test)
            m["fold"] = i
            all_metrics.append(m)
            logger.info(f"Fold {i}: {m}")

        avg = {k: float(np.mean([d[k] for d in all_metrics if k in d])) for k in all_metrics[0] if k != "fold"}
        self.results.append({"name": self.name, "avg": avg, "folds": all_metrics})
        return avg


if __name__ == "__main__":
    print("impact_harness.py loaded. Ready for clean experiments.")