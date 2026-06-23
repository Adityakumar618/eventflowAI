"""
Optuna Hyperparameter Search — LightGBM P50 Model
==================================================
Goal: minimize P50 MAE on temporal CV (3 folds)
Budget: 60 trials, ~5 minutes
Expected gain: 10-20% MAE reduction from best defaults
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import json
import joblib
import optuna
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

OUT_DIR = Path("experiments/results")

print("=" * 60)
print("OPTUNA HYPERPARAMETER SEARCH — LightGBM P50")
print("60 trials, 3-fold temporal CV, minimize P50 MAE")
print("=" * 60)

# ── Build dataset (same as previous experiments) ─────────────────────────────
raw = pd.read_csv("data/raw/astram_events.csv")
for col in ["start_datetime","closed_datetime","resolved_datetime"]:
    raw[col] = pd.to_datetime(raw[col], errors="coerce")

raw["event_observed"] = raw["closed_datetime"].notna().astype(int)
max_date = raw["start_datetime"].max()

def compute_dur(row):
    if pd.notna(row["closed_datetime"]):
        return max((row["closed_datetime"] - row["start_datetime"]).total_seconds()/3600, 0.05)
    elif pd.notna(row["resolved_datetime"]):
        return max((row["resolved_datetime"] - row["start_datetime"]).total_seconds()/3600, 0.05)
    return max((max_date - row["start_datetime"]).total_seconds()/3600, 0.05)

raw["duration_hrs"] = raw.apply(compute_dur, axis=1)

for col in ["event_cause","zone","corridor","police_station"]:
    le = LabelEncoder()
    raw[col+"_enc"] = le.fit_transform(raw[col].fillna("unknown").astype(str))

raw["hour"]       = raw["start_datetime"].dt.hour
raw["dow"]        = raw["start_datetime"].dt.dayofweek
raw["month"]      = raw["start_datetime"].dt.month
raw["is_weekend"] = raw["dow"].isin([5,6]).astype(int)
raw["is_rush"]    = raw["hour"].apply(lambda h: 1 if (8<=h<=11) or (17<=h<=20) else 0)
raw["is_night"]   = ((raw["hour"]>=22)|(raw["hour"]<=5)).astype(int)
raw["hour_sin"]   = np.sin(raw["hour"] * 2 * np.pi / 24)
raw["hour_cos"]   = np.cos(raw["hour"] * 2 * np.pi / 24)
raw["requires_road_closure"] = raw["requires_road_closure"].fillna(0).astype(int)
raw["is_weather"] = raw["event_cause"].isin(
    ["water_logging","tree_fall","fog/low_visibility","debris"]).astype(int)

obs = raw[raw["event_observed"]==1]
for col_name, agg_fn in [
    ("cause_mean",   "mean"),
    ("cause_median", "median"),
    ("cause_p90",    lambda x: x.quantile(0.9)),
    ("cause_p10",    lambda x: x.quantile(0.1)),
]:
    s = obs.groupby("event_cause")["duration_hrs"].agg(**{col_name: agg_fn}).reset_index()
    raw = raw.merge(s, on="event_cause", how="left")

raw = raw.merge(obs.groupby("police_station")["duration_hrs"].agg(
    station_mean="mean").reset_index(), on="police_station", how="left")
raw = raw.merge(obs.groupby("corridor")["duration_hrs"].agg(
    corridor_mean="mean", corridor_cnt="count").reset_index(), on="corridor", how="left")

gm = obs["duration_hrs"].median()
for col in ["cause_mean","cause_median","cause_p90","cause_p10",
            "station_mean","corridor_mean","corridor_cnt"]:
    raw[col] = raw[col].fillna(gm)

raw["cause_x_rush"]    = raw["event_cause_enc"] * raw["is_rush"]
raw["cause_x_night"]   = raw["event_cause_enc"] * raw["is_night"]
raw["cause_x_zone"]    = raw["event_cause_enc"] * raw["zone_enc"]
raw["cause_x_closure"] = raw["event_cause_enc"] * raw["requires_road_closure"]
raw["station_x_cause"] = raw["police_station_enc"] * raw["event_cause_enc"]

FEATURES = [
    "hour","dow","month","is_weekend","is_rush","is_night","hour_sin","hour_cos",
    "is_weather","requires_road_closure",
    "event_cause_enc","zone_enc","corridor_enc","police_station_enc",
    "cause_mean","cause_median","cause_p90","cause_p10","station_mean",
    "corridor_mean","corridor_cnt",
    "cause_x_rush","cause_x_night","cause_x_zone","cause_x_closure","station_x_cause",
]
for col in FEATURES:
    raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0)

raw = raw.sort_values("start_datetime").reset_index(drop=True)
raw["start_dt_naive"] = raw["start_datetime"].dt.tz_localize(None)

# ── Folds ─────────────────────────────────────────────────────────────────────
folds = [
    {"train_end": pd.Timestamp("2024-01-31"),
     "test_start": pd.Timestamp("2024-02-01"), "test_end": pd.Timestamp("2024-02-29")},
    {"train_end": pd.Timestamp("2024-02-29"),
     "test_start": pd.Timestamp("2024-03-01"), "test_end": pd.Timestamp("2024-03-31")},
    {"train_end": pd.Timestamp("2024-03-31"),
     "test_start": pd.Timestamp("2024-04-01"), "test_end": pd.Timestamp("2024-04-30")},
]
TRAIN_MASK = (raw["event_observed"]==1) & (raw["duration_hrs"]<=48)

for f in folds:
    f["tr"] = raw[raw["start_dt_naive"] <= f["train_end"]].index
    f["te"] = raw[(raw["start_dt_naive"] >= f["test_start"]) &
                  (raw["start_dt_naive"] <= f["test_end"])].index

# ── Objective function ─────────────────────────────────────────────────────────
def objective(trial):
    params = {
        "objective":       "quantile",
        "alpha":           0.50,
        "n_estimators":    trial.suggest_int("n_estimators", 200, 800),
        "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves":      trial.suggest_int("num_leaves", 15, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
        "reg_alpha":       trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        "reg_lambda":      trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state":    42,
        "verbosity":       -1,
    }

    fold_maes = []
    for fold in folds:
        tr = raw.loc[fold["tr"]]
        te = raw.loc[fold["te"]]
        tr_obs = tr[TRAIN_MASK.loc[fold["tr"]]]
        te_obs = te[(te["event_observed"]==1) & (te["duration_hrs"]<=48)]

        if len(tr_obs) < 20 or len(te_obs) < 5:
            continue

        model = lgb.LGBMRegressor(**params)
        model.fit(tr_obs[FEATURES], tr_obs["duration_hrs"])
        y_pred = np.maximum(model.predict(te_obs[FEATURES]), 0.05)
        mae = mean_absolute_error(te_obs["duration_hrs"].values, y_pred)
        fold_maes.append(mae)

    return np.mean(fold_maes) if fold_maes else 999.0

# ── Run Optuna ────────────────────────────────────────────────────────────────
print("\nRunning 60 trials...")
study = optuna.create_study(direction="minimize",
                             sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=60, show_progress_bar=False)

best = study.best_params
best_mae = study.best_value

print(f"\nBest MAE: {best_mae:.4f}h")
print(f"Best params:")
for k, v in best.items():
    print(f"  {k}: {v}")

# ── Baseline comparison ───────────────────────────────────────────────────────
print("\n[BASELINE] Default LGB params:")
baseline_maes = []
for fold in folds:
    tr = raw.loc[fold["tr"]]
    te = raw.loc[fold["te"]]
    tr_obs = tr[TRAIN_MASK.loc[fold["tr"]]]
    te_obs = te[(te["event_observed"]==1) & (te["duration_hrs"]<=48)]
    if len(tr_obs) < 20 or len(te_obs) < 5:
        continue
    m = lgb.LGBMRegressor(objective="quantile", alpha=0.5,
                           n_estimators=500, learning_rate=0.03, num_leaves=63,
                           random_state=42, verbosity=-1)
    m.fit(tr_obs[FEATURES], tr_obs["duration_hrs"])
    y_pred = np.maximum(m.predict(te_obs[FEATURES]), 0.05)
    baseline_maes.append(mean_absolute_error(te_obs["duration_hrs"].values, y_pred))

baseline_mae = np.mean(baseline_maes)
improvement  = (baseline_mae - best_mae) / baseline_mae * 100
print(f"  Baseline MAE: {baseline_mae:.4f}h")
print(f"  Tuned MAE:    {best_mae:.4f}h")
print(f"  Improvement:  {improvement:.1f}%")

# ── Train final model with best params ────────────────────────────────────────
print("\n[FINAL MODEL] Training on all data with best params...")
all_obs = raw[TRAIN_MASK].copy()
final_params = {
    "objective": "quantile", "alpha": 0.50,
    "random_state": 42, "verbosity": -1,
    **best
}
final_model = lgb.LGBMRegressor(**final_params)
final_model.fit(all_obs[FEATURES], all_obs["duration_hrs"])

joblib.dump(final_model, "models/lgb_q50_tuned.pkl")
print("Saved tuned model -> models/lgb_q50_tuned.pkl")

# ── Save Optuna results ────────────────────────────────────────────────────────
optuna_results = {
    "baseline_mae": round(baseline_mae, 4),
    "tuned_mae":    round(best_mae, 4),
    "improvement_pct": round(improvement, 1),
    "best_params":  best,
    "n_trials":     60,
    "verdict": "USE_TUNED" if improvement > 5 else "DEFAULT_IS_FINE",
}
with open(OUT_DIR / "optuna_results.json", "w") as f:
    json.dump(optuna_results, f, indent=2)
print(f"Saved Optuna results -> {OUT_DIR / 'optuna_results.json'}")

print("\n" + "="*60)
print("OPTUNA SEARCH COMPLETE")
print(f"  Improvement: {improvement:.1f}%")
print(f"  Verdict: {'USE TUNED MODEL' if improvement > 5 else 'DEFAULT IS ALREADY NEAR-OPTIMAL'}")
print("="*60)
