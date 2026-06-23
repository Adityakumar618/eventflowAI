"""
Advanced Feature Engineering Experiment
========================================
4 feature innovations stacked on top of the best baseline (MAE 3.22h):

1. TEMPORAL CAUSE STATS    — rolling 30-day mean per cause (drift-aware)
2. CONCURRENT EVENT LOAD   — # active events same zone at event start (cascade proxy)
3. LOO TARGET ENCODING     — police_station encoded as leave-one-out mean duration
4. RESIDUAL CORRECTION     — second LGB trained on first LGB's signed residuals

Each feature group added incrementally. We compare against:
  Baseline: Optuna-tuned LGB, global cause stats, raw label encoding
  Target:   Beat 3.22h MAE
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import json
import joblib
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

OUT_DIR = Path("experiments/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("ADVANCED FEATURE ENGINEERING EXPERIMENT")
print("Pushing beyond MAE 3.22h")
print("=" * 65)

# ── Load and prepare base data ────────────────────────────────────────────────
raw = pd.read_csv("data/raw/astram_events.csv")
for col in ["start_datetime", "closed_datetime", "resolved_datetime"]:
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
raw["requires_road_closure"] = raw["requires_road_closure"].fillna(0).astype(int)

for col in ["event_cause", "zone", "corridor", "police_station"]:
    le = LabelEncoder()
    raw[col+"_enc"] = le.fit_transform(raw[col].fillna("unknown").astype(str))

raw["hour"]      = raw["start_datetime"].dt.hour
raw["dow"]       = raw["start_datetime"].dt.dayofweek
raw["month"]     = raw["start_datetime"].dt.month
raw["is_weekend"]= raw["dow"].isin([5,6]).astype(int)
raw["is_rush"]   = raw["hour"].apply(lambda h: 1 if (8<=h<=11) or (17<=h<=20) else 0)
raw["is_night"]  = ((raw["hour"]>=22)|(raw["hour"]<=5)).astype(int)
raw["hour_sin"]  = np.sin(raw["hour"] * 2 * np.pi / 24)
raw["hour_cos"]  = np.cos(raw["hour"] * 2 * np.pi / 24)
raw["is_weather"]= raw["event_cause"].isin(
    ["water_logging","tree_fall","fog/low_visibility","debris"]).astype(int)

raw = raw.sort_values("start_datetime").reset_index(drop=True)
raw["start_dt_naive"] = raw["start_datetime"].dt.tz_localize(None)
raw["start_ts"]  = raw["start_dt_naive"].astype(np.int64) // 10**9

obs = raw[raw["event_observed"]==1]

# Global stats (baseline features)
for col_name, fn in [("cause_mean","mean"), ("cause_median","median"),
                      ("cause_p90", lambda x: x.quantile(0.9)),
                      ("cause_p10", lambda x: x.quantile(0.1))]:
    s = obs.groupby("event_cause")["duration_hrs"].agg(**{col_name: fn}).reset_index()
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

# ── FEATURE 1: Rolling 30-day cause mean (drift-aware) ───────────────────────
print("\n[FEATURE 1] Rolling 30-day cause mean (drift-aware)")
print("  Standard global mean ignores temporal drift.")
print("  Monsoon VB takes 2x longer than dry season VB.")

rolling_cause_means = []
WINDOW_DAYS = 30
WINDOW_SEC  = WINDOW_DAYS * 86400

for i, row in raw.iterrows():
    t  = row["start_ts"]
    ec = row["event_cause"]
    # Look back 30 days, only observed events
    past = raw.iloc[max(0, i-500):i]
    mask = (
        (past["event_cause"] == ec) &
        (past["event_observed"] == 1) &
        (past["start_ts"] >= t - WINDOW_SEC) &
        (past["start_ts"] <  t)
    )
    matched = past[mask]
    if len(matched) >= 3:
        rolling_cause_means.append(matched["duration_hrs"].mean())
    else:
        rolling_cause_means.append(row["cause_mean"])  # fallback to global

raw["cause_rolling_30d"] = rolling_cause_means
print(f"  Done. Corr with cause_mean: {raw['cause_rolling_30d'].corr(raw['cause_mean']):.3f}")
print(f"  Corr with duration: {raw[raw['event_observed']==1]['cause_rolling_30d'].corr(raw[raw['event_observed']==1]['duration_hrs']):.3f}")

# ── FEATURE 2: Concurrent active events in same zone ─────────────────────────
print("\n[FEATURE 2] Concurrent active events in same zone")
print("  More active events = resource competition = longer resolution")

concurrent_zone = []
for i, row in raw.iterrows():
    t = row["start_ts"]
    z = row["zone"]
    # Events that started before this one and are still (estimated) active
    # Use their start + cause_mean as estimated end time
    recent = raw.iloc[max(0, i-300):i]
    # Events that started within last 4 hours (likely still active)
    active_mask = (
        (recent["zone"] == z) &
        (recent["start_ts"] >= t - 4*3600) &
        (recent["start_ts"] < t)
    )
    concurrent_zone.append(int(active_mask.sum()))

raw["concurrent_zone_events"] = concurrent_zone

# Also: concurrent in same corridor (tighter geographic scope)
concurrent_corridor = []
for i, row in raw.iterrows():
    t = row["start_ts"]
    c = row["corridor"]
    recent = raw.iloc[max(0, i-200):i]
    active_mask = (
        (recent["corridor"] == c) &
        (recent["start_ts"] >= t - 2*3600) &
        (recent["start_ts"] < t)
    )
    concurrent_corridor.append(int(active_mask.sum()))

raw["concurrent_corridor_events"] = concurrent_corridor
print(f"  Zone concurrency range: [{raw['concurrent_zone_events'].min()}, {raw['concurrent_zone_events'].max()}]")
print(f"  Corr with duration (observed): "
      f"{raw[raw['event_observed']==1]['concurrent_zone_events'].corr(raw[raw['event_observed']==1]['duration_hrs']):.3f}")

# ── FEATURE 3: Leave-one-out (LOO) target encoding for police_station ─────────
print("\n[FEATURE 3] LOO target encoding for police_station")
print("  Raw label encoding loses ordinal info. LOO avoids leakage.")

# LOO: for each event, encode police_station as mean duration of OTHER events at same station
station_dur_sum   = obs.groupby("police_station")["duration_hrs"].sum()
station_dur_count = obs.groupby("police_station")["duration_hrs"].count()
global_mean = obs["duration_hrs"].mean()

loo_encoding = []
for _, row in raw.iterrows():
    s  = row["police_station"]
    d  = row["duration_hrs"] if row["event_observed"] == 1 else 0
    n  = station_dur_count.get(s, 0)
    sm = station_dur_sum.get(s, global_mean * n)

    if n > 1 and row["event_observed"] == 1:
        # Leave out current event
        loo_mean = (sm - d) / (n - 1)
    elif n > 0:
        loo_mean = sm / n
    else:
        loo_mean = global_mean
    loo_encoding.append(loo_mean)

raw["station_loo_encoded"] = loo_encoding
print(f"  LOO encoding range: [{raw['station_loo_encoded'].min():.2f}, {raw['station_loo_encoded'].max():.2f}]")

# ── ADVANCED FEATURE SET ──────────────────────────────────────────────────────
BASE_FEATURES = [
    "hour","dow","month","is_weekend","is_rush","is_night","hour_sin","hour_cos",
    "is_weather","requires_road_closure",
    "event_cause_enc","zone_enc","corridor_enc","police_station_enc",
    "cause_mean","cause_median","cause_p90","cause_p10","station_mean",
    "corridor_mean","corridor_cnt",
    "cause_x_rush","cause_x_night","cause_x_zone","cause_x_closure","station_x_cause",
]

ADV_FEATURES = BASE_FEATURES + [
    "cause_rolling_30d",          # Feature 1
    "concurrent_zone_events",     # Feature 2
    "concurrent_corridor_events", # Feature 2
    "station_loo_encoded",        # Feature 3
]

for col in ADV_FEATURES:
    raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0)

# ── TEMPORAL CV SETUP ─────────────────────────────────────────────────────────
folds = [
    {"name":"Feb", "train_end": pd.Timestamp("2024-01-31"),
     "test_start": pd.Timestamp("2024-02-01"), "test_end": pd.Timestamp("2024-02-29")},
    {"name":"Mar", "train_end": pd.Timestamp("2024-02-29"),
     "test_start": pd.Timestamp("2024-03-01"), "test_end": pd.Timestamp("2024-03-31")},
    {"name":"Apr", "train_end": pd.Timestamp("2024-03-31"),
     "test_start": pd.Timestamp("2024-04-01"), "test_end": pd.Timestamp("2024-04-30")},
]
TRAIN_MASK = (raw["event_observed"]==1) & (raw["duration_hrs"]<=48)
for f in folds:
    f["tr"] = raw[raw["start_dt_naive"] <= f["train_end"]].index
    f["te"] = raw[(raw["start_dt_naive"] >= f["test_start"]) &
                  (raw["start_dt_naive"] <= f["test_end"])].index

BEST_PARAMS = dict(
    objective="quantile", alpha=0.50,
    n_estimators=263, learning_rate=0.010852, num_leaves=103,
    min_child_samples=44, colsample_bytree=0.863, subsample=0.863,
    reg_alpha=0.000355, reg_lambda=0.001142,
    random_state=42, verbosity=-1,
)

def cv_mae(feature_set, label=""):
    fold_maes = []
    for f in folds:
        tr = raw.loc[f["tr"]]; te = raw.loc[f["te"]]
        tr_obs = tr[TRAIN_MASK.loc[f["tr"]]]
        te_obs = te[(te["event_observed"]==1) & (te["duration_hrs"]<=48)]
        if len(tr_obs) < 20 or len(te_obs) < 5:
            continue
        m = lgb.LGBMRegressor(**BEST_PARAMS)
        m.fit(tr_obs[feature_set], tr_obs["duration_hrs"])
        yp = np.maximum(m.predict(te_obs[feature_set]), 0.05)
        mae = mean_absolute_error(te_obs["duration_hrs"].values, yp)
        fold_maes.append(mae)
    avg = np.mean(fold_maes) if fold_maes else 999
    print(f"  {label:<40} MAE = {avg:.4f}h  folds={[round(x,2) for x in fold_maes]}")
    return avg

print("\n" + "="*65)
print("INCREMENTAL FEATURE ABLATION")
print("="*65)
print(f"\n  {'Feature Set':<40} {'Result':>12}")
print("  " + "-"*55)

mae_baseline = cv_mae(BASE_FEATURES,    "Baseline (global stats)")
mae_rolling  = cv_mae(BASE_FEATURES + ["cause_rolling_30d"],
                       "+rolling_30d cause")
mae_conc     = cv_mae(BASE_FEATURES + ["cause_rolling_30d",
                                        "concurrent_zone_events",
                                        "concurrent_corridor_events"],
                       "+concurrent zone/corridor")
mae_loo      = cv_mae(ADV_FEATURES,     "+LOO station encoding")

# ── FEATURE 4: Residual correction model ─────────────────────────────────────
print("\n[FEATURE 4] Residual correction model")
print("  Train model 1 (P50) -> compute signed residuals")
print("  Train model 2 on residuals -> subtract from P50")

def cv_mae_residual(feature_set):
    fold_maes = []
    for f in folds:
        tr = raw.loc[f["tr"]]; te = raw.loc[f["te"]]
        tr_obs = tr[TRAIN_MASK.loc[f["tr"]]]
        te_obs = te[(te["event_observed"]==1) & (te["duration_hrs"]<=48)]
        if len(tr_obs) < 30 or len(te_obs) < 5:
            continue

        # Model 1: P50 quantile
        m1 = lgb.LGBMRegressor(**BEST_PARAMS)
        m1.fit(tr_obs[feature_set], tr_obs["duration_hrs"])
        tr_pred = np.maximum(m1.predict(tr_obs[feature_set]), 0.05)
        tr_residuals = tr_obs["duration_hrs"].values - tr_pred  # signed error

        # Model 2: Residual correction (predict error to fix systematic bias)
        # Use MAE objective for residuals (robust to outliers)
        m2_params = {**BEST_PARAMS, "objective": "regression_l1", "alpha": None}
        m2_params.pop("alpha")
        m2 = lgb.LGBMRegressor(**m2_params)
        m2.fit(tr_obs[feature_set], tr_residuals)

        # Corrected prediction
        te_pred1 = np.maximum(m1.predict(te_obs[feature_set]), 0.05)
        te_correction = m2.predict(te_obs[feature_set])
        te_pred_final = np.maximum(te_pred1 + te_correction * 0.5, 0.05)  # blend

        mae = mean_absolute_error(te_obs["duration_hrs"].values, te_pred_final)
        fold_maes.append(mae)
    avg = np.mean(fold_maes) if fold_maes else 999
    print(f"  {'Residual corrected P50':<40} MAE = {avg:.4f}h  folds={[round(x,2) for x in fold_maes]}")
    return avg

mae_residual = cv_mae_residual(ADV_FEATURES)

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("IMPROVEMENT SUMMARY vs BASELINE 3.22h")
print("="*65)

results = {
    "Baseline (global stats)":          mae_baseline,
    "+rolling_30d cause":               mae_rolling,
    "+concurrent events":               mae_conc,
    "+LOO station (all adv features)":  mae_loo,
    "+Residual correction":             mae_residual,
}
best_name = min(results, key=results.get)
best_mae  = results[best_name]
total_gain = (3.22 - best_mae) / 3.22 * 100

print(f"\n{'Method':<45} {'MAE':>8} {'vs 3.22h':>10}")
print("-"*65)
for name, mae in results.items():
    delta = mae - 3.22
    flag  = " <-- BEST" if name == best_name else ""
    print(f"  {name:<43} {mae:>6.4f}h {delta:>+8.3f}h{flag}")

print(f"\n  Total improvement from 3.22h: {total_gain:.1f}%")

# Save best advanced feature set
winner_features = ADV_FEATURES if best_mae == mae_loo or best_mae == mae_residual else BASE_FEATURES
with open(OUT_DIR / "advanced_features_results.json", "w") as f:
    json.dump({
        "baseline_mae": 3.22,
        "results": {k: round(v,4) for k,v in results.items()},
        "best_method": best_name,
        "best_mae": round(best_mae, 4),
        "total_improvement_pct": round(total_gain, 1),
        "advanced_features": ADV_FEATURES,
    }, f, indent=2)

print(f"\nSaved -> experiments/results/advanced_features_results.json")
print("\n[OK] ADVANCED FEATURE EXPERIMENT DONE")
