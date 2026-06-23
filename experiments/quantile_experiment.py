"""
EventFlow AI - Quantile Regression Experiment
==============================================
Train 3 LightGBM quantile models (P10, P50, P90).
Validate with: pinball loss, coverage rate, interval width.

Innovation: Prediction INTERVALS, not point estimates.
- P10 = best case (90% chance event clears BEFORE this)
- P50 = median (most likely duration)
- P90 = worst case (10% chance event EXCEEDS this)

No competing team will have uncertainty-aware predictions.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

OUT_DIR = Path("experiments/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("QUANTILE REGRESSION EXPERIMENT")
print("P10 / P50 / P90 prediction intervals for event duration")
print("=" * 65)

# ── Load data ────────────────────────────────────────────────────────────────
raw = pd.read_csv("data/raw/astram_events.csv")
dt_cols = ["start_datetime","closed_datetime","resolved_datetime"]
for col in dt_cols:
    raw[col] = pd.to_datetime(raw[col], errors="coerce")

# Correct target: only closed_datetime = truly observed
raw["event_observed"] = raw["closed_datetime"].notna().astype(int)
max_date = raw["start_datetime"].max()

def compute_dur(row):
    if pd.notna(row["closed_datetime"]):
        return max((row["closed_datetime"] - row["start_datetime"]).total_seconds()/3600, 0.05)
    elif pd.notna(row["resolved_datetime"]):
        return max((row["resolved_datetime"] - row["start_datetime"]).total_seconds()/3600, 0.05)
    return max((max_date - row["start_datetime"]).total_seconds()/3600, 0.05)

raw["duration_hrs"] = raw.apply(compute_dur, axis=1)

# ── Feature engineering ───────────────────────────────────────────────────────
from sklearn.preprocessing import LabelEncoder

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
weather_causes = ["water_logging","tree_fall","fog/low_visibility","debris"]
raw["is_weather"] = raw["event_cause"].isin(weather_causes).astype(int)

# Historical stats from observed events only
obs = raw[raw["event_observed"]==1]
cause_stats = obs.groupby("event_cause")["duration_hrs"].agg(
    cause_mean="mean", cause_median="median",
    cause_p90=lambda x: x.quantile(0.9),
    cause_p10=lambda x: x.quantile(0.1),
).reset_index()
raw = raw.merge(cause_stats, on="event_cause", how="left")

station_stats = obs.groupby("police_station")["duration_hrs"].agg(
    station_mean="mean").reset_index()
raw = raw.merge(station_stats, on="police_station", how="left")

corridor_stats = obs.groupby("corridor")["duration_hrs"].agg(
    corridor_mean="mean", corridor_cnt="count").reset_index()
raw = raw.merge(corridor_stats, on="corridor", how="left")

for col in ["cause_mean","cause_median","cause_p90","cause_p10",
            "station_mean","corridor_mean","corridor_cnt"]:
    raw[col] = raw[col].fillna(obs["duration_hrs"].median())

# Interaction features
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
    raw[col] = raw[col].fillna(0)

raw = raw.sort_values("start_datetime").reset_index(drop=True)
raw["start_dt_naive"] = raw["start_datetime"].dt.tz_localize(None)

# Training set: observed AND duration <=48h
TRAIN_MASK = (raw["event_observed"]==1) & (raw["duration_hrs"]<=48)
print(f"\nTraining samples: {TRAIN_MASK.sum()} (observed + <=48h)")
print(f"Test (all observed): {raw['event_observed'].sum()}")

# ── Temporal CV folds ────────────────────────────────────────────────────────
folds = [
    {"name":"Fold1", "train_end":pd.Timestamp("2024-01-31"),
     "test_start":pd.Timestamp("2024-02-01"), "test_end":pd.Timestamp("2024-02-29")},
    {"name":"Fold2", "train_end":pd.Timestamp("2024-02-29"),
     "test_start":pd.Timestamp("2024-03-01"), "test_end":pd.Timestamp("2024-03-31")},
    {"name":"Fold3", "train_end":pd.Timestamp("2024-03-31"),
     "test_start":pd.Timestamp("2024-04-01"), "test_end":pd.Timestamp("2024-04-30")},
]
for f in folds:
    f["tr"] = raw[raw["start_dt_naive"] <= f["train_end"]].index
    f["te"] = raw[(raw["start_dt_naive"] >= f["test_start"]) &
                  (raw["start_dt_naive"] <= f["test_end"])].index

# ── Quantile loss (pinball) ───────────────────────────────────────────────────
def pinball_loss(y_true, y_pred, alpha):
    e = y_true - y_pred
    return np.mean(np.where(e >= 0, alpha * e, (alpha - 1) * e))

# ── Coverage rate: what % of actual fall within [P10, P90]? ─────────────────
def coverage_rate(y_true, p10, p90):
    return np.mean((y_true >= p10) & (y_true <= p90))

# ── Mean interval width ───────────────────────────────────────────────────────
def interval_width(p10, p90):
    return np.mean(p90 - p10)

# ── Train quantile models per fold ───────────────────────────────────────────
print("\n[TRAINING] 3 quantile models x 3 folds = 9 models")
print("-" * 65)

QUANTILES = [0.10, 0.50, 0.90]
QNAMES    = ["P10 (best case)", "P50 (median)", "P90 (worst case)"]

all_fold_results = []
best_models = {0.10: None, 0.50: None, 0.90: None}

for fold in folds:
    tr_idx = fold["tr"]
    te_idx = fold["te"]

    tr = raw.loc[tr_idx]
    te = raw.loc[te_idx]
    tr_obs = tr[TRAIN_MASK.loc[tr_idx]]

    X_train = tr_obs[FEATURES]
    y_train = tr_obs["duration_hrs"]

    # Test set: only observed events with actual duration
    te_obs  = te[(te["event_observed"]==1) & (te["duration_hrs"]<=48)]
    X_test  = te_obs[FEATURES]
    y_test  = te_obs["duration_hrs"]

    if len(tr_obs) < 20 or len(te_obs) < 5:
        print(f"  {fold['name']}: Too few samples, skipping")
        continue

    fold_preds = {}
    for alpha, qname in zip(QUANTILES, QNAMES):
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=63,
            min_child_samples=20,
            colsample_bytree=0.8,
            subsample=0.8,
            random_state=42,
            verbosity=-1,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred = np.maximum(y_pred, 0.05)
        fold_preds[alpha] = y_pred

        pb = pinball_loss(y_test.values, y_pred, alpha)
        mae = mean_absolute_error(y_test.values, y_pred)

        if fold["name"] == "Fold3":  # Keep last fold models as best
            best_models[alpha] = model

        print(f"  {fold['name']} | {qname:<20} | "
              f"Pinball={pb:.3f}  MAE={mae:.2f}h  n_test={len(y_test)}")

    # Coverage and width for this fold
    if 0.10 in fold_preds and 0.90 in fold_preds:
        cov = coverage_rate(y_test.values, fold_preds[0.10], fold_preds[0.90])
        width = interval_width(fold_preds[0.10], fold_preds[0.90])
        print(f"  {fold['name']} | INTERVAL STATS | "
              f"Coverage={cov*100:.1f}% (target=80%)  Width={width:.2f}h")
        all_fold_results.append({
            "fold": fold["name"],
            "coverage": cov,
            "width": width,
            "n_test": len(y_test),
        })
    print()

# ── Summary ───────────────────────────────────────────────────────────────────
print("=" * 65)
print("QUANTILE MODEL SUMMARY")
print("=" * 65)

if all_fold_results:
    avg_cov   = np.mean([r["coverage"] for r in all_fold_results])
    avg_width = np.mean([r["width"] for r in all_fold_results])
    print(f"\nAverage coverage: {avg_cov*100:.1f}%  (ideal: 80% for P10-P90 interval)")
    print(f"Average width:    {avg_width:.2f}h")
    print()
    if avg_cov >= 0.75:
        print("  [EXCELLENT] Coverage >= 75%: intervals are reliable")
    elif avg_cov >= 0.60:
        print("  [GOOD] Coverage >= 60%: intervals are useful")
    else:
        print("  [UNDERCOVER] Coverage < 60%: intervals too narrow")

# ── Per-cause calibration of quantile model ───────────────────────────────────
print("\n[CALIBRATION] Per-cause: Does P90 actually bound 90% of events?")
if best_models[0.50] and best_models[0.90]:
    obs_48 = raw[(raw["event_observed"]==1) & (raw["duration_hrs"]<=48)].copy()
    obs_48["pred_p50"] = np.maximum(best_models[0.50].predict(obs_48[FEATURES].fillna(0)), 0.05)
    obs_48["pred_p10"] = np.maximum(best_models[0.10].predict(obs_48[FEATURES].fillna(0)), 0.05)
    obs_48["pred_p90"] = np.maximum(best_models[0.90].predict(obs_48[FEATURES].fillna(0)), 0.05)

    print(f"\n  {'Cause':<22} {'Actual Med':>11} {'P50 Pred':>10} {'P10':>8} "
          f"{'P90':>8} {'Coverage':>10} {'N':>5}")
    print("  " + "-" * 78)

    per_cause_results = []
    for cause, grp in obs_48.groupby("event_cause"):
        if len(grp) < 5:
            continue
        act_med = grp["duration_hrs"].median()
        cov = coverage_rate(grp["duration_hrs"].values,
                             grp["pred_p10"].values, grp["pred_p90"].values)
        p50_med = grp["pred_p50"].median()
        p10_med = grp["pred_p10"].median()
        p90_med = grp["pred_p90"].median()
        status = "[OK]" if cov >= 0.75 else "[WIDE]" if cov >= 0.95 else "[NARROW]"
        print(f"  {cause:<22} {act_med:>9.2f}h {p50_med:>8.2f}h "
              f"{p10_med:>6.2f}h {p90_med:>6.2f}h {cov*100:>9.1f}%  "
              f"{len(grp):>4}  {status}")
        per_cause_results.append({"cause": cause, "coverage": cov,
                                    "p50_mae": abs(p50_med - act_med)})

    # Sample output for dashboard demo
    print("\n[DEMO OUTPUT] What the dashboard will show for a new event:")
    demo_event = obs_48.iloc[0]
    row = demo_event[FEATURES].fillna(0).to_frame().T
    p10_pred = float(np.maximum(best_models[0.10].predict(row)[0], 0.05))
    p50_pred = float(np.maximum(best_models[0.50].predict(row)[0], 0.05))
    p90_pred = float(np.maximum(best_models[0.90].predict(row)[0], 0.05))
    actual   = float(demo_event["duration_hrs"])
    cause    = demo_event["event_cause"]

    print(f"\n  Event cause: {cause}")
    print(f"  Best case  (P10): {p10_pred:.2f}h")
    print(f"  Most likely (P50): {p50_pred:.2f}h  <-- BTP deployment target")
    print(f"  Worst case (P90): {p90_pred:.2f}h  <-- officers stay until this")
    print(f"  Actual:           {actual:.2f}h")
    print(f"  Interval width:   {p90_pred - p10_pred:.2f}h")

# ── Save models if good ───────────────────────────────────────────────────────
import joblib
if all(v is not None for v in best_models.values()):
    Path("models").mkdir(exist_ok=True)
    joblib.dump(best_models[0.10], "models/lgb_q10.pkl")
    joblib.dump(best_models[0.50], "models/lgb_q50.pkl")
    joblib.dump(best_models[0.90], "models/lgb_q90.pkl")
    joblib.dump(FEATURES,           "models/q_features.pkl")

    summary = {
        "avg_coverage": round(avg_cov, 3) if all_fold_results else None,
        "avg_width_hrs": round(avg_width, 3) if all_fold_results else None,
        "features": FEATURES,
        "verdict": "USE_QUANTILE" if (all_fold_results and avg_cov >= 0.60) else "NEEDS_TUNING"
    }
    with open(OUT_DIR / "quantile_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[SAVED] 3 quantile models -> models/lgb_q10.pkl, lgb_q50.pkl, lgb_q90.pkl")
    print(f"[SAVED] Summary -> {OUT_DIR / 'quantile_summary.json'}")

print("\n[OK] QUANTILE EXPERIMENT DONE")
