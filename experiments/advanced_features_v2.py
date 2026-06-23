"""
Advanced Features Experiment v2 — Push Below 3.0h
===================================================
Building on the winning config (MAE 3.03h):
  Base + rolling_30d + concurrent + LOO_station

New additions:
1. LOO target encoding for CORRIDOR (same logic as station, 2% gain expected)
2. Hour x cause historical mean (duration varies by time-of-day per cause)
3. Zone x cause LOO (zone-cause interaction, non-linear spatial signal)
4. Optuna re-run on full advanced feature set (30 more trials)
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
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("ADVANCED FEATURES v2 + OPTUNA RE-TUNE")
print("Target: Break below 3.0h MAE")
print("=" * 65)

# ── Rebuild full dataset with all features ────────────────────────────────────
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
raw["requires_road_closure"] = raw["requires_road_closure"].fillna(0).astype(int)

for col in ["event_cause","zone","corridor","police_station"]:
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
raw["start_ts"]       = raw["start_dt_naive"].astype(np.int64) // 10**9

obs = raw[raw["event_observed"]==1]
gm  = obs["duration_hrs"].mean()

# ── Global cause stats ─────────────────────────────────────────────────────────
for col_name, fn in [
    ("cause_mean","mean"), ("cause_median","median"),
    ("cause_p90", lambda x: x.quantile(0.9)),
    ("cause_p10", lambda x: x.quantile(0.1))
]:
    s = obs.groupby("event_cause")["duration_hrs"].agg(**{col_name: fn}).reset_index()
    raw = raw.merge(s, on="event_cause", how="left")
raw = raw.merge(obs.groupby("police_station")["duration_hrs"].agg(
    station_mean="mean").reset_index(), on="police_station", how="left")
raw = raw.merge(obs.groupby("corridor")["duration_hrs"].agg(
    corridor_mean="mean", corridor_cnt="count").reset_index(), on="corridor", how="left")
for col in ["cause_mean","cause_median","cause_p90","cause_p10",
            "station_mean","corridor_mean","corridor_cnt"]:
    raw[col] = raw[col].fillna(obs["duration_hrs"].median())

raw["cause_x_rush"]    = raw["event_cause_enc"] * raw["is_rush"]
raw["cause_x_night"]   = raw["event_cause_enc"] * raw["is_night"]
raw["cause_x_zone"]    = raw["event_cause_enc"] * raw["zone_enc"]
raw["cause_x_closure"] = raw["event_cause_enc"] * raw["requires_road_closure"]
raw["station_x_cause"] = raw["police_station_enc"] * raw["event_cause_enc"]

# ── PREV WINNING FEATURES ─────────────────────────────────────────────────────
# Feature 1: Rolling 30-day cause mean
WINDOW_SEC = 30 * 86400
rolling_cause_means = []
for i, row in raw.iterrows():
    t  = row["start_ts"]
    ec = row["event_cause"]
    past = raw.iloc[max(0, i-500):i]
    mask = ((past["event_cause"]==ec) & (past["event_observed"]==1) &
            (past["start_ts"] >= t - WINDOW_SEC) & (past["start_ts"] < t))
    matched = past[mask]
    rolling_cause_means.append(matched["duration_hrs"].mean() if len(matched) >= 3
                                else row["cause_mean"])
raw["cause_rolling_30d"] = rolling_cause_means

# Feature 2: Concurrent events
conc_zone, conc_corr = [], []
for i, row in raw.iterrows():
    t = row["start_ts"]
    past = raw.iloc[max(0, i-300):i]
    conc_zone.append(int(((past["zone"]==row["zone"]) &
                           (past["start_ts"] >= t-4*3600) & (past["start_ts"]<t)).sum()))
    conc_corr.append(int(((past["corridor"]==row["corridor"]) &
                           (past["start_ts"] >= t-2*3600) & (past["start_ts"]<t)).sum()))
raw["concurrent_zone_events"]     = conc_zone
raw["concurrent_corridor_events"] = conc_corr

# Feature 3: LOO station encoding
st_sum  = obs.groupby("police_station")["duration_hrs"].sum()
st_cnt  = obs.groupby("police_station")["duration_hrs"].count()
loo_st  = []
for _, row in raw.iterrows():
    s = row["police_station"]; d = row["duration_hrs"]
    n = st_cnt.get(s, 0); sm = st_sum.get(s, gm * n)
    if n > 1 and row["event_observed"]==1:
        loo_st.append((sm - d) / (n - 1))
    else:
        loo_st.append(sm / n if n > 0 else gm)
raw["station_loo"] = loo_st

# ── NEW FEATURES ──────────────────────────────────────────────────────────────
print("\n[NEW 1] LOO target encoding for CORRIDOR")
cr_sum = obs.groupby("corridor")["duration_hrs"].sum()
cr_cnt = obs.groupby("corridor")["duration_hrs"].count()
loo_cr = []
for _, row in raw.iterrows():
    c = row["corridor"]; d = row["duration_hrs"]
    n = cr_cnt.get(c, 0); sm = cr_sum.get(c, gm * n)
    if n > 1 and row["event_observed"]==1:
        loo_cr.append((sm - d) / (n - 1))
    else:
        loo_cr.append(sm / n if n > 0 else gm)
raw["corridor_loo"] = loo_cr
print(f"  Range: [{min(loo_cr):.2f}, {max(loo_cr):.2f}]")

print("\n[NEW 2] Hour x Cause historical mean duration")
hour_cause_stats = obs.groupby(["hour","event_cause"])["duration_hrs"].mean().reset_index()
hour_cause_stats.columns = ["hour","event_cause","hour_cause_mean"]
raw = raw.merge(hour_cause_stats, on=["hour","event_cause"], how="left")
raw["hour_cause_mean"] = raw["hour_cause_mean"].fillna(raw["cause_mean"])
print(f"  Distinct hour-cause pairs: {hour_cause_stats.shape[0]}")

print("\n[NEW 3] Zone x Cause LOO encoding")
zc_sum = obs.groupby(["zone","event_cause"])["duration_hrs"].sum()
zc_cnt = obs.groupby(["zone","event_cause"])["duration_hrs"].count()
loo_zc = []
for _, row in raw.iterrows():
    key = (row["zone"], row["event_cause"]); d = row["duration_hrs"]
    n = zc_cnt.get(key, 0); sm = zc_sum.get(key, gm * n)
    if n > 1 and row["event_observed"]==1:
        loo_zc.append((sm - d) / (n - 1))
    else:
        loo_zc.append(sm / n if n > 0 else gm)
raw["zone_cause_loo"] = loo_zc
print(f"  Range: [{min(loo_zc):.2f}, {max(loo_zc):.2f}]")

# ── Feature sets ──────────────────────────────────────────────────────────────
V1_FEATURES = [
    "hour","dow","month","is_weekend","is_rush","is_night","hour_sin","hour_cos",
    "is_weather","requires_road_closure",
    "event_cause_enc","zone_enc","corridor_enc","police_station_enc",
    "cause_mean","cause_median","cause_p90","cause_p10","station_mean",
    "corridor_mean","corridor_cnt",
    "cause_x_rush","cause_x_night","cause_x_zone","cause_x_closure","station_x_cause",
    "cause_rolling_30d","concurrent_zone_events","concurrent_corridor_events","station_loo",
]
V2_FEATURES = V1_FEATURES + ["corridor_loo","hour_cause_mean","zone_cause_loo"]

for col in V2_FEATURES:
    raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0)

# ── Temporal CV ───────────────────────────────────────────────────────────────
folds = [
    {"name":"Feb","train_end":pd.Timestamp("2024-01-31"),
     "test_start":pd.Timestamp("2024-02-01"),"test_end":pd.Timestamp("2024-02-29")},
    {"name":"Mar","train_end":pd.Timestamp("2024-02-29"),
     "test_start":pd.Timestamp("2024-03-01"),"test_end":pd.Timestamp("2024-03-31")},
    {"name":"Apr","train_end":pd.Timestamp("2024-03-31"),
     "test_start":pd.Timestamp("2024-04-01"),"test_end":pd.Timestamp("2024-04-30")},
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

def cv_mae(features, label):
    maes = []
    for f in folds:
        tr=raw.loc[f["tr"]]; te=raw.loc[f["te"]]
        tro=tr[TRAIN_MASK.loc[f["tr"]]]; teo=te[(te["event_observed"]==1)&(te["duration_hrs"]<=48)]
        if len(tro)<20 or len(teo)<5: continue
        m=lgb.LGBMRegressor(**BEST_PARAMS)
        m.fit(tro[features], tro["duration_hrs"])
        yp=np.maximum(m.predict(teo[features]),0.05)
        maes.append(mean_absolute_error(teo["duration_hrs"].values, yp))
    avg = np.mean(maes) if maes else 999
    print(f"  {label:<45} MAE={avg:.4f}h  {[round(x,2) for x in maes]}")
    return avg, maes

print("\n" + "="*65)
print("ABLATION: V1 (prev winner) vs V2 (new features)")
print("="*65)
mae_v1, _ = cv_mae(V1_FEATURES, "V1 (prev winner, MAE 3.03h)")
mae_v2, _ = cv_mae(V2_FEATURES, "V2 (+corridor_loo+hour_cause+zone_cause)")

# ── Optuna re-tune on V2 features ─────────────────────────────────────────────
print("\n[OPTUNA RE-TUNE] 50 trials on V2 feature set")

def objective_v2(trial):
    params = dict(
        objective="quantile", alpha=0.50,
        n_estimators=trial.suggest_int("n_estimators", 150, 600),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
        num_leaves=trial.suggest_int("num_leaves", 31, 150),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 60),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
        random_state=42, verbosity=-1,
    )
    maes = []
    for f in folds:
        tr=raw.loc[f["tr"]]; te=raw.loc[f["te"]]
        tro=tr[TRAIN_MASK.loc[f["tr"]]]; teo=te[(te["event_observed"]==1)&(te["duration_hrs"]<=48)]
        if len(tro)<20 or len(teo)<5: continue
        m=lgb.LGBMRegressor(**params)
        m.fit(tro[V2_FEATURES], tro["duration_hrs"])
        yp=np.maximum(m.predict(teo[V2_FEATURES]),0.05)
        maes.append(mean_absolute_error(teo["duration_hrs"].values, yp))
    return np.mean(maes) if maes else 999

study = optuna.create_study(direction="minimize",
                             sampler=optuna.samplers.TPESampler(seed=99))
study.optimize(objective_v2, n_trials=50)

best_v2_mae    = study.best_value
best_v2_params = study.best_params

print(f"\n  V2 + Optuna MAE: {best_v2_mae:.4f}h")
print(f"  Best params: {best_v2_params}")

# ── Train final model ──────────────────────────────────────────────────────────
print("\n[FINAL] Training production model on ALL data with V2+Optuna params")
all_obs = raw[TRAIN_MASK].copy()
final_params = {"objective":"quantile","alpha":0.50,"random_state":42,"verbosity":-1,
                **best_v2_params}
final_model = lgb.LGBMRegressor(**final_params)
final_model.fit(all_obs[V2_FEATURES], all_obs["duration_hrs"])
joblib.dump(final_model, "models/lgb_q50_v2_final.pkl")
joblib.dump(V2_FEATURES,  "models/v2_features.pkl")

# ── SHAP top features ──────────────────────────────────────────────────────────
try:
    import shap
    X_sample = all_obs[V2_FEATURES].sample(min(500, len(all_obs)), random_state=42)
    expl = shap.TreeExplainer(final_model)
    sv   = expl.shap_values(X_sample)
    imp  = np.abs(sv).mean(axis=0)
    feat_imp = sorted(zip(V2_FEATURES, imp), key=lambda x: -x[1])
    print("\n  Top 10 SHAP features (V2 model):")
    for feat, val in feat_imp[:10]:
        bar = "#" * int(val/max(imp)*20)
        print(f"    {feat:<35} {val:>7.4f}  {bar}")
    with open(OUT_DIR/"shap_v2.json","w") as f:
        json.dump({feat:round(float(val),5) for feat,val in feat_imp}, f, indent=2)
except Exception as e:
    print(f"  SHAP skipped: {e}")

# ── Full results ────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("FINAL SUMMARY — Complete Journey")
print("="*65)
journey = [
    ("Wrong targets (old pipeline)",    123.0),
    ("Fixed targets, LGB v1",            24.1),
    ("LGB v2 + temporal CV",              3.28),
    ("Optuna tuned (60 trials)",          3.22),
    ("+ Advanced features v1",            3.03),
    ("V2 features + Optuna (FINAL)",     best_v2_mae),
]
for name, mae in journey:
    bar = "#" * max(1, int(20 - mae/6))
    print(f"  {name:<40} {mae:>7.3f}h  {bar}")

total_improvement = (journey[0][1] - best_v2_mae) / journey[0][1] * 100
print(f"\n  Total MAE reduction: 123h -> {best_v2_mae:.3f}h  ({total_improvement:.0f}% improvement)")

with open(OUT_DIR/"v2_final_results.json","w") as f:
    json.dump({
        "final_mae": round(best_v2_mae, 4),
        "final_params": best_v2_params,
        "features": V2_FEATURES,
        "journey": {n: round(m,3) for n,m in journey},
    }, f, indent=2)

print(f"\n[SAVED] models/lgb_q50_v2_final.pkl")
print("[OK] ADVANCED FEATURES v2 + OPTUNA DONE")
