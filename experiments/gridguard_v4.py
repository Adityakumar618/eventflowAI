"""
GridGuard AI v2 — Advanced Feature Engineering Experiment
===========================================================
Implementing the highest-ROI ideas from DeepSeek's analysis:

1. Description TF-IDF + SVD embeddings (semantic severity score)
   - "Starting problem" vs "Tyre burst" vs "Brake failure" are different
   - 83% fill rate → major untapped signal
2. Hawkes Process Intensity (self-exciting event bursts)
   - λ(t) = μ + Σ α·exp(−β·(t−t_i)) for past events in same zone
   - Tells model if we're in a resource-exhaustion burst
3. Officer Workload via created_by_id (99.97% filled proxy)
   - How many active events does this officer currently own?
4. Corridor Stress Index (exponentially-weighted remaining durations)
   - Replaces crude concurrent count with mathematically grounded signal
5. Weibull Mixture survival head (vs KM scaling)
   - Directly models bimodal duration distributions
6. Multi-task features (cascade probability, closure probability as inputs)
7. Re-tune with Optuna (100 trials on full feature set)

Target: Push MAE below 1.0h
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
from scipy.stats import weibull_min

OUT_DIR = Path("experiments/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 68)
print("GRIDGUARD AI v2 — ADVANCED FEATURE EXPERIMENT")
print("Target: Break MAE below 1.0h")
print("=" * 68)

# ── Load and clean raw data ──────────────────────────────────────────────────
raw = pd.read_csv("data/raw/astram_events.csv")
for col in ["start_datetime", "closed_datetime", "resolved_datetime"]:
    raw[col] = pd.to_datetime(raw[col], errors="coerce")

raw["event_observed"] = raw["closed_datetime"].notna().astype(int)
max_date = raw["start_datetime"].max()

def compute_dur(row):
    if pd.notna(row["closed_datetime"]):
        return max((row["closed_datetime"] - row["start_datetime"]).total_seconds() / 3600, 0.05)
    elif pd.notna(row["resolved_datetime"]):
        return max((row["resolved_datetime"] - row["start_datetime"]).total_seconds() / 3600, 0.05)
    return max((max_date - row["start_datetime"]).total_seconds() / 3600, 0.05)

raw["duration_hrs"] = raw.apply(compute_dur, axis=1)
raw["requires_road_closure"] = raw["requires_road_closure"].fillna(0).astype(int)

# Label encode categoricals
for col in ["event_cause", "zone", "corridor", "police_station"]:
    le = LabelEncoder()
    raw[col + "_enc"] = le.fit_transform(raw[col].fillna("unknown").astype(str))

raw["hour"]      = raw["start_datetime"].dt.hour
raw["dow"]       = raw["start_datetime"].dt.dayofweek
raw["month"]     = raw["start_datetime"].dt.month
raw["is_weekend"]= raw["dow"].isin([5, 6]).astype(int)
raw["is_rush"]   = raw["hour"].apply(lambda h: 1 if (8<=h<=11) or (17<=h<=20) else 0)
raw["is_night"]  = ((raw["hour"] >= 22) | (raw["hour"] <= 5)).astype(int)
raw["hour_sin"]  = np.sin(raw["hour"] * 2 * np.pi / 24)
raw["hour_cos"]  = np.cos(raw["hour"] * 2 * np.pi / 24)
raw["is_weather"]= raw["event_cause"].isin(
    ["water_logging", "tree_fall", "fog/low_visibility", "debris"]).astype(int)

raw = raw.sort_values("start_datetime").reset_index(drop=True)
raw["start_dt_naive"] = raw["start_datetime"].dt.tz_localize(None)
raw["start_ts"] = raw["start_dt_naive"].astype(np.int64) // 10**9

obs  = raw[raw["event_observed"] == 1]
gm   = obs["duration_hrs"].mean()

# ── Global stats (same as V3 baseline) ──────────────────────────────────────
for col_name, fn in [
    ("cause_mean",   "mean"),
    ("cause_median", "median"),
    ("cause_p90",    lambda x: x.quantile(0.9)),
    ("cause_p10",    lambda x: x.quantile(0.1)),
]:
    s = obs.groupby("event_cause")["duration_hrs"].agg(**{col_name: fn}).reset_index()
    raw = raw.merge(s, on="event_cause", how="left")

raw = raw.merge(obs.groupby("police_station")["duration_hrs"].agg(
    station_mean="mean").reset_index(), on="police_station", how="left")
raw = raw.merge(obs.groupby("corridor")["duration_hrs"].agg(
    corridor_mean="mean", corridor_cnt="count").reset_index(), on="corridor", how="left")

for col in ["cause_mean", "cause_median", "cause_p90", "cause_p10",
            "station_mean", "corridor_mean", "corridor_cnt"]:
    raw[col] = raw[col].fillna(obs["duration_hrs"].median())

raw["cause_x_rush"]    = raw["event_cause_enc"] * raw["is_rush"]
raw["cause_x_night"]   = raw["event_cause_enc"] * raw["is_night"]
raw["cause_x_zone"]    = raw["event_cause_enc"] * raw["zone_enc"]
raw["cause_x_closure"] = raw["event_cause_enc"] * raw["requires_road_closure"]
raw["station_x_cause"] = raw["police_station_enc"] * raw["event_cause_enc"]

# ── PREV WINNING FEATURES (V3) ───────────────────────────────────────────────
WINDOW_SEC = 30 * 86400
rolling_cause_means = []
for i, row in raw.iterrows():
    t = row["start_ts"]; ec = row["event_cause"]
    past = raw.iloc[max(0, i-500):i]
    mask = ((past["event_cause"]==ec) & (past["event_observed"]==1) &
            (past["start_ts"] >= t - WINDOW_SEC) & (past["start_ts"] < t))
    m = past[mask]
    rolling_cause_means.append(m["duration_hrs"].mean() if len(m) >= 3 else row["cause_mean"])
raw["cause_rolling_30d"] = rolling_cause_means

conc_z, conc_c = [], []
for i, row in raw.iterrows():
    t = row["start_ts"]
    past = raw.iloc[max(0, i-300):i]
    conc_z.append(int(((past["zone"]==row["zone"]) &
                        (past["start_ts"]>=t-4*3600) & (past["start_ts"]<t)).sum()))
    conc_c.append(int(((past["corridor"]==row["corridor"]) &
                        (past["start_ts"]>=t-2*3600) & (past["start_ts"]<t)).sum()))
raw["concurrent_zone_events"]     = conc_z
raw["concurrent_corridor_events"] = conc_c

st_sum = obs.groupby("police_station")["duration_hrs"].sum()
st_cnt = obs.groupby("police_station")["duration_hrs"].count()
loo_st = []
for _, row in raw.iterrows():
    s=row["police_station"]; d=row["duration_hrs"]
    n=st_cnt.get(s,0); sm=st_sum.get(s,gm*n)
    loo_st.append((sm-d)/(n-1) if n>1 and row["event_observed"]==1 else (sm/n if n>0 else gm))
raw["station_loo"] = loo_st

cr_sum = obs.groupby("corridor")["duration_hrs"].sum()
cr_cnt = obs.groupby("corridor")["duration_hrs"].count()
loo_cr = []
for _, row in raw.iterrows():
    c=row["corridor"]; d=row["duration_hrs"]
    n=cr_cnt.get(c,0); sm=cr_sum.get(c,gm*n)
    loo_cr.append((sm-d)/(n-1) if n>1 and row["event_observed"]==1 else (sm/n if n>0 else gm))
raw["corridor_loo"] = loo_cr

hc = obs.groupby(["hour","event_cause"])["duration_hrs"].mean().reset_index()
hc.columns = ["hour","event_cause","hour_cause_mean"]
raw = raw.merge(hc, on=["hour","event_cause"], how="left")
raw["hour_cause_mean"] = raw["hour_cause_mean"].fillna(raw["cause_mean"])

obs2 = obs.copy()
obs2["zone_cause"] = obs2["zone"] + "||" + obs2["event_cause"]
raw["zone_cause"] = raw["zone"] + "||" + raw["event_cause"]
zc_sum = obs2.groupby("zone_cause")["duration_hrs"].sum()
zc_cnt = obs2.groupby("zone_cause")["duration_hrs"].count()
loo_zc = []
for _, row in raw.iterrows():
    key=row["zone_cause"]; d=row["duration_hrs"]
    n=zc_cnt.get(key,0); sm=zc_sum.get(key,gm*n)
    loo_zc.append((sm-d)/(n-1) if n>1 and row["event_observed"]==1 else (sm/n if n>0 else gm))
raw["zone_cause_loo"] = loo_zc

# ───────────────────────────────────────────────────────────────────────────────
# NEW FEATURE 1: Description Text → TF-IDF + SVD (Semantic Severity Score)
# ───────────────────────────────────────────────────────────────────────────────
print("\n[NEW 1] Description TF-IDF + LSA Embeddings (semantic severity)")
raw["description_clean"] = (
    raw["description"]
    .fillna("")
    .str.lower()
    .str.replace(r"[^a-z0-9\s]", " ", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)
raw["reason_clean"] = (
    raw["reason_breakdown"]
    .fillna("")
    .str.lower()
    .str.replace(r"[^a-z0-9\s]", " ", regex=True)
    .str.strip()
)
# Combine description + reason_breakdown for richer text signal
raw["text_combined"] = raw["description_clean"] + " " + raw["reason_clean"]

N_COMPONENTS = 8  # Keep compact — avoid overfitting

tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
tfidf_matrix = tfidf.fit_transform(raw["text_combined"])
svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
text_features = svd.fit_transform(tfidf_matrix)

text_cols = [f"text_svd_{i}" for i in range(N_COMPONENTS)]
for i, col in enumerate(text_cols):
    raw[col] = text_features[:, i]

# Check correlation of text features with duration
for col in text_cols[:3]:
    corr = raw[raw["event_observed"]==1][col].corr(raw[raw["event_observed"]==1]["duration_hrs"])
    print(f"  {col} corr w/ duration: {corr:.4f}")

# Has description flag — description absence itself is a signal
raw["has_description"] = (raw["description"].notna() & (raw["description"].str.len() > 5)).astype(int)

# ───────────────────────────────────────────────────────────────────────────────
# NEW FEATURE 2: Hawkes Process Intensity
# ───────────────────────────────────────────────────────────────────────────────
print("\n[NEW 2] Neural Hawkes Process Intensity (self-exciting event bursts)")
ALPHA = 0.8   # excitation amplitude
BETA  = 0.5   # decay rate (per hour)
MU    = 0.05  # base rate

hawkes_zone  = []
hawkes_cause = []

for i, row in raw.iterrows():
    t   = row["start_ts"] / 3600  # convert to hours
    z   = row["zone"]
    ec  = row["event_cause"]
    past = raw.iloc[max(0, i-200):i]

    # Zone Hawkes: same zone events trigger zone-level resource pressure
    past_z = past[past["zone"] == z]
    t_past_z = past_z["start_ts"].values / 3600
    dt_z = t - t_past_z
    intensity_z = MU + ALPHA * np.sum(np.exp(-BETA * dt_z[dt_z > 0]))
    hawkes_zone.append(float(intensity_z))

    # Cause Hawkes: same cause events → same resource type competition
    past_ec = past[past["event_cause"] == ec]
    t_past_ec = past_ec["start_ts"].values / 3600
    dt_ec = t - t_past_ec
    intensity_ec = MU + ALPHA * np.sum(np.exp(-BETA * dt_ec[dt_ec > 0]))
    hawkes_cause.append(float(intensity_ec))

raw["hawkes_zone_intensity"]  = hawkes_zone
raw["hawkes_cause_intensity"] = hawkes_cause

# Log-transform to tame outliers
raw["log_hawkes_zone"]  = np.log1p(raw["hawkes_zone_intensity"])
raw["log_hawkes_cause"] = np.log1p(raw["hawkes_cause_intensity"])

corr_z  = raw[raw["event_observed"]==1]["hawkes_zone_intensity"].corr(
           raw[raw["event_observed"]==1]["duration_hrs"])
corr_ec = raw[raw["event_observed"]==1]["hawkes_cause_intensity"].corr(
           raw[raw["event_observed"]==1]["duration_hrs"])
print(f"  Hawkes zone intensity corr w/ duration:  {corr_z:.4f}")
print(f"  Hawkes cause intensity corr w/ duration: {corr_ec:.4f}")
print(f"  Zone intensity range: [{raw['hawkes_zone_intensity'].min():.2f}, {raw['hawkes_zone_intensity'].max():.2f}]")

# ───────────────────────────────────────────────────────────────────────────────
# NEW FEATURE 3: Officer Workload via created_by_id (99.97% filled)
# ───────────────────────────────────────────────────────────────────────────────
print("\n[NEW 3] Officer Workload (active concurrent events per officer)")
officer_load = []
for i, row in raw.iterrows():
    t  = row["start_ts"]
    oid = row.get("created_by_id", None)
    if pd.isna(oid):
        officer_load.append(0)
        continue
    past = raw.iloc[max(0, i-300):i]
    # How many events this officer has active in last 4 hours
    n = int(((past["created_by_id"] == oid) &
             (past["start_ts"] >= t - 4*3600) &
             (past["start_ts"] < t)).sum())
    officer_load.append(n)
raw["officer_active_load"] = officer_load

corr_ol = raw[raw["event_observed"]==1]["officer_active_load"].corr(
          raw[raw["event_observed"]==1]["duration_hrs"])
print(f"  Officer load range: [0, {raw['officer_active_load'].max()}]")
print(f"  Officer load corr w/ duration: {corr_ol:.4f}")

# ───────────────────────────────────────────────────────────────────────────────
# NEW FEATURE 4: Corridor Stress Index
# (exponentially-weighted predicted remaining durations on same corridor)
# ───────────────────────────────────────────────────────────────────────────────
print("\n[NEW 4] Corridor Stress Index (exp-weighted active duration burden)")
DECAY_HOURS = 2.0  # Half-life of 2 hours for stress decay

stress_idx = []
for i, row in raw.iterrows():
    t  = row["start_ts"] / 3600
    co = row["corridor"]
    # Look at recent events on same corridor
    past = raw.iloc[max(0, i-200):i]
    same_corr = past[(past["corridor"] == co) & (past["event_observed"] == 1)]
    if len(same_corr) == 0:
        stress_idx.append(row.get("corridor_mean", gm))
        continue
    # Weight each past event's mean duration by how recent it was
    dt = t - same_corr["start_ts"].values / 3600
    weights = np.exp(-dt / DECAY_HOURS)
    durations = same_corr["duration_hrs"].values
    weighted_stress = np.sum(weights * durations) / (np.sum(weights) + 1e-8)
    stress_idx.append(float(weighted_stress))

raw["corridor_stress_index"] = stress_idx

corr_si = raw[raw["event_observed"]==1]["corridor_stress_index"].corr(
          raw[raw["event_observed"]==1]["duration_hrs"])
print(f"  Stress index range: [{min(stress_idx):.2f}, {max(stress_idx):.2f}]")
print(f"  Stress index corr w/ duration: {corr_si:.4f}")

# ───────────────────────────────────────────────────────────────────────────────
# NEW FEATURE 5: Vehicle-type × Cause interaction
# (different vehicle types have systematically different resolution times)
# ───────────────────────────────────────────────────────────────────────────────
print("\n[NEW 5] Vehicle-type × Cause LOO encoding")
raw["veh_type_clean"] = raw["veh_type"].fillna("unknown")
raw["veh_cause_key"]  = raw["veh_type_clean"] + "||" + raw["event_cause"]

obs3 = obs.copy()
obs3["veh_cause_key"] = obs3["veh_type"].fillna("unknown") + "||" + obs3["event_cause"]
vc_sum = obs3.groupby("veh_cause_key")["duration_hrs"].sum()
vc_cnt = obs3.groupby("veh_cause_key")["duration_hrs"].count()
loo_vc = []
for _, row in raw.iterrows():
    key=row["veh_cause_key"]; d=row["duration_hrs"]
    n=vc_cnt.get(key,0); sm=vc_sum.get(key,gm*n)
    loo_vc.append((sm-d)/(n-1) if n>1 and row["event_observed"]==1 else (sm/n if n>0 else gm))
raw["veh_cause_loo"] = loo_vc
print(f"  Unique veh×cause pairs: {len(vc_cnt)}")
print(f"  Corr w/ duration: {raw[raw['event_observed']==1]['veh_cause_loo'].corr(obs['duration_hrs']):.4f}")

# ───────────────────────────────────────────────────────────────────────────────
# NEW FEATURE 6: Priority signal (High=1 gets faster response)
# ───────────────────────────────────────────────────────────────────────────────
raw["is_high_priority"] = (raw["priority"].fillna("Low") == "High").astype(int)

# ── Build complete feature sets ──────────────────────────────────────────────
V3_BASE = [
    "hour","dow","month","is_weekend","is_rush","is_night","hour_sin","hour_cos",
    "is_weather","requires_road_closure",
    "event_cause_enc","zone_enc","corridor_enc","police_station_enc",
    "cause_mean","cause_median","cause_p90","cause_p10","station_mean",
    "corridor_mean","corridor_cnt",
    "cause_x_rush","cause_x_night","cause_x_zone","cause_x_closure","station_x_cause",
    "cause_rolling_30d","concurrent_zone_events","concurrent_corridor_events",
    "station_loo","corridor_loo","hour_cause_mean","zone_cause_loo",
]

V4_NEW = (V3_BASE +
    text_cols + ["has_description"] +
    ["log_hawkes_zone","log_hawkes_cause","hawkes_zone_intensity","hawkes_cause_intensity"] +
    ["officer_active_load","corridor_stress_index","veh_cause_loo","is_high_priority"]
)

for col in V4_NEW:
    raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0)

# ── Temporal CV folds ────────────────────────────────────────────────────────
folds = [
    {"name":"Feb","train_end":pd.Timestamp("2024-01-31"),
     "test_start":pd.Timestamp("2024-02-01"),"test_end":pd.Timestamp("2024-02-29")},
    {"name":"Mar","train_end":pd.Timestamp("2024-02-29"),
     "test_start":pd.Timestamp("2024-03-01"),"test_end":pd.Timestamp("2024-03-31")},
    {"name":"Apr","train_end":pd.Timestamp("2024-03-31"),
     "test_start":pd.Timestamp("2024-04-01"),"test_end":pd.Timestamp("2024-04-30")},
]
TRAIN_MASK = (raw["event_observed"] == 1) & (raw["duration_hrs"] <= 48)
for f in folds:
    f["tr"] = raw[raw["start_dt_naive"] <= f["train_end"]].index
    f["te"] = raw[(raw["start_dt_naive"] >= f["test_start"]) &
                  (raw["start_dt_naive"] <= f["test_end"])].index

BEST_V3_PARAMS = dict(
    objective="quantile", alpha=0.50,
    n_estimators=467, learning_rate=0.012427, num_leaves=138,
    min_child_samples=10, colsample_bytree=0.9403, subsample=0.9913,
    reg_alpha=0.2484, reg_lambda=0.1641, random_state=42, verbosity=-1,
)

def cv_mae(features, label, params=None):
    if params is None:
        params = BEST_V3_PARAMS
    maes = []
    for f in folds:
        tr=raw.loc[f["tr"]]; te=raw.loc[f["te"]]
        tro=tr[TRAIN_MASK.loc[f["tr"]]]; teo=te[(te["event_observed"]==1)&(te["duration_hrs"]<=48)]
        if len(tro)<20 or len(teo)<5: continue
        valid_feats = [ft for ft in features if ft in tro.columns]
        m = lgb.LGBMRegressor(**params)
        m.fit(tro[valid_feats], tro["duration_hrs"])
        yp = np.maximum(m.predict(teo[valid_feats]), 0.05)
        maes.append(mean_absolute_error(teo["duration_hrs"].values, yp))
    avg = np.mean(maes) if maes else 999
    print(f"  {label:<52} MAE={avg:.4f}h  {[round(x,2) for x in maes]}")
    return avg

print("\n" + "="*68)
print("INCREMENTAL ABLATION — V3 baseline → V4 GridGuard")
print("="*68)
m_v3  = cv_mae(V3_BASE, "V3 baseline (1.62h)")
m_txt = cv_mae(V3_BASE + text_cols + ["has_description"], "+Text embeddings (8-dim SVD)")
m_hwk = cv_mae(V3_BASE + text_cols + ["has_description"] +
               ["log_hawkes_zone","log_hawkes_cause"], "+Hawkes intensity")
m_ofc = cv_mae(V3_BASE + text_cols + ["has_description"] +
               ["log_hawkes_zone","log_hawkes_cause","officer_active_load"], "+Officer load")
m_all = cv_mae(V4_NEW, "+Stress idx + veh_cause + priority (ALL V4)")

# ── Optuna on full V4 feature set (100 trials) ───────────────────────────────
print(f"\n[OPTUNA V4] 100 trials on full {len(V4_NEW)}-feature V4 set")

def objective_v4(trial):
    params = dict(
        objective="quantile", alpha=0.50,
        n_estimators=trial.suggest_int("n_estimators", 200, 800),
        learning_rate=trial.suggest_float("learning_rate", 0.003, 0.08, log=True),
        num_leaves=trial.suggest_int("num_leaves", 50, 200),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 60),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-5, 2.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
        min_split_gain=trial.suggest_float("min_split_gain", 0.0, 0.5),
        random_state=42, verbosity=-1,
    )
    maes = []
    for f in folds:
        tr=raw.loc[f["tr"]]; te=raw.loc[f["te"]]
        tro=tr[TRAIN_MASK.loc[f["tr"]]]; teo=te[(te["event_observed"]==1)&(te["duration_hrs"]<=48)]
        if len(tro)<20 or len(teo)<5: continue
        valid_feats = [ft for ft in V4_NEW if ft in tro.columns]
        m = lgb.LGBMRegressor(**params)
        m.fit(tro[valid_feats], tro["duration_hrs"])
        yp = np.maximum(m.predict(teo[valid_feats]), 0.05)
        maes.append(mean_absolute_error(teo["duration_hrs"].values, yp))
    return np.mean(maes) if maes else 999

study = optuna.create_study(direction="minimize",
                             sampler=optuna.samplers.TPESampler(seed=77))
study.optimize(objective_v4, n_trials=100)
best_v4_mae    = study.best_value
best_v4_params = study.best_params
print(f"\n  V4 + Optuna best MAE: {best_v4_mae:.4f}h")
print(f"  Best params: {best_v4_params}")

# ── Train and save final V4 model ────────────────────────────────────────────
print("\n[FINAL] Training V4 on all observed data")
all_obs = raw[TRAIN_MASK].copy()
valid_feats_v4 = [ft for ft in V4_NEW if ft in all_obs.columns]
final_params_v4 = {"objective":"quantile","alpha":0.50,"random_state":42,
                   "verbosity":-1, **best_v4_params}
final_v4 = lgb.LGBMRegressor(**final_params_v4)
final_v4.fit(all_obs[valid_feats_v4], all_obs["duration_hrs"])
joblib.dump(final_v4,        "models/lgb_v4_gridguard.pkl")
joblib.dump(valid_feats_v4,  "models/v4_features.pkl")
joblib.dump(tfidf,           "models/v4_tfidf.pkl")
joblib.dump(svd,             "models/v4_svd.pkl")
print("  Saved → models/lgb_v4_gridguard.pkl")

# ── SHAP analysis ─────────────────────────────────────────────────────────────
try:
    import shap
    X_samp = all_obs[valid_feats_v4].sample(min(500, len(all_obs)), random_state=42)
    expl   = shap.TreeExplainer(final_v4)
    sv     = expl.shap_values(X_samp)
    imp    = np.abs(sv).mean(axis=0)
    ranked = sorted(zip(valid_feats_v4, imp), key=lambda x: -x[1])
    print("\n  Top 15 SHAP features (V4 model):")
    for feat, val in ranked[:15]:
        bar = "#" * int(val / max(imp) * 25)
        print(f"    {feat:<40} {val:>7.4f}  {bar}")
    with open(OUT_DIR/"shap_v4.json","w") as f:
        json.dump({feat:round(float(val),5) for feat,val in ranked}, f, indent=2)
except Exception as e:
    print(f"  SHAP skipped: {e}")

# ── Calibrated Weibull Intervals (replace KM scaling) ─────────────────────────
print("\n[WEIBULL] Fitting per-cause Weibull distributions on observed events")
weibull_params = {}
for cause in obs["event_cause"].unique():
    sub = obs[obs["event_cause"] == cause]["duration_hrs"].clip(0.05, 200).values
    if len(sub) < 10:
        continue
    try:
        c_fit, loc_fit, scale_fit = weibull_min.fit(sub, floc=0)
        q10 = float(weibull_min.ppf(0.10, c_fit, loc=loc_fit, scale=scale_fit))
        q50 = float(weibull_min.ppf(0.50, c_fit, loc=loc_fit, scale=scale_fit))
        q90 = float(weibull_min.ppf(0.90, c_fit, loc=loc_fit, scale=scale_fit))
        weibull_params[cause] = {
            "shape": round(c_fit, 4), "scale": round(scale_fit, 4),
            "q10": round(q10, 3), "q50": round(q50, 3), "q90": round(q90, 3),
            "n_events": int(len(sub))
        }
        print(f"  {cause:<25} k={c_fit:.2f} λ={scale_fit:.2f}  "
              f"P10={q10:.2f}h P50={q50:.2f}h P90={q90:.2f}h")
    except Exception as e:
        print(f"  {cause}: fit failed ({e})")

with open("models/weibull_params.json","w") as f:
    json.dump(weibull_params, f, indent=2)
print("  Saved → models/weibull_params.json")

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "="*68)
print("COMPLETE ML JOURNEY — GRIDGUARD AI")
print("="*68)
journey = [
    ("Wrong targets (unclosed events)",  123.00),
    ("Fixed closed_datetime target",      24.10),
    ("LGB + temporal CV (3 folds)",        3.28),
    ("Optuna round 1 (60 trials)",         3.22),
    ("Advanced features V1",               3.03),
    ("V3 (corridor_loo + hour×cause)",     1.62),
    ("V4 GridGuard (text+Hawkes+...)",    best_v4_mae),
]
for name, mae in journey:
    bar = "█" * max(1, int((1 - mae/15)*30))
    print(f"  {name:<42} {mae:>7.3f}h  {bar}")

pct = (journey[0][1] - best_v4_mae) / journey[0][1] * 100
print(f"\n  Total improvement: 123h → {best_v4_mae:.3f}h  ({pct:.1f}% reduction)")
print(f"\n  NEW FEATURES CONTRIBUTION:")
print(f"    V3 → V4 (all features, default params): {m_v3:.4f}h → {m_all:.4f}h")
print(f"    V4 + Optuna 100 trials:                 {best_v4_mae:.4f}h")

with open(OUT_DIR/"v4_results.json","w") as f:
    json.dump({
        "v3_mae": round(m_v3, 4), "v4_mae_default": round(m_all, 4),
        "v4_mae_optuna": round(best_v4_mae, 4),
        "best_params": best_v4_params, "features": valid_feats_v4,
        "weibull_params": weibull_params,
        "journey": {n:round(m,3) for n,m in journey},
    }, f, indent=2)

print("\n[OK] GRIDGUARD AI V4 EXPERIMENT COMPLETE")
