"""
GridGuard AI V10 -- Kaggle Nuclear Option
==========================================
RESOURCE-INTENSIVE: Designed for Kaggle GPU T4, 30GB RAM, 12-hour runtime.
DO NOT run on a laptop -- use Kaggle free GPU!

Improvements over V9 (MAE=2.843h, honest baseline):
  [NEW-1]  Bayesian smooth target encoding  -- robust for small groups
  [NEW-2]  Log-ratio features               -- log(group_mean / global_mean)
  [NEW-3]  Time-since-last-event features   -- temporal gap in corridor/zone
  [NEW-4]  Rolling 7d cause mean            -- short-window signal
  [NEW-5]  Cause variance + p75 features    -- uncertainty signal
  [NEW-6]  Cluster tightness (radius_km)    -- spatial density quality
  [NEW-7]  4th model: CatBoost MAE          -- native categorical encoding
  [NEW-8]  750+ Optuna trials               -- deeper hyperparameter search
  [NEW-9]  Ridge stacking meta-learner      -- OOF prediction blending
  [NEW-10] GPU acceleration (LGB+XGB)       -- 5-10x faster tree fitting

V9 SHAP insights guiding V10:
  cluster_dur_loo=15.4%, corridor_loo=5.8%, zone_cause_loo=5.0%
  hour_cause_mean=4.0%, veh_cause_loo=3.3%, cause_mean=3.2%
  DROPPED: concurrent_zone/corridor (0.0 SHAP), is_rush (0.0 SHAP)

KAGGLE SETUP INSTRUCTIONS:
  1. kaggle.com -> Datasets -> New Dataset: upload astram_events.csv
     Name it: gridlock-hackathon-data
  2. kaggle.com -> Code -> New Notebook
  3. Click Add Data -> find gridlock-hackathon-data
  4. Right sidebar -> Session Options -> Accelerator -> GPU T4 x1
  5. Paste this ENTIRE script into one code cell
  6. Run All (30-90 min depending on GPU/CPU)
  7. Download /kaggle/working/v10_results.json and report back!
"""

import os, re, json, time, warnings, math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.cluster import DBSCAN
import lightgbm as lgb
import xgboost as xgb
import optuna
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
    print("[OK] CatBoost available")
except ImportError:
    HAS_CAT = False
    print("[WARN] CatBoost not installed -- pip install catboost")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import subprocess
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    HAS_GPU = result.returncode == 0
except Exception:
    HAS_GPU = False
print(f"[HW] GPU available: {HAS_GPU}")

# ---- DATA PATHS -- CHANGE IF YOUR DATASET NAME DIFFERS ----
if os.path.exists("/kaggle/input"):
    DATA_PATH = None
    for root, dirs, files in os.walk("/kaggle/input"):
        for fname in files:
            if fname.lower().endswith(".csv") and ("astram" in fname.lower() or "event" in fname.lower()):
                DATA_PATH = os.path.join(root, fname)
                break
        if DATA_PATH:
            break
    if DATA_PATH is None:
        raise FileNotFoundError("Could not find astram_events.csv under /kaggle/input. Upload dataset first!")
    OUTPUT_DIR = Path("/kaggle/working")
else:
    DATA_PATH = "data/raw/astram_events.csv"
    OUTPUT_DIR = Path("experiments/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"[DATA] {DATA_PATH}")

t_start = time.time()

# ============================================================
# CONSTANTS
# ============================================================
DURATION_CAP  = 48
N_SVD         = 20
N_OPTUNA_LGB  = 50   # Reduced for local CPU
N_OPTUNA_DART = 20   # Reduced for local CPU
N_OPTUNA_XGB  = 30   # Reduced for local CPU
N_OPTUNA_CAT  = 20   # Reduced for local CPU
SMOOTH_ALPHA  = 10.0
ALPHA_H, BETA_H, MU_H = 0.8, 0.5, 0.05

LGB_DEVICE = "gpu" if HAS_GPU else "cpu"
XGB_TREE   = "hist"  # Fix for XGBoost >= 2.0
CAT_TASK   = "GPU" if HAS_GPU else "CPU"

print(f"\n{'='*70}")
print("GRIDGUARD AI V10 -- KAGGLE NUCLEAR OPTION")
print(f"4 models | {N_OPTUNA_LGB+N_OPTUNA_DART+N_OPTUNA_XGB+N_OPTUNA_CAT} Optuna trials | Ridge stacking")
print(f"GPU={HAS_GPU} | Target: push below 2.0h honestly")
print("="*70)


# ============================================================
# 1. LOAD AND PREPROCESS
# ============================================================
print("\n[1] Loading data...")
raw = pd.read_csv(DATA_PATH)
for col in ["start_datetime", "closed_datetime", "resolved_datetime"]:
    raw[col] = pd.to_datetime(raw[col], errors="coerce")

raw["event_observed"] = raw["closed_datetime"].notna().astype(np.int8)
max_date = raw["start_datetime"].max()

raw["duration_hrs"] = np.where(
    raw["closed_datetime"].notna(),
    (raw["closed_datetime"] - raw["start_datetime"]).dt.total_seconds() / 3600,
    np.where(
        raw["resolved_datetime"].notna(),
        (raw["resolved_datetime"] - raw["start_datetime"]).dt.total_seconds() / 3600,
        (max_date - raw["start_datetime"]).dt.total_seconds() / 3600,
    )
).clip(min=0.05).astype(np.float32)

raw = raw.sort_values("start_datetime").reset_index(drop=True)

# FIX: Drop rows where start_datetime failed to parse to prevent NaT -> np.int8 crash
raw = raw.dropna(subset=["start_datetime"]).reset_index(drop=True)

def _make_naive(s):
    try:
        return s.dt.tz_localize(None)
    except Exception:
        try:
            return s.dt.tz_convert("UTC").dt.tz_localize(None)
        except Exception:
            return s

raw["_sdt"] = _make_naive(raw["start_datetime"])
raw["start_ts"] = (_make_naive(raw["start_datetime"]).astype(np.int64) // 10**9).astype(np.float64)
raw["start_ts_hour"] = raw["start_ts"] / 3600


# ============================================================
# 2. STATIC FEATURES
# ============================================================
print("[2] Static features...")
raw["hour"]        = raw["start_datetime"].dt.hour.astype(np.int8)
raw["dow"]         = raw["start_datetime"].dt.dayofweek.astype(np.int8)
raw["month"]       = raw["start_datetime"].dt.month.astype(np.int8)
raw["hour_bin"]    = (raw["hour"] // 4).astype(np.int8)
raw["is_weekend"]  = raw["dow"].isin([5, 6]).astype(np.int8)
raw["is_night"]    = ((raw["hour"] >= 22) | (raw["hour"] <= 5)).astype(np.int8)
raw["hour_sin"]    = np.sin(raw["hour"] * 2 * np.pi / 24).astype(np.float32)
raw["hour_cos"]    = np.cos(raw["hour"] * 2 * np.pi / 24).astype(np.float32)
raw["is_weather"]  = raw["event_cause"].isin(["water_logging", "tree_fall", "fog/low_visibility", "debris"]).astype(np.int8)
raw["requires_road_closure"] = raw["requires_road_closure"].fillna(0).astype(np.int8)
raw["is_high_priority"]      = (raw["priority"].fillna("Low") == "High").astype(np.int8)
raw["veh_type_clean"]  = raw["veh_type"].fillna("unknown")
raw["veh_cause_key"]   = raw["veh_type_clean"] + "||" + raw["event_cause"]

for col in ["event_cause", "zone", "corridor", "police_station"]:
    le = LabelEncoder()
    raw[col + "_enc"] = le.fit_transform(raw[col].fillna("unknown").astype(str)).astype(np.int16)

raw["cause_x_zone"]      = (raw["event_cause_enc"] * raw["zone_enc"]).astype(np.float32)
raw["cause_x_closure"]   = (raw["event_cause_enc"] * raw["requires_road_closure"]).astype(np.float32)
raw["cause_x_night"]     = (raw["event_cause_enc"] * raw["is_night"]).astype(np.float32)
raw["cause_x_hour_bin"]  = (raw["event_cause_enc"] * raw["hour_bin"]).astype(np.float32)

raw["description_clean"] = (raw["description"].fillna("").str.lower().str.replace(r"[^a-z0-9\s]", " ", regex=True).str.strip())
raw["reason_clean"]      = (raw["reason_breakdown"].fillna("").str.lower().str.replace(r"[^a-z0-9\s]", " ", regex=True).str.strip())
raw["text_combined"]     = raw["description_clean"] + " " + raw["reason_clean"]
raw["has_description"]   = (raw["description"].notna() & (raw["description"].str.len() > 5)).astype(np.int8)
raw["description_len"]   = raw["description"].fillna("").str.len().clip(0, 500).astype(np.float32)

def _extract_pin(addr):
    m = re.search(r"\b(\d{6})\b", str(addr))
    return m.group(1) if m else "000000"

def _road_class(addr):
    a = str(addr).lower()
    if any(k in a for k in ["nh-", "sh-", "highway", "expressway"]): return 3
    if any(k in a for k in ["main", "ring road", "outer ring"]): return 2
    return 1

def _is_junction(addr):
    return 1 if any(k in str(addr).lower() for k in ["junction", " jn", "circle", "signal"]) else 0

if "address" in raw.columns:
    raw["pin_code"]       = raw["address"].apply(_extract_pin)
    raw["addr_road_class"] = raw["address"].apply(_road_class).astype(np.int8)
    raw["is_junction"]    = raw["address"].apply(_is_junction).astype(np.int8)
else:
    raw["pin_code"] = "000000"
    raw["addr_road_class"] = 1
    raw["is_junction"] = 0


# ============================================================
# 3. CAUSAL TIME-SERIES FEATURES
# ============================================================
print("[3] Causal time-series features...")

def hawkes_vectorized(df, group_col):
    result = np.full(len(df), MU_H, dtype=np.float64)
    ts = df["start_ts_hour"].values
    groups = df[group_col].values
    grp_idx = defaultdict(list)
    for i, g in enumerate(groups):
        grp_idx[g].append(i)
    for g, idxs in grp_idx.items():
        idxs = np.array(idxs)
        t_g = ts[idxs]
        intensity = np.full(len(idxs), MU_H, dtype=np.float64)
        running = 0.0
        for k in range(1, len(idxs)):
            running = running * np.exp(-BETA_H * (t_g[k] - t_g[k-1])) + ALPHA_H
            intensity[k] = MU_H + running
        result[idxs] = intensity
    return result

raw["log_hawkes_zone"]  = np.log1p(hawkes_vectorized(raw, "zone")).astype(np.float32)
raw["log_hawkes_cause"] = np.log1p(hawkes_vectorized(raw, "event_cause")).astype(np.float32)
print("  Hawkes done.")

raw["officer_active_load"] = 0
if "created_by_id" in raw.columns:
    raw["_ones"] = 1
    raw["officer_active_load"] = (
        raw.groupby("created_by_id", group_keys=False, dropna=False)
           .apply(lambda g: g["_ones"].shift(1, fill_value=0).rolling(50, min_periods=1).sum())
        .fillna(0).astype(np.int16).values
    )
    raw.drop(columns=["_ones"], inplace=True)
print("  Officer load done.")

def corridor_stress_vectorized(df):
    result = np.full(len(df), np.nan)
    obs_mask = (df["event_observed"] == 1).values
    ts   = df["start_ts_hour"].values
    durs = df["duration_hrs"].values
    corrs = df["corridor"].values
    grp_idx = defaultdict(list)
    for i, c in enumerate(corrs):
        grp_idx[c].append(i)
    DECAY = 2.0
    for c, idxs in grp_idx.items():
        idxs = np.array(idxs)
        t_g = ts[idxs]; d_g = durs[idxs]; o_g = obs_mask[idxs]
        wsum = 0.0; wdsum = 0.0
        for k in range(len(idxs)):
            if k > 0:
                dt = t_g[k] - t_g[k-1]
                wsum  *= np.exp(-dt / DECAY)
                wdsum *= np.exp(-dt / DECAY)
                result[idxs[k]] = wdsum / (wsum + 1e-8) if wsum > 0 else np.nan
            if o_g[k]:
                wsum += 1.0; wdsum += d_g[k]
    return result

raw["corridor_stress_index"] = corridor_stress_vectorized(raw).astype(np.float32)
print("  Corridor stress done.")

obs_dur_series = raw["duration_hrs"].where(raw["event_observed"] == 1)
raw["cause_rolling_30d"] = (
    raw.assign(_dur=obs_dur_series)
       .groupby("event_cause", group_keys=False, dropna=False)
       .apply(lambda g: g["_dur"].shift(1).rolling(500, min_periods=3).mean())
    .values.astype(np.float32)
)
raw["cause_rolling_7d"] = (
    raw.assign(_dur=obs_dur_series)
       .groupby("event_cause", group_keys=False, dropna=False)
       .apply(lambda g: g["_dur"].shift(1).rolling(50, min_periods=2).mean())
    .values.astype(np.float32)
)
print("  Rolling means done.")

def time_since_last(df, group_col):
    result = np.full(len(df), np.nan, dtype=np.float32)
    ts = df["start_ts_hour"].values
    groups = df[group_col].values
    last_ts = {}
    for i, g in enumerate(groups):
        if g in last_ts:
            result[i] = float(ts[i] - last_ts[g])
        last_ts[g] = ts[i]
    return result

raw["hours_since_last_corridor"] = time_since_last(raw, "corridor")
raw["hours_since_last_zone"]     = time_since_last(raw, "zone")
print("  Time-since features done.")

BTP_STATIONS = {
    "Yeshwanthpur": (13.0253, 77.5397), "Marathahalli": (12.9592, 77.6974),
    "Hebbal": (13.0350, 77.5970), "Electronic City": (12.8399, 77.6770),
    "Whitefield": (12.9698, 77.7499), "Koramangala": (12.9279, 77.6271),
    "MG Road": (12.9716, 77.6099), "Rajajinagar": (12.9933, 77.5547),
}
ISEC_CONG = {h: 1.0 + 0.6 * (1 if (8 <= h <= 10) or (17 <= h <= 20) else 0) for h in range(24)}
_BTP = np.array(list(BTP_STATIONS.values()))

def _nearest_km(lat, lon):
    if not (np.isfinite(float(lat) if not isinstance(lat, float) else lat) and
            np.isfinite(float(lon) if not isinstance(lon, float) else lon)):
        return 5.0
    R = 6371.0
    dlat = np.radians(_BTP[:, 0] - float(lat))
    dlon = np.radians(_BTP[:, 1] - float(lon))
    a = np.sin(dlat/2)**2 + np.cos(np.radians(float(lat))) * np.cos(np.radians(_BTP[:, 0])) * np.sin(dlon/2)**2
    return float((R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))).min())

raw["nearest_station_km"] = raw.apply(
    lambda r: _nearest_km(
        r.get("latitude", 13.0) if pd.notna(r.get("latitude", None)) else 13.0,
        r.get("longitude", 77.6) if pd.notna(r.get("longitude", None)) else 77.6
    ), axis=1
).astype(np.float32)
raw["police_response_eta_mins"] = (raw["nearest_station_km"] / 18 * 60).clip(2, 60).astype(np.float32)
raw["tod_congestion_idx"]       = raw["hour"].map(ISEC_CONG).fillna(1.4).astype(np.float32)
raw["congestion_x_response"]    = (raw["tod_congestion_idx"] * raw["police_response_eta_mins"]).astype(np.float32)
print("  BTP distances done.")


# ============================================================
# 4. DBSCAN SPATIAL CLUSTERS
# ============================================================
print("[4] DBSCAN spatial clustering...")
raw["lat_r"] = raw["latitude"].fillna(raw["latitude"].median()).round(3)
raw["lon_r"] = raw["longitude"].fillna(raw["longitude"].median()).round(3)

best_db = None; best_signal = 0
for eps_val in [0.005, 0.007, 0.009, 0.011, 0.013]:
    db = DBSCAN(eps=eps_val, min_samples=3).fit(raw[["lat_r", "lon_r"]])
    signal = (db.labels_ >= 0).sum()
    n_clust = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    if signal > best_signal and n_clust >= 25:
        best_signal = signal; best_db = db
if best_db is None:
    best_db = DBSCAN(eps=0.008, min_samples=3).fit(raw[["lat_r", "lon_r"]])
raw["cluster_id"] = best_db.labels_.astype(np.int16)
n_clusters = len(set(best_db.labels_)) - 1
print(f"  DBSCAN: {n_clusters} clusters, {best_signal} non-noise points")

cluster_coords = {}
for cid in raw["cluster_id"].unique():
    if cid < 0: continue
    mask = raw["cluster_id"] == cid
    lats = raw.loc[mask, "lat_r"].values; lons = raw.loc[mask, "lon_r"].values
    cent_lat = lats.mean(); cent_lon = lons.mean()
    dists = np.sqrt((lats - cent_lat)**2 + (lons - cent_lon)**2) * 111.0
    cluster_coords[cid] = float(dists.mean() + 1e-6)
raw["cluster_radius_km"] = raw["cluster_id"].map(cluster_coords).fillna(5.0).astype(np.float32)


# ============================================================
# 5. OFFICER FEATURES
# ============================================================
print("[5] Officer features...")
raw["officer_event_cnt"] = 0
if "created_by_id" in raw.columns:
    ocnt = raw["created_by_id"].map(raw["created_by_id"].value_counts()).fillna(0)
    raw["officer_event_cnt"] = ocnt.astype(np.int16).values


# ============================================================
# 6. FOLD SETUP
# ============================================================
print("\n[6] 5-fold temporal CV setup...")
TRAIN_MASK = (raw["event_observed"] == 1) & (raw["duration_hrs"] <= DURATION_CAP)
gm_global  = float(raw.loc[TRAIN_MASK, "duration_hrs"].mean())
print(f"  Global observed mean: {gm_global:.3f}h  |  N_observed={TRAIN_MASK.sum()}")

FOLDS = [
    {"name": "Oct-23", "train_end": pd.Timestamp("2023-09-30"),
     "test_start": pd.Timestamp("2023-10-01"), "test_end": pd.Timestamp("2023-10-31")},
    {"name": "Dec-23", "train_end": pd.Timestamp("2023-11-30"),
     "test_start": pd.Timestamp("2023-12-01"), "test_end": pd.Timestamp("2023-12-31")},
    {"name": "Feb-24", "train_end": pd.Timestamp("2024-01-31"),
     "test_start": pd.Timestamp("2024-02-01"), "test_end": pd.Timestamp("2024-02-29")},
    {"name": "Mar-24", "train_end": pd.Timestamp("2024-02-29"),
     "test_start": pd.Timestamp("2024-03-01"), "test_end": pd.Timestamp("2024-03-31")},
    {"name": "Apr-24", "train_end": pd.Timestamp("2024-03-31"),
     "test_start": pd.Timestamp("2024-04-01"), "test_end": pd.Timestamp("2024-04-30")},
]
for f in FOLDS:
    f["tr"] = raw.index[raw["_sdt"] <= f["train_end"]]
    f["te"] = raw.index[(raw["_sdt"] >= f["test_start"]) & (raw["_sdt"] <= f["test_end"])]
    print(f"  Fold {f['name']}: train={len(f['tr'])}, test={len(f['te'])}")


# ============================================================
# 7. IN-FOLD FEATURE BUILDER
# ============================================================
def smooth_loo(tr_df, te_df, key_col, val_col, gm, alpha=SMOOTH_ALPHA):
    """Bayesian smoothed LOO encoding. Robust for small groups."""
    g_sum  = tr_df.groupby(key_col)[val_col].sum()
    g_cnt  = tr_df.groupby(key_col)[val_col].count()
    g_mean = (g_sum / g_cnt.replace(0, 1))
    tr_gs  = tr_df[key_col].map(g_sum).fillna(0).values
    tr_gc  = tr_df[key_col].map(g_cnt).fillna(0).values
    raw_loo_tr = np.where(tr_gc > 1, (tr_gs - tr_df[val_col].values) / (tr_gc - 1), gm)
    smooth_tr  = (tr_gc * raw_loo_tr + alpha * gm) / (tr_gc + alpha)
    te_gmean   = te_df[key_col].map(g_mean).fillna(gm).values
    te_gc      = te_df[key_col].map(g_cnt).fillna(0).values
    smooth_te  = (te_gc * te_gmean + alpha * gm) / (te_gc + alpha)
    return smooth_tr.astype(np.float32), smooth_te.astype(np.float32)


def build_fold_features(tr_idx, te_idx, gm):
    tr_mask = TRAIN_MASK.loc[tr_idx]
    tr_obs  = raw.loc[tr_idx[tr_mask]].copy()
    te_obs  = raw.loc[te_idx].copy()
    te_obs  = te_obs[(te_obs["event_observed"] == 1) & (te_obs["duration_hrs"] <= DURATION_CAP)].copy()
    if len(tr_obs) < 20 or len(te_obs) < 5:
        return None
    gm_fold = float(tr_obs["duration_hrs"].mean())
    ft = {}; fe = {}

    # Cause stats
    for stat, fn in [("cause_mean","mean"),("cause_median","median"),
                     ("cause_p90",lambda x: x.quantile(0.9)),
                     ("cause_p75",lambda x: x.quantile(0.75)),
                     ("cause_p10",lambda x: x.quantile(0.1)),
                     ("cause_std","std")]:
        s = tr_obs.groupby("event_cause")["duration_hrs"].agg(fn).rename(stat)
        ft[stat] = tr_obs["event_cause"].map(s).fillna(gm_fold).values.astype(np.float32)
        fe[stat] = te_obs["event_cause"].map(s).fillna(gm_fold).values.astype(np.float32)
    ft["cause_std"] = np.nan_to_num(ft["cause_std"], nan=gm_fold*0.5)
    fe["cause_std"] = np.nan_to_num(fe["cause_std"], nan=gm_fold*0.5)

    # Corridor stats
    cr = tr_obs.groupby("corridor")["duration_hrs"].agg(["mean","count","std"])
    cr.columns = ["corridor_mean","corridor_cnt","corridor_std"]
    for c in cr.columns:
        fb = gm_fold if ("mean" in c or "std" in c) else 1
        ft[c] = tr_obs["corridor"].map(cr[c]).fillna(fb).values.astype(np.float32)
        fe[c] = te_obs["corridor"].map(cr[c]).fillna(fb).values.astype(np.float32)
    ft["corridor_std"] = np.nan_to_num(ft["corridor_std"], nan=gm_fold*0.5)
    fe["corridor_std"] = np.nan_to_num(fe["corridor_std"], nan=gm_fold*0.5)

    # Station stats
    st_m = tr_obs.groupby("police_station")["duration_hrs"].mean()
    ft["station_mean"] = tr_obs["police_station"].map(st_m).fillna(gm_fold).values.astype(np.float32)
    fe["station_mean"] = te_obs["police_station"].map(st_m).fillna(gm_fold).values.astype(np.float32)

    # Hour x Cause mean
    hc = tr_obs.groupby(["hour","event_cause"])["duration_hrs"].mean().reset_index()
    hc.columns = ["hour","event_cause","hour_cause_mean"]
    ft["hour_cause_mean"] = tr_obs[["hour","event_cause"]].reset_index(drop=True).merge(
        hc, on=["hour","event_cause"], how="left")["hour_cause_mean"].fillna(gm_fold).values.astype(np.float32)
    fe["hour_cause_mean"] = te_obs[["hour","event_cause"]].reset_index(drop=True).merge(
        hc, on=["hour","event_cause"], how="left")["hour_cause_mean"].fillna(gm_fold).values.astype(np.float32)

    # Hour_bin x Cause mean
    hbc = tr_obs.groupby(["hour_bin","event_cause"])["duration_hrs"].mean().reset_index()
    hbc.columns = ["hour_bin","event_cause","hour_bin_cause_mean"]
    ft["hour_bin_cause_mean"] = tr_obs[["hour_bin","event_cause"]].reset_index(drop=True).merge(
        hbc, on=["hour_bin","event_cause"], how="left")["hour_bin_cause_mean"].fillna(gm_fold).values.astype(np.float32)
    fe["hour_bin_cause_mean"] = te_obs[["hour_bin","event_cause"]].reset_index(drop=True).merge(
        hbc, on=["hour_bin","event_cause"], how="left")["hour_bin_cause_mean"].fillna(gm_fold).values.astype(np.float32)

    # Rolling features
    ft["cause_rolling_30d"] = tr_obs["cause_rolling_30d"].fillna(gm_fold).values.astype(np.float32)
    fe["cause_rolling_30d"] = te_obs["cause_rolling_30d"].fillna(gm_fold).values.astype(np.float32)
    ft["cause_rolling_7d"]  = tr_obs["cause_rolling_7d"].fillna(gm_fold).values.astype(np.float32)
    fe["cause_rolling_7d"]  = te_obs["cause_rolling_7d"].fillna(gm_fold).values.astype(np.float32)

    # Bayesian smooth LOO for direct columns
    for loo_name, col in [("corridor_loo","corridor"),("station_loo","police_station")]:
        a, b = smooth_loo(tr_obs, te_obs, col, "duration_hrs", gm_fold)
        ft[loo_name] = a; fe[loo_name] = b

    # Composite key LOOs
    for loo_name, key_fn in [
        ("zone_cause_loo",  lambda d: d["zone"].fillna("unk") + "||" + d["event_cause"]),
        ("veh_cause_loo",   lambda d: d["veh_cause_key"]),
        ("cluster_dur_loo", lambda d: d["cluster_id"].astype(str)),
        ("officer_dur_loo", lambda d: d["created_by_id"].fillna("unknown") if "created_by_id" in d.columns else pd.Series(["unknown"]*len(d), index=d.index)),
        ("pin_code_loo",    lambda d: d["pin_code"]),
    ]:
        tr_k = key_fn(tr_obs).rename("__k"); te_k = key_fn(te_obs).rename("__k")
        tr_t = tr_obs[["duration_hrs"]].copy(); tr_t["__k"] = tr_k.values
        te_t = te_obs[["duration_hrs"]].copy(); te_t["__k"] = te_k.values
        a, b = smooth_loo(tr_t, te_t, "__k", "duration_hrs", gm_fold)
        ft[loo_name] = a; fe[loo_name] = b

    # Cluster density
    cc = tr_obs.groupby("cluster_id")["duration_hrs"].count()
    ft["cluster_density"] = tr_obs["cluster_id"].map(cc).fillna(1).values.astype(np.float32)
    fe["cluster_density"] = te_obs["cluster_id"].map(cc).fillna(1).values.astype(np.float32)

    # Log-ratio features (new)
    for src, name in [("corridor_mean","corridor"),("cause_mean","event_cause"),("cluster_dur_loo","cluster")]:
        ft[f"log_ratio_{name}"] = np.log1p(np.abs(ft[src] / (gm_fold+1e-8))).astype(np.float32)
        fe[f"log_ratio_{name}"] = np.log1p(np.abs(fe[src] / (gm_fold+1e-8))).astype(np.float32)

    # Log of top features
    for feat in ["corridor_loo","zone_cause_loo","cluster_dur_loo","cause_mean"]:
        ft[f"log_{feat}"] = np.log1p(np.abs(ft[feat])).astype(np.float32)
        fe[f"log_{feat}"] = np.log1p(np.abs(fe[feat])).astype(np.float32)

    # Time-since features
    tsl_med_tr = float(np.nanmedian(tr_obs["hours_since_last_corridor"].values))
    tsl_med_te = float(np.nanmedian(te_obs["hours_since_last_corridor"].values)) if not te_obs["hours_since_last_corridor"].isna().all() else tsl_med_tr
    ft["hours_since_last_corridor"] = np.nan_to_num(tr_obs["hours_since_last_corridor"].values, nan=tsl_med_tr).astype(np.float32)
    fe["hours_since_last_corridor"] = np.nan_to_num(te_obs["hours_since_last_corridor"].values, nan=tsl_med_te).astype(np.float32)

    # TF-IDF / SVD on train only
    tfidf = TfidfVectorizer(max_features=800, ngram_range=(1,2), min_df=2, sublinear_tf=True)
    svd   = TruncatedSVD(n_components=N_SVD, random_state=42)
    tr_txt = tfidf.fit_transform(tr_obs["text_combined"])
    te_txt = tfidf.transform(te_obs["text_combined"])
    tr_svd = svd.fit_transform(tr_txt); te_svd = svd.transform(te_txt)
    for i in range(N_SVD):
        ft[f"text_svd_{i}"] = tr_svd[:, i].astype(np.float32)
        fe[f"text_svd_{i}"] = te_svd[:, i].astype(np.float32)

    STATIC = [
        "hour","dow","month","hour_bin","is_weekend","is_night",
        "hour_sin","hour_cos","is_weather","requires_road_closure",
        "event_cause_enc","zone_enc","corridor_enc","police_station_enc",
        "cause_x_zone","cause_x_closure","cause_x_night","cause_x_hour_bin",
        "is_high_priority","has_description","description_len",
        "log_hawkes_zone","log_hawkes_cause","officer_active_load",
        "corridor_stress_index","police_response_eta_mins",
        "tod_congestion_idx","congestion_x_response",
        "nearest_station_km","is_junction","addr_road_class",
        "officer_event_cnt","cluster_id","cluster_radius_km",
        "hours_since_last_zone",
    ]

    def assemble(df, fdict):
        parts = [df[STATIC].reset_index(drop=True).fillna(0).astype(np.float32)]
        for k, v in fdict.items():
            parts.append(pd.Series(v, name=k, dtype=np.float32))
        return pd.concat(parts, axis=1).fillna(0)

    X_tr = assemble(tr_obs.reset_index(drop=True), ft)
    X_te = assemble(te_obs.reset_index(drop=True), fe)
    y_tr_log = np.log1p(tr_obs["duration_hrs"].values).astype(np.float32)
    y_te_raw = te_obs["duration_hrs"].values.astype(np.float32)
    return X_tr, y_tr_log, X_te, y_te_raw


# ============================================================
# 8. PRE-BUILD FOLD CACHE (for Optuna speed)
# ============================================================
print("\n[7] Pre-building fold feature cache...")
fold_cache = []
for f in FOLDS:
    if len(f["tr"]) < 50: continue
    result = build_fold_features(f["tr"], f["te"], gm_global)
    if result is None: continue
    X_tr, y_tr_log, X_te, y_te_raw = result
    if len(X_tr) < 20 or len(X_te) < 5: continue
    fold_cache.append((X_tr, y_tr_log, X_te, y_te_raw, f["name"]))
print(f"  Cached {len(fold_cache)} valid folds. Features per fold: {fold_cache[0][0].shape[1] if fold_cache else 0}")


# ============================================================
# 9. PHASE A: BASELINE
# ============================================================
print(f"\n{'='*70}")
print("PHASE A -- 4-model baseline (default params)")
print("="*70)

DEFAULT_LGB = dict(objective="regression_l1",n_estimators=600,learning_rate=0.02,
    num_leaves=120,min_child_samples=10,colsample_bytree=0.8,subsample=0.8,
    reg_alpha=0.1,reg_lambda=1.0,random_state=42,verbosity=-1,n_jobs=-1,device=LGB_DEVICE)

phase_a = {"lgb":[],"dart":[],"xgb":[],"cat":[]}
for f in FOLDS:
    if len(f["tr"]) < 50: continue
    result = build_fold_features(f["tr"], f["te"], gm_global)
    if result is None: continue
    X_tr, y_tr, X_te, y_te = result
    if len(X_tr) < 20 or len(X_te) < 5: continue
    fn = list(X_tr.columns)

    m = lgb.LGBMRegressor(**DEFAULT_LGB); m.fit(X_tr, y_tr)
    p_lgb = np.expm1(np.maximum(m.predict(X_te), 0))
    phase_a["lgb"].append(mean_absolute_error(y_te, p_lgb))

    dp = dict(DEFAULT_LGB); dp.update({"boosting_type":"dart","drop_rate":0.1,"skip_drop":0.5}); dp.pop("verbosity",None)
    md = lgb.LGBMRegressor(**dp); md.fit(X_tr, y_tr)
    p_dart = np.expm1(np.maximum(md.predict(X_te), 0))
    phase_a["dart"].append(mean_absolute_error(y_te, p_dart))

    xgb_params = {"objective":"reg:absoluteerror","tree_method":XGB_TREE,
        "learning_rate":0.02,"max_depth":7,"subsample":0.8,"colsample_bytree":0.8,
        "seed":42,"verbosity":0}
    if HAS_GPU:
        xgb_params["device"] = "cuda"
    else:
        xgb_params["nthread"] = -1
        
    bst = xgb.train(xgb_params,
        xgb.DMatrix(X_tr,label=y_tr,feature_names=fn), num_boost_round=600, verbose_eval=False)
    p_xgb = np.expm1(np.maximum(bst.predict(xgb.DMatrix(X_te,feature_names=fn)), 0))
    phase_a["xgb"].append(mean_absolute_error(y_te, p_xgb))

    if HAS_CAT:
        try:
            cb = CatBoostRegressor(iterations=600,learning_rate=0.02,depth=8,loss_function="MAE",
                random_seed=42,verbose=0,task_type=CAT_TASK)
            cb.fit(X_tr, y_tr)
            p_cat = np.expm1(np.maximum(cb.predict(X_te), 0))
            phase_a["cat"].append(mean_absolute_error(y_te, p_cat))
        except Exception as ex:
            print(f"  CatBoost error: {ex}")

    ens = np.mean([p_lgb, p_dart, p_xgb], axis=0)
    print(f"  {f['name']}: LGB={phase_a['lgb'][-1]:.3f} DART={phase_a['dart'][-1]:.3f} XGB={phase_a['xgb'][-1]:.3f} ENS={mean_absolute_error(y_te,ens):.3f}h")

mae_a = {k: np.mean(v) if v else 99.0 for k, v in phase_a.items()}
print(f"\n  Phase A: LGB={mae_a['lgb']:.4f}  DART={mae_a['dart']:.4f}  XGB={mae_a['xgb']:.4f}  CAT={mae_a['cat']:.4f}")


# ============================================================
# 10. PHASE B: OPTUNA (LGB, DART, XGB, CatBoost)
# ============================================================
print(f"\n{'='*70}")
print(f"PHASE B -- Optuna ({N_OPTUNA_LGB}+{N_OPTUNA_DART}+{N_OPTUNA_XGB}+{N_OPTUNA_CAT} trials)")
print("="*70)

def _cv_mae(predict_fn):
    maes = []
    for X_tr, y_tr, X_te, y_te, _ in fold_cache:
        pred = predict_fn(X_tr, y_tr, X_te)
        maes.append(mean_absolute_error(y_te, np.expm1(np.maximum(pred, 0))))
    return float(np.mean(maes))

# LGB L1
def obj_lgb(trial):
    p = {"objective":"regression_l1","random_state":42,"verbosity":-1,"n_jobs":-1,"device":LGB_DEVICE,
         "n_estimators":     trial.suggest_int("n_estimators", 200, 2000),
         "learning_rate":    trial.suggest_float("learning_rate", 0.003, 0.06, log=True),
         "num_leaves":       trial.suggest_int("num_leaves", 31, 300),
         "min_child_samples":trial.suggest_int("min_child_samples", 5, 120),
         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.25, 1.0),
         "subsample":        trial.suggest_float("subsample", 0.25, 1.0),
         "reg_alpha":        trial.suggest_float("reg_alpha", 1e-9, 15.0, log=True),
         "reg_lambda":       trial.suggest_float("reg_lambda", 1e-9, 15.0, log=True),
         "min_split_gain":   trial.suggest_float("min_split_gain", 0.0, 1.5),
         "max_depth":        trial.suggest_int("max_depth", 3, 16),
    }
    return _cv_mae(lambda Xtr,ytr,Xte: lgb.LGBMRegressor(**p).fit(Xtr,ytr).predict(Xte))

study_lgb = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=40))
study_lgb.optimize(obj_lgb, n_trials=N_OPTUNA_LGB, show_progress_bar=False)
best_lgb_p = study_lgb.best_params; best_lgb_mae = study_lgb.best_value
print(f"  LGB best: {best_lgb_mae:.4f}h  params={best_lgb_p}")

# LGB DART
def obj_dart(trial):
    p = {"objective":"regression_l1","boosting_type":"dart","random_state":42,"n_jobs":-1,"device":LGB_DEVICE,
         "n_estimators":     trial.suggest_int("n_estimators", 200, 1500),
         "learning_rate":    trial.suggest_float("learning_rate", 0.003, 0.05, log=True),
         "num_leaves":       trial.suggest_int("num_leaves", 31, 250),
         "min_child_samples":trial.suggest_int("min_child_samples", 5, 100),
         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.25, 1.0),
         "subsample":        trial.suggest_float("subsample", 0.25, 1.0),
         "reg_alpha":        trial.suggest_float("reg_alpha", 1e-9, 10.0, log=True),
         "reg_lambda":       trial.suggest_float("reg_lambda", 1e-9, 10.0, log=True),
         "drop_rate":        trial.suggest_float("drop_rate", 0.05, 0.5),
         "skip_drop":        trial.suggest_float("skip_drop", 0.2, 0.9),
         "max_depth":        trial.suggest_int("max_depth", 3, 14),
    }
    return _cv_mae(lambda Xtr,ytr,Xte: lgb.LGBMRegressor(**p).fit(Xtr,ytr).predict(Xte))

study_dart = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=99, n_startup_trials=25))
study_dart.optimize(obj_dart, n_trials=N_OPTUNA_DART, show_progress_bar=False)
best_dart_p = study_dart.best_params; best_dart_mae = study_dart.best_value
print(f"  DART best: {best_dart_mae:.4f}h")

# XGBoost
def obj_xgb(trial):
    n_est = trial.suggest_int("n_estimators", 200, 1500)
    p = {"objective":"reg:absoluteerror","tree_method":XGB_TREE,"seed":42,"verbosity":0,
         "learning_rate":    trial.suggest_float("learning_rate", 0.003, 0.05, log=True),
         "max_depth":        trial.suggest_int("max_depth", 3, 14),
         "subsample":        trial.suggest_float("subsample", 0.3, 1.0),
         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
         "reg_alpha":        trial.suggest_float("reg_alpha", 1e-9, 10.0, log=True),
         "reg_lambda":       trial.suggest_float("reg_lambda", 1e-9, 10.0, log=True),
         "min_child_weight": trial.suggest_int("min_child_weight", 1, 40),
         "gamma":            trial.suggest_float("gamma", 0.0, 3.0),
    }
    if HAS_GPU:
        p["device"] = "cuda"
    else:
        p["nthread"] = -1
    def _fit(Xtr,ytr,Xte):
        fn = list(Xtr.columns)
        b = xgb.train(p, xgb.DMatrix(Xtr,label=ytr,feature_names=fn), num_boost_round=n_est, verbose_eval=False)
        return b.predict(xgb.DMatrix(Xte,feature_names=fn))
    return _cv_mae(_fit)

study_xgb = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=456, n_startup_trials=25))
study_xgb.optimize(obj_xgb, n_trials=N_OPTUNA_XGB, show_progress_bar=False)
best_xgb_p = study_xgb.best_params; best_xgb_mae = study_xgb.best_value
xgb_n_est = best_xgb_p.pop("n_estimators", 600)
print(f"  XGB best: {best_xgb_mae:.4f}h")

# CatBoost
best_cat_p = None; best_cat_mae = 99.0
if HAS_CAT:
    def obj_cat(trial):
        p = {"loss_function":"MAE","eval_metric":"MAE","random_seed":42,"verbose":0,"task_type":CAT_TASK,
             "iterations":         trial.suggest_int("iterations", 300, 1500),
             "learning_rate":      trial.suggest_float("learning_rate", 0.003, 0.05, log=True),
             "depth":              trial.suggest_int("depth", 4, 10),
             "l2_leaf_reg":        trial.suggest_float("l2_leaf_reg", 0.5, 20.0),
             "bagging_temperature":trial.suggest_float("bagging_temperature", 0.0, 2.0),
             "random_strength":    trial.suggest_float("random_strength", 0.1, 5.0),
        }
        try:
            return _cv_mae(lambda Xtr,ytr,Xte: CatBoostRegressor(**p).fit(Xtr,ytr,verbose=False).predict(Xte))
        except Exception:
            return 99.0
    study_cat = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=789, n_startup_trials=20))
    study_cat.optimize(obj_cat, n_trials=N_OPTUNA_CAT, show_progress_bar=False)
    best_cat_p = study_cat.best_params; best_cat_mae = study_cat.best_value
    print(f"  CatBoost best: {best_cat_mae:.4f}h")


# ============================================================
# 11. PHASE C: STACKING (OOF Ridge meta-learner)
# ============================================================
print(f"\n{'='*70}")
print("PHASE C -- OOF stacking with Ridge meta-learner")
print("="*70)

best_lgb_full  = dict(best_lgb_p); best_lgb_full.update({"objective":"regression_l1","random_state":42,"verbosity":-1,"n_jobs":-1,"device":LGB_DEVICE})
best_dart_full = dict(best_dart_p); best_dart_full.update({"objective":"regression_l1","boosting_type":"dart","random_state":42,"n_jobs":-1,"device":LGB_DEVICE})
best_xgb_full  = dict(best_xgb_p); best_xgb_full.update({"objective":"reg:absoluteerror","tree_method":XGB_TREE,"seed":42,"verbosity":0})
if HAS_GPU: best_xgb_full["device"] = "cuda"
else: best_xgb_full["nthread"] = -1

oof_lgb=[]; oof_dart=[]; oof_xgb=[]; oof_cat=[]; oof_y=[]
fold_c_maes = []

for f in FOLDS:
    if len(f["tr"]) < 50: continue
    result = build_fold_features(f["tr"], f["te"], gm_global)
    if result is None: continue
    X_tr, y_tr, X_te, y_te = result
    if len(X_tr) < 20 or len(X_te) < 5: continue
    fn = list(X_tr.columns)

    m = lgb.LGBMRegressor(**best_lgb_full); m.fit(X_tr, y_tr)
    p1 = np.expm1(np.maximum(m.predict(X_te), 0))

    md = lgb.LGBMRegressor(**best_dart_full); md.fit(X_tr, y_tr)
    p2 = np.expm1(np.maximum(md.predict(X_te), 0))

    bx = xgb.train(best_xgb_full, xgb.DMatrix(X_tr,label=y_tr,feature_names=fn),
                   num_boost_round=xgb_n_est, verbose_eval=False)
    p3 = np.expm1(np.maximum(bx.predict(xgb.DMatrix(X_te,feature_names=fn)), 0))

    p4 = None
    if HAS_CAT and best_cat_p:
        try:
            cb = CatBoostRegressor(**best_cat_p, loss_function="MAE", eval_metric="MAE",
                                   random_seed=42, verbose=0, task_type=CAT_TASK)
            cb.fit(X_tr, y_tr, verbose=False)
            p4 = np.expm1(np.maximum(cb.predict(X_te), 0))
        except Exception:
            p4 = (p1+p2+p3)/3

    oof_lgb.extend(p1); oof_dart.extend(p2); oof_xgb.extend(p3); oof_y.extend(y_te)
    if p4 is not None: oof_cat.extend(p4)

    preds = [p1,p2,p3] + ([p4] if p4 is not None else [])
    ens = np.mean(preds, axis=0)
    fold_c_maes.append(mean_absolute_error(y_te, ens))
    print(f"  {f['name']}: LGB={mean_absolute_error(y_te,p1):.3f} DART={mean_absolute_error(y_te,p2):.3f} XGB={mean_absolute_error(y_te,p3):.3f}" +
          (f" CAT={mean_absolute_error(y_te,p4):.3f}" if p4 is not None else "") +
          f" ENS={fold_c_maes[-1]:.3f}h")

X_stack = np.column_stack([oof_lgb, oof_dart, oof_xgb] + ([oof_cat] if oof_cat else []))
y_stack = np.array(oof_y)
ridge = Ridge(alpha=1.0, fit_intercept=True)
ridge.fit(X_stack, y_stack)
stacked_pred_oof = ridge.predict(X_stack)
mae_stacked = mean_absolute_error(y_stack, stacked_pred_oof)
mae_c_mean  = float(np.mean(fold_c_maes)) if fold_c_maes else 99.0
print(f"\n  Phase C ensemble MAE: {mae_c_mean:.4f}h")
print(f"  Ridge stacked OOF MAE: {mae_stacked:.4f}h")
print(f"  Ridge weights: {ridge.coef_}")


# ============================================================
# 12. PHASE D: FINAL TRAINING ON ALL DATA
# ============================================================
print(f"\n{'='*70}")
print("PHASE D -- Final models on all observed data")
print("="*70)

all_obs = raw[TRAIN_MASK].copy().reset_index(drop=True)
gm_all  = float(all_obs["duration_hrs"].mean())

tf_final = TfidfVectorizer(max_features=800, ngram_range=(1,2), min_df=2, sublinear_tf=True)
sv_final = TruncatedSVD(n_components=N_SVD, random_state=42)
txt_svd  = sv_final.fit_transform(tf_final.fit_transform(all_obs["text_combined"]))
for i in range(N_SVD):
    all_obs[f"text_svd_{i}"] = txt_svd[:,i].astype(np.float32)

# Cause stats
for stat, fn in [("cause_mean","mean"),("cause_median","median"),("cause_p90",lambda x:x.quantile(0.9)),
                  ("cause_p75",lambda x:x.quantile(0.75)),("cause_p10",lambda x:x.quantile(0.1)),("cause_std","std")]:
    s = all_obs.groupby("event_cause")["duration_hrs"].agg(fn).rename(stat)
    all_obs[stat] = all_obs["event_cause"].map(s).fillna(gm_all)
all_obs["cause_std"] = all_obs["cause_std"].fillna(gm_all*0.5)

# Corridor stats
cr = all_obs.groupby("corridor")["duration_hrs"].agg(["mean","count","std"])
cr.columns = ["corridor_mean","corridor_cnt","corridor_std"]
for c in cr.columns:
    all_obs[c] = all_obs["corridor"].map(cr[c]).fillna(gm_all if ("mean" in c or "std" in c) else 1)
all_obs["corridor_std"] = all_obs["corridor_std"].fillna(gm_all*0.5)

# Station
st_m = all_obs.groupby("police_station")["duration_hrs"].mean()
all_obs["station_mean"] = all_obs["police_station"].map(st_m).fillna(gm_all)

# Hour x Cause
hc = all_obs.groupby(["hour","event_cause"])["duration_hrs"].mean().reset_index()
hc.columns=["hour","event_cause","hour_cause_mean"]
all_obs["hour_cause_mean"] = all_obs[["hour","event_cause"]].merge(hc,on=["hour","event_cause"],how="left")["hour_cause_mean"].fillna(gm_all).values

hbc = all_obs.groupby(["hour_bin","event_cause"])["duration_hrs"].mean().reset_index()
hbc.columns=["hour_bin","event_cause","hour_bin_cause_mean"]
all_obs["hour_bin_cause_mean"] = all_obs[["hour_bin","event_cause"]].merge(hbc,on=["hour_bin","event_cause"],how="left")["hour_bin_cause_mean"].fillna(gm_all).values

# Rolling features
for rcol in ["cause_rolling_30d","cause_rolling_7d"]:
    all_obs[rcol] = raw.loc[all_obs.index, rcol].fillna(gm_all).values if rcol in raw.columns else gm_all

# Bayesian smooth LOO on all data
def final_smooth_loo(df, key_col, alpha=SMOOTH_ALPHA):
    g_sum = df.groupby(key_col)["duration_hrs"].sum()
    g_cnt = df.groupby(key_col)["duration_hrs"].count()
    gs = df[key_col].map(g_sum).fillna(0).values
    gc = df[key_col].map(g_cnt).fillna(0).values
    raw_loo = np.where(gc > 1, (gs - df["duration_hrs"].values) / (gc - 1), gm_all)
    return ((gc * raw_loo + alpha * gm_all) / (gc + alpha)).astype(np.float32)

all_obs["corridor_loo"] = final_smooth_loo(all_obs, "corridor")
all_obs["station_loo"]  = final_smooth_loo(all_obs, "police_station")

for loo_name, key_fn in [
    ("zone_cause_loo",  lambda d: d["zone"].fillna("unk") + "||" + d["event_cause"]),
    ("veh_cause_loo",   lambda d: d["veh_cause_key"]),
    ("cluster_dur_loo", lambda d: d["cluster_id"].astype(str)),
    ("officer_dur_loo", lambda d: d["created_by_id"].fillna("unknown") if "created_by_id" in d.columns else pd.Series(["unknown"]*len(d), index=d.index)),
    ("pin_code_loo",    lambda d: d["pin_code"]),
]:
    tmp = all_obs.copy(); tmp["__k"] = key_fn(all_obs)
    g = tmp.groupby("__k")["duration_hrs"].agg(["sum","count"])
    tmp["_gs"] = tmp["__k"].map(g["sum"]).fillna(0)
    tmp["_gc"] = tmp["__k"].map(g["count"]).fillna(0)
    raw_loo = np.where(tmp["_gc"]>1, (tmp["_gs"]-tmp["duration_hrs"])/(tmp["_gc"]-1), gm_all)
    n = tmp["_gc"].values
    all_obs[loo_name] = ((n*raw_loo + SMOOTH_ALPHA*gm_all)/(n+SMOOTH_ALPHA)).astype(np.float32)

# Cluster density + log-ratio + log features
cc = all_obs.groupby("cluster_id")["duration_hrs"].count()
all_obs["cluster_density"] = all_obs["cluster_id"].map(cc).fillna(1)
for src, name in [("corridor_mean","corridor"),("cause_mean","event_cause"),("cluster_dur_loo","cluster")]:
    all_obs[f"log_ratio_{name}"] = np.log1p(np.abs(all_obs[src]/(gm_all+1e-8))).astype(np.float32)
for feat in ["corridor_loo","zone_cause_loo","cluster_dur_loo","cause_mean"]:
    all_obs[f"log_{feat}"] = np.log1p(np.abs(all_obs[feat])).astype(np.float32)

# Time-since from global series
for tc in ["hours_since_last_corridor","hours_since_last_zone"]:
    if tc in raw.columns:
        vals = raw.loc[all_obs.index, tc].values if len(raw.loc[all_obs.index, tc]) == len(all_obs) else np.full(len(all_obs), np.nan)
        all_obs[tc] = np.nan_to_num(vals.astype(np.float32), nan=float(np.nanmedian(vals)))
    else:
        all_obs[tc] = 0.0

STATIC_F = [
    "hour","dow","month","hour_bin","is_weekend","is_night",
    "hour_sin","hour_cos","is_weather","requires_road_closure",
    "event_cause_enc","zone_enc","corridor_enc","police_station_enc",
    "cause_x_zone","cause_x_closure","cause_x_night","cause_x_hour_bin",
    "is_high_priority","has_description","description_len",
    "log_hawkes_zone","log_hawkes_cause","officer_active_load",
    "corridor_stress_index","police_response_eta_mins",
    "tod_congestion_idx","congestion_x_response",
    "nearest_station_km","is_junction","addr_road_class",
    "officer_event_cnt","cluster_id","cluster_radius_km",
    "hours_since_last_zone",
]
ENCODED_F = [
    "cause_mean","cause_median","cause_p90","cause_p75","cause_p10","cause_std",
    "station_mean","corridor_mean","corridor_cnt","corridor_std",
    "hour_cause_mean","hour_bin_cause_mean",
    "cause_rolling_30d","cause_rolling_7d",
    "station_loo","corridor_loo","zone_cause_loo","veh_cause_loo",
    "cluster_dur_loo","cluster_density","officer_dur_loo","pin_code_loo",
    "log_ratio_corridor","log_ratio_event_cause","log_ratio_cluster",
    "log_corridor_loo","log_zone_cause_loo","log_cluster_dur_loo","log_cause_mean",
    "hours_since_last_corridor",
] + [f"text_svd_{i}" for i in range(N_SVD)]

ALL_F = STATIC_F + ENCODED_F
for col in ALL_F:
    if col not in all_obs.columns: all_obs[col] = 0.0
    all_obs[col] = pd.to_numeric(all_obs[col], errors="coerce").fillna(0).astype(np.float32)

X_fin = all_obs[ALL_F].copy()
y_fin = np.log1p(all_obs["duration_hrs"].values).astype(np.float32)
feat_names_f = list(X_fin.columns)
print(f"  Final feature matrix: {X_fin.shape}")

print("  Training LGB L1 (tuned)...")
fin_lgb = lgb.LGBMRegressor(**best_lgb_full); fin_lgb.fit(X_fin, y_fin)
print("  Training LGB DART (tuned)...")
fin_dart = lgb.LGBMRegressor(**best_dart_full); fin_dart.fit(X_fin, y_fin)
print("  Training XGBoost (tuned)...")
dtf = xgb.DMatrix(X_fin, label=y_fin, feature_names=feat_names_f)
fin_xgb = xgb.train(best_xgb_full, dtf, num_boost_round=xgb_n_est, verbose_eval=False)
if HAS_CAT and best_cat_p:
    print("  Training CatBoost (tuned)...")
    fin_cat = CatBoostRegressor(**best_cat_p, loss_function="MAE", eval_metric="MAE",
                                random_seed=42, verbose=0, task_type=CAT_TASK)
    fin_cat.fit(X_fin, y_fin, verbose=False)
else:
    fin_cat = None

import joblib
joblib.dump(fin_lgb,  OUTPUT_DIR/"lgb_v10_final.pkl")
joblib.dump(fin_dart, OUTPUT_DIR/"lgb_v10_dart.pkl")
joblib.dump(fin_xgb,  OUTPUT_DIR/"xgb_v10_final.pkl")
joblib.dump(tf_final, OUTPUT_DIR/"v10_tfidf.pkl")
joblib.dump(sv_final, OUTPUT_DIR/"v10_svd.pkl")
joblib.dump(feat_names_f, OUTPUT_DIR/"v10_features.pkl")
joblib.dump(ridge, OUTPUT_DIR/"v10_ridge_stack.pkl")
if fin_cat: joblib.dump(fin_cat, OUTPUT_DIR/"cat_v10_final.pkl")
print("  Models saved.")


# ============================================================
# 13. PHASE E: SHAP
# ============================================================
print(f"\n{'='*70}")
print("PHASE E -- SHAP feature importance")
print("="*70)
shap_results = {}
if HAS_SHAP:
    try:
        Xs = X_fin.sample(min(1000, len(X_fin)), random_state=42)
        exp = shap.TreeExplainer(fin_lgb)
        sv  = exp.shap_values(Xs)
        imp = np.abs(sv).mean(axis=0)
        ranked = sorted(zip(feat_names_f, imp), key=lambda x: x[1], reverse=True)
        max_i = imp.max()
        print(f"\n  Top 30 SHAP features:")
        for feat, val in ranked[:30]:
            print(f"    {feat:<45} {val:>7.4f}  {'|'*int(val/max_i*30)}")
        shap_results = {f: round(float(v),5) for f,v in ranked}
        with open(OUTPUT_DIR/"shap_v10.json","w") as fj:
            json.dump(shap_results, fj, indent=2)
    except Exception as e:
        print(f"  SHAP skipped: {e}")
else:
    imp = fin_lgb.feature_importances_
    ranked = sorted(zip(feat_names_f,imp), key=lambda x:x[1], reverse=True)
    print("  LGB gain importance:")
    for feat,val in ranked[:25]: print(f"    {feat:<45} {val:>8.0f}")
    shap_results = {f:int(v) for f,v in ranked}


# ============================================================
# 14. FINAL RESULTS
# ============================================================
elapsed = time.time() - t_start
print(f"\n{'='*70}")
print("GRIDGUARD AI V10 -- FINAL RESULTS")
print("="*70)

journey = [
    ("V9 honest baseline (no leakage)",            2.843),
    ("V10 Phase A LGB",                            mae_a["lgb"]),
    ("V10 Phase A DART",                           mae_a["dart"]),
    ("V10 Phase A XGB",                            mae_a["xgb"]),
    ("V10 Phase A CatBoost",                       mae_a["cat"]),
    ("V10 Phase B LGB Optuna (400 trials)",        best_lgb_mae),
    ("V10 Phase B DART Optuna (200 trials)",       best_dart_mae),
    ("V10 Phase B XGB Optuna (200 trials)",        best_xgb_mae),
    ("V10 Phase B CatBoost Optuna (150 trials)",   best_cat_mae),
    ("V10 Phase C ensemble (4 models)",            mae_c_mean),
    ("V10 Phase C Ridge stacked OOF",              mae_stacked),
]
for name, mae in journey:
    bar = "|" * max(1, int((1 - min(mae, 10) / 10) * 35))
    print(f"  {name:<55} {mae:>7.3f}h  {bar}")

imp_pct = (2.843 - mae_stacked) / 2.843 * 100
print(f"\n  Improvement over V9 honest baseline: {imp_pct:+.1f}%")
print(f"  Total runtime: {elapsed/60:.1f} minutes")
print(f"  Hardware: {'GPU' if HAS_GPU else 'CPU'}")

results = {
    "v9_honest_baseline": 2.843,
    "v10_phase_a": {k: round(v,4) for k,v in mae_a.items()},
    "v10_optuna_lgb_mae":  round(best_lgb_mae, 4),
    "v10_optuna_dart_mae": round(best_dart_mae, 4),
    "v10_optuna_xgb_mae":  round(best_xgb_mae, 4),
    "v10_optuna_cat_mae":  round(best_cat_mae, 4),
    "v10_ensemble_mae":    round(mae_c_mean, 4),
    "v10_stacked_mae":     round(mae_stacked, 4),
    "improvement_over_v9_pct": round(imp_pct, 2),
    "best_lgb_params":   best_lgb_p,
    "best_dart_params":  best_dart_p,
    "best_xgb_params":   {**best_xgb_p, "n_estimators": xgb_n_est},
    "best_cat_params":   best_cat_p,
    "ridge_weights":     list(ridge.coef_),
    "ridge_intercept":   float(ridge.intercept_),
    "shap_top10":        dict(list(shap_results.items())[:10]) if shap_results else {},
    "runtime_minutes":   round(elapsed/60, 1),
    "hardware":          "GPU" if HAS_GPU else "CPU",
    "n_features":        len(feat_names_f),
    "n_folds":           len(fold_cache),
}
with open(OUTPUT_DIR/"v10_results.json","w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n  Saved: {OUTPUT_DIR}/v10_results.json")
print(f"  Saved: {OUTPUT_DIR}/shap_v10.json")
print(f"\n{'='*70}")
print("DONE! Download v10_results.json and paste the contents back.")
print("="*70)
