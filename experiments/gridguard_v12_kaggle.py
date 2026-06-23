"""
GridGuard + EventFlow Prism V12 FINAL — ULTIMATE HYBRID
================================================================================
KAGGLE GPU T4 x1 | ~55 min runtime | Best possible MAE in 1 hour

ARCHITECTURE:
  5 base models: LGB L1 (GPU), LGB DART (GPU), CatBoost (GPU), HGB (CPU), ET (CPU)
  3 quantile models: Q35, Q50, Q65 via LGB GPU
  8-column GBM meta-stacker + Isotonic calibration
  PuLP prescriptive officer allocation

KEY IMPROVEMENTS OVER VANILLA V12:
  [1] LGB DART back -- proven best single model at 2.815h in V10
  [2] CatBoost GPU -- native categorical encoding, 5th diverse model
  [3] Dropped zero-SHAP features (is_weekend, requires_road_closure,
      has_description, is_junction, tod_congestion_idx)
  [4] 8-col meta-stacker vs 6-col -- richer blending signal
  [5] Trial budget: LGB=60, DART=30, CAT=30, HGB=15, ET=15 = 150 total
      All GPU models finish in ~20 min; CPU models ~30 min = ~55 min total
  [6] active_overlap_2km censored expiry fixed (DURATION_CAP not 1e9)
  [7] Phase D bug fixed (correct te_idx for all-data training)

KAGGLE SETUP:
  1. Upload astram_events.csv as dataset "gridlock-hackathon-data"
  2. New Notebook -> Add Data -> GPU T4 x1
  3. Paste ENTIRE script into ONE cell -> Run All
  4. Download v12_results.json + models when done
"""

import os, re, json, time, warnings, math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.cluster import DBSCAN
import lightgbm as lgb
import optuna
import joblib

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---- Optional dependencies ----
try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
    print("[OK] CatBoost available.")
except ImportError:
    HAS_CAT = False
    print("[WARN] CatBoost not installed -- will skip (pip install catboost).")

try:
    import pulp
    HAS_PULP = True
    print("[OK] PuLP available.")
except ImportError:
    HAS_PULP = False
    print("[WARN] PuLP not installed -- prescriptive phase will be skipped.")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import subprocess
    HAS_GPU = subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
except Exception:
    HAS_GPU = False

LGB_DEVICE = "gpu" if HAS_GPU else "cpu"
CAT_DEVICE = "GPU" if HAS_GPU else "CPU"
print(f"[HW] GPU={HAS_GPU} | LGB_DEVICE={LGB_DEVICE}")

# ============================================================
# PATHS
# ============================================================
if os.path.exists("/kaggle/input"):
    DATA_PATH = None
    for root, _, files in os.walk("/kaggle/input"):
        for fname in files:
            if fname.lower().endswith(".csv") and ("astram" in fname.lower() or "event" in fname.lower()):
                DATA_PATH = os.path.join(root, fname)
                break
        if DATA_PATH: break
    if DATA_PATH is None:
        raise FileNotFoundError("Upload astram_events.csv to Kaggle input first!")
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
DURATION_CAP    = 48.0
N_HASH          = 256
N_HASH_SVD      = 32
N_FOLDS         = 5
PURGE_HOURS     = 48
STEIN_PRIOR     = 10.0
EWMA_HALFLIFE_D = 14.0
KNN_K           = 8
OVERLAP_RADIUS_KM = 2.0
CENSORED_WEIGHT = 0.40
# Trial budget tuned for ~55 min on Kaggle GPU P100
N_OPTUNA_LGB  = 100  # GPU -- P100 is 2x faster than T4
N_OPTUNA_DART = 50   # GPU -- P100 is 2x faster than T4
N_OPTUNA_CAT  = 50   # GPU -- P100 is 2x faster than T4
N_OPTUNA_HGB  = 15   # CPU -- same speed on any Kaggle tier
N_OPTUNA_ET   = 15   # CPU -- same speed on any Kaggle tier
QUANTILE_ALPHAS = (0.35, 0.50, 0.65)

_total_trials = N_OPTUNA_LGB + N_OPTUNA_DART + N_OPTUNA_CAT + N_OPTUNA_HGB + N_OPTUNA_ET
print(f"\n{'='*75}")
print("GRIDGUARD + EVENTFLOW PRISM V12 FINAL — ULTIMATE HYBRID")
print(f"5 base models | {_total_trials} Optuna trials | 8-col GBM meta | PuLP prescriptive")
print(f"GPU={HAS_GPU} | Estimated runtime: ~55 min on Kaggle GPU T4")
print("="*75)


# ============================================================
# 1. LOAD & PREPROCESS
# ============================================================
print("\n[1] Loading data...")
raw = pd.read_csv(DATA_PATH, low_memory=False)
for col in ["start_datetime", "closed_datetime", "resolved_datetime"]:
    raw[col] = pd.to_datetime(raw[col], errors="coerce")
raw = raw[raw["start_datetime"].notna()].reset_index(drop=True)

def _to_int8(s, fill=0):
    return pd.to_numeric(s, errors="coerce").fillna(fill).astype(np.int8)

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

def _naive(s):
    try: return s.dt.tz_localize(None)
    except Exception:
        try: return s.dt.tz_convert("UTC").dt.tz_localize(None)
        except Exception: return s

raw["_sdt"]         = _naive(raw["start_datetime"])
raw["start_ts"]     = (raw["_sdt"].astype("int64") // 10**9).astype(np.float64)
raw["start_ts_hour"] = (raw["start_ts"] / 3600.0).astype(np.float32)

# Core flags (zero-SHAP features removed: is_weekend, requires_road_closure,
#              has_description, is_junction -- they add noise, not signal)
raw["is_planned"]      = _to_int8((raw.get("event_type", pd.Series("unplanned")) == "planned").astype(int))
raw["hour"]            = _to_int8(raw["start_datetime"].dt.hour, 12)
raw["dow"]             = _to_int8(raw["start_datetime"].dt.dayofweek, 2)
raw["month"]           = _to_int8(raw["start_datetime"].dt.month, 6)
raw["week_of_year"]    = _to_int8(raw["start_datetime"].dt.isocalendar().week.astype("Int64"), 26)
raw["is_night"]        = ((raw["hour"] >= 22) | (raw["hour"] <= 5)).astype(np.int8)
raw["is_rush"]         = raw["hour"].isin(list(range(8,11)) + list(range(17,21))).astype(np.int8)
raw["is_high_priority"] = _to_int8((raw.get("priority", "Low") == "High").astype(int))
raw["is_weather"]      = raw.get("event_cause", pd.Series("")).isin(
    ["water_logging", "tree_fall", "fog/low_visibility", "debris"]).astype(np.int8)

# Fourier harmonics (proven signal in V12)
for period, name, base_col in [(24, "hour", "hour"), (7, "dow", "dow"), (12, "month", "month"), (52, "week", "week_of_year")]:
    base = raw[base_col]
    raw[f"{name}_sin1"] = np.sin(2 * np.pi * base / period).astype(np.float32)
    raw[f"{name}_cos1"] = np.cos(2 * np.pi * base / period).astype(np.float32)
    raw[f"{name}_sin2"] = np.sin(4 * np.pi * base / period).astype(np.float32)
    raw[f"{name}_cos2"] = np.cos(4 * np.pi * base / period).astype(np.float32)

# Label encode
for col in ["event_cause", "zone", "corridor", "police_station"]:
    le = LabelEncoder()
    raw[col + "_enc"] = le.fit_transform(raw[col].fillna("unknown").astype(str)).astype(np.int16)

raw["veh_type_clean"]    = raw.get("veh_type", pd.Series("unknown")).fillna("unknown")
raw["veh_cause_key"]     = raw["veh_type_clean"] + "||" + raw.get("event_cause", pd.Series(""))
raw["zone_cause_key"]    = raw["zone"].fillna("unk") + "||" + raw.get("event_cause", pd.Series(""))
raw["corridor_cause_key"] = raw["corridor"].fillna("unk") + "||" + raw.get("event_cause", pd.Series(""))

# Text
_desc   = raw.get("description", pd.Series("")).fillna("").astype(str)
_reason = raw.get("reason_breakdown", pd.Series("")).fillna("").astype(str)
raw["text_combined"]  = (_desc.str.lower().str.replace(r"[^a-z0-9\s]", " ", regex=True) + " " +
                         _reason.str.lower().str.replace(r"[^a-z0-9\s]", " ", regex=True)).str.strip()
raw["description_len"] = _desc.str.len().clip(0, 500).astype(np.float32)

# Coords
raw["lat"] = raw["latitude"].fillna(raw["latitude"].median()).astype(np.float32)
raw["lon"] = raw["longitude"].fillna(raw["longitude"].median()).astype(np.float32)

# Address
def _extract_pin(addr):
    m = re.search(r"\b(\d{6})\b", str(addr))
    return m.group(1) if m else "000000"
def _road_class(addr):
    a = str(addr).lower()
    if any(k in a for k in ["nh-", "sh-", "highway", "expressway"]): return 3
    if any(k in a for k in ["main", "ring road", "outer ring"]): return 2
    return 1

if "address" in raw.columns:
    raw["pin_code"]       = raw["address"].apply(_extract_pin)
    raw["addr_road_class"] = _to_int8(raw["address"].apply(_road_class), 1)
else:
    raw["pin_code"]       = "000000"
    raw["addr_road_class"] = np.int8(1)

TRAIN_MASK = (raw["event_observed"] == 1) & (raw["duration_hrs"] <= DURATION_CAP)
gm_global  = float(raw.loc[TRAIN_MASK, "duration_hrs"].mean())
print(f"  N={len(raw)} | observed={TRAIN_MASK.sum()} | censored={len(raw)-TRAIN_MASK.sum()} | gm={gm_global:.3f}h")

# ============================================================
# GRAPH FEATURES — load precomputed centrality from corridor_graph_features.json
# Keys: centralities -> "corridor:<name>" -> {betweenness, closeness, degree}
#       corridor_stats -> "<name>" -> {impact_score, med_duration_hrs, closure_rate}
# Falls back to zeros if file not present (safe on Kaggle).
# ============================================================
_GRAPH_FEATURES_PATHS = [
    "data/precomputed/corridor_graph_features.json",          # local
    "/kaggle/input/gridlock-hackathon-data/corridor_graph_features.json",  # Kaggle upload
]
_gf_cent    = {}  # corridor_name -> betweenness, closeness
_gf_cstats  = {}  # corridor_name -> impact_score, med_duration_hrs
for _gfp in _GRAPH_FEATURES_PATHS:
    try:
        import json as _json
        with open(_gfp) as _f:
            _gf = _json.load(_f)
        # centralities keyed as "corridor:<name>"
        for k, v in _gf.get("centralities", {}).items():
            if k.startswith("corridor:"):
                _gf_cent[k[9:]] = v  # strip prefix
        _gf_cstats = _gf.get("corridor_stats", {})
        print(f"  [OK] Graph features loaded from {_gfp} | {len(_gf_cent)} corridors")
        break
    except FileNotFoundError:
        continue
    except Exception as _e:
        print(f"  [WARN] Graph features load error: {_e}")
        break

if _gf_cent:
    _corr_norm = raw["corridor"].fillna("").str.strip().str.lower()
    raw["graph_betweenness"]   = _corr_norm.map(lambda c: float(_gf_cent.get(c, {}).get("betweenness",  0.0))).astype(np.float32)
    raw["graph_closeness"]     = _corr_norm.map(lambda c: float(_gf_cent.get(c, {}).get("closeness",    0.0))).astype(np.float32)
    raw["graph_impact_score"]  = _corr_norm.map(lambda c: float(_gf_cstats.get(raw["corridor"].iloc[0] if False else c, {}).get("impact_score", 0.0))).astype(np.float32)
    # corridor_stats keys are Title-case; map again with original corridor values
    _cs_map = {k.strip().lower(): float(v.get("impact_score", 0.0)) for k, v in _gf_cstats.items()}
    raw["graph_impact_score"]  = _corr_norm.map(lambda c: _cs_map.get(c, 0.0)).astype(np.float32)
else:
    raw["graph_betweenness"]  = np.float32(0.0)
    raw["graph_closeness"]    = np.float32(0.0)
    raw["graph_impact_score"] = np.float32(0.0)
    print("  [WARN] Graph features not found — using zeros (won't hurt model)")


# ============================================================
# 2. GLOBAL CAUSAL FEATURES
# ============================================================
print("[2] Global causal features...")

def hawkes_intensity(df, group_col, alpha=0.8, beta=0.5, mu=0.05):
    result = np.full(len(df), mu, dtype=np.float64)
    ts = df["start_ts_hour"].values
    groups = df[group_col].values
    grp_idx = defaultdict(list)
    for i, g in enumerate(groups): grp_idx[g].append(i)
    for g, idxs in grp_idx.items():
        idxs = np.array(idxs)
        t_g  = ts[idxs]
        intensity = np.full(len(idxs), mu, dtype=np.float64)
        running = 0.0
        for k in range(1, len(idxs)):
            running = running * np.exp(-beta * (t_g[k] - t_g[k-1])) + alpha
            intensity[k] = mu + running
        result[idxs] = intensity
    return np.log1p(result).astype(np.float32)

raw["log_hawkes_cause"] = hawkes_intensity(raw, "event_cause")
raw["log_hawkes_zone"]  = hawkes_intensity(raw, "zone")

def ewma_by_group(df, group_col, val_col, halflife_days=EWMA_HALFLIFE_D):
    out = np.full(len(df), np.nan, dtype=np.float32)
    hl_sec = halflife_days * 86400.0
    ts = df["start_ts"].values; groups = df[group_col].values
    vals = df[val_col].values; obs = df["event_observed"].values
    grp_idx = defaultdict(list)
    for i, g in enumerate(groups): grp_idx[g].append(i)
    for _, idxs in grp_idx.items():
        idxs  = np.array(idxs)
        order = np.argsort(ts[idxs])
        idxs  = idxs[order]
        ema   = np.nan
        for k, i in enumerate(idxs):
            if k > 0 and np.isfinite(ema): out[i] = ema
            if obs[i] == 1:
                v = vals[i]
                if not np.isfinite(ema): ema = v
                else:
                    dt = max(ts[i] - ts[idxs[k-1]], 1.0)
                    alpha_ema = 1 - np.exp(-np.log(2) * dt / hl_sec)
                    ema = alpha_ema * v + (1 - alpha_ema) * ema
    return out

obs_dur = raw["duration_hrs"].where(raw["event_observed"] == 1)
raw["ewma_cause"]    = ewma_by_group(raw.assign(_dur=obs_dur), "event_cause", "_dur")
raw["ewma_corridor"] = ewma_by_group(raw.assign(_dur=obs_dur), "corridor",    "_dur")
raw["ewma_zone"]     = ewma_by_group(raw.assign(_dur=obs_dur), "zone",        "_dur")

def time_since_last(df, group_col):
    result = np.full(len(df), np.nan, dtype=np.float32)
    ts = df["start_ts_hour"].values; groups = df[group_col].values; last = {}
    for i, g in enumerate(groups):
        if g in last: result[i] = ts[i] - last[g]
        last[g] = ts[i]
    return result

raw["hours_since_last_corridor"] = time_since_last(raw, "corridor")
raw["hours_since_last_zone"]     = time_since_last(raw, "zone")

raw["officer_event_cnt"] = 0
if "created_by_id" in raw.columns:
    raw["officer_event_cnt"] = raw["created_by_id"].map(
        raw["created_by_id"].value_counts()).fillna(0).astype(np.int16).values

# Active Overlap (fixed: censored events expire at DURATION_CAP, not 1e9)
print("  Computing active_overlap_2km...")
def compute_active_overlap(df, radius_km=OVERLAP_RADIUS_KM):
    n = len(df)
    result = np.zeros(n, dtype=np.float32)
    if n < 2: return result
    sub = df.reset_index(drop=True)
    ts = sub["start_ts_hour"].values; lat = sub["lat"].values
    lon = sub["lon"].values; dur = sub["duration_hrs"].values
    obs = sub["event_observed"].values
    order = np.argsort(ts)
    active = []
    for qi in order:
        t_q = ts[qi]
        active = [(e, la, lo) for e, la, lo in active if e > t_q]
        cnt = 0
        for e, la, lo in active:
            dlat = np.radians(lat[qi] - la)
            dlon = np.radians(lon[qi] - lo)
            a    = np.sin(dlat/2)**2 + np.cos(np.radians(lat[qi])) * np.cos(np.radians(la)) * np.sin(dlon/2)**2
            if 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))) <= radius_km:
                cnt += 1
        result[qi] = cnt
        # FIX: censored events expire at DURATION_CAP (not 1e9) to avoid O(N^2) blowup
        end_t = ts[qi] + dur[qi] if obs[qi] == 1 else t_q + DURATION_CAP
        active.append((end_t, lat[qi], lon[qi]))
    return pd.Series(result, index=df.index).astype(np.float32)

_sorted = raw.sort_values("start_ts")
raw["active_overlap_2km"] = compute_active_overlap(_sorted).reindex(raw.index).fillna(0)


# ============================================================
# 3. SPATIAL (KNN + DBSCAN)
# ============================================================
print("[3] Spatial features...")
raw["lat_r"] = raw["lat"].round(3)
raw["lon_r"] = raw["lon"].round(3)

db = DBSCAN(eps=0.008, min_samples=3).fit(raw[["lat_r", "lon_r"]])
raw["cluster_id"] = db.labels_.astype(np.int16)
cluster_obs = raw[raw["cluster_id"] >= 0]
cstats = cluster_obs.groupby("cluster_id")["duration_hrs"].agg(["mean", "count"])
raw["cluster_dur_mean"] = raw["cluster_id"].map(cstats["mean"]).fillna(gm_global).astype(np.float32)
raw["cluster_count"]    = raw["cluster_id"].map(cstats["count"]).fillna(0).astype(np.float32)
print(f"  DBSCAN: {(db.labels_ >= 0).sum()} points in {db.labels_.max()+1} clusters")

# KNN spatial LOO for observed events
def knn_duration_loo(obs_df, k=KNN_K):
    out = np.full(len(obs_df), np.nan, dtype=np.float32)
    if len(obs_df) < k + 1: return out
    coords = obs_df[["lat", "lon"]].values
    durs   = obs_df["duration_hrs"].values
    tree   = cKDTree(coords)
    for i, (c, d) in enumerate(zip(coords, durs)):
        dists, idxs = tree.query(c, k=k+1)
        neighbors   = idxs[idxs != i][:k]
        if len(neighbors) > 0: out[i] = np.mean(durs[neighbors])
    return out

def knn_duration_features(tr_obs, te_df, k=KNN_K):
    out    = np.full(len(te_df), np.nan, dtype=np.float32)
    if len(tr_obs) < k: return out, None, None
    coords = tr_obs[["lat", "lon"]].values
    durs   = tr_obs["duration_hrs"].values
    tree   = cKDTree(coords)
    te_coords = te_df[["lat", "lon"]].values
    dists, idxs = tree.query(te_coords, k=k)
    out = np.mean(durs[idxs], axis=1).astype(np.float32)
    return out, tree, durs


# ============================================================
# 4. PURGED EXPANDING-WINDOW CV
# ============================================================
print("\n[4] Purged expanding-window CV setup...")
ts_all   = raw["start_ts"].values
n_total  = len(raw)
purge_sec = PURGE_HOURS * 3600
min_train = max(n_total // (N_FOLDS + 1), 500)
step      = (n_total - min_train) // N_FOLDS
fold_specs = []

for fi in range(N_FOLDS):
    train_end_idx  = min_train + fi * step
    test_start_idx = train_end_idx
    test_end_idx   = min(train_end_idx + step, n_total)
    if test_end_idx <= test_start_idx: continue
    tr_cutoff = ts_all[train_end_idx-1] - purge_sec if train_end_idx > 0 else 0
    tr_idx    = np.where(ts_all <= tr_cutoff)[0]
    te_idx    = np.arange(test_start_idx, test_end_idx)
    fold_specs.append({"name": f"Fold{fi+1}", "tr": tr_idx, "te": te_idx})

for f in fold_specs:
    print(f"  {f['name']}: train={len(f['tr'])} test={len(f['te'])} (purge {PURGE_HOURS}h)")


# ============================================================
# 5. IN-FOLD FEATURE BUILDER
# ============================================================
STATIC_COLS = [
    "hour", "dow", "month", "week_of_year", "is_night", "is_rush", "is_planned",
    "is_weather", "is_high_priority", "description_len",
    "addr_road_class", "event_cause_enc", "zone_enc", "corridor_enc", "police_station_enc",
    "officer_event_cnt", "cluster_id", "cluster_dur_mean", "cluster_count",
    "log_hawkes_cause", "log_hawkes_zone", "ewma_cause", "ewma_corridor", "ewma_zone",
    "hours_since_last_corridor", "hours_since_last_zone", "active_overlap_2km",
    "hour_sin1", "hour_cos1", "hour_sin2", "hour_cos2",
    "dow_sin1",  "dow_cos1",
    "month_sin1", "month_cos1", "week_sin1", "week_cos1",
    # Phase 2 graph centrality features (pre-computed from NetworkX corridor graph)
    "graph_betweenness", "graph_closeness", "graph_impact_score",
]

def stein_encode(tr_df, te_df, key_col, val_col, gm, prior=STEIN_PRIOR):
    g      = tr_df.groupby(key_col)[val_col]
    g_mean = g.mean(); g_cnt = g.count(); g_sum = g.sum()
    tr_cnt = tr_df[key_col].map(g_cnt).fillna(0).values
    tr_loo = np.where(tr_cnt > 1,
        (tr_df[key_col].map(g_sum).fillna(0).values - tr_df[val_col].values) / (tr_cnt - 1), gm)
    tr_final = (tr_cnt * tr_loo + prior * gm) / (tr_cnt + prior)
    te_mean  = te_df[key_col].map(g_mean).fillna(gm).values
    te_cnt   = te_df[key_col].map(g_cnt).fillna(0).values
    te_final = (te_cnt * te_mean + prior * gm) / (te_cnt + prior)
    return tr_final.astype(np.float32), te_final.astype(np.float32)

def build_fold_features(tr_idx, te_idx, hash_vec, gm):
    tr_all = raw.iloc[tr_idx].copy()
    te_all = raw.iloc[te_idx].copy()
    tr_obs = tr_all[TRAIN_MASK.iloc[tr_idx].values].copy()
    te_obs = te_all[TRAIN_MASK.iloc[te_idx].values].copy()
    if len(tr_obs) < 30 or len(te_obs) < 5: return None

    gm_fold = float(tr_obs["duration_hrs"].mean())
    ft, fe  = {}, {}

    # James-Stein target encoding
    for name, col in [("stein_cause",     "event_cause"),
                      ("stein_corridor",  "corridor"),
                      ("stein_zone_cause","zone_cause_key"),
                      ("stein_veh_cause", "veh_cause_key"),
                      ("stein_station",   "police_station"),
                      ("stein_pin",       "pin_code")]:
        _, fe[name] = stein_encode(tr_obs, te_obs, col, "duration_hrs", gm_fold)
        g_mean = tr_obs.groupby(col)["duration_hrs"].mean()
        ft[name] = tr_all[col].map(g_mean).fillna(gm_fold).values.astype(np.float32)

    # Cause stats
    for stat, fn in [("cause_mean", "mean"), ("cause_p90", lambda x: x.quantile(0.9))]:
        s = tr_obs.groupby("event_cause")["duration_hrs"].agg(fn).rename(stat)
        ft[stat] = tr_all["event_cause"].map(s).fillna(gm_fold).values.astype(np.float32)
        fe[stat] = te_obs["event_cause"].map(s).fillna(gm_fold).values.astype(np.float32)

    # KNN spatial
    km_tr        = knn_duration_loo(tr_obs)
    km_te, _, _  = knn_duration_features(tr_obs, te_obs)
    ft["knn_dur_loo"] = km_tr
    fe["knn_dur_loo"] = km_te

    # HashingVectorizer + SVD on text
    svd = TruncatedSVD(n_components=N_HASH_SVD, random_state=42)
    H_tr = hash_vec.transform(tr_all["text_combined"].fillna(""))
    H_te = hash_vec.transform(te_obs["text_combined"].fillna(""))
    T_tr = svd.fit_transform(H_tr)
    T_te = svd.transform(H_te)
    for k in range(N_HASH_SVD):
        ft[f"hash_svd_{k}"] = T_tr[:, k].astype(np.float32)
        fe[f"hash_svd_{k}"] = T_te[:, k].astype(np.float32)

    def assemble(df, fdict):
        parts = []
        for c in STATIC_COLS:
            if c in df.columns:
                parts.append(df[c].values.astype(np.float32))
            else:
                parts.append(np.zeros(len(df), dtype=np.float32))
        for k, v in fdict.items():
            arr = np.array(v, dtype=np.float32)
            if len(arr) != len(df): arr = np.full(len(df), np.nan, dtype=np.float32)
            parts.append(arr)
        cols = STATIC_COLS + list(fdict.keys())
        return pd.DataFrame(np.column_stack(parts), columns=cols).fillna(0)

    X_tr = assemble(tr_all.reset_index(drop=True), ft)
    X_te = assemble(te_obs.reset_index(drop=True), fe)
    # Train on log(1+y) to handle right-skewed durations
    y_tr_log = np.log1p(tr_all["duration_hrs"].clip(0.05, DURATION_CAP).values).astype(np.float32)
    w_tr     = np.where(tr_all["event_observed"].values == 1, 1.0, CENSORED_WEIGHT).astype(np.float32)
    y_te     = te_obs["duration_hrs"].values.astype(np.float32)
    return X_tr, y_tr_log, w_tr, X_te, y_te, list(X_tr.columns)

hash_vec = HashingVectorizer(n_features=N_HASH, alternate_sign=False, ngram_range=(1, 2))

print("\n[5] Building fold cache...")
fold_cache = []
for spec in fold_specs:
    res = build_fold_features(spec["tr"], spec["te"], hash_vec, gm_global)
    if res is None: continue
    X_tr, y_tr, w_tr, X_te, y_te, fn = res
    if len(X_tr) < 50 or len(X_te) < 5: continue
    fold_cache.append((X_tr, y_tr, w_tr, X_te, y_te, fn, spec["name"]))
print(f"  Cached {len(fold_cache)} folds | features: {len(fold_cache[0][5]) if fold_cache else 0}")


# ============================================================
# 6. PHASE B: OPTUNA TUNING
# ============================================================
print(f"\n{'='*75}")
print(f"PHASE B — Optuna Tuning ({N_OPTUNA_LGB} LGB, {N_OPTUNA_DART} DART, {N_OPTUNA_CAT} CAT, {N_OPTUNA_HGB} HGB, {N_OPTUNA_ET} ET)")
print("="*75)

def _cv_mae(predict_fn):
    maes = []
    for X_tr, y_tr, w_tr, X_te, y_te, fn, _ in fold_cache:
        pred = predict_fn(X_tr, y_tr, w_tr, X_te)
        maes.append(mean_absolute_error(y_te, np.expm1(np.maximum(pred, 0))))
    return float(np.mean(maes))

# --- LGB L1 ---
def obj_lgb(trial):
    p = {
        "objective": "regression_l1", "device": LGB_DEVICE, "verbosity": -1, "n_jobs": -1, "random_state": 42,
        "n_estimators":      trial.suggest_int("n_estimators",      200, 1500),
        "learning_rate":     trial.suggest_float("learning_rate",    1e-3, 5e-2, log=True),
        "num_leaves":        trial.suggest_int("num_leaves",         31, 255),
        "min_child_samples": trial.suggest_int("min_child_samples",  5, 100),
        "subsample":         trial.suggest_float("subsample",        0.4, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha",        1e-8, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda",       1e-8, 10.0, log=True),
    }
    return _cv_mae(lambda Xtr, ytr, wtr, Xte: lgb.LGBMRegressor(**p).fit(Xtr, ytr, sample_weight=wtr).predict(Xte))

study_lgb = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=20))
study_lgb.optimize(obj_lgb, n_trials=N_OPTUNA_LGB, show_progress_bar=False)
best_lgb_p = study_lgb.best_params
print(f"  LGB  best: {study_lgb.best_value:.4f}h")

# --- LGB DART (proven best single model at 2.815h in V10) ---
def obj_dart(trial):
    p = {
        "objective": "regression_l1", "boosting_type": "dart", "device": LGB_DEVICE, "verbosity": -1,
        "n_jobs": -1, "random_state": 42,
        "n_estimators":      trial.suggest_int("n_estimators",      300, 1500),
        "learning_rate":     trial.suggest_float("learning_rate",   1e-3, 5e-2, log=True),
        "num_leaves":        trial.suggest_int("num_leaves",        31, 255),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample":         trial.suggest_float("subsample",       0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.3, 0.9),
        "reg_alpha":         trial.suggest_float("reg_alpha",       1e-8, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda",      1e-8, 10.0, log=True),
        "drop_rate":         trial.suggest_float("drop_rate",       0.1, 0.5),
        "skip_drop":         trial.suggest_float("skip_drop",       0.3, 0.7),
    }
    return _cv_mae(lambda Xtr, ytr, wtr, Xte: lgb.LGBMRegressor(**p).fit(Xtr, ytr, sample_weight=wtr).predict(Xte))

study_dart = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=45, n_startup_trials=10))
study_dart.optimize(obj_dart, n_trials=N_OPTUNA_DART, show_progress_bar=False)
best_dart_p = study_dart.best_params
print(f"  DART best: {study_dart.best_value:.4f}h")

# --- CatBoost (GPU, native categorical signal) ---
if HAS_CAT:
    def obj_cat(trial):
        p = {
            "loss_function": "MAE", "task_type": CAT_DEVICE, "verbose": 0, "random_seed": 42,
            "iterations":         trial.suggest_int("iterations",         200, 800),
            "learning_rate":      trial.suggest_float("learning_rate",    1e-3, 5e-2, log=True),
            "depth":              trial.suggest_int("depth",              4, 10),
            "l2_leaf_reg":        trial.suggest_float("l2_leaf_reg",      1e-1, 20.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
            "random_strength":    trial.suggest_float("random_strength",  0.5, 5.0),
        }
        return _cv_mae(lambda Xtr, ytr, wtr, Xte: CatBoostRegressor(**p).fit(Xtr, ytr, sample_weight=wtr).predict(Xte))

    study_cat = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=46, n_startup_trials=10))
    study_cat.optimize(obj_cat, n_trials=N_OPTUNA_CAT, show_progress_bar=False)
    best_cat_p = study_cat.best_params
    print(f"  CAT  best: {study_cat.best_value:.4f}h")
else:
    best_cat_p = None
    print("  CAT  skipped (not installed)")

# --- HGB (CPU) ---
def obj_hgb(trial):
    p = {
        "loss": "absolute_error", "random_state": 42,
        "max_iter":        trial.suggest_int("max_iter",         100, 800),
        "learning_rate":   trial.suggest_float("learning_rate",  1e-3, 5e-2, log=True),
        "max_leaf_nodes":  trial.suggest_int("max_leaf_nodes",   15, 127),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 100),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-8, 10.0, log=True),
    }
    return _cv_mae(lambda Xtr, ytr, wtr, Xte: HistGradientBoostingRegressor(**p).fit(Xtr, ytr, sample_weight=wtr).predict(Xte))

study_hgb = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=43, n_startup_trials=8))
study_hgb.optimize(obj_hgb, n_trials=N_OPTUNA_HGB, show_progress_bar=False)
best_hgb_p = study_hgb.best_params
print(f"  HGB  best: {study_hgb.best_value:.4f}h")

# --- ExtraTrees (CPU) ---
def obj_et(trial):
    p = {
        "n_estimators":    trial.suggest_int("n_estimators",    100, 500),
        "max_depth":       trial.suggest_int("max_depth",       5, 25),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 20),
        "max_features":    trial.suggest_float("max_features",  0.3, 0.9),
        "random_state": 42, "n_jobs": -1
    }
    return _cv_mae(lambda Xtr, ytr, wtr, Xte: ExtraTreesRegressor(**p).fit(Xtr, ytr, sample_weight=wtr).predict(Xte))

study_et = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=44, n_startup_trials=6))
study_et.optimize(obj_et, n_trials=N_OPTUNA_ET, show_progress_bar=False)
best_et_p = study_et.best_params
print(f"  ET   best: {study_et.best_value:.4f}h")


# ============================================================
# 7. PHASE C: OOF 8-COLUMN STACKING & ISOTONIC META-LEARNER
# ============================================================
print(f"\n{'='*75}")
print("PHASE C — 8-col OOF Stacking (LGB+DART+CAT+HGB+ET+Q35+Q50+Q65) + GBM Meta + Isotonic")
print("="*75)

best_lgb_full  = dict(best_lgb_p);  best_lgb_full.update( {"objective": "regression_l1", "device": LGB_DEVICE, "verbosity": -1, "n_jobs": -1, "random_state": 42})
best_dart_full = dict(best_dart_p); best_dart_full.update({"objective": "regression_l1", "boosting_type": "dart", "device": LGB_DEVICE, "verbosity": -1, "n_jobs": -1, "random_state": 42})
best_hgb_full  = dict(best_hgb_p);  best_hgb_full.update( {"loss": "absolute_error", "random_state": 42})
best_et_full   = dict(best_et_p);   best_et_full.update(  {"random_state": 42, "n_jobs": -1})
if HAS_CAT and best_cat_p:
    best_cat_full = dict(best_cat_p); best_cat_full.update({"loss_function": "MAE", "task_type": CAT_DEVICE, "verbose": 0, "random_seed": 42})

oof_lgb  = []; oof_dart = []; oof_cat  = []; oof_hgb  = []; oof_et   = []
oof_q35  = []; oof_q50  = []; oof_q65  = []; oof_y    = []

for X_tr, y_tr, w_tr, X_te, y_te, fn, fname in fold_cache:
    p_lgb  = np.expm1(np.maximum(lgb.LGBMRegressor(**best_lgb_full).fit(X_tr, y_tr, sample_weight=w_tr).predict(X_te), 0))
    p_dart = np.expm1(np.maximum(lgb.LGBMRegressor(**best_dart_full).fit(X_tr, y_tr, sample_weight=w_tr).predict(X_te), 0))
    p_hgb  = np.expm1(np.maximum(HistGradientBoostingRegressor(**best_hgb_full).fit(X_tr, y_tr, sample_weight=w_tr).predict(X_te), 0))
    p_et   = np.expm1(np.maximum(ExtraTreesRegressor(**best_et_full).fit(X_tr, y_tr, sample_weight=w_tr).predict(X_te), 0))
    p_q35  = np.expm1(np.maximum(lgb.LGBMRegressor(objective="quantile", alpha=0.35, n_estimators=300, device=LGB_DEVICE, verbosity=-1).fit(X_tr, y_tr, sample_weight=w_tr).predict(X_te), 0))
    p_q50  = np.expm1(np.maximum(lgb.LGBMRegressor(objective="quantile", alpha=0.50, n_estimators=300, device=LGB_DEVICE, verbosity=-1).fit(X_tr, y_tr, sample_weight=w_tr).predict(X_te), 0))
    p_q65  = np.expm1(np.maximum(lgb.LGBMRegressor(objective="quantile", alpha=0.65, n_estimators=300, device=LGB_DEVICE, verbosity=-1).fit(X_tr, y_tr, sample_weight=w_tr).predict(X_te), 0))

    if HAS_CAT and best_cat_p:
        p_cat = np.expm1(np.maximum(CatBoostRegressor(**best_cat_full).fit(X_tr, y_tr, sample_weight=w_tr).predict(X_te), 0))
    else:
        p_cat = (p_lgb + p_dart) / 2  # fallback if no catboost

    oof_lgb.extend(p_lgb); oof_dart.extend(p_dart); oof_cat.extend(p_cat)
    oof_hgb.extend(p_hgb); oof_et.extend(p_et)
    oof_q35.extend(p_q35); oof_q50.extend(p_q50); oof_q65.extend(p_q65)
    oof_y.extend(y_te)

    ens = np.mean([p_lgb, p_dart, p_q50], axis=0)
    print(f"  {fname}: LGB={mean_absolute_error(y_te, p_lgb):.3f} DART={mean_absolute_error(y_te, p_dart):.3f} "
          f"CAT={mean_absolute_error(y_te, p_cat):.3f} Q50={mean_absolute_error(y_te, p_q50):.3f} ENS={mean_absolute_error(y_te, ens):.3f}h")

# 8-column meta-stacker input
X_stack = np.column_stack([oof_lgb, oof_dart, oof_cat, oof_hgb, oof_et, oof_q35, oof_q50, oof_q65])
y_stack = np.array(oof_y)

# GBM Meta-blender
gbm_meta = GradientBoostingRegressor(loss="absolute_error", n_estimators=150, max_depth=3, random_state=42)
gbm_meta.fit(X_stack, y_stack)
stacked_preds = gbm_meta.predict(X_stack)

# Isotonic Calibration
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(stacked_preds, y_stack)
final_oof_calibrated = iso.transform(stacked_preds)

mae_stack = mean_absolute_error(y_stack, final_oof_calibrated)
mae_simple_ens = mean_absolute_error(y_stack, np.mean([oof_lgb, oof_dart, oof_q50], axis=0))
print(f"\n  Simple DART+LGB+Q50 ensemble MAE : {mae_simple_ens:.4f}h")
print(f"  GBM Stacked + Isotonic MAE       : {mae_stack:.4f}h")
# Use whichever is better
best_mae = min(mae_stack, mae_simple_ens)
use_stack = mae_stack <= mae_simple_ens
print(f"  Using: {'GBM Stack' if use_stack else 'Simple Ensemble'} (MAE={best_mae:.4f}h)")


# ============================================================
# 8. PHASE D: FINAL MODELS ON ALL DATA
# ============================================================
print(f"\n{'='*75}")
print("PHASE D — Final Models on All Data")
print("="*75)

# FIX: pass all indices to get a valid full-data feature matrix
X_all, y_all, w_all, _, _, final_cols = build_fold_features(
    np.arange(len(raw)), np.arange(len(raw)), hash_vec, gm_global)

print("  Training final base models...")
final_lgb  = lgb.LGBMRegressor(**best_lgb_full).fit(X_all, y_all, sample_weight=w_all)
final_dart = lgb.LGBMRegressor(**best_dart_full).fit(X_all, y_all, sample_weight=w_all)
final_hgb  = HistGradientBoostingRegressor(**best_hgb_full).fit(X_all, y_all, sample_weight=w_all)
final_et   = ExtraTreesRegressor(**best_et_full).fit(X_all, y_all, sample_weight=w_all)
final_q35  = lgb.LGBMRegressor(objective="quantile", alpha=0.35, n_estimators=300, device=LGB_DEVICE, verbosity=-1).fit(X_all, y_all, sample_weight=w_all)
final_q50  = lgb.LGBMRegressor(objective="quantile", alpha=0.50, n_estimators=300, device=LGB_DEVICE, verbosity=-1).fit(X_all, y_all, sample_weight=w_all)
final_q65  = lgb.LGBMRegressor(objective="quantile", alpha=0.65, n_estimators=300, device=LGB_DEVICE, verbosity=-1).fit(X_all, y_all, sample_weight=w_all)

if HAS_CAT and best_cat_p:
    final_cat = CatBoostRegressor(**best_cat_full).fit(X_all, y_all, sample_weight=w_all)
else:
    final_cat = None

joblib.dump(final_lgb,  OUTPUT_DIR / "v12_lgb.pkl")
joblib.dump(final_dart, OUTPUT_DIR / "v12_dart.pkl")
joblib.dump(gbm_meta,   OUTPUT_DIR / "v12_gbm_meta.pkl")
joblib.dump(iso,        OUTPUT_DIR / "v12_isotonic.pkl")
print("  Models saved.")

# SHAP on final LGB
if HAS_SHAP:
    try:
        exp = shap.TreeExplainer(final_lgb)
        sv  = exp.shap_values(X_all.sample(min(1000, len(X_all)), random_state=42))
        imp = np.abs(sv).mean(axis=0)
        ranked = sorted(zip(final_cols, imp), key=lambda x: x[1], reverse=True)
        print("\n  Top 20 SHAP features:")
        for f, v in ranked[:20]: print(f"    {f:<40} {v:.4f}")
        shap_dict = {f: round(float(v), 5) for f, v in ranked}
        with open(OUTPUT_DIR / "shap_v12.json", "w") as fp:
            json.dump(shap_dict, fp, indent=2)
    except Exception as e:
        print(f"  SHAP Error: {e}")


# ============================================================
# 9. PHASE E: PRESCRIPTIVE OFFICER ALLOCATION (PuLP)
# ============================================================
if HAS_PULP:
    print(f"\n{'='*75}")
    print("PHASE E — Prescriptive Traffic Allocation (PuLP)")
    print("="*75)

    try:
        recent_te_idx = np.arange(max(0, len(raw) - 20), len(raw))
        recent_tr_idx = np.arange(len(raw) - 20)
        res = build_fold_features(recent_tr_idx, recent_te_idx, hash_vec, gm_global)
        if res is not None:
            recent_X, _, _, _, _, _ = res
            recent = raw.iloc[recent_te_idx].copy().reset_index(drop=True)

            def _pred(m):
                return np.expm1(np.maximum(m.predict(recent_X), 0))

            r_lgb  = _pred(final_lgb)
            r_dart = _pred(final_dart)
            r_q50  = _pred(final_q50)
            r_hgb  = np.expm1(np.maximum(final_hgb.predict(recent_X), 0))
            r_et   = np.expm1(np.maximum(final_et.predict(recent_X), 0))
            r_q35  = _pred(final_q35)
            r_q65  = _pred(final_q65)
            r_cat  = _pred(final_cat) if final_cat is not None else (r_lgb + r_dart) / 2

            r_stack = np.column_stack([r_lgb, r_dart, r_cat, r_hgb, r_et, r_q35, r_q50, r_q65])
            if use_stack:
                r_preds = iso.transform(gbm_meta.predict(r_stack))
            else:
                r_preds = np.mean([r_lgb, r_dart, r_q50], axis=0)

            recent["predicted_duration"]  = r_preds
            recent["priority_weight"]     = recent["is_high_priority"].map({1: 2.5, 0: 1.0})
            recent["overlap_weight"]      = (1 + recent["active_overlap_2km"] * 0.2)
            recent["impact_score"]        = (
                recent["predicted_duration"] *
                recent["priority_weight"] *
                recent["overlap_weight"]
            )

            TOTAL_OFFICERS = 8
            prob = pulp.LpProblem("Officer_Allocation", pulp.LpMaximize)
            x    = pulp.LpVariable.dicts("deploy", range(len(recent)), cat="Binary")
            prob += pulp.lpSum(x[i] * recent.loc[i, "impact_score"] * 0.35 for i in range(len(recent)))
            prob += pulp.lpSum(x[i] for i in range(len(recent))) <= TOTAL_OFFICERS
            prob.solve(pulp.PULP_CBC_CMD(msg=0))

            deployed = [i for i in range(len(recent)) if x[i].varValue == 1.0]
            print(f"  PuLP Status: {pulp.LpStatus[prob.status]}")
            print(f"  Deploying {len(deployed)} of {TOTAL_OFFICERS} officers to top-impact events:\n")
            for i in deployed:
                r = recent.iloc[i]
                print(f"   -> Zone: {r['zone']:<20} Cause: {r['event_cause']:<25} "
                      f"PredDur: {r['predicted_duration']:.2f}h  Impact: {r['impact_score']:.2f}")
        else:
            print("  [WARN] Not enough recent data for PuLP demo.")
    except Exception as e:
        print(f"  [ERROR] PuLP phase failed: {e}")


# ============================================================
# 10. SAVE RESULTS
# ============================================================
runtime_min = (time.time() - t_start) / 60

results = {
    "v9_honest_baseline": 2.843,
    "v10_dart_best":      2.815,
    "v12_simple_ens_mae": round(mae_simple_ens, 4),
    "v12_stacked_mae":    round(mae_stack, 4),
    "v12_best_mae":       round(best_mae, 4),
    "improvement_over_v9_pct": round((best_mae - 2.843) / 2.843 * 100, 2),
    "improvement_over_v10_dart_pct": round((best_mae - 2.815) / 2.815 * 100, 2),
    "best_lgb_params":  best_lgb_p,
    "best_dart_params": best_dart_p,
    "best_hgb_params":  best_hgb_p,
    "best_et_params":   best_et_p,
    "best_cat_params":  best_cat_p if HAS_CAT else None,
    "runtime_minutes":  round(runtime_min, 1),
    "n_folds":          len(fold_cache),
    "n_features":       len(final_cols),
    "hardware":         "GPU" if HAS_GPU else "CPU",
    "models": ["LGB_L1", "LGB_DART", "CatBoost", "HGB", "ExtraTrees", "Q35", "Q50", "Q65"],
}

with open(OUTPUT_DIR / "v12_results.json", "w") as fp:
    json.dump(results, fp, indent=2)

print(f"\n{'='*75}")
print("GRIDGUARD V12 FINAL — COMPLETE")
print(f"  V9 honest baseline    : 2.843h")
print(f"  V10 DART best         : 2.815h")
print(f"  V12 simple ensemble   : {mae_simple_ens:.4f}h")
print(f"  V12 GBM stack+isotonic: {mae_stack:.4f}h")
print(f"  V12 BEST              : {best_mae:.4f}h  ({results['improvement_over_v9_pct']:+.1f}% vs V9)")
print(f"  Runtime               : {runtime_min:.1f} min | Hardware: {'GPU' if HAS_GPU else 'CPU'}")
print(f"  Saved: v12_results.json | shap_v12.json | models/*.pkl")
print("="*75)
print("\nDONE! Download v12_results.json and paste back for analysis.")
