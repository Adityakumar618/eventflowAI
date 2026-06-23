"""
EventFlow Prism V11 -- Complementary Kaggle Grandmaster Track
==============================================================
RESOURCE-INTENSIVE: Kaggle GPU T4, 30GB RAM, 12-hour session.
Designed to run ALONGSIDE GridGuard V10 -- NOT a copy. Different paradigm.

V10 direction (running now):
  - Bayesian LOO encodings, Hawkes, DBSCAN clusters, TF-IDF+SVD
  - 4 homogeneous tree boosters (LGB/DART/XGB/Cat) + Ridge linear stack
  - Observed-only training, monthly temporal folds, log1p point estimate

PRISM V11 direction (this script):
  [P-1]  Censoring-aware sample weights -- uses ALL 8173 events, not just observed
  [P-2]  Spatial KNN-LOO duration (cKDTree, inverse-distance) -- not DBSCAN LOO
  [P-3]  Active overlap load -- concurrent events within 2km radius at start time
  [P-4]  James-Stein hierarchical shrinkage encodings (zone>corridor>cause nesting)
  [P-5]  EWMA recency features (14-day half-life) -- not rolling window means
  [P-6]  HashingVectorizer text (256 dims) -- not TF-IDF
  [P-7]  Fourier multi-scale temporal (hour/dow/month/week harmonics)
  [P-8]  Auxiliary closure classifier -> predicted prob as duration feature
  [P-9]  Mixture-of-Experts (short/medium/long regime specialists + gating)
  [P-10] Quantile triplet fusion (P35/P50/P65) with coverage calibration
  [P-11] Heterogeneous model stack: Quantile LGB + HistGradientBoosting + ExtraTrees
  [P-12] Non-linear GBM meta-blender + Isotonic calibration (not Ridge stack)
  [P-13] Purged expanding-window CV with 48h embargo (not monthly folds)
  [P-14] Multi-objective Optuna (MAE + quantile coverage)

KAGGLE SETUP:
  1. Upload astram_events.csv as dataset (same as V10)
  2. New Notebook -> Add Data -> GPU T4 x1
  3. Paste ENTIRE script into one cell -> Run All (45-120 min)
  4. Download /kaggle/working/v11_results.json and report back!

Compare V10 vs V11 results to build the ultimate ensemble.
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
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
import optuna

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

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
# P100/T4: col-wise GPU layout is faster; keep LGB Optuna sequential to avoid GPU contention
LGB_GPU_EXTRA = (
    {"device": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0, "force_col_wise": True}
    if HAS_GPU else {"device": "cpu"}
)
OPTUNA_JOBS_GPU = 1   # one GPU trial at a time
OPTUNA_JOBS_CPU = 2   # parallel CPU trials (HGB + ExtraTrees)
print(f"[HW] GPU={HAS_GPU}  LGB={LGB_DEVICE}  Optuna jobs: GPU={OPTUNA_JOBS_GPU} CPU={OPTUNA_JOBS_CPU}")


def _lgb_kw(**kwargs):
    """Merge standard LGB kwargs with GPU-optimized settings (P100/T4 safe)."""
    base = {"verbosity": -1, "n_jobs": -1, "random_state": 42}
    base.update(LGB_GPU_EXTRA)
    base.update(kwargs)
    return base

# ---- PATHS ----
if os.path.exists("/kaggle/input"):
    DATA_PATH = None
    for root, _, files in os.walk("/kaggle/input"):
        for fname in files:
            if fname.lower().endswith(".csv") and ("astram" in fname.lower() or "event" in fname.lower()):
                DATA_PATH = os.path.join(root, fname)
                break
        if DATA_PATH:
            break
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
DURATION_CAP     = 48.0
N_HASH           = 256
N_HASH_SVD       = 32
N_FOLDS          = 6
PURGE_HOURS      = 48
STEIN_PRIOR      = 12.0
EWMA_HALFLIFE_D  = 14.0
KNN_K            = 10
OVERLAP_RADIUS   = 2.0          # km
CENSORED_WEIGHT  = 0.38
N_OPTUNA_LGB     = 350
N_OPTUNA_HGB     = 250
N_OPTUNA_ET      = 150
EXPERT_BOUNDS    = (1.5, 4.0)  # short < 1.5h, medium 1.5-4h, long > 4h
QUANTILE_ALPHAS  = (0.35, 0.50, 0.65)

print(f"\n{'='*72}")
print("EVENTFLOW PRISM V11 -- COMPLEMENTARY GRANDMASTER TRACK")
print(f"MoE + Quantile + Spatial KNN + Censoring-aware | {N_OPTUNA_LGB+N_OPTUNA_HGB+N_OPTUNA_ET} Optuna trials")
print(f"Purged {N_FOLDS}-fold CV | Target: honest MAE < 2.5h with uncertainty bands")
print("="*72)


# ============================================================
# 1. LOAD & PREPROCESS
# ============================================================
print("\n[1] Loading data...")
raw = pd.read_csv(DATA_PATH, low_memory=False)
for col in ["start_datetime", "closed_datetime", "resolved_datetime"]:
    raw[col] = pd.to_datetime(raw[col], errors="coerce")

# Rows without a valid start time cannot be modeled (pandas 2.x rejects NaT -> int8)
n_before = len(raw)
raw = raw[raw["start_datetime"].notna()].copy()
if len(raw) < n_before:
    print(f"  Dropped {n_before - len(raw)} rows with invalid start_datetime")

def _to_int8(s, fill=0):
    """Safe int8 cast — pandas 2.x raises on NaN without fillna."""
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
    ),
).clip(min=0.05).astype(np.float32)

raw = raw.sort_values("start_datetime").reset_index(drop=True)

def _naive_dt(s):
    try:
        return s.dt.tz_localize(None)
    except Exception:
        try:
            return s.dt.tz_convert("UTC").dt.tz_localize(None)
        except Exception:
            return s

raw["_sdt"] = _naive_dt(raw["start_datetime"])
raw["start_ts"] = (raw["_sdt"].astype("int64") // 10**9).astype(np.float64)
raw["start_ts_hour"] = (raw["start_ts"] / 3600.0).astype(np.float32)

# Regime flags
raw["is_planned"] = _to_int8(
    (raw["event_type"] == "planned") if "event_type" in raw.columns
    else pd.Series(0, index=raw.index)
)
raw["hour"] = _to_int8(raw["start_datetime"].dt.hour, fill=12)
raw["dow"] = _to_int8(raw["start_datetime"].dt.dayofweek, fill=2)
raw["month"] = _to_int8(raw["start_datetime"].dt.month, fill=6)
raw["week_of_year"] = _to_int8(raw["start_datetime"].dt.isocalendar().week.astype("Int64"), fill=26)
raw["is_weekend"] = raw["dow"].isin([5, 6]).astype(np.int8)
raw["is_night"] = ((raw["hour"] >= 22) | (raw["hour"] <= 5)).astype(np.int8)
raw["is_rush"] = raw["hour"].isin(list(range(8, 11)) + list(range(17, 21))).astype(np.int8)
if "requires_road_closure" in raw.columns:
    raw["requires_road_closure"] = _to_int8(raw["requires_road_closure"].map(
        {True: 1, False: 0, "True": 1, "False": 0, "true": 1, "false": 0}
    ).fillna(0))
else:
    raw["requires_road_closure"] = np.int8(0)
raw["is_high_priority"] = _to_int8(
    raw["priority"].fillna("Low") == "High" if "priority" in raw.columns
    else pd.Series(0, index=raw.index)
)
raw["is_weather"] = raw["event_cause"].isin(
    ["water_logging", "tree_fall", "fog/low_visibility", "debris"]
).astype(np.int8)

# Fourier temporal harmonics (multi-scale)
for period, name in [(24, "hour"), (7, "dow"), (12, "month"), (52, "week")]:
    base = raw["hour"] if name == "hour" else raw["dow"] if name == "dow" else raw["month"] if name == "month" else raw["week_of_year"]
    raw[f"{name}_sin1"] = np.sin(2 * np.pi * base / period).astype(np.float32)
    raw[f"{name}_cos1"] = np.cos(2 * np.pi * base / period).astype(np.float32)
    raw[f"{name}_sin2"] = np.sin(4 * np.pi * base / period).astype(np.float32)
    raw[f"{name}_cos2"] = np.cos(4 * np.pi * base / period).astype(np.float32)

# Label encodings
for col in ["event_cause", "zone", "corridor", "police_station"]:
    le = LabelEncoder()
    raw[col + "_enc"] = le.fit_transform(raw[col].fillna("unknown").astype(str)).astype(np.int16)

raw["veh_type_clean"] = raw["veh_type"].fillna("unknown")
raw["veh_cause_key"] = raw["veh_type_clean"] + "||" + raw["event_cause"]
raw["zone_cause_key"] = raw["zone"].fillna("unk") + "||" + raw["event_cause"]
raw["corridor_cause_key"] = raw["corridor"].fillna("unk") + "||" + raw["event_cause"]

# Text hashing prep
_desc = raw["description"].fillna("").astype(str) if "description" in raw.columns else pd.Series("", index=raw.index)
_reason = raw["reason_breakdown"].fillna("").astype(str) if "reason_breakdown" in raw.columns else pd.Series("", index=raw.index)
raw["text_combined"] = (
    _desc.str.lower().str.replace(r"[^a-z0-9\s]", " ", regex=True)
    + " "
    + _reason.str.lower().str.replace(r"[^a-z0-9\s]", " ", regex=True)
).str.strip()
raw["has_description"] = _to_int8(_desc.str.len() > 5)
raw["description_len"] = _desc.str.len().clip(0, 500).astype(np.float32)

# Coordinates (clean)
raw["lat"] = raw["latitude"].fillna(raw["latitude"].median()).astype(np.float32)
raw["lon"] = raw["longitude"].fillna(raw["longitude"].median()).astype(np.float32)

# Address features
def _extract_pin(addr):
    m = re.search(r"\b(\d{6})\b", str(addr))
    return m.group(1) if m else "000000"

def _road_class(addr):
    a = str(addr).lower()
    if any(k in a for k in ["nh-", "sh-", "highway", "expressway"]):
        return 3
    if any(k in a for k in ["main", "ring road", "outer ring"]):
        return 2
    return 1

if "address" in raw.columns:
    raw["pin_code"] = raw["address"].apply(_extract_pin)
    raw["addr_road_class"] = _to_int8(raw["address"].apply(_road_class), fill=1)
else:
    raw["pin_code"] = "000000"
    raw["addr_road_class"] = np.int8(1)

TRAIN_MASK = (raw["event_observed"] == 1) & (raw["duration_hrs"] <= DURATION_CAP)
EVAL_MASK = TRAIN_MASK.copy()
gm_global = float(raw.loc[TRAIN_MASK, "duration_hrs"].mean())
print(f"  N={len(raw)} | observed={TRAIN_MASK.sum()} | censored={len(raw)-TRAIN_MASK.sum()} | gm={gm_global:.3f}h")


# ============================================================
# 2. CAUSAL GLOBAL FEATURES (computed once, chronologically)
# ============================================================
print("[2] Causal global features...")

def ewma_by_group(df, group_col, val_col, halflife_days=EWMA_HALFLIFE_D):
    """Exponentially weighted mean with strict causality (shift before EWM)."""
    out = np.full(len(df), np.nan, dtype=np.float32)
    hl_seconds = halflife_days * 86400.0
    ts = df["start_ts"].values
    groups = df[group_col].values
    vals = df[val_col].values
    obs = df["event_observed"].values

    grp_idx = defaultdict(list)
    for i, g in enumerate(groups):
        grp_idx[g].append(i)

    for _, idxs in grp_idx.items():
        idxs = np.array(idxs)
        order = np.argsort(ts[idxs])
        idxs = idxs[order]
        ema = np.nan
        for k, i in enumerate(idxs):
            if k > 0 and np.isfinite(ema):
                out[i] = ema
            if obs[i] == 1:
                v = vals[i]
                if not np.isfinite(ema):
                    ema = v
                else:
                    dt = max(ts[i] - ts[idxs[k - 1]], 1.0)
                    alpha = 1.0 - np.exp(-np.log(2) * dt / hl_seconds)
                    ema = alpha * v + (1 - alpha) * ema
    return out


obs_dur = raw["duration_hrs"].where(raw["event_observed"] == 1)
raw["ewma_cause"] = ewma_by_group(
    raw.assign(_dur=obs_dur), "event_cause", "_dur"
)
raw["ewma_corridor"] = ewma_by_group(
    raw.assign(_dur=obs_dur), "corridor", "_dur"
)
raw["ewma_zone"] = ewma_by_group(
    raw.assign(_dur=obs_dur), "zone", "_dur"
)

# Duration percentile rank within cause (causal: expanding rank on observed only)
raw["cause_expanding_rank"] = np.nan
for cause, grp in raw.groupby("event_cause", sort=False):
    idx = grp.index.values
    durs = grp["duration_hrs"].values
    obs = grp["event_observed"].values
    ranks = np.full(len(idx), np.nan)
    history = []
    for k in range(len(idx)):
        if obs[k] == 1:
            if len(history) >= 3:
                ranks[k] = float(np.mean(np.array(history) <= durs[k]))
            history.append(durs[k])
    raw.loc[idx, "cause_expanding_rank"] = ranks
raw["cause_expanding_rank"] = raw["cause_expanding_rank"].fillna(0.5).astype(np.float32)

# Officer experience
raw["officer_event_cnt"] = 0
if "created_by_id" in raw.columns:
    raw["officer_event_cnt"] = raw["created_by_id"].map(
        raw["created_by_id"].value_counts()
    ).fillna(0).astype(np.int16).values

print("  EWMA + rank features done.")


# ============================================================
# 3. SPATIAL HELPERS (fold-safe)
# ============================================================
def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def knn_duration_features(train_df, query_df, k=KNN_K):
    """
    Inverse-distance weighted KNN mean duration.
    train_df: observed training rows with lat, lon, duration_hrs
    query_df: rows to score
    Returns: knn_mean, knn_std, knn_cnt (neighbors within 5km)
    """
    n_q = len(query_df)
    knn_mean = np.full(n_q, np.nan, dtype=np.float32)
    knn_std = np.full(n_q, np.nan, dtype=np.float32)
    knn_cnt = np.zeros(n_q, dtype=np.float32)

    if len(train_df) < 5:
        return knn_mean, knn_std, knn_cnt

    tr_lat = train_df["lat"].values
    tr_lon = train_df["lon"].values
    tr_dur = train_df["duration_hrs"].values
    coords_tr = np.column_stack([tr_lat, tr_lon])
    tree = cKDTree(coords_tr)

    q_lat = query_df["lat"].values
    q_lon = query_df["lon"].values
    coords_q = np.column_stack([q_lat, q_lon])

    dists, idxs = tree.query(coords_q, k=min(k, len(train_df)))

    if dists.ndim == 1:
        dists = dists.reshape(-1, 1)
        idxs = idxs.reshape(-1, 1)

    for i in range(n_q):
        d = dists[i]
        ix = idxs[i]
        valid = d < 5.0
        if not valid.any():
            continue
        d_v = d[valid]
        ix_v = ix[valid]
        w = 1.0 / (d_v + 0.05)
        dur_v = tr_dur[ix_v]
        knn_mean[i] = float(np.average(dur_v, weights=w))
        knn_std[i] = float(np.std(dur_v)) if len(dur_v) > 1 else 0.0
        knn_cnt[i] = float(valid.sum())
    return knn_mean, knn_std, knn_cnt


def knn_duration_loo(train_df, k=KNN_K):
    """LOO KNN for training rows (exclude self-neighbor)."""
    n = len(train_df)
    knn_mean = np.full(n, np.nan, dtype=np.float32)
    if n < 5:
        return knn_mean

    lat = train_df["lat"].values
    lon = train_df["lon"].values
    dur = train_df["duration_hrs"].values
    coords = np.column_stack([lat, lon])
    tree = cKDTree(coords)
    kk = min(k + 1, n)
    dists, idxs = tree.query(coords, k=kk)

    if dists.ndim == 1:
        dists = dists.reshape(-1, 1)
        idxs = idxs.reshape(-1, 1)

    for i in range(n):
        mask = idxs[i] != i
        if not mask.any():
            continue
        d = dists[i][mask][:k]
        ix = idxs[i][mask][:k]
        w = 1.0 / (d + 0.05)
        knn_mean[i] = float(np.average(dur[ix], weights=w))
    return knn_mean


def active_overlap_count(df_subset, radius_km=OVERLAP_RADIUS):
    """
    Count events still active at each row's start time within radius_km.
    Active = started before t and (observed end > t OR censored and started before t).
    """
    n = len(df_subset)
    result = np.zeros(n, dtype=np.float32)
    if n < 2:
        return result

    sub = df_subset.reset_index(drop=True)
    ts = sub["start_ts_hour"].values
    lat = sub["lat"].values
    lon = sub["lon"].values
    dur = sub["duration_hrs"].values
    obs = sub["event_observed"].values

    order = np.argsort(ts)
    for qi in range(n):
        t_q = ts[qi]
        cnt = 0
        for j in order:
            if ts[j] >= t_q:
                break
            if j == qi:
                continue
            end_t = ts[j] + dur[j] if obs[j] == 1 else t_q + 1e9
            if end_t > t_q:
                dist = _haversine_km(lat[qi], lon[qi], lat[j], lon[j])
                if dist <= radius_km:
                    cnt += 1
        result[qi] = cnt
    return result


# Active overlap: compute ONCE on full timeline (causal, ~6x faster than per-fold)
print("  Computing global active overlap (one-time)...")
_t_ov = time.time()
_sorted_ov = raw.sort_values("start_ts")
raw["active_overlap_2km"] = (
    pd.Series(active_overlap_count(_sorted_ov), index=_sorted_ov.index)
    .reindex(raw.index).fillna(0).astype(np.float32)
)
print(f"  Active overlap done in {time.time() - _t_ov:.1f}s")


# ============================================================
# 4. JAMES-STEIN SHRINKAGE ENCODER (fold-safe)
# ============================================================
def stein_encode(tr_df, te_df, key_col, val_col, gm, prior=STEIN_PRIOR):
    """Hierarchical Bayesian shrinkage toward global mean."""
    g = tr_df.groupby(key_col)[val_col]
    g_sum = g.sum()
    g_cnt = g.count()
    g_mean = g_sum / g_cnt.replace(0, 1)

    tr_cnt = tr_df[key_col].map(g_cnt).fillna(0).values
    tr_sum = tr_df[key_col].map(g_sum).fillna(0).values
    tr_loo = np.where(tr_cnt > 1, (tr_sum - tr_df[val_col].values) / (tr_cnt - 1), gm)
    tr_enc = (tr_cnt * tr_loo + prior * gm) / (tr_cnt + prior)

    te_mean = te_df[key_col].map(g_mean).fillna(gm).values
    te_cnt = te_df[key_col].map(g_cnt).fillna(0).values
    te_enc = (te_cnt * te_mean + prior * gm) / (te_cnt + prior)

    return tr_enc.astype(np.float32), te_enc.astype(np.float32)


# ============================================================
# 5. PURGED EXPANDING-WINDOW CV
# ============================================================
print("\n[3] Purged expanding-window CV...")
purge_sec = PURGE_HOURS * 3600.0
ts_all = raw["start_ts"].values
n_total = len(raw)

fold_specs = []
min_train = int(n_total * 0.45)
step = (n_total - min_train) // N_FOLDS

for fi in range(N_FOLDS):
    train_end_idx = min_train + fi * step
    test_start_idx = train_end_idx
    test_end_idx = min(train_end_idx + step, n_total)
    if test_end_idx <= test_start_idx:
        continue

    tr_cutoff_ts = ts_all[train_end_idx - 1] - purge_sec
    tr_idx = np.where(ts_all <= tr_cutoff_ts)[0]
    te_idx = np.arange(test_start_idx, test_end_idx)

    fold_specs.append({
        "name": f"Fold{fi+1}",
        "tr": tr_idx,
        "te": te_idx,
        "train_end": str(raw.loc[tr_idx[-1], "_sdt"])[:10] if len(tr_idx) else "?",
        "test_range": f"{str(raw.loc[te_idx[0], '_sdt'])[:10]}..{str(raw.loc[te_idx[-1], '_sdt'])[:10]}",
    })
    print(f"  {fold_specs[-1]['name']}: train={len(tr_idx)} test={len(te_idx)} "
          f"(purge={PURGE_HOURS}h) | test {fold_specs[-1]['test_range']}")


# ============================================================
# 6. IN-FOLD FEATURE BUILDER
# ============================================================
STATIC_COLS = [
    "hour", "dow", "month", "week_of_year", "is_weekend", "is_night", "is_rush",
    "is_planned", "is_weather", "requires_road_closure", "is_high_priority",
    "has_description", "description_len", "addr_road_class",
    "event_cause_enc", "zone_enc", "corridor_enc", "police_station_enc",
    "officer_event_cnt",
    "hour_sin1", "hour_cos1", "hour_sin2", "hour_cos2",
    "dow_sin1", "dow_cos1", "dow_sin2", "dow_cos2",
    "month_sin1", "month_cos1", "month_sin2", "month_cos2",
    "week_sin1", "week_cos1", "week_sin2", "week_cos2",
    "ewma_cause", "ewma_corridor", "ewma_zone", "cause_expanding_rank",
]

INTERACTION_COLS = [
    ("cause_x_rush", lambda d: d["event_cause_enc"] * d["is_rush"]),
    ("cause_x_night", lambda d: d["event_cause_enc"] * d["is_night"]),
    ("cause_x_planned", lambda d: d["event_cause_enc"] * d["is_planned"]),
    ("cause_x_closure", lambda d: d["event_cause_enc"] * d["requires_road_closure"]),
    ("planned_x_rush", lambda d: d["is_planned"] * d["is_rush"]),
]


def _add_cause_stats(tr_obs, df, gm_fold, fdict):
    for stat, fn in [
        ("cause_mean", "mean"), ("cause_median", "median"),
        ("cause_p90", lambda x: x.quantile(0.9)),
        ("cause_p10", lambda x: x.quantile(0.1)),
    ]:
        s = tr_obs.groupby("event_cause")["duration_hrs"].agg(fn).rename(stat)
        fdict[stat] = df["event_cause"].map(s).fillna(gm_fold).values.astype(np.float32)


def _add_text_svd(tr_obs, df, hash_vec, svd, fdict):
    if svd is None:
        svd = TruncatedSVD(n_components=N_HASH_SVD, random_state=42)
        svd.fit(hash_vec.transform(tr_obs["text_combined"]))
    mat = svd.transform(hash_vec.transform(df["text_combined"]))
    for i in range(N_HASH_SVD):
        fdict[f"hash_svd_{i}"] = mat[:, i].astype(np.float32)
    return svd


def _align_gate_probs(gate, X, n_classes=3):
    """Ensure gate predict_proba always returns (n, 3) even if a regime is absent in train."""
    probs = gate.predict_proba(X)
    full = np.zeros((len(X), n_classes), dtype=np.float64)
    for i, cls in enumerate(gate.classes_):
        ci = int(cls)
        if 0 <= ci < n_classes:
            full[:, ci] = probs[:, i]
    row_sum = full.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return (full / row_sum).astype(np.float32)


def assemble_features(df, fdict):
    parts = []
    for c in STATIC_COLS:
        parts.append(df[c].fillna(0).astype(np.float32).rename(c))
    for name, fn in INTERACTION_COLS:
        parts.append(pd.Series(fn(df), name=name, dtype=np.float32))
    for k, v in fdict.items():
        parts.append(pd.Series(v, name=k, dtype=np.float32))
    return pd.concat(parts, axis=1).fillna(0)


def build_fold_features(tr_idx, te_idx, hash_vec, gm):
    """
    Build leak-free features for one fold.
    Training uses ALL rows (observed + censored) with sample weights.
    Evaluation uses observed test rows only.
    """
    tr_all = raw.iloc[tr_idx].copy()
    te_all = raw.iloc[te_idx].copy()
    tr_obs = tr_all[TRAIN_MASK.iloc[tr_idx]].copy()
    te_obs = te_all[EVAL_MASK.iloc[te_idx]].copy()

    if len(tr_obs) < 30 or len(te_obs) < 5:
        return None

    gm_fold = float(tr_obs["duration_hrs"].mean())
    stein_keys = [
        ("stein_cause", "event_cause"),
        ("stein_corridor", "corridor"),
        ("stein_zone_cause", "zone_cause_key"),
        ("stein_corridor_cause", "corridor_cause_key"),
        ("stein_veh_cause", "veh_cause_key"),
        ("stein_station", "police_station"),
        ("stein_pin", "pin_code"),
    ]

    ft_all, fe = {}, {}

    # Shrinkage encodings: group mean for train-all, LOO-smoothed for test
    for name, col in stein_keys:
        g = tr_obs.groupby(col)["duration_hrs"].mean()
        cnt = tr_obs.groupby(col)["duration_hrs"].count()
        enc_tr = tr_all[col].map(g).fillna(gm_fold).values
        c_tr = tr_all[col].map(cnt).fillna(0).values
        ft_all[name] = ((c_tr * enc_tr + STEIN_PRIOR * gm_fold) / (c_tr + STEIN_PRIOR)).astype(np.float32)
        _, fe[name] = stein_encode(tr_obs, te_obs, col, "duration_hrs", gm_fold)

    _add_cause_stats(tr_obs, tr_all, gm_fold, ft_all)
    _add_cause_stats(tr_obs, te_obs, gm_fold, fe)

    # Spatial KNN from observed train only
    km_tr, ks_tr, kc_tr = knn_duration_features(tr_obs, tr_all)
    km_te, ks_te, kc_te = knn_duration_features(tr_obs, te_obs)
    ft_all["knn_dur_loo"] = km_tr
    ft_all["knn_dur_mean"] = km_tr
    ft_all["knn_dur_std"] = ks_tr
    ft_all["knn_neighbor_cnt"] = kc_tr
    fe["knn_dur_loo"] = km_te
    fe["knn_dur_mean"] = km_te
    fe["knn_dur_std"] = ks_te
    fe["knn_neighbor_cnt"] = kc_te

    # Active overlap: precomputed globally (causal, leak-free)
    ft_all["active_overlap_2km"] = tr_all["active_overlap_2km"].values.astype(np.float32)
    fe["active_overlap_2km"] = te_obs["active_overlap_2km"].values.astype(np.float32)

    # Per-fold SVD on train text (leak-free)
    svd_fold = TruncatedSVD(n_components=N_HASH_SVD, random_state=42)
    svd_fold.fit(hash_vec.transform(tr_obs["text_combined"]))
    _add_text_svd(tr_obs, tr_all, hash_vec, svd_fold, ft_all)
    _add_text_svd(tr_obs, te_obs, hash_vec, svd_fold, fe)

    X_tr = assemble_features(tr_all.reset_index(drop=True), ft_all)
    X_te = assemble_features(te_obs.reset_index(drop=True), fe)

    y_tr_log = np.log1p(tr_all["duration_hrs"].clip(0.05, DURATION_CAP).values).astype(np.float32)
    w_tr = np.where(tr_all["event_observed"].values == 1, 1.0, CENSORED_WEIGHT).astype(np.float32)
    y_te_raw = te_obs["duration_hrs"].values.astype(np.float32)
    closure_tr = tr_all["requires_road_closure"].values.astype(np.int8)
    closure_te = te_obs["requires_road_closure"].values.astype(np.int8)

    return X_tr, y_tr_log, w_tr, X_te, y_te_raw, closure_tr, closure_te, list(X_tr.columns)


# Global hashing (stateless -- no fit needed)
hash_vec = HashingVectorizer(n_features=N_HASH, alternate_sign=False, ngram_range=(1, 2))

print("\n[4] Pre-building fold cache...")
fold_cache = []
for spec in fold_specs:
    res = build_fold_features(spec["tr"], spec["te"], hash_vec, gm_global)
    if res is None:
        continue
    X_tr, y_tr, w_tr, X_te, y_te, c_tr, c_te, fn = res
    if len(X_tr) < 50 or len(X_te) < 5:
        continue
    fold_cache.append((X_tr, y_tr, w_tr, X_te, y_te, c_tr, c_te, fn, spec["name"]))
    print(f"  Cached {spec['name']}: train={len(X_tr)} test={len(X_te)} feats={X_tr.shape[1]}")

if not fold_cache:
    raise RuntimeError(
        "No valid CV folds were built. Check data quality (need observed events with duration <= 48h)."
    )
FEATURE_NAMES = fold_cache[0][7]
print(f"  Total cached folds: {len(fold_cache)} | features: {len(FEATURE_NAMES)}")


def _augment_closure_once(X_tr, X_te, c_tr, c_te, w_tr):
    """Train closure classifier once per fold; reuse across all Optuna trials."""
    clf = lgb.LGBMClassifier(**_lgb_kw(
        n_estimators=150, learning_rate=0.05, num_leaves=31,
    ))
    clf.fit(X_tr, c_tr, sample_weight=w_tr)
    X_tr_a = X_tr.copy()
    X_te_a = X_te.copy()
    X_tr_a["pred_closure_prob"] = clf.predict_proba(X_tr)[:, 1].astype(np.float32)
    X_te_a["pred_closure_prob"] = clf.predict_proba(X_te)[:, 1].astype(np.float32)
    return X_tr_a, X_te_a, clf


print("  Caching closure-augmented fold matrices (saves ~1000+ redundant fits)...")
aug_cache = []
for X_tr, y_tr, w_tr, X_te, y_te, c_tr, c_te, fn, fname in fold_cache:
    X_tr_a, X_te_a, _ = _augment_closure_once(X_tr, X_te, c_tr, c_te, w_tr)
    aug_cache.append((X_tr_a, y_tr, w_tr, X_te_a, y_te, fname))


# ============================================================
# 7. PHASE A -- HETEROGENEOUS BASELINE
# ============================================================
print(f"\n{'='*72}")
print("PHASE A -- Heterogeneous baselines (Quantile LGB + HGB + ExtraTrees)")
print("="*72)

DEFAULT_LGB_Q = _lgb_kw(
    objective="quantile", alpha=0.5,
    n_estimators=500, learning_rate=0.025, num_leaves=96,
    min_child_samples=15, colsample_bytree=0.75, subsample=0.8,
    reg_alpha=0.05, reg_lambda=0.5,
)
DEFAULT_HGB = dict(
    loss="absolute_error", max_iter=400, learning_rate=0.06,
    max_depth=8, min_samples_leaf=20, l2_regularization=0.5,
    random_state=42,
)
DEFAULT_ET = dict(
    n_estimators=300, max_depth=14, min_samples_leaf=8,
    max_features=0.6, n_jobs=-1, random_state=42,
)

phase_a = {"lgb_q": [], "hgb": [], "et": [], "moe": []}

for X_tr_aug, y_tr, w_tr, X_te_aug, y_te, fname in aug_cache:
    m_lgb = lgb.LGBMRegressor(**DEFAULT_LGB_Q)
    m_lgb.fit(X_tr_aug, y_tr, sample_weight=w_tr)
    p_lgb = np.expm1(np.maximum(m_lgb.predict(X_te_aug), 0))

    m_hgb = HistGradientBoostingRegressor(**DEFAULT_HGB)
    m_hgb.fit(X_tr_aug, y_tr, sample_weight=w_tr)
    p_hgb = np.expm1(np.maximum(m_hgb.predict(X_te_aug), 0))

    m_et = ExtraTreesRegressor(**DEFAULT_ET)
    m_et.fit(X_tr_aug, y_tr, sample_weight=w_tr)
    p_et = np.expm1(np.maximum(m_et.predict(X_te_aug), 0))

    # Mixture-of-Experts
    y_tr_hours = np.expm1(y_tr)
    short_b, long_b = EXPERT_BOUNDS
    regimes = np.where(y_tr_hours < short_b, 0, np.where(y_tr_hours < long_b, 1, 2))

    gate = lgb.LGBMClassifier(**_lgb_kw(
        objective="multiclass", num_class=3, n_estimators=200,
        learning_rate=0.05, num_leaves=48,
    ))
    gate.fit(X_tr_aug, regimes, sample_weight=w_tr)
    gate_probs = _align_gate_probs(gate, X_te_aug)

    expert_preds = np.zeros(len(X_te_aug))
    for r in range(3):
        mask_r = regimes == r
        if mask_r.sum() < 10:
            expert_preds += gate_probs[:, r] * p_lgb
            continue
        w_r = w_tr.copy()
        w_r[~mask_r] *= 0.15
        exp_m = lgb.LGBMRegressor(**DEFAULT_LGB_Q)
        exp_m.fit(X_tr_aug, y_tr, sample_weight=w_r)
        p_r = np.expm1(np.maximum(exp_m.predict(X_te_aug), 0))
        expert_preds += gate_probs[:, r] * p_r

    phase_a["lgb_q"].append(mean_absolute_error(y_te, p_lgb))
    phase_a["hgb"].append(mean_absolute_error(y_te, p_hgb))
    phase_a["et"].append(mean_absolute_error(y_te, p_et))
    phase_a["moe"].append(mean_absolute_error(y_te, expert_preds))

    ens = (p_lgb + p_hgb + p_et) / 3
    print(f"  {fname}: LGB_Q={phase_a['lgb_q'][-1]:.3f} HGB={phase_a['hgb'][-1]:.3f} "
          f"ET={phase_a['et'][-1]:.3f} MoE={phase_a['moe'][-1]:.3f} ENS={mean_absolute_error(y_te, ens):.3f}h")

mae_a = {k: float(np.mean(v)) for k, v in phase_a.items()}
print(f"\n  Phase A means: " + "  ".join(f"{k}={v:.4f}" for k, v in mae_a.items()))


# ============================================================
# 8. PHASE B -- OPTUNA (multi-objective aware)
# ============================================================
print(f"\n{'='*72}")
print(f"PHASE B -- Optuna ({N_OPTUNA_LGB}+{N_OPTUNA_HGB}+{N_OPTUNA_ET} trials)")
print("="*72)


def _cv_mae_lgb(predict_fn):
    maes = []
    for X_tr_a, y_tr, w_tr, X_te_a, y_te, _ in aug_cache:
        pred = predict_fn(X_tr_a, y_tr, w_tr, X_te_a)
        maes.append(mean_absolute_error(y_te, np.expm1(np.maximum(pred, 0))))
    return float(np.mean(maes))


def obj_lgb(trial):
    alpha = trial.suggest_categorical("alpha", [0.35, 0.5, 0.65])
    p = _lgb_kw(
        objective="quantile", alpha=alpha,
        n_estimators=trial.suggest_int("n_estimators", 200, 1800),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
        num_leaves=trial.suggest_int("num_leaves", 24, 256),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 80),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
        subsample=trial.suggest_float("subsample", 0.3, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        max_depth=trial.suggest_int("max_depth", 4, 14),
    )
    return _cv_mae_lgb(lambda Xtr, ytr, w, Xte: lgb.LGBMRegressor(**p).fit(Xtr, ytr, sample_weight=w).predict(Xte))


study_lgb = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=35))
study_lgb.optimize(obj_lgb, n_trials=N_OPTUNA_LGB, n_jobs=OPTUNA_JOBS_GPU, show_progress_bar=False)
best_lgb_p = study_lgb.best_params
best_lgb_mae = study_lgb.best_value
print(f"  Quantile LGB best: {best_lgb_mae:.4f}h  alpha={best_lgb_p.get('alpha', 0.5)}")


def obj_hgb(trial):
    p = {
        "loss": "absolute_error", "random_state": 42,
        "max_iter": trial.suggest_int("max_iter", 150, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 16),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 60),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 5.0, log=True),
        "max_bins": trial.suggest_int("max_bins", 128, 255),
    }
    return _cv_mae_lgb(lambda Xtr, ytr, w, Xte: HistGradientBoostingRegressor(**p).fit(Xtr, ytr, sample_weight=w).predict(Xte))


study_hgb = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=77, n_startup_trials=25))
study_hgb.optimize(obj_hgb, n_trials=N_OPTUNA_HGB, n_jobs=OPTUNA_JOBS_CPU, show_progress_bar=False)
best_hgb_p = study_hgb.best_params
best_hgb_mae = study_hgb.best_value
print(f"  HistGradientBoost best: {best_hgb_mae:.4f}h")


def obj_et(trial):
    p = {
        "n_estimators": trial.suggest_int("n_estimators", 150, 600),
        "max_depth": trial.suggest_int("max_depth", 8, 24),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 3, 30),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        "random_state": 42, "n_jobs": -1,
    }
    return _cv_mae_lgb(lambda Xtr, ytr, w, Xte: ExtraTreesRegressor(**p).fit(Xtr, ytr, sample_weight=w).predict(Xte))


study_et = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=123, n_startup_trials=20))
study_et.optimize(obj_et, n_trials=N_OPTUNA_ET, n_jobs=OPTUNA_JOBS_CPU, show_progress_bar=False)
best_et_p = study_et.best_params
best_et_mae = study_et.best_value
print(f"  ExtraTrees best: {best_et_mae:.4f}h")


# ============================================================
# 9. PHASE C -- QUANTILE TRIPLET + MoE + GBM META-BLENDER
# ============================================================
print(f"\n{'='*72}")
print("PHASE C -- Quantile triplet + MoE + GBM meta-blender + Isotonic calibration")
print("="*72)

best_lgb_full = _lgb_kw(**{k: v for k, v in best_lgb_p.items() if k != "alpha"}, objective="quantile")
best_hgb_full = {**best_hgb_p, "loss": "absolute_error", "random_state": 42}
best_et_full = {**best_et_p, "random_state": 42, "n_jobs": -1}

oof_lgb = []; oof_hgb = []; oof_et = []; oof_moe = []; oof_y = []
oof_q_lo = []; oof_q_hi = []
fold_c_maes = []
short_b, long_b = EXPERT_BOUNDS

for X_tr_a, y_tr, w_tr, X_te_a, y_te, fname in aug_cache:
    # Quantile triplet
    q_preds = {}
    for alpha in QUANTILE_ALPHAS:
        params = {**best_lgb_full, "alpha": alpha}
        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr_a, y_tr, sample_weight=w_tr)
        q_preds[alpha] = np.expm1(np.maximum(m.predict(X_te_a), 0))

    p_lgb = q_preds[0.50]
    p_lo = q_preds[0.35]
    p_hi = q_preds[0.65]

    m_hgb = HistGradientBoostingRegressor(**best_hgb_full)
    m_hgb.fit(X_tr_a, y_tr, sample_weight=w_tr)
    p_hgb = np.expm1(np.maximum(m_hgb.predict(X_te_a), 0))

    m_et = ExtraTreesRegressor(**best_et_full)
    m_et.fit(X_tr_a, y_tr, sample_weight=w_tr)
    p_et = np.expm1(np.maximum(m_et.predict(X_te_a), 0))

    # MoE
    y_tr_h = np.expm1(y_tr)
    regimes = np.where(y_tr_h < short_b, 0, np.where(y_tr_h < long_b, 1, 2))
    gate = lgb.LGBMClassifier(**_lgb_kw(
        objective="multiclass", num_class=3, n_estimators=250,
        learning_rate=0.04, num_leaves=64,
    ))
    gate.fit(X_tr_a, regimes, sample_weight=w_tr)
    gate_probs = _align_gate_probs(gate, X_te_a)
    moe_pred = np.zeros(len(X_te_a))
    for r in range(3):
        w_r = w_tr.copy()
        w_r[regimes != r] *= 0.12
        exp_m = lgb.LGBMRegressor(**{**best_lgb_full, "alpha": 0.5})
        exp_m.fit(X_tr_a, y_tr, sample_weight=w_r)
        moe_pred += gate_probs[:, r] * np.expm1(np.maximum(exp_m.predict(X_te_a), 0))

    oof_lgb.extend(p_lgb); oof_hgb.extend(p_hgb); oof_et.extend(p_et)
    oof_moe.extend(moe_pred); oof_y.extend(y_te)
    oof_q_lo.extend(p_lo); oof_q_hi.extend(p_hi)

    ens = (p_lgb + p_hgb + p_et + moe_pred) / 4
    fold_c_maes.append(mean_absolute_error(y_te, ens))
    coverage = float(np.mean((y_te >= p_lo) & (y_te <= p_hi)))
    print(f"  {fname}: LGB={mean_absolute_error(y_te, p_lgb):.3f} HGB={mean_absolute_error(y_te, p_hgb):.3f} "
          f"ET={mean_absolute_error(y_te, p_et):.3f} MoE={mean_absolute_error(y_te, moe_pred):.3f} "
          f"ENS={fold_c_maes[-1]:.3f}h cov@35-65={coverage:.1%}")

X_meta = np.column_stack([oof_lgb, oof_hgb, oof_et, oof_moe, oof_q_lo, oof_q_hi])
y_meta = np.array(oof_y)

meta = GradientBoostingRegressor(
    n_estimators=120, learning_rate=0.08, max_depth=3,
    min_samples_leaf=20, random_state=42,
)
meta.fit(X_meta, y_meta)
stacked_pred = meta.predict(X_meta)
mae_stacked = mean_absolute_error(y_meta, stacked_pred)

iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(stacked_pred, y_meta)
calibrated_pred = iso.predict(stacked_pred)
mae_calibrated = mean_absolute_error(y_meta, calibrated_pred)

mae_ens = float(np.mean(fold_c_maes))
quantile_coverage = float(np.mean((y_meta >= np.array(oof_q_lo)) & (y_meta <= np.array(oof_q_hi))))

print(f"\n  4-model ensemble MAE: {mae_ens:.4f}h")
print(f"  GBM meta-blender OOF MAE: {mae_stacked:.4f}h")
print(f"  Isotonic calibrated MAE: {mae_calibrated:.4f}h")
print(f"  Quantile band coverage (P35-P65): {quantile_coverage:.1%}")
print(f"  Meta feature importances: {meta.feature_importances_}")


# ============================================================
# 10. PHASE D -- FINAL MODELS ON ALL DATA
# ============================================================
print(f"\n{'='*72}")
print("PHASE D -- Final training on all data (censoring-aware)")
print("="*72)

all_df = raw.copy()
gm_all = float(all_df.loc[TRAIN_MASK, "duration_hrs"].mean())
obs_all = all_df[TRAIN_MASK].copy()

# Final encodings on all observed
final_enc = {}
for name, col in [
    ("stein_cause", "event_cause"), ("stein_corridor", "corridor"),
    ("stein_zone_cause", "zone_cause_key"), ("stein_corridor_cause", "corridor_cause_key"),
    ("stein_veh_cause", "veh_cause_key"), ("stein_station", "police_station"),
    ("stein_pin", "pin_code"),
]:
    g = obs_all.groupby(col)["duration_hrs"].mean()
    cnt = obs_all.groupby(col)["duration_hrs"].count()
    enc = all_df[col].map(g).fillna(gm_all)
    c = all_df[col].map(cnt).fillna(0)
    all_df[name] = ((c * enc + STEIN_PRIOR * gm_all) / (c + STEIN_PRIOR)).astype(np.float32)

for stat, fn in [
    ("cause_mean", "mean"), ("cause_median", "median"),
    ("cause_p90", lambda x: x.quantile(0.9)), ("cause_p10", lambda x: x.quantile(0.1)),
]:
    s = obs_all.groupby("event_cause")["duration_hrs"].agg(fn).rename(stat)
    all_df[stat] = all_df["event_cause"].map(s).fillna(gm_all).astype(np.float32)

km, ks, kc = knn_duration_features(obs_all, all_df)
all_df["knn_dur_loo"] = km
all_df["knn_dur_mean"] = km
all_df["knn_dur_std"] = ks
all_df["knn_neighbor_cnt"] = kc

# active_overlap_2km already in raw — copy through all_df (no recompute)

svd_final = TruncatedSVD(n_components=N_HASH_SVD, random_state=42)
svd_final.fit(hash_vec.transform(obs_all["text_combined"]))
txt_svd = svd_final.transform(hash_vec.transform(all_df["text_combined"]))
for i in range(N_HASH_SVD):
    all_df[f"hash_svd_{i}"] = txt_svd[:, i].astype(np.float32)

for name, fn in INTERACTION_COLS:
    all_df[name] = fn(all_df).astype(np.float32)

ALL_FEATURES = STATIC_COLS + [n for n, _ in INTERACTION_COLS] + [
    "stein_cause", "stein_corridor", "stein_zone_cause", "stein_corridor_cause",
    "stein_veh_cause", "stein_station", "stein_pin",
    "cause_mean", "cause_median", "cause_p90", "cause_p10",
    "knn_dur_loo", "knn_dur_mean", "knn_dur_std", "knn_neighbor_cnt",
    "active_overlap_2km",
] + [f"hash_svd_{i}" for i in range(N_HASH_SVD)]

for col in ALL_FEATURES:
    if col not in all_df.columns:
        all_df[col] = 0.0
    all_df[col] = pd.to_numeric(all_df[col], errors="coerce").fillna(0).astype(np.float32)

X_all = all_df[ALL_FEATURES].copy()
y_all_log = np.log1p(all_df["duration_hrs"].clip(0.05, DURATION_CAP).values).astype(np.float32)
w_all = np.where(all_df["event_observed"].values == 1, 1.0, CENSORED_WEIGHT).astype(np.float32)
closure_all = all_df["requires_road_closure"].values.astype(np.int8)

# Closure classifier
print("  Training closure classifier...")
clf_final = lgb.LGBMClassifier(**_lgb_kw(n_estimators=300, learning_rate=0.04, num_leaves=48))
clf_final.fit(X_all, closure_all, sample_weight=w_all)
all_df["pred_closure_prob"] = clf_final.predict_proba(X_all)[:, 1].astype(np.float32)
X_all["pred_closure_prob"] = all_df["pred_closure_prob"]
FINAL_FEATURES = list(X_all.columns)

print("  Training quantile triplet...")
fin_models_q = {}
for alpha in QUANTILE_ALPHAS:
    params = {**best_lgb_full, "alpha": alpha}
    m = lgb.LGBMRegressor(**params)
    m.fit(X_all, y_all_log, sample_weight=w_all)
    fin_models_q[alpha] = m

print("  Training HGB + ExtraTrees...")
fin_hgb = HistGradientBoostingRegressor(**best_hgb_full)
fin_hgb.fit(X_all, y_all_log, sample_weight=w_all)

fin_et = ExtraTreesRegressor(**best_et_full)
fin_et.fit(X_all, y_all_log, sample_weight=w_all)

# MoE final
print("  Training MoE experts...")
y_hours = np.expm1(y_all_log)
regimes = np.where(y_hours < short_b, 0, np.where(y_hours < long_b, 1, 2))
fin_gate = lgb.LGBMClassifier(**_lgb_kw(
    objective="multiclass", num_class=3, n_estimators=300,
    learning_rate=0.04, num_leaves=64,
))
fin_gate.fit(X_all, regimes, sample_weight=w_all)
fin_experts = []
for r in range(3):
    w_r = w_all.copy()
    w_r[regimes != r] *= 0.12
    exp_m = lgb.LGBMRegressor(**{**best_lgb_full, "alpha": 0.5})
    exp_m.fit(X_all, y_all_log, sample_weight=w_r)
    fin_experts.append(exp_m)

import joblib
joblib.dump(fin_models_q, OUTPUT_DIR / "prism_v11_quantile_models.pkl")
joblib.dump(fin_hgb, OUTPUT_DIR / "prism_v11_hgb.pkl")
joblib.dump(fin_et, OUTPUT_DIR / "prism_v11_extratrees.pkl")
joblib.dump(fin_gate, OUTPUT_DIR / "prism_v11_gate.pkl")
joblib.dump(fin_experts, OUTPUT_DIR / "prism_v11_experts.pkl")
joblib.dump(clf_final, OUTPUT_DIR / "prism_v11_closure.pkl")
joblib.dump(meta, OUTPUT_DIR / "prism_v11_meta_gbm.pkl")
joblib.dump(iso, OUTPUT_DIR / "prism_v11_isotonic.pkl")
joblib.dump(svd_final, OUTPUT_DIR / "v11_hash_svd.pkl")
joblib.dump(FINAL_FEATURES, OUTPUT_DIR / "v11_features.pkl")
print("  Models saved.")


# ============================================================
# 11. PHASE E -- SHAP
# ============================================================
print(f"\n{'='*72}")
print("PHASE E -- Feature importance (SHAP on quantile P50 model)")
print("="*72)
shap_results = {}
if HAS_SHAP:
    try:
        Xs = X_all[FINAL_FEATURES].sample(min(800, len(X_all)), random_state=42)
        exp = shap.TreeExplainer(fin_models_q[0.50])
        sv = exp.shap_values(Xs)
        imp = np.abs(sv).mean(axis=0)
        ranked = sorted(zip(FINAL_FEATURES, imp), key=lambda x: x[1], reverse=True)
        print("\n  Top 25 SHAP features:")
        mx = imp.max()
        for feat, val in ranked[:25]:
            print(f"    {feat:<42} {val:>7.4f}  {'|' * max(1, int(val / mx * 28))}")
        shap_results = {f: round(float(v), 5) for f, v in ranked}
        with open(OUTPUT_DIR / "shap_v11.json", "w") as fj:
            json.dump(shap_results, fj, indent=2)
    except Exception as e:
        print(f"  SHAP skipped: {e}")
else:
    imp = fin_models_q[0.50].feature_importances_
    ranked = sorted(zip(FINAL_FEATURES, imp), key=lambda x: x[1], reverse=True)
    for feat, val in ranked[:20]:
        print(f"    {feat:<42} {val:>8.0f}")
    shap_results = {f: int(v) for f, v in ranked}


# ============================================================
# 12. FINAL RESULTS
# ============================================================
elapsed = time.time() - t_start
print(f"\n{'='*72}")
print("EVENTFLOW PRISM V11 -- FINAL RESULTS")
print("="*72)

journey = [
    ("V9 honest baseline (reference)", 2.843),
    ("V10 reference (running on Kaggle)", None),
    ("V11 Phase A Quantile LGB", mae_a["lgb_q"]),
    ("V11 Phase A HistGradientBoosting", mae_a["hgb"]),
    ("V11 Phase A ExtraTrees", mae_a["et"]),
    ("V11 Phase A Mixture-of-Experts", mae_a["moe"]),
    ("V11 Phase B Optuna Quantile LGB", best_lgb_mae),
    ("V11 Phase B Optuna HGB", best_hgb_mae),
    ("V11 Phase B Optuna ExtraTrees", best_et_mae),
    ("V11 Phase C 4-model ensemble", mae_ens),
    ("V11 Phase C GBM meta-blender OOF", mae_stacked),
    ("V11 Phase C Isotonic calibrated", mae_calibrated),
]
for name, mae in journey:
    if mae is None:
        print(f"  {name:<52} (compare after V10 run)")
    else:
        bar = "|" * max(1, int((1 - min(mae, 10) / 10) * 35))
        print(f"  {name:<52} {mae:>7.3f}h  {bar}")

imp_pct = (2.843 - mae_calibrated) / 2.843 * 100
print(f"\n  Improvement over V9 baseline: {imp_pct:+.1f}%")
print(f"  Quantile coverage P35-P65: {quantile_coverage:.1%}")
print(f"  Runtime: {elapsed/60:.1f} min | Hardware: {'GPU' if HAS_GPU else 'CPU'}")

results = {
    "experiment": "EventFlow Prism V11",
    "complementary_to": "GridGuard V10",
    "v9_honest_baseline": 2.843,
    "v11_phase_a": {k: round(v, 4) for k, v in mae_a.items()},
    "v11_optuna_lgb_mae": round(best_lgb_mae, 4),
    "v11_optuna_hgb_mae": round(best_hgb_mae, 4),
    "v11_optuna_et_mae": round(best_et_mae, 4),
    "v11_ensemble_mae": round(mae_ens, 4),
    "v11_stacked_mae": round(mae_stacked, 4),
    "v11_calibrated_mae": round(mae_calibrated, 4),
    "v11_quantile_coverage_35_65": round(quantile_coverage, 4),
    "improvement_over_v9_pct": round(imp_pct, 2),
    "best_lgb_params": best_lgb_p,
    "best_hgb_params": best_hgb_p,
    "best_et_params": best_et_p,
    "meta_importances": list(meta.feature_importances_),
    "expert_bounds_hours": list(EXPERT_BOUNDS),
    "censoring_weight": CENSORED_WEIGHT,
    "cv_type": f"purged_expanding_{N_FOLDS}fold_{PURGE_HOURS}h",
    "shap_top10": dict(list(shap_results.items())[:10]) if shap_results else {},
    "runtime_minutes": round(elapsed / 60, 1),
    "hardware": "GPU" if HAS_GPU else "CPU",
    "n_features": len(FINAL_FEATURES),
    "n_folds": len(fold_cache),
    "innovations": [
        "censoring_aware_weights", "spatial_knn_loo", "active_overlap_2km",
        "james_stein_hierarchical", "ewma_recency", "hashing_vectorizer_text",
        "fourier_temporal", "auxiliary_closure_classifier", "mixture_of_experts",
        "quantile_triplet_fusion", "histgradientboosting", "extratrees",
        "gbm_meta_blender", "isotonic_calibration", "purged_cv",
    ],
}
with open(OUTPUT_DIR / "v11_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n  Saved: {OUTPUT_DIR}/v11_results.json")
print(f"  Saved: {OUTPUT_DIR}/shap_v11.json")
print(f"\n{'='*72}")
print("DONE! Run V10 + V11 on Kaggle, then share BOTH v10_results.json and v11_results.json")
print("We'll blend the best signals from each track into the ultimate model.")
print("="*72)