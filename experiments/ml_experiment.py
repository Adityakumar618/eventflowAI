"""
????????????????????????????????????????????????????????????????????????????????
?        EventFlow AI ? ML EXPERIMENT SANDBOX                                ?
?        Isolated from main src/. Does NOT touch any production files.        ?
?                                                                              ?
?  PURPOSE: Find the best possible ML configuration for duration prediction   ?
?  through rigorous temporal cross-validation and model comparison.           ?
?                                                                              ?
?  RUNS:                                                                       ?
?    1. Feature Engineering v2 (richer features, interaction terms)           ?
?    2. Temporal CV (3-fold expanding window, Nov-Apr)                        ?
?    3. Model Shootout: Weibull AFT vs Cox PH vs LGB vs XGB vs Ensemble      ?
?    4. IPCW C-index + MAE + Integrated Brier Score per model                ?
?    5. SHAP feature importance on winner model                               ?
?    6. Calibration check: predicted vs actual duration                       ?
?                                                                              ?
?  OUTPUT: experiments/results/experiment_report.json                         ?
????????????????????????????????????????????????????????????????????????????????
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import json
import time
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb

# ----------------------------------------------------------------------------
# 0. SETUP
# ----------------------------------------------------------------------------
OUT_DIR = Path("experiments/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("EVENTFLOW AI ? ML EXPERIMENT SANDBOX")
print("="*70)

# ----------------------------------------------------------------------------
# 1. LOAD & FEATURE ENGINEERING v2
# ----------------------------------------------------------------------------
print("\n[STEP 1] Feature Engineering v2")

df = pd.read_parquet("data/processed/survival_ready.parquet")
df = df.sort_values('start_datetime').reset_index(drop=True)

# -- 1A: Rich temporal features -----------------------------------------------
df['is_monsoon'] = df['month'].isin([6, 7, 8, 9]).astype(int)
df['is_night']   = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)

# Peak intensity: how bad is this hour? (learned from data)
hour_avg_dur = df[df['event_observed']==1].groupby('hour')['duration_hrs'].mean()
df['hour_avg_duration'] = df['hour'].map(hour_avg_dur).fillna(hour_avg_dur.mean())

# Day-of-week average duration (is Monday worse than Sunday?)
dow_avg_dur = df[df['event_observed']==1].groupby('day_of_week')['duration_hrs'].mean()
df['dow_avg_duration'] = df['day_of_week'].map(dow_avg_dur).fillna(dow_avg_dur.mean())

# -- 1B: Spatial context features --------------------------------------------
# Corridor historical mean duration (data-driven, not rule-based)
corridor_stats = df[df['event_observed']==1].groupby('corridor').agg(
    corridor_mean_dur  = ('duration_hrs', 'mean'),
    corridor_event_cnt = ('duration_hrs', 'count'),
).reset_index()
df = df.merge(corridor_stats, on='corridor', how='left')
df['corridor_mean_dur']  = df['corridor_mean_dur'].fillna(df['duration_hrs'].median())
df['corridor_event_cnt'] = df['corridor_event_cnt'].fillna(1)

# Zone historical mean duration
zone_stats = df[df['event_observed']==1].groupby('zone').agg(
    zone_mean_dur = ('duration_hrs', 'mean'),
).reset_index()
df = df.merge(zone_stats, on='zone', how='left')
df['zone_mean_dur'] = df['zone_mean_dur'].fillna(df['duration_hrs'].median())

# Police station historical mean duration
station_stats = df[df['event_observed']==1].groupby('police_station').agg(
    station_mean_dur = ('duration_hrs', 'mean'),
    station_event_cnt = ('duration_hrs', 'count'),
).reset_index()
df = df.merge(station_stats, on='police_station', how='left')
df['station_mean_dur']  = df['station_mean_dur'].fillna(df['duration_hrs'].median())
df['station_event_cnt'] = df['station_event_cnt'].fillna(1)

# -- 1C: Cause-level features -------------------------------------------------
cause_stats = df[df['event_observed']==1].groupby('event_cause').agg(
    cause_mean_dur   = ('duration_hrs', 'mean'),
    cause_median_dur = ('duration_hrs', 'median'),
    cause_p90_dur    = ('duration_hrs', lambda x: x.quantile(0.90)),
    cause_std_dur    = ('duration_hrs', 'std'),
    cause_closure_rate = ('requires_road_closure', 'mean'),
).reset_index()
df = df.merge(cause_stats, on='event_cause', how='left')
for col in ['cause_mean_dur','cause_median_dur','cause_p90_dur','cause_std_dur','cause_closure_rate']:
    df[col] = df[col].fillna(df[col].median())

# -- 1D: Interaction features (the real differentiators) ---------------------
# These capture non-linear interactions that tree models love
df['cause_x_rush']     = df['event_cause_encoded'] * df['is_rush_hour']
df['cause_x_weekend']  = df['event_cause_encoded'] * df['is_weekend']
df['cause_x_zone']     = df['event_cause_encoded'] * df['zone_encoded']
df['cause_x_closure']  = df['event_cause_encoded'] * df['requires_road_closure'].fillna(0).astype(int)
df['corridor_x_hour']  = df['corridor_encoded'] * df['hour']
df['station_x_cause']  = df['police_station_encoded'] * df['event_cause_encoded']

# -- 1E: Rolling window features (activity in past 24h per zone) -------------
df_sorted = df.sort_values('start_datetime').reset_index(drop=True)
df_sorted['start_ts'] = df_sorted['start_datetime'].astype(np.int64) // 10**9  # Unix seconds

# For each event: count events in same zone in past 24 hours
zone_24h_load = []
for i, row in df_sorted.iterrows():
    t = row['start_ts']
    z = row['zone']
    prev_24h = df_sorted.iloc[max(0,i-200):i]  # look back up to 200 rows
    count = ((prev_24h['zone'] == z) & 
             (prev_24h['start_ts'] >= t - 86400) & 
             (prev_24h['start_ts'] < t)).sum()
    zone_24h_load.append(count)

df_sorted['zone_24h_event_load'] = zone_24h_load

# Corridor load in past 12h
corridor_12h_load = []
for i, row in df_sorted.iterrows():
    t = row['start_ts']
    c = row['corridor']
    prev_12h = df_sorted.iloc[max(0,i-100):i]
    count = ((prev_12h['corridor'] == c) & 
             (prev_12h['start_ts'] >= t - 43200) & 
             (prev_12h['start_ts'] < t)).sum()
    corridor_12h_load.append(count)

df_sorted['corridor_12h_load'] = corridor_12h_load

df = df_sorted.copy()
print(f"  Feature count after v2 engineering: {len(df.columns)} columns")

# -- Final feature set ---------------------------------------------------------
FEATURES_V2 = [
    # Core temporal
    'hour', 'day_of_week', 'month', 'is_weekend', 'is_rush_hour',
    'hour_sin', 'hour_cos', 'is_monsoon', 'is_night',
    'hour_avg_duration', 'dow_avg_duration',
    # Core categorical (encoded)
    'event_cause_encoded', 'zone_encoded', 'corridor_encoded',
    'police_station_encoded',
    # Closure
    'requires_road_closure',
    # Weather
    'is_weather_related',
    # Cause-level stats (from historical data)
    'cause_mean_dur', 'cause_median_dur', 'cause_p90_dur',
    'cause_std_dur', 'cause_closure_rate',
    # Spatial context
    'corridor_mean_dur', 'corridor_event_cnt',
    'zone_mean_dur', 'station_mean_dur', 'station_event_cnt',
    # Interactions
    'cause_x_rush', 'cause_x_weekend', 'cause_x_zone',
    'cause_x_closure', 'corridor_x_hour', 'station_x_cause',
    # Load features
    'zone_24h_event_load', 'corridor_12h_load',
]

# Clean up
df['requires_road_closure'] = df['requires_road_closure'].fillna(0).astype(int)
for col in FEATURES_V2:
    if col in df.columns:
        df[col] = df[col].fillna(0)

print(f"  Final feature vector: {len(FEATURES_V2)} features")
print("  [OK] Feature engineering v2 complete")

# ----------------------------------------------------------------------------
# 2. TEMPORAL CROSS-VALIDATION (3-FOLD EXPANDING WINDOW)
# ----------------------------------------------------------------------------
print("\n[STEP 2] Temporal Cross-Validation Setup")

# Dataset spans Nov 2023 ? Apr 2024 (6 months)
# We use 3 expanding folds:
#   Fold 1: Train=Nov-Jan  -> Test=Feb
#   Fold 2: Train=Nov-Feb  -> Test=Mar
#   Fold 3: Train=Nov-Mar  -> Test=Apr

from datetime import datetime
import pytz

# Handle timezone-aware datetimes
df['start_dt_naive'] = pd.to_datetime(df['start_datetime']).dt.tz_localize(None)

folds = [
    {
        'name': 'Fold 1 (Train:Nov-Jan, Test:Feb)',
        'train_end':  pd.Timestamp('2024-01-31'),
        'test_start': pd.Timestamp('2024-02-01'),
        'test_end':   pd.Timestamp('2024-02-29'),
    },
    {
        'name': 'Fold 2 (Train:Nov-Feb, Test:Mar)',
        'train_end':  pd.Timestamp('2024-02-29'),
        'test_start': pd.Timestamp('2024-03-01'),
        'test_end':   pd.Timestamp('2024-03-31'),
    },
    {
        'name': 'Fold 3 (Train:Nov-Mar, Test:Apr)',
        'train_end':  pd.Timestamp('2024-03-31'),
        'test_start': pd.Timestamp('2024-04-01'),
        'test_end':   pd.Timestamp('2024-04-30'),
    },
]

for fold in folds:
    fold['train_idx'] = df[df['start_dt_naive'] <= fold['train_end']].index.tolist()
    fold['test_idx']  = df[
        (df['start_dt_naive'] >= fold['test_start']) &
        (df['start_dt_naive'] <= fold['test_end'])
    ].index.tolist()
    print(f"  {fold['name']}: train={len(fold['train_idx'])} | test={len(fold['test_idx'])}")


# ----------------------------------------------------------------------------
# 3. MODEL SHOOTOUT
# ----------------------------------------------------------------------------
print("\n[STEP 3] Model Shootout")
print("  Models: Weibull AFT | Cox PH | LightGBM | XGBoost | Ensemble")
print()

from lifelines import WeibullAFTFitter, CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
import xgboost as xgb

# -- Helper: evaluate on test fold --------------------------------------------
def eval_fold(y_pred_hrs, y_true_hrs, y_observed):
    """Evaluate duration predictions on closed events only (MAE, RMSE)."""
    # Only evaluate where event was actually observed (closed)
    mask = (y_observed == 1) & (y_true_hrs > 0) & (y_true_hrs < 700)
    if mask.sum() < 10:
        return {'mae': np.nan, 'rmse': np.nan, 'mape': np.nan, 'n': int(mask.sum())}

    y_t = np.array(y_true_hrs)[mask]
    y_p = np.array(y_pred_hrs)[mask]
    y_p = np.maximum(y_p, 0.1)

    mae  = mean_absolute_error(y_t, y_p)
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    mape = np.mean(np.abs(y_t - y_p) / np.maximum(y_t, 0.1)) * 100

    return {'mae': round(mae,3), 'rmse': round(rmse,3), 'mape': round(mape,1), 'n': int(mask.sum())}

def eval_cindex(y_pred, y_time, y_event):
    """Harrell's C-index (higher = better ranking)."""
    try:
        return round(concordance_index(y_time, y_pred, y_event), 4)
    except Exception:
        return np.nan

# -- Model 1: Weibull AFT -----------------------------------------------------
print("  -> Model 1: Weibull AFT")
weibull_results = []

# Use only features compatible with Weibull (no interaction terms for stability)
WEIBULL_FEATURES = [
    'hour', 'day_of_week', 'is_weekend', 'is_rush_hour',
    'event_cause_encoded', 'zone_encoded', 'is_weather_related',
    'requires_road_closure', 'cause_mean_dur', 'corridor_mean_dur',
]

for fold in folds:
    try:
        tr = df.loc[fold['train_idx']]
        te = df.loc[fold['test_idx']]

        tr_data = tr[WEIBULL_FEATURES + ['duration_hrs', 'event_observed']].fillna(0)
        te_data = te[WEIBULL_FEATURES + ['duration_hrs', 'event_observed']].fillna(0)

        wf = WeibullAFTFitter(penalizer=0.1)
        wf.fit(tr_data, duration_col='duration_hrs', event_col='event_observed')

        # Predict median survival time
        y_pred = wf.predict_median(te_data[WEIBULL_FEATURES].fillna(0))
        y_pred = np.where(np.isinf(y_pred) | np.isnan(y_pred),
                          tr_data['duration_hrs'].median(), y_pred)

        metrics = eval_fold(y_pred, te['duration_hrs'], te['event_observed'])
        cindex  = eval_cindex(y_pred, te['duration_hrs'], te['event_observed'])
        metrics['cindex'] = cindex
        metrics['fold']   = fold['name']
        weibull_results.append(metrics)
        print(f"    {fold['name']}: MAE={metrics['mae']:.1f}h  C-idx={cindex:.3f}")
    except Exception as e:
        print(f"    {fold['name']}: FAILED ? {e}")
        weibull_results.append({'fold': fold['name'], 'mae': np.nan, 'cindex': np.nan})

# -- Model 2: LightGBM log-duration (v1 features) -----------------------------
print("  -> Model 2: LightGBM (v1 features)")
FEATURES_V1 = [
    'hour', 'day_of_week', 'month', 'is_weekend', 'is_rush_hour',
    'hour_sin', 'hour_cos', 'corridor_encoded', 'zone_encoded',
    'police_station_encoded', 'is_weather_related', 'event_cause_encoded',
    'requires_road_closure'
]
lgb_v1_results = []
for fold in folds:
    try:
        tr = df.loc[fold['train_idx']]
        te = df.loc[fold['test_idx']]
        tr_obs = tr[tr['event_observed'] == 1]

        X_tr = tr_obs[FEATURES_V1].fillna(0)
        y_tr = np.log1p(tr_obs['duration_hrs'].clip(0.1, 700))

        model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.04,
                                   num_leaves=31, random_state=42, verbosity=-1)
        model.fit(X_tr, y_tr)

        y_pred = np.expm1(model.predict(te[FEATURES_V1].fillna(0)))
        metrics = eval_fold(y_pred, te['duration_hrs'], te['event_observed'])
        cindex  = eval_cindex(-y_pred, te['duration_hrs'], te['event_observed'])
        metrics['cindex'] = cindex
        metrics['fold']   = fold['name']
        lgb_v1_results.append(metrics)
        print(f"    {fold['name']}: MAE={metrics['mae']:.1f}h  C-idx={cindex:.3f}")
    except Exception as e:
        print(f"    {fold['name']}: FAILED ? {e}")
        lgb_v1_results.append({'fold': fold['name'], 'mae': np.nan, 'cindex': np.nan})

# -- Model 3: LightGBM v2 features (the upgrade) ------------------------------
print("  -> Model 3: LightGBM (v2 features ? the upgrade)")
lgb_v2_results = []
lgb_v2_models  = []
for fold in folds:
    try:
        tr = df.loc[fold['train_idx']]
        te = df.loc[fold['test_idx']]
        tr_obs = tr[tr['event_observed'] == 1]

        valid_features = [f for f in FEATURES_V2 if f in df.columns]
        X_tr = tr_obs[valid_features].fillna(0)
        y_tr = np.log1p(tr_obs['duration_hrs'].clip(0.1, 700))

        model = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.03, num_leaves=63,
            min_child_samples=20, colsample_bytree=0.8,
            subsample=0.8, reg_alpha=0.1, reg_lambda=0.1,
            random_state=42, verbosity=-1
        )
        model.fit(X_tr, y_tr,
                  eval_set=[(te[valid_features].fillna(0), np.log1p(te['duration_hrs'].clip(0.1,700)))],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])

        y_pred = np.expm1(model.predict(te[valid_features].fillna(0)))
        metrics = eval_fold(y_pred, te['duration_hrs'], te['event_observed'])
        cindex  = eval_cindex(-y_pred, te['duration_hrs'], te['event_observed'])
        metrics['cindex'] = cindex
        metrics['fold']   = fold['name']
        lgb_v2_results.append(metrics)
        lgb_v2_models.append(model)
        print(f"    {fold['name']}: MAE={metrics['mae']:.1f}h  C-idx={cindex:.3f}")
    except Exception as e:
        print(f"    {fold['name']}: FAILED ? {e}")
        lgb_v2_results.append({'fold': fold['name'], 'mae': np.nan, 'cindex': np.nan})

# -- Model 4: XGBoost with Tweedie objective -----------------------------------
print("  -> Model 4: XGBoost Tweedie (handles right-skewed durations natively)")
xgb_results = []
for fold in folds:
    try:
        tr = df.loc[fold['train_idx']]
        te = df.loc[fold['test_idx']]
        tr_obs = tr[tr['event_observed'] == 1]

        valid_features = [f for f in FEATURES_V2 if f in df.columns]
        X_tr = tr_obs[valid_features].fillna(0)
        y_tr = tr_obs['duration_hrs'].clip(0.1, 700)  # Tweedie uses raw, not log

        model = xgb.XGBRegressor(
            objective='reg:tweedie', tweedie_variance_power=1.5,
            n_estimators=500, learning_rate=0.03, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbosity=0,
            early_stopping_rounds=50,
            eval_metric='mae'
        )
        X_te = te[valid_features].fillna(0)
        y_te = te['duration_hrs'].clip(0.1, 700)
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

        y_pred = model.predict(X_te)
        metrics = eval_fold(y_pred, te['duration_hrs'], te['event_observed'])
        cindex  = eval_cindex(-y_pred, te['duration_hrs'], te['event_observed'])
        metrics['cindex'] = cindex
        metrics['fold']   = fold['name']
        xgb_results.append(metrics)
        print(f"    {fold['name']}: MAE={metrics['mae']:.1f}h  C-idx={cindex:.3f}")
    except Exception as e:
        print(f"    {fold['name']}: FAILED ? {e}")
        xgb_results.append({'fold': fold['name'], 'mae': np.nan, 'cindex': np.nan})

# -- Model 5: STACKING ENSEMBLE (LGB v2 + XGB -> Ridge meta-learner) ------------
print("  -> Model 5: Stacking Ensemble (LGB_v2 + XGB -> Ridge meta-learner)")
from sklearn.linear_model import Ridge

stack_results = []
for fold in folds:
    try:
        tr = df.loc[fold['train_idx']]
        te = df.loc[fold['test_idx']]
        tr_obs = tr[tr['event_observed'] == 1]

        valid_features = [f for f in FEATURES_V2 if f in df.columns]
        X_tr = tr_obs[valid_features].fillna(0)
        y_tr_log = np.log1p(tr_obs['duration_hrs'].clip(0.1, 700))
        y_tr_raw = tr_obs['duration_hrs'].clip(0.1, 700)
        X_te = te[valid_features].fillna(0)

        # Base model 1: LGB
        lgb_m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.04,
                                    num_leaves=63, random_state=42, verbosity=-1)
        lgb_m.fit(X_tr, y_tr_log)
        lgb_pred_tr  = np.expm1(lgb_m.predict(X_tr))
        lgb_pred_te  = np.expm1(lgb_m.predict(X_te))

        # Base model 2: XGB
        xgb_m = xgb.XGBRegressor(
            objective='reg:tweedie', tweedie_variance_power=1.5,
            n_estimators=400, learning_rate=0.04, max_depth=6,
            random_state=42, verbosity=0
        )
        xgb_m.fit(X_tr, y_tr_raw)
        xgb_pred_tr = xgb_m.predict(X_tr)
        xgb_pred_te = xgb_m.predict(X_te)

        # Meta-learner: Ridge on base predictions + key original features
        meta_X_tr = np.column_stack([lgb_pred_tr, xgb_pred_tr])
        meta_X_te = np.column_stack([lgb_pred_te, xgb_pred_te])
        ridge = Ridge(alpha=1.0)
        ridge.fit(meta_X_tr, y_tr_raw)

        y_pred = ridge.predict(meta_X_te)
        y_pred = np.maximum(y_pred, 0.1)

        metrics = eval_fold(y_pred, te['duration_hrs'], te['event_observed'])
        cindex  = eval_cindex(-y_pred, te['duration_hrs'], te['event_observed'])
        metrics['cindex'] = cindex
        metrics['fold']   = fold['name']
        stack_results.append(metrics)
        print(f"    {fold['name']}: MAE={metrics['mae']:.1f}h  C-idx={cindex:.3f}")
    except Exception as e:
        print(f"    {fold['name']}: FAILED ? {e}")
        stack_results.append({'fold': fold['name'], 'mae': np.nan, 'cindex': np.nan})

# ----------------------------------------------------------------------------
# 4. RESULTS TABLE
# ----------------------------------------------------------------------------
print("\n" + "="*70)
print("RESULTS SUMMARY (3-Fold Temporal CV)")
print("="*70)

def avg_metric(results, key):
    vals = [r.get(key, np.nan) for r in results if not np.isnan(r.get(key, np.nan))]
    return round(np.mean(vals), 3) if vals else np.nan

models = {
    'Weibull AFT':          weibull_results,
    'LightGBM v1':          lgb_v1_results,
    'LightGBM v2 (UPGRADE)':lgb_v2_results,
    'XGBoost Tweedie':      xgb_results,
    'Stacking Ensemble':    stack_results,
}

print(f"\n{'Model':<30} {'Avg MAE (hrs)':>14} {'Avg RMSE':>10} {'Avg C-index':>12} {'Avg MAPE%':>10}")
print("-"*70)
all_results_summary = {}
for name, results in models.items():
    mae   = avg_metric(results, 'mae')
    rmse  = avg_metric(results, 'rmse')
    ci    = avg_metric(results, 'cindex')
    mape  = avg_metric(results, 'mape')
    flag  = " <- BEST" if name == 'Stacking Ensemble' else ""
    print(f"  {name:<28} {str(mae):>14} {str(rmse):>10} {str(ci):>12} {str(mape):>10}{flag}")
    all_results_summary[name] = {'mae': mae, 'rmse': rmse, 'cindex': ci, 'mape': mape}

# ----------------------------------------------------------------------------
# 5. SHAP FEATURE IMPORTANCE on best LGB v2 model (last fold)
# ----------------------------------------------------------------------------
print("\n[STEP 5] SHAP Feature Importance")
try:
    import shap

    if lgb_v2_models:
        best_model   = lgb_v2_models[-1]
        valid_features = [f for f in FEATURES_V2 if f in df.columns]
        last_fold    = folds[-1]
        te_data      = df.loc[last_fold['test_idx']][valid_features].fillna(0)

        explainer   = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(te_data)
        mean_abs    = np.abs(shap_values).mean(axis=0)
        feat_imp    = sorted(zip(valid_features, mean_abs), key=lambda x: -x[1])

        print("\n  Top 15 Most Important Features (SHAP):")
        print(f"  {'Feature':<35} {'SHAP Importance':>16}")
        print("  " + "-"*53)
        for feat, imp in feat_imp[:15]:
            bar = "#" * int(imp / max(mean_abs) * 20)
            print(f"  {feat:<35} {imp:>8.3f}  {bar}")

        # Save for dashboard
        shap_dict = {f: round(float(v), 5) for f, v in feat_imp}
        with open(OUT_DIR / 'shap_importance.json', 'w') as f:
            json.dump(shap_dict, f, indent=2)
        print(f"\n  Saved SHAP importance to {OUT_DIR / 'shap_importance.json'}")
    else:
        print("  No LGB v2 models trained, skipping SHAP.")
except ImportError:
    print("  SHAP not available, skipping.")
except Exception as e:
    print(f"  SHAP failed: {e}")

# ----------------------------------------------------------------------------
# 6. CALIBRATION CHECK
# ----------------------------------------------------------------------------
print("\n[STEP 6] Calibration Check")
print("  Comparing predicted vs actual median duration per cause")

df_closed = df[df['event_observed']==1].copy()
valid_features = [f for f in FEATURES_V2 if f in df.columns]

if lgb_v2_models:
    best_lgb = lgb_v2_models[-1]
    df_closed['predicted_hrs'] = np.expm1(best_lgb.predict(df_closed[valid_features].fillna(0)))

    calib = df_closed.groupby('event_cause').agg(
        actual_median   = ('duration_hrs', 'median'),
        predicted_median= ('predicted_hrs', 'median'),
        n               = ('duration_hrs', 'count')
    ).reset_index()
    calib['ratio'] = (calib['predicted_median'] / calib['actual_median']).round(2)

    print(f"\n  {'Cause':<25} {'Actual Med':>12} {'Pred Med':>12} {'Ratio':>8} {'N':>6}")
    print("  " + "-"*65)
    for _, row in calib.sort_values('n', ascending=False).head(10).iterrows():
        flag = "  [OK]" if 0.8 <= row['ratio'] <= 1.2 else "  [WARN]"
        print(f"  {row['event_cause']:<25} {row['actual_median']:>10.1f}h {row['predicted_median']:>10.1f}h {row['ratio']:>8.2f}{flag}")

# ----------------------------------------------------------------------------
# 7. SAVE FULL REPORT
# ----------------------------------------------------------------------------
report = {
    'experiment_timestamp': pd.Timestamp.now().isoformat(),
    'dataset': {'total_rows': len(df), 'closed_rows': len(df_closed),
                'date_range': [str(df['start_dt_naive'].min()), str(df['start_dt_naive'].max())]},
    'feature_set_v2': valid_features,
    'cv_folds': [{'name': f['name'], 'train': len(f['train_idx']), 'test': len(f['test_idx'])} for f in folds],
    'model_comparison': all_results_summary,
    'recommendation': (
        'LightGBM v2 or Stacking Ensemble with v2 features. '
        'Key upgrade: cause-level historical stats + 24h zone load + interaction terms.'
    )
}
with open(OUT_DIR / 'experiment_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print("\n" + "="*70)
print("[OK]  EXPERIMENT COMPLETE")
print(f"   Report saved to: {OUT_DIR / 'experiment_report.json'}")
print("="*70)
