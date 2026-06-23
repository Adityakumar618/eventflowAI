"""
Impact v1p5 Smoke Experiment: Value of Shared V9-style LOO + Cluster Features
============================================================================
Phase 1-4 from the plan.

Goal (smoke but GM-minded):
- Define a composite congestion_impact target.
- Compare two feature sets using the new core_features + advanced_event_fe:
  1. Basic (use_v9_style_loo=False) — our previous isolated impact FE.
  2. Enhanced (use_v9_style_loo=True) — adds high-quality LOO (corridor, officer, cluster etc.) + cluster signals from shared core.
- Use purged temporal CV via harness.
- Strong baseline: V9 duration predictions + simple heuristic for impact.
- Metrics overall + on planned regime.
- Everything additive. Does not modify any V9 code/models.

This is the first measurement of whether harvesting V9 brilliance helps the impact problem.

Run with:
  python experiments/impact_v1p5_shared_features.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import time
from sklearn.metrics import mean_absolute_error

from inference import GridGuardV9Inference
from advanced_event_fe import EventImpactFeatureEngineer, PLANNED_CAUSES
from impact_harness import purged_temporal_splits, evaluate_impact, log_experiment

BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "data" / "processed" / "survival_ready.parquet"
RAW = BASE / "data" / "raw" / "astram_events.csv"

print("=" * 70)
print("IMPACT v1p5 — SHARED V9-STYLE FEATURES SMOKE (Phase 1-4)")
print("Does harvesting V9 LOO/Cluster features improve congestion impact prediction?")
print("=" * 70)

# === 1. Load & prepare data ===
print("\n[1] Loading data...")
df = pd.read_parquet(PROCESSED)
df = df.sort_values("start_datetime").reset_index(drop=True)

# Ensure key columns
if "event_type" not in df.columns:
    df["event_type"] = "unplanned"  # fallback
df["is_planned"] = (df["event_type"] == "planned").astype(int)
df["regime"] = df["is_planned"].map({1: "planned", 0: "unplanned"})

# Build observed duration if missing (safety)
if "duration_hrs" not in df.columns:
    df["duration_hrs"] = 2.0

# === 2. Define composite impact target (business-oriented, respecting censoring lessons) ===
# Use only observed <=48h events for stats. CIU proxies network effect.
# Critically: clip durations like V9/survival pipelines do.
corr_counts = df["corridor"].value_counts().to_dict()
df["corridor_centrality"] = df["corridor"].map(lambda x: corr_counts.get(x, 10) / max(corr_counts.values())).clip(0.05, 1.0)

def build_impact_target(dframe):
    # Respect observed + cap like the rest of the project
    obs = dframe.get("event_observed", pd.Series(1, index=dframe.index)) == 1
    dur = dframe["duration_hrs"].clip(0.1, 48.0)
    closure = dframe.get("requires_road_closure", 0).fillna(0).astype(float)
    centrality = dframe["corridor_centrality"]
    is_rush = dframe.get("is_rush", pd.Series(0, index=dframe.index)).fillna(0).astype(float)
    rush_mult = 1.0 + 0.4 * is_rush
    
    impact = dur * (1.0 + 1.6 * closure) * centrality * rush_mult
    impact = impact.where(obs, impact * 0.6)  # downweight heavily censored
    return impact.clip(0.2, 25.0)   # realistic operational range for impact proxy

df["impact_target"] = build_impact_target(df)
df["log_impact"] = np.log1p(df["impact_target"])

print(f"Impact target stats: mean={df['impact_target'].mean():.2f}, median={df['impact_target'].median():.2f}")
print(f"Planned fraction: {df['is_planned'].mean()*100:.1f}%")

# === 3. Generate strong V9 duration baseline (for comparison) ===
print("\n[2] Generating V9 duration predictions as strong baseline (inference only)...")
v9_engine = GridGuardV9Inference(enable_mappls=False)

def get_v9_dur(row):
    try:
        p = v9_engine.predict({
            "event_cause": row.get("event_cause", "vehicle_breakdown"),
            "corridor": row.get("corridor", "Non-corridor"),
            "zone": row.get("zone", "unknown"),
            "hour": int(row.get("hour", 12)),
            "lat": float(row.get("latitude", 13.0)),
            "lon": float(row.get("longitude", 77.6)),
            "requires_road_closure": bool(row.get("requires_road_closure", False)),
            "priority": row.get("priority", "Low"),
            "description": str(row.get("description", "")),
            "police_station": row.get("police_station", "unknown"),
        })
        return p["predicted_hours"]
    except Exception:
        return 2.0

# For smoke speed we sample, but use full for fairness on small data
df["v9_pred_dur"] = df.apply(get_v9_dur, axis=1)

# Heuristic impact from V9 dur (strong baseline)
df["v9_heuristic_impact"] = (
    df["v9_pred_dur"] * 
    (1 + 1.8 * df["requires_road_closure"].fillna(0)) * 
    df["corridor_centrality"] * 
    (1.3 if "is_rush" in df else 1.0)
).clip(0.2, 60)

print("V9 heuristic impact MAE vs true impact (full data):",
      round(mean_absolute_error(df["impact_target"], df["v9_heuristic_impact"]), 3))

# === 4. Feature functions (basic vs enhanced) ===
print("\n[3] Preparing feature functions...")

def basic_features(d):
    fe = EventImpactFeatureEngineer(use_v9_style_loo=False)
    fe.fit(d.iloc[: max(50, len(d)//2)])
    out = fe.transform(d)
    feats = [c for c in fe.get_feature_names() if c in out.columns]
    return out[feats].fillna(0), feats   # return DataFrame

def enhanced_features(d):
    fe = EventImpactFeatureEngineer(use_v9_style_loo=True)
    fe.fit(d.iloc[: max(50, len(d)//2)])
    out = fe.transform(d)
    feats = [c for c in fe.get_feature_names() if c in out.columns]
    return out[feats].fillna(0), feats   # return DataFrame

# === 5. CV Experiment ===
print("\n[4] Running purged temporal CV...")

n_splits = 4
splits = purged_temporal_splits(df, n_splits=n_splits, purge_days=1)

results = []

for name, feat_fn in [("basic", basic_features), ("enhanced", enhanced_features)]:
    fold_maes = []
    fold_planned_maes = []
    
    for fold, (tr_idx, te_idx) in enumerate(splits):
        X_tr_df, feat_names = feat_fn(df.iloc[tr_idx])
        X_te_df, _ = feat_fn(df.iloc[te_idx])
        
        y_tr = df.iloc[tr_idx]["log_impact"].values
        y_te = df.iloc[te_idx]["impact_target"].values
        
        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=48,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.2,
            random_state=42 + fold,
            verbosity=-1
        )
        model.fit(X_tr_df, y_tr)
        pred_log = model.predict(X_te_df)
        pred = np.expm1(np.clip(pred_log, -1, 5))
        
        mae = mean_absolute_error(y_te, pred)
        fold_maes.append(mae)
        
        planned_mask = df.iloc[te_idx]["is_planned"] == 1
        if planned_mask.sum() > 3:
            pmae = mean_absolute_error(y_te[planned_mask], pred[planned_mask])
            fold_planned_maes.append(pmae)
        
        print(f"  {name} fold {fold}: overall MAE={mae:.3f} | planned_n={planned_mask.sum()}")
    
    avg_mae = np.mean(fold_maes)
    avg_planned = np.mean(fold_planned_maes) if fold_planned_maes else np.nan
    results.append({
        "name": name,
        "avg_mae": round(avg_mae, 3),
        "avg_planned_mae": round(avg_planned, 3) if not np.isnan(avg_planned) else "N/A",
        "n_folds": len(fold_maes)
    })

# === 6. V9 heuristic baseline on same folds ===
print("\n[5] V9 heuristic baseline on same CV folds...")
v9_fold_maes = []
v9_planned_maes = []

for tr_idx, te_idx in splits:
    y_te = df.iloc[te_idx]["impact_target"].values
    v9_imp = df.iloc[te_idx]["v9_heuristic_impact"].values
    v9_fold_maes.append(mean_absolute_error(y_te, v9_imp))
    
    planned_mask = df.iloc[te_idx]["is_planned"] == 1
    if planned_mask.sum() > 3:
        v9_planned_maes.append(mean_absolute_error(y_te[planned_mask], v9_imp[planned_mask]))

results.append({
    "name": "v9_heuristic_baseline",
    "avg_mae": round(np.mean(v9_fold_maes), 3),
    "avg_planned_mae": round(np.mean(v9_planned_maes), 3) if v9_planned_maes else "N/A",
    "n_folds": len(v9_fold_maes)
})

print("\n=== RESULTS ===")
for r in results:
    print(f"{r['name']:25s}  overall_MAE={r['avg_mae']}   planned_MAE={r['avg_planned_mae']}")

# === 7. Save ===
exp_dir = log_experiment(
    "impact_v1p5_shared_features",
    {"results": results},
    {"n_splits": n_splits, "target": "log_impact", "model": "LGBM regression"},
    ["basic_vs_enhanced_vs_v9_heuristic"],
)

print(f"\nResults logged to {exp_dir}")
print("Smoke experiment complete.")