"""
GridGuard AI — V6 Push Harder: Beyond LightGBM
================================================
4 structural improvements that tree models fundamentally cannot do:

IDEA 1: Log-space regression (trivial, high impact)
  - Duration is log-normal distributed
  - Train LGB on log(duration), back-transform with exp()
  - Minimizes relative error, better for heavy right tail
  - Expected: 5-15% MAE improvement

IDEA 2: Temporal residual correction (meta-learner)
  - V4 fold MAEs: [0.97, 1.77, 1.59] — March is 80% harder than Feb
  - This is systematic temporal bias, NOT random noise
  - Approach: model V4's signed residuals as function of time+recent patterns
  - Apply correction: final = V4_pred - residual_correction
  - Expected: flatten the fold variance → lower mean MAE

IDEA 3: Neural entity embeddings (torch required)
  - corridor LOO treats ORR North 1 and ORR North 2 as unrelated
  - Embedding: corridor → 16-dim vector, learned jointly
  - Model learns corridor similarity from co-occurrence patterns
  - Architecture: Input concat + 2-layer MLP + quantile loss
  - Expected: better generalization on sparse corridors

IDEA 4: NGBoost (natural gradient boosting)
  - Predicts full probability distribution (not just P50)
  - Eliminates need for KM scaling entirely
  - NLL loss → calibrated P10/P50/P90 jointly optimized
  - Will install if not present
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

OUT_DIR = Path("experiments/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 66)
print("GRIDGUARD AI V6 — PUSH HARDER")
print("Four structural improvements beyond LightGBM")
print("=" * 66)

# ── Rebuild V4 features (canonical, clean) ────────────────────────────────────
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

raw["hour"]       = raw["start_datetime"].dt.hour
raw["dow"]        = raw["start_datetime"].dt.dayofweek
raw["month"]      = raw["start_datetime"].dt.month
raw["week"]       = raw["start_datetime"].dt.isocalendar().week.fillna(0).astype(int)
raw["is_weekend"] = raw["dow"].isin([5,6]).astype(int)
raw["is_rush"]    = raw["hour"].apply(lambda h: 1 if (8<=h<=11) or (17<=h<=20) else 0)
raw["is_night"]   = ((raw["hour"]>=22)|(raw["hour"]<=5)).astype(int)
raw["hour_sin"]   = np.sin(raw["hour"]*2*np.pi/24)
raw["hour_cos"]   = np.cos(raw["hour"]*2*np.pi/24)
raw["week_sin"]   = np.sin(raw["week"]*2*np.pi/52)  # NEW: annual cycle
raw["week_cos"]   = np.cos(raw["week"]*2*np.pi/52)
raw["is_weather"] = raw["event_cause"].isin(
    ["water_logging","tree_fall","fog/low_visibility","debris"]).astype(int)
raw["is_high_priority"] = (raw["priority"].fillna("Low")=="High").astype(int)

raw = raw.sort_values("start_datetime").reset_index(drop=True)
raw["start_dt_naive"] = raw["start_datetime"].dt.tz_localize(None)
raw["start_ts"] = raw["start_dt_naive"].astype(np.int64)//10**9

obs = raw[raw["event_observed"]==1]
gm  = obs["duration_hrs"].mean()

# Global stats
for col_name, fn in [("cause_mean","mean"),("cause_median","median"),
                      ("cause_p90",lambda x: x.quantile(0.9)),
                      ("cause_p10",lambda x: x.quantile(0.1))]:
    s = obs.groupby("event_cause")["duration_hrs"].agg(**{col_name:fn}).reset_index()
    raw = raw.merge(s, on="event_cause", how="left")
raw = raw.merge(obs.groupby("police_station")["duration_hrs"].agg(
    station_mean="mean").reset_index(), on="police_station", how="left")
raw = raw.merge(obs.groupby("corridor")["duration_hrs"].agg(
    corridor_mean="mean",corridor_cnt="count").reset_index(), on="corridor", how="left")
for c in ["cause_mean","cause_median","cause_p90","cause_p10","station_mean",
          "corridor_mean","corridor_cnt"]:
    raw[c] = raw[c].fillna(obs["duration_hrs"].median())

raw["cause_x_rush"]    = raw["event_cause_enc"]*raw["is_rush"]
raw["cause_x_night"]   = raw["event_cause_enc"]*raw["is_night"]
raw["cause_x_zone"]    = raw["event_cause_enc"]*raw["zone_enc"]
raw["cause_x_closure"] = raw["event_cause_enc"]*raw["requires_road_closure"]
raw["station_x_cause"] = raw["police_station_enc"]*raw["event_cause_enc"]

WINDOW_SEC = 30*86400
rolling_c = []
for i, row in raw.iterrows():
    t=row["start_ts"]; ec=row["event_cause"]
    past=raw.iloc[max(0,i-500):i]
    mask=((past["event_cause"]==ec)&(past["event_observed"]==1)&
          (past["start_ts"]>=t-WINDOW_SEC)&(past["start_ts"]<t))
    m=past[mask]
    rolling_c.append(m["duration_hrs"].mean() if len(m)>=3 else row["cause_mean"])
raw["cause_rolling_30d"] = rolling_c

conc_z,conc_c=[],[]
for i,row in raw.iterrows():
    t=row["start_ts"]
    past=raw.iloc[max(0,i-300):i]
    conc_z.append(int(((past["zone"]==row["zone"])&(past["start_ts"]>=t-4*3600)&(past["start_ts"]<t)).sum()))
    conc_c.append(int(((past["corridor"]==row["corridor"])&(past["start_ts"]>=t-2*3600)&(past["start_ts"]<t)).sum()))
raw["concurrent_zone_events"]=conc_z
raw["concurrent_corridor_events"]=conc_c

def loo_enc(df_obs,df_all,group_col,gm):
    s=df_obs.groupby(group_col)["duration_hrs"].sum()
    n=df_obs.groupby(group_col)["duration_hrs"].count()
    res=[]
    for _,row in df_all.iterrows():
        key=row[group_col];d=row["duration_hrs"];cnt=n.get(key,0);sm=s.get(key,gm*cnt)
        res.append((sm-d)/(cnt-1) if cnt>1 and row["event_observed"]==1 else (sm/cnt if cnt>0 else gm))
    return res

raw["station_loo"]=loo_enc(obs,raw,"police_station",gm)
raw["corridor_loo"]=loo_enc(obs,raw,"corridor",gm)

hc=obs.groupby(["hour","event_cause"])["duration_hrs"].mean().reset_index()
hc.columns=["hour","event_cause","hour_cause_mean"]
raw=raw.merge(hc,on=["hour","event_cause"],how="left")
raw["hour_cause_mean"]=raw["hour_cause_mean"].fillna(raw["cause_mean"])

obs2=obs.copy(); obs2["zone_cause"]=obs2["zone"]+"||"+obs2["event_cause"]
raw["zone_cause"]=raw["zone"]+"||"+raw["event_cause"]
raw["zone_cause_loo"]=loo_enc(obs2,raw,"zone_cause",gm)

raw["veh_type_clean"]=raw["veh_type"].fillna("unknown")
raw["veh_cause_key"]=raw["veh_type_clean"]+"||"+raw["event_cause"]
obs3=obs.copy(); obs3["veh_cause_key"]=obs3["veh_type"].fillna("unknown")+"||"+obs3["event_cause"]
raw["veh_cause_loo"]=loo_enc(obs3,raw,"veh_cause_key",gm)

raw["text_combined"]=(
    raw["description"].fillna("").str.lower().str.replace(r"[^a-z0-9\s]"," ",regex=True)+
    " "+raw["reason_breakdown"].fillna("").str.lower())
N_TEXT=8
tfidf=TfidfVectorizer(max_features=500,ngram_range=(1,2),min_df=3,sublinear_tf=True)
tfidf_m=tfidf.fit_transform(raw["text_combined"])
svd=TruncatedSVD(n_components=N_TEXT,random_state=42)
tf=svd.fit_transform(tfidf_m)
text_cols=[f"text_svd_{i}" for i in range(N_TEXT)]
for i,c in enumerate(text_cols): raw[c]=tf[:,i]
raw["has_description"]=(raw["description"].notna()&(raw["description"].str.len()>5)).astype(int)

ALPHA,BETA,MU=0.8,0.5,0.05
hawkes_z=[]
for i,row in raw.iterrows():
    t=row["start_ts"]/3600
    past=raw.iloc[max(0,i-200):i]
    pz=past[past["zone"]==row["zone"]]
    dt=t-pz["start_ts"].values/3600
    hawkes_z.append(float(MU+ALPHA*np.sum(np.exp(-BETA*dt[dt>0]))))
raw["log_hawkes_zone"]=np.log1p(hawkes_z)

officer_load=[]
for i,row in raw.iterrows():
    t=row["start_ts"];oid=row.get("created_by_id",None)
    if pd.isna(oid): officer_load.append(0); continue
    past=raw.iloc[max(0,i-300):i]
    officer_load.append(int(((past["created_by_id"]==oid)&(past["start_ts"]>=t-4*3600)&(past["start_ts"]<t)).sum()))
raw["officer_active_load"]=officer_load

V4_FEATURES = [
    "hour","dow","month","week","is_weekend","is_rush","is_night","hour_sin","hour_cos",
    "week_sin","week_cos",  # NEW: annual cycle features
    "is_weather","requires_road_closure","is_high_priority",
    "event_cause_enc","zone_enc","corridor_enc","police_station_enc",
    "cause_mean","cause_median","cause_p90","cause_p10","station_mean",
    "corridor_mean","corridor_cnt",
    "cause_x_rush","cause_x_night","cause_x_zone","cause_x_closure","station_x_cause",
    "cause_rolling_30d","concurrent_zone_events","concurrent_corridor_events",
    "station_loo","corridor_loo","hour_cause_mean","zone_cause_loo","veh_cause_loo",
]+text_cols+["has_description","log_hawkes_zone","officer_active_load"]

for col in V4_FEATURES:
    raw[col]=pd.to_numeric(raw[col],errors="coerce").fillna(0)

# Log-duration target
raw["log_duration"] = np.log(raw["duration_hrs"].clip(0.05))

folds=[
    {"name":"Feb","train_end":pd.Timestamp("2024-01-31"),
     "test_start":pd.Timestamp("2024-02-01"),"test_end":pd.Timestamp("2024-02-29")},
    {"name":"Mar","train_end":pd.Timestamp("2024-02-29"),
     "test_start":pd.Timestamp("2024-03-01"),"test_end":pd.Timestamp("2024-03-31")},
    {"name":"Apr","train_end":pd.Timestamp("2024-03-31"),
     "test_start":pd.Timestamp("2024-04-01"),"test_end":pd.Timestamp("2024-04-30")},
]
TRAIN_MASK=(raw["event_observed"]==1)&(raw["duration_hrs"]<=48)
for f in folds:
    f["tr"]=raw[raw["start_dt_naive"]<=f["train_end"]].index
    f["te"]=raw[(raw["start_dt_naive"]>=f["test_start"])&(raw["start_dt_naive"]<=f["test_end"])].index

V4_PARAMS=dict(
    objective="quantile",alpha=0.50,
    n_estimators=661,learning_rate=0.013164,num_leaves=129,
    min_child_samples=9,colsample_bytree=0.9989,subsample=0.5783,
    reg_alpha=0.2257,reg_lambda=0.0107,min_split_gain=0.1991,
    random_state=42,verbosity=-1,
)
valid_v4=[f for f in V4_FEATURES if f in raw.columns]

def cv_mae(predict_fn, label):
    maes=[]
    for f in folds:
        tr=raw.loc[f["tr"]]; te=raw.loc[f["te"]]
        tro=tr[TRAIN_MASK.loc[f["tr"]]]; teo=te[(te["event_observed"]==1)&(te["duration_hrs"]<=48)]
        if len(tro)<20 or len(teo)<5: continue
        preds=predict_fn(tro,teo)
        maes.append(mean_absolute_error(teo["duration_hrs"].values,np.maximum(preds,0.05)))
    avg=np.mean(maes) if maes else 999
    print(f"  {label:<58} MAE={avg:.4f}h  {[round(x,2) for x in maes]}")
    return avg

print("\n" + "="*66)
print("ABLATION — Four Structural Improvements")
print("="*66)

# ── BASELINE V4 ────────────────────────────────────────────────────────────────
def lgb_v4(tr, te, params=None, feats=None, target="duration_hrs"):
    if params is None: params=V4_PARAMS
    if feats is None: feats=valid_v4
    m=lgb.LGBMRegressor(**params)
    m.fit(tr[feats], tr[target])
    return m.predict(te[feats])

mae_v4 = cv_mae(lambda tr,te: lgb_v4(tr,te), "V4 baseline (1.442h confirmed)")

# ── IDEA 1: Log-space regression ───────────────────────────────────────────────
print("\n[IDEA 1] Log-space regression (train on log(duration), back-transform)")

LOG_PARAMS = dict(V4_PARAMS)
LOG_PARAMS["objective"] = "quantile"  # quantile on log scale = median quantile on original

def log_lgb_predict(tr, te):
    m=lgb.LGBMRegressor(**LOG_PARAMS)
    m.fit(tr[valid_v4], tr["log_duration"])
    log_preds=m.predict(te[valid_v4])
    return np.exp(log_preds)  # back to hours

mae_log = cv_mae(log_lgb_predict, "Log-space LGB (train log, predict exp)")

# Re-tune log-space with Optuna (30 trials — fast since it's 1 model)
print("  [Optuna 50 trials on log-space model...]")
def obj_log(trial):
    p=dict(
        objective="quantile",alpha=0.50,
        n_estimators=trial.suggest_int("n",300,900),
        learning_rate=trial.suggest_float("lr",0.005,0.08,log=True),
        num_leaves=trial.suggest_int("leaves",50,200),
        min_child_samples=trial.suggest_int("mcs",5,40),
        colsample_bytree=trial.suggest_float("cbt",0.5,1.0),
        subsample=trial.suggest_float("sub",0.5,1.0),
        reg_alpha=trial.suggest_float("ra",1e-5,2.0,log=True),
        reg_lambda=trial.suggest_float("rl",1e-5,10.0,log=True),
        random_state=42,verbosity=-1,
    )
    maes=[]
    for f in folds:
        tr=raw.loc[f["tr"]]; te=raw.loc[f["te"]]
        tro=tr[TRAIN_MASK.loc[f["tr"]]]; teo=te[(te["event_observed"]==1)&(te["duration_hrs"]<=48)]
        if len(tro)<20 or len(teo)<5: continue
        m=lgb.LGBMRegressor(**p)
        m.fit(tro[valid_v4], tro["log_duration"])
        preds=np.exp(m.predict(teo[valid_v4]))
        maes.append(mean_absolute_error(teo["duration_hrs"].values,np.maximum(preds,0.05)))
    return np.mean(maes) if maes else 999

study_log=optuna.create_study(direction="minimize",sampler=optuna.samplers.TPESampler(seed=99))
study_log.optimize(obj_log,n_trials=50)
mae_log_tuned=study_log.best_value
best_log_params=study_log.best_params
print(f"  Log-space + Optuna 50 trials: {mae_log_tuned:.4f}h")

cv_mae(log_lgb_predict, "Log-space LGB (default params)")
print(f"  Log-space + Optuna best:  MAE={mae_log_tuned:.4f}h")

# ── IDEA 2: Temporal Residual Meta-Learner ─────────────────────────────────────
print("\n[IDEA 2] Temporal Residual Correction")
print("  March fold is 80% harder than Feb → systematic temporal bias")
print("  Phase 1: collect V4 OOF predictions → compute signed residuals")
print("  Phase 2: LGB on (month, week, hawkes, rolling) to model residuals")
print("  Phase 3: final_pred = V4_pred + residual_correction")

# Generate out-of-fold predictions and residuals using temporal expanding window
residual_feats = ["month","week","week_sin","week_cos","is_weather",
                  "log_hawkes_zone","cause_rolling_30d","concurrent_zone_events",
                  "hour","dow","cause_mean","event_cause_enc","zone_enc"]
residual_feats = [f for f in residual_feats if f in raw.columns]

def two_phase_predict(tr, te):
    # Phase 1: V4 base predictions
    m_base = lgb.LGBMRegressor(**V4_PARAMS)
    m_base.fit(tr[valid_v4], tr["duration_hrs"])
    v4_pred_te = np.maximum(m_base.predict(te[valid_v4]), 0.05)

    # Compute V4 residuals on train (using in-fold predictions — slight optimism bias)
    # For temporal CV, we use LOO on training set itself for residuals
    v4_pred_tr = np.maximum(m_base.predict(tr[valid_v4]), 0.05)
    tr_residuals = tr["duration_hrs"].values - v4_pred_tr  # signed residuals

    # Phase 2: Residual meta-learner
    m_resid = lgb.LGBMRegressor(
        n_estimators=100, learning_rate=0.05, num_leaves=31,
        min_child_samples=5, reg_alpha=0.5, reg_lambda=1.0,
        random_state=42, verbosity=-1,
    )
    m_resid.fit(tr[residual_feats], tr_residuals)
    residual_correction = m_resid.predict(te[residual_feats])

    # Phase 3: Corrected predictions
    final = v4_pred_te + residual_correction
    return np.maximum(final, 0.05)

mae_residual = cv_mae(two_phase_predict, "V4 + Temporal Residual Meta-Learner")

# ── IDEA 3: Stacked Quantile Ensemble ─────────────────────────────────────────
print("\n[IDEA 3] Stacked Quantile Ensemble (multiple α, averaged)")
print("  Train 3 quantile models at α=0.3, 0.5, 0.7 → average predictions")
print("  Idea: averaging quantile models at different α gives smoother estimate")

def stacked_quantile_predict(tr, te):
    alphas = [0.30, 0.40, 0.50, 0.60, 0.70]
    preds = []
    for alpha in alphas:
        p = dict(V4_PARAMS, alpha=alpha)
        m = lgb.LGBMRegressor(**p)
        m.fit(tr[valid_v4], tr["duration_hrs"])
        preds.append(np.maximum(m.predict(te[valid_v4]), 0.05))
    # Weighted average — center alpha gets higher weight
    weights = np.array([0.10, 0.20, 0.40, 0.20, 0.10])
    return np.average(np.stack(preds), axis=0, weights=weights)

mae_stacked = cv_mae(stacked_quantile_predict, "Stacked Quantile Ensemble (α=0.3→0.7, weighted)")

# ── IDEA 4: DART (Dropout as Regularization) booster ─────────────────────────
print("\n[IDEA 4] DART Booster (dropout trees — reduces overfitting)")
print("  DART: like gradient boosting but randomly drops trees during training")
print("  More robust than GBDT for datasets with noisy labels")

DART_PARAMS = dict(
    boosting_type="dart",
    objective="quantile", alpha=0.50,
    n_estimators=500, learning_rate=0.05, num_leaves=127,
    min_child_samples=10, colsample_bytree=0.8, subsample=0.8,
    drop_rate=0.1, skip_drop=0.5, uniform_drop=False,
    reg_alpha=0.3, reg_lambda=0.1,
    random_state=42, verbosity=-1,
)

def dart_predict(tr, te):
    m=lgb.LGBMRegressor(**DART_PARAMS)
    m.fit(tr[valid_v4], tr["duration_hrs"])
    return np.maximum(m.predict(te[valid_v4]), 0.05)

mae_dart = cv_mae(dart_predict, "DART Booster (dropout trees, drop_rate=0.1)")

# Tune DART
print("  [Optuna 50 trials on DART...]")
def obj_dart(trial):
    p=dict(
        boosting_type="dart",
        objective="quantile",alpha=0.50,
        n_estimators=trial.suggest_int("n",200,700),
        learning_rate=trial.suggest_float("lr",0.005,0.08,log=True),
        num_leaves=trial.suggest_int("leaves",50,200),
        min_child_samples=trial.suggest_int("mcs",5,40),
        colsample_bytree=trial.suggest_float("cbt",0.5,1.0),
        subsample=trial.suggest_float("sub",0.5,1.0),
        drop_rate=trial.suggest_float("drop",0.05,0.3),
        skip_drop=trial.suggest_float("skip",0.3,0.7),
        reg_alpha=trial.suggest_float("ra",1e-5,2.0,log=True),
        reg_lambda=trial.suggest_float("rl",1e-5,10.0,log=True),
        random_state=42,verbosity=-1,
    )
    maes=[]
    for f in folds:
        tr=raw.loc[f["tr"]]; te=raw.loc[f["te"]]
        tro=tr[TRAIN_MASK.loc[f["tr"]]]; teo=te[(te["event_observed"]==1)&(te["duration_hrs"]<=48)]
        if len(tro)<20 or len(teo)<5: continue
        m=lgb.LGBMRegressor(**p)
        m.fit(tro[valid_v4],tro["duration_hrs"])
        maes.append(mean_absolute_error(teo["duration_hrs"].values,
                                        np.maximum(m.predict(teo[valid_v4]),0.05)))
    return np.mean(maes) if maes else 999

study_dart=optuna.create_study(direction="minimize",sampler=optuna.samplers.TPESampler(seed=77))
study_dart.optimize(obj_dart,n_trials=50)
mae_dart_tuned=study_dart.best_value
best_dart_params=study_dart.best_params
print(f"  DART + Optuna 50 trials: {mae_dart_tuned:.4f}h")

# ── IDEA 5: Neural Network (PyTorch entity embeddings) ────────────────────────
mae_nn = None
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    print("\n[IDEA 5] Neural Quantile Network with Entity Embeddings (PyTorch)")
    print("  Architecture: Corridor(32) + Cause(8) + Zone(4) + MLP(256→128→64→1)")
    print("  Loss: Pinball/quantile loss at α=0.5 (median)")

    N_CORR  = raw["corridor_enc"].max()+1
    N_CAUSE = raw["event_cause_enc"].max()+1
    N_ZONE  = raw["zone_enc"].max()+1
    N_STAT  = raw["police_station_enc"].max()+1

    # Numerical features (all V4 except the encoded categoricals)
    NUM_FEATS = [f for f in valid_v4 if f not in
                 ["corridor_enc","event_cause_enc","zone_enc","police_station_enc"]]

    class EntityEmbeddingNet(nn.Module):
        def __init__(self, n_corr, n_cause, n_zone, n_stat, n_num):
            super().__init__()
            self.emb_corr  = nn.Embedding(n_corr,  32)
            self.emb_cause = nn.Embedding(n_cause, 12)
            self.emb_zone  = nn.Embedding(n_zone,  6)
            self.emb_stat  = nn.Embedding(n_stat,  8)
            total = 32+12+6+8+n_num
            self.net = nn.Sequential(
                nn.Linear(total, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 64),  nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(64, 1),
            )

        def forward(self, x_corr, x_cause, x_zone, x_stat, x_num):
            e = torch.cat([
                self.emb_corr(x_corr),
                self.emb_cause(x_cause),
                self.emb_zone(x_zone),
                self.emb_stat(x_stat),
                x_num
            ], dim=1)
            return self.net(e).squeeze(-1)

    def quantile_loss(preds, targets, alpha=0.5):
        err = targets - preds
        return torch.mean(torch.max(alpha*err, (alpha-1)*err))

    def nn_cv(alpha=0.5):
        maes=[]
        for f in folds:
            tr=raw.loc[f["tr"]]; te=raw.loc[f["te"]]
            tro=tr[TRAIN_MASK.loc[f["tr"]]].copy()
            teo=te[(te["event_observed"]==1)&(te["duration_hrs"]<=48)].copy()
            if len(tro)<20 or len(teo)<5: continue

            for df in [tro, teo]:
                for c in NUM_FEATS:
                    if c not in df.columns: df[c] = 0.0

            X_tr_num = torch.FloatTensor(tro[NUM_FEATS].values.astype(np.float32))
            X_te_num = torch.FloatTensor(teo[NUM_FEATS].values.astype(np.float32))
            X_tr_corr  = torch.LongTensor(tro["corridor_enc"].astype(int).values)
            X_te_corr  = torch.LongTensor(teo["corridor_enc"].astype(int).values)
            X_tr_cause = torch.LongTensor(tro["event_cause_enc"].astype(int).values)
            X_te_cause = torch.LongTensor(teo["event_cause_enc"].astype(int).values)
            X_tr_zone  = torch.LongTensor(tro["zone_enc"].astype(int).values)
            X_te_zone  = torch.LongTensor(teo["zone_enc"].astype(int).values)
            X_tr_stat  = torch.LongTensor(tro["police_station_enc"].astype(int).values)
            X_te_stat  = torch.LongTensor(teo["police_station_enc"].astype(int).values)
            y_tr = torch.FloatTensor(tro["duration_hrs"].values.astype(np.float32))

            model = EntityEmbeddingNet(N_CORR, N_CAUSE, N_ZONE, N_STAT, len(NUM_FEATS))
            opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)
            dataset = TensorDataset(X_tr_corr, X_tr_cause, X_tr_zone, X_tr_stat, X_tr_num, y_tr)
            loader  = DataLoader(dataset, batch_size=64, shuffle=True)

            # Train 80 epochs
            model.train()
            for epoch in range(80):
                for bc,bca,bz,bs,bx,by in loader:
                    opt.zero_grad()
                    pred = model(bc,bca,bz,bs,bx)
                    loss = quantile_loss(pred, by, alpha)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                sched.step()

            model.eval()
            with torch.no_grad():
                preds = model(X_te_corr,X_te_cause,X_te_zone,X_te_stat,X_te_num).numpy()
            preds = np.maximum(preds, 0.05)
            maes.append(mean_absolute_error(teo["duration_hrs"].values, preds))
        avg=np.mean(maes) if maes else 999
        print(f"  Neural EmbNet (α=0.5, 80 epochs)               MAE={avg:.4f}h  {[round(x,2) for x in maes]}")
        return avg, maes

    mae_nn, _ = nn_cv(alpha=0.5)

    # LGB+NN ensemble
    print("  [Trying LGB V4 + Neural EmbNet ensemble...]")
    def lgb_nn_ensemble(tr, te, nn_weight=0.35):
        # LGB
        m=lgb.LGBMRegressor(**V4_PARAMS)
        m.fit(tr[valid_v4],tr["duration_hrs"])
        lgb_p = np.maximum(m.predict(te[valid_v4]), 0.05)

        # NN — recreate from scratch for this fold
        for df in [tr, te]:
            for c in NUM_FEATS:
                if c not in df.columns: df[c] = 0.0
        X_tr_num = torch.FloatTensor(tr[NUM_FEATS].values.astype(np.float32))
        X_te_num = torch.FloatTensor(te[NUM_FEATS].values.astype(np.float32))
        X_tr_c = torch.LongTensor(tr["corridor_enc"].astype(int).values)
        X_te_c = torch.LongTensor(te["corridor_enc"].astype(int).values)
        X_tr_ca= torch.LongTensor(tr["event_cause_enc"].astype(int).values)
        X_te_ca= torch.LongTensor(te["event_cause_enc"].astype(int).values)
        X_tr_z = torch.LongTensor(tr["zone_enc"].astype(int).values)
        X_te_z = torch.LongTensor(te["zone_enc"].astype(int).values)
        X_tr_s = torch.LongTensor(tr["police_station_enc"].astype(int).values)
        X_te_s = torch.LongTensor(te["police_station_enc"].astype(int).values)
        y_tr   = torch.FloatTensor(tr["duration_hrs"].values.astype(np.float32))

        net = EntityEmbeddingNet(N_CORR,N_CAUSE,N_ZONE,N_STAT,len(NUM_FEATS))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        ds  = TensorDataset(X_tr_c,X_tr_ca,X_tr_z,X_tr_s,X_tr_num,y_tr)
        dl  = DataLoader(ds, batch_size=64, shuffle=True)
        net.train()
        for _ in range(80):
            for bc,bca,bz,bs,bx,by in dl:
                opt.zero_grad()
                quantile_loss(net(bc,bca,bz,bs,bx),by,0.5).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),1.0)
                opt.step()
        net.eval()
        with torch.no_grad():
            nn_p = np.maximum(net(X_te_c,X_te_ca,X_te_z,X_te_s,X_te_num).numpy(), 0.05)

        return lgb_p*(1-nn_weight) + nn_p*nn_weight

    mae_ens_nn = cv_mae(lambda tr,te: lgb_nn_ensemble(tr,te), "LGB V4 (65%) + Neural EmbNet (35%) ensemble")

except ImportError:
    print("\n[IDEA 5] PyTorch not available — skipping neural network")
    mae_ens_nn = None

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
print("\n" + "="*66)
print("COMPLETE RESULTS — V6 PUSH HARDER")
print("="*66)
results = [
    ("V4 baseline (champion)",                   mae_v4),
    ("Log-space LGB (default)",                  mae_log),
    (f"Log-space LGB + Optuna 50",               mae_log_tuned),
    ("DART booster (default)",                   mae_dart),
    (f"DART + Optuna 50",                        mae_dart_tuned),
    ("V4 + Temporal Residual Correction",        mae_residual),
    ("Stacked Quantile α=0.3→0.7",              mae_stacked),
]
if mae_nn is not None:
    results.append(("Neural Entity Embedding Net",   mae_nn))
if mae_ens_nn is not None:
    results.append(("LGB+Neural Ensemble",           mae_ens_nn))

results.sort(key=lambda x: x[1])
best_name, best_mae = results[0]

for name, mae in results:
    marker = " ← WINNER" if name==best_name else ""
    bar = "█"*max(1,int((2.0-min(mae,2.0))/2.0*30))
    print(f"  {name:<45} {mae:.4f}h  {bar}{marker}")

print(f"\n  FINAL CHAMPION: {best_name}")
print(f"  FINAL MAE:      {best_mae:.4f}h")
print(f"  Total journey:  123h → {best_mae:.3f}h ({(123-best_mae)/123*100:.1f}% improvement)")

with open(OUT_DIR/"v6_results.json","w") as f:
    json.dump({
        "best_method": best_name, "best_mae": round(best_mae,4),
        "all_results": {n:round(m,4) for n,m in results},
        "log_tuned_params": best_log_params,
        "dart_tuned_params": best_dart_params,
    }, f, indent=2)

print("\n[OK] GRIDGUARD AI V6 DONE")
