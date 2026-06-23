"""
Final targeted experiment: V4 + lat/lon direct spatial coordinates
===================================================================
Hypothesis: Direct lat/lon as features solves the Non-corridor black hole
WITHOUT destroying corridor_loo. The model can learn spatial gradients
(e.g., south Bengaluru faster than north) without fragmenting LOO history.

Also test: ghost closures with stricter filter (only short-duration ghosts)
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

OUT_DIR = Path("experiments/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 62)
print("FINAL: V4 + Lat/Lon spatial features (no corridor rename)")
print("=" * 62)

raw = pd.read_csv("data/raw/astram_events.csv")
for col in ["start_datetime","closed_datetime","resolved_datetime","modified_datetime"]:
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
raw["is_weekend"] = raw["dow"].isin([5,6]).astype(int)
raw["is_rush"]    = raw["hour"].apply(lambda h: 1 if (8<=h<=11) or (17<=h<=20) else 0)
raw["is_night"]   = ((raw["hour"]>=22)|(raw["hour"]<=5)).astype(int)
raw["hour_sin"]   = np.sin(raw["hour"]*2*np.pi/24)
raw["hour_cos"]   = np.cos(raw["hour"]*2*np.pi/24)
raw["is_weather"] = raw["event_cause"].isin(
    ["water_logging","tree_fall","fog/low_visibility","debris"]).astype(int)
raw["is_high_priority"] = (raw["priority"].fillna("Low")=="High").astype(int)
raw["is_non_corridor"]  = (raw["corridor"]=="Non-corridor").astype(int)  # flag

raw = raw.sort_values("start_datetime").reset_index(drop=True)
raw["start_dt_naive"] = raw["start_datetime"].dt.tz_localize(None)
raw["start_ts"] = raw["start_dt_naive"].astype(np.int64)//10**9

obs = raw[raw["event_observed"]==1]
gm  = obs["duration_hrs"].mean()

# Normalize spatial coordinates
raw["lat_norm"] = (raw["latitude"]  - raw["latitude"].mean())  / raw["latitude"].std()
raw["lon_norm"] = (raw["longitude"] - raw["longitude"].mean()) / raw["longitude"].std()
# Radial distance from city center (MG Road ~13.0, 77.6)
raw["dist_center"] = np.sqrt((raw["latitude"]-13.0)**2 + (raw["longitude"]-77.6)**2)

print(f"Spatial features added: lat_norm, lon_norm, dist_center")
print(f"Non-corridor events: {raw['is_non_corridor'].sum()} ({raw['is_non_corridor'].mean()*100:.1f}%)")

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

# V4 features (confirmed champion)
V4_BASE = [
    "hour","dow","month","is_weekend","is_rush","is_night","hour_sin","hour_cos",
    "is_weather","requires_road_closure","is_high_priority",
    "event_cause_enc","zone_enc","corridor_enc","police_station_enc",
    "cause_mean","cause_median","cause_p90","cause_p10","station_mean",
    "corridor_mean","corridor_cnt",
    "cause_x_rush","cause_x_night","cause_x_zone","cause_x_closure","station_x_cause",
    "cause_rolling_30d","concurrent_zone_events","concurrent_corridor_events",
    "station_loo","corridor_loo","hour_cause_mean","zone_cause_loo","veh_cause_loo",
]+text_cols+["has_description","log_hawkes_zone","officer_active_load"]

# V4 + Spatial
V4_SPATIAL = V4_BASE + ["lat_norm","lon_norm","dist_center","is_non_corridor"]

for col in V4_SPATIAL:
    raw[col]=pd.to_numeric(raw[col],errors="coerce").fillna(0)

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

LGB_V4_PARAMS=dict(
    objective="quantile",alpha=0.50,
    n_estimators=661,learning_rate=0.013164,num_leaves=129,
    min_child_samples=9,colsample_bytree=0.9989,subsample=0.5783,
    reg_alpha=0.2257,reg_lambda=0.0107,min_split_gain=0.1991,
    random_state=42,verbosity=-1,
)

def cv_mae(features, label):
    maes=[]
    for f in folds:
        tr=raw.loc[f["tr"]]; te=raw.loc[f["te"]]
        tro=tr[TRAIN_MASK.loc[f["tr"]]]; teo=te[(te["event_observed"]==1)&(te["duration_hrs"]<=48)]
        if len(tro)<20 or len(teo)<5: continue
        vf=[ft for ft in features if ft in tro.columns]
        m=lgb.LGBMRegressor(**LGB_V4_PARAMS)
        m.fit(tro[vf],tro["duration_hrs"])
        maes.append(mean_absolute_error(teo["duration_hrs"].values,np.maximum(m.predict(teo[vf]),0.05)))
    avg=np.mean(maes) if maes else 999
    print(f"  {label:<55} MAE={avg:.4f}h  {[round(x,2) for x in maes]}")
    return avg

print("\n" + "="*62)
print("DIRECT SPATIAL vs MICRO-GRID LOO COMPARISON")
print("="*62)
mae_v4_base    = cv_mae(V4_BASE,    "V4 base (confirmed champion)")
mae_v4_spatial = cv_mae(V4_SPATIAL, "V4 + lat/lon/dist/non_corridor_flag")

# Non-corridor specific MAE analysis
print("\n[NON-CORRIDOR SPECIFIC ANALYSIS]")
for f in folds:
    tr=raw.loc[f["tr"]]; te=raw.loc[f["te"]]
    tro=tr[TRAIN_MASK.loc[f["tr"]]]
    teo_nc=te[(te["event_observed"]==1)&(te["duration_hrs"]<=48)&(te["corridor"]=="Non-corridor")]
    teo_co=te[(te["event_observed"]==1)&(te["duration_hrs"]<=48)&(te["corridor"]!="Non-corridor")]
    if len(tro)<20: continue
    vf=[ft for ft in V4_SPATIAL if ft in tro.columns]
    m=lgb.LGBMRegressor(**LGB_V4_PARAMS)
    m.fit(tro[vf],tro["duration_hrs"])
    mae_nc=mean_absolute_error(teo_nc["duration_hrs"].values,np.maximum(m.predict(teo_nc[vf]),0.05)) if len(teo_nc)>0 else None
    mae_co=mean_absolute_error(teo_co["duration_hrs"].values,np.maximum(m.predict(teo_co[vf]),0.05)) if len(teo_co)>0 else None
    print(f"  {f['name']}: Non-corridor MAE={mae_nc:.3f}h | Named-corridor MAE={mae_co:.3f}h" if mae_nc and mae_co else "")

print("\n" + "="*62)
print("DEFINITIVE RESULT")
print("="*62)
if mae_v4_spatial < mae_v4_base:
    improvement = (mae_v4_base - mae_v4_spatial)/mae_v4_base*100
    print(f"  V4 + Spatial WINS: {mae_v4_spatial:.4f}h ({improvement:.1f}% better than V4)")
    print(f"  -> Integrate lat/lon into production model")
    best_final = mae_v4_spatial
    best_label = "V4 + Spatial direct coordinates"
else:
    print(f"  V4 BASE remains CHAMPION: {mae_v4_base:.4f}h")
    print(f"  -> V4 is our final production model. No further ML improvements possible.")
    best_final = mae_v4_base
    best_label = "V4 (final, production)"

print(f"\n  CONFIRMED FINAL MODEL: {best_label}")
print(f"  CONFIRMED FINAL MAE:   {best_final:.4f}h")
print(f"  Total improvement:     123.0h → {best_final:.3f}h ({(123-best_final)/123*100:.1f}%)")
print("\n[OK] FINAL SPATIAL EXPERIMENT DONE")
