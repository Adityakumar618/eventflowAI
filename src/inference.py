"""
GridGuard AI - Production Inference Engine (V9)
================================================
Ensemble of 3 models: LGB L1 + LGB DART + XGBoost L1
All trained on log1p(duration_hrs) target, expm1 at output.

New V9 features vs V7:
  - cluster_dur_loo  : DBSCAN spatial cluster history (top feature 15.4% SHAP)
  - officer_dur_loo  : per-officer resolution speed history
  - pin_code_loo     : area-level (PIN code) duration signal
  - zone_cause_loo   : fixed composite LOO key (was broken in V4/V7)
  - veh_cause_loo    : fixed composite LOO key (was broken in V4/V7)
  - is_junction      : address junction flag
  - addr_road_class  : highway/main/other from address text
  - officer_event_cnt: officer workload count
  - description_len  : text length signal
"""
import re, os, json, time, logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1); dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def _extract_pin(address):
    m = re.search(r"\b(\d{6})\b", str(address))
    return m.group(1) if m else "000000"

def _road_class(address):
    a = str(address).lower()
    if any(k in a for k in ["nh-", "sh-", "highway", "expressway"]): return 3
    if any(k in a for k in ["main", "ring road", "outer ring"]): return 2
    return 1

def _is_junction(address):
    return 1 if any(k in str(address).lower() for k in ["junction", " jn", "circle", "signal"]) else 0

try:
    from src.mappls_feature_engine import BTP_STATIONS, ISEC_CONGESTION
except Exception:
    BTP_STATIONS = {"Yeshwanthpur":(13.0253,77.5397),"Marathahalli":(12.9592,77.6974),
                    "Hebbal":(13.0350,77.5970),"Electronic City":(12.8399,77.6770),"Whitefield":(12.9698,77.7499)}
    ISEC_CONGESTION = {h: 1.0+0.5*(1 if 8<=h<=10 or 17<=h<=20 else 0) for h in range(24)}

_BTP = np.array(list(BTP_STATIONS.values()))


class GridGuardV9Inference:
    """V9 ensemble inference: LGB L1 + LGB DART + XGBoost, 65 features."""

    MODEL_VERSION = "V9"

    def __init__(self, enable_mappls=False, mappls_client_id=None, mappls_client_secret=None):
        t0 = time.time()
        m = BASE_DIR / "models"
        self.lgb_model  = joblib.load(m / "lgb_v9_final.pkl")
        self.dart_model = joblib.load(m / "lgb_v9_dart.pkl")
        self.xgb_model  = joblib.load(m / "xgb_v9_final.pkl")
        self.tfidf      = joblib.load(m / "v9_tfidf.pkl")
        self.svd        = joblib.load(m / "v9_svd.pkl")
        self.features   = joblib.load(m / "v9_features.pkl")
        self.dbscan     = joblib.load(m / "dbscan_v9.pkl")
        # OOF inverse-MAE weights from Phase C (LGB=2.934h, DART=3.176h, XGB=3.212h)
        w = np.array([1/2.934, 1/3.176, 1/3.212]); self._w = w / w.sum()
        self._build_lookup_tables()
        self.mappls_engine = None
        if enable_mappls and mappls_client_id:
            try:
                from src.mappls_feature_engine import MapplsFeatureEngine
                self.mappls_engine = MapplsFeatureEngine(mappls_client_id, mappls_client_secret,
                    cache_dir=str(BASE_DIR/"data"/"mappls_cache"))
            except Exception as e:
                logger.warning("Mappls init: %s", e)
        logger.info("V9 engine ready in %.2fs | %d features | 3-model ensemble", time.time()-t0, len(self.features))

    def _build_lookup_tables(self):
        from sklearn.preprocessing import LabelEncoder
        raw = pd.read_csv(BASE_DIR/"data"/"raw"/"astram_events.csv")
        for c in ["start_datetime","closed_datetime","resolved_datetime"]:
            raw[c] = pd.to_datetime(raw[c], errors="coerce")
        md = raw["start_datetime"].max()
        raw["duration_hrs"] = np.where(raw["closed_datetime"].notna(),
            (raw["closed_datetime"]-raw["start_datetime"]).dt.total_seconds()/3600,
            np.where(raw["resolved_datetime"].notna(),
                (raw["resolved_datetime"]-raw["start_datetime"]).dt.total_seconds()/3600,
                (md-raw["start_datetime"]).dt.total_seconds()/3600)).clip(min=0.05)
        obs = raw[raw["closed_datetime"].notna() & (raw["duration_hrs"]<=48)].copy()
        self.gm = float(obs["duration_hrs"].mean())
        self.encoders = {}
        for col in ["event_cause","zone","corridor","police_station"]:
            le = LabelEncoder(); le.fit(raw[col].fillna("unknown").astype(str))
            self.encoders[col] = le
        self.cause_stats = obs.groupby("event_cause")["duration_hrs"].agg(
            cause_mean="mean", cause_median="median",
            cause_p90=lambda x: x.quantile(0.9), cause_p10=lambda x: x.quantile(0.1)
        ).to_dict("index")
        cr = obs.groupby("corridor")["duration_hrs"].agg(["mean","count"])
        self.corridor_stats = {k: {"corridor_mean": r["mean"], "corridor_cnt": r["count"]} for k,r in cr.iterrows()}
        self.station_stats = obs.groupby("police_station")["duration_hrs"].mean().to_dict()
        obs["hour"] = obs["start_datetime"].dt.hour
        self.hour_cause_mean = obs.groupby(["hour","event_cause"])["duration_hrs"].mean().to_dict()
        def _loo(df, col):
            return df.groupby(col)["duration_hrs"].mean().to_dict()
        self.corridor_loo = _loo(obs, "corridor")
        self.station_loo  = _loo(obs, "police_station")
        obs["_zc"] = obs["zone"].fillna("unk") + "||" + obs["event_cause"]
        self.zone_cause_loo = _loo(obs, "_zc")
        obs["_vc"] = obs["veh_type"].fillna("unknown") + "||" + obs["event_cause"]
        self.veh_cause_loo = _loo(obs, "_vc")
        raw["lat_r"] = raw["latitude"].round(3); raw["lon_r"] = raw["longitude"].round(3)
        raw["cluster_id"] = self.dbscan.labels_
        obs["cluster_id"] = raw.loc[obs.index, "cluster_id"].values
        self.cluster_loo = _loo(obs[obs["cluster_id"]>=0], "cluster_id")
        self.cluster_density = obs["cluster_id"].value_counts().to_dict()
        self._core_lats    = raw.loc[raw["cluster_id"]>=0, "lat_r"].values
        self._core_lons    = raw.loc[raw["cluster_id"]>=0, "lon_r"].values
        self._core_labels  = raw.loc[raw["cluster_id"]>=0, "cluster_id"].values
        if "created_by_id" in obs.columns:
            self.officer_loo = _loo(obs.dropna(subset=["created_by_id"]), "created_by_id")
            self.officer_cnt = obs["created_by_id"].value_counts().to_dict()
        else:
            self.officer_loo = {}; self.officer_cnt = {}
        if "address" in raw.columns:
            obs["_pin"] = raw.loc[obs.index, "address"].apply(_extract_pin)
            self.pin_loo = _loo(obs, "_pin")
        else:
            self.pin_loo = {}
        self.corridor_stress = self.corridor_loo.copy()

    def _predict_cluster(self, lat, lon):
        if len(self._core_lats) == 0: return -1
        d = _haversine_km(round(lat,3), round(lon,3), self._core_lats, self._core_lons)
        i = int(np.argmin(d))
        return int(self._core_labels[i]) if d[i] < 1.0 else -1

    def _text_features(self, desc, reason):
        text = (str(desc or "").lower() + " " + str(reason or "").lower()).strip()
        try:
            v = self.tfidf.transform([text]); s = self.svd.transform(v)[0]
            return {f"text_svd_{i}": float(s[i]) for i in range(len(s))}
        except Exception:
            return {f"text_svd_{i}": 0.0 for i in range(8)}

    def predict(self, event_dict: Dict[str, Any], return_interval: bool = False) -> Dict[str, Any]:
        t0 = time.time()
        cause   = str(event_dict.get("event_cause", "vehicle_breakdown"))
        corr    = str(event_dict.get("corridor", "Non-corridor"))
        station = str(event_dict.get("police_station", "unknown"))
        zone    = str(event_dict.get("zone", "unknown"))
        hour    = int(event_dict.get("hour", 12))
        lat     = float(event_dict.get("lat", 13.0))
        lon     = float(event_dict.get("lon", 77.6))
        dow     = int(event_dict.get("dow", 0))
        month   = int(event_dict.get("month", 1))
        address = str(event_dict.get("address", ""))
        veh     = str(event_dict.get("veh_type", "unknown"))
        oid     = event_dict.get("created_by_id", None)
        row = {}
        def enc(col, val):
            try: return int(self.encoders[col].transform([str(val)])[0])
            except: return 0
        row["event_cause_enc"]=enc("event_cause",cause); row["zone_enc"]=enc("zone",zone)
        row["corridor_enc"]=enc("corridor",corr); row["police_station_enc"]=enc("police_station",station)
        row["hour"]=hour; row["dow"]=dow; row["month"]=month
        row["is_weekend"]=1 if dow in(5,6) else 0
        row["is_rush"]=1 if(8<=hour<=11)or(17<=hour<=20) else 0
        row["is_night"]=1 if(hour>=22 or hour<=5) else 0
        row["hour_sin"]=np.sin(hour*2*np.pi/24); row["hour_cos"]=np.cos(hour*2*np.pi/24)
        row["requires_road_closure"]=1 if event_dict.get("requires_road_closure") else 0
        row["is_high_priority"]=1 if str(event_dict.get("priority","Low"))=="High" else 0
        row["is_weather"]=1 if cause in("water_logging","tree_fall","fog/low_visibility","debris") else 0
        desc=str(event_dict.get("description",""))
        row["has_description"]=1 if len(desc)>5 else 0; row["description_len"]=min(len(desc),500)
        cs=self.cause_stats.get(cause,{})
        row["cause_mean"]=cs.get("cause_mean",self.gm); row["cause_median"]=cs.get("cause_median",self.gm)
        row["cause_p90"]=cs.get("cause_p90",self.gm*3); row["cause_p10"]=cs.get("cause_p10",self.gm*0.3)
        co=self.corridor_stats.get(corr,{})
        row["corridor_mean"]=co.get("corridor_mean",self.gm); row["corridor_cnt"]=co.get("corridor_cnt",1)
        row["station_mean"]=self.station_stats.get(station,self.gm)
        row["corridor_loo"]=self.corridor_loo.get(corr,self.gm)
        row["station_loo"]=self.station_loo.get(station,self.gm)
        row["zone_cause_loo"]=self.zone_cause_loo.get(f"{zone}||{cause}",self.gm)
        row["veh_cause_loo"]=self.veh_cause_loo.get(f"{veh}||{cause}",self.gm)
        row["hour_cause_mean"]=self.hour_cause_mean.get((hour,cause),row["cause_mean"])
        row["cause_x_rush"]=row["event_cause_enc"]*row["is_rush"]
        row["cause_x_night"]=row["event_cause_enc"]*row["is_night"]
        row["cause_x_zone"]=row["event_cause_enc"]*row["zone_enc"]
        row["cause_x_closure"]=row["event_cause_enc"]*row["requires_road_closure"]
        row["cause_rolling_30d"]=event_dict.get("cause_rolling_30d",row["cause_mean"])
        row["concurrent_zone_events"]=int(event_dict.get("concurrent_zone_events",0))
        row["concurrent_corridor_events"]=int(event_dict.get("concurrent_corridor_events",0))
        row["officer_active_load"]=int(event_dict.get("officer_active_load",0))
        row["log_hawkes_zone"]=float(event_dict.get("log_hawkes_zone",0.05))
        row["log_hawkes_cause"]=float(event_dict.get("log_hawkes_cause",0.05))
        row["corridor_stress_index"]=self.corridor_stress.get(corr,self.gm)
        dists=_haversine_km(lat,lon,_BTP[:,0],_BTP[:,1])
        nkm=float(dists.min())
        row["nearest_station_km"]=nkm
        row["police_response_eta_mins"]=max(2.0,(nkm/18)*60)
        row["tod_congestion_idx"]=float(ISEC_CONGESTION.get(hour,1.4))
        row["congestion_x_response"]=row["tod_congestion_idx"]*row["police_response_eta_mins"]
        row["is_junction"]=float(_is_junction(address)); row["addr_road_class"]=float(_road_class(address))
        row["officer_event_cnt"]=float(self.officer_cnt.get(oid,0)) if oid else 0.0
        row["officer_dur_loo"]=float(self.officer_loo.get(oid,self.gm)) if oid else self.gm
        cid=self._predict_cluster(lat,lon)
        row["cluster_id"]=float(cid)
        row["cluster_dur_loo"]=float(self.cluster_loo.get(cid,self.gm))
        row["cluster_density"]=float(self.cluster_density.get(cid,1))
        row["pin_code_loo"]=float(self.pin_loo.get(_extract_pin(address),self.gm))
        row.update(self._text_features(event_dict.get("description",""), event_dict.get("reason_breakdown","")))
        mappls_enriched = False
        if self.mappls_engine:
            try:
                row.update(self.mappls_engine.get_features(lat=lat,lon=lon,hour=hour,event_cause=cause,timeout=12.0))
                mappls_enriched = True
            except Exception: pass
        df = pd.DataFrame([row])
        for f in self.features:
            if f not in df.columns: df[f]=0.0
        df = df.reindex(columns=self.features, fill_value=0.0).astype(np.float32)
        log_lgb  = float(self.lgb_model.predict(df)[0])
        log_dart = float(self.dart_model.predict(df)[0])
        try:
            import xgboost as xgb
            log_xgb = float(self.xgb_model.predict(xgb.DMatrix(df, feature_names=list(df.columns)))[0])
        except Exception:
            log_xgb = (log_lgb+log_dart)/2
        lb = self._w[0]*log_lgb + self._w[1]*log_dart + self._w[2]*log_xgb
        pred = max(0.05, float(np.expm1(max(lb,0.0))))
        p10=pred*0.40; p90=pred*2.60
        if cs:
            r10=cs.get("cause_p10",p10)/max(cs.get("cause_mean",pred),0.01)
            r90=cs.get("cause_p90",p90)/max(cs.get("cause_mean",pred),0.01)
            p10=max(0.05,pred*r10); p90=max(pred*1.1,pred*r90)
        return {"predicted_hours":round(pred,2),"p10_hours":round(p10,2),"p90_hours":round(p90,2),
                "cluster_id":int(cid),"cluster_dur_loo":round(row["cluster_dur_loo"],3),
                "mappls_enriched":mappls_enriched,"latency_ms":round((time.time()-t0)*1000,1),"model_version":"V9"}


# Backwards-compatible alias
GridGuardInference = GridGuardV9Inference

_engine: Optional[GridGuardV9Inference] = None
def get_inference_engine(mappls_client_id=None, mappls_client_secret=None):
    global _engine
    if _engine is None:
        _engine = GridGuardV9Inference(mappls_client_id=mappls_client_id, mappls_client_secret=mappls_client_secret)
    return _engine


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    engine = GridGuardV9Inference(enable_mappls=False)
    tests = [
        {"event_cause":"vehicle_breakdown","zone":"North","corridor":"Bellary Road",
         "police_station":"Yelahanka","hour":8,"lat":13.035,"lon":77.597,
         "description":"truck broke down blocking 2 lanes","address":"Bellary Road near Yelahanka Junction"},
        {"event_cause":"water_logging","zone":"South","corridor":"Hosur Road",
         "police_station":"Electronic City","hour":17,"lat":12.845,"lon":77.660,
         "address":"Hosur Road Electronic City Main Road"},
    ]
    print("\n" + "="*62)
    print("GRIDGUARD AI V9 INFERENCE SMOKE TEST")
    print("="*62)
    for ev in tests:
        r = engine.predict(ev)
        print(f"\n  {ev['event_cause']:<24} zone={ev['zone']:<8} hour={ev['hour']:02d}h")
        print(f"  -> {r['predicted_hours']:.2f}h  [P10={r['p10_hours']:.2f} P90={r['p90_hours']:.2f}]  cluster={r['cluster_id']}  {r['latency_ms']:.0f}ms")
    print("\n[OK] V9 inference engine working.")
