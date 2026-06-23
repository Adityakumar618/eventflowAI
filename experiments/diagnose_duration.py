import pandas as pd
import numpy as np

df = pd.read_parquet("data/processed/survival_ready.parquet")

print("=== DURATION DISTRIBUTION PER CAUSE ===")
closed = df[df["event_observed"]==1]
stats = closed.groupby("event_cause")["duration_hrs"].describe(percentiles=[.1,.25,.5,.75,.9,.95])
print(stats[["count","10%","25%","50%","75%","90%","95%","max"]].round(1).to_string())

print()
print("=== EVENTS BY DURATION BUCKET ===")
buckets = [0, 2, 6, 12, 24, 72, 720]
labels  = ["0-2h","2-6h","6-12h","12-24h","24-72h","72-720h"]
df["dur_bucket"] = pd.cut(df["duration_hrs"], bins=buckets, labels=labels)
print(df["dur_bucket"].value_counts().sort_index().to_string())

print()
print("=== VEHICLE BREAKDOWN SANITY CHECK ===")
vb = df[df["event_cause"]=="vehicle_breakdown"]
print("Total:", len(vb), "| Observed:", int(vb["event_observed"].sum()))
print("Median duration:", round(vb["duration_hrs"].median(),1), "h")
print("<=  2h:", int((vb["duration_hrs"]<=2).sum()), f"({(vb['duration_hrs']<=2).mean()*100:.0f}%)")
print("<=  6h:", int((vb["duration_hrs"]<=6).sum()), f"({(vb['duration_hrs']<=6).mean()*100:.0f}%)")
print("<= 24h:", int((vb["duration_hrs"]<=24).sum()), f"({(vb['duration_hrs']<=24).mean()*100:.0f}%)")
print("= 720h (cap):", int((vb["duration_hrs"]>=719).sum()), "events")

print()
print("=== ACCIDENT SANITY CHECK ===")
ac = df[df["event_cause"]=="accident"]
print("Total:", len(ac), "| Median:", round(ac["duration_hrs"].median(),1), "h")
print("= 720h cap:", int((ac["duration_hrs"]>=719).sum()))

print()
print("KEY DIAGNOSIS:")
print("If median VB = 600h, the closed_datetime is NOT operational resolution time.")
print("It is administrative ticket closure. Officers cleared the road hours earlier.")
print()
print("resolved_datetime null rate:")
raw = pd.read_csv("data/raw/astram_events.csv")
print("resolved_datetime:", raw["resolved_datetime"].isna().mean()*100, "% null")
print("closed_datetime:  ", raw["closed_datetime"].isna().mean()*100, "% null")
print()
print("=== Sample VB durations (closed_datetime - start_datetime) ===")
raw_dt = raw.copy()
raw_dt["start_datetime"]  = pd.to_datetime(raw_dt["start_datetime"],  errors="coerce")
raw_dt["closed_datetime"] = pd.to_datetime(raw_dt["closed_datetime"], errors="coerce")
raw_dt["resolved_datetime"] = pd.to_datetime(raw_dt["resolved_datetime"], errors="coerce")
raw_dt["dur_closed"]   = (raw_dt["closed_datetime"]   - raw_dt["start_datetime"]).dt.total_seconds()/3600
raw_dt["dur_resolved"] = (raw_dt["resolved_datetime"] - raw_dt["start_datetime"]).dt.total_seconds()/3600

vb_raw = raw_dt[raw_dt["event_cause"]=="vehicle_breakdown"]
print("Median dur from closed_datetime:  ", round(vb_raw["dur_closed"].median(),1),"h")
print("Median dur from resolved_datetime:", round(vb_raw["dur_resolved"].dropna().median(),1),"h")
print("N with resolved_datetime:", vb_raw["dur_resolved"].notna().sum())
