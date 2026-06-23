"""
Kaggle Grandmaster-level Deep EDA for Event-Driven Congestion Impact
Focus: Planned & Unplanned events, impact proxies, feature discovery.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

PYTHON_PATH_NOTE = "Use full python path with DS stack: C:\\Users\\Nisha kumari\\AppData\\Local\\Python\\bin\\python.exe"

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw" / "astram_events.csv"
PROCESSED = BASE / "data" / "processed" / "survival_ready.parquet"

print("=" * 80)
print("EVENTFLOW AI - GRANDMASTER EDA: EVENT-DRIVEN CONGESTION")
print("=" * 80)

df = pd.read_csv(RAW)
print(f"\nLoaded raw: {df.shape[0]} rows, {df.shape[1]} cols")

# Parse dates early
for col in ['start_datetime', 'closed_datetime', 'end_datetime', 'created_date']:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

df['hour'] = df['start_datetime'].dt.hour
df['dow'] = df['start_datetime'].dt.dayofweek
df['month'] = df['start_datetime'].dt.month
df['is_weekend'] = df['dow'].isin([5,6]).astype(int)
df['is_rush'] = ((df['hour'] >= 8) & (df['hour'] <= 11) | (df['hour'] >= 17) & (df['hour'] <= 20)).astype(int)

df['observed'] = df['closed_datetime'].notna().astype(int)
df['dur_hrs'] = np.where(
    df['observed'] == 1,
    (df['closed_datetime'] - df['start_datetime']).dt.total_seconds() / 3600,
    np.nan
)
df['dur_hrs'] = df['dur_hrs'].clip(lower=0.05)

# === 1. Planned vs Unplanned core stats ===
print("\n" + "="*60)
print("1. PLANNED vs UNPLANNED CORE BREAKDOWN")
print("="*60)
print(df['event_type'].value_counts(dropna=False))

print("\n--- Requires road closure rate ---")
print(pd.crosstab(df['event_type'], df['requires_road_closure'], normalize='index').round(3))

print("\n--- High priority rate ---")
print(pd.crosstab(df['event_type'], df['priority'] == 'High', normalize='index').round(3))

obs = df[df['observed']==1]
print("\n--- Observed duration quantiles (hrs) ---")
for et in ['planned', 'unplanned']:
    sub = obs[obs['event_type']==et]['dur_hrs']
    q = {q: round(sub.quantile(q), 2) for q in [0.25, 0.5, 0.75, 0.9, 0.95]}
    print(f"{et:12s} n={len(sub):5d}  Q25={q[0.25]:.1f}  med={q[0.5]:.1f}  Q75={q[0.75]:.1f}  P90={q[0.9]:.1f}")

# === 2. Event causes - focus on event-driven ===
print("\n" + "="*60)
print("2. EVENT-CAUSE ANALYSIS (CONSTRUCTION, PUBLIC, VIP etc.)")
print("="*60)

cause_counts = df['event_cause'].value_counts()
print("Top causes overall:")
print(cause_counts.head(12))

planned = df[df['event_type'] == 'planned']
print("\nPlanned events causes:")
print(planned['event_cause'].value_counts())

# Proxy "event_driven" flag
event_keywords = ['construction', 'public_event', 'procession', 'vip', 'rally', 'festival', 'match', 'cricket', 'event', 'meeting', 'function', 'protest']
df['desc_lower'] = df['description'].fillna('').astype(str).str.lower()
df['is_event_driven_text'] = df['desc_lower'].apply(lambda x: any(kw in x for kw in event_keywords)).astype(int)

print("\n--- 'is_event_driven_text' hits in description ---")
print(df.groupby('event_type')['is_event_driven_text'].sum())

# Impact by cause for "planned-like"
print("\n--- Closure rate + median dur for key 'planned-like' causes ---")
key_causes = ['construction', 'public_event', 'water_logging', 'pot_holes', 'accident', 'tree_fall', 'vehicle_breakdown']
for cause in key_causes:
    sub = df[df['event_cause'] == cause]
    if len(sub) < 20: continue
    closure_rt = sub['requires_road_closure'].mean()
    med_dur = sub[sub['observed']==1]['dur_hrs'].median() if sub['observed'].sum() > 5 else np.nan
    high_p = (sub['priority'] == 'High').mean()
    print(f"{cause:20s} n={len(sub):5d}  closure={closure_rt:.2f}  med_dur={med_dur:.2f}h  high_pri={high_p:.2f}")

# === 3. Corridor & Zone concentration (congestion hotspots) ===
print("\n" + "="*60)
print("3. SPATIAL CONCENTRATION - CORRIDOR / ZONE (PROXY FOR CENTRALITY)")
print("="*60)

corridor_stats = df.groupby('corridor').agg(
    total=('id', 'count'),
    planned=('event_type', lambda x: (x=='planned').sum()),
    closure_rate=('requires_road_closure', 'mean'),
    high_pri_rate=('priority', lambda x: (x=='High').mean()),
    med_dur=('dur_hrs', 'median')
).reset_index()
corridor_stats['planned_pct'] = (corridor_stats['planned'] / corridor_stats['total'] * 100).round(1)
corridor_stats = corridor_stats[corridor_stats['total'] >= 30].sort_values('planned', ascending=False)

print("Top corridors by # planned events (min 30 total events):")
print(corridor_stats[['corridor', 'total', 'planned', 'planned_pct', 'closure_rate', 'med_dur']].head(12).to_string(index=False))

print("\n--- Zone stats ---")
zone_stats = df.groupby('zone').agg(
    n=('id','count'),
    planned_n=('event_type', lambda x:(x=='planned').sum()),
    avg_closure=('requires_road_closure','mean')
).sort_values('planned_n', ascending=False)
print(zone_stats.head(8))

# === 4. Temporal patterns for planned / event driven ===
print("\n" + "="*60)
print("4. TEMPORAL PATTERNS - WHEN DO PLANNED / EVENT-DRIVEN HAPPEN?")
print("="*60)

print("Planned by hour bucket:")
planned['hour_bucket'] = pd.cut(planned['hour'], bins=[-1,6,11,16,20,24], labels=['night','morning','midday','evening','late'])
print(planned['hour_bucket'].value_counts(normalize=True).round(3))

print("\nPlanned vs Unplanned rush hour rate:")
print(df.groupby('event_type')['is_rush'].mean().round(3))

print("\nWeekday vs weekend planned rate:")
print(df.groupby('is_weekend')['event_type'].value_counts(normalize=True).unstack().round(3))

# === 5. Text & description signals for event type ===
print("\n" + "="*60)
print("5. TEXT MINING IN DESCRIPTIONS (EVENT SIGNALS)")
print("="*60)
sample_planned_desc = planned['description'].dropna().astype(str).str.lower().head(30).tolist()
print("Sample planned descriptions (lower):")
for d in sample_planned_desc[:8]:
    if len(d) > 10:
        print("  -", d[:120])

# Keyword frequency
all_desc = df['desc_lower']
kw_counter = Counter()
for kw in ['rally','procession','vip','festival','cricket','match','public','construction work','bwssb','kride','metro','road work','meeting']:
    cnt = all_desc.str.contains(kw).sum()
    if cnt > 10:
        kw_counter[kw] = cnt
print("\nStrong description keywords (global count >10):", dict(kw_counter.most_common(10)))

# === 6. Impact proxy creation (for target engineering) ===
print("\n" + "="*60)
print("6. POTENTIAL IMPACT PROXY TARGETS (for modeling)")
print("="*60)

# Rough congestion impact score = dur * closure_factor * rush_factor * (high frequency corridor bonus)
corr_freq = df['corridor'].value_counts().to_dict()
df['corridor_freq'] = df['corridor'].map(lambda x: corr_freq.get(x, 10))

impact_df = obs.copy()
impact_df['impact_proxy'] = (
    impact_df['dur_hrs'].clip(0, 12) * 
    (1 + impact_df['requires_road_closure'] * 0.8) *
    (1 + impact_df['is_rush'] * 0.4) *
    np.log1p(impact_df['corridor_freq']) / 4.0   # centrality proxy
)

print("Impact proxy (observed only) stats:")
print(impact_df['impact_proxy'].describe().round(2))

print("\nTop 5 highest impact proxies (cause, corridor, dur, score):")
top_imp = impact_df.nlargest(5, 'impact_proxy')[['event_cause','corridor','dur_hrs','requires_road_closure','impact_proxy']]
print(top_imp.to_string(index=False))

# === 7. Concurrent load opportunity (already partially used) ===
print("\n" + "="*60)
print("7. CONCURRENT / CASCADE OPPORTUNITY BY EVENT TYPE")
print("="*60)

# Simple proxy for same-corridor same-hour density
df_sorted = df.sort_values('start_datetime')
df_sorted['corridor_hour_count'] = df_sorted.groupby([df_sorted['corridor'], df_sorted['start_datetime'].dt.floor('H')]).cumcount()
print("Max same-corridor-hour events (proxy for load):", df_sorted['corridor_hour_count'].max())
print("High load examples (count >=3):")
high_load = df_sorted[df_sorted['corridor_hour_count'] >= 2].groupby('corridor').size().sort_values(ascending=False).head(5)
print(high_load)

# === 8. Data quality for planned specifically ===
print("\n" + "="*60)
print("8. DATA QUALITY & CENSORING FOR PLANNED")
print("="*60)
print("Planned observed rate:", planned['observed'].mean().round(3))
print("Planned has end_datetime filled:", planned['end_datetime'].notna().mean().round(3))
print("Planned missing description rate:", planned['description'].isna().mean().round(3))
print("Planned missing zone:", planned['zone'].isna().mean().round(3))

print("\n" + "="*80)
print("KEY INSIGHTS FOR FEATURE ENGINEERING (GRANDMASTER DIRECTION)")
print("="*80)
print("""
1. Planned events are rare (~5.7%). Must treat separately or with strong 'is_planned' flag + interactions.
2. Construction + public_event are the main 'planned-like' levers for proactive planning.
3. Corridor frequency is a very strong proxy for network centrality / exposure.
4. Rush hour + requires_closure dramatically increase effective impact.
5. Descriptions contain weak but usable signals for event type (construction work, BWSSB, VIP).
6. Duration distributions differ: planned may have better end times or be longer running infrastructure.
7. High value: engineer 'advance knowledge' features + expected scale (even if proxy).
8. Opportunity: predict not only duration but a composite 'Congestion Impact Units' + secondary risk.
""")

# Save some key aggregates for later use in FE
out_dir = BASE / "data" / "precomputed"
out_dir.mkdir(exist_ok=True, parents=True)

corridor_stats.to_parquet(out_dir / "corridor_impact_stats.parquet", index=False)
print(f"\nSaved corridor_impact_stats.parquet for use in advanced FE.")
print("EDA complete.")