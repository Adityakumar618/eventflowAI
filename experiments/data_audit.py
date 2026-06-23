import pandas as pd
import numpy as np

df = pd.read_csv('data/raw/astram_events.csv')
df['closed_datetime']   = pd.to_datetime(df['closed_datetime'],   errors='coerce')
df['modified_datetime'] = pd.to_datetime(df['modified_datetime'], errors='coerce')
df['start_datetime']    = pd.to_datetime(df['start_datetime'],    errors='coerce')

mask_ghost = (df['status'] == 'closed') & df['closed_datetime'].isna() & df['modified_datetime'].notna()
mask_true  = df['closed_datetime'].notna()

print("=== GHOST CLOSURE AUDIT ===")
print(f"Total events:                       {len(df)}")
print(f"Has closed_datetime (TRUE obs):     {mask_true.sum()}")
print(f"status=closed + NO closed_datetime: {((df['status']=='closed') & df['closed_datetime'].isna()).sum()}")
print(f"Ghost (closed+no_closed+has_mod):   {mask_ghost.sum()}")
print(f"Active events:                      {(df['status']=='active').sum()}")
print()

ghost = df[mask_ghost].copy()
ghost['ghost_dur'] = (ghost['modified_datetime'] - ghost['start_datetime']).dt.total_seconds()/3600
ghost = ghost[ghost['ghost_dur'] > 0]
print(f"Ghost closures with valid duration: {len(ghost)}")
print(f"Ghost duration median:              {ghost['ghost_dur'].median():.2f}h")
print(f"Ghost duration mean:                {ghost['ghost_dur'].mean():.2f}h")
print(f"Ghost duration p90:                 {ghost['ghost_dur'].quantile(0.9):.2f}h")
print(f"Ghost duration max:                 {ghost['ghost_dur'].max():.2f}h")
print()

print("=== NON-CORRIDOR AUDIT ===")
nc = df[df['corridor'] == 'Non-corridor']
unique_coords = nc.groupby(['latitude','longitude']).ngroups
print(f"Non-corridor events:         {len(nc)} ({len(nc)/len(df)*100:.1f}%)")
print(f"Unique lat/lon in Non-corr:  {unique_coords}")
print()

print("=== CIVIC vs TRAFFIC MEDIAN DURATIONS (cap 200h) ===")
true_obs = df[mask_true].copy()
true_obs['dur'] = (true_obs['closed_datetime'] - true_obs['start_datetime']).dt.total_seconds()/3600
true_obs = true_obs[(true_obs['dur'] > 0) & (true_obs['dur'] <= 200)]
stats = true_obs.groupby('event_cause')['dur'].agg(['median','mean','count']).sort_values('median')
print(stats.to_string())
