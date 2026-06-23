import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def generate_eda_charts():
    out_dir = Path("data/charts")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_parquet("data/processed/survival_ready.parquet")
    
    # 1. Event Cause Distribution
    plt.figure(figsize=(12, 6))
    cause_counts = df['event_cause'].value_counts().head(10)
    sns.barplot(x=cause_counts.values, y=cause_counts.index, palette="viridis")
    plt.title("Top 10 Event Causes in Bengaluru (ASTraM)")
    plt.xlabel("Number of Events")
    plt.tight_layout()
    plt.savefig(out_dir / "01_event_causes.png", dpi=300)
    plt.close()
    
    # 2. Temporal Heatmap: Hour vs Day of Week
    plt.figure(figsize=(10, 6))
    heatmap_data = pd.crosstab(df['day_of_week'], df['hour'])
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    sns.heatmap(heatmap_data, cmap="YlOrRd", yticklabels=days)
    plt.title("Event Frequency by Hour and Day of Week")
    plt.xlabel("Hour of Day")
    plt.ylabel("Day of Week")
    plt.tight_layout()
    plt.savefig(out_dir / "02_temporal_heatmap.png", dpi=300)
    plt.close()
    
    # 3. Duration distributions by event cause
    plt.figure(figsize=(12, 6))
    top_causes = cause_counts.index[:5]
    sns.boxplot(data=df[df['event_cause'].isin(top_causes)], x="duration_hrs", y="event_cause")
    plt.title("Duration Distribution by Top Event Causes")
    plt.xlabel("Duration (Hours)")
    plt.xlim(0, df['duration_hrs'].quantile(0.95)) # Cap at 95th percentile for readability
    plt.tight_layout()
    plt.savefig(out_dir / "03_duration_boxplots.png", dpi=300)
    plt.close()
    
    # 4. Top Corridors
    plt.figure(figsize=(12, 6))
    corridor_counts = df['corridor'].value_counts().head(10)
    sns.barplot(x=corridor_counts.values, y=corridor_counts.index, palette="magma")
    plt.title("Top 10 Affected Corridors")
    plt.tight_layout()
    plt.savefig(out_dir / "04_top_corridors.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    print("Generating EDA charts...")
    generate_eda_charts()
    print("Charts saved to data/charts/")
