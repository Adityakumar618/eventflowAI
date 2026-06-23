import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from scipy.stats import linregress

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


class ResponseEfficiencyAnalyzer:
    def compute_station_scores(self, df: pd.DataFrame, models_dir: str = "data/precomputed") -> pd.DataFrame:
        logging.info("Computing police station efficiency scores...")
        closed = df[df['event_observed'] == 1].copy()
        
        if 'police_station' not in closed.columns:
            logging.warning("No police_station column found.")
            return pd.DataFrame()

        # Average predicted duration (from KM median or a simple global mean)
        global_median = closed['duration_hrs'].median()

        station_stats = closed.groupby('police_station').agg(
            n_events          = ('duration_hrs', 'count'),
            mean_actual_hrs   = ('duration_hrs', 'mean'),
            median_actual_hrs = ('duration_hrs', 'median'),
        ).round(2)

        station_stats = station_stats[station_stats['n_events'] >= 10]
        station_stats['response_ratio'] = (
            station_stats['mean_actual_hrs'] / global_median
        ).round(3)

        station_stats['efficiency_score'] = (
            100 * (1 - ((station_stats['response_ratio'] - 1).clip(0, 2) / 2))
        ).round(1)

        station_stats['performance_band'] = station_stats['efficiency_score'].apply(
            lambda s: 'EXCELLENT' if s >= 85 else
                      'GOOD'      if s >= 70 else
                      'AVERAGE'   if s >= 55 else
                      'NEEDS REVIEW'
        )

        station_stats = station_stats.sort_values('efficiency_score', ascending=False)

        out = Path(models_dir)
        out.mkdir(parents=True, exist_ok=True)
        station_stats.to_parquet(out / 'station_efficiency.parquet')
        logging.info(f"Saved station efficiency to {out / 'station_efficiency.parquet'}")
        return station_stats

    def generate_insights(self, station_stats: pd.DataFrame) -> list:
        insights = []
        if len(station_stats) == 0:
            return ["Insufficient data to generate insights."]

        best = station_stats.iloc[0]
        insights.append(
            f"🏆 **{best.name}** leads with {best['efficiency_score']:.0f}/100 efficiency "
            f"across {int(best['n_events'])} events — "
            f"average resolution time {best['mean_actual_hrs']:.1f}h."
        )

        needs_review = station_stats[station_stats['performance_band'] == 'NEEDS REVIEW']
        if len(needs_review) > 0:
            worst = needs_review.iloc[-1]
            insights.append(
                f"⚠️ **{worst.name}** is taking {worst['response_ratio']:.2f}× the city average — "
                f"consider resource review or workload balancing."
            )

        excellent = station_stats[station_stats['performance_band'] == 'EXCELLENT']
        insights.append(
            f"✅ **{len(excellent)} stations** rated EXCELLENT out of {len(station_stats)} with sufficient data."
        )
        return insights


class TrendIntelligenceEngine:
    def compute_trends(self, df: pd.DataFrame) -> dict:
        logging.info("Computing trend intelligence...")

        df_ts = df.copy()
        df_ts['period'] = df_ts['start_datetime'].dt.to_period('M')
        monthly = df_ts.groupby('period').agg(
            event_count   = ('duration_hrs', 'count'),
            median_duration = ('duration_hrs', 'median'),
        )

        x = np.arange(len(monthly))

        def safe_linregress(y_vals):
            try:
                result = linregress(x, y_vals)
                return result.slope, result.pvalue
            except Exception:
                return 0.0, 1.0

        freq_slope, freq_p  = safe_linregress(monthly['event_count'].values)
        dur_slope,  dur_p   = safe_linregress(monthly['median_duration'].values)

        # Corridor trends
        corridor_trends = {}
        for corridor, group in df_ts.groupby('corridor'):
            monthly_c = group.groupby('period')['duration_hrs'].count()
            if len(monthly_c) >= 3:
                s, _ = safe_linregress(monthly_c.values.astype(float))
                corridor_trends[str(corridor)] = round(float(s), 2)

        deteriorating = sorted(corridor_trends.items(), key=lambda x: -x[1])[:5]
        improving     = sorted(corridor_trends.items(), key=lambda x:  x[1])[:5]

        # Cause trends
        cause_trends = {}
        for cause, group in df_ts.groupby('event_cause'):
            monthly_c = group.groupby('period')['duration_hrs'].count()
            if len(monthly_c) >= 3:
                s, _ = safe_linregress(monthly_c.values.astype(float))
                cause_trends[str(cause)] = round(float(s), 2)

        result = {
            'frequency_trend': {
                'slope':     round(float(freq_slope), 2),
                'pvalue':    round(float(freq_p), 4),
                'direction': 'INCREASING' if freq_slope > 0 else 'DECREASING',
                'insight':   (
                    f"Event frequency is {'increasing' if freq_slope > 0 else 'decreasing'} "
                    f"by {abs(freq_slope):.1f} events/month (p={freq_p:.3f})"
                )
            },
            'duration_trend': {
                'slope':     round(float(dur_slope), 3),
                'pvalue':    round(float(dur_p), 4),
                'direction': 'GETTING SLOWER' if dur_slope > 0 else 'GETTING FASTER',
                'insight':   (
                    f"Median resolution time is "
                    f"{'increasing' if dur_slope > 0 else 'decreasing'} "
                    f"by {abs(dur_slope):.1f}h/month"
                )
            },
            'deteriorating_corridors': [{'corridor': c, 'slope': s} for c, s in deteriorating],
            'improving_corridors':     [{'corridor': c, 'slope': s} for c, s in improving],
            'cause_trends':            cause_trends,
            'monthly_summary':         {
                str(p): {'count': int(row['event_count']), 'median_hrs': round(float(row['median_duration']), 2)}
                for p, row in monthly.iterrows()
            }
        }

        out = Path("data/precomputed")
        out.mkdir(parents=True, exist_ok=True)
        with open(out / 'trend_analysis.json', 'w') as f:
            json.dump(result, f, indent=2)
        logging.info("Trend analysis saved.")
        return result


if __name__ == "__main__":
    df = pd.read_parquet("data/processed/survival_ready.parquet")

    resp = ResponseEfficiencyAnalyzer()
    stats = resp.compute_station_scores(df)
    print("\n=== TOP 5 STATIONS ===")
    print(stats.head())

    trend = TrendIntelligenceEngine()
    res = trend.compute_trends(df)
    print("\n=== TRENDS ===")
    print(res['frequency_trend']['insight'])
    print(res['duration_trend']['insight'])
