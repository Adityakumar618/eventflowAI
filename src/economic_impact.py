import pandas as pd

class EconomicImpactQuantifier:
    """
    Converts STIS (minutes of delay) into INR economic cost.
    Based on ISEC Bengaluru study: Bengaluru loses ₹11.7 billion/year to traffic congestion
    ≈ ₹32 million/day ≈ ₹1.34 million/hour of city-wide delay
    """

    HOURLY_CITY_COST_LAKH = 134  # ₹1.34 crore/hour ≈ ₹134 lakh/hour

    TRAFFIC_VOLUME_INDEX = {
        'deep_night':     0.1,
        'morning':        1.2,
        'midday':         0.9,
        'evening':        2.0,   # Peak
        'late_night':     0.3,
    }

    def compute(self, event: dict, stis_result: dict, duration_hrs: float) -> dict:
        network_fraction = stis_result['components']['junction_centrality']
        if pd.isna(network_fraction) or network_fraction <= 0:
            network_fraction = 0.05 # default assumption 5% of network
            
        hour = event.get('hour', 12)
        bucket = 'midday'
        if 6 <= hour < 11: bucket = 'morning'
        elif 11 <= hour < 16: bucket = 'midday'
        elif 16 <= hour < 20: bucket = 'evening'
        elif 20 <= hour < 24: bucket = 'late_night'
        else: bucket = 'deep_night'
        
        volume_mult = self.TRAFFIC_VOLUME_INDEX.get(bucket, 1.0)
        
        # We also scale by the actual Mappls time delta if available
        # 1.0 would mean complete gridlock of that network fraction
        impact_severity = stis_result['components']['mappls_time_delta_mins'] / 30.0
        impact_severity = max(0.1, min(1.0, impact_severity))
        
        cost_lakh = (
            self.HOURLY_CITY_COST_LAKH
            * network_fraction
            * duration_hrs
            * volume_mult
            * impact_severity
        )
        
        affected_commuters = int(8_000_000 * network_fraction * volume_mult / 2)
        cost_per_commuter = (cost_lakh * 100_000) / max(affected_commuters, 1)

        return {
            'total_cost_lakh': round(cost_lakh, 1),
            'total_cost_display': f"₹{cost_lakh:.1f} lakh" if cost_lakh < 100 else f"₹{cost_lakh/100:.2f} crore",
            'affected_commuters': affected_commuters,
            'cost_per_commuter_rs': round(cost_per_commuter, 0),
            'methodology': "Based on ISEC Bengaluru study: ₹11.7B annual congestion cost, scaled by centrality, duration, and traffic volume.",
            'plain_english': f"This event is estimated to cost Bengaluru ₹{cost_lakh:.1f} lakh over {duration_hrs:.1f} hours."
        }
