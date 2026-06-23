import pandas as pd
import yaml
from pathlib import Path
import logging
from src.mappls_client import MapplsClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def capacity_reduction_score(event: dict) -> float:
    """
    Proxy for how much road capacity is reduced by this event.
    """
    # If number of lanes is not available, assume 2 lanes per direction (4 total)
    base_lanes = event.get('number_of_lanes', 4)
    if pd.isna(base_lanes) or base_lanes == 0:
        base_lanes = 4

    blocked_lanes_map = {
        'vehicle_breakdown': 1,      
        'accident':          2,      
        'tree_fall':         base_lanes,  
        'water_logging':     base_lanes,  
        'construction':      max(1, base_lanes // 2),  
        'procession':        base_lanes,  
        'vip_movement':      base_lanes,  
        'congestion':        0,       
        'pot_holes':         0.5,     
    }
    cause = event.get('event_cause', 'unknown')
    blocked = blocked_lanes_map.get(cause, 1)

    capacity_reduction = min(1.0, blocked / max(base_lanes, 1))

    # Scale by whether road closure is required
    closure = event.get('requires_road_closure', 0)
    if closure == 1:
        capacity_reduction = min(1.0, capacity_reduction * 1.5)

    return round(capacity_reduction, 3)

class STISCalculator:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.mappls = MapplsClient(self.config['MAPPLS_API_KEY'], self.config['DATA_CACHE_DIR'])
        
        self.MAJOR_OD_PAIRS = [
            ("Kempegowda Bus Station", "Electronic City", (12.977, 77.572), (12.843, 77.674)),
            ("Hebbal", "Silk Board", (13.035, 77.597), (12.917, 77.624)),
            ("Mekhri Circle", "Marathahalli", (13.009, 77.577), (12.954, 77.700)),
            ("Yeshwanthpura", "Koramangala", (13.022, 77.540), (12.935, 77.624)),
        ]

    def compute_time_delta(self, event: dict) -> dict:
        results = []
        event_coords = (event['latitude'], event['longitude'])
        
        if pd.isna(event_coords[0]) or pd.isna(event_coords[1]):
            return {'impact_mins_max': 0, 'od_pairs_affected': 0, 'details': []}

        for (orig_name, dest_name, orig_coords, dest_coords) in self.MAJOR_OD_PAIRS:
            try:
                normal_time = self.mappls.get_route(orig_coords, dest_coords)
                avoid_time  = self.mappls.get_route_avoiding(orig_coords, dest_coords, event_coords)
                
                # Check if API returned successful responses
                if 'routes' in normal_time and len(normal_time['routes']) > 0 and \
                   'routes' in avoid_time and len(avoid_time['routes']) > 0:
                   
                    normal_secs = normal_time['routes'][0]['duration']
                    avoid_secs  = avoid_time['routes'][0]['duration']

                    if avoid_secs > normal_secs:
                        delta_mins = (avoid_secs - normal_secs) / 60
                    else:
                        delta_mins = 0

                    results.append({
                        'od_pair': f"{orig_name} -> {dest_name}",
                        'normal_mins': round(normal_secs / 60, 1),
                        'detour_mins': round(avoid_secs / 60, 1),
                        'impact_mins': round(delta_mins, 1),
                        'impact_pct':  round(delta_mins / max(normal_secs / 60, 1) * 100, 1),
                    })
            except Exception as e:
                logging.warning(f"Failed to compute delta for {orig_name}->{dest_name}: {e}")
                continue

        if not results:
            return {'impact_mins_max': 0, 'od_pairs_affected': 0, 'details': []}

        return {
            'impact_mins_max':   max(r['impact_mins'] for r in results),
            'impact_mins_avg':   round(sum(r['impact_mins'] for r in results) / len(results), 1),
            'od_pairs_affected': sum(1 for r in results if r['impact_mins'] > 2),
            'worst_od':          max(results, key=lambda x: x['impact_mins']),
            'details':           results,
        }

    def compute(self, event: dict, centrality_score: float = 0.5) -> dict:
        cap  = capacity_reduction_score(event)
        
        # Centrality is 0.5 default if graph is too large to compute
        # It gets overridden if pre-computed values exist
        cent = centrality_score
        
        mpl  = self.compute_time_delta(event)

        # Normalize Mappls delta: 0 mins = 0.0, 30+ mins = 1.0
        mpl_norm = min(1.0, mpl.get('impact_mins_max', 0) / 30.0)

        w_cap = self.config.get('STIS_WEIGHT_CAPACITY', 0.35)
        w_cent = self.config.get('STIS_WEIGHT_CENTRALITY', 0.30)
        w_rout = self.config.get('STIS_WEIGHT_ROUTING', 0.35)
        
        raw_score = w_cap * cap + w_cent * cent + w_rout * mpl_norm
        stis_score = round(raw_score * 10, 1)

        worst_od = mpl.get('worst_od', {})
        worst_od_name = worst_od.get('od_pair', 'local')
        impact_mins = mpl.get('impact_mins_max', 0)

        return {
            'stis': stis_score,
            'severity_label': (
                'CRITICAL' if stis_score >= 7 else
                'HIGH'     if stis_score >= 4 else
                'MEDIUM'   if stis_score >= 2 else
                'LOW'
            ),
            'components': {
                'capacity_reduction': round(cap, 3),
                'junction_centrality': round(cent, 4),
                'mappls_time_delta_mins': impact_mins,
                'od_pairs_affected': mpl.get('od_pairs_affected', 0),
            },
            'plain_english': (
                f"This event reduces road capacity by {int(cap*100)}%, "
                f"is located at a junction carrying {int(cent*100)}% of network traffic, "
                f"and forces commuters on the {worst_od_name} "
                f"corridor to take a {impact_mins:.0f}-minute detour."
            )
        }
