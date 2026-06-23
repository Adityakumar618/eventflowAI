import pandas as pd
import yaml
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def audit_field_timing(df: pd.DataFrame, field: str) -> dict:
    """
    For fields without explicit set_timestamp, use proxy:
    If field is null for events that are still 'active', 
    it was likely not set at T=0.
    """
    if field not in df.columns:
        return {'field': field, 'error': 'Not in dataframe'}
        
    active_events = df[df['status'] == 'active']
    closed_events = df[df['status'].isin(['closed', 'resolved'])]
    
    if len(active_events) == 0:
        return {'field': field, 'error': 'No active events to compare'}
        
    null_rate_active = active_events[field].isna().mean()
    null_rate_closed = closed_events[field].isna().mean()

    # If null_rate_active >> null_rate_closed, field is set post-resolution -> leakage
    leakage_signal = null_rate_active - null_rate_closed

    return {
        'field': field,
        'null_rate_active': round(null_rate_active, 3),
        'null_rate_closed': round(null_rate_closed, 3),
        'leakage_signal': round(leakage_signal, 3),
        'verdict': 'LEAKAGE RISK' if leakage_signal > 0.1 else 'SAFE'
    }

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    raw_path = Path(config["DATA_RAW_DIR"]) / "astram_events.csv"
    logging.info(f"Loading raw data from {raw_path}")
    df = pd.read_csv(raw_path)
    
    fields_to_check = ['priority', 'requires_road_closure', 'corridor', 'number_of_lanes']
    
    logging.info("\n--- LEAKAGE AUDIT RESULTS ---")
    for field in fields_to_check:
        res = audit_field_timing(df, field)
        logging.info(f"Field: {res.get('field', field)}")
        if 'error' in res:
            logging.info(f"  Error: {res['error']}")
        else:
            logging.info(f"  Nulls in ACTIVE: {res['null_rate_active']:.1%}")
            logging.info(f"  Nulls in CLOSED: {res['null_rate_closed']:.1%}")
            logging.info(f"  Leakage Signal:  {res['leakage_signal']:.3f}")
            logging.info(f"  VERDICT:         {res['verdict']}")
        logging.info("-" * 30)
