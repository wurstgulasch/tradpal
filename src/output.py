import json
import pandas as pd
from config.settings import OUTPUT_FILE

def save_signals_to_json(df):
    """
    Saves the signals as a JSON file.
    """
    # Only rows with signals
    signals = df[(df['Buy_Signal'] == 1) | (df['Sell_Signal'] == 1)].copy()
    
    # Convert to dict
    signals_dict = signals.to_dict(orient='records')
    
    # Save JSON
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(signals_dict, f, indent=4, default=str)
    
    print(f"Signals saved to {OUTPUT_FILE}")

def get_latest_signals(df):
    """
    Returns the latest signals as JSON.
    """
    latest = df.tail(1).to_dict(orient='records')[0]
    return json.dumps(latest, default=str, indent=4)
