import json
import pandas as pd
from typing import List, Dict, Any, Union, Optional
from config.settings import OUTPUT_FORMAT, OUTPUT_FILE, LOG_LEVEL, LOG_FILE, JSON_INDENT

def save_signals_to_json(signals: Union[pd.DataFrame, List[Dict[str, Any]]], filename: Optional[str] = None) -> None:
    """
    Saves signals to JSON file. Accepts both DataFrame and list formats.
    """
    output_file = filename or OUTPUT_FILE

    if isinstance(signals, pd.DataFrame):
        # Only rows with signals
        signals_data = signals[(signals['Buy_Signal'] == 1) | (signals['Sell_Signal'] == 1)].copy()
        signals_dict = signals_data.to_dict(orient='records')
    elif isinstance(signals, list):
        # Validate that list contains dictionaries
        if not all(isinstance(item, dict) for item in signals):
            raise TypeError("List must contain dictionaries")
        signals_dict = signals
    else:
        raise TypeError("Signals must be a DataFrame or list of dictionaries")

    # Save JSON with explicit encoding
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(signals_dict, f, indent=JSON_INDENT, default=str)

    print(f"Signals saved to {output_file}")

def load_signals_from_json(filename: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Loads signals from JSON file.
    """
    input_file = filename or OUTPUT_FILE

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            signals = json.load(f)
        return signals
    except FileNotFoundError:
        raise FileNotFoundError(f"File {input_file} not found")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON: {e}", e.doc, e.pos)

def format_signal_data(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Formats signal data for JSON output.
    """
    if df.empty:
        return []

    # Convert DataFrame to list of dictionaries
    signals = []
    for _, row in df.iterrows():
        # Get timestamp - prefer 'timestamp' column over index
        if 'timestamp' in row and not pd.isna(row['timestamp']):
            timestamp = row['timestamp']
            if hasattr(timestamp, 'isoformat'):
                timestamp_str = timestamp.isoformat()
            else:
                timestamp_str = str(timestamp)
        else:
            timestamp = row.name
            timestamp_str = timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp)

        signal_dict = {
            'timestamp': timestamp_str,
            'open': float(row['open']) if 'open' in row and not pd.isna(row['open']) else None,
            'high': float(row['high']) if 'high' in row and not pd.isna(row['high']) else None,
            'low': float(row['low']) if 'low' in row and not pd.isna(row['low']) else None,
            'close': float(row['close']) if 'close' in row and not pd.isna(row['close']) else None,
            'volume': float(row['volume']) if 'volume' in row and not pd.isna(row['volume']) else None,
            'Buy_Signal': bool(row.get('Buy_Signal', False)),
            'Sell_Signal': bool(row.get('Sell_Signal', False))
        }

        # Add indicator values if they exist
        indicators = ['EMA9', 'EMA21', 'RSI', 'BB_upper', 'BB_middle', 'BB_lower', 'ATR', 'ADX']
        for indicator in indicators:
            if indicator in row and not pd.isna(row[indicator]):
                signal_dict[indicator] = float(row[indicator])
            else:
                signal_dict[indicator] = None

        # Add risk management values if they exist - in a separate risk_management dict
        risk_management = {}
        risk_fields = ['Position_Size_Absolute', 'Position_Size_Percent', 'Stop_Loss_Buy', 'Take_Profit_Buy', 'Stop_Loss', 'Take_Profit', 'Leverage']
        for field in risk_fields:
            if field in row and not pd.isna(row[field]):
                # Convert field names to snake_case for the risk_management dict
                key = field.lower()
                if field == 'Position_Size_Absolute':
                    key = 'position_size'
                elif field == 'Position_Size_Percent':
                    key = 'position_size_percent'
                elif field == 'Stop_Loss_Buy':
                    key = 'stop_loss'
                elif field == 'Take_Profit_Buy':
                    key = 'take_profit'
                risk_management[key] = float(row[field])

        if risk_management:
            signal_dict['risk_management'] = risk_management

        signals.append(signal_dict)

    return signals

def get_latest_signals(df: pd.DataFrame) -> str:
    """
    Returns the latest signals as JSON.
    """
    latest = df.tail(1).to_dict(orient='records')[0]
    return json.dumps(latest, default=str, indent=4)
