import ccxt
import pandas as pd
from datetime import datetime, timedelta
from config.settings import EXCHANGE, SYMBOL, TIMEFRAME, LOOKBACK_DAYS
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def fetch_data(limit=200):
    """
    Fetches recent price data using ccxt.
    For continuous monitoring, fetches only recent candles for efficiency.
    """
    exchange_class = getattr(ccxt, EXCHANGE)

    # Use API credentials from environment if available
    api_key = os.getenv('TRADPAL_API_KEY')
    api_secret = os.getenv('TRADPAL_API_SECRET')

    if api_key and api_secret:
        exchange = exchange_class({
            'apiKey': api_key,
            'secret': api_secret,
        })
    else:
        exchange = exchange_class()

    # For continuous monitoring, fetch only recent data (last 200 candles)
    # This is much more efficient than fetching 7 days of data every time
    ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=limit)

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    return df

def fetch_historical_data(symbol=None, exchange_name=None, timeframe=None, limit=None):
    """
    Fetches historical data with configurable parameters.
    Used for both backtesting and live data fetching.
    """
    # Use provided parameters or defaults
    symbol = symbol or SYMBOL
    exchange_name = exchange_name or EXCHANGE
    timeframe = timeframe or TIMEFRAME
    limit = limit or 1000

    exchange_class = getattr(ccxt, exchange_name)

    # Use API credentials from environment if available
    api_key = os.getenv('TRADPAL_API_KEY')
    api_secret = os.getenv('TRADPAL_API_SECRET')

    if api_key and api_secret:
        exchange = exchange_class({
            'apiKey': api_key,
            'secret': api_secret,
        })
    else:
        exchange = exchange_class()

    # For backtesting, fetch historical data
    if limit > 200:
        since = exchange.parse8601((datetime.now() - timedelta(days=LOOKBACK_DAYS)).isoformat())
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
    else:
        # For live monitoring, fetch recent data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    return df

def validate_data(df):
    """
    Validates OHLCV DataFrame structure and data integrity.
    """
    # Check if input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise AttributeError("Input must be a pandas DataFrame")

    # Check if DataFrame is not empty
    if len(df) == 0:
        raise ValueError("DataFrame is empty")

    # Check for required columns (they might be in index or columns)
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing required columns. Required: {required_columns}")

    # Check data types and convert if possible
    critical_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in critical_columns:
        try:
            # Try to convert to numeric
            pd.to_numeric(df[col], errors='coerce')
        except (ValueError, TypeError):
            raise ValueError(f"Column '{col}' contains non-numeric data")

    # Check for NaN values in critical columns
    for col in ['open', 'high', 'low', 'close']:
        if df[col].isna().any():
            raise ValueError(f"NaN values found in {col} column")

    # Check for negative prices or volumes (only if numeric)
    try:
        if (df[critical_columns[:-1]] < 0).any().any():  # Exclude volume for now
            raise ValueError("Negative values found in price columns")
        if (df['volume'] < 0).any():
            raise ValueError("Negative values found in volume column")
    except TypeError:
        raise ValueError("Cannot perform numeric comparisons on non-numeric data")

    # Check OHLC relationships (only if numeric)
    try:
        invalid_ohlc = (
            (df['high'] < df['low']) |  # High should be >= low
            (df['open'] < df['low']) | (df['open'] > df['high']) |  # Open should be between low and high
            (df['close'] < df['low']) | (df['close'] > df['high'])   # Close should be between low and high
        )
        if invalid_ohlc.any():
            raise ValueError("Invalid OHLC relationships detected")
    except TypeError:
        raise ValueError("Cannot validate OHLC relationships with non-numeric data")

    # Check timestamp ordering if timestamp is index
    if isinstance(df.index, pd.DatetimeIndex):
        if not df.index.is_monotonic_increasing:
            raise ValueError("Timestamps are not in chronological order")

    return True
