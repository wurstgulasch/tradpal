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

def fetch_historical_data():
    """
    Fetches full historical data for backtesting or initial analysis.
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

    since = exchange.parse8601((datetime.now() - timedelta(days=LOOKBACK_DAYS)).isoformat())

    ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since)

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    return df
