import ccxt
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict
from config.settings import (
    SYMBOL, EXCHANGE, TIMEFRAME, DEFAULT_DATA_LIMIT, HISTORICAL_DATA_LIMIT, 
    MAX_RETRIES_LIVE, MAX_RETRIES_HISTORICAL, CACHE_TTL_LIVE, CACHE_TTL_HISTORICAL, 
    KRAKEN_MAX_PER_REQUEST, DEFAULT_HISTORICAL_DAYS, LOOKBACK_DAYS, WEBSOCKET_DATA_ENABLED
)
import os
import asyncio
import aiohttp
from dotenv import load_dotenv
from .cache import cache_api_call
from .input_validation import InputValidator, validate_api_inputs
from .error_handling import error_boundary, NetworkError, DataError
import time
import random

# Load environment variables
load_dotenv()

# Import WebSocket functionality
try:
    from .websocket_data_fetcher import (
        WebSocketDataFetcher, fetch_realtime_data, is_websocket_available
    )
    WEBSOCKET_FETCHER_AVAILABLE = True
except ImportError:
    WEBSOCKET_FETCHER_AVAILABLE = False

class AdaptiveRateLimiter:
    """Adaptive Rate Limiter für Exchange-API-Aufrufe."""

    def __init__(self, exchange_name: str = EXCHANGE):
        self.exchange_name = exchange_name
        self.request_times = []
        self.rate_limits = self._get_exchange_limits()
        self.backoff_factor = 1.0
        self.consecutive_errors = 0

    def _get_exchange_limits(self) -> Dict[str, int]:
        """Hole Rate-Limits für verschiedene Exchanges."""
        limits = {
            'kraken': {'requests_per_second': 1, 'requests_per_minute': 20},
            'binance': {'requests_per_second': 10, 'requests_per_minute': 1200},
            'coinbase': {'requests_per_second': 10, 'requests_per_minute': 100},
            'default': {'requests_per_second': 1, 'requests_per_minute': 60}
        }
        return limits.get(self.exchange_name.lower(), limits['default'])

    def can_make_request(self) -> bool:
        """Prüfe, ob ein Request gemacht werden kann."""
        now = time.time()

        # Entferne alte Requests außerhalb des Zeitfensters
        self.request_times = [t for t in self.request_times if now - t < 60]

        # Prüfe Rate-Limits
        requests_per_minute = len(self.request_times)
        requests_per_second = len([t for t in self.request_times if now - t < 1])

        return (requests_per_second < self.rate_limits['requests_per_second'] and
                requests_per_minute < self.rate_limits['requests_per_minute'])

    def record_request(self):
        """Zeichne einen Request auf."""
        self.request_times.append(time.time())

    def get_backoff_time(self, retry_count: int) -> float:
        """Berechne Backoff-Zeit basierend auf Retry-Count und Fehlern."""
        base_backoff = min(2 ** retry_count, 300)  # Max 5 Minuten

        # Erhöhe Backoff bei aufeinanderfolgenden Fehlern
        if self.consecutive_errors > 2:
            base_backoff *= 2

        # Füge Jitter hinzu, um Thundering Herd zu vermeiden
        jitter = random.uniform(0.1, 1.0) * base_backoff * 0.1

        return base_backoff + jitter

    def record_error(self):
        """Zeichne einen Fehler auf."""
        self.consecutive_errors += 1

    def record_success(self):
        """Zeichne einen Erfolg auf."""
        self.consecutive_errors = max(0, self.consecutive_errors - 1)

# Global Rate Limiter Instanz
rate_limiter = AdaptiveRateLimiter()

def adaptive_retry(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator für adaptive Retries mit Rate-Limiting."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    # Warte auf Rate-Limit
                    while not rate_limiter.can_make_request():
                        backoff_time = rate_limiter.get_backoff_time(attempt)
                        time.sleep(min(backoff_time, 1.0))  # Max 1 Sekunde warten

                    # Mache Request
                    rate_limiter.record_request()
                    result = func(*args, **kwargs)

                    # Erfolg aufzeichnen
                    rate_limiter.record_success()
                    return result

                except Exception as e:
                    last_exception = e
                    rate_limiter.record_error()

                    if attempt < max_retries:
                        backoff_time = rate_limiter.get_backoff_time(attempt)
                        time.sleep(backoff_time)

            # Alle Retries gescheitert
            raise last_exception

        return wrapper
    return decorator

@adaptive_retry(max_retries=MAX_RETRIES_LIVE)
@cache_api_call(ttl_seconds=CACHE_TTL_LIVE)  # Cache for live data
def fetch_data(limit: int = DEFAULT_DATA_LIMIT) -> pd.DataFrame:
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

@adaptive_retry(max_retries=MAX_RETRIES_HISTORICAL)
@cache_api_call(ttl_seconds=CACHE_TTL_HISTORICAL)  # Cache for historical data
def fetch_historical_data(symbol=SYMBOL, exchange_name=EXCHANGE, timeframe=TIMEFRAME, limit=HISTORICAL_DATA_LIMIT, start_date=None):
    """
    Fetches historical OHLCV data from specified exchange using ccxt.
    Supports pagination for large datasets.
    """
    # Validate inputs
    validate_api_inputs(symbol=symbol, exchange=exchange_name, timeframe=timeframe, limit=limit)

    # Override exchange if specified
    if exchange_name != EXCHANGE:
        exchange_name = exchange_name

    # Get exchange class
    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class()

    # Set since based on start_date or default
    if start_date:
        since = exchange.parse8601(start_date.isoformat())
    else:
        since = exchange.parse8601((datetime.now() - timedelta(days=DEFAULT_HISTORICAL_DAYS)).isoformat())  # Default historical period

    # For backtesting, fetch historical data with pagination if needed
    all_ohlcv = []
    remaining_limit = limit
    max_per_request = KRAKEN_MAX_PER_REQUEST  # Kraken's limit for 1m timeframe

    while remaining_limit > 0:
        fetch_limit = min(remaining_limit, max_per_request)
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, fetch_limit)
        except Exception as e:
            # Return retry on recoverable errors
            return "retry"
        
        if not ohlcv:
            break
        
        all_ohlcv.extend(ohlcv)
        
        # Update since for next batch
        last_timestamp = ohlcv[-1][0]
        since = last_timestamp + (exchange.timeframes[timeframe] * 1000)  # Next candle
        
        remaining_limit -= len(ohlcv)
        
        # Safety check to avoid infinite loops
        if len(ohlcv) < fetch_limit:
            break

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    # Validate the fetched data if not empty
    if len(df) > 0:
        validate_data(df)

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


def fetch_data_realtime(symbol: str = SYMBOL, exchange: str = EXCHANGE,
                       timeframe: str = TIMEFRAME, duration: int = 60) -> pd.DataFrame:
    """
    Fetch real-time data using WebSocket if available, otherwise fall back to REST API.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe for candlestick data
        duration: Duration in seconds to collect data (for WebSocket mode)
        
    Returns:
        DataFrame with OHLCV data
    """
    if WEBSOCKET_DATA_ENABLED and WEBSOCKET_FETCHER_AVAILABLE and is_websocket_available():
        try:
            # Use async WebSocket fetching
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            data = loop.run_until_complete(
                fetch_realtime_data(symbol, exchange, timeframe, duration)
            )
            loop.close()
            
            if not data.empty:
                return data
            else:
                # Fall back to REST API if WebSocket returns no data
                return fetch_data()
        except Exception as e:
            # Fall back to REST API on error
            return fetch_data()
    else:
        # Use standard REST API
        return fetch_data()
