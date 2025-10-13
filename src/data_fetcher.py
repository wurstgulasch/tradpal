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
# Import cache functionality
try:
    from .cache import cache_api_call
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    def cache_api_call(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
# Import validation functionality
try:
    from .input_validation import InputValidator, validate_api_inputs
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
    def validate_api_inputs(*args, **kwargs):
        pass
    class InputValidator:
        pass

# Import error handling functionality
try:
    from .error_handling import error_boundary, NetworkError, DataError
    ERROR_HANDLING_AVAILABLE = True
except ImportError:
    ERROR_HANDLING_AVAILABLE = False
    def error_boundary(func):
        return func
    class NetworkError(Exception):
        pass
    class DataError(Exception):
        pass
import time
import random
import numpy as np

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
def fetch_historical_data(symbol=SYMBOL, exchange_name=EXCHANGE, timeframe=TIMEFRAME, limit=HISTORICAL_DATA_LIMIT, start_date=None, show_progress=True):
    """
    Fetches historical OHLCV data from specified exchange using ccxt.
    Supports advanced pagination for large datasets with progress tracking and adaptive batching.

    Args:
        symbol: Trading symbol (e.g., 'BTC/USDT')
        exchange_name: Exchange name (e.g., 'kraken', 'binance')
        timeframe: Timeframe (e.g., '1m', '1h', '1d')
        limit: Maximum number of candles to fetch
        start_date: Start date for historical data (datetime object)
        show_progress: Whether to show progress indicators

    Returns:
        DataFrame with OHLCV data or "retry" on recoverable errors
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
        since = exchange.parse8601((datetime.now() - timedelta(days=DEFAULT_HISTORICAL_DAYS)).isoformat())

    # Advanced pagination with adaptive batching and progress tracking
    all_ohlcv = []
    remaining_limit = limit
    total_fetched = 0

    # Adaptive batch sizing based on exchange and timeframe
    base_batch_size = _get_adaptive_batch_size(exchange_name, timeframe)
    current_batch_size = base_batch_size
    consecutive_errors = 0
    max_consecutive_errors = 3

    # Progress tracking
    start_time = time.time()
    last_progress_time = start_time

    if show_progress:
        print(f"📊 Fetching {limit} candles of {symbol} from {exchange_name} ({timeframe})...")

    while remaining_limit > 0:
        # Check if we exceeded max consecutive errors
        if consecutive_errors >= max_consecutive_errors:
            if show_progress:
                print(f"❌ Too many consecutive errors ({consecutive_errors}), aborting")
            return "retry"

        try:
            # Adaptive batch size reduction on errors
            fetch_limit = min(remaining_limit, current_batch_size)

            # Rate limiting
            rate_limiter.record_request()
            if not rate_limiter.can_make_request():
                sleep_time = rate_limiter.get_backoff_time(1)
                if show_progress:
                    print(f"⏳ Rate limit reached, waiting {sleep_time:.1f}s...")
                time.sleep(sleep_time)

            # Fetch batch
            batch_start_time = time.time()
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, fetch_limit)
            batch_duration = time.time() - batch_start_time

            if not ohlcv:
                if show_progress:
                    print("ℹ️  No more data available from exchange")
                break

            # Validate batch data
            if not _validate_batch_data(ohlcv):
                consecutive_errors += 1
                current_batch_size = max(10, current_batch_size // 2)  # Reduce batch size
                if show_progress:
                    print(f"⚠️  Invalid batch data, reducing batch size to {current_batch_size}")
                continue

            # Add batch to results
            batch_size = len(ohlcv)
            all_ohlcv.extend(ohlcv)
            total_fetched += batch_size
            remaining_limit -= batch_size
            consecutive_errors = 0  # Reset error counter

            # Update since for next batch (next candle after last one)
            last_timestamp = ohlcv[-1][0]
            # Convert timeframe to milliseconds
            timeframe_ms = _timeframe_to_ms(timeframe)
            since = last_timestamp + timeframe_ms

            # Progress reporting
            current_time = time.time()
            if show_progress and (current_time - last_progress_time > 5):  # Update every 5 seconds
                elapsed = current_time - start_time
                progress = (total_fetched / limit) * 100 if limit > 0 else 0
                rate = total_fetched / elapsed if elapsed > 0 else 0
                eta = (remaining_limit / rate) if rate > 0 else 0

                print(f"📈 Progress: {total_fetched}/{limit} candles ({progress:.1f}%) | "
                      f"Rate: {rate:.1f} candles/sec | ETA: {eta:.0f}s | "
                      f"Batch: {batch_size} in {batch_duration:.2f}s")
                last_progress_time = current_time

            # Safety check to avoid infinite loops
            if batch_size < fetch_limit:
                if show_progress:
                    print(f"ℹ️  Reached end of available data ({batch_size}/{fetch_limit} candles in last batch)")
                break

            # Adaptive batch size adjustment based on performance
            if batch_duration > 5:  # Slow response
                current_batch_size = max(10, current_batch_size // 2)
            elif batch_duration < 1 and current_batch_size < base_batch_size * 2:  # Fast response
                current_batch_size = min(base_batch_size * 2, current_batch_size * 2)

        except Exception as e:
            # Check if this is a rate limit error
            error_str = str(type(e).__name__).lower()
            error_msg = str(e).lower()

            if 'ratelimit' in error_str or 'ratelimit' in error_msg or 'rate limit' in error_msg or 'rate_limit' in error_msg:
                consecutive_errors += 1
                sleep_time = rate_limiter.get_backoff_time(consecutive_errors)
                if show_progress:
                    print(f"🚦 Rate limit exceeded, waiting {sleep_time:.1f}s... (attempt {consecutive_errors}/{max_consecutive_errors})")
                time.sleep(sleep_time)
                current_batch_size = max(10, current_batch_size // 2)

            elif 'network' in error_str or 'network' in error_msg or 'connection' in error_msg:
                consecutive_errors += 1
                sleep_time = rate_limiter.get_backoff_time(consecutive_errors)
                if show_progress:
                    print(f"🌐 Network error, retrying in {sleep_time:.1f}s... (attempt {consecutive_errors}/{max_consecutive_errors})")
                time.sleep(sleep_time)

            else:
                # Check if this is a recoverable error
                if any(keyword in error_msg for keyword in ['timeout', 'connection', 'temporary']):
                    consecutive_errors += 1
                    sleep_time = rate_limiter.get_backoff_time(consecutive_errors)
                    if show_progress:
                        print(f"⚠️  Temporary error, retrying in {sleep_time:.1f}s... (attempt {consecutive_errors}/{max_consecutive_errors})")
                    time.sleep(sleep_time)
                    current_batch_size = max(10, current_batch_size // 2)
                else:
                    # Non-recoverable error
                    if show_progress:
                        print(f"❌ Non-recoverable error: {e}")
                    return "retry"
    if show_progress:
        total_time = time.time() - start_time
        final_rate = total_fetched / total_time if total_time > 0 else 0
        print(f"✅ Completed: {total_fetched} candles fetched in {total_time:.1f}s "
              f"(avg {final_rate:.1f} candles/sec)")

    # Create DataFrame
    if not all_ohlcv:
        if show_progress:
            print("⚠️  No data fetched")
        return pd.DataFrame()

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    # Remove duplicates (can happen with pagination)
    df = df[~df.index.duplicated(keep='first')]

    # Validate the fetched data if not empty
    if len(df) > 0:
        validate_data(df)

    return df

def _timeframe_to_ms(timeframe: str) -> int:
    """
    Convert timeframe string to milliseconds.
    
    Args:
        timeframe: Timeframe string (e.g., '1m', '1h', '1d')
        
    Returns:
        Timeframe in milliseconds
    """
    # Timeframe multipliers
    timeframe_map = {
        '1m': 60 * 1000,
        '5m': 5 * 60 * 1000,
        '15m': 15 * 60 * 1000,
        '30m': 30 * 60 * 1000,
        '1h': 60 * 60 * 1000,
        '4h': 4 * 60 * 60 * 1000,
        '1d': 24 * 60 * 60 * 1000,
        '1w': 7 * 24 * 60 * 60 * 1000,
    }
    
    return timeframe_map.get(timeframe, 60 * 60 * 1000)  # Default to 1h

def _get_adaptive_batch_size(exchange_name: str, timeframe: str) -> int:
    """
    Get adaptive batch size based on exchange and timeframe.

    Args:
        exchange_name: Name of the exchange
        timeframe: Timeframe string

    Returns:
        Recommended batch size
    """
    # Base batch sizes by exchange
    exchange_batches = {
        'kraken': KRAKEN_MAX_PER_REQUEST,
        'binance': 1000,
        'coinbase': 300,
        'default': 500
    }

    base_batch = exchange_batches.get(exchange_name.lower(), exchange_batches['default'])

    # Adjust for timeframe (shorter timeframes = smaller batches)
    timeframe_multipliers = {
        '1m': 1.0,
        '5m': 1.2,
        '15m': 1.5,
        '30m': 2.0,
        '1h': 3.0,
        '4h': 4.0,
        '1d': 5.0,
        '1w': 7.0
    }

    multiplier = timeframe_multipliers.get(timeframe, 1.0)
    return int(base_batch / multiplier)

def _validate_batch_data(ohlcv_batch: list) -> bool:
    """
    Validate a batch of OHLCV data.

    Args:
        ohlcv_batch: List of OHLCV candles

    Returns:
        True if batch is valid, False otherwise
    """
    if not ohlcv_batch:
        return False

    try:
        for candle in ohlcv_batch:
            if len(candle) != 6:  # OHLCV + timestamp
                return False

            timestamp, open_price, high, low, close, volume = candle

            # Check for None/NaN values
            if any(x is None or (isinstance(x, float) and np.isnan(x))
                   for x in [timestamp, open_price, high, low, close, volume]):
                return False

            # Basic OHLC validation
            if not (low <= open_price <= high and low <= close <= high):
                return False

            # Check for negative values
            if any(x < 0 for x in [open_price, high, low, close, volume]):
                return False

        return True

    except (TypeError, ValueError):
        return False

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
