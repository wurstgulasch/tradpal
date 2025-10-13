"""
Modular Data Fetcher for TradPal Indicator System

This module provides a unified interface for fetching market data from various sources
including exchanges (CCXT), Yahoo Finance, Alpha Vantage, and Polygon.io.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from config.settings import (
    SYMBOL, TIMEFRAME, DEFAULT_DATA_LIMIT, HISTORICAL_DATA_LIMIT,
    MAX_RETRIES_LIVE, MAX_RETRIES_HISTORICAL, CACHE_TTL_LIVE, CACHE_TTL_HISTORICAL,
    DEFAULT_HISTORICAL_DAYS, WEBSOCKET_DATA_ENABLED
)
import time
import logging

# Import ccxt for exchange integration (needed for testing)
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    ccxt = None

# Import modular data sources
try:
    from .data_sources.factory import create_data_source
except ImportError:
    from data_sources.factory import create_data_source

# Import cache functionality
try:
    from .cache import cache_api_call
    CACHE_AVAILABLE = True
except ImportError:
    try:
        from cache import cache_api_call
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
    try:
        from input_validation import InputValidator, validate_api_inputs
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
    try:
        from error_handling import error_boundary, NetworkError, DataError
        ERROR_HANDLING_AVAILABLE = True
    except ImportError:
        ERROR_HANDLING_AVAILABLE = False
        def error_boundary(func):
            return func
        class NetworkError(Exception):
            pass
        class DataError(Exception):
            pass

# Import WebSocket functionality
try:
    from .websocket_data_fetcher import (
        WebSocketDataFetcher, fetch_realtime_data, is_websocket_available
    )
    WEBSOCKET_FETCHER_AVAILABLE = True
except ImportError:
    try:
        from websocket_data_fetcher import (
            WebSocketDataFetcher, fetch_realtime_data, is_websocket_available
        )
        WEBSOCKET_FETCHER_AVAILABLE = True
    except ImportError:
        WEBSOCKET_FETCHER_AVAILABLE = False

logger = logging.getLogger(__name__)

# Global data source instance
_data_source = None

class AdaptiveRateLimiter:
    """
    Adaptive rate limiter for API calls with exchange-specific limits.
    """

    def __init__(self, exchange_name: str = 'kraken'):
        """
        Initialize rate limiter for specific exchange.

        Args:
            exchange_name: Exchange name ('binance', 'kraken', 'coinbase', etc.)
        """
        self.exchange_name = exchange_name
        self.request_times = []
        self.consecutive_errors = 0

        # Exchange-specific rate limits
        self.rate_limits = {
            'kraken': {'requests_per_second': 1, 'requests_per_minute': 20},
            'binance': {'requests_per_second': 10, 'requests_per_minute': 1200},
            'coinbase': {'requests_per_second': 3, 'requests_per_minute': 100},
            'default': {'requests_per_second': 1, 'requests_per_minute': 60}
        }

        # Get limits for this exchange, fallback to default
        exchange_limits = self.rate_limits.get(exchange_name, self.rate_limits['default'])
        self.rate_limits = exchange_limits

    def can_make_request(self) -> bool:
        """
        Check if a request can be made based on rate limits.

        Returns:
            True if request can be made, False otherwise
        """
        now = time.time()

        # Clean old requests (keep only last minute)
        self.request_times = [req for req in self.request_times if now - req < 60]

        # Check per-second limit
        recent_requests = [req for req in self.request_times if now - req < 1]
        if len(recent_requests) >= self.rate_limits['requests_per_second']:
            return False

        # Check per-minute limit
        if len(self.request_times) >= self.rate_limits['requests_per_minute']:
            return False

        return True

    def record_request(self):
        """Record a successful request."""
        self.request_times.append(time.time())

    def record_error(self):
        """Record an error."""
        self.consecutive_errors += 1

    def record_success(self):
        """Record a successful request and reduce error count."""
        self.consecutive_errors = max(0, self.consecutive_errors - 1)

    def get_backoff_time(self, attempt: int) -> float:
        """
        Get backoff time for retry attempts.

        Args:
            attempt: Current attempt number (0-based)

        Returns:
            Backoff time in seconds
        """
        base_backoff = min(2 ** attempt, 300)  # Exponential backoff, capped at 300s
        error_multiplier = 2 ** self.consecutive_errors
        return min(base_backoff * error_multiplier, 300)

def adaptive_retry(max_retries: int = 3):
    """
    Decorator for adaptive retry with rate limiting.

    Args:
        max_retries: Maximum number of retry attempts

    Returns:
        Decorated function with retry logic
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            limiter = AdaptiveRateLimiter()
            for attempt in range(max_retries + 1):
                if not limiter.can_make_request():
                    backoff = limiter.get_backoff_time(attempt)
                    time.sleep(backoff)

                try:
                    result = func(*args, **kwargs)
                    limiter.record_success()
                    limiter.record_request()
                    return result
                except Exception as e:
                    limiter.record_error()
                    limiter.record_request()  # Record the failed attempt
                    if attempt == max_retries:
                        raise e
                    backoff = limiter.get_backoff_time(attempt)
                    time.sleep(backoff)
            return None
        return wrapper
    return decorator

def _get_adaptive_batch_size(exchange: str = 'binance', timeframe: str = '1m') -> int:
    """
    Get adaptive batch size based on exchange and timeframe.

    Args:
        exchange: Exchange name
        timeframe: Timeframe string

    Returns:
        Optimal batch size for fetching
    """
    # Base batch sizes by exchange
    base_sizes = {
        'binance': 1000,
        'kraken': 720,  # Kraken's max per request for 1m
        'coinbase': 300,
        'default': 500
    }

    base_size = base_sizes.get(exchange, base_sizes['default'])

    # Adjust for timeframe (smaller batches for shorter timeframes)
    timeframe_multipliers = {
        '1m': 1.0,
        '5m': 1.2,
        '15m': 1.5,
        '1h': 2.0,
        '4h': 3.0,
        '1d': 5.0
    }

    multiplier = timeframe_multipliers.get(timeframe, 1.0)

    return int(base_size * multiplier)

def _validate_batch_data(batch: list) -> bool:
    """
    Validate a batch of OHLCV data.

    Args:
        batch: List of OHLCV data points [timestamp, open, high, low, close, volume]

    Returns:
        True if batch is valid, False otherwise
    """
    if not batch:
        return False

    for candle in batch:
        # Check length (timestamp + OHLCV = 6 elements)
        if len(candle) != 6:
            return False

        timestamp, open_price, high, low, close, volume = candle

        # Check for None values
        if any(val is None for val in candle):
            return False

        # Check OHLC relationships
        try:
            if high < low or open_price < low or open_price > high or close < low or close > high:
                return False
        except TypeError:
            return False

        # Check for negative values
        if any(val < 0 for val in [open_price, high, low, close, volume]):
            return False

    return True

def _timeframe_to_ms(timeframe: str) -> int:
    """
    Convert timeframe string to milliseconds.

    Args:
        timeframe: Timeframe string (e.g., '1m', '1h', '1d')

    Returns:
        Timeframe in milliseconds
    """
    timeframe_map = {
        '1m': 60000,
        '2m': 120000,
        '5m': 300000,
        '15m': 900000,
        '30m': 1800000,
        '1h': 3600000,
        '4h': 14400000,
        '1d': 86400000,
        '1w': 604800000,
        '1M': 2592000000,
    }

    return timeframe_map.get(timeframe, 3600000)  # Default to 1h

def get_data_source():
    """Get or create the configured data source instance."""
    global _data_source
    if _data_source is None:
        _data_source = create_data_source()
    return _data_source

def set_data_source(data_source_name: str):
    """Set the data source by name."""
    global _data_source
    _data_source = create_data_source(data_source_name)

@cache_api_call(ttl_seconds=CACHE_TTL_LIVE)  # Cache for live data
def fetch_data(limit: int = DEFAULT_DATA_LIMIT, symbol: str = SYMBOL,
               timeframe: str = TIMEFRAME) -> pd.DataFrame:
    """
    Fetches recent price data using the configured data source.
    For continuous monitoring, fetches only recent candles for efficiency.

    Args:
        limit: Number of recent candles to fetch
        symbol: Trading symbol (e.g., 'BTC/USDT')
        timeframe: Timeframe string (e.g., '1m', '1h', '1d')

    Returns:
        DataFrame with OHLCV data
    """
    data_source = get_data_source()

    # Calculate start date for recent data
    end_date = datetime.now()
    # Estimate start date based on limit and timeframe
    timeframe_minutes = _timeframe_to_minutes(timeframe)
    start_date = end_date - timedelta(minutes=timeframe_minutes * limit)

    logger.info(f"Fetching recent data: {symbol} {timeframe} from {start_date} to {end_date}")

    try:
        df = data_source.fetch_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )

        if df.empty:
            logger.warning("No data fetched")
            return pd.DataFrame()

        # Validate data
        validate_data(df)
        return df

    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        raise

@cache_api_call(ttl_seconds=CACHE_TTL_HISTORICAL)  # Cache for historical data
def fetch_historical_data(symbol: str = SYMBOL, timeframe: str = TIMEFRAME,
                         limit: int = HISTORICAL_DATA_LIMIT, start_date: Optional[datetime] = None,
                         show_progress: bool = True) -> pd.DataFrame:
    """
    Fetches historical OHLCV data using the configured data source.
    Supports advanced pagination for large datasets with progress tracking.

    Args:
        symbol: Trading symbol (e.g., 'BTC/USDT')
        timeframe: Timeframe string (e.g., '1m', '1h', '1d')
        limit: Maximum number of candles to fetch
        start_date: Start date for historical data (datetime object)
        show_progress: Whether to show progress indicators

    Returns:
        DataFrame with OHLCV data
    """
    # Validate inputs
    if VALIDATION_AVAILABLE:
        validate_api_inputs(symbol=symbol, timeframe=timeframe, limit=limit)

    data_source = get_data_source()

    # Set default start date if not provided
    if start_date is None:
        start_date = datetime.now() - timedelta(days=DEFAULT_HISTORICAL_DAYS)

    end_date = datetime.now()

    logger.info(f"Fetching historical data: {symbol} {timeframe} from {start_date} to {end_date}")

    if show_progress:
        print(f"📊 Fetching historical data for {symbol} ({timeframe})...")

    start_time = time.time()

    try:
        df = data_source.fetch_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )

        if df.empty:
            if show_progress:
                print("⚠️  No data fetched")
            return pd.DataFrame()

        # Validate data
        validate_data(df)

        if show_progress:
            total_time = time.time() - start_time
            print(f"✅ Completed: {len(df)} candles fetched in {total_time:.1f}s")

        return df

    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        if show_progress:
            print(f"❌ Error fetching historical data: {e}")
        raise

def _timeframe_to_minutes(timeframe: str) -> int:
    """
    Convert timeframe string to minutes.

    Args:
        timeframe: Timeframe string (e.g., '1m', '1h', '1d')

    Returns:
        Timeframe in minutes
    """
    timeframe_map = {
        '1m': 1,
        '2m': 2,
        '5m': 5,
        '15m': 15,
        '30m': 30,
        '1h': 60,
        '4h': 240,
        '1d': 1440,
        '1w': 10080,
        '1M': 43200,
    }

    return timeframe_map.get(timeframe, 60)  # Default to 1h

def validate_data(df: pd.DataFrame) -> bool:
    """
    Validates OHLCV DataFrame structure and data integrity.

    Args:
        df: DataFrame to validate

    Returns:
        True if valid

    Raises:
        ValueError: If validation fails
    """
    # Check if input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    # Check if DataFrame is not empty
    if len(df) == 0:
        raise ValueError("DataFrame is empty")

    # Check for required columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Check data types and convert if possible
    for col in required_columns:
        try:
            # Try to convert to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except (ValueError, TypeError):
            raise ValueError(f"Column '{col}' contains non-numeric data")

    # Check for NaN values in critical columns
    for col in ['open', 'high', 'low', 'close']:
        if df[col].isna().any():
            raise ValueError(f"NaN values found in {col} column")

    # Check for negative prices or volumes
    try:
        if (df[['open', 'high', 'low', 'close']] < 0).any().any():
            raise ValueError("Negative values found in price columns")
        if (df['volume'] < 0).any():
            raise ValueError("Negative values found in volume column")
    except TypeError:
        raise ValueError("Cannot perform numeric comparisons on non-numeric data")

    # Check OHLC relationships
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

def fetch_data_realtime(symbol: str = SYMBOL, timeframe: str = TIMEFRAME,
                       duration: int = 60) -> pd.DataFrame:
    """
    Fetch real-time data using WebSocket if available, otherwise fall back to configured data source.

    Args:
        symbol: Trading symbol
        timeframe: Timeframe for candlestick data
        duration: Duration in seconds to collect data (for WebSocket mode)

    Returns:
        DataFrame with OHLCV data
    """
    if WEBSOCKET_DATA_ENABLED and WEBSOCKET_FETCHER_AVAILABLE and is_websocket_available():
        try:
            import asyncio
            # Use async WebSocket fetching
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            data = loop.run_until_complete(
                fetch_realtime_data(symbol, timeframe, duration)
            )
            loop.close()

            if not data.empty:
                return data
            else:
                # Fall back to configured data source
                return fetch_data(symbol=symbol, timeframe=timeframe)
        except Exception as e:
            logger.warning(f"WebSocket fetch failed, falling back to REST: {e}")
            # Fall back to configured data source
            return fetch_data(symbol=symbol, timeframe=timeframe)
    else:
        # Use configured data source
        return fetch_data(symbol=symbol, timeframe=timeframe)
