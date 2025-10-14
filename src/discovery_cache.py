"""
Enhanced Caching System for Discovery Mode

Provides intelligent caching specifically optimized for genetic algorithm optimization
with shared data access and reduced API calls.
"""

import functools
import hashlib
import json
import os
import pickle
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple, Callable
import pandas as pd
import logging

from config.settings import (
    REDIS_ENABLED, REDIS_HOST, REDIS_PORT, REDIS_DB,
    REDIS_PASSWORD, REDIS_TTL_INDICATORS, REDIS_TTL_API,
    REDIS_MAX_CONNECTIONS
)

logger = logging.getLogger(__name__)

class DiscoveryDataCache:
    """
    Specialized cache for Discovery mode that shares data across multiple backtests.

    This cache is designed to minimize API calls during GA optimization by:
    1. Caching complete datasets for specific time ranges
    2. Sharing data across multiple backtest evaluations
    3. Using intelligent cache keys based on data parameters
    """

    def __init__(self, cache_dir: str = "cache/discovery", ttl_hours: int = 24):
        """
        Initialize discovery data cache.

        Args:
            cache_dir: Directory for cache files
            ttl_hours: Cache TTL in hours (longer for discovery mode)
        """
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_hours * 3600
        os.makedirs(cache_dir, exist_ok=True)
        self._memory_cache = {}  # In-memory cache for current session

    def _get_cache_key(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> str:
        """
        Generate a unique cache key for the data request.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            start_date: Start date string
            end_date: End date string

        Returns:
            Unique cache key
        """
        key_data = f"{symbol}_{timeframe}_{start_date}_{end_date}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> str:
        """Generate cache file path for a key."""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")

    def _is_expired(self, cache_path: str) -> bool:
        """Check if cache file is expired."""
        if not os.path.exists(cache_path):
            return True

        file_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        return datetime.now() - file_time > timedelta(seconds=self.ttl_seconds)

    def get(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Get cached data if available and not expired.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            start_date: Start date string
            end_date: End date string

        Returns:
            Cached DataFrame or None
        """
        cache_key = self._get_cache_key(symbol, timeframe, start_date, end_date)

        # Check memory cache first (fastest)
        if cache_key in self._memory_cache:
            logger.debug(f"Discovery cache hit (memory): {symbol} {timeframe}")
            return self._memory_cache[cache_key]

        # Check file cache
        cache_path = self._get_cache_path(cache_key)
        if not self._is_expired(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                    # Store in memory cache for faster future access
                    self._memory_cache[cache_key] = data
                    logger.debug(f"Discovery cache hit (file): {symbol} {timeframe}")
                    return data
            except (FileNotFoundError, pickle.PickleError, EOFError) as e:
                logger.warning(f"Cache file corrupted, will refetch: {e}")

        logger.debug(f"Discovery cache miss: {symbol} {timeframe}")
        return None

    def set(self, symbol: str, timeframe: str, start_date: str, end_date: str, data: pd.DataFrame) -> None:
        """
        Cache the data.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            start_date: Start date string
            end_date: End date string
            data: DataFrame to cache
        """
        if data is None or data.empty:
            return

        cache_key = self._get_cache_key(symbol, timeframe, start_date, end_date)

        # Store in memory cache
        self._memory_cache[cache_key] = data

        # Store in file cache
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            logger.debug(f"Cached discovery data: {symbol} {timeframe} ({len(data)} rows)")
        except (OSError, pickle.PickleError) as e:
            logger.warning(f"Failed to cache data: {e}")

    def clear(self) -> None:
        """Clear all cached data."""
        self._memory_cache.clear()

        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl'):
                try:
                    os.remove(os.path.join(self.cache_dir, filename))
                except OSError:
                    pass

        logger.info("Discovery cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        # Count actual files in cache directory
        try:
            file_cache_size = len([f for f in os.listdir(self.cache_dir) if f.endswith('.pkl') and os.path.isfile(os.path.join(self.cache_dir, f))])
        except (OSError, FileNotFoundError):
            file_cache_size = 0

        memory_cache_size = len(self._memory_cache)

        return {
            'memory_cache_entries': memory_cache_size,
            'file_cache_entries': file_cache_size,
            'cache_dir': self.cache_dir,
            'ttl_hours': self.ttl_seconds / 3600
        }


# Global discovery cache instance
_discovery_cache = None

def get_discovery_cache() -> DiscoveryDataCache:
    """Get or create the global discovery cache instance."""
    global _discovery_cache
    if _discovery_cache is None:
        _discovery_cache = DiscoveryDataCache()
    return _discovery_cache

def cached_discovery_fetch(symbol: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch data with discovery-optimized caching.

    This function first checks the discovery cache, and only makes an API call
    if the data is not cached or expired.

    Args:
        symbol: Trading symbol
        timeframe: Timeframe string
        start_date: Start date string
        end_date: End date string

    Returns:
        DataFrame with OHLCV data
    """
    cache = get_discovery_cache()

    logger.info(f"Discovery cache: Checking cache for {symbol} {timeframe} {start_date} to {end_date}")

    # Try to get from cache first
    cached_data = cache.get(symbol, timeframe, start_date, end_date)
    if cached_data is not None:
        logger.info(f"Discovery cache: HIT - Using cached data ({len(cached_data)} rows)")
        return cached_data

    # Cache miss - fetch from API
    try:
        from .data_fetcher import fetch_historical_data
    except ImportError:
        from data_fetcher import fetch_historical_data

    logger.info(f"Discovery cache: MISS - Fetching fresh data: {symbol} {timeframe} {start_date} to {end_date}")

    # Calculate limit based on timeframe and date range
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    days_diff = (end_dt - start_dt).days

    # Estimate limit based on timeframe
    if timeframe == '1m':
        limit = days_diff * 24 * 60  # minutes per day
    elif timeframe == '5m':
        limit = days_diff * 24 * 12  # 5-minute bars per day
    elif timeframe == '15m':
        limit = days_diff * 24 * 4   # 15-minute bars per day
    elif timeframe == '30m':
        limit = days_diff * 24 * 2   # 30-minute bars per day
    elif timeframe == '1h':
        limit = days_diff * 24       # hours per day
    elif timeframe == '4h':
        limit = days_diff * 6        # 4-hour bars per day
    elif timeframe == '1d':
        limit = days_diff            # days
    else:
        limit = days_diff * 24       # default to hourly

    # Add some buffer
    limit = int(limit * 1.1)

    data = fetch_historical_data(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_dt,
        limit=limit,
        show_progress=False  # Don't show progress for cached fetches
    )

    # Cache the result
    if not data.empty:
        logger.info(f"Discovery cache: Storing {len(data)} rows in cache")
        cache.set(symbol, timeframe, start_date, end_date, data)
    else:
        logger.warning(f"Discovery cache: No data to cache (empty DataFrame)")

    return data

def clear_discovery_cache():
    """Clear the discovery cache."""
    cache = get_discovery_cache()
    cache.clear()

def get_discovery_cache_stats() -> Dict[str, Any]:
    """Get discovery cache statistics."""
    cache = get_discovery_cache()
    return cache.get_stats()