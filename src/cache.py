"""
Caching utilities for the trading indicator system.
Provides caching for expensive operations like indicator calculations and API calls.
"""

import functools
import hashlib
import json
import os
import pickle
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple
import pandas as pd


class Cache:
    """Simple file-based cache with TTL support."""

    def __init__(self, cache_dir: str = "cache", ttl_seconds: int = 3600):
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_seconds
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_path(self, key: str) -> str:
        """Generate cache file path for a key."""
        # Create a safe filename from the key
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{safe_key}.pkl")

    def _is_expired(self, cache_path: str) -> bool:
        """Check if cache file is expired."""
        if not os.path.exists(cache_path):
            return True

        file_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        return datetime.now() - file_time > timedelta(seconds=self.ttl_seconds)

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if it exists and is not expired."""
        cache_path = self._get_cache_path(key)

        if self._is_expired(cache_path):
            return None

        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except (FileNotFoundError, pickle.PickleError):
            return None

    def set(self, key: str, value: Any) -> None:
        """Store value in cache."""
        cache_path = self._get_cache_path(key)

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
        except (OSError, pickle.PickleError):
            # Silently fail if caching fails
            pass

    def clear(self) -> None:
        """Clear all cache files."""
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl'):
                try:
                    os.remove(os.path.join(self.cache_dir, filename))
                except OSError:
                    pass


# Global cache instance
_indicator_cache = Cache(cache_dir="cache/indicators", ttl_seconds=300)  # 5 minutes for indicators
_api_cache = Cache(cache_dir="cache/api", ttl_seconds=60)  # 1 minute for API calls


def cache_indicators(ttl_seconds: int = 300):
    """Decorator for caching indicator calculations."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key_data = {
                'func': func.__name__,
                'args': str(args),
                'kwargs': str(sorted(kwargs.items()))
            }
            key = json.dumps(key_data, sort_keys=True)

            # Try to get from cache
            cached_result = _indicator_cache.get(key)
            if cached_result is not None:
                return cached_result

            # Calculate and cache result
            result = func(*args, **kwargs)
            _indicator_cache.set(key, result)
            return result

        return wrapper
    return decorator


def cache_api_call(ttl_seconds: int = 60):
    """Decorator for caching API calls."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key_data = {
                'func': func.__name__,
                'args': str(args),
                'kwargs': str(sorted(kwargs.items()))
            }
            key = json.dumps(key_data, sort_keys=True)

            # Try to get from cache
            cached_result = _api_cache.get(key)
            if cached_result is not None:
                return cached_result

            # Make API call and cache result
            result = func(*args, **kwargs)
            _api_cache.set(key, result)
            return result

        return wrapper
    return decorator


def clear_all_caches():
    """Clear all caches."""
    _indicator_cache.clear()
    _api_cache.clear()


def get_cache_stats() -> Dict[str, int]:
    """Get cache statistics."""
    indicator_files = len([f for f in os.listdir("cache/indicators") if f.endswith('.pkl')]) if os.path.exists("cache/indicators") else 0
    api_files = len([f for f in os.listdir("cache/api") if f.endswith('.pkl')]) if os.path.exists("cache/api") else 0

    return {
        'indicator_cache_size': indicator_files,
        'api_cache_size': api_files
    }