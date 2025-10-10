"""
Caching utilities for the trading indicator system.
Provides caching for expensive operations like indicator calculations and API calls.
Supports both file-based caching and Redis for distributed setups.
"""

import functools
import hashlib
import json
import os
import pickle
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple
import pandas as pd
import logging

# Setup logging
logger = logging.getLogger(__name__)

# Try to import Redis
try:
    import redis
    from redis.connection import ConnectionPool
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.info("Redis not available. Using file-based cache only.")
    logger.info("Install with: pip install redis")

from config.settings import (
    REDIS_ENABLED, REDIS_HOST, REDIS_PORT, REDIS_DB, 
    REDIS_PASSWORD, REDIS_TTL_INDICATORS, REDIS_TTL_API,
    REDIS_MAX_CONNECTIONS
)


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


class RedisCache:
    """Redis-based cache with TTL support for distributed setups."""
    
    def __init__(self, host: str = REDIS_HOST, port: int = REDIS_PORT, 
                 db: int = REDIS_DB, password: Optional[str] = REDIS_PASSWORD,
                 ttl_seconds: int = 3600, max_connections: int = REDIS_MAX_CONNECTIONS):
        """
        Initialize Redis cache.
        
        Args:
            host: Redis server host
            port: Redis server port
            db: Redis database number
            password: Redis password (if required)
            ttl_seconds: Default TTL for cache entries
            max_connections: Maximum number of connections in the pool
        """
        self.ttl_seconds = ttl_seconds
        self.redis_client = None
        
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available. RedisCache will not function.")
            return
        
        if not REDIS_ENABLED:
            logger.info("Redis caching is disabled in settings.")
            return
        
        try:
            # Create connection pool
            pool = ConnectionPool(
                host=host,
                port=port,
                db=db,
                password=password,
                max_connections=max_connections,
                decode_responses=False  # We'll handle encoding/decoding ourselves
            )
            self.redis_client = redis.Redis(connection_pool=pool)
            
            # Test connection
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage in Redis."""
        if isinstance(value, pd.DataFrame):
            # For DataFrames, use pickle for efficient serialization
            return b"PDF:" + pickle.dumps(value)
        else:
            # For other types, use JSON for security
            try:
                json_str = json.dumps(value, default=str)
                return b"JSON:" + json_str.encode("utf-8")
            except (TypeError, ValueError):
                # Fallback to pickle for complex objects that JSON cannot handle
                return b"PICKLE:" + pickle.dumps(value)
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from Redis."""
        try:
            if data.startswith(b"PDF:"):
                # DataFrame serialized with pickle
                return pickle.loads(data[4:])
            elif data.startswith(b"JSON:"):
                # JSON serialized data
                json_str = data[5:].decode("utf-8")
                return json.loads(json_str)
            elif data.startswith(b"PICKLE:"):
                # Fallback pickle data
                return pickle.loads(data[7:])
            else:
                # Legacy pickle data (for backward compatibility)
                return pickle.loads(data)
        except Exception as e:
            logger.error(f"Failed to deserialize value: {e}")
            logger.error(f"Failed to deserialize value: {e}")
            return None
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from Redis cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        if not self.redis_client:
            return None
        
        try:
            data = self.redis_client.get(key)
            if data:
                return self._deserialize_value(data)
            return None
        except Exception as e:
            logger.error(f"Error getting value from Redis: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """
        Store value in Redis cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: TTL in seconds (uses default if not specified)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.redis_client:
            return False
        
        try:
            ttl = ttl_seconds if ttl_seconds is not None else self.ttl_seconds
            serialized = self._serialize_value(value)
            self.redis_client.setex(key, ttl, serialized)
            return True
        except Exception as e:
            logger.error(f"Error setting value in Redis: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete value from Redis cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        if not self.redis_client:
            return False
        
        try:
            self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Error deleting value from Redis: {e}")
            return False
    
    def clear(self, pattern: str = "*") -> bool:
        """
        Clear cache entries matching pattern.
        
        Args:
            pattern: Key pattern to match (default: all keys)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.redis_client:
            return False
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
            return True
        except Exception as e:
            logger.error(f"Error clearing Redis cache: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists, False otherwise
        """
        if not self.redis_client:
            return False
        
        try:
            return self.redis_client.exists(key) > 0
        except Exception as e:
            logger.error(f"Error checking key existence in Redis: {e}")
            return False
    
    def get_ttl(self, key: str) -> Optional[int]:
        """
        Get remaining TTL for a key.
        
        Args:
            key: Cache key
            
        Returns:
            Remaining TTL in seconds, or None if key doesn't exist
        """
        if not self.redis_client:
            return None
        
        try:
            ttl = self.redis_client.ttl(key)
            return ttl if ttl > 0 else None
        except Exception as e:
            logger.error(f"Error getting TTL from Redis: {e}")
            return None


class HybridCache:
    """
    Hybrid cache that uses Redis for distributed caching with file-based fallback.
    
    This cache tries Redis first, then falls back to file-based caching if Redis
    is unavailable. This provides the benefits of distributed caching when available
    while maintaining functionality when Redis is not configured.
    """
    
    def __init__(self, cache_dir: str = "cache", ttl_seconds: int = 3600):
        """
        Initialize hybrid cache.
        
        Args:
            cache_dir: Directory for file-based cache
            ttl_seconds: Default TTL for cache entries
        """
        self.ttl_seconds = ttl_seconds
        
        # Initialize both cache backends
        self.redis_cache = RedisCache(ttl_seconds=ttl_seconds) if REDIS_ENABLED and REDIS_AVAILABLE else None
        self.file_cache = Cache(cache_dir=cache_dir, ttl_seconds=ttl_seconds)
        
        # Determine which cache to use
        self.use_redis = (self.redis_cache and 
                         self.redis_cache.redis_client is not None and 
                         REDIS_ENABLED)
        
        if self.use_redis:
            logger.info("Using Redis cache for distributed caching")
        else:
            logger.info("Using file-based cache (Redis not available or disabled)")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (tries Redis first, then file-based)."""
        if self.use_redis:
            value = self.redis_cache.get(key)
            if value is not None:
                return value
        
        # Fallback to file-based cache
        return self.file_cache.get(key)
    
    def set(self, key: str, value: Any) -> None:
        """Store value in cache (both Redis and file-based for redundancy)."""
        if self.use_redis:
            self.redis_cache.set(key, value)
        
        # Always store in file-based cache as backup
        self.file_cache.set(key, value)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        if self.use_redis:
            self.redis_cache.clear()
        
        self.file_cache.clear()


class CacheManager(HybridCache):
    """
    CacheManager class for backward compatibility with sentiment analysis module.
    
    This is an alias for HybridCache to maintain compatibility with existing code
    that expects a CacheManager class.
    """
    pass


# Global cache instances
_indicator_cache = HybridCache(cache_dir="cache/indicators", ttl_seconds=REDIS_TTL_INDICATORS)
_api_cache = HybridCache(cache_dir="cache/api", ttl_seconds=REDIS_TTL_API)


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