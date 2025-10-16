"""
Cache Manager for data service caching functionality.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import pandas as pd

# Optional Redis import
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Manages caching for data service operations.

    Provides Redis-based caching with TTL support and fallback to in-memory cache.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379", default_ttl: int = 3600):
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.redis_client = None
        self.memory_cache: Dict[str, Tuple[Any, datetime]] = {}
        self._initialized = False

    async def initialize(self):
        """Initialize the cache manager."""
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(self.redis_url)
                self.redis_client.ping()
                logger.info("Redis cache initialized")
            except Exception as e:
                logger.warning(f"Redis initialization failed: {e}")
                self.redis_client = None

        self._initialized = True
        logger.info("Cache manager initialized")

    async def cleanup(self):
        """Cleanup cache resources."""
        if self.redis_client:
            try:
                self.redis_client.close()
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}")

        self.memory_cache.clear()
        self._initialized = False
        logger.info("Cache manager cleaned up")

    async def cache_data(self, key: str, data: Any, ttl_seconds: Optional[int] = None) -> bool:
        """
        Cache data with optional TTL.

        Args:
            key: Cache key
            data: Data to cache (DataFrame, dict, etc.)
            ttl_seconds: Time to live in seconds

        Returns:
            Success status
        """
        if not self._initialized:
            await self.initialize()

        ttl = ttl_seconds or self.default_ttl
        expiration = datetime.now() + timedelta(seconds=ttl)

        try:
            if self.redis_client:
                # Serialize data for Redis
                if isinstance(data, pd.DataFrame):
                    cache_data = {
                        'type': 'dataframe',
                        'data': data.to_json(orient='index', date_format='iso')
                    }
                else:
                    cache_data = {
                        'type': 'json',
                        'data': json.dumps(data, default=str)
                    }

                self.redis_client.setex(key, ttl, json.dumps(cache_data))
                logger.debug(f"Cached data in Redis: {key}")
                return True

            else:
                # Use memory cache
                self.memory_cache[key] = (data, expiration)
                logger.debug(f"Cached data in memory: {key}")
                return True

        except Exception as e:
            logger.error(f"Cache storage failed for key {key}: {e}")
            return False

    async def get_cached_data(self, key: str) -> Optional[Any]:
        """
        Retrieve cached data.

        Args:
            key: Cache key

        Returns:
            Cached data or None if not found/expired
        """
        if not self._initialized:
            await self.initialize()

        try:
            if self.redis_client:
                # Get from Redis
                cached_data = self.redis_client.get(key)
                if not cached_data:
                    return None

                cache_data = json.loads(cached_data)

                if cache_data['type'] == 'dataframe':
                    df = pd.read_json(cache_data['data'], orient='index')
                    df.index = pd.to_datetime(df.index)
                    return df
                else:
                    return json.loads(cache_data['data'])

            else:
                # Get from memory cache
                if key not in self.memory_cache:
                    return None

                data, expiration = self.memory_cache[key]
                if datetime.now() > expiration:
                    # Expired, remove from cache
                    del self.memory_cache[key]
                    return None

                return data

        except Exception as e:
            logger.error(f"Cache retrieval failed for key {key}: {e}")
            return None

    async def invalidate_cache(self, key_pattern: str = "*") -> int:
        """
        Invalidate cache entries matching pattern.

        Args:
            key_pattern: Pattern to match (Redis keys command style)

        Returns:
            Number of entries invalidated
        """
        if not self._initialized:
            await self.initialize()

        try:
            if self.redis_client:
                # Find keys matching pattern
                keys = self.redis_client.keys(key_pattern)
                if keys:
                    deleted = self.redis_client.delete(*keys)
                    logger.info(f"Invalidated {deleted} Redis cache entries")
                    return deleted
                return 0

            else:
                # Clear memory cache
                cleared = len(self.memory_cache)
                self.memory_cache.clear()
                logger.info(f"Cleared {cleared} memory cache entries")
                return cleared

        except Exception as e:
            logger.error(f"Cache invalidation failed: {e}")
            return 0

    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Cache statistics
        """
        if not self._initialized:
            await self.initialize()

        stats = {
            'cache_type': 'redis' if self.redis_client else 'memory',
            'initialized': self._initialized
        }

        try:
            if self.redis_client:
                info = self.redis_client.info()
                stats.update({
                    'total_keys': info.get('db0', {}).get('keys', 0),
                    'used_memory': info.get('used_memory_human', 'unknown'),
                    'connected_clients': info.get('connected_clients', 0)
                })
            else:
                stats.update({
                    'total_keys': len(self.memory_cache),
                    'used_memory': 'in_memory'
                })

        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            stats['error'] = str(e)

        return stats

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform cache health check.

        Returns:
            Health check result
        """
        health = {
            'component': 'cache_manager',
            'status': 'healthy',
            'timestamp': datetime.now().isoformat()
        }

        if not self._initialized:
            health['status'] = 'not_initialized'
            return health

        try:
            if self.redis_client:
                self.redis_client.ping()
                health['redis_connection'] = 'ok'
            else:
                health['cache_type'] = 'memory_only'

            # Test basic cache operation
            test_key = f"health_check_{datetime.now().timestamp()}"
            await self.cache_data(test_key, {"test": "data"}, ttl_seconds=10)
            retrieved = await self.get_cached_data(test_key)

            if retrieved and retrieved.get('test') == 'data':
                health['cache_operation'] = 'ok'
            else:
                health['cache_operation'] = 'failed'
                health['status'] = 'degraded'

        except Exception as e:
            health['status'] = 'unhealthy'
            health['error'] = str(e)

        return health