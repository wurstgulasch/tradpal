#!/usr/bin/env python3
"""
Data Service Client - Client for interacting with the Data Service.

Provides methods to:
- Fetch real-time and historical data
- Manage data caching
- Monitor data quality
- Handle service health checks
"""

import asyncio
import aiohttp
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json

from config.settings import DATA_SERVICE_URL, API_KEY, API_SECRET


class DataServiceClient:
    """Client for the Data Service microservice"""

    def __init__(self, base_url: str = DATA_SERVICE_URL):
        self.base_url = base_url.rstrip('/')
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(__name__)

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def initialize(self) -> None:
        """Initialize the client"""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                headers={
                    'X-API-Key': API_KEY,
                    'Content-Type': 'application/json'
                }
            )

    async def close(self) -> None:
        """Close the client"""
        if self.session:
            await self.session.close()
            self.session = None

    async def health_check(self) -> bool:
        """Check if the service is healthy"""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                return response.status == 200
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False

    async def fetch_realtime_data(self, symbol: str, timeframe: str, exchange: str = "binance") -> Dict[str, Any]:
        """Fetch real-time market data"""
        try:
            params = {
                'symbol': symbol,
                'timeframe': timeframe,
                'exchange': exchange,
                'realtime': True
            }

            async with self.session.get(f"{self.base_url}/data/fetch", params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error = await response.text()
                    raise Exception(f"Data fetch failed: {error}")

        except Exception as e:
            self.logger.error(f"Failed to fetch realtime data: {e}")
            raise

    async def fetch_historical_data(self, symbol: str, timeframe: str, start_date: str, end_date: Optional[str] = None,
                                   exchange: str = "binance") -> Dict[str, Any]:
        """Fetch historical market data"""
        try:
            params = {
                'symbol': symbol,
                'timeframe': timeframe,
                'exchange': exchange,
                'start_date': start_date
            }

            if end_date:
                params['end_date'] = end_date

            async with self.session.get(f"{self.base_url}/data/fetch", params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error = await response.text()
                    raise Exception(f"Historical data fetch failed: {error}")

        except Exception as e:
            self.logger.error(f"Failed to fetch historical data: {e}")
            raise

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            async with self.session.get(f"{self.base_url}/cache/stats") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {}
        except Exception as e:
            self.logger.error(f"Failed to get cache stats: {e}")
            return {}

    async def clear_cache(self, pattern: Optional[str] = None) -> bool:
        """Clear cache entries"""
        try:
            params = {}
            if pattern:
                params['pattern'] = pattern

            async with self.session.delete(f"{self.base_url}/cache", params=params) as response:
                return response.status == 200

        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
            return False

    async def get_data_quality_report(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Get data quality report"""
        try:
            params = {'symbol': symbol, 'timeframe': timeframe}

            async with self.session.get(f"{self.base_url}/quality/report", params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {}

        except Exception as e:
            self.logger.error(f"Failed to get quality report: {e}")
            return {}

    async def validate_data_integrity(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data integrity"""
        try:
            async with self.session.post(f"{self.base_url}/validate", json=data) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error = await response.text()
                    raise Exception(f"Data validation failed: {error}")

        except Exception as e:
            self.logger.error(f"Failed to validate data: {e}")
            raise