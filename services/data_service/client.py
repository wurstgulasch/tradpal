#!/usr/bin/env python3
"""
Data Service Client - Client for interacting with the Data Service.

Provides methods to:
- Fetch real-time and historical data
- Manage data caching
- Monitor data quality
- Handle service health checks
Enhanced with Zero-Trust Security (mTLS + JWT).
"""

import asyncio
import aiohttp
import logging
import ssl
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json

from config.service_settings import (
    DATA_SERVICE_URL, ENABLE_MTLS, MTLS_CERT_PATH, MTLS_KEY_PATH, CA_CERT_PATH
)
from config.core_settings import API_KEY, API_SECRET

# Circuit Breaker imports
from services.infrastructure_service.circuit_breaker_service import (
    get_http_circuit_breaker,
    CircuitBreakerConfig,
    SERVICE_CIRCUIT_BREAKER_CONFIGS
)


class DataService:
    """Client for the Data Service microservice with Zero-Trust Security"""

    def __init__(self, base_url: str = DATA_SERVICE_URL):
        self.base_url = base_url.rstrip('/')
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(__name__)
        self.jwt_token: Optional[str] = None

        # mTLS configuration
        self.mtls_enabled = ENABLE_MTLS or False
        self.ssl_context: Optional[ssl.SSLContext] = None

        # Circuit Breaker configuration
        self.circuit_breaker_config = SERVICE_CIRCUIT_BREAKER_CONFIGS.get(
            'data_service',
            CircuitBreakerConfig()
        )
        self.http_breaker = None

        if self.mtls_enabled:
            self._setup_mtls()

    def _setup_mtls(self):
        """Setup mutual TLS configuration"""
        try:
            from pathlib import Path
            self.ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)

            # Load client certificate and key
            if MTLS_CERT_PATH and MTLS_KEY_PATH:
                cert_path = Path(MTLS_CERT_PATH)
                key_path = Path(MTLS_KEY_PATH)

                if cert_path.exists() and key_path.exists():
                    self.ssl_context.load_cert_chain(str(cert_path), str(key_path))
                    self.logger.info("✅ mTLS client certificate loaded for Data Service")
                else:
                    self.logger.warning("⚠️  mTLS certificate files not found, disabling mTLS")
                    self.mtls_enabled = False

            # Load CA certificate for server verification
            if CA_CERT_PATH and Path(CA_CERT_PATH).exists():
                self.ssl_context.load_verify_locations(CA_CERT_PATH)
                self.ssl_context.verify_mode = ssl.CERT_REQUIRED

        except Exception as e:
            self.logger.error(f"❌ Failed to setup mTLS for Data Service: {e}")
            self.mtls_enabled = False

    async def authenticate(self) -> bool:
        """Authenticate with security service and get JWT token"""
        try:
            from services.infrastructure_service.security_service.client import SecurityServiceClient

            security_client = SecurityServiceClient()
            success = await security_client.authenticate("data_service_client")

            if success:
                self.jwt_token = "authenticated"  # Placeholder for actual token
                self.logger.info("✅ Data service client authenticated")
                return True
            else:
                self.logger.error("❌ Data service client authentication failed")
                return False

        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            return False

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def initialize(self) -> None:
        """Initialize the client"""
        if self.session is None:
            headers = {
                'X-API-Key': API_KEY,
                'Content-Type': 'application/json'
            }
            if self.jwt_token:
                headers['Authorization'] = f'Bearer {self.jwt_token}'

            connector = None
            if self.mtls_enabled and self.ssl_context:
                connector = aiohttp.TCPConnector(ssl=self.ssl_context)

            self.session = aiohttp.ClientSession(
                headers=headers,
                connector=connector
            )

            # Initialize Circuit Breaker
            self.http_breaker = await get_http_circuit_breaker(
                "data_service",
                self.circuit_breaker_config,
                self.session
            )

    async def close(self) -> None:
        """Close the client"""
        if self.session:
            await self.session.close()
            self.session = None

    async def health_check(self) -> bool:
        """Check if the service is healthy"""
        try:
            if self.http_breaker:
                response = await self.http_breaker.get(f"{self.base_url}/health")
                return response.status == 200
            else:
                # Fallback without circuit breaker
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

            if self.http_breaker:
                response = await self.http_breaker.get(f"{self.base_url}/data/fetch", params=params)
                if response.status == 200:
                    return await response.json()
                else:
                    error = await response.text()
                    raise Exception(f"Data fetch failed: {error}")
            else:
                # Fallback without circuit breaker
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
                                   exchange: str = "binance", data_source: str = "kaggle") -> Dict[str, Any]:
        """Fetch historical market data"""
        try:
            params = {
                'symbol': symbol,
                'timeframe': timeframe,
                'exchange': exchange,
                'start_date': start_date,
                'data_source': data_source
            }

            if end_date:
                params['end_date'] = end_date

            if self.http_breaker:
                response = await self.http_breaker.get(f"{self.base_url}/data/fetch", params=params)
                if response.status == 200:
                    return await response.json()
                else:
                    error = await response.text()
                    raise Exception(f"Historical data fetch failed: {error}")
            else:
                # Fallback without circuit breaker
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
            if self.http_breaker:
                response = await self.http_breaker.get(f"{self.base_url}/cache/stats")
                if response.status == 200:
                    return await response.json()
                else:
                    return {}
            else:
                # Fallback without circuit breaker
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

            if self.http_breaker:
                response = await self.http_breaker.delete(f"{self.base_url}/cache", params=params)
                return response.status == 200
            else:
                # Fallback without circuit breaker
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

    # Data Mesh Methods

    async def register_data_product(self, name: str, domain: str, description: str,
                                  schema: Dict[str, Any], owners: List[str]) -> Dict[str, Any]:
        """
        Register a new data product in the Data Mesh.

        Args:
            name: Data product name
            domain: Data domain
            description: Product description
            schema: Data schema
            owners: List of owners

        Returns:
            Registration result
        """
        try:
            payload = {
                "name": name,
                "domain": domain,
                "description": description,
                "schema": schema,
                "owners": owners
            }

            if self.http_breaker:
                response = await self.http_breaker.post(
                    f"{self.base_url}/data-mesh/products/register",
                    json=payload
                )
                if response.status == 200:
                    return await response.json()
                else:
                    error = await response.text()
                    raise Exception(f"Data product registration failed: {error}")
            else:
                # Fallback without circuit breaker
                async with self.session.post(
                    f"{self.base_url}/data-mesh/products/register",
                    json=payload
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error = await response.text()
                        raise Exception(f"Data product registration failed: {error}")

        except Exception as e:
            self.logger.error(f"Data product registration failed: {e}")
            raise

    async def store_market_data(self, symbol: str, timeframe: str, data: Dict[str, Any],
                               metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Store market data in the Data Mesh.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            data: OHLCV data as dict
            metadata: Additional metadata

        Returns:
            Storage result
        """
        try:
            payload = {
                "symbol": symbol,
                "timeframe": timeframe,
                "data": data,
                "metadata": metadata or {}
            }

            async with self.session.post(
                f"{self.base_url}/data-mesh/market-data/store",
                json=payload
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error = await response.text()
                    raise Exception(f"Market data storage failed: {error}")

        except Exception as e:
            self.logger.error(f"Market data storage failed: {e}")
            raise

    async def retrieve_market_data(self, symbol: str, timeframe: str,
                                  start_date: Optional[str] = None,
                                  end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve market data from the Data Mesh.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            start_date: Start date filter
            end_date: End date filter

        Returns:
            Retrieved data
        """
        try:
            params = {"symbol": symbol, "timeframe": timeframe}
            if start_date:
                params["start_date"] = start_date
            if end_date:
                params["end_date"] = end_date

            async with self.session.get(
                f"{self.base_url}/data-mesh/market-data/retrieve",
                params=params
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error = await response.text()
                    raise Exception(f"Market data retrieval failed: {error}")

        except Exception as e:
            self.logger.error(f"Market data retrieval failed: {e}")
            raise

    async def store_ml_features(self, feature_set_name: str, features: Dict[str, Any],
                               metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store ML features in the Feature Store.

        Args:
            feature_set_name: Name of the feature set
            features: Features data as dict
            metadata: Feature set metadata

        Returns:
            Storage result
        """
        try:
            payload = {
                "feature_set_name": feature_set_name,
                "features": features,
                "metadata": metadata
            }

            async with self.session.post(
                f"{self.base_url}/data-mesh/features/store",
                json=payload
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error = await response.text()
                    raise Exception(f"ML features storage failed: {error}")

        except Exception as e:
            self.logger.error(f"ML features storage failed: {e}")
            raise

    async def retrieve_ml_features(self, feature_set_name: str,
                                  feature_names: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve ML features from the Feature Store.

        Args:
            feature_set_name: Name of the feature set
            feature_names: Comma-separated feature names (optional)

        Returns:
            Retrieved features
        """
        try:
            params = {"feature_set_name": feature_set_name}
            if feature_names:
                params["feature_names"] = feature_names

            async with self.session.get(
                f"{self.base_url}/data-mesh/features/retrieve",
                params=params
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error = await response.text()
                    raise Exception(f"ML features retrieval failed: {error}")

        except Exception as e:
            self.logger.error(f"ML features retrieval failed: {e}")
            raise

    async def archive_historical_data(self, symbol: str, timeframe: str,
                                     start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Archive historical data to Data Lake.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date

        Returns:
            Archival result
        """
        try:
            payload = {
                "symbol": symbol,
                "timeframe": timeframe,
                "start_date": start_date,
                "end_date": end_date
            }

            async with self.session.post(
                f"{self.base_url}/data-mesh/archive",
                json=payload
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error = await response.text()
                    raise Exception(f"Data archival failed: {error}")

        except Exception as e:
            self.logger.error(f"Data archival failed: {e}")
            raise

    async def get_data_mesh_status(self) -> Dict[str, Any]:
        """
        Get comprehensive Data Mesh status.

        Returns:
            Data Mesh health and statistics
        """
        try:
            async with self.session.get(f"{self.base_url}/data-mesh/status") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error = await response.text()
                    raise Exception(f"Data Mesh status check failed: {error}")

        except Exception as e:
            self.logger.error(f"Data Mesh status check failed: {e}")
            raise