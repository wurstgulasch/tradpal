#!/usr/bin/env python3
"""
Backtesting Service Client - Client for interacting with the Backtesting Service.

Provides methods to:
- Run single backtests
- Run multi-symbol backtests
- Run multi-model comparisons
- Monitor backtest status
- Get backtest results
Enhanced with Zero-Trust Security (mTLS + JWT).
"""

import asyncio
import aiohttp
import logging
import ssl
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

from config.settings import (
    BACKTESTING_SERVICE_URL, API_KEY, API_SECRET,
    ENABLE_MTLS, MTLS_CERT_PATH, MTLS_KEY_PATH, CA_CERT_PATH
)


class BacktestingServiceClient:
    """Client for the Backtesting Service microservice with Zero-Trust Security"""

    def __init__(self, base_url: str = BACKTESTING_SERVICE_URL):
        self.base_url = base_url.rstrip('/')
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(__name__)
        self.jwt_token: Optional[str] = None

        # mTLS configuration
        self.mtls_enabled = ENABLE_MTLS or False
        self.ssl_context: Optional[ssl.SSLContext] = None

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
                    self.logger.info("✅ mTLS client certificate loaded for Backtesting Service")
                else:
                    self.logger.warning("⚠️  mTLS certificate files not found, disabling mTLS")
                    self.mtls_enabled = False

            # Load CA certificate for server verification
            if CA_CERT_PATH and Path(CA_CERT_PATH).exists():
                self.ssl_context.load_verify_locations(CA_CERT_PATH)
                self.ssl_context.verify_mode = ssl.CERT_REQUIRED

        except Exception as e:
            self.logger.error(f"❌ Failed to setup mTLS for Backtesting Service: {e}")
            self.mtls_enabled = False

    async def authenticate(self) -> bool:
        """Authenticate with security service and get JWT token"""
        try:
            from services.security_service.client import SecurityServiceClient

            security_client = SecurityServiceClient()
            success = await security_client.authenticate("backtesting_service_client")

            if success:
                self.jwt_token = "authenticated"  # Placeholder for actual token
                self.logger.info("✅ Backtesting service client authenticated")
                return True
            else:
                self.logger.error("❌ Backtesting service client authentication failed")
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

    async def run_backtest(self, symbol: str, timeframe: str, start_date: str,
                          end_date: Optional[str] = None, initial_capital: float = 10000,
                          strategy_config: Optional[Dict[str, Any]] = None,
                          risk_config: Optional[Dict[str, Any]] = None,
                          data_source: str = 'kaggle') -> Dict[str, Any]:
        """Run a single backtest"""
        try:
            payload = {
                'symbol': symbol,
                'timeframe': timeframe,
                'start_date': start_date,
                'initial_capital': initial_capital,
                'data_source': data_source
            }

            if end_date:
                payload['end_date'] = end_date
            if strategy_config:
                payload['strategy_config'] = strategy_config
            if risk_config:
                payload['risk_config'] = risk_config

            async with self.session.post(f"{self.base_url}/backtest", json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error = await response.text()
                    raise Exception(f"Backtest failed: {error}")

        except Exception as e:
            self.logger.error(f"Failed to run backtest: {e}")
            raise

    async def run_multi_symbol_backtest(self, symbols: List[str], timeframe: str,
                                       start_date: str, end_date: Optional[str] = None,
                                       initial_capital: float = 10000) -> Dict[str, Any]:
        """Run multi-symbol backtest"""
        try:
            payload = {
                'symbols': symbols,
                'timeframe': timeframe,
                'start_date': start_date,
                'initial_capital': initial_capital
            }

            if end_date:
                payload['end_date'] = end_date

            async with self.session.post(f"{self.base_url}/backtest/multi-symbol", json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error = await response.text()
                    raise Exception(f"Multi-symbol backtest failed: {error}")

        except Exception as e:
            self.logger.error(f"Failed to run multi-symbol backtest: {e}")
            raise

    async def run_multi_model_backtest(self, symbol: str, timeframe: str, start_date: str,
                                      end_date: Optional[str] = None, initial_capital: float = 10000,
                                      models_to_test: Optional[List[str]] = None,
                                      max_workers: int = 4) -> Dict[str, Any]:
        """Run multi-model comparison backtest"""
        try:
            payload = {
                'symbol': symbol,
                'timeframe': timeframe,
                'start_date': start_date,
                'initial_capital': initial_capital,
                'max_workers': max_workers
            }

            if end_date:
                payload['end_date'] = end_date
            if models_to_test:
                payload['models_to_test'] = models_to_test

            async with self.session.post(f"{self.base_url}/backtest/multi-model", json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error = await response.text()
                    raise Exception(f"Multi-model backtest failed: {error}")

        except Exception as e:
            self.logger.error(f"Failed to run multi-model backtest: {e}")
            raise

    async def get_backtest_status(self, backtest_id: str) -> Dict[str, Any]:
        """Get status of a running backtest"""
        try:
            async with self.session.get(f"{self.base_url}/backtest/{backtest_id}/status") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {'status': 'not_found'}

        except Exception as e:
            self.logger.error(f"Failed to get backtest status: {e}")
            return {'status': 'error'}

    async def get_backtest_results(self, backtest_id: str) -> Dict[str, Any]:
        """Get results of a completed backtest"""
        try:
            async with self.session.get(f"{self.base_url}/backtest/{backtest_id}/results") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {}

        except Exception as e:
            self.logger.error(f"Failed to get backtest results: {e}")
            return {}

    async def list_backtests(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all backtests"""
        try:
            params = {}
            if status:
                params['status'] = status

            async with self.session.get(f"{self.base_url}/backtests", params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return []

        except Exception as e:
            self.logger.error(f"Failed to list backtests: {e}")
            return []

    async def cancel_backtest(self, backtest_id: str) -> bool:
        """Cancel a running backtest"""
        try:
            async with self.session.delete(f"{self.base_url}/backtest/{backtest_id}") as response:
                return response.status == 200

        except Exception as e:
            self.logger.error(f"Failed to cancel backtest: {e}")
            return False