#!/usr/bin/env python3
"""
Core Service Client

Async client for communicating with the Core Service API.
Provides signal generation, indicator calculation, and strategy execution.
Enhanced with Zero-Trust Security (mTLS + JWT).
"""

import asyncio
import aiohttp
import logging
import ssl
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
from pathlib import Path

from config.settings import (
    CORE_SERVICE_URL, API_KEY, ENABLE_MTLS,
    MTLS_CERT_PATH, MTLS_KEY_PATH, CA_CERT_PATH
)

logger = logging.getLogger(__name__)


class CoreServiceClient:
    """Async client for Core Service API with Zero-Trust Security"""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or CORE_SERVICE_URL or "http://localhost:8002"
        self.session: Optional[aiohttp.ClientSession] = None
        self.api_key = API_KEY
        self.jwt_token: Optional[str] = None

        # mTLS configuration
        self.mtls_enabled = ENABLE_MTLS or False
        self.ssl_context: Optional[ssl.SSLContext] = None

        if self.mtls_enabled:
            self._setup_mtls()

    def _setup_mtls(self):
        """Setup mutual TLS configuration"""
        try:
            self.ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)

            # Load client certificate and key
            if MTLS_CERT_PATH and MTLS_KEY_PATH:
                cert_path = Path(MTLS_CERT_PATH)
                key_path = Path(MTLS_KEY_PATH)

                if cert_path.exists() and key_path.exists():
                    self.ssl_context.load_cert_chain(str(cert_path), str(key_path))
                    logger.info("✅ mTLS client certificate loaded")
                else:
                    logger.warning("⚠️  mTLS certificate files not found, disabling mTLS")
                    self.mtls_enabled = False

            # Load CA certificate for server verification
            if CA_CERT_PATH and Path(CA_CERT_PATH).exists():
                self.ssl_context.load_verify_locations(CA_CERT_PATH)
                self.ssl_context.verify_mode = ssl.CERT_REQUIRED
                logger.info("✅ mTLS CA certificate loaded")

        except Exception as e:
            logger.error(f"❌ Failed to setup mTLS: {e}")
            self.mtls_enabled = False

    @asynccontextmanager
    async def _get_session(self):
        """Get or create HTTP session with security"""
        if self.session is None:
            headers = {}
            if self.api_key:
                headers["X-API-Key"] = self.api_key
            if self.jwt_token:
                headers["Authorization"] = f"Bearer {self.jwt_token}"

            connector = None
            if self.mtls_enabled and self.ssl_context:
                connector = aiohttp.TCPConnector(ssl=self.ssl_context)

            self.session = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
                connector=connector
            )

        try:
            yield self.session
        except Exception:
            if self.session:
                await self.session.close()
                self.session = None
            raise

    async def authenticate(self) -> bool:
        """Authenticate with security service and get JWT token"""
        try:
            from services.security_service.client import SecurityServiceClient

            security_client = SecurityServiceClient()
            success = await security_client.authenticate("core_service_client")

            if success:
                # Get the token from security client
                # Note: In production, this should be handled more securely
                self.jwt_token = "authenticated"  # Placeholder
                logger.info("✅ Core service client authenticated")
                return True
            else:
                logger.error("❌ Core service client authentication failed")
                return False

        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False

    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None

    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None

    async def health_check(self) -> bool:
        """Check if the core service is healthy"""
        try:
            async with self._get_session() as session:
                async with session.get(f"{self.base_url}/health") as response:
                    return response.status == 200
        except Exception as e:
            logger.warning(f"Core service health check failed: {e}")
            return False

    async def generate_signals(
        self,
        symbol: str,
        timeframe: str,
        data: List[Dict[str, Any]],
        strategy_config: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate trading signals.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            data: OHLCV data as list of dicts
            strategy_config: Strategy configuration

        Returns:
            List of trading signals
        """
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "data": data
        }

        if strategy_config:
            payload["strategy_config"] = strategy_config

        async with self._get_session() as session:
            async with session.post(
                f"{self.base_url}/signals/generate",
                json=payload
            ) as response:
                response.raise_for_status()
                result = await response.json()

                if result.get("success"):
                    return result.get("signals", [])
                else:
                    raise RuntimeError(f"Signal generation failed: {result}")

    async def calculate_indicators(
        self,
        symbol: str,
        timeframe: str,
        data: List[Dict[str, Any]],
        indicators: List[str]
    ) -> Dict[str, Any]:
        """
        Calculate technical indicators.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            data: OHLCV data as list of dicts
            indicators: List of indicators to calculate

        Returns:
            Calculated indicators
        """
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "data": data,
            "indicators": indicators
        }

        async with self._get_session() as session:
            async with session.post(
                f"{self.base_url}/indicators/calculate",
                json=payload
            ) as response:
                response.raise_for_status()
                result = await response.json()

                if result.get("success"):
                    return result.get("indicators", {})
                else:
                    raise RuntimeError(f"Indicator calculation failed: {result}")

    async def execute_strategy(
        self,
        symbol: str,
        timeframe: str,
        signal: Dict[str, Any],
        capital: float,
        risk_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute trading strategy.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            signal: Trading signal
            capital: Available capital
            risk_config: Risk management configuration

        Returns:
            Strategy execution result
        """
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "signal": signal,
            "capital": capital,
            "risk_config": risk_config
        }

        async with self._get_session() as session:
            async with session.post(
                f"{self.base_url}/strategy/execute",
                json=payload
            ) as response:
                response.raise_for_status()
                result = await response.json()

                if result.get("success"):
                    return result.get("execution", {})
                else:
                    raise RuntimeError(f"Strategy execution failed: {result}")

    async def get_market_analysis(self, symbol: str, timeframe: str = "1h") -> Dict[str, Any]:
        """
        Get market analysis for a symbol.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Market analysis data
        """
        async with self._get_session() as session:
            async with session.get(
                f"{self.base_url}/analysis/market/{symbol}?timeframe={timeframe}"
            ) as response:
                response.raise_for_status()
                return await response.json()

    async def list_strategies(self) -> List[str]:
        """List available trading strategies"""
        async with self._get_session() as session:
            async with session.get(f"{self.base_url}/strategies") as response:
                response.raise_for_status()
                result = await response.json()
                return result.get("strategies", [])

    async def list_indicators(self) -> List[str]:
        """List available technical indicators"""
        async with self._get_session() as session:
            async with session.get(f"{self.base_url}/indicators") as response:
                response.raise_for_status()
                result = await response.json()
                return result.get("indicators", [])

    async def get_performance_metrics(self, symbol: str) -> Dict[str, Any]:
        """
        Get performance metrics for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Performance metrics
        """
        async with self._get_session() as session:
            async with session.get(f"{self.base_url}/performance/{symbol}") as response:
                response.raise_for_status()
                return await response.json()