#!/usr/bin/env python3
"""
Discovery Service Client

Async client for communicating with the Discovery Service API.
Provides genetic algorithm optimization for trading indicators.
Enhanced with Zero-Trust Security (mTLS + JWT).
"""

import asyncio
import aiohttp
import logging
import ssl
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from config.settings import (
    DISCOVERY_SERVICE_URL, API_KEY,
    ENABLE_MTLS, MTLS_CERT_PATH, MTLS_KEY_PATH, CA_CERT_PATH
)

logger = logging.getLogger(__name__)


class DiscoveryServiceClient:
    """Async client for Discovery Service API with Zero-Trust Security"""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or DISCOVERY_SERVICE_URL or "http://localhost:8001"
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
            from pathlib import Path
            self.ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)

            # Load client certificate and key
            if MTLS_CERT_PATH and MTLS_KEY_PATH:
                cert_path = Path(MTLS_CERT_PATH)
                key_path = Path(MTLS_KEY_PATH)

                if cert_path.exists() and key_path.exists():
                    self.ssl_context.load_cert_chain(str(cert_path), str(key_path))
                    logger.info("✅ mTLS client certificate loaded for Discovery Service")
                else:
                    logger.warning("⚠️  mTLS certificate files not found, disabling mTLS")
                    self.mtls_enabled = False

            # Load CA certificate for server verification
            if CA_CERT_PATH and Path(CA_CERT_PATH).exists():
                self.ssl_context.load_verify_locations(CA_CERT_PATH)
                self.ssl_context.verify_mode = ssl.CERT_REQUIRED

        except Exception as e:
            logger.error(f"❌ Failed to setup mTLS for Discovery Service: {e}")
            self.mtls_enabled = False

    async def authenticate(self) -> bool:
        """Authenticate with security service and get JWT token"""
        try:
            from services.security_service.client import SecurityServiceClient

            security_client = SecurityServiceClient()
            success = await security_client.authenticate("discovery_service_client")

            if success:
                self.jwt_token = "authenticated"  # Placeholder for actual token
                logger.info("✅ Discovery service client authenticated")
                return True
            else:
                logger.error("❌ Discovery service client authentication failed")
                return False

        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False

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
                timeout=aiohttp.ClientTimeout(total=300),  # Longer timeout for optimization
                connector=connector
            )

        try:
            yield self.session
        except Exception:
            if self.session:
                await self.session.close()
                self.session = None
            raise

    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None

    async def health_check(self) -> bool:
        """Check if the discovery service is healthy"""
        try:
            async with self._get_session() as session:
                async with session.get(f"{self.base_url}/api/v1/health") as response:
                    return response.status == 200
        except Exception as e:
            logger.warning(f"Discovery service health check failed: {e}")
            return False

    async def run_parameter_discovery(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        population_size: int = 50,
        generations: int = 20,
        use_walk_forward: bool = True
    ) -> Dict[str, Any]:
        """
        Start a parameter discovery optimization run.

        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1d', '1h')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            population_size: GA population size
            generations: Number of GA generations
            use_walk_forward: Use walk-forward analysis

        Returns:
            Optimization results
        """
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "start_date": start_date,
            "end_date": end_date,
            "population_size": population_size,
            "generations": generations,
            "use_walk_forward": use_walk_forward
        }

        async with self._get_session() as session:
            async with session.post(
                f"{self.base_url}/api/v1/optimization/start",
                json=payload
            ) as response:
                response.raise_for_status()
                result = await response.json()

                if result.get("success"):
                    optimization_id = result["optimization_id"]
                    logger.info(f"Optimization started: {optimization_id}")

                    # Wait for completion (polling)
                    return await self._wait_for_optimization(optimization_id)
                else:
                    raise RuntimeError(f"Optimization failed to start: {result}")

    async def _wait_for_optimization(self, optimization_id: str, poll_interval: int = 5) -> Dict[str, Any]:
        """Wait for optimization to complete by polling status"""
        while True:
            status = await self.get_optimization_status(optimization_id)

            if status["status"] == "completed":
                # Get detailed results
                results = await self.get_optimization_results(optimization_id)
                return results["results"]
            elif status["status"] == "failed":
                raise RuntimeError(f"Optimization failed: {status.get('error_message', 'Unknown error')}")
            elif status["status"] == "cancelled":
                raise RuntimeError("Optimization was cancelled")

            await asyncio.sleep(poll_interval)

    async def get_optimization_status(self, optimization_id: str) -> Dict[str, Any]:
        """
        Get the status of a specific optimization run.

        Args:
            optimization_id: Unique optimization identifier

        Returns:
            Optimization status information
        """
        async with self._get_session() as session:
            async with session.get(
                f"{self.base_url}/api/v1/optimization/{optimization_id}/status"
            ) as response:
                response.raise_for_status()
                return await response.json()

    async def get_optimization_results(self, optimization_id: str) -> Dict[str, Any]:
        """
        Get detailed results of a completed optimization.

        Args:
            optimization_id: Unique optimization identifier

        Returns:
            Detailed optimization results
        """
        async with self._get_session() as session:
            async with session.get(
                f"{self.base_url}/api/v1/optimization/{optimization_id}/results"
            ) as response:
                response.raise_for_status()
                return await response.json()

    async def list_active_optimizations(self) -> List[Dict[str, Any]]:
        """List all currently active optimization runs"""
        async with self._get_session() as session:
            async with session.get(f"{self.base_url}/api/v1/optimization/active") as response:
                response.raise_for_status()
                result = await response.json()
                return result.get("optimizations", [])

    async def cancel_optimization(self, optimization_id: str) -> Dict[str, Any]:
        """
        Cancel an active optimization run.

        Args:
            optimization_id: Unique optimization identifier

        Returns:
            Cancellation confirmation
        """
        async with self._get_session() as session:
            async with session.delete(
                f"{self.base_url}/api/v1/optimization/{optimization_id}"
            ) as response:
                response.raise_for_status()
                return await response.json()

    async def get_indicator_combinations(self) -> List[str]:
        """Get available indicator combinations for optimization"""
        async with self._get_session() as session:
            async with session.get(f"{self.base_url}/api/v1/indicator-combinations") as response:
                response.raise_for_status()
                result = await response.json()
                return result.get("combinations", [])