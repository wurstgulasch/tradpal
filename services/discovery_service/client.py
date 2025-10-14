#!/usr/bin/env python3
"""
Discovery Service Client

Async client for communicating with the Discovery Service API.
Provides genetic algorithm optimization for trading indicators.
"""

import asyncio
import aiohttp
import logging
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from config.settings import DISCOVERY_SERVICE_URL, API_KEY

logger = logging.getLogger(__name__)


class DiscoveryServiceClient:
    """Async client for Discovery Service API"""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or DISCOVERY_SERVICE_URL or "http://localhost:8001"
        self.session: Optional[aiohttp.ClientSession] = None
        self.api_key = API_KEY

    @asynccontextmanager
    async def _get_session(self):
        """Get or create HTTP session"""
        if self.session is None:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            self.session = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=300)  # Longer timeout for optimization
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