#!/usr/bin/env python3
"""
TradPal Circuit Breaker Service Client

Async client for interacting with the Circuit Breaker Service.
Provides methods for monitoring and managing circuit breakers.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import aiohttp
from contextlib import asynccontextmanager

from config.settings import CIRCUIT_BREAKER_SERVICE_URL, CIRCUIT_BREAKER_SERVICE_TIMEOUT

logger = logging.getLogger(__name__)


class CircuitBreakerServiceClient:
    """Async client for Circuit Breaker Service communication"""

    def __init__(self, base_url: Optional[str] = None, timeout: float = 30.0):
        self.base_url = base_url or CIRCUIT_BREAKER_SERVICE_URL
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None

    @asynccontextmanager
    async def _get_session(self):
        """Get or create HTTP session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        try:
            yield self._session
        finally:
            pass  # Keep session alive for reuse

    async def close(self):
        """Close the HTTP session"""
        if self._session and not self._session.closed:
            await self._session.close()

    async def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        async with self._get_session() as session:
            async with session.get(f"{self.base_url}/health") as response:
                response.raise_for_status()
                return await response.json()

    async def list_circuit_breakers(self) -> List[str]:
        """List all registered circuit breakers"""
        async with self._get_session() as session:
            async with session.get(f"{self.base_url}/circuit-breakers") as response:
                response.raise_for_status()
                data = await response.json()
                return data["circuit_breakers"]

    async def get_circuit_breaker_info(self, name: str) -> Dict[str, Any]:
        """Get detailed information about a circuit breaker"""
        async with self._get_session() as session:
            async with session.get(f"{self.base_url}/circuit-breakers/{name}") as response:
                response.raise_for_status()
                return await response.json()

    async def get_circuit_breaker_metrics(self, name: str) -> Dict[str, Any]:
        """Get metrics for a specific circuit breaker"""
        async with self._get_session() as session:
            async with session.get(f"{self.base_url}/circuit-breakers/{name}/metrics") as response:
                response.raise_for_status()
                return await response.json()

    async def reset_circuit_breaker(self, name: str) -> Dict[str, Any]:
        """Reset a circuit breaker to closed state"""
        async with self._get_session() as session:
            async with session.post(f"{self.base_url}/circuit-breakers/{name}/reset") as response:
                response.raise_for_status()
                return await response.json()

    async def reset_all_circuit_breakers(self) -> Dict[str, Any]:
        """Reset all circuit breakers"""
        async with self._get_session() as session:
            async with session.post(f"{self.base_url}/circuit-breakers/reset-all") as response:
                response.raise_for_status()
                return await response.json()

    async def get_circuit_breaker_state(self, name: str) -> Dict[str, Any]:
        """Get the current state of a circuit breaker"""
        async with self._get_session() as session:
            async with session.get(f"{self.base_url}/circuit-breakers/{name}/state") as response:
                response.raise_for_status()
                return await response.json()

    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data for all circuit breakers"""
        async with self._get_session() as session:
            async with session.get(f"{self.base_url}/dashboard") as response:
                response.raise_for_status()
                return await response.json()

    async def get_alerts(self) -> Dict[str, Any]:
        """Get alerts for circuit breakers"""
        async with self._get_session() as session:
            async with session.get(f"{self.base_url}/alerts") as response:
                response.raise_for_status()
                return await response.json()

    async def update_circuit_breaker_config(self, name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Update configuration for a circuit breaker"""
        async with self._get_session() as session:
            async with session.put(
                f"{self.base_url}/circuit-breakers/{name}/config",
                json=config
            ) as response:
                response.raise_for_status()
                return await response.json()

    async def is_circuit_breaker_open(self, name: str) -> bool:
        """Check if a circuit breaker is open"""
        state_info = await self.get_circuit_breaker_state(name)
        return state_info["is_open"]

    async def is_circuit_breaker_closed(self, name: str) -> bool:
        """Check if a circuit breaker is closed"""
        state_info = await self.get_circuit_breaker_state(name)
        return state_info["is_closed"]

    async def is_circuit_breaker_half_open(self, name: str) -> bool:
        """Check if a circuit breaker is half-open"""
        state_info = await self.get_circuit_breaker_state(name)
        return state_info["is_half_open"]


# Global client instance
circuit_breaker_client = CircuitBreakerServiceClient()


async def authenticate():
    """Authenticate with the circuit breaker service"""
    try:
        await circuit_breaker_client.health_check()
        logger.info("Circuit Breaker Service authentication successful")
        return True
    except Exception as e:
        logger.error(f"Circuit Breaker Service authentication failed: {e}")
        return False


async def get_circuit_breaker_status(name: str) -> Dict[str, Any]:
    """Convenience function to get circuit breaker status"""
    return await circuit_breaker_client.get_circuit_breaker_info(name)


async def reset_circuit_breaker_if_needed(name: str, max_failures: int = 5) -> bool:
    """Reset circuit breaker if it has too many failures"""
    try:
        metrics = await circuit_breaker_client.get_circuit_breaker_metrics(name)
        if metrics["failures_total"] >= max_failures:
            await circuit_breaker_client.reset_circuit_breaker(name)
            logger.info(f"Reset circuit breaker {name} due to {metrics['failures_total']} failures")
            return True
    except Exception as e:
        logger.error(f"Failed to check/reset circuit breaker {name}: {e}")
    return False


async def monitor_circuit_breakers() -> Dict[str, Any]:
    """Monitor all circuit breakers and return summary"""
    try:
        return await circuit_breaker_client.get_dashboard_data()
    except Exception as e:
        logger.error(f"Failed to monitor circuit breakers: {e}")
        return {"error": str(e)}


async def check_service_health() -> bool:
    """Check if circuit breaker service is healthy"""
    try:
        health = await circuit_breaker_client.health_check()
        return health["status"] == "healthy"
    except Exception:
        return False