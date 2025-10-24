"""
TradPal API Gateway Service Client
Async client for API gateway communication
"""

import asyncio
import aiohttp
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class APIGatewayClient:
    """
    Async client for API Gateway Service communication.
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def initialize(self):
        """Initialize the client session."""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        logger.info(f"API Gateway client initialized with base URL: {self.base_url}")

    async def close(self):
        """Close the client session."""
        if self.session:
            await self.session.close()
            self.session = None

    async def authenticate(self, token: str) -> Dict[str, Any]:
        """
        Authenticate with the API gateway.

        Args:
            token: JWT authentication token

        Returns:
            Authentication response
        """
        if not self.session:
            await self.initialize()

        try:
            async with self.session.post(
                f"{self.base_url}/auth/verify",
                json={"token": token}
            ) as response:
                return await response.json()
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return {"success": False, "error": str(e)}

    async def route_request(self, service: str, endpoint: str, method: str = "GET",
                          data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Route a request through the API gateway.

        Args:
            service: Target service name
            endpoint: Service endpoint
            method: HTTP method
            data: Request data

        Returns:
            Response from the target service
        """
        if not self.session:
            await self.initialize()

        try:
            url = f"{self.base_url}/api/{service}{endpoint}"
            async with self.session.request(method, url, json=data) as response:
                return await response.json()
        except Exception as e:
            logger.error(f"Request routing failed: {e}")
            return {"success": False, "error": str(e)}

    async def get_service_health(self) -> Dict[str, Any]:
        """Get health status of all services."""
        return await self.route_request("health", "", "GET")

    async def get_service_discovery(self) -> Dict[str, Any]:
        """Get service discovery information."""
        return await self.route_request("discovery", "", "GET")