#!/usr/bin/env python3
"""
Security Service Client

Async client for communicating with the Security Service API.
Provides mTLS credentials, JWT tokens, and secrets management.
"""

import asyncio
import aiohttp
import logging
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from config.settings import SECURITY_SERVICE_URL, API_KEY

logger = logging.getLogger(__name__)


class SecurityServiceClient:
    """Async client for Security Service API"""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or SECURITY_SERVICE_URL or "http://localhost:8004"
        self.session: Optional[aiohttp.ClientSession] = None
        self.api_key = API_KEY
        self._jwt_token: Optional[str] = None

    @asynccontextmanager
    async def _get_session(self):
        """Get or create HTTP session"""
        if self.session is None:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            if self._jwt_token:
                headers["Authorization"] = f"Bearer {self._jwt_token}"

            self.session = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
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
        """Check if the security service is healthy"""
        try:
            async with self._get_session() as session:
                async with session.get(f"{self.base_url}/health") as response:
                    return response.status == 200
        except Exception as e:
            logger.warning(f"Security service health check failed: {e}")
            return False

    async def authenticate(self, service_name: str = "tradpal_orchestrator") -> bool:
        """
        Authenticate with the security service and get JWT token.

        Args:
            service_name: Name of the service requesting authentication

        Returns:
            True if authentication successful
        """
        try:
            payload = {
                "service_name": service_name,
                "permissions": ["read", "write"]
            }

            async with self._get_session() as session:
                async with session.post(
                    f"{self.base_url}/tokens/generate",
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        self._jwt_token = result.get("token")
                        logger.info(f"Authenticated as {service_name}")
                        return True
                    else:
                        logger.error(f"Authentication failed: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False

    async def validate_token(self) -> Dict[str, Any]:
        """Validate the current JWT token"""
        if not self._jwt_token:
            return {"valid": False, "error": "No token available"}

        try:
            async with self._get_session() as session:
                async with session.post(f"{self.base_url}/tokens/validate") as response:
                    response.raise_for_status()
                    return await response.json()
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            return {"valid": False, "error": str(e)}

    async def issue_service_credentials(self, service_name: str) -> Dict[str, Any]:
        """
        Issue mTLS credentials for a service.

        Args:
            service_name: Name of the service

        Returns:
            Service credentials
        """
        payload = {"service_name": service_name}

        async with self._get_session() as session:
            async with session.post(
                f"{self.base_url}/credentials/issue",
                json=payload
            ) as response:
                response.raise_for_status()
                return await response.json()

    async def store_secret(self, path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store a secret.

        Args:
            path: Secret path
            data: Secret data

        Returns:
            Storage result
        """
        payload = {"path": path, "data": data}

        async with self._get_session() as session:
            async with session.post(
                f"{self.base_url}/secrets/store",
                json=payload
            ) as response:
                response.raise_for_status()
                return await response.json()

    async def retrieve_secret(self, path: str) -> Dict[str, Any]:
        """
        Retrieve a secret.

        Args:
            path: Secret path

        Returns:
            Secret data
        """
        async with self._get_session() as session:
            async with session.get(f"{self.base_url}/secrets/{path}") as response:
                response.raise_for_status()
                return await response.json()

    async def list_credentials(self) -> Dict[str, Any]:
        """List all service credentials"""
        async with self._get_session() as session:
            async with session.get(f"{self.base_url}/credentials") as response:
                response.raise_for_status()
                return await response.json()

    async def list_tokens(self) -> Dict[str, Any]:
        """List all active tokens"""
        async with self._get_session() as session:
            async with session.get(f"{self.base_url}/tokens") as response:
                response.raise_for_status()
                return await response.json()