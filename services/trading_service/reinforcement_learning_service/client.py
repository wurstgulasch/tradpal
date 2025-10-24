"""
TradPal Reinforcement Learning Service Client
Async client for communicating with the Reinforcement Learning Service
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
import aiohttp
import asyncio
from datetime import datetime

from config.settings import config

logger = logging.getLogger(__name__)


class ReinforcementLearningServiceClient:
    """Async client for Reinforcement Learning Service communication"""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or config.get('service', 'REINFORCEMENT_LEARNING_SERVICE_URL', fallback='http://localhost:8013')
        self.session: Optional[aiohttp.ClientSession] = None
        self.auth_token: Optional[str] = None

    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def initialize(self):
        """Initialize the client with authentication"""
        if self.session is None:
            self.session = aiohttp.ClientSession()

        # Authenticate if needed
        await self.authenticate()

    async def authenticate(self):
        """Authenticate with the service"""
        try:
            # For now, assume no authentication needed
            # In production, this would handle JWT or other auth
            self.auth_token = "authenticated"
            logger.info("Reinforcement Learning Service client authenticated")
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            raise

    async def close(self):
        """Close the client session"""
        if self.session:
            await self.session.close()
            self.session = None

    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request with error handling"""
        if not self.session:
            await self.initialize()

        url = f"{self.base_url}{endpoint}"

        try:
            async with self.session.request(method, url, **kwargs) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Request failed: {response.status} - {error_text}")
        except Exception as e:
            logger.error(f"Request to {url} failed: {str(e)}")
            raise

    async def train_agent(self, symbol: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Train a reinforcement learning agent"""
        data = {
            "symbol": symbol,
            "config": config
        }
        return await self._make_request("POST", "/train", json=data)

    async def get_action(self, symbol: str, state: List[float], agent_id: Optional[str] = None) -> Tuple[int, float]:
        """Get trading action from RL agent"""
        data = {
            "symbol": symbol,
            "state": state,
            "agent_id": agent_id
        }
        response = await self._make_request("POST", "/action", json=data)
        return response["action"], response["confidence"]

    async def list_agents(self) -> List[Dict[str, Any]]:
        """List available trained agents"""
        response = await self._make_request("GET", "/agents")
        return response.get("agents", [])

    async def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get information about a specific agent"""
        return await self._make_request("GET", f"/agent/{agent_id}")

    async def get_training_status(self, agent_id: str) -> Dict[str, Any]:
        """Get training status of an agent"""
        return await self._make_request("GET", f"/status/{agent_id}")

    async def delete_agent(self, agent_id: str) -> bool:
        """Delete a trained agent"""
        response = await self._make_request("DELETE", f"/agent/{agent_id}")
        return response.get("success", False)

    async def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        return await self._make_request("GET", "/health")


# Global client instance
_rl_client: Optional[ReinforcementLearningServiceClient] = None


async def get_reinforcement_learning_client() -> ReinforcementLearningServiceClient:
    """Get or create Reinforcement Learning service client"""
    global _rl_client

    if _rl_client is None:
        _rl_client = ReinforcementLearningServiceClient()

    if _rl_client.session is None:
        await _rl_client.initialize()

    return _rl_client