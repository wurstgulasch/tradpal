"""
TradPal ML Training Service Client
Async client for communicating with the ML Training Service
"""

import logging
from typing import Dict, Any, Optional, List
import aiohttp
import asyncio
from datetime import datetime

from config.settings import config

logger = logging.getLogger(__name__)


class MLTrainingServiceClient:
    """Async client for ML Training Service communication"""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or config.get('service', 'ML_TRAINING_SERVICE_URL', fallback='http://localhost:8012')
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
            logger.info("ML Training Service client authenticated")
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

    async def train_model(self, symbol: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Train a machine learning model"""
        data = {
            "symbol": symbol,
            "config": config
        }
        return await self._make_request("POST", "/train", json=data)

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available trained models"""
        response = await self._make_request("GET", "/models")
        return response.get("models", [])

    async def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        return await self._make_request("GET", f"/model/{model_id}")

    async def delete_model(self, model_id: str) -> bool:
        """Delete a trained model"""
        response = await self._make_request("DELETE", f"/model/{model_id}")
        return response.get("success", False)

    async def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        return await self._make_request("GET", "/health")


# Global client instance
_ml_training_client: Optional[MLTrainingServiceClient] = None


async def get_ml_training_client() -> MLTrainingServiceClient:
    """Get or create ML Training service client"""
    global _ml_training_client

    if _ml_training_client is None:
        _ml_training_client = MLTrainingServiceClient()

    if _ml_training_client.session is None:
        await _ml_training_client.initialize()

    return _ml_training_client