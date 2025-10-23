"""
Trading AI Service Client
Client for communicating with the Trading AI Service
"""

import logging
import asyncio
from typing import Dict, Any, Optional
import aiohttp
from datetime import datetime

logger = logging.getLogger(__name__)


class TradingAIServiceClient:
    """Client for Trading AI Service communication"""

    def __init__(self, base_url: str = "http://localhost:8003"):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.auth_token: Optional[str] = None

    async def authenticate(self) -> bool:
        """Authenticate with the service"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            # For now, just check if service is available
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    logger.info("✅ Trading AI Service authentication successful")
                    return True
                else:
                    logger.warning(f"⚠️ Trading AI Service health check failed: {response.status}")
                    return False

        except Exception as e:
            logger.error(f"❌ Trading AI Service authentication failed: {e}")
            return False

    async def get_signal(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get trading signal from AI service"""
        try:
            if not self.session:
                await self.authenticate()

            payload = {
                "symbol": symbol,
                "market_data": market_data,
                "timestamp": datetime.now().isoformat()
            }

            async with self.session.post(f"{self.base_url}/signal", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result
                else:
                    logger.warning(f"⚠️ AI signal request failed: {response.status}")
                    return {"action": "HOLD", "confidence": 0.0}

        except Exception as e:
            logger.error(f"❌ Failed to get AI signal: {e}")
            return {"action": "HOLD", "confidence": 0.0, "error": str(e)}

    async def get_model_status(self) -> Dict[str, Any]:
        """Get AI model status"""
        try:
            if not self.session:
                await self.authenticate()

            async with self.session.get(f"{self.base_url}/model/status") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"status": "unknown", "error": f"HTTP {response.status}"}

        except Exception as e:
            logger.error(f"❌ Failed to get model status: {e}")
            return {"status": "error", "error": str(e)}

    async def update_model(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update AI model"""
        try:
            if not self.session:
                await self.authenticate()

            async with self.session.post(f"{self.base_url}/model/update", json=model_data) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"success": False, "error": f"HTTP {response.status}"}

        except Exception as e:
            logger.error(f"❌ Failed to update model: {e}")
            return {"success": False, "error": str(e)}

    async def close(self):
        """Close the client session"""
        if self.session:
            await self.session.close()
            self.session = None