"""
TradPal Market Regime Service Client
Async client for communicating with the Market Regime Service
"""

import logging
from typing import Dict, Any, Optional, List
import aiohttp
import asyncio
from datetime import datetime

from config.settings import config

logger = logging.getLogger(__name__)


class MarketRegimeServiceClient:
    """Async client for Market Regime Service communication"""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or config.get('service', 'MARKET_REGIME_SERVICE_URL', fallback='http://localhost:8014')
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
            logger.info("Market Regime Service client authenticated")
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

    async def analyze_market_regime(self, symbol: str, data: List[Dict[str, Any]], lookback_periods: int = 20) -> Dict[str, Any]:
        """Analyze market regime for given data"""
        payload = {
            "symbol": symbol,
            "data": data,
            "lookback_periods": lookback_periods
        }
        return await self._make_request("POST", "/analyze", json=payload)

    async def analyze_multi_timeframe_regime(self, symbol: str, timeframes: List[str], data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze market regime across multiple timeframes"""
        payload = {
            "symbol": symbol,
            "timeframes": timeframes,
            "data": data
        }
        return await self._make_request("POST", "/analyze-multi", json=payload)

    async def get_available_regimes(self) -> List[str]:
        """List all available market regime types"""
        response = await self._make_request("GET", "/regimes")
        return response.get("regimes", [])

    async def get_regime_statistics(self, symbol: str) -> Dict[str, Any]:
        """Get regime statistics for a symbol"""
        return await self._make_request("GET", f"/statistics/{symbol}")

    async def predict_regime(self, symbol: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict market regime for future data"""
        return await self._make_request("POST", f"/predict/{symbol}", json=data)

    async def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        return await self._make_request("GET", "/health")


# Global client instance
_regime_client: Optional[MarketRegimeServiceClient] = None


async def get_market_regime_client() -> MarketRegimeServiceClient:
    """Get or create Market Regime service client"""
    global _regime_client

    if _regime_client is None:
        _regime_client = MarketRegimeServiceClient()

    if _regime_client.session is None:
        await _regime_client.initialize()

    return _regime_client