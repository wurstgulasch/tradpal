"""
TradPal Risk Management Service Client
Async client for communicating with the Risk Management Service
"""

import logging
from typing import Dict, Any, Optional, List
import aiohttp
import asyncio
from datetime import datetime

from config.settings import config

logger = logging.getLogger(__name__)


class RiskManagementServiceClient:
    """Async client for Risk Management Service communication"""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or config.get('service', 'RISK_MANAGEMENT_SERVICE_URL', fallback='http://localhost:8015')
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
            logger.info("Risk Management Service client authenticated")
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

    async def calculate_position_size(self, capital: float, entry_price: float, stop_loss: float, risk_config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimal position size"""
        payload = {
            "capital": capital,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "risk_config": risk_config
        }
        return await self._make_request("POST", "/position-size", json=payload)

    async def assess_portfolio_risk(self, positions: List[Dict[str, Any]], capital: float, risk_config: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall portfolio risk"""
        payload = {
            "positions": positions,
            "capital": capital,
            "risk_config": risk_config
        }
        return await self._make_request("POST", "/portfolio-risk", json=payload)

    async def calculate_kelly_criterion(self, win_rate: float, win_loss_ratio: float) -> float:
        """Calculate Kelly Criterion"""
        params = {"win_rate": win_rate, "win_loss_ratio": win_loss_ratio}
        response = await self._make_request("GET", "/kelly-criterion", params=params)
        return response["kelly_percentage"]

    async def calculate_var(self, confidence: float, returns: List[float]) -> float:
        """Calculate Value at Risk"""
        payload = {"returns": returns}
        response = await self._make_request("GET", f"/var/{confidence}", json=payload)
        return response["var"]

    async def validate_trade(self, trade: Dict[str, Any], risk_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate trade against risk criteria"""
        payload = {"trade": trade, "risk_config": risk_config}
        return await self._make_request("POST", "/validate-trade", json=payload)

    async def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        return await self._make_request("GET", "/health")


# Global client instance
_risk_client: Optional[RiskManagementServiceClient] = None


async def get_risk_management_client() -> RiskManagementServiceClient:
    """Get or create Risk Management service client"""
    global _risk_client

    if _risk_client is None:
        _risk_client = RiskManagementServiceClient()

    if _risk_client.session is None:
        await _risk_client.initialize()

    return _risk_client