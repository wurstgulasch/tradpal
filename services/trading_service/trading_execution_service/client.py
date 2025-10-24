"""
TradPal Trading Execution Service Client
Async client for communicating with the Trading Execution Service
"""

import logging
from typing import Dict, Any, Optional, List
import aiohttp
import asyncio
from datetime import datetime

from config.settings import config

logger = logging.getLogger(__name__)


class TradingExecutionServiceClient:
    """Async client for Trading Execution Service communication"""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or config.get('service', 'TRADING_EXECUTION_SERVICE_URL', fallback='http://localhost:8016')
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
            logger.info("Trading Execution Service client authenticated")
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

    async def place_order(self, symbol: str, side: str, order_type: str = "market",
                         quantity: float = 0.0, price: Optional[float] = None,
                         stop_price: Optional[float] = None, time_in_force: str = "GTC") -> Dict[str, Any]:
        """Place a trading order"""
        payload = {
            "symbol": symbol,
            "side": side,
            "order_type": order_type,
            "quantity": quantity,
            "price": price,
            "stop_price": stop_price,
            "time_in_force": time_in_force
        }
        return await self._make_request("POST", "/order", json=payload)

    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order"""
        return await self._make_request("DELETE", f"/order/{order_id}")

    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status"""
        return await self._make_request("GET", f"/order/{order_id}")

    async def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get all open orders"""
        response = await self._make_request("GET", "/orders")
        return response.get("orders", [])

    async def get_portfolio(self) -> Dict[str, Any]:
        """Get current portfolio information"""
        return await self._make_request("GET", "/portfolio")

    async def get_account_balance(self) -> Dict[str, Any]:
        """Get account balance information"""
        return await self._make_request("GET", "/balance")

    async def close_position(self, symbol: str, quantity: Optional[float] = None) -> Dict[str, Any]:
        """Close a position"""
        params = {"symbol": symbol}
        if quantity is not None:
            params["quantity"] = quantity
        return await self._make_request("POST", "/close-position", params=params)

    async def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        return await self._make_request("GET", "/health")


# Global client instance
_execution_client: Optional[TradingExecutionServiceClient] = None


async def get_trading_execution_client() -> TradingExecutionServiceClient:
    """Get or create Trading Execution service client"""
    global _execution_client

    if _execution_client is None:
        _execution_client = TradingExecutionServiceClient()

    if _execution_client.session is None:
        await _execution_client.initialize()

    return _execution_client