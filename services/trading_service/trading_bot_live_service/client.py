#!/usr/bin/env python3
"""
Trading Bot Live Service Client - Async client for live trading service.

Provides async HTTP client for:
- Starting and stopping trading sessions
- Order management and position tracking
- Risk parameter updates
- Performance monitoring
- Emergency controls
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

import aiohttp
from aiohttp import ClientTimeout

from config.settings import TRADING_BOT_LIVE_SERVICE_URL, REQUEST_TIMEOUT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StartTradingRequest:
    """Request data for starting trading."""
    symbol: str
    strategy: str
    timeframe: str = "1m"
    capital: float = 10000.0
    risk_per_trade: float = 0.01
    max_positions: int = 1
    enable_paper_trading: bool = True


@dataclass
class StopTradingRequest:
    """Request data for stopping trading."""
    symbol: str
    close_positions: bool = True


@dataclass
class OrderRequest:
    """Request data for manual order placement."""
    symbol: str
    side: str
    quantity: float
    order_type: str = "market"
    price: Optional[float] = None


@dataclass
class RiskUpdateRequest:
    """Request data for updating risk parameters."""
    symbol: str
    risk_per_trade: Optional[float] = None
    max_positions: Optional[int] = None
    max_drawdown: Optional[float] = None


@dataclass
class TradeExecutionRequest:
    """Request data for trade execution."""
    symbol: str
    signal: str
    price: float
    confidence: float
    risk_parameters: Dict[str, Any]


@dataclass
class ClosePositionRequest:
    """Request data for closing positions."""
    position_id: str
    reason: str = "Manual close"


@dataclass
class ChartRequest:
    """Request data for chart data."""
    symbol: str
    timeframe: str
    start_date: str
    end_date: str


class TradingBotLiveServiceClient:
    """Async client for Trading Bot Live Service."""

    def __init__(self, base_url: str = TRADING_BOT_LIVE_SERVICE_URL):
        self.base_url = base_url.rstrip('/')
        self.timeout = ClientTimeout(total=REQUEST_TIMEOUT)

        # HTTP session
        self._session: Optional[aiohttp.ClientSession] = None

        logger.info(f"Trading Bot Live Service Client initialized with URL: {self.base_url}")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def start(self):
        """Start the HTTP session."""
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=self.timeout)

    async def close(self):
        """Close the HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make HTTP request to service."""
        if not self._session:
            await self.start()

        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        try:
            async with self._session.request(method, url, json=data, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")

        except aiohttp.ClientError as e:
            logger.error(f"Request failed: {e}")
            raise Exception(f"Service request failed: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Check service health."""
        return await self._make_request("GET", "/health")

    async def start_trading(self, request: StartTradingRequest) -> Dict[str, Any]:
        """
        Start live trading.

        Args:
            request: Trading start request data

        Returns:
            Trading start response
        """
        data = {
            "symbol": request.symbol,
            "strategy": request.strategy,
            "timeframe": request.timeframe,
            "capital": request.capital,
            "risk_per_trade": request.risk_per_trade,
            "max_positions": request.max_positions,
            "enable_paper_trading": request.enable_paper_trading
        }

        response = await self._make_request("POST", "/start", data)

        logger.info(f"Trading started for {request.symbol} with {request.strategy}")
        return response

    async def stop_trading(self, request: StopTradingRequest) -> Dict[str, Any]:
        """
        Stop live trading.

        Args:
            request: Trading stop request data

        Returns:
            Trading stop response
        """
        data = {
            "symbol": request.symbol,
            "close_positions": request.close_positions
        }

        response = await self._make_request("POST", "/stop", data)

        logger.info(f"Trading stopped for {request.symbol}")
        return response

    async def place_order(self, request: OrderRequest) -> Dict[str, Any]:
        """
        Place a manual order.

        Args:
            request: Order request data

        Returns:
            Order placement response
        """
        data = {
            "symbol": request.symbol,
            "side": request.side,
            "quantity": request.quantity,
            "order_type": request.order_type,
            "price": request.price
        }

        response = await self._make_request("POST", "/order", data)

        logger.info(f"Order placed: {request.side} {request.quantity} {request.symbol}")
        return response

    async def execute_trade(self, request: TradeExecutionRequest) -> Dict[str, Any]:
        """
        Execute a trade.

        Args:
            request: Trade execution request data

        Returns:
            Trade execution response
        """
        data = {
            "symbol": request.symbol,
            "signal": request.signal,
            "price": request.price,
            "confidence": request.confidence,
            "risk_parameters": request.risk_parameters
        }

        response = await self._make_request("POST", "/execute-trade", data)

        logger.info(f"Trade executed for {request.symbol}")
        return response

    async def get_trading_status(self) -> Dict[str, Any]:
        """Get status of all trading sessions."""
        return await self._make_request("GET", "/status")

    async def get_symbol_status(self, symbol: str) -> Dict[str, Any]:
        """Get trading status for a specific symbol."""
        return await self._make_request("GET", f"/status/{symbol}")

    async def get_positions(self) -> Dict[str, Any]:
        """Get all open positions."""
        return await self._make_request("GET", "/positions")

    async def get_symbol_positions(self, symbol: str) -> List[Dict[str, Any]]:
        """Get open positions for a specific symbol."""
        response = await self._make_request("GET", f"/positions/{symbol}")
        return response.get("positions", [])

    async def get_performance(self) -> Dict[str, Any]:
        """Get overall trading performance."""
        return await self._make_request("GET", "/performance")

    async def get_symbol_performance(self, symbol: str) -> Dict[str, Any]:
        """Get performance metrics for a specific symbol."""
        return await self._make_request("GET", f"/performance/{symbol}")

    async def get_order_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent order history."""
        params = {"limit": limit}
        response = await self._make_request("GET", "/orders", params=params)
        return response.get("orders", [])

    async def get_symbol_orders(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get order history for a specific symbol."""
        params = {"limit": limit}
        response = await self._make_request("GET", f"/orders/{symbol}", params=params)
        return response.get("orders", [])

    async def close_position(self, request: ClosePositionRequest) -> Dict[str, Any]:
        """
        Close a position.

        Args:
            request: Close position request data

        Returns:
            Close position response
        """
        data = {
            "position_id": request.position_id,
            "reason": request.reason
        }

        response = await self._make_request("POST", "/close-position", data)

        logger.info(f"Position {request.position_id} closed")
        return response

    async def get_configuration(self) -> Dict[str, Any]:
        """Get current trading configuration."""
        return await self._make_request("GET", "/config")

    async def start_trading_session(
        self,
        request: StartTradingRequest,
        monitor_duration: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Start a trading session and optionally monitor it.

        Args:
            request: Trading start request
            monitor_duration: Duration to monitor in seconds (None for indefinite)

        Returns:
            Session start result
        """
        # Start trading
        start_result = await self.start_trading(request)

        if monitor_duration:
            # Start monitoring task
            asyncio.create_task(self._monitor_session(request.symbol, monitor_duration))

        return start_result

    async def _monitor_session(self, symbol: str, duration: int):
        """Monitor a trading session for a specified duration."""
        try:
            await asyncio.sleep(duration)

            # Get final status
            status = await self.get_symbol_status(symbol)
            performance = await self.get_symbol_performance(symbol)

            logger.info(f"Monitoring completed for {symbol}")
            logger.info(f"Final status: {status}")
            logger.info(f"Performance: {performance}")

        except Exception as e:
            logger.error(f"Monitoring error for {symbol}: {e}")

    async def get_service_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics."""
        health = await self.health_check()
        status = await self.get_trading_status()
        performance = await self.get_performance()
        positions = await self.get_positions()

        return {
            "service_health": health,
            "trading_status": status,
            "overall_performance": performance,
            "open_positions": positions,
            "total_open_positions": len(positions)
        }