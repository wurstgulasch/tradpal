#!/usr/bin/env python3
"""
Web UI Service Client - Async client for web interface service.

Provides async HTTP client for:
- Dashboard data retrieval
- Strategy management
- Backtesting operations
- Live trading monitoring
- User authentication
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

import aiohttp
from aiohttp import ClientTimeout

from config.settings import WEB_UI_SERVICE_URL, REQUEST_TIMEOUT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LoginRequest:
    """Login request data."""
    username: str
    password: str


@dataclass
class StrategyConfig:
    """Strategy configuration data."""
    name: str
    symbol: str
    timeframe: str
    parameters: Dict[str, Any]


@dataclass
class ChartRequest:
    """Chart data request."""
    symbol: str
    timeframe: str = "1h"
    limit: int = 100


@dataclass
class BacktestRequest:
    """Backtest request data."""
    strategy_name: str
    symbol: str
    start_date: str
    end_date: str
    initial_capital: float


class WebUIServiceClient:
    """Async client for Web UI Service."""

    def __init__(self, base_url: str = WEB_UI_SERVICE_URL):
        self.base_url = base_url.rstrip('/')
        self.timeout = ClientTimeout(total=REQUEST_TIMEOUT)

        # HTTP session
        self._session: Optional[aiohttp.ClientSession] = None

        # Authentication
        self._auth_token: Optional[str] = None

        logger.info(f"Web UI Service Client initialized with URL: {self.base_url}")

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
        params: Optional[Dict[str, Any]] = None,
        auth_required: bool = True
    ) -> Dict[str, Any]:
        """Make HTTP request to service."""
        if not self._session:
            await self.start()

        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        headers = {}
        if auth_required and self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"

        try:
            async with self._session.request(method, url, json=data, params=params, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 401:
                    raise Exception("Authentication required")
                else:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")

        except aiohttp.ClientError as e:
            logger.error(f"Request failed: {e}")
            raise Exception(f"Service request failed: {e}")

    async def login(self, username: str, password: str) -> Dict[str, Any]:
        """
        User login.

        Args:
            username: Username
            password: Password

        Returns:
            Login response with token
        """
        data = {"username": username, "password": password}
        response = await self._make_request("POST", "/auth/login", data, auth_required=False)

        if response.get("success"):
            self._auth_token = response.get("token")

        logger.info(f"Login successful for user: {username}")
        return response

    async def logout(self):
        """User logout."""
        self._auth_token = None
        logger.info("User logged out")

    async def health_check(self) -> Dict[str, Any]:
        """Check service health."""
        return await self._make_request("GET", "/health", auth_required=False)

    async def get_dashboard(self) -> Dict[str, Any]:
        """Get dashboard data (alias for get_dashboard_data)."""
        return await self.get_dashboard_data()

    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data."""
        return await self._make_request("GET", "/dashboard")

    async def get_live_dashboard_data(self) -> Dict[str, Any]:
        """Get live dashboard data with real-time updates."""
        return await self._make_request("GET", "/dashboard/live")

    async def get_strategies(self) -> List[Dict[str, Any]]:
        """Get available trading strategies."""
        response = await self._make_request("GET", "/strategies")
        return response.get("strategies", [])

    async def create_strategy(self, config: StrategyConfig) -> Dict[str, Any]:
        """
        Create a new trading strategy.

        Args:
            config: Strategy configuration

        Returns:
            Creation response
        """
        data = {
            "name": config.name,
            "symbol": config.symbol,
            "timeframe": config.timeframe,
            "parameters": config.parameters
        }

        response = await self._make_request("POST", "/strategies", data)

        logger.info(f"Strategy created: {config.name}")
        return response

    async def get_strategy(self, strategy_id: str) -> Dict[str, Any]:
        """Get strategy details."""
        return await self._make_request("GET", f"/strategies/{strategy_id}")

    async def update_strategy(self, strategy_id: str, config: StrategyConfig) -> Dict[str, Any]:
        """
        Update an existing strategy.

        Args:
            strategy_id: Strategy ID
            config: New strategy configuration

        Returns:
            Update response
        """
        data = {
            "name": config.name,
            "symbol": config.symbol,
            "timeframe": config.timeframe,
            "parameters": config.parameters
        }

        response = await self._make_request("PUT", f"/strategies/{strategy_id}", data)

        logger.info(f"Strategy updated: {strategy_id}")
        return response

    async def delete_strategy(self, strategy_id: str) -> Dict[str, Any]:
        """Delete a strategy."""
        response = await self._make_request("DELETE", f"/strategies/{strategy_id}")

        logger.info(f"Strategy deleted: {strategy_id}")
        return response

    async def run_backtest(self, request: BacktestRequest) -> Dict[str, Any]:
        """
        Run a backtest for a strategy.

        Args:
            request: Backtest request data

        Returns:
            Backtest response
        """
        data = {
            "strategy_name": request.strategy_name,
            "symbol": request.symbol,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "initial_capital": request.initial_capital
        }

        response = await self._make_request("POST", "/backtest", data)

        logger.info(f"Backtest started for {request.strategy_name}")
        return response

    async def get_backtest_results(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent backtest results."""
        params = {"limit": limit}
        response = await self._make_request("GET", "/backtests", params=params)
        return response.get("backtests", [])

    async def get_backtest_details(self, backtest_id: str) -> Dict[str, Any]:
        """Get detailed backtest results."""
        return await self._make_request("GET", f"/backtests/{backtest_id}")

    async def get_trading_status(self) -> Dict[str, Any]:
        """Get live trading status."""
        return await self._make_request("GET", "/trading/status")

    async def get_trading_performance(self) -> Dict[str, Any]:
        """Get live trading performance."""
        return await self._make_request("GET", "/trading/performance")

    async def get_chart_data(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> Dict[str, Any]:
        """Get chart data for visualization."""
        params = {"timeframe": timeframe, "limit": limit}
        return await self._make_request("GET", f"/charts/{symbol}", params=params)

    async def get_portfolio_analytics(self) -> Dict[str, Any]:
        """Get portfolio analytics."""
        return await self._make_request("GET", "/analytics/portfolio")

    async def get_risk_analytics(self) -> Dict[str, Any]:
        """Get risk analytics."""
        return await self._make_request("GET", "/analytics/risk")

    async def get_settings(self) -> Dict[str, Any]:
        """Get application settings."""
        return await self._make_request("GET", "/settings")

    async def update_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update application settings.

        Args:
            settings: New settings

        Returns:
            Update response
        """
        response = await self._make_request("PUT", "/settings", settings)

        logger.info("Settings updated")
        return response

    async def get_service_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics."""
        health = await self.health_check()
        dashboard = await self.get_dashboard_data()
        strategies = await self.get_strategies()
        backtests = await self.get_backtest_results(limit=5)

        return {
            "service_health": health,
            "dashboard_summary": dashboard,
            "total_strategies": len(strategies),
            "recent_backtests": backtests,
            "active_user": self._auth_token is not None
        }