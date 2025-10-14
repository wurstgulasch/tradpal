#!/usr/bin/env python3
"""
Risk Service Client

Async client for communicating with the Risk Service API.
Provides position sizing, risk assessment, and portfolio management.
"""

import asyncio
import aiohttp
import logging
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from config.settings import RISK_SERVICE_URL, API_KEY

logger = logging.getLogger(__name__)


class RiskServiceClient:
    """Async client for Risk Service API"""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or RISK_SERVICE_URL or "http://localhost:8002"
        self.session: Optional[aiohttp.ClientSession] = None
        self.api_key = API_KEY

    @asynccontextmanager
    async def _get_session(self):
        """Get or create HTTP session"""
        if self.session is None:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

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
        """Check if the risk service is healthy"""
        try:
            async with self._get_session() as session:
                async with session.get(f"{self.base_url}/health") as response:
                    return response.status == 200
        except Exception as e:
            logger.warning(f"Risk service health check failed: {e}")
            return False

    async def calculate_position_sizing(
        self,
        symbol: str,
        capital: float,
        entry_price: float,
        position_type: str,
        atr_value: Optional[float] = None,
        volatility: Optional[float] = None,
        risk_percentage: float = 0.01
    ) -> Dict[str, Any]:
        """
        Calculate optimal position sizing with risk management.

        Args:
            symbol: Trading symbol
            capital: Available capital
            entry_price: Entry price
            position_type: Position type (long/short)
            atr_value: ATR value for SL/TP calculation
            volatility: Current volatility
            risk_percentage: Risk percentage per trade

        Returns:
            Position sizing details
        """
        payload = {
            "symbol": symbol,
            "capital": capital,
            "entry_price": entry_price,
            "position_type": position_type,
            "risk_percentage": risk_percentage
        }

        if atr_value is not None:
            payload["atr_value"] = atr_value
        if volatility is not None:
            payload["volatility"] = volatility

        async with self._get_session() as session:
            async with session.post(
                f"{self.base_url}/position/sizing",
                json=payload
            ) as response:
                response.raise_for_status()
                result = await response.json()

                if result.get("success"):
                    return result.get("position_sizing", {})
                else:
                    raise RuntimeError(f"Position sizing failed: {result}")

    async def assess_portfolio_risk(
        self,
        returns_data: List[float],
        time_horizon: str = "daily"
    ) -> Dict[str, Any]:
        """
        Assess portfolio risk with comprehensive metrics.

        Args:
            returns_data: List of return values
            time_horizon: Risk assessment horizon

        Returns:
            Risk assessment results
        """
        payload = {
            "returns_data": returns_data,
            "time_horizon": time_horizon
        }

        async with self._get_session() as session:
            async with session.post(
                f"{self.base_url}/portfolio/assess",
                json=payload
            ) as response:
                response.raise_for_status()
                result = await response.json()

                if result.get("success"):
                    return result.get("risk_assessment", {})
                else:
                    raise RuntimeError(f"Risk assessment failed: {result}")

    async def get_portfolio_exposure(self) -> Dict[str, Any]:
        """Get current portfolio risk exposure"""
        async with self._get_session() as session:
            async with session.get(f"{self.base_url}/portfolio/exposure") as response:
                response.raise_for_status()
                return await response.json()

    async def update_risk_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update risk management parameters.

        Args:
            parameters: Dictionary of parameters to update

        Returns:
            Update confirmation
        """
        async with self._get_session() as session:
            async with session.put(
                f"{self.base_url}/parameters",
                json=parameters
            ) as response:
                response.raise_for_status()
                return await response.json()

    async def get_risk_parameters(self) -> Dict[str, Any]:
        """Get current risk management parameters"""
        async with self._get_session() as session:
            async with session.get(f"{self.base_url}/parameters") as response:
                response.raise_for_status()
                return await response.json()

    async def get_position_metrics(self, symbol: str) -> Dict[str, Any]:
        """
        Get risk metrics for a specific position.

        Args:
            symbol: Trading symbol

        Returns:
            Position risk metrics
        """
        async with self._get_session() as session:
            async with session.get(f"{self.base_url}/metrics/{symbol}") as response:
                response.raise_for_status()
                return await response.json()

    async def close_position(self, symbol: str) -> Dict[str, Any]:
        """
        Close/remove a position from portfolio tracking.

        Args:
            symbol: Trading symbol

        Returns:
            Close confirmation
        """
        async with self._get_session() as session:
            async with session.delete(f"{self.base_url}/portfolio/{symbol}") as response:
                response.raise_for_status()
                return await response.json()

    async def get_risk_history(self, limit: int = 10) -> Dict[str, Any]:
        """
        Get risk assessment history.

        Args:
            limit: Maximum number of history entries to return

        Returns:
            Risk history data
        """
        async with self._get_session() as session:
            async with session.get(f"{self.base_url}/history?limit={limit}") as response:
                response.raise_for_status()
                return await response.json()