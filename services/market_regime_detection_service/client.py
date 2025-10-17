"""
Market Regime Detection Service Client
Client for market regime classification and clustering analysis.
"""

import asyncio
import aiohttp
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


class MarketRegimeDetectionServiceClient:
    """Client for Market Regime Detection Service"""

    def __init__(self, base_url: str = "http://localhost:8005"):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None

    async def initialize(self) -> None:
        """Initialize the client"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )

    async def close(self) -> None:
        """Close the client"""
        if self.session:
            await self.session.close()

    async def get_market_regime(self, symbol: str, lookback_periods: int = 100) -> Dict[str, Any]:
        """
        Get current market regime classification for a symbol.

        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            lookback_periods: Number of periods to analyze for regime detection

        Returns:
            Market regime classification with confidence scores
        """
        try:
            params = {
                "symbol": symbol,
                "lookback_periods": lookback_periods
            }

            async with self.session.get(f"{self.base_url}/regime/{symbol}",
                                      params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get market regime: {response.status}")
                    return {}

        except Exception as e:
            logger.error(f"Error getting market regime: {e}")
            return {}

    async def get_regime_features(self, symbol: str) -> Dict[str, Any]:
        """
        Get regime classification features for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Feature vector used for regime classification
        """
        try:
            async with self.session.get(f"{self.base_url}/features/{symbol}") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get regime features: {response.status}")
                    return {}

        except Exception as e:
            logger.error(f"Error getting regime features: {e}")
            return {}

    async def get_regime_history(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """
        Get historical regime classifications for a symbol.

        Args:
            symbol: Trading symbol
            days: Number of days of historical data

        Returns:
            Historical regime data with timestamps
        """
        try:
            params = {"days": days}

            async with self.session.get(f"{self.base_url}/history/{symbol}",
                                      params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get regime history: {response.status}")
                    return {}

        except Exception as e:
            logger.error(f"Error getting regime history: {e}")
            return {}

    async def get_regime_transition_probability(self, from_regime: str, to_regime: str) -> Dict[str, Any]:
        """
        Get transition probability between market regimes.

        Args:
            from_regime: Source regime
            to_regime: Target regime

        Returns:
            Transition probability data
        """
        try:
            params = {
                "from_regime": from_regime,
                "to_regime": to_regime
            }

            async with self.session.get(f"{self.base_url}/transition",
                                      params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get transition probability: {response.status}")
                    return {}

        except Exception as e:
            logger.error(f"Error getting transition probability: {e}")
            return {}

    async def get_volatility_regime(self, symbol: str) -> Dict[str, Any]:
        """
        Get volatility regime classification for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Volatility regime data
        """
        try:
            async with self.session.get(f"{self.base_url}/volatility/{symbol}") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get volatility regime: {response.status}")
                    return {}

        except Exception as e:
            logger.error(f"Error getting volatility regime: {e}")
            return {}

    async def get_trend_regime(self, symbol: str) -> Dict[str, Any]:
        """
        Get trend regime classification for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Trend regime data
        """
        try:
            async with self.session.get(f"{self.base_url}/trend/{symbol}") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get trend regime: {response.status}")
                    return {}

        except Exception as e:
            logger.error(f"Error getting trend regime: {e}")
            return {}

    async def get_regime_clusters(self, symbol: str) -> Dict[str, Any]:
        """
        Get regime clustering information for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Clustering data and centroids
        """
        try:
            async with self.session.get(f"{self.base_url}/clusters/{symbol}") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get regime clusters: {response.status}")
                    return {}

        except Exception as e:
            logger.error(f"Error getting regime clusters: {e}")
            return {}

    async def update_regime_model(self, symbol: str, new_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the regime detection model with new data.

        Args:
            symbol: Trading symbol
            new_data: New market data for model update

        Returns:
            Update status and new model parameters
        """
        try:
            async with self.session.post(f"{self.base_url}/update/{symbol}",
                                       json=new_data) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to update regime model: {response.status}")
                    return {}

        except Exception as e:
            logger.error(f"Error updating regime model: {e}")
            return {}

    async def get_regime_statistics(self, symbol: str) -> Dict[str, Any]:
        """
        Get statistical information about regime classifications.

        Args:
            symbol: Trading symbol

        Returns:
            Regime statistics and performance metrics
        """
        try:
            async with self.session.get(f"{self.base_url}/statistics/{symbol}") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get regime statistics: {response.status}")
                    return {}

        except Exception as e:
            logger.error(f"Error getting regime statistics: {e}")
            return {}