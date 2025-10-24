"""
Alternative Data Service Client
Client for accessing alternative data sources and sentiment analysis.
"""

import asyncio
import aiohttp
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


class AlternativeDataService:
    """Client for Alternative Data Service"""

    def __init__(self, base_url: str = "http://localhost:8004"):
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

    async def get_sentiment_data(self, symbol: str, timeframe: str = "1h",
                               limit: int = 100) -> Dict[str, Any]:
        """
        Get sentiment analysis data for a symbol.

        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            timeframe: Data timeframe
            limit: Number of data points

        Returns:
            Sentiment data with scores and sources
        """
        try:
            params = {
                "symbol": symbol,
                "timeframe": timeframe,
                "limit": limit
            }

            async with self.session.get(f"{self.base_url}/sentiment/{symbol}",
                                      params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get sentiment data: {response.status}")
                    return {}

        except Exception as e:
            logger.error(f"Error getting sentiment data: {e}")
            return {}

    async def get_onchain_metrics(self, symbol: str, metrics: List[str] = None) -> Dict[str, Any]:
        """
        Get on-chain metrics for a symbol.

        Args:
            symbol: Trading symbol
            metrics: List of specific metrics to retrieve

        Returns:
            On-chain metrics data
        """
        try:
            params = {"symbol": symbol}
            if metrics:
                params["metrics"] = ",".join(metrics)

            async with self.session.get(f"{self.base_url}/onchain/{symbol}",
                                      params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get on-chain metrics: {response.status}")
                    return {}

        except Exception as e:
            logger.error(f"Error getting on-chain metrics: {e}")
            return {}

    async def get_economic_indicators(self, indicators: List[str] = None) -> Dict[str, Any]:
        """
        Get economic indicators data.

        Args:
            indicators: List of specific indicators to retrieve

        Returns:
            Economic indicators data
        """
        try:
            params = {}
            if indicators:
                params["indicators"] = ",".join(indicators)

            async with self.session.get(f"{self.base_url}/economic",
                                      params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get economic indicators: {response.status}")
                    return {}

        except Exception as e:
            logger.error(f"Error getting economic indicators: {e}")
            return {}

    async def get_composite_score(self, symbol: str) -> Dict[str, Any]:
        """
        Get composite alternative data score for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Composite score combining all alternative data sources
        """
        try:
            async with self.session.get(f"{self.base_url}/composite/{symbol}") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get composite score: {response.status}")
                    return {}

        except Exception as e:
            logger.error(f"Error getting composite score: {e}")
            return {}

    async def get_market_regime_context(self, symbol: str) -> Dict[str, Any]:
        """
        Get market regime context from alternative data.

        Args:
            symbol: Trading symbol

        Returns:
            Market regime context data
        """
        try:
            async with self.session.get(f"{self.base_url}/regime/{symbol}") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get regime context: {response.status}")
                    return {}

        except Exception as e:
            logger.error(f"Error getting regime context: {e}")
            return {}

    async def get_fear_greed_index(self) -> Dict[str, Any]:
        """
        Get Fear & Greed Index data.

        Returns:
            Fear & Greed Index data
        """
        try:
            async with self.session.get(f"{self.base_url}/fear-greed") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get Fear & Greed Index: {response.status}")
                    return {}

        except Exception as e:
            logger.error(f"Error getting Fear & Greed Index: {e}")
            return {}

    async def get_news_sentiment(self, symbol: str, hours: int = 24) -> Dict[str, Any]:
        """
        Get news sentiment analysis for a symbol.

        Args:
            symbol: Trading symbol
            hours: Hours of news to analyze

        Returns:
            News sentiment data
        """
        try:
            params = {"hours": hours}

            async with self.session.get(f"{self.base_url}/news/{symbol}",
                                      params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get news sentiment: {response.status}")
                    return {}

        except Exception as e:
            logger.error(f"Error getting news sentiment: {e}")
            return {}

    async def get_social_sentiment(self, symbol: str, platform: str = "twitter") -> Dict[str, Any]:
        """
        Get social media sentiment for a symbol.

        Args:
            symbol: Trading symbol
            platform: Social platform (twitter, reddit, etc.)

        Returns:
            Social sentiment data
        """
        try:
            params = {"platform": platform}

            async with self.session.get(f"{self.base_url}/social/{symbol}",
                                      params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get social sentiment: {response.status}")
                    return {}

        except Exception as e:
            logger.error(f"Error getting social sentiment: {e}")
            return {}