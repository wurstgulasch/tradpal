#!/usr/bin/env python3
"""
Economic Data Collector - Macro Economic Indicators

Collects economic indicators that impact financial markets:
- Fed Funds Rate (US Federal Reserve interest rate)
- CPI (Consumer Price Index - inflation)
- Unemployment Rate
- GDP Growth
- PMI (Purchasing Managers Index)
- Other key economic indicators
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import json

import aiohttp
import pandas as pd
import numpy as np

from .__init__ import EconomicData, EconomicIndicator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EconomicDataCollector:
    """
    Collector for macroeconomic indicators from various sources.
    """

    def __init__(self):
        self.session = None
        self.api_keys = {}
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour for economic data

        # Load API keys
        self._load_api_keys()

    def _load_api_keys(self):
        """Load API keys from environment."""
        import os
        self.api_keys = {
            'fred': os.getenv('FRED_API_KEY'),  # Federal Reserve Economic Data
            'alpha_vantage': os.getenv('ALPHA_VANTAGE_API_KEY'),
            'quandl': os.getenv('QUANDL_API_KEY'),
            'bls': os.getenv('BLS_API_KEY'),  # Bureau of Labor Statistics
        }

    async def initialize(self):
        """Initialize the collector."""
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
            logger.info("Economic Data Collector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Economic Data Collector: {e}")
            raise

    async def close(self):
        """Close resources."""
        if self.session:
            await self.session.close()

    async def get_indicators(self, indicators: Optional[List[str]] = None) -> List[EconomicData]:
        """
        Get economic indicators.

        Args:
            indicators: Specific indicators to fetch (optional)

        Returns:
            List of economic data points
        """
        try:
            # Default indicators if none specified
            if indicators is None:
                indicators = [
                    'fed_funds_rate',
                    'cpi',
                    'unemployment_rate',
                    'gdp_growth',
                    'pmi'
                ]

            # Map string indicators to enum
            indicator_enums = []
            for indicator in indicators:
                try:
                    indicator_enum = EconomicIndicator(indicator.upper())
                    indicator_enums.append(indicator_enum)
                except ValueError:
                    logger.warning(f"Unknown indicator: {indicator}")
                    continue

            # Collect data in parallel
            tasks = [self._get_indicator_data(indicator) for indicator in indicator_enums]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            economic_data = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"Indicator collection failed for {indicator_enums[i]}: {result}")
                    continue

                if result:
                    economic_data.extend(result)

            return economic_data

        except Exception as e:
            logger.error(f"Economic indicators collection failed: {e}")
            return []

    async def _get_indicator_data(self, indicator: EconomicIndicator) -> List[EconomicData]:
        """Get data for a specific economic indicator."""
        try:
            if indicator == EconomicIndicator.FED_FUNDS_RATE:
                return await self._get_fed_funds_rate()
            elif indicator == EconomicIndicator.CPI:
                return await self._get_cpi()
            elif indicator == EconomicIndicator.UNEMPLOYMENT_RATE:
                return await self._get_unemployment_rate()
            elif indicator == EconomicIndicator.GDP_GROWTH:
                return await self._get_gdp_growth()
            elif indicator == EconomicIndicator.PMI:
                return await self._get_pmi()
            else:
                return []
        except Exception as e:
            logger.error(f"Economic indicator {indicator} failed: {e}")
            return []

    async def _get_fed_funds_rate(self) -> List[EconomicData]:
        """Get Federal Funds Rate."""
        try:
            # Try FRED API first
            if self.api_keys.get('fred'):
                return await self._get_fred_data('FEDFUNDS', EconomicIndicator.FED_FUNDS_RATE)

            # Fallback to Alpha Vantage
            if self.api_keys.get('alpha_vantage'):
                return await self._get_alpha_vantage_fed_funds()

            # Return mock data
            return [EconomicData(
                indicator=EconomicIndicator.FED_FUNDS_RATE,
                value=5.25 + np.random.uniform(-0.25, 0.25),  # Current range
                timestamp=datetime.utcnow(),
                source='mock',
                metadata={'note': 'API keys not configured'}
            )]

        except Exception as e:
            logger.error(f"Fed Funds Rate fetch failed: {e}")
            return []

    async def _get_cpi(self) -> List[EconomicData]:
        """Get Consumer Price Index."""
        try:
            # Try FRED API
            if self.api_keys.get('fred'):
                return await self._get_fred_data('CPIAUCSL', EconomicIndicator.CPI)

            # Try BLS API
            if self.api_keys.get('bls'):
                return await self._get_bls_cpi()

            # Return mock data
            return [EconomicData(
                indicator=EconomicIndicator.CPI,
                value=300 + np.random.uniform(-5, 5),  # Current level
                timestamp=datetime.utcnow(),
                source='mock',
                metadata={'note': 'API keys not configured'}
            )]

        except Exception as e:
            logger.error(f"CPI fetch failed: {e}")
            return []

    async def _get_unemployment_rate(self) -> List[EconomicData]:
        """Get Unemployment Rate."""
        try:
            # Try FRED API
            if self.api_keys.get('fred'):
                return await self._get_fred_data('UNRATE', EconomicIndicator.UNEMPLOYMENT_RATE)

            # Try BLS API
            if self.api_keys.get('bls'):
                return await self._get_bls_unemployment()

            # Return mock data
            return [EconomicData(
                indicator=EconomicIndicator.UNEMPLOYMENT_RATE,
                value=4.0 + np.random.uniform(-0.5, 0.5),  # Current level
                timestamp=datetime.utcnow(),
                source='mock',
                metadata={'note': 'API keys not configured'}
            )]

        except Exception as e:
            logger.error(f"Unemployment Rate fetch failed: {e}")
            return []

    async def _get_gdp_growth(self) -> List[EconomicData]:
        """Get GDP Growth."""
        try:
            # Try FRED API
            if self.api_keys.get('fred'):
                return await self._get_fred_data('A191RL1Q225SBEA', EconomicIndicator.GDP_GROWTH)

            # Return mock data
            return [EconomicData(
                indicator=EconomicIndicator.GDP_GROWTH,
                value=2.5 + np.random.uniform(-1.0, 1.0),  # Quarterly growth
                timestamp=datetime.utcnow(),
                source='mock',
                metadata={'note': 'API keys not configured'}
            )]

        except Exception as e:
            logger.error(f"GDP Growth fetch failed: {e}")
            return []

    async def _get_pmi(self) -> List[EconomicData]:
        """Get Purchasing Managers Index."""
        try:
            # PMI data is typically from private sources like ISM
            # For demo, we'll use mock data
            return [EconomicData(
                indicator=EconomicIndicator.PMI,
                value=50 + np.random.uniform(-5, 5),  # Above 50 = expansion
                timestamp=datetime.utcnow(),
                source='mock',
                metadata={'note': 'PMI requires premium API access'}
            )]

        except Exception as e:
            logger.error(f"PMI fetch failed: {e}")
            return []

    # API-specific implementations
    async def _get_fred_data(self, series_id: str, indicator: EconomicIndicator) -> List[EconomicData]:
        """Get data from FRED API."""
        try:
            url = f"https://api.stlouisfed.org/fred/series/observations"
            params = {
                'series_id': series_id,
                'api_key': self.api_keys['fred'],
                'file_type': 'json',
                'limit': 1,
                'sort_order': 'desc'
            }

            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"FRED API error: {response.status}")

                data = await response.json()

            if data.get('observations'):
                latest = data['observations'][0]
                if latest.get('value') != '.':
                    return [EconomicData(
                        indicator=indicator,
                        value=float(latest['value']),
                        timestamp=datetime.strptime(latest['date'], '%Y-%m-%d'),
                        source='fred',
                        metadata={'series_id': series_id}
                    )]
            return []

        except Exception as e:
            logger.error(f"FRED data fetch failed for {series_id}: {e}")
            raise

    async def _get_alpha_vantage_fed_funds(self) -> List[EconomicData]:
        """Get Fed Funds Rate from Alpha Vantage."""
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'FEDERAL_FUNDS_RATE',
                'apikey': self.api_keys['alpha_vantage']
            }

            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"Alpha Vantage API error: {response.status}")

                data = await response.json()

            if data.get('data'):
                latest = data['data'][0]
                return [EconomicData(
                    indicator=EconomicIndicator.FED_FUNDS_RATE,
                    value=float(latest['value']),
                    timestamp=datetime.strptime(latest['date'], '%Y-%m-%d'),
                    source='alpha_vantage',
                    metadata={'function': 'FEDERAL_FUNDS_RATE'}
                )]
            return []

        except Exception as e:
            logger.error(f"Alpha Vantage Fed Funds fetch failed: {e}")
            raise

    async def _get_bls_cpi(self) -> List[EconomicData]:
        """Get CPI from Bureau of Labor Statistics."""
        try:
            # BLS API requires specific series IDs and registration
            # This is a simplified example
            url = "https://api.bls.gov/publicAPI/v2/timeseries/data/CUUR0000SA0"
            headers = {'Content-Type': 'application/json'}
            payload = {
                'seriesid': ['CUUR0000SA0'],
                'startyear': str(datetime.utcnow().year),
                'endyear': str(datetime.utcnow().year),
                'registrationkey': self.api_keys.get('bls')
            }

            async with self.session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    raise Exception(f"BLS API error: {response.status}")

                data = await response.json()

            # Parse BLS response format
            if data.get('Results', {}).get('series'):
                series = data['Results']['series'][0]
                if series.get('data'):
                    latest = series['data'][0]
                    return [EconomicData(
                        indicator=EconomicIndicator.CPI,
                        value=float(latest['value']),
                        timestamp=datetime.strptime(f"{latest['year']}-{latest['period'][1:]}", '%Y-%m'),
                        source='bls',
                        metadata={'series_id': 'CUUR0000SA0'}
                    )]
            return []

        except Exception as e:
            logger.error(f"BLS CPI fetch failed: {e}")
            raise

    async def _get_bls_unemployment(self) -> List[EconomicData]:
        """Get Unemployment Rate from Bureau of Labor Statistics."""
        try:
            url = "https://api.bls.gov/publicAPI/v2/timeseries/data/LNS14000000"
            headers = {'Content-Type': 'application/json'}
            payload = {
                'seriesid': ['LNS14000000'],
                'startyear': str(datetime.utcnow().year),
                'endyear': str(datetime.utcnow().year),
                'registrationkey': self.api_keys.get('bls')
            }

            async with self.session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    raise Exception(f"BLS API error: {response.status}")

                data = await response.json()

            if data.get('Results', {}).get('series'):
                series = data['Results']['series'][0]
                if series.get('data'):
                    latest = series['data'][0]
                    return [EconomicData(
                        indicator=EconomicIndicator.UNEMPLOYMENT_RATE,
                        value=float(latest['value']),
                        timestamp=datetime.strptime(f"{latest['year']}-{latest['period'][1:]}", '%Y-%m'),
                        source='bls',
                        metadata={'series_id': 'LNS14000000'}
                    )]
            return []

        except Exception as e:
            logger.error(f"BLS Unemployment fetch failed: {e}")
            raise

    async def get_fed_speech_sentiment(self) -> Dict[str, Any]:
        """Analyze sentiment of recent Fed speeches."""
        try:
            # This would integrate with news APIs to get Fed speeches
            # For now, return mock analysis
            return {
                'sentiment': 'neutral',
                'confidence': 0.6,
                'key_themes': ['inflation', 'employment', 'growth'],
                'hawkish_dovish_score': 0.1,  # Slightly hawkish
                'timestamp': datetime.utcnow(),
                'source': 'mock'
            }
        except Exception as e:
            logger.error(f"Fed speech sentiment analysis failed: {e}")
            return {}

    async def predict_rate_decisions(self) -> Dict[str, float]:
        """Predict upcoming Fed rate decisions based on economic data."""
        try:
            # Simple prediction model based on current indicators
            indicators = await self.get_indicators(['cpi', 'unemployment_rate', 'gdp_growth'])

            # Mock prediction logic
            cpi_value = next((d.value for d in indicators if d.indicator == EconomicIndicator.CPI), 300)
            unemployment = next((d.value for d in indicators if d.indicator == EconomicIndicator.UNEMPLOYMENT_RATE), 4.0)

            # Simple rule: If CPI > 310 and unemployment < 4.5, likely rate hike
            if cpi_value > 310 and unemployment < 4.5:
                rate_hike_probability = 0.7
            elif cpi_value < 290 or unemployment > 5.0:
                rate_hike_probability = 0.2
            else:
                rate_hike_probability = 0.5

            return {
                'rate_hike_probability': rate_hike_probability,
                'expected_change': 0.25 if rate_hike_probability > 0.6 else 0.0,
                'next_meeting': (datetime.utcnow() + timedelta(days=30)).date(),
                'based_on_indicators': ['cpi', 'unemployment', 'gdp']
            }

        except Exception as e:
            logger.error(f"Rate decision prediction failed: {e}")
            return {}

    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get economic data collection metrics."""
        return {
            "configured_apis": {
                "fred": self.api_keys.get('fred') is not None,
                "alpha_vantage": self.api_keys.get('alpha_vantage') is not None,
                "quandl": self.api_keys.get('quandl') is not None,
                "bls": self.api_keys.get('bls') is not None
            },
            "cache_size": len(self.cache),
            "cache_ttl": self.cache_ttl
        }

    async def get_metrics(self) -> Dict[str, Any]:
        """Get economic data collection metrics."""
        return await self.get_metrics_summary()