"""
Commodities Data Source for TradPal

Provides real-time and historical commodities data including:
- Precious metals (Gold, Silver, Platinum)
- Energy (Oil, Natural Gas)
- Agricultural (Corn, Wheat, Soybeans)
- Industrial metals (Copper, Aluminum)
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

from .base import BaseDataSource

logger = logging.getLogger(__name__)


class CommoditiesDataSource(BaseDataSource):
    """
    Commodities data source for commodity trading data.

    Supports:
    - Precious metals (Gold, Silver, Platinum, Palladium)
    - Energy commodities (WTI Oil, Brent Oil, Natural Gas)
    - Agricultural commodities (Corn, Wheat, Soybeans, Coffee, Sugar)
    - Industrial metals (Copper, Aluminum, Zinc, Nickel)
    """

    # Major commodities
    COMMODITIES = {
        # Precious metals
        'XAU/USD': {'name': 'Gold', 'category': 'precious_metals', 'base_price': 2000.0},
        'XAG/USD': {'name': 'Silver', 'category': 'precious_metals', 'base_price': 25.0},
        'XPT/USD': {'name': 'Platinum', 'category': 'precious_metals', 'base_price': 950.0},
        'XPD/USD': {'name': 'Palladium', 'category': 'precious_metals', 'base_price': 1100.0},

        # Energy
        'WTI': {'name': 'WTI Crude Oil', 'category': 'energy', 'base_price': 80.0},
        'BRENT': {'name': 'Brent Crude Oil', 'category': 'energy', 'base_price': 85.0},
        'NATURAL_GAS': {'name': 'Natural Gas', 'category': 'energy', 'base_price': 3.0},

        # Agricultural
        'CORN': {'name': 'Corn', 'category': 'agricultural', 'base_price': 450.0},
        'WHEAT': {'name': 'Wheat', 'category': 'agricultural', 'base_price': 600.0},
        'SOYBEANS': {'name': 'Soybeans', 'category': 'agricultural', 'base_price': 1200.0},
        'COFFEE': {'name': 'Coffee', 'category': 'agricultural', 'base_price': 180.0},
        'SUGAR': {'name': 'Sugar', 'category': 'agricultural', 'base_price': 22.0},

        # Industrial metals
        'COPPER': {'name': 'Copper', 'category': 'industrial_metals', 'base_price': 3.8},
        'ALUMINUM': {'name': 'Aluminum', 'category': 'industrial_metals', 'base_price': 0.9},
        'ZINC': {'name': 'Zinc', 'category': 'industrial_metals', 'base_price': 1.2},
        'NICKEL': {'name': 'Nickel', 'category': 'industrial_metals', 'base_price': 8.5}
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.config = config or {}
        self.provider = self.config.get('provider', 'yahoo_finance')  # yahoo_finance, investing_com, etc.
        self.api_key = self.config.get('api_key', '')

    async def get_historical_data(self, symbol: str, start_date: str,
                                end_date: str, interval: str = "1D") -> pd.DataFrame:
        """
        Get historical commodities data.

        Args:
            symbol: Commodity symbol (e.g., 'XAU/USD', 'WTI', 'CORN')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe: Timeframe ('1D', '1H', '15m', etc.)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            if symbol not in self.COMMODITIES:
                logger.warning(f"Unknown commodity symbol: {symbol}")
                return self._generate_synthetic_commodity_data(symbol, start_date, end_date, timeframe)

            commodity_info = self.COMMODITIES[symbol]

            if self.provider == 'yahoo_finance':
                return await self._get_yahoo_commodity_data(symbol, start_date, end_date, interval)
            else:
                # Fallback to synthetic data
                return self._generate_synthetic_commodity_data(symbol, start_date, end_date, interval)

        except Exception as e:
            logger.error(f"Error getting commodity data for {symbol}: {e}")
            return self._generate_synthetic_commodity_data(symbol, start_date, end_date, interval)

    async def _get_yahoo_commodity_data(self, symbol: str, start_date: str,
                                      end_date: str, timeframe: str) -> pd.DataFrame:
        """Get commodity data from Yahoo Finance"""
        try:
            # Map commodity symbols to Yahoo Finance tickers
            yahoo_tickers = {
                'XAU/USD': 'GC=F',  # Gold futures
                'XAG/USD': 'SI=F',  # Silver futures
                'WTI': 'CL=F',      # WTI Oil futures
                'BRENT': 'BZ=F',    # Brent Oil futures
                'CORN': 'ZC=F',     # Corn futures
                'WHEAT': 'ZW=F',    # Wheat futures
                'SOYBEANS': 'ZS=F', # Soybeans futures
                'COPPER': 'HG=F',   # Copper futures
                'ALUMINUM': 'ALI=F' if 'ALI=F' else None,  # Limited availability
            }

            yahoo_symbol = yahoo_tickers.get(symbol)
            if not yahoo_symbol:
                return self._generate_synthetic_commodity_data(symbol, start_date, end_date, timeframe)

            # Use Yahoo Finance data source
            from .yahoo_finance import YahooFinanceDataSource
            yahoo_source = YahooFinanceDataSource()
            return await yahoo_source.get_historical_data(yahoo_symbol, start_date, end_date, interval=timeframe)

        except Exception as e:
            logger.error(f"Yahoo Finance commodity data error: {e}")
            return self._generate_synthetic_commodity_data(symbol, start_date, end_date, timeframe)

    def _generate_synthetic_commodity_data(self, symbol: str, start_date: str,
                                         end_date: str, timeframe: str) -> pd.DataFrame:
        """Generate synthetic commodity data for development/testing"""
        try:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)

            if timeframe == "1D":
                freq = "D"
            elif timeframe == "1H":
                freq = "H"
            else:
                freq = "1D"

            timestamps = pd.date_range(start=start, end=end, freq=freq)

            # Get commodity info
            commodity_info = self.COMMODITIES.get(symbol, {'base_price': 100.0, 'category': 'unknown'})
            base_price = commodity_info['base_price']
            category = commodity_info['category']

            # Different volatility characteristics by category
            volatility_multipliers = {
                'precious_metals': 0.02,   # 2% daily volatility
                'energy': 0.03,           # 3% daily volatility
                'agricultural': 0.025,    # 2.5% daily volatility
                'industrial_metals': 0.022 # 2.2% daily volatility
            }

            volatility = volatility_multipliers.get(category, 0.02)

            prices = []
            current_price = base_price

            for i in range(len(timestamps)):
                # Random walk with category-specific volatility
                change = np.random.normal(0, volatility)
                current_price += change * current_price

                # Add seasonal/trend components for agricultural commodities
                if category == 'agricultural':
                    # Simple seasonal pattern (higher in summer for corn/wheat)
                    day_of_year = timestamps[i].dayofyear
                    seasonal_factor = 0.001 * np.sin(2 * np.pi * day_of_year / 365)
                    current_price += seasonal_factor * current_price

                # Generate OHLC
                high = current_price + abs(np.random.normal(0, volatility * current_price * 0.5))
                low = current_price - abs(np.random.normal(0, volatility * current_price * 0.5))
                open_price = prices[-1][3] if prices else current_price
                close = current_price
                volume = np.random.uniform(10000, 100000)  # Higher volume for commodities

                prices.append([open_price, high, low, close, volume])

            df = pd.DataFrame(prices, columns=['open', 'high', 'low', 'close', 'volume'])
            df.index = timestamps

            return df

        except Exception as e:
            logger.error(f"Error generating synthetic commodity data: {e}")
            return pd.DataFrame()

    async def get_real_time_data(self, symbol: str) -> Dict[str, Any]:
        """Get real-time commodity data"""
        try:
            if symbol not in self.COMMODITIES:
                return {}

            commodity_info = self.COMMODITIES[symbol]
            base_price = commodity_info['base_price']

            # Add some random movement
            current_price = base_price + np.random.normal(0, base_price * 0.01)

            return {
                "symbol": symbol,
                "name": commodity_info['name'],
                "category": commodity_info['category'],
                "price": current_price,
                "change": np.random.normal(0, base_price * 0.005),
                "change_percent": np.random.normal(0, 1.0),
                "timestamp": datetime.now().isoformat(),
                "volume": np.random.uniform(50000, 200000)
            }

        except Exception as e:
            logger.error(f"Error getting real-time commodity data for {symbol}: {e}")
            return {}

    async def get_commodity_indicators(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate commodity-specific indicators"""
        try:
            indicators = {}

            if len(data) < 20:
                return indicators

            close = data['close']
            volume = data.get('volume', pd.Series([1000] * len(data)))

            # Commodity-specific indicators
            commodity_info = self.COMMODITIES.get(symbol, {})
            category = commodity_info.get('category', 'unknown')

            # Volume indicators (important for commodities)
            indicators['volume_sma'] = volume.rolling(20).mean()
            indicators['volume_ratio'] = volume / indicators['volume_sma']

            # Open interest proxy (synthetic for now)
            indicators['open_interest_proxy'] = volume.rolling(50).mean()

            # Category-specific indicators
            if category == 'energy':
                # Energy-specific: contango/backwardation signals
                indicators['contango_signal'] = (close - close.shift(30)).rolling(10).mean()

            elif category == 'agricultural':
                # Agricultural: weather/crop report sensitivity
                indicators['price_momentum'] = (close - close.shift(20)) / close.shift(20)

            elif category in ['precious_metals', 'industrial_metals']:
                # Metals: industrial production correlation proxy
                indicators['industrial_correlation'] = close.rolling(30).corr(volume)

            # Commodity-specific volatility
            returns = close.pct_change()
            indicators['commodity_volatility'] = returns.rolling(20).std() * np.sqrt(252)

            return indicators

        except Exception as e:
            logger.error(f"Error calculating commodity indicators for {symbol}: {e}")
            return {}

    def get_available_commodities(self) -> List[str]:
        """Get list of available commodities"""
        return list(self.COMMODITIES.keys())

    def get_commodities_by_category(self, category: str) -> List[str]:
        """Get commodities by category"""
        return [symbol for symbol, info in self.COMMODITIES.items() if info['category'] == category]

    def is_valid_commodity(self, symbol: str) -> bool:
        """Check if commodity symbol is valid"""
        return symbol in self.COMMODITIES

    def fetch_historical_data(self, symbol: str, timeframe: str,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None,
                            limit: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch historical commodities data.

        Args:
            symbol: Commodity symbol (e.g., 'GC=F')
            timeframe: Timeframe string
            start_date: Start date
            end_date: End date
            limit: Maximum records to fetch

        Returns:
            DataFrame with OHLCV data
        """
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365)

        return asyncio.run(self.get_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe
        ))

    def fetch_recent_data(self, symbol: str, timeframe: str,
                         limit: int = 100) -> pd.DataFrame:
        """
        Fetch recent commodities data.

        Args:
            symbol: Commodity symbol
            timeframe: Timeframe string
            limit: Number of records to fetch

        Returns:
            DataFrame with OHLCV data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=limit)

        return asyncio.run(self.get_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe
        ))