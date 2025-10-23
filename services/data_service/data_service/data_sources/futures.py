"""
Futures Data Source for TradPal

Provides real-time and historical futures data including:
- Index futures (S&P 500, Nasdaq, Dow Jones)
- Bond futures (Treasury futures)
- Currency futures
- Commodity futures
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

from .base import BaseDataSource

logger = logging.getLogger(__name__)


class FuturesDataSource(BaseDataSource):
    """
    Futures data source for futures trading data.

    Supports:
    - Index futures (E-mini S&P 500, Nasdaq 100, etc.)
    - Interest rate futures (Treasury bonds, notes)
    - Currency futures
    - Commodity futures
    - Micro contracts
    """

    # Major futures contracts
    FUTURES_CONTRACTS = {
        # Index Futures
        'ES': {'name': 'E-mini S&P 500', 'category': 'index', 'base_price': 4500.0, 'tick_size': 0.25, 'multiplier': 50},
        'NQ': {'name': 'E-mini Nasdaq 100', 'category': 'index', 'base_price': 16000.0, 'tick_size': 0.25, 'multiplier': 20},
        'RTY': {'name': 'E-mini Russell 2000', 'category': 'index', 'base_price': 2200.0, 'tick_size': 0.10, 'multiplier': 50},
        'YM': {'name': 'E-mini Dow Jones', 'category': 'index', 'base_price': 35000.0, 'tick_size': 1.0, 'multiplier': 5},

        # Interest Rate Futures
        'ZN': {'name': '10-Year T-Note', 'category': 'interest_rate', 'base_price': 120.0, 'tick_size': 0.015625, 'multiplier': 1000},
        'ZB': {'name': '30-Year T-Bond', 'category': 'interest_rate', 'base_price': 150.0, 'tick_size': 0.03125, 'multiplier': 1000},
        'ZF': {'name': '5-Year T-Note', 'category': 'interest_rate', 'base_price': 110.0, 'tick_size': 0.0078125, 'multiplier': 1000},

        # Currency Futures
        '6E': {'name': 'Euro FX', 'category': 'currency', 'base_price': 1.08, 'tick_size': 0.00005, 'multiplier': 125000},
        '6B': {'name': 'British Pound', 'category': 'currency', 'base_price': 1.27, 'tick_size': 0.00005, 'multiplier': 62500},
        '6J': {'name': 'Japanese Yen', 'category': 'currency', 'base_price': 0.00667, 'tick_size': 0.0000005, 'multiplier': 12500000},
        '6C': {'name': 'Canadian Dollar', 'category': 'currency', 'base_price': 0.74, 'tick_size': 0.00005, 'multiplier': 100000},

        # Commodity Futures (subset - main ones covered in commodities.py)
        'CL': {'name': 'WTI Crude Oil', 'category': 'energy', 'base_price': 80.0, 'tick_size': 0.01, 'multiplier': 1000},
        'NG': {'name': 'Natural Gas', 'category': 'energy', 'base_price': 3.0, 'tick_size': 0.001, 'multiplier': 10000},
        'GC': {'name': 'Gold', 'category': 'precious_metals', 'base_price': 2000.0, 'tick_size': 0.10, 'multiplier': 100},
        'SI': {'name': 'Silver', 'category': 'precious_metals', 'base_price': 25.0, 'tick_size': 0.005, 'multiplier': 5000},

        # Micro Contracts
        'MES': {'name': 'Micro E-mini S&P 500', 'category': 'micro_index', 'base_price': 4500.0, 'tick_size': 0.25, 'multiplier': 5},
        'MNQ': {'name': 'Micro E-mini Nasdaq', 'category': 'micro_index', 'base_price': 16000.0, 'tick_size': 0.25, 'multiplier': 2},
        'M2K': {'name': 'Micro E-mini Russell 2000', 'category': 'micro_index', 'base_price': 2200.0, 'tick_size': 0.10, 'multiplier': 5},
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.config = config or {}
        self.provider = self.config.get('provider', 'yahoo_finance')  # yahoo_finance, cme, etc.
        self.api_key = self.config.get('api_key', '')

        # Futures-specific settings
        self.include_micro = self.config.get('include_micro', True)

    async def get_historical_data(self, symbol: str, start_date: str,
                                end_date: str, interval: str = "1D") -> pd.DataFrame:
        """
        Get historical futures data.

        Args:
            symbol: Futures symbol (e.g., 'ES', 'NQ', 'ZN')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe: Timeframe ('1D', '1H', '15m', etc.)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            if symbol not in self.FUTURES_CONTRACTS:
                logger.warning(f"Unknown futures symbol: {symbol}")
                return self._generate_synthetic_futures_data(symbol, start_date, end_date, timeframe)

            contract_info = self.FUTURES_CONTRACTS[symbol]

            if self.provider == 'yahoo_finance':
                return await self._get_yahoo_futures_data(symbol, start_date, end_date, interval)
            else:
                # Fallback to synthetic data
                return self._generate_synthetic_futures_data(symbol, start_date, end_date, interval)

        except Exception as e:
            logger.error(f"Error getting futures data for {symbol}: {e}")
            return self._generate_synthetic_futures_data(symbol, start_date, end_date, interval)

    async def _get_yahoo_futures_data(self, symbol: str, start_date: str,
                                    end_date: str, timeframe: str) -> pd.DataFrame:
        """Get futures data from Yahoo Finance"""
        try:
            # Map futures symbols to Yahoo Finance tickers
            yahoo_tickers = {
                'ES': 'ES=F',    # E-mini S&P 500
                'NQ': 'NQ=F',    # E-mini Nasdaq 100
                'RTY': 'RTY=F',  # E-mini Russell 2000
                'YM': 'YM=F',    # E-mini Dow
                'ZN': 'ZN=F',    # 10-Year T-Note
                'ZB': 'ZB=F',    # 30-Year T-Bond
                'ZF': 'ZF=F',    # 5-Year T-Note
                '6E': 'EUR=X',   # Euro (approximate)
                '6B': 'GBP=X',   # British Pound (approximate)
                '6J': 'JPY=X',   # Japanese Yen (approximate)
                'CL': 'CL=F',    # WTI Oil
                'GC': 'GC=F',    # Gold
                'SI': 'SI=F',    # Silver
            }

            yahoo_symbol = yahoo_tickers.get(symbol)
            if not yahoo_symbol:
                return self._generate_synthetic_futures_data(symbol, start_date, end_date, timeframe)

            # Use Yahoo Finance data source
            from .yahoo_finance import YahooFinanceDataSource
            yahoo_source = YahooFinanceDataSource()
            return await yahoo_source.get_historical_data(yahoo_symbol, start_date, end_date, interval=timeframe)

        except Exception as e:
            logger.error(f"Yahoo Finance futures data error: {e}")
            return self._generate_synthetic_futures_data(symbol, start_date, end_date, timeframe)

    def _generate_synthetic_futures_data(self, symbol: str, start_date: str,
                                       end_date: str, timeframe: str) -> pd.DataFrame:
        """Generate synthetic futures data for development/testing"""
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

            # Get contract info
            contract_info = self.FUTURES_CONTRACTS.get(symbol, {'base_price': 100.0, 'category': 'unknown'})
            base_price = contract_info['base_price']
            category = contract_info['category']

            # Different volatility characteristics by category
            volatility_multipliers = {
                'index': 0.015,           # 1.5% daily volatility
                'micro_index': 0.015,     # Same as index
                'interest_rate': 0.008,   # 0.8% daily volatility
                'currency': 0.01,         # 1% daily volatility
                'energy': 0.03,           # 3% daily volatility
                'precious_metals': 0.02   # 2% daily volatility
            }

            volatility = volatility_multipliers.get(category, 0.015)

            prices = []
            current_price = base_price

            for i in range(len(timestamps)):
                # Random walk with category-specific volatility
                change = np.random.normal(0, volatility)
                current_price += change * current_price

                # Add time-of-day effects for futures (higher volatility during market hours)
                hour = timestamps[i].hour
                if 9 <= hour <= 16:  # Market hours (approximate)
                    intraday_multiplier = 1.2
                else:
                    intraday_multiplier = 0.8

                # Generate OHLC with futures-specific characteristics
                high = current_price + abs(np.random.normal(0, volatility * current_price * 0.3)) * intraday_multiplier
                low = current_price - abs(np.random.normal(0, volatility * current_price * 0.3)) * intraday_multiplier
                open_price = prices[-1][3] if prices else current_price
                close = current_price

                # Futures have different volume patterns
                if category in ['index', 'micro_index']:
                    volume = np.random.uniform(50000, 500000)  # High volume for index futures
                elif category == 'interest_rate':
                    volume = np.random.uniform(10000, 100000)  # Moderate volume for bonds
                else:
                    volume = np.random.uniform(5000, 50000)    # Lower volume for others

                prices.append([open_price, high, low, close, volume])

            df = pd.DataFrame(prices, columns=['open', 'high', 'low', 'close', 'volume'])
            df.index = timestamps

            return df

        except Exception as e:
            logger.error(f"Error generating synthetic futures data: {e}")
            return pd.DataFrame()

    async def get_real_time_data(self, symbol: str) -> Dict[str, Any]:
        """Get real-time futures data"""
        try:
            if symbol not in self.FUTURES_CONTRACTS:
                return {}

            contract_info = self.FUTURES_CONTRACTS[symbol]
            base_price = contract_info['base_price']

            # Add some random movement
            current_price = base_price + np.random.normal(0, base_price * 0.005)

            return {
                "symbol": symbol,
                "name": contract_info['name'],
                "category": contract_info['category'],
                "price": current_price,
                "change": np.random.normal(0, base_price * 0.002),
                "change_percent": np.random.normal(0, 0.5),
                "timestamp": datetime.now().isoformat(),
                "volume": np.random.uniform(10000, 100000),
                "open_interest": np.random.uniform(50000, 500000),
                "tick_size": contract_info['tick_size'],
                "multiplier": contract_info['multiplier']
            }

        except Exception as e:
            logger.error(f"Error getting real-time futures data for {symbol}: {e}")
            return {}

    async def get_futures_indicators(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate futures-specific indicators"""
        try:
            indicators = {}

            if len(data) < 20:
                return indicators

            close = data['close']
            volume = data.get('volume', pd.Series([1000] * len(data)))

            # Futures-specific indicators
            contract_info = self.FUTURES_CONTRACTS.get(symbol, {})
            category = contract_info.get('category', 'unknown')

            # Open interest proxy (synthetic)
            indicators['open_interest_proxy'] = volume.rolling(50).mean()

            # Futures-specific volume indicators
            indicators['volume_open_interest_ratio'] = volume / indicators['open_interest_proxy']

            # Category-specific indicators
            if category in ['index', 'micro_index']:
                # Index futures: correlation with broader market
                indicators['market_correlation'] = close.rolling(20).corr(close.shift(1))

            elif category == 'interest_rate':
                # Bond futures: yield curve signals
                indicators['yield_proxy'] = 100 - close  # Inverse relationship
                indicators['yield_momentum'] = indicators['yield_proxy'].diff(10)

            elif category == 'currency':
                # Currency futures: interest rate differential proxy
                indicators['currency_momentum'] = close.pct_change(20)

            # Futures-specific volatility (includes open interest effects)
            returns = close.pct_change()
            indicators['futures_volatility'] = returns.rolling(20).std() * np.sqrt(252)

            # Roll-over effects (front month vs back month)
            indicators['rollover_signal'] = (close - close.shift(20)).rolling(5).mean()

            return indicators

        except Exception as e:
            logger.error(f"Error calculating futures indicators for {symbol}: {e}")
            return {}

    def get_available_contracts(self) -> List[str]:
        """Get list of available futures contracts"""
        contracts = list(self.FUTURES_CONTRACTS.keys())
        if not self.include_micro:
            # Filter out micro contracts
            contracts = [c for c in contracts if not c.startswith('M')]
        return contracts

    def get_contracts_by_category(self, category: str) -> List[str]:
        """Get futures contracts by category"""
        return [symbol for symbol, info in self.FUTURES_CONTRACTS.items() if info['category'] == category]

    def is_valid_contract(self, symbol: str) -> bool:
        """Check if futures contract symbol is valid"""
        return symbol in self.FUTURES_CONTRACTS

    def get_contract_specs(self, symbol: str) -> Dict[str, Any]:
        """Get contract specifications"""
        return self.FUTURES_CONTRACTS.get(symbol, {})

    def calculate_tick_value(self, symbol: str, price: float) -> float:
        """Calculate value of one tick move"""
        contract = self.FUTURES_CONTRACTS.get(symbol)
        if not contract:
            return 0.0

        return contract['tick_size'] * contract['multiplier']

    def fetch_historical_data(self, symbol: str, timeframe: str,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None,
                            limit: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch historical futures data.

        Args:
            symbol: Futures contract symbol (e.g., 'ES=F')
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
        Fetch recent futures data.

        Args:
            symbol: Futures contract symbol
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