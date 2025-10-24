"""
Yahoo Finance Data Source for TradPal Indicator System

This module provides data fetching from Yahoo Finance using the yfinance library.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None

from .base import BaseDataSource

logger = logging.getLogger(__name__)

class YahooFinanceDataSource(BaseDataSource):
    """
    Data source for fetching financial data from Yahoo Finance.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Yahoo Finance data source.

        Args:
            config: Optional configuration dictionary
        """
        super().__init__("Yahoo Finance", config)

        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance library is required for Yahoo Finance data source")

        # Default configuration
        self.config.setdefault('timeout', 30)
        self.config.setdefault('auto_adjust', True)
        self.config.setdefault('prepost', False)

    def fetch_historical_data(self, symbol: str, timeframe: str,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None,
                            limit: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from Yahoo Finance.

        Args:
            symbol: Trading symbol (e.g., 'BTC-USD' for Bitcoin)
            timeframe: Timeframe string ('1d', '1h', '1wk', etc.)
            start_date: Start date for historical data
            end_date: End date for historical data
            limit: Maximum number of candles (ignored by yfinance)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Convert symbol format if needed (BTC/USDT -> BTC-USD)
            yahoo_symbol = self._convert_symbol_format(symbol)

            # Set default dates if not provided
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                start_date = end_date - timedelta(days=365)  # Default 1 year

            # Map timeframe to yfinance interval
            interval = self._map_timeframe(timeframe)

            self.logger.info(f"Fetching {yahoo_symbol} {interval} data from {start_date.date()} to {end_date.date()}")

            # Create ticker and fetch data
            ticker = yf.Ticker(yahoo_symbol)

            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                timeout=self.config['timeout'],
                auto_adjust=self.config['auto_adjust'],
                prepost=self.config['prepost']
            )

            if df.empty:
                self.logger.warning(f"No data received for {yahoo_symbol}")
                return pd.DataFrame()

            # Convert to TradPal format
            df = self._convert_to_tradpal_format(df)

            # Apply limit if specified
            if limit and len(df) > limit:
                df = df.tail(limit)

            # Validate data
            self.validate_data(df)

            self.logger.info(f"Successfully fetched {len(df)} candles for {yahoo_symbol}")
            return df

        except Exception as e:
            self.logger.error(f"Error fetching data from Yahoo Finance: {e}")
            raise

    def fetch_recent_data(self, symbol: str, timeframe: str,
                         limit: int = 100) -> pd.DataFrame:
        """
        Fetch recent OHLCV data from Yahoo Finance.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            limit: Number of recent candles to fetch

        Returns:
            DataFrame with OHLCV data
        """
        # For recent data, fetch from a recent start date
        end_date = datetime.now()
        start_date = end_date - timedelta(days=limit * self._timeframe_to_days(timeframe))

        return self.fetch_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )

    def _convert_symbol_format(self, symbol: str) -> str:
        """
        Convert symbol format to Yahoo Finance format.

        Args:
            symbol: Input symbol (e.g., 'BTC/USDT')

        Returns:
            Yahoo Finance symbol (e.g., 'BTC-USD')
        """
        # Replace '/' with '-' for Yahoo Finance format
        return symbol.replace('/', '-')

    def _map_timeframe(self, timeframe: str) -> str:
        """
        Map TradPal timeframe to Yahoo Finance interval.

        Args:
            timeframe: TradPal timeframe string

        Returns:
            Yahoo Finance interval string
        """
        # Yahoo Finance intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        timeframe_map = {
            '1m': '1m',
            '2m': '2m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '60m',  # Yahoo uses '60m' for 1h
            '4h': '1h',
            '1d': '1d',
            '5d': '5d',
            '1w': '1wk',
            '1M': '1mo'
        }

        return timeframe_map.get(timeframe, '1d')  # Default to daily

    def _timeframe_to_days(self, timeframe: str) -> int:
        """
        Convert timeframe to approximate days for date calculation.

        Args:
            timeframe: Timeframe string

        Returns:
            Approximate days
        """
        timeframe_days = {
            '1m': 1/1440,  # 1 minute
            '5m': 5/1440,
            '15m': 15/1440,
            '30m': 30/1440,
            '1h': 1/24,
            '4h': 4/24,
            '1d': 1,
            '5d': 5,
            '1w': 7,
            '1M': 30
        }

        return timeframe_days.get(timeframe, 1)

    def _convert_to_tradpal_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert Yahoo Finance DataFrame to TradPal OHLCV format.

        Args:
            df: Yahoo Finance DataFrame

        Returns:
            DataFrame in TradPal format
        """
        # Yahoo Finance columns: Open, High, Low, Close, Volume, Dividends, Stock Splits
        # TradPal format: open, high, low, close, volume

        # Select only OHLCV columns and rename to lowercase
        columns_map = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }

        # Keep only the columns we need
        df = df[list(columns_map.keys())].copy()

        # Rename columns
        df = df.rename(columns=columns_map)

        # Ensure timestamp index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        return df