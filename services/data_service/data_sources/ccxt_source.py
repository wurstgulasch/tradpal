"""
CCXT Data Source for TradPal Indicator System

This module provides data fetching from cryptocurrency exchanges using the CCXT library.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import logging
import time

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    ccxt = None

from .base import BaseDataSource

logger = logging.getLogger(__name__)

class CCXTDataSource(BaseDataSource):
    """
    Data source for fetching data from cryptocurrency exchanges via CCXT.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize CCXT data source.

        Args:
            config: Optional configuration dictionary
        """
        super().__init__("CCXT", config)

        if not CCXT_AVAILABLE:
            raise ImportError("ccxt library is required for CCXT data source")

        # Default configuration
        self.config.setdefault('exchange', 'binance')  # Default exchange
        self.config.setdefault('timeout', 30000)  # 30 seconds
        self.config.setdefault('enableRateLimit', True)

        # Initialize exchange
        self.exchange = None
        self._init_exchange()

    def _init_exchange(self):
        """Initialize the CCXT exchange instance."""
        try:
            exchange_class = getattr(ccxt, self.config['exchange'])
            self.exchange = exchange_class({
                'timeout': self.config['timeout'],
                'enableRateLimit': self.config['enableRateLimit'],
            })

            # Load markets
            self.exchange.load_markets()
            self.logger.info(f"Initialized {self.config['exchange']} exchange")

        except Exception as e:
            self.logger.error(f"Failed to initialize {self.config['exchange']} exchange: {e}")
            raise

    def fetch_historical_data(self, symbol: str, timeframe: str,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None,
                            limit: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from CCXT exchange.

        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe string ('1m', '1h', '1d')
            start_date: Start date for historical data
            end_date: End date for historical data
            limit: Maximum number of candles

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Set defaults
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                # Default to 1000 candles back
                timeframe_minutes = self._timeframe_to_minutes(timeframe)
                start_date = end_date - timedelta(minutes=timeframe_minutes * (limit or 1000))

            if limit is None:
                limit = 1000  # CCXT default

            self.logger.info(f"Fetching {symbol} {timeframe} data from {start_date} to {end_date}")

            # Convert dates to timestamps
            since = int(start_date.timestamp() * 1000)

            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=limit
            )

            if not ohlcv:
                self.logger.warning(f"No data received for {symbol}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = self._ohlcv_to_dataframe(ohlcv)

            # Filter by end_date if specified
            if end_date:
                df = df[df.index <= end_date]

            # Validate data
            self.validate_data(df)

            self.logger.info(f"Successfully fetched {len(df)} candles for {symbol}")
            return df

        except Exception as e:
            self.logger.error(f"Error fetching data from CCXT: {e}")
            raise

    def fetch_recent_data(self, symbol: str, timeframe: str,
                         limit: int = 100) -> pd.DataFrame:
        """
        Fetch recent OHLCV data from CCXT exchange.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            limit: Number of recent candles to fetch

        Returns:
            DataFrame with OHLCV data
        """
        return self.fetch_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit
        )

    def _ohlcv_to_dataframe(self, ohlcv: List[List]) -> pd.DataFrame:
        """
        Convert CCXT OHLCV list to pandas DataFrame.

        Args:
            ohlcv: OHLCV data from CCXT [timestamp, open, high, low, close, volume]

        Returns:
            DataFrame with OHLCV data
        """
        if not ohlcv:
            return pd.DataFrame()

        # Create DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # Convert timestamp to datetime index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')

        # Ensure numeric types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """
        Convert timeframe string to minutes.

        Args:
            timeframe: CCXT timeframe string

        Returns:
            Timeframe in minutes
        """
        timeframe_map = {
            '1m': 1,
            '3m': 3,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '2h': 120,
            '4h': 240,
            '6h': 360,
            '8h': 480,
            '12h': 720,
            '1d': 1440,
            '3d': 4320,
            '1w': 10080,
            '1M': 43200,
        }

        return timeframe_map.get(timeframe, 60)  # Default to 1h

    def get_available_symbols(self) -> List[str]:
        """
        Get list of available trading symbols on the exchange.

        Returns:
            List of trading symbols
        """
        if not self.exchange:
            return []

        try:
            return list(self.exchange.markets.keys())
        except Exception as e:
            self.logger.error(f"Error getting available symbols: {e}")
            return []

    def get_exchange_info(self) -> Dict[str, Any]:
        """
        Get information about the current exchange.

        Returns:
            Dictionary with exchange information
        """
        if not self.exchange:
            return {}

        return {
            'name': self.exchange.name,
            'id': self.exchange.id,
            'countries': getattr(self.exchange, 'countries', []),
            'urls': getattr(self.exchange, 'urls', {}),
            'has': getattr(self.exchange, 'has', {}),
            'timeframes': getattr(self.exchange, 'timeframes', {}),
            'markets_count': len(self.exchange.markets) if self.exchange.markets else 0
        }