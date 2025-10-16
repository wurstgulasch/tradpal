"""
Base Data Source Interface for TradPal Indicator System

This module defines the abstract base class for all data sources,
ensuring consistent API across different data providers.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from datetime import datetime
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class BaseDataSource(ABC):
    """
    Abstract base class for all data sources in the TradPal system.

    This class defines the standard interface that all data sources must implement,
    ensuring consistency and interchangeability between different data providers.
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data source.

        Args:
            name: Human-readable name of the data source
            config: Optional configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def fetch_historical_data(self, symbol: str, timeframe: str,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None,
                            limit: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a given symbol and timeframe.

        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe string (e.g., '1m', '1h', '1d')
            start_date: Start date for historical data
            end_date: End date for historical data
            limit: Maximum number of candles to fetch

        Returns:
            DataFrame with OHLCV data indexed by timestamp
        """
        pass

    @abstractmethod
    def fetch_recent_data(self, symbol: str, timeframe: str,
                         limit: int = 100) -> pd.DataFrame:
        """
        Fetch recent OHLCV data for a given symbol and timeframe.

        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe string (e.g., '1m', '1h', '1d')
            limit: Number of recent candles to fetch

        Returns:
            DataFrame with OHLCV data indexed by timestamp
        """
        pass

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate the structure and integrity of fetched data.

        Args:
            df: DataFrame to validate

        Returns:
            True if data is valid, False otherwise

        Raises:
            ValueError: If critical validation fails
        """
        if df.empty:
            self.logger.warning("DataFrame is empty")
            return False

        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Check for NaN values in critical columns
        for col in ['open', 'high', 'low', 'close']:
            if df[col].isna().any():
                raise ValueError(f"NaN values found in {col} column")

        # Check OHLC relationships
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['open'] < df['low']) | (df['open'] > df['high']) |
            (df['close'] < df['low']) | (df['close'] > df['high'])
        )

        if invalid_ohlc.any():
            invalid_count = invalid_ohlc.sum()
            self.logger.warning(f"Found {invalid_count} invalid OHLC relationships")
            # Don't raise error for now, just log warning

        return True

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this data source.

        Returns:
            Dictionary with source information
        """
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'config': self.config
        }

    def is_available(self) -> bool:
        """
        Check if this data source is available and functional.

        Returns:
            True if available, False otherwise
        """
        try:
            # Simple availability check - try to fetch a small amount of data
            df = self.fetch_recent_data('BTC/USDT', '1d', limit=1)
            return not df.empty
        except Exception as e:
            self.logger.error(f"Data source availability check failed: {e}")
            return False