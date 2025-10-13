"""
Modular Data Sources for TradPal Indicator System

This module provides a unified interface for fetching market data from various sources
including exchanges (CCXT), Yahoo Finance, Alpha Vantage, and Polygon.io.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class DataSource(ABC):
    """Abstract base class for all data sources."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def fetch_historical_data(self, symbol: str, timeframe: str, start_date: datetime,
                            end_date: Optional[datetime] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch historical OHLCV data.

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe string (e.g., '1m', '1h', '1d')
            start_date: Start date for data
            end_date: End date for data (optional)
            limit: Maximum number of records to fetch (optional)

        Returns:
            DataFrame with OHLCV data and timestamp index
        """
        pass

    @abstractmethod
    def get_supported_timeframes(self) -> List[str]:
        """Return list of supported timeframes for this data source."""
        pass

    @abstractmethod
    def get_supported_symbols(self) -> List[str]:
        """Return list of supported symbols for this data source."""
        pass

    def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol is supported by this data source."""
        return symbol in self.get_supported_symbols()

    def validate_timeframe(self, timeframe: str) -> bool:
        """Validate if timeframe is supported by this data source."""
        return timeframe in self.get_supported_timeframes()