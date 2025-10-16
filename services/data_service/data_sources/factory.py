"""
Data Source Factory for TradPal Indicator System

This module provides a factory function to create data source instances
based on configuration and availability.
"""

import logging
from typing import Optional, Dict, Any
from .base import BaseDataSource

logger = logging.getLogger(__name__)

# Import data source implementations
try:
    from .yahoo_finance import YahooFinanceDataSource
    YAHOO_AVAILABLE = True
except ImportError:
    YAHOO_AVAILABLE = False
    logger.warning("Yahoo Finance data source not available")

try:
    from .ccxt_source import CCXTDataSource
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    logger.warning("CCXT data source not available")

try:
    from .kaggle import KaggleDataSource
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
    logger.warning("Kaggle data source not available")

class DataSourceFactory:
    """
    Factory class for creating data source instances.
    """

    @staticmethod
    def create_data_source(name: Optional[str] = None,
                          config: Optional[Dict[str, Any]] = None) -> BaseDataSource:
        """
        Create a data source instance based on name and configuration.

        Args:
            name: Name of the data source ('yahoo_finance', 'ccxt', 'kaggle')
            config: Optional configuration dictionary

        Returns:
            Data source instance

        Raises:
            ValueError: If data source name is not recognized or not available
        """
        if name is None:
            # Auto-select best available data source
            return DataSourceFactory._auto_select_data_source(config)

        name = name.lower()

        if name == 'yahoo_finance' or name == 'yahoo':
            if not YAHOO_AVAILABLE:
                raise ValueError("Yahoo Finance data source is not available. Install yfinance.")
            return YahooFinanceDataSource(config)

        elif name == 'ccxt':
            if not CCXT_AVAILABLE:
                raise ValueError("CCXT data source is not available. Install ccxt.")
            return CCXTDataSource(config)

        elif name == 'kaggle':
            if not KAGGLE_AVAILABLE:
                raise ValueError("Kaggle data source is not available. Install kaggle and configure API key.")
            return KaggleDataSource(config)

        else:
            available_sources = []
            if YAHOO_AVAILABLE:
                available_sources.append('yahoo_finance')
            if CCXT_AVAILABLE:
                available_sources.append('ccxt')
            if KAGGLE_AVAILABLE:
                available_sources.append('kaggle')

            raise ValueError(f"Unknown data source '{name}'. Available sources: {available_sources}")

    @staticmethod
    def _auto_select_data_source(config: Optional[Dict[str, Any]] = None) -> BaseDataSource:
        """
        Auto-select the best available data source based on priority.

        Priority order: kaggle (for historical data), yahoo_finance, ccxt

        Args:
            config: Optional configuration dictionary

        Returns:
            Best available data source instance
        """
        # Priority order for auto-selection
        priority_order = [
            ('kaggle', KAGGLE_AVAILABLE),
            ('yahoo_finance', YAHOO_AVAILABLE),
            ('ccxt', CCXT_AVAILABLE)
        ]

        for source_name, available in priority_order:
            if available:
                logger.info(f"Auto-selected data source: {source_name}")
                return DataSourceFactory.create_data_source(source_name, config)

        raise ValueError("No data sources are available. Please install required dependencies.")

    @staticmethod
    def get_available_sources() -> Dict[str, bool]:
        """
        Get availability status of all data sources.

        Returns:
            Dictionary mapping source names to availability status
        """
        return {
            'yahoo_finance': YAHOO_AVAILABLE,
            'ccxt': CCXT_AVAILABLE,
            'kaggle': KAGGLE_AVAILABLE
        }

    @staticmethod
    def list_sources() -> list:
        """
        List all available data source names.

        Returns:
            List of available data source names
        """
        available = DataSourceFactory.get_available_sources()
        return [name for name, is_available in available.items() if is_available]

# Convenience function for backward compatibility
def create_data_source(name: Optional[str] = None,
                      config: Optional[Dict[str, Any]] = None) -> BaseDataSource:
    """
    Create a data source instance (convenience function).

    Args:
        name: Name of the data source
        config: Optional configuration dictionary

    Returns:
        Data source instance
    """
    return DataSourceFactory.create_data_source(name, config)