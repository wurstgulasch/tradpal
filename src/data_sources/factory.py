"""
Data Source Factory

Creates and returns the appropriate data source implementation based on configuration.
"""

from typing import Dict, Any, Optional
import logging
from config.settings import DATA_SOURCE, DATA_SOURCE_CONFIG

from . import DataSource
from .yahoo_finance import YahooFinanceDataSource
from .ccxt_source import CCXTDataSource
from .funding_rate_source import FundingRateDataSource

logger = logging.getLogger(__name__)

def create_data_source(data_source_name: Optional[str] = None) -> DataSource:
    """
    Factory function to create data source instances.

    Args:
        data_source_name: Name of data source to create. If None, uses DATA_SOURCE from config.

    Returns:
        DataSource instance

    Raises:
        ValueError: If data source is not supported
    """
    if data_source_name is None:
        data_source_name = DATA_SOURCE

    config = DATA_SOURCE_CONFIG.get(data_source_name)
    if not config:
        raise ValueError(f"No configuration found for data source: {data_source_name}")

    logger.info(f"Creating data source: {data_source_name}")

    if data_source_name == 'yahoo_finance':
        return YahooFinanceDataSource(config)
    elif data_source_name == 'ccxt':
        return CCXTDataSource(config)
    elif data_source_name == 'funding_rate':
        return FundingRateDataSource(config)
    elif data_source_name == 'alpha_vantage':
        # Placeholder for future implementation
        raise NotImplementedError("Alpha Vantage data source not yet implemented")
    elif data_source_name == 'polygon':
        # Placeholder for future implementation
        raise NotImplementedError("Polygon.io data source not yet implemented")
    else:
        raise ValueError(f"Unsupported data source: {data_source_name}")

def get_available_data_sources() -> Dict[str, str]:
    """
    Get dictionary of available data sources with descriptions.

    Returns:
        Dict mapping data source names to descriptions
    """
    return {
        'yahoo_finance': 'Yahoo Finance - Best for historical data and traditional assets',
        'ccxt': 'CCXT - Best for crypto exchanges with real-time data',
        'funding_rate': 'Funding Rate - Specialized for perpetual futures funding rate analysis',
        'alpha_vantage': 'Alpha Vantage - Premium financial data API (not implemented)',
        'polygon': 'Polygon.io - High-performance financial market data (not implemented)'
    }