#!/usr/bin/env python3
"""
Test script for multi-market data sources
"""

import asyncio
import logging
from datetime import datetime, timedelta
from services.data_service.data_service.data_sources.factory import DataSourceFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_data_sources():
    """Test the new multi-market data sources"""
    logger.info("Testing multi-market data sources...")

    # Test configuration
    config = {
        'cache_dir': 'cache',
        'timeout': 30,
        'retries': 3
    }

    # Test each data source
    test_cases = [
        ('forex', 'EURUSD=X'),
        ('commodities', 'GC=F'),  # Gold futures
        ('futures', 'ES=F'),      # E-mini S&P 500
    ]

    for source_name, symbol in test_cases:
        try:
            logger.info(f"Testing {source_name} data source with symbol {symbol}")

            # Create data source
            data_source = DataSourceFactory.create_data_source(source_name, config)

            # Test data fetching
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)

            data = await data_source.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval='1d'
            )

            if data is not None and not data.empty:
                logger.info(f"✓ {source_name}: Successfully fetched {len(data)} records")
                logger.info(f"  Sample data: {data.head(1).to_dict()}")
            else:
                logger.warning(f"✗ {source_name}: No data returned")

        except Exception as e:
            logger.error(f"✗ {source_name}: Error - {e}")

    logger.info("Multi-market data source testing completed")

if __name__ == "__main__":
    asyncio.run(test_data_sources())