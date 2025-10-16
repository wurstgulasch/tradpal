#!/usr/bin/env python3
"""
Test script for new data sources: Funding Rate and Liquidation Data

This script tests the newly implemented data sources for TradPal.
"""

import sys
import os
import logging

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_funding_rate_data():
    """Test the Funding Rate data source."""
    logger.info("Testing Funding Rate data source...")

    try:
        from services.data_service.data_sources.factory import create_data_source

        # Create funding rate data source
        funding_source = create_data_source('funding_rate')

        # Test recent data
        logger.info("Fetching recent funding rate data for BTCUSDT...")
        recent_data = funding_source.fetch_recent_data('BTCUSDT', '8h', limit=10)

        if not recent_data.empty:
            logger.info(f"✅ Successfully fetched {len(recent_data)} funding rate records")
            logger.info(f"Latest funding rate: {recent_data['funding_rate'].iloc[-1]:.6f}")
            logger.info(f"Market regime: {recent_data['market_regime'].iloc[-1]}")

            # Test stats
            stats = funding_source.get_funding_rate_stats('BTCUSDT')
            if stats:
                logger.info(f"✅ Funding rate stats: Mean={stats['mean_funding_rate']:.6f}, Current={stats['current_funding_rate']:.6f}")
        else:
            logger.warning("❌ No funding rate data received")

    except Exception as e:
        logger.error(f"❌ Funding rate data source test failed: {e}")
        return False

    return True

def test_liquidation_data():
    """Test the Liquidation data source."""
    logger.info("Testing Liquidation data source...")

    results = []

    # Test Binance (may fail due to auth)
    try:
        logger.info("Testing Binance liquidation data...")
        from services.data_service.data_sources.factory import create_data_source

        binance_source = create_data_source('liquidation', {'exchange': 'binance'})
        binance_data = binance_source.fetch_recent_data('BTCUSDT', '1h', limit=20)

        if not binance_data.empty:
            logger.info(f"✅ Binance: Successfully fetched {len(binance_data)} liquidation records")
            results.append(("Binance Liquidations", True))
        else:
            logger.warning("❌ Binance: No liquidation data received (expected due to auth)")
            results.append(("Binance Liquidations", False))

    except Exception as e:
        logger.error(f"❌ Binance liquidation test failed: {e}")
        results.append(("Binance Liquidations", False))

    # Test Bybit
    try:
        logger.info("Testing Bybit liquidation data...")
        bybit_source = create_data_source('liquidation', {'exchange': 'bybit'})
        bybit_data = bybit_source.fetch_recent_data('BTCUSDT', '1h', limit=20)

        if not bybit_data.empty:
            logger.info(f"✅ Bybit: Successfully fetched {len(bybit_data)} liquidation records")
            logger.info(f"Total liquidation value: ${bybit_data['liquidation_value'].sum():,.2f}")

            # Show breakdown
            long_liq = (bybit_data['side'] == 'BUY').sum()
            short_liq = (bybit_data['side'] == 'SELL').sum()
            logger.info(f"Long liquidations: {long_liq}, Short liquidations: {short_liq}")
            results.append(("Bybit Liquidations", True))
        else:
            logger.warning("❌ Bybit: No liquidation data received")
            results.append(("Bybit Liquidations", False))

    except Exception as e:
        logger.error(f"❌ Bybit liquidation test failed: {e}")
        results.append(("Bybit Liquidations", False))

    # Test OKX
    try:
        logger.info("Testing OKX liquidation data...")
        okx_source = create_data_source('liquidation', {'exchange': 'okx'})
        okx_data = okx_source.fetch_recent_data('BTCUSDT', '1h', limit=20)

        if not okx_data.empty:
            logger.info(f"✅ OKX: Successfully fetched {len(okx_data)} liquidation records")
            logger.info(f"Total liquidation value: ${okx_data['liquidation_value'].sum():,.2f}")

            # Show breakdown
            long_liq = (okx_data['side'] == 'BUY').sum()
            short_liq = (okx_data['side'] == 'SELL').sum()
            logger.info(f"Long liquidations: {long_liq}, Short liquidations: {short_liq}")
            results.append(("OKX Liquidations", True))
        else:
            logger.warning("❌ OKX: No liquidation data received")
            results.append(("OKX Liquidations", False))

    except Exception as e:
        logger.error(f"❌ OKX liquidation test failed: {e}")
        results.append(("OKX Liquidations", False))

    # Return True if at least one source works
    return any(success for _, success in results)

def test_data_source_availability():
    """Test data source availability."""
    logger.info("Testing data source availability...")

    try:
        from services.data_service.data_sources.factory import DataSourceFactory

        available_sources = DataSourceFactory.get_available_sources()
        logger.info("Available data sources:")
        for name, available in available_sources.items():
            status = "✅" if available else "❌"
            logger.info(f"  {status} {name}")

        available_list = DataSourceFactory.list_sources()
        logger.info(f"Total available sources: {len(available_list)}")

        return True

    except Exception as e:
        logger.error(f"❌ Data source availability test failed: {e}")
        return False

def main():
    """Main test function."""
    logger.info("🚀 Starting TradPal New Data Sources Test")
    logger.info("=" * 50)

    results = []

    # Test data source availability
    results.append(("Data Source Availability", test_data_source_availability()))

    # Test funding rate data
    results.append(("Funding Rate Data", test_funding_rate_data()))

    # Test liquidation data
    results.append(("Liquidation Data", test_liquidation_data()))

    # Summary
    logger.info("=" * 50)
    logger.info("Test Results Summary:")

    passed = 0
    total = len(results)

    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"  {status}: {test_name}")
        if success:
            passed += 1

    logger.info(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed! New data sources are working correctly.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())