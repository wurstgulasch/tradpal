#!/usr/bin/env python3
"""
Test script for modular data sources.

Tests the new data source architecture with different providers.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def test_yahoo_finance():
    """Test Yahoo Finance data source."""
    print("🧪 Testing Yahoo Finance data source...")

    try:
        from src.data_sources.factory import create_data_source

        # Create Yahoo Finance data source
        data_source = create_data_source('yahoo_finance')

        # Test data fetch
        symbol = 'BTC/USDT'
        timeframe = '1d'
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()

        print(f"Fetching {symbol} {timeframe} data from {start_date.date()} to {end_date.date()}")

        df = data_source.fetch_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )

        if df.empty:
            print("❌ No data received from Yahoo Finance")
            return False

        print(f"✅ Received {len(df)} candles from Yahoo Finance")
        print(f"   Date range: {df.index.min()} to {df.index.max()}")
        print(f"   Sample data:\n{df.head(3)}")

        return True

    except Exception as e:
        print(f"❌ Yahoo Finance test failed: {e}")
        return False

def test_ccxt():
    """Test CCXT data source."""
    print("🧪 Testing CCXT data source...")

    try:
        from src.data_sources.factory import create_data_source

        # Create CCXT data source
        data_source = create_data_source('ccxt')

        # Test data fetch
        symbol = 'BTC/USDT'
        timeframe = '1h'
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()

        print(f"Fetching {symbol} {timeframe} data from {start_date.date()} to {end_date.date()}")

        df = data_source.fetch_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            limit=100
        )

        if df.empty:
            print("❌ No data received from CCXT")
            return False

        print(f"✅ Received {len(df)} candles from CCXT")
        print(f"   Date range: {df.index.min()} to {df.index.max()}")
        print(f"   Sample data:\n{df.head(3)}")

        return True

    except Exception as e:
        print(f"❌ CCXT test failed: {e}")
        return False

def test_data_fetcher_integration():
    """Test the updated data_fetcher.py with modular sources."""
    print("🧪 Testing data_fetcher.py integration...")

    try:
        from src.data_fetcher import fetch_historical_data, set_data_source

        # Test with Yahoo Finance
        print("Testing with Yahoo Finance...")
        set_data_source('yahoo_finance')

        df = fetch_historical_data(
            symbol='BTC/USDT',
            timeframe='1d',
            limit=30,
            show_progress=False
        )

        if df.empty:
            print("❌ No data from data_fetcher with Yahoo Finance")
            return False

        print(f"✅ data_fetcher works with Yahoo Finance: {len(df)} candles")

        # Test with CCXT
        print("Testing with CCXT...")
        set_data_source('ccxt')

        df = fetch_historical_data(
            symbol='BTC/USDT',
            timeframe='1h',
            limit=24,
            show_progress=False
        )

        if df.empty:
            print("❌ No data from data_fetcher with CCXT")
            return False

        print(f"✅ data_fetcher works with CCXT: {len(df)} candles")

        return True

    except Exception as e:
        print(f"❌ data_fetcher integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting modular data source tests...\n")

    results = []

    # Test individual data sources
    results.append(("Yahoo Finance", test_yahoo_finance()))
    print()
    results.append(("CCXT", test_ccxt()))
    print()

    # Test integration
    results.append(("Data Fetcher Integration", test_data_fetcher_integration()))
    print()

    # Summary
    print("📊 Test Results:")
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if success:
            passed += 1

    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("🎉 All tests passed! Modular data sources are working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())