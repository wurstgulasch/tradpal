#!/usr/bin/env python3
"""
Test Script for Volatility Data Source as Liquidation Proxy

Tests the new volatility indicators as alternative to liquidation data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from services.data_service.data_sources.volatility import VolatilityDataSource
from services.data_service.data_sources.liquidation import LiquidationDataSource
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)

def test_volatility_data_source():
    """Test the volatility data source directly."""
    print("� **Testing Volatility Data Source**")
    print("=" * 50)

    try:
        source = VolatilityDataSource()

        # Test current data
        current_data = source.fetch_current_data('BTCUSDT')
        print(f"✅ Current volatility data: {len(current_data)} records")
        if not current_data.empty:
            print(f"   Columns: {list(current_data.columns)}")
            print(f"   Sample values:")
            print(current_data.head(3))

        # Test historical data
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')

        historical_data = source.fetch_historical_data(
            symbol='BTCUSDT',
            timeframe='1d',
            start_date=start_date,
            end_date=end_date
        )

        print(f"✅ Historical volatility data: {len(historical_data)} records")
        if not historical_data.empty:
            print(f"   Date range: {historical_data.index.min()} to {historical_data.index.max()}")
            print(f"   Columns: {list(historical_data.columns)}")
            print(f"   Sample values:")
            print(historical_data.head(3))

        return historical_data

    except Exception as e:
        print(f"❌ Volatility data source failed: {e}")
        return None

def test_liquidation_fallback():
    """Test liquidation data source with fallback to volatility."""
    print("\n� **Testing Liquidation Fallback System**")
    print("=" * 50)

    try:
        source = LiquidationDataSource()

        # Test current data
        current_data = source.fetch_current_data('BTCUSDT')
        print(f"✅ Current liquidation data: {len(current_data)} records")
        if not current_data.empty:
            print(f"   Columns: {list(current_data.columns)}")
            print(f"   Sample values:")
            print(current_data.head(3))

        # Test historical data (should fallback to volatility)
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')

        historical_data = source.fetch_historical_data(
            symbol='BTCUSDT',
            timeframe='1d',
            start_date=start_date,
            end_date=end_date
        )

        print(f"✅ Historical liquidation data (with fallback): {len(historical_data)} records")
        if not historical_data.empty:
            print(f"   Date range: {historical_data.index.min()} to {historical_data.index.max()}")
            print(f"   Columns: {list(historical_data.columns)}")
            print(f"   Sample values:")
            print(historical_data.head(3))

        return historical_data

    except Exception as e:
        print(f"❌ Liquidation fallback failed: {e}")
        return None

def main():
    """Main test function."""
    print("🧪 **TradPal Volatility as Liquidation Proxy Test**")
    print("=" * 60)

    # Test volatility data source
    volatility_data = test_volatility_data_source()

    # Test liquidation fallback
    liquidation_data = test_liquidation_fallback()

    # Summary
    print("\n📋 **Test Summary**")
    print("=" * 30)

    success_count = 0
    total_tests = 2

    if volatility_data is not None and not volatility_data.empty:
        success_count += 1
        print("✅ Volatility Data Source: PASSED")
    else:
        print("❌ Volatility Data Source: FAILED")

    if liquidation_data is not None and not liquidation_data.empty:
        success_count += 1
        print("✅ Liquidation Fallback: PASSED")
    else:
        print("❌ Liquidation Fallback: FAILED")

    print(f"\n🎯 **Overall Result: {success_count}/{total_tests} tests passed**")

    if success_count == total_tests:
        print("🎉 Volatility indicators successfully implemented as liquidation proxy!")
    else:
        print("⚠️  Some features need attention")

if __name__ == "__main__":
    main()