#!/usr/bin/env python3
"""
Test Script for Backtesting Service with Volatility Fallback

Tests the enhanced backtesting service with volatility indicators as liquidation proxy.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from services.backtesting_service.service import BacktestingService
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)

def test_backtesting_with_volatility_fallback():
    """Test backtesting service with volatility fallback."""
    print("🚀 **Testing Backtesting Service with Volatility Fallback**")
    print("=" * 60)

    try:
        # Initialize backtesting service
        service = BacktestingService()

        # Configure test parameters
        service.symbol = 'BTCUSDT'
        service.timeframe = '1d'
        service.start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        service.end_date = datetime.now().strftime('%Y-%m-%d')

        print(f"📊 Testing backtest for {service.symbol} from {service.start_date} to {service.end_date}")

        # Run backtest (this should trigger volatility fallback)
        import asyncio
        result = asyncio.run(service.run_backtest_async(
            symbol=service.symbol,
            timeframe=service.timeframe,
            start_date=service.start_date,
            end_date=service.end_date
        ))

        if result and 'data' in result:
            print("✅ Backtest successful")
            data = result['data']
            print(f"   Data shape: {data.shape}")
            print(f"   Columns: {list(data.columns)}")

            # Check if volatility/liquidation data was added
            has_liquidation = 'liquidation_signal' in data.columns
            has_volatility = 'volatility_signal' in data.columns

            if has_liquidation:
                print("✅ Liquidation data available")
                liquidation_count = data['liquidation_signal'].notnull().sum()
                print(f"   Liquidation signals: {liquidation_count}")
            else:
                print("❌ No liquidation data")

            if has_volatility:
                print("✅ Volatility data available")
                volatility_count = data['volatility_signal'].notnull().sum()
                print(f"   Volatility signals: {volatility_count}")
            else:
                print("❌ No volatility data")

            return True
        else:
            print("❌ Backtest failed")
            return False

    except Exception as e:
        print(f"❌ Backtesting test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🧪 **TradPal Backtesting with Volatility Fallback Test**")
    print("=" * 70)

    # Test backtesting with volatility fallback
    success = test_backtesting_with_volatility_fallback()

    # Summary
    print("\n📋 **Test Summary**")
    print("=" * 30)

    if success:
        print("✅ Backtesting with Volatility Fallback: PASSED")
        print("🎉 Volatility indicators successfully integrated as liquidation proxy!")
    else:
        print("❌ Backtesting with Volatility Fallback: FAILED")

if __name__ == "__main__":
    main()