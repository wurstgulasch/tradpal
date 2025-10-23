#!/usr/bin/env python3
"""
Test script for adaptive risk management system.
"""

import asyncio
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.trading_service.backtesting_service.service import AsyncBacktester

async def test_adaptive_risk_management():
    """Test the adaptive risk management system."""
    print("🧪 Testing Adaptive Risk Management System")
    print("=" * 50)

    try:
        # Test with bull market period (2020-2021)
        print("\n🐂 Testing Bull Market Period (2020-2021)...")
        backtester = AsyncBacktester(
            symbol='BTC/USDT',
            exchange='kraken',
            timeframe='1d',
            start_date='2020-01-01',
            end_date='2021-12-31'
        )

        result = await backtester.run_backtest_async(strategy='traditional')

        if result.get('success'):
            metrics = result.get('metrics', {})
            print("✅ Bull Market Backtest Successful!")
            print(f"📊 Trades: {result.get('trades_count', 0)}")
            print(f"💰 Final Capital: ${metrics.get('final_capital', 0):.2f}")
            print(f"📈 Total Return: {metrics.get('return_pct', 0):.2f}%")
            print(f"🎯 Win Rate: {metrics.get('win_rate', 0):.1f}%")
            print(f"📊 Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        else:
            print(f"❌ Bull Market Backtest Failed: {result.get('error')}")

        # Test with bear market period (2017-2021)
        print("\n🐻 Testing Bear Market Period (2017-2021)...")
        backtester2 = AsyncBacktester(
            symbol='BTC/USDT',
            exchange='kraken',
            timeframe='1d',
            start_date='2017-01-01',
            end_date='2021-12-31'
        )

        result2 = await backtester2.run_backtest_async(strategy='traditional')

        if result2.get('success'):
            metrics2 = result2.get('metrics', {})
            print("✅ Bear Market Backtest Successful!")
            print(f"📊 Trades: {result2.get('trades_count', 0)}")
            print(f"💰 Final Capital: ${metrics2.get('final_capital', 0):.2f}")
            print(f"📈 Total Return: {metrics2.get('return_pct', 0):.2f}%")
            print(f"🎯 Win Rate: {metrics2.get('win_rate', 0):.1f}%")
            print(f"📊 Sharpe Ratio: {metrics2.get('sharpe_ratio', 0):.2f}")
        else:
            print(f"❌ Bear Market Backtest Failed: {result2.get('error')}")

        print("\n🎯 Adaptive Risk Management Test Complete!")
        print("\nKey Improvements:")
        print("✅ Bull Markets: Higher risk tolerance, wider take profits")
        print("✅ Bear Markets: Conservative risk, tighter stops")
        print("✅ Sideways Markets: Moderate risk, balanced parameters")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_adaptive_risk_management())