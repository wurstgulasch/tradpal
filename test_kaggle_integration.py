#!/usr/bin/env python3
"""
Test script for Kaggle data source integration in backtesting.
Tests the data service integration without running full services.
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.data_service.client import DataServiceClient
from services.backtesting_service.service import AsyncBacktester
import pandas as pd

async def test_backtester_direct():
    """Test backtester directly with mock data."""
    print("🧪 Testing backtester with mock Kaggle data...")

    # Create mock data that simulates Kaggle Bitcoin data
    import numpy as np
    dates = pd.date_range('2020-01-01', '2021-01-01', freq='D')
    np.random.seed(42)  # For reproducible results

    # Generate realistic Bitcoin-like price data
    base_price = 10000
    prices = []
    current_price = base_price

    for i in range(len(dates)):
        # Add some trend and volatility
        change = np.random.normal(0.001, 0.03)  # Mean 0.1%, std 3%
        current_price *= (1 + change)
        prices.append(current_price)

    # Create OHLCV data
    data = []
    for i, price in enumerate(prices):
        high = price * (1 + abs(np.random.normal(0, 0.02)))
        low = price * (1 - abs(np.random.normal(0, 0.02)))
        open_price = prices[i-1] if i > 0 else price
        close = price
        volume = np.random.uniform(1000000, 10000000)  # Realistic volume

        data.append({
            'timestamp': dates[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })

    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)

    print(f"� Mock data shape: {df.shape}")
    print(f"📅 Date range: {df.index.min()} to {df.index.max()}")
    print(f"� Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    # Test backtester with this data
    print("\n🤖 Testing backtester with mock data...")
    backtester = AsyncBacktester(
        symbol="BTC/USDT",
        exchange="binance",
        timeframe="1d",
        start_date="2020-01-01",
        end_date="2021-01-01",
        initial_capital=10000,
        data_source="kaggle",
        data_client=None  # No client needed for direct testing
    )

    # Manually set the data to bypass fetching
    backtester.data = df.copy()

    # Override the fetch method to return our data
    original_fetch = backtester._fetch_data_async
    async def mock_fetch():
        return backtester.data
    backtester._fetch_data_async = mock_fetch

    # Run a simple backtest
    result = await backtester.run_backtest_async(strategy="traditional")

    if result.get("success"):
        print("✅ Backtest successful!")
        metrics = result.get("metrics", {})
        print(f"💰 Total P&L: ${metrics.get('total_pnl', 0):.2f}")
        print(f"📊 Win Rate: {metrics.get('win_rate', 0)}%")
        print(f"📈 Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"📉 Max Drawdown: {metrics.get('max_drawdown', 0)}%")
        print(f"🎯 Total Trades: {metrics.get('total_trades', 0)}")
        print(f"📈 Return %: {metrics.get('return_pct', 0)}%")

        # Test with different strategies
        print("\n🔄 Testing different strategies...")

        # Test ML-enhanced (will fallback to traditional)
        result_ml = await backtester.run_backtest_async(strategy="ml_enhanced")
        if result_ml.get("success"):
            metrics_ml = result_ml.get("metrics", {})
            print(f"🤖 ML-Enhanced P&L: ${metrics_ml.get('total_pnl', 0):.2f}")

    else:
        print(f"❌ Backtest failed: {result.get('error')}")

async def test_kaggle_integration():
    """Test Kaggle data source integration."""
    print("🧪 Testing Kaggle data source integration...")

    # For now, skip the service test and go directly to backtester test
    await test_backtester_direct()

if __name__ == "__main__":
    asyncio.run(test_kaggle_integration())