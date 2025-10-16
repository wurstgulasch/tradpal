#!/usr/bin/env python3
"""
Test script for bull market trading strategy.
Tests the backtesting system with a simulated bull market (2020-2021 Bitcoin rally).
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.backtesting_service.service import AsyncBacktester
import pandas as pd
import numpy as np

async def create_bull_market_data():
    """Create realistic bull market data (similar to 2020-2021 Bitcoin rally)."""
    print("📈 Creating bull market data (2020-2021 style)...")

    dates = pd.date_range('2020-03-01', '2021-11-30', freq='D')
    np.random.seed(123)  # Different seed for different pattern

    # Start with realistic post-COVID Bitcoin price (~$5000 in March 2020)
    base_price = 5000
    prices = [base_price]

    # Simulate bull market: COVID recovery + institutional adoption
    for i in range(1, len(dates)):
        date = dates[i]

        # Base parameters for bull market
        drift = 0.0015  # 0.15% daily upward drift (bull market)
        volatility = 0.03  # 3% daily volatility

        # Adjust based on market phases
        if date.month in [3, 4]:  # COVID crash recovery
            drift = 0.002  # Strong recovery
            volatility = 0.04
        elif date.month in [5, 6, 7]:  # Consolidation
            drift = 0.0005  # Slow growth
            volatility = 0.025
        elif date.month >= 8:  # Institutional adoption phase
            drift = 0.0025  # Strong bull run
            volatility = 0.035

        # Add news events and adoption milestones
        if np.random.random() < 0.03:  # 3% chance of major event
            event_impact = np.random.normal(0.05, 0.08)  # Big positive moves
            drift += event_impact

        # Generate price change
        change = np.random.normal(drift, volatility)
        new_price = prices[-1] * (1 + change)

        # Ensure minimum price (no crashes in bull market)
        new_price = max(new_price, base_price * 0.7)

        prices.append(new_price)

    # Create OHLCV data
    data = []
    for i, price in enumerate(prices):
        volatility_daily = 0.02 + np.random.random() * 0.02

        high = price * (1 + abs(np.random.normal(0, volatility_daily)))
        low = price * (1 - abs(np.random.normal(0, volatility_daily)))
        open_price = prices[i-1] if i > 0 else price

        high = max(high, open_price, price)
        low = min(low, open_price, price)

        volume = 50000 + np.random.random() * 200000
        volume *= (1 + np.random.normal(0, 0.3))  # Volume varies

        data.append({
            'timestamp': dates[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': max(volume, 10000)
        })

    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)

    print(f"📊 Bull market data shape: {df.shape}")
    print(f"📅 Date range: {df.index.min()} to {df.index.max()}")
    print(f"💰 Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"📈 Total return: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:.1f}%")

    return df

async def test_bull_market_strategy():
    """Test strategy in bull market conditions."""
    print("🚀 Testing bull market strategy...")

    # Create bull market data
    df = await create_bull_market_data()

    # Test backtester
    backtester = AsyncBacktester(
        symbol="BTC/USDT",
        exchange="binance",
        timeframe="1d",
        start_date="2020-03-01",
        end_date="2021-11-30",
        initial_capital=10000,
        data_source="kaggle",
        data_client=None
    )

    backtester.data = df.copy()
    async def mock_fetch():
        return backtester.data
    backtester._fetch_data_async = mock_fetch

    # Run backtest
    result = await backtester.run_backtest_async(strategy="traditional")

    if result.get("success"):
        print("✅ Bull market backtest successful!")
        metrics = result.get("metrics", {})
        print(f"💰 Total P&L: ${metrics.get('total_pnl', 0):.2f}")
        print(f"📊 Win Rate: {metrics.get('win_rate', 0)}%")
        print(f"📈 Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"📉 Max Drawdown: {metrics.get('max_drawdown', 0)}%")
        print(f"🎯 Total Trades: {metrics.get('total_trades', 0)}")
        print(f"📈 Return %: {metrics.get('return_pct', 0)}%")

        # Compare with buy-and-hold
        buy_hold_return = ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 10000
        print(f"\n📊 Benchmark Comparison:")
        print(f"📈 Buy & Hold P&L: ${buy_hold_return:.2f}")
        print(f"🤖 Strategy P&L: ${metrics.get('total_pnl', 0):.2f}")
        print(f"⚡ Outperformance: ${metrics.get('total_pnl', 0) - buy_hold_return:.2f}")

        # Analyze trades
        trades = result.get("trades", [])
        if trades:
            winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
            print(f"\n✅ Winning trades: {len(winning_trades)}/{len(trades)}")
            if winning_trades:
                avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades)
                print(f"💰 Average win: ${avg_win:.2f}")

    else:
        print(f"❌ Backtest failed: {result.get('error')}")

async def main():
    """Main test function."""
    print("🏆 Starting bull market strategy tests...")
    await test_bull_market_strategy()
    print("✅ Bull market tests completed!")

if __name__ == "__main__":
    asyncio.run(main())