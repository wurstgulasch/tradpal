#!/usr/bin/env python3
"""
Improved test script for backtesting with realistic market data.
Tests the backtesting system with proper indicators and realistic Bitcoin data.
"""

import asyncio
import sys
import os
import pytest
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.trading_service.backtesting_service.service import AsyncBacktester
import pandas as pd
import numpy as np

async def create_realistic_btc_data():
    """Create realistic Bitcoin price data from 2017-2021."""
    print("📊 Creating realistic Bitcoin market data...")

    dates = pd.date_range('2017-01-01', '2021-12-31', freq='D')
    np.random.seed(42)  # For reproducible results

    # Start with realistic Bitcoin price (~$1000 in 2017)
    base_price = 1000
    prices = [base_price]

    # Simulate market cycles: 2017-2018 bull, 2018-2019 bear, 2019-2021 bull
    for i in range(1, len(dates)):
        date = dates[i]

        # Base volatility
        volatility = 0.03  # 3% daily volatility

        # Adjust volatility and drift based on market cycle
        if date.year == 2017:
            # Bull market - higher volatility, upward bias
            drift = 0.002  # 0.2% daily upward drift
            volatility = 0.04
        elif date.year == 2018:
            # Bear market - high volatility, downward bias
            drift = -0.0015  # -0.15% daily drift
            volatility = 0.05
        elif date.year == 2019:
            # Recovery - moderate volatility, slight upward bias
            drift = 0.001
            volatility = 0.035
        else:  # 2020-2021
            # Massive bull run - high volatility, strong upward bias
            drift = 0.003
            volatility = 0.06

        # Add some random events (news, halving effects, etc.)
        if np.random.random() < 0.02:  # 2% chance of major event
            event_impact = np.random.normal(0, 0.1)  # Big move
            drift += event_impact

        # Generate price change
        change = np.random.normal(drift, volatility)
        new_price = prices[-1] * (1 + change)

        # Ensure price doesn't go negative
        new_price = max(new_price, 10)

        prices.append(new_price)

    # Create OHLCV data with realistic spreads
    data = []
    for i, price in enumerate(prices):
        # Generate realistic OHLC from close price
        volatility_daily = 0.02 + np.random.random() * 0.03  # 2-5% daily range

        high = price * (1 + abs(np.random.normal(0, volatility_daily)))
        low = price * (1 - abs(np.random.normal(0, volatility_daily)))
        open_price = prices[i-1] if i > 0 else price

        # Ensure OHLC relationships are correct
        high = max(high, open_price, price)
        low = min(low, open_price, price)

        # Realistic volume (Bitcoin trading volume)
        base_volume = 100000 + np.random.random() * 500000
        volume = base_volume * (1 + np.random.normal(0, 0.5))  # Volume varies

        data.append({
            'timestamp': dates[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': max(volume, 1000)  # Minimum volume
        })

    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)

    print(f"📊 Data shape: {df.shape}")
    print(f"📅 Date range: {df.index.min()} to {df.index.max()}")
    print(f"💰 Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"📈 Total return: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:.1f}%")

    return df

@pytest.mark.asyncio
async def test_improved_backtesting():
    """Test backtesting with improved indicators and realistic data."""
    print("🧪 Testing improved backtesting system...")

    # Create realistic data
    df = await create_realistic_btc_data()

    # Test backtester with this data
    print("\n🤖 Testing backtester with realistic data...")
    backtester = AsyncBacktester(
        symbol="BTC/USDT",
        exchange="binance",
        timeframe="1d",
        start_date="2017-01-01",
        end_date="2021-12-31",
        initial_capital=10000,
        data_source="kaggle",
        data_client=None  # No client needed for direct testing
    )

    # Manually set the data to bypass fetching
    backtester.data = df.copy()

    # Override the fetch method to return our data
    async def mock_fetch():
        return backtester.data
    backtester._fetch_data_async = mock_fetch

    # Run backtest with traditional strategy
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

        # Analyze trades in detail
        trades = result.get("trades", [])
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) <= 0]

        print(f"\n📊 Detailed Trade Analysis:")
        print(f"✅ Winning trades: {len(winning_trades)}")
        print(f"❌ Losing trades: {len(losing_trades)}")

        if winning_trades:
            avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades)
            max_win = max(t['pnl'] for t in winning_trades)
            print(f"💰 Average win: ${avg_win:.2f}")
            print(f"🏆 Max win: ${max_win:.2f}")

        if losing_trades:
            avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades)
            max_loss = min(t['pnl'] for t in losing_trades)
            print(f"💸 Average loss: ${avg_loss:.2f}")
            print(f"📉 Max loss: ${max_loss:.2f}")

        # Show sample trades
        print(f"\n📋 Sample Trades:")
        for i, trade in enumerate(trades[:5]):  # Show first 5 trades
            print(f"Trade {i+1}: {trade['type']} | Entry: ${trade['entry_price']:.2f} | Exit: ${trade['exit_price']:.2f} | P&L: ${trade['pnl']:.2f}")

        # Test with different strategies
        print("\n🔄 Testing different strategies...")

        # Test ML-enhanced (will fallback to traditional)
        result_ml = await backtester.run_backtest_async(strategy="ml_enhanced")
        if result_ml.get("success"):
            metrics_ml = result_ml.get("metrics", {})
            print(f"🤖 ML-Enhanced P&L: ${metrics_ml.get('total_pnl', 0):.2f}")
            print(f"🤖 ML-Enhanced Win Rate: {metrics_ml.get('win_rate', 0)}%")

        # Compare with buy-and-hold
        buy_hold_return = ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 10000
        print(f"\n📊 Benchmark Comparison:")
        print(f"📈 Buy & Hold P&L: ${buy_hold_return:.2f}")
        print(f"🤖 Strategy P&L: ${metrics.get('total_pnl', 0):.2f}")
        print(f"⚡ Outperformance: ${metrics.get('total_pnl', 0) - buy_hold_return:.2f}")

    else:
        print(f"❌ Backtest failed: {result.get('error')}")

async def main():
    """Main test function."""
    print("🚀 Starting improved backtesting tests...")
    await test_improved_backtesting()
    print("✅ Tests completed!")

if __name__ == "__main__":
    asyncio.run(main())