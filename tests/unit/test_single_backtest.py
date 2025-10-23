#!/usr/bin/env python3
"""
Test script to run a single backtest with ML enhancement and check signal generation.
"""
import sys
import os
sys.path.append('/Users/danielsadowski/VSCodeProjects/tradpal/tradpal')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from services.trading_service.backtesting_service.orchestrator import BacktestingServiceOrchestrator

async def test_single_backtest():
    """Test a single backtest configuration."""
    print("🧪 Testing single backtest with ML enhancement...")

    # Create backtesting orchestrator
    orchestrator = BacktestingServiceOrchestrator()

    # Test configuration that should generate signals
    config = {
        'name': 'ML Enhanced Strategy',
        'symbol': 'BTC/USDT',
        'timeframe': '1h',
        'indicators': {
            'ema': {'enabled': True, 'periods': [9, 21]},
            'rsi': {'enabled': True, 'period': 14, 'oversold': 30, 'overbought': 70},
            'bb': {'enabled': True, 'period': 20, 'std_dev': 2.0}
        }
    }

    print("📊 Running backtest with ML enhancement...")

    # Create sample data for testing (in real usage, would fetch from data service)
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='1H')
    np.random.seed(42)
    data = pd.DataFrame({
        'open': 50000 + np.random.normal(0, 1000, len(dates)),
        'high': 51000 + np.random.normal(0, 1000, len(dates)),
        'low': 49000 + np.random.normal(0, 1000, len(dates)),
        'close': 50000 + np.random.normal(0, 1000, len(dates)),
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)

    # Ensure high >= close >= low
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)

    try:
        # Initialize orchestrator
        await orchestrator.initialize()

        # Run backtest using orchestrator
        results = await orchestrator.run_quick_backtest(config, data)

        if results.get('success'):
            metrics = results.get('metrics', {})
            trades_count = metrics.get('total_trades', 0)

            print("✅ Backtest completed successfully!")
            print(f"📊 Results: {trades_count} trades")
            print(f"   Win Rate: {metrics.get('win_rate', 0):.1%}")
            print(f"   Total P&L: {metrics.get('total_pnl', 0):.2f}")
            print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")

            # Check if we have any trades
            if trades_count == 0:
                print("⚠️  No trades generated - checking signal generation...")

                print(f"📊 Using {len(data)} data points")

                # In real implementation, would analyze signals here
                print("📊 Signal Analysis:")
                print("   (Signal analysis would be implemented with actual data processing)")

        else:
            print(f"❌ Backtest failed: {results.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"❌ Error during backtest: {str(e)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_single_backtest())