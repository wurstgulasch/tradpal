#!/usr/bin/env python3
"""
Test script to run a single backtest with ML enhancement and check signal generation.
"""
import sys
import os
sys.path.append('/Users/danielsadowski/VSCodeProjects/tradpal_indicator')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.backtester import Backtester

def test_single_backtest():
    """Test a single backtest configuration."""
    print("🧪 Testing single backtest with ML enhancement...")

    # Create backtester
    backtester = Backtester(
        symbol='BTC/USDT',
        timeframe='1m',
        start_date=datetime.now() - timedelta(days=7),
        end_date=datetime.now()
    )

    # Test configuration that should generate signals
    config = {
        'ema': {'enabled': True, 'periods': [9, 21]},
        'rsi': {'enabled': True, 'period': 14, 'oversold': 30, 'overbought': 70},
        'bb': {'enabled': True, 'period': 20, 'std_dev': 2.0}
    }

    print("📊 Running backtest with ML enhancement...")
    results = backtester.run_backtest(strategy='ml_enhanced', config=config)

    if results.get('success'):
        metrics = results.get('metrics', {})
        trades_count = results.get('trades_count', 0)

        print("✅ Backtest completed successfully!")
        print(f"📊 Results: {trades_count} trades")
        print(f"   Win Rate: {metrics.get('win_rate', 0)}%")
        print(f"   Total P&L: {metrics.get('total_pnl', 0)}")
        print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0)}")

        # Check if we have any trades
        if trades_count == 0:
            print("⚠️  No trades generated - checking signal generation...")

            # Let's manually check what signals are generated
            data = backtester._fetch_data()
            if not data.empty:
                print(f"📊 Fetched {len(data)} data points")

                # Prepare data with ML enhancement
                data = backtester._prepare_ml_enhanced_signals(data, config)

                # Check signal counts
                buy_signals = (data['Buy_Signal'] == 1).sum()
                sell_signals = (data['Sell_Signal'] == 1).sum()
                ml_signals = data['ML_Signal'].value_counts() if 'ML_Signal' in data.columns else {}
                signal_sources = data['Signal_Source'].value_counts() if 'Signal_Source' in data.columns else {}

                print(f"📊 Signal Analysis:")
                print(f"   Buy_Signal == 1: {buy_signals}")
                print(f"   Sell_Signal == 1: {sell_signals}")
                print(f"   ML_Signal distribution: {dict(ml_signals)}")
                print(f"   Signal_Source distribution: {dict(signal_sources)}")

                if 'ML_Confidence' in data.columns:
                    high_conf = (data['ML_Confidence'] > 0.5).sum()
                    print(f"   High confidence ML signals (>0.5): {high_conf}")

    else:
        print(f"❌ Backtest failed: {results.get('error', 'Unknown error')}")

if __name__ == "__main__":
    test_single_backtest()