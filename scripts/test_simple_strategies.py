#!/usr/bin/env python3
"""
Test simpler trading strategies for better robustness.
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.backtester import Backtester
from main import calculate_buy_hold_performance

def test_simple_strategies():
    """Test simple trading strategies."""
    print("🧪 Testing Simple Trading Strategies")

    # Simple strategy configurations - using the correct format expected by the system
    strategies = {
        "ema_crossover": {
            "name": "EMA Crossover (9, 21)",
            "config": {
                "ema": {"enabled": True, "periods": [9, 21]},
                "rsi": {"enabled": False},
                "bb": {"enabled": False},
                "atr": {"enabled": True, "period": 14}
            }
        },
        "rsi_only": {
            "name": "RSI Only (14, 30/70)",
            "config": {
                "ema": {"enabled": False},
                "rsi": {"enabled": True, "period": 14, "oversold": 30, "overbought": 70},
                "bb": {"enabled": False},
                "atr": {"enabled": True, "period": 14}
            }
        },
        "bb_only": {
            "name": "Bollinger Bands Only (20, 2)",
            "config": {
                "ema": {"enabled": False},
                "rsi": {"enabled": False},
                "bb": {"enabled": True, "period": 20, "std_dev": 2.0},
                "atr": {"enabled": True, "period": 14}
            }
        },
        "ema_rsi": {
            "name": "EMA + RSI (9/21 + 14)",
            "config": {
                "ema": {"enabled": True, "periods": [9, 21]},
                "rsi": {"enabled": True, "period": 14, "oversold": 30, "overbought": 70},
                "bb": {"enabled": False},
                "atr": {"enabled": True, "period": 14}
            }
        },
        "ema_bb": {
            "name": "EMA + BB (9/21 + 20)",
            "config": {
                "ema": {"enabled": True, "periods": [9, 21]},
                "rsi": {"enabled": False},
                "bb": {"enabled": True, "period": 20, "std_dev": 2.0},
                "atr": {"enabled": True, "period": 14}
            }
        },
        "rsi_bb": {
            "name": "RSI + BB (14 + 20)",
            "config": {
                "ema": {"enabled": False},
                "rsi": {"enabled": True, "period": 14, "oversold": 30, "overbought": 70},
                "bb": {"enabled": True, "period": 20, "std_dev": 2.0},
                "atr": {"enabled": True, "period": 14}
            }
        }
    }

    # Test parameters
    symbol = 'BTC/USDT'
    timeframe = '1h'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)

    print(f"📊 Testing on {symbol} {timeframe}")

    # Calculate Buy & Hold
    try:
        bh_performance = calculate_buy_hold_performance(symbol, 'kraken', timeframe)
        print(f"📊 Buy & Hold: {bh_performance:.2f}%")
    except Exception as e:
        print(f"⚠️  Buy & Hold failed: {e}")
        bh_performance = 0

    results = {}

    # Test each strategy
    for strategy_key, strategy_info in strategies.items():
        print(f"🔄 Testing {strategy_info['name']}...")

        try:
            backtester = Backtester(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                config=strategy_info['config']
            )

            result = backtester.run_backtest(strategy='traditional')

            if result.get('success'):
                metrics = result.get('metrics', {})
                return_pct = metrics.get('return_pct', 0)
                win_rate = metrics.get('win_rate', 0)
                total_trades = metrics.get('total_trades', 0)

                results[strategy_key] = {
                    'name': strategy_info['name'],
                    'return_pct': return_pct,
                    'win_rate': win_rate,
                    'total_trades': total_trades,
                    'vs_buy_hold': return_pct - bh_performance
                }

                print(f"   ✅ Return: {return_pct:.2f}%, Win Rate: {win_rate:.1f}%, Trades: {total_trades}")
            else:
                print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"   ❌ Error: {str(e)}")

    # Find best strategy
    if results:
        best_key = max(results.keys(), key=lambda k: results[k]['return_pct'])
        best = results[best_key]
        print(f"\n🏆 Best Strategy: {best['name']}")
        print(f"   📈 Return: {best['return_pct']:.2f}%")
        print(f"   💪 vs Buy & Hold: {best['vs_buy_hold']:+.2f}%")

        return best

    return None

if __name__ == '__main__':
    best_strategy = test_simple_strategies()
    if best_strategy:
        print("\n✅ Simple strategies test completed")
    else:
        print("\n❌ No successful strategies")