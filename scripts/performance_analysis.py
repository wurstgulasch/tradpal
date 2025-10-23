#!/usr/bin/env python3
"""
Comprehensive Performance Analysis for TradPal Trading Bot
Tests performance across different market periods and conditions
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent))

from services.trading_service.backtesting_service.service import AsyncBacktester

async def run_comprehensive_backtests():
    """Run comprehensive backtests across different market periods."""

    print("🚀 Starting Comprehensive Performance Analysis...")
    print("=" * 60)

    # Test periods
    test_periods = [
        {
            "name": "Full Period (2012-2024)",
            "start": "2012-01-01",
            "end": "2024-12-31",
            "description": "Complete market cycle including multiple bull/bear markets"
        },
        {
            "name": "Bitcoin Bear Market (2017-2021)",
            "start": "2017-01-01",
            "end": "2021-12-31",
            "description": "Major bear market from ATH to bottom"
        },
        {
            "name": "Bitcoin Bull Market (2020-2021)",
            "start": "2020-01-01",
            "end": "2021-11-30",
            "description": "Strong bull market recovery"
        },
        {
            "name": "Sideways Market (2022-2023)",
            "start": "2022-01-01",
            "end": "2023-12-31",
            "description": "Choppy sideways market with mean reversion opportunities"
        },
        {
            "name": "Recent Bull Market (2023-2024)",
            "start": "2023-01-01",
            "end": "2024-10-01",
            "description": "Recent bull market phase"
        }
    ]

    results = {}

    for period in test_periods:
        print(f"\n📊 Testing: {period['name']}")
        print(f"   Period: {period['start']} to {period['end']}")
        print(f"   Description: {period['description']}")
        print("-" * 50)

        try:
            # Create backtester
            backtester = AsyncBacktester(
                symbol="BTC/USDT",
                exchange="binance",
                timeframe="1d",
                start_date=period["start"],
                end_date=period["end"],
                initial_capital=10000,
                data_source="kaggle"
            )

            # Run backtest
            result = await backtester.run_backtest_async(strategy="traditional")

            if result.get("success"):
                metrics = result["metrics"]
                trades = result["trades"]

                # Calculate Buy & Hold return
                buy_hold_return = calculate_buy_hold_return(backtester.data)

                # Store results
                results[period["name"]] = {
                    "period": period,
                    "strategy_metrics": metrics,
                    "buy_hold_return": buy_hold_return,
                    "outperformance": metrics.get("return_pct", 0) - buy_hold_return,
                    "trades_count": len(trades),
                    "winning_trades": sum(1 for t in trades if t.get("pnl", 0) > 0),
                    "losing_trades": sum(1 for t in trades if t.get("pnl", 0) <= 0),
                    "sample_trades": trades[:5] if trades else []
                }

                print("✅ Strategy Results:")
                print(".2f")
                print(".2f")
                print(".2f")
                print(".2f")
                print(f"   📊 Trades: {len(trades)} (Win: {results[period['name']]['winning_trades']}, Loss: {results[period['name']]['losing_trades']})")
                print(".2f")
                print(".2f")
            else:
                print(f"❌ Backtest failed: {result.get('error', 'Unknown error')}")
                results[period["name"]] = {"error": result.get("error", "Unknown error")}

        except Exception as e:
            print(f"❌ Error testing {period['name']}: {str(e)}")
            results[period["name"]] = {"error": str(e)}

    # Save results
    output_file = "docs/performance_analysis_results.json"
    os.makedirs("docs", exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n💾 Results saved to {output_file}")

    # Generate summary
    generate_performance_summary(results)

    return results

def calculate_buy_hold_return(data):
    """Calculate buy & hold return for comparison."""
    if data is None or data.empty:
        return 0

    try:
        start_price = data['close'].iloc[0]
        end_price = data['close'].iloc[-1]
        return ((end_price - start_price) / start_price) * 100
    except:
        return 0

def generate_performance_summary(results):
    """Generate a comprehensive performance summary."""

    print("\n" + "="*80)
    print("🎯 COMPREHENSIVE PERFORMANCE SUMMARY")
    print("="*80)

    successful_tests = {k: v for k, v in results.items() if "error" not in v}

    if not successful_tests:
        print("❌ No successful tests to summarize")
        return

    # Overall statistics
    total_outperformance = sum(r["outperformance"] for r in successful_tests.values())
    avg_outperformance = total_outperformance / len(successful_tests)

    total_strategy_return = sum(r["strategy_metrics"]["return_pct"] for r in successful_tests.values())
    avg_strategy_return = total_strategy_return / len(successful_tests)

    total_bh_return = sum(r["buy_hold_return"] for r in successful_tests.values())
    avg_bh_return = total_bh_return / len(successful_tests)

    print("
📊 OVERALL PERFORMANCE (2012-2024):"    print(".2f"    print(".2f"    print(".2f"    print(".2f"
    # Best and worst periods
    best_period = max(successful_tests.keys(), key=lambda x: successful_tests[x]["outperformance"])
    worst_period = min(successful_tests.keys(), key=lambda x: successful_tests[x]["outperformance"])

    print("
🏆 BEST PERIOD:"    print(f"   {best_period}")
    print(".2f"
    print("
📉 WORST PERIOD:"    print(f"   {worst_period}")
    print(".2f"
    # Market condition analysis
    print("
📈 MARKET CONDITION ANALYSIS:"    for period_name, data in successful_tests.items():
        bh_return = data["buy_hold_return"]
        if bh_return > 20:
            condition = "🐂 BULL MARKET"
        elif bh_return < -20:
            condition = "🐻 BEAR MARKET"
        else:
            condition = "🔄 SIDEWAYS MARKET"

        print(f"   {period_name}: {condition} (Buy&Hold: {bh_return:+.1f}%)")

    print("
✅ ANALYSIS COMPLETE!"    print("="*80)

if __name__ == "__main__":
    asyncio.run(run_comprehensive_backtests())