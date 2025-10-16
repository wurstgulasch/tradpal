#!/usr/bin/env python3
"""
Complete Performance Analysis for TradPal Trading Bot
Tests all market periods with detailed results
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
from pathlib import Path

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent))

from services.backtesting_service.service import AsyncBacktester

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

async def run_complete_performance_analysis():
    """Run complete performance analysis for all market periods."""

    print("🚀 Starting Complete Performance Analysis...")
    print("=" * 60)

    # Test periods with detailed analysis
    test_periods = [
        {
            "name": "Complete History (2012-2024)",
            "start": "2012-01-01",
            "end": "2024-12-31",
            "description": "Complete Bitcoin history from $13 to $100k+",
            "market_type": "COMPLETE"
        },
        {
            "name": "Bear Market (2017-2021)",
            "start": "2017-01-01",
            "end": "2021-12-31",
            "description": "From ATH $20k to bottom $30k - major bear market",
            "market_type": "BEAR"
        },
        {
            "name": "Bull Market Recovery (2020-2021)",
            "start": "2020-03-01",
            "end": "2021-11-30",
            "description": "COVID recovery and institutional adoption",
            "market_type": "BULL"
        },
        {
            "name": "Sideways Market (2022-2023)",
            "start": "2022-01-01",
            "end": "2023-12-31",
            "description": "Choppy consolidation after Terra/LUNA collapse",
            "market_type": "SIDEWAYS"
        },
        {
            "name": "Recent Bull Market (2023-2024)",
            "start": "2023-01-01",
            "end": "2024-10-01",
            "description": "ETF approval and renewed institutional interest",
            "market_type": "BULL_RECENT"
        }
    ]

    results = {}

    for period in test_periods:
        print(f"\n📊 Testing: {period['name']}")
        print(f"   Period: {period['start']} to {period['end']}")
        print(f"   Market: {period['market_type']}")
        print(f"   Description: {period['description']}")
        print("-" * 60)

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

                # Calculate additional metrics
                outperformance = metrics.get("return_pct", 0) - buy_hold_return
                total_return = metrics.get("return_pct", 0)

                # Store results
                results[period["name"]] = {
                    "period": period,
                    "strategy_metrics": metrics,
                    "buy_hold_return": buy_hold_return,
                    "outperformance": outperformance,
                    "total_return": total_return,
                    "trades_count": len(trades),
                    "winning_trades": sum(1 for t in trades if t.get("pnl", 0) > 0),
                    "losing_trades": sum(1 for t in trades if t.get("pnl", 0) <= 0),
                    "win_rate": (sum(1 for t in trades if t.get("pnl", 0) > 0) / len(trades) * 100) if trades else 0,
                    "sample_trades": trades[:8] if trades else [],  # More samples
                    "data_points": len(backtester.data) if backtester.data is not None else 0
                }

                print("✅ STRATEGY RESULTS:"                print(".2f"                print(".2f"                print(".2f"                print(".2f"                print(".2f"                print(f"   📊 Trades: {len(trades)} (Win: {results[period['name']]['winning_trades']}, Loss: {results[period['name']]['losing_trades']})")
                print(".2f"                print(f"   📈 Data Points: {results[period['name']]['data_points']}")

                # Show sample trades
                if trades and len(trades) > 0:
                    print("   🔄 Sample Trades:")
                    for i, trade in enumerate(trades[:3]):
                        trade_type = trade.get('type', 'unknown')
                        pnl = trade.get('pnl', 0)
                        entry = trade.get('entry_price', 0)
                        exit = trade.get('exit_price', 0)
                        print(f"      {i+1}. {trade_type.upper()} | Entry: ${entry:.2f} | Exit: ${exit:.2f} | P&L: ${pnl:+.2f}")

            else:
                print(f"❌ Backtest failed: {result.get('error', 'Unknown error')}")
                results[period["name"]] = {"error": result.get("error", "Unknown error")}

        except Exception as e:
            print(f"❌ Error testing {period['name']}: {str(e)}")
            results[period["name"]] = {"error": str(e)}

    # Save results
    output_file = "docs/complete_performance_results.json"
    os.makedirs("docs", exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n💾 Complete results saved to {output_file}")

    # Generate comprehensive summary
    generate_comprehensive_summary(results)

    return results

def generate_comprehensive_summary(results):
    """Generate comprehensive performance summary."""

    print("\n" + "="*100)
    print("🎯 COMPREHENSIVE PERFORMANCE SUMMARY - ALL MARKET PERIODS")
    print("="*100)

    successful_tests = {k: v for k, v in results.items() if "error" not in v}

    if not successful_tests:
        print("❌ No successful tests to summarize")
        return

    print("
📊 PERFORMANCE OVERVIEW:"    print("<10")
    print("-" * 80)
    for period_name, data in successful_tests.items():
        strategy_return = data["total_return"]
        bh_return = data["buy_hold_return"]
        outperformance = data["outperformance"]
        trades = data["trades_count"]
        win_rate = data["win_rate"]

        status = "✅ POSITIVE" if strategy_return > 0 else "❌ NEGATIVE"
        out_status = "🚀 OUTPERFORM" if outperformance > 0 else "📉 UNDERPERFORM"

        print("<10")

    # Overall statistics
    total_strategy_return = sum(r["total_return"] for r in successful_tests.values())
    avg_strategy_return = total_strategy_return / len(successful_tests)

    total_bh_return = sum(r["buy_hold_return"] for r in successful_tests.values())
    avg_bh_return = total_bh_return / len(successful_tests)

    total_outperformance = sum(r["outperformance"] for r in successful_tests.values())
    avg_outperformance = total_outperformance / len(successful_tests)

    print("
🎯 OVERALL AVERAGE PERFORMANCE:"    print(".2f"    print(".2f"    print(".2f"
    # Best and worst periods
    best_period = max(successful_tests.keys(), key=lambda x: successful_tests[x]["outperformance"])
    worst_period = min(successful_tests.keys(), key=lambda x: successful_tests[x]["outperformance"])

    print("
🏆 BEST OUTPERFORMANCE:"    print(f"   {best_period}")
    print(".2f"
    print("
📉 WORST OUTPERFORMANCE:"    print(f"   {worst_period}")
    print(".2f"
    # Market condition analysis
    print("
📈 MARKET REGIME ANALYSIS:"    for period_name, data in successful_tests.items():
        bh_return = data["buy_hold_return"]
        strategy_return = data["total_return"]
        outperformance = data["outperformance"]

        if bh_return > 30:
            condition = "🐂 STRONG BULL"
        elif bh_return > 10:
            condition = "🐂 BULL"
        elif bh_return > -10:
            condition = "🔄 SIDEWAYS"
        elif bh_return > -30:
            condition = "🐻 WEAK BEAR"
        else:
            condition = "🐻 STRONG BEAR"

        strategy_status = "✅ PROFITABLE" if strategy_return > 0 else "❌ LOSING"
        out_status = "🚀 OUTPERFORMING" if outperformance > 0 else "📉 UNDERPERFORMING"

        print(f"   {period_name}: {condition} | Strategy: {strategy_status} | vs Buy&Hold: {out_status}")

    print("
✅ COMPLETE ANALYSIS FINISHED!"    print("="*100)

if __name__ == "__main__":
    asyncio.run(run_complete_performance_analysis())