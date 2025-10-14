#!/usr/bin/env python3
"""
Demo script for Backtesting Service.

This script demonstrates how to use the Backtesting Service
for various types of backtesting operations.
"""

import asyncio
import json
import time
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from services.backtesting_service.service import BacktestingService

# Mock EventSystem for demo purposes
class Event:
    def __init__(self, type, data):
        self.type = type
        self.data = data

class EventSystem:
    def __init__(self):
        self.subscribers = {}

    def subscribe(self, event_type, handler):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)

    async def publish(self, event):
        if event.type in self.subscribers:
            for handler in self.subscribers[event.type]:
                handler(event)


async def demo_single_backtest():
    """Demonstrate single backtest execution."""
    print("🔍 Running Single Backtest Demo")
    print("-" * 40)

    # Initialize service
    event_system = EventSystem()
    service = BacktestingService(event_system=event_system)

    # Run backtest
    result = await service.run_backtest_async(
        symbol="BTC/USDT",
        timeframe="1d",
        start_date="2024-01-01",
        end_date="2024-03-01",
        strategy="traditional",
        initial_capital=10000.0,
        backtest_id="demo_single_backtest"
    )

    if result.get("success"):
        metrics = result.get("metrics", {})
        print("✅ Backtest completed successfully!")
        print(f"   Trades: {result.get('trades_count', 0)}")
        print(f"   Total P&L: ${metrics.get('total_pnl', 0):.2f}")
        print(f"   Win Rate: {metrics.get('win_rate', 0):.1f}%")
        print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"   Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%")
    else:
        print(f"❌ Backtest failed: {result.get('error')}")

    return result


async def demo_multi_symbol_backtest():
    """Demonstrate multi-symbol backtest execution."""
    print("\n🔍 Running Multi-Symbol Backtest Demo")
    print("-" * 40)

    event_system = EventSystem()
    service = BacktestingService(event_system=event_system)

    symbols = ["BTC/USDT", "ETH/USDT"]

    result = await service.run_multi_symbol_backtest_async(
        symbols=symbols,
        timeframe="1d",
        start_date="2024-01-01",
        end_date="2024-02-01",
        initial_capital=5000.0,
        max_workers=2,
        backtest_id="demo_multi_symbol"
    )

    successful = result.get("successful_backtests", [])
    failed = result.get("failed_backtests", [])

    print(f"📊 Multi-symbol backtest completed:")
    print(f"   Successful: {len(successful)} backtests")
    print(f"   Failed: {len(failed)} backtests")

    aggregated = result.get("aggregated_metrics", {})
    print(f"   Avg Total P&L: ${aggregated.get('avg_total_pnl', 0):.2f}")
    print(f"   Avg Win Rate: {aggregated.get('avg_win_rate', 0):.1f}%")
    print(f"   Avg Sharpe Ratio: {aggregated.get('avg_sharpe_ratio', 0):.2f}")
    return result


async def demo_event_driven_backtest():
    """Demonstrate event-driven backtest execution."""
    print("\n🔍 Running Event-Driven Backtest Demo")
    print("-" * 40)

    event_system = EventSystem()
    service = BacktestingService(event_system=event_system)

    # Completion handler
    results_received = []

    def handle_completion(event):
        results_received.append(event.data)
        print(f"📨 Received completion event for backtest: {event.data.get('backtest_id')}")

    # Subscribe to completion events
    event_system.subscribe("backtest.completed", handle_completion)

    # Publish backtest request
    await event_system.publish(Event(
        type="backtest.request",
        data={
            "backtest_id": "demo_event_backtest",
            "symbol": "BTC/USDT",
            "timeframe": "1d",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "strategy": "traditional",
            "initial_capital": 10000.0
        }
    ))

    # Wait for completion
    timeout = 60  # 60 seconds
    start_time = time.time()

    while time.time() - start_time < timeout and not results_received:
        await asyncio.sleep(1)

    if results_received:
        result = results_received[0].get("result", {})
        print("✅ Event-driven backtest completed!")
        print(f"   Result: {result.get('success', False)}")
    else:
        print("❌ Event-driven backtest timed out")

    return results_received[0] if results_received else None


async def demo_backtest_status_monitoring():
    """Demonstrate backtest status monitoring."""
    print("\n🔍 Running Status Monitoring Demo")
    print("-" * 40)

    event_system = EventSystem()
    service = BacktestingService(event_system=event_system)

    # Start a backtest
    await service.run_backtest_async(
        symbol="BTC/USDT",
        timeframe="1d",
        start_date="2024-01-01",
        end_date="2024-02-01",
        strategy="traditional",
        backtest_id="demo_status_monitoring"
    )

    # Monitor status
    for i in range(10):  # Check 10 times
        status = await service.get_backtest_status("demo_status_monitoring")
        print(f"Status check {i+1}: {status.get('status', 'unknown')}")

        if status.get("status") in ["completed", "failed"]:
            break

        await asyncio.sleep(2)

    # Get final result
    final_status = await service.get_backtest_status("demo_status_monitoring")
    if final_status.get("status") == "completed":
        result = final_status.get("result", {})
        print(f"✅ Final result: {result.get('trades_count', 0)} trades")
    else:
        print(f"❌ Final status: {final_status.get('status')}")

    return final_status


async def demo_active_backtests():
    """Demonstrate active backtests listing."""
    print("\n🔍 Running Active Backtests Demo")
    print("-" * 40)

    event_system = EventSystem()
    service = BacktestingService(event_system=event_system)

    # Start multiple backtests
    backtest_ids = []
    for i in range(3):
        backtest_id = f"demo_active_{i+1}"
        await service.run_backtest_async(
            symbol="BTC/USDT",
            timeframe="1d",
            start_date="2024-01-01",
            end_date="2024-01-15",
            strategy="traditional",
            backtest_id=backtest_id
        )
        backtest_ids.append(backtest_id)

    # List active backtests
    active = await service.list_active_backtests()
    print(f"📊 Active backtests: {len(active)}")
    for backtest in active:
        print(f"   - {backtest.get('backtest_id')}: {backtest.get('status')}")

    # Wait for completion
    await asyncio.sleep(10)

    # List again
    active_after = await service.list_active_backtests()
    print(f"📊 Active backtests after completion: {len(active_after)}")

    return active


async def run_all_demos():
    """Run all demo functions."""
    print("🚀 Backtesting Service Demo Suite")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    demos = [
        ("Single Backtest", demo_single_backtest),
        ("Multi-Symbol Backtest", demo_multi_symbol_backtest),
        ("Event-Driven Backtest", demo_event_driven_backtest),
        ("Status Monitoring", demo_backtest_status_monitoring),
        ("Active Backtests Listing", demo_active_backtests)
    ]

    results = {}

    for demo_name, demo_func in demos:
        try:
            print(f"\n{'='*20} {demo_name} {'='*20}")
            result = await demo_func()
            results[demo_name] = {"success": True, "result": result}
            print(f"✅ {demo_name} completed successfully")
        except Exception as e:
            print(f"❌ {demo_name} failed: {str(e)}")
            results[demo_name] = {"success": False, "error": str(e)}

    # Summary
    print("\n" + "=" * 50)
    print("📊 DEMO SUMMARY")
    print("=" * 50)

    successful = sum(1 for result in results.values() if result["success"])
    total = len(results)

    print(f"Total Demos: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")

    if successful == total:
        print("🎉 All demos completed successfully!")
    else:
        print("⚠️  Some demos failed - check output above")

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Save results
    results_file = f"output/backtesting_service_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump({
            "demo_run": {
                "timestamp": datetime.now().isoformat(),
                "total_demos": total,
                "successful": successful,
                "failed": total - successful
            },
            "results": results
        }, f, indent=2, default=str)

    print(f"📄 Detailed results saved to: {results_file}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backtesting Service Demo")
    parser.add_argument("--single", action="store_true", help="Run only single backtest demo")
    parser.add_argument("--multi-symbol", action="store_true", help="Run only multi-symbol demo")
    parser.add_argument("--event-driven", action="store_true", help="Run only event-driven demo")
    parser.add_argument("--status", action="store_true", help="Run only status monitoring demo")
    parser.add_argument("--active", action="store_true", help="Run only active backtests demo")

    args = parser.parse_args()

    if args.single:
        asyncio.run(demo_single_backtest())
    elif args.multi_symbol:
        asyncio.run(demo_multi_symbol_backtest())
    elif args.event_driven:
        asyncio.run(demo_event_driven_backtest())
    elif args.status:
        asyncio.run(demo_backtest_status_monitoring())
    elif args.active:
        asyncio.run(demo_active_backtests())
    else:
        asyncio.run(run_all_demos())