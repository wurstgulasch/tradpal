#!/usr/bin/env python3
"""
Discovery Service Demo - Demonstration of genetic algorithm optimization.

This script demonstrates:
- Starting optimization runs
- Monitoring optimization progress
- Retrieving optimization results
- Event-driven optimization tracking
"""

import asyncio
import json
import time
from datetime import datetime

# Import service
from services.discovery_service.service import DiscoveryService, EventSystem

async def demo_single_optimization():
    """Demonstrate single optimization run."""
    print("🧬 Running Single Optimization Demo")
    print("-" * 40)

    event_system = EventSystem()
    service = DiscoveryService(event_system=event_system)

    # Run optimization
    result = await service.run_optimization_async(
        optimization_id="demo_single_opt",
        symbol="BTC/USDT",
        timeframe="1d",
        start_date="2024-01-01",
        end_date="2024-12-31",
        population_size=20,  # Small for demo
        generations=5,      # Small for demo
        use_walk_forward=True
    )

    if result["success"]:
        print("✅ Optimization completed successfully!")
        print(f"   Optimization ID: {result['optimization_id']}")
        print(f"   Best Fitness: {result['best_fitness']:.2f}")
        print(f"   Total Evaluations: {result['total_evaluations']}")
        print(f"   Duration: {result['duration_seconds']:.1f} seconds")

        if result.get('best_config'):
            config = result['best_config']
            print(f"   Best Configuration: {config.get('combination_name', 'Unknown')}")
            if 'ema' in config and config['ema'].get('enabled'):
                print(f"   EMA Periods: {config['ema']['periods']}")
            if 'rsi' in config and config['rsi'].get('enabled'):
                print(f"   RSI Period: {config['rsi']['period']}")

    else:
        print(f"❌ Optimization failed: {result.get('error')}")

    return result

async def demo_optimization_status_monitoring():
    """Demonstrate monitoring optimization status."""
    print("\n🔍 Running Status Monitoring Demo")
    print("-" * 40)

    event_system = EventSystem()
    service = DiscoveryService(event_system=event_system)

    # Start optimization
    optimization_id = "demo_status_monitor"

    # Start optimization in background
    task = asyncio.create_task(service.run_optimization_async(
        optimization_id=optimization_id,
        symbol="BTC/USDT",
        timeframe="1d",
        start_date="2024-01-01",
        end_date="2024-12-31",
        population_size=30,
        generations=10,
        use_walk_forward=False  # Faster for demo
    ))

    # Monitor status
    print("Monitoring optimization progress...")
    for i in range(15):  # Monitor for up to 15 checks
        status = await service.get_optimization_status(optimization_id)

        if status.get('status') == 'completed':
            print("✅ Optimization completed!")
            print(f"   Best Fitness: {status['best_fitness']:.2f}")
            print(f"   Duration: {status['duration_seconds']:.1f}s")
            break
        elif status.get('status') == 'running':
            print(f"🔄 Status check {i+1}: Running...")
        elif status.get('status') == 'failed':
            print(f"❌ Optimization failed: {status.get('error_message')}")
            break

        await asyncio.sleep(2)  # Wait 2 seconds between checks

    # Wait for completion if still running
    if not task.done():
        await task

    return await service.get_optimization_status(optimization_id)

async def demo_multiple_optimizations():
    """Demonstrate running multiple optimizations."""
    print("\n🎯 Running Multiple Optimizations Demo")
    print("-" * 40)

    event_system = EventSystem()
    service = DiscoveryService(event_system=event_system)

    # Define different optimization scenarios
    scenarios = [
        {
            "id": "demo_short_term",
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-06-01",
            "end_date": "2024-12-31",
            "population_size": 15,
            "generations": 3
        },
        {
            "id": "demo_long_term",
            "symbol": "BTC/USDT",
            "timeframe": "1d",
            "start_date": "2023-01-01",
            "end_date": "2024-12-31",
            "population_size": 20,
            "generations": 5
        }
    ]

    results = []

    for scenario in scenarios:
        print(f"Starting optimization: {scenario['id']}")
        print(f"   Symbol: {scenario['symbol']}, Timeframe: {scenario['timeframe']}")
        print(f"   Period: {scenario['start_date']} to {scenario['end_date']}")

        result = await service.run_optimization_async(
            optimization_id=scenario["id"],
            symbol=scenario["symbol"],
            timeframe=scenario["timeframe"],
            start_date=scenario["start_date"],
            end_date=scenario["end_date"],
            population_size=scenario["population_size"],
            generations=scenario["generations"],
            use_walk_forward=True
        )

        if result["success"]:
            print(f"✅ {scenario['id']} completed - Best Fitness: {result['best_fitness']:.2f}")
        else:
            print(f"❌ {scenario['id']} failed: {result.get('error')}")

        results.append(result)

    # Compare results
    print("\n📊 Comparison of Optimization Results:")
    print("-" * 50)

    successful_results = [r for r in results if r["success"]]
    if successful_results:
        for result in successful_results:
            opt_id = result['optimization_id']
            fitness = result['best_fitness']
            duration = result['duration_seconds']
            print(f"   {opt_id}: Fitness={fitness:.2f}, Duration={duration:.1f}s")

        best_result = max(successful_results, key=lambda x: x['best_fitness'])
        print(f"\n🏆 Best Overall: {best_result['optimization_id']} (Fitness: {best_result['best_fitness']:.2f})")

    return results

async def demo_event_driven_optimization():
    """Demonstrate event-driven optimization tracking."""
    print("\n📡 Running Event-Driven Optimization Demo")
    print("-" * 40)

    event_system = EventSystem()
    service = DiscoveryService(event_system=event_system)

    # Event tracking
    events_received = []

    async def handle_started(event):
        events_received.append(event)
        print(f"🚀 Optimization started: {event.data['optimization_id']}")

    async def handle_completed(event):
        events_received.append(event)
        data = event.data
        print(f"✅ Optimization completed: {data['optimization_id']}")
        print(f"   Best Fitness: {data['best_fitness']:.2f}")
        print(f"   Duration: {data['duration']:.1f}s")

    async def handle_failed(event):
        events_received.append(event)
        data = event.data
        print(f"❌ Optimization failed: {data['optimization_id']} - {data['error']}")

    # Subscribe to events
    event_system.subscribe("discovery.optimization.started", handle_started)
    event_system.subscribe("discovery.optimization.completed", handle_completed)
    event_system.subscribe("discovery.optimization.failed", handle_failed)

    # Run optimization
    result = await service.run_optimization_async(
        optimization_id="demo_event_driven",
        symbol="BTC/USDT",
        timeframe="1d",
        start_date="2024-01-01",
        end_date="2024-06-30",
        population_size=25,
        generations=8,
        use_walk_forward=True
    )

    # Wait a bit for all events to be processed
    await asyncio.sleep(0.5)

    print(f"\n📊 Event Summary:")
    print(f"   Events Received: {len(events_received)}")
    event_types = [event.type for event in events_received]
    print(f"   Event Types: {event_types}")

    return result, events_received

async def demo_active_optimizations():
    """Demonstrate listing and managing active optimizations."""
    print("\n📋 Running Active Optimizations Demo")
    print("-" * 40)

    event_system = EventSystem()
    service = DiscoveryService(event_system=event_system)

    # Start multiple optimizations
    optimization_ids = []

    for i in range(3):
        opt_id = f"demo_active_{i+1}"
        optimization_ids.append(opt_id)

        # Start optimization (will run in background)
        asyncio.create_task(service.run_optimization_async(
            optimization_id=opt_id,
            symbol="BTC/USDT",
            timeframe="1d",
            start_date="2024-01-01",
            end_date="2024-03-01",
            population_size=15,
            generations=3,
            use_walk_forward=False
        ))

    # Monitor active optimizations
    print("Monitoring active optimizations...")
    for check in range(10):
        active = await service.list_active_optimizations()
        print(f"Check {check+1}: {len(active)} active optimizations")

        if len(active) == 0:
            print("All optimizations completed!")
            break

        await asyncio.sleep(3)

    # Final status check
    final_active = await service.list_active_optimizations()
    print(f"Final active count: {len(final_active)}")

    return final_active

async def demo_indicator_combinations():
    """Demonstrate available indicator combinations."""
    print("\n📊 Running Indicator Combinations Demo")
    print("-" * 40)

    service = DiscoveryService()

    combinations = service.INDICATOR_COMBINATIONS

    print(f"Available Indicator Combinations: {len(combinations)}")
    print("\nSample Combinations:")
    print("-" * 30)

    # Show first 10 combinations
    for i, (key, combo) in enumerate(list(combinations.items())[:10]):
        print(f"{i+1:2d}. {combo['name']}")
        print(f"      Indicators: {', '.join(combo['indicators'])}")

    if len(combinations) > 10:
        print(f"      ... and {len(combinations) - 10} more combinations")

    print(f"\n💡 Total Possible Configurations: ~{len(combinations) * 1000:,}")  # Rough estimate
    print("   (Each combination has ~1000 parameter variations)")

    return combinations

async def run_all_demos():
    """Run all demo functions."""
    print("🚀 Discovery Service Demo Suite")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    demos = [
        ("Single Optimization", demo_single_optimization),
        ("Status Monitoring", demo_optimization_status_monitoring),
        ("Multiple Optimizations", demo_multiple_optimizations),
        ("Event-Driven Tracking", demo_event_driven_optimization),
        ("Active Optimizations", demo_active_optimizations),
        ("Indicator Combinations", demo_indicator_combinations)
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
    results_file = f"output/discovery_service_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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

    parser = argparse.ArgumentParser(description="Discovery Service Demo")
    parser.add_argument("--single", action="store_true", help="Run only single optimization demo")
    parser.add_argument("--status", action="store_true", help="Run only status monitoring demo")
    parser.add_argument("--multiple", action="store_true", help="Run only multiple optimizations demo")
    parser.add_argument("--events", action="store_true", help="Run only event-driven demo")
    parser.add_argument("--active", action="store_true", help="Run only active optimizations demo")
    parser.add_argument("--combinations", action="store_true", help="Run only indicator combinations demo")

    args = parser.parse_args()

    if args.single:
        asyncio.run(demo_single_optimization())
    elif args.status:
        asyncio.run(demo_optimization_status_monitoring())
    elif args.multiple:
        asyncio.run(demo_multiple_optimizations())
    elif args.events:
        asyncio.run(demo_event_driven_optimization())
    elif args.active:
        asyncio.run(demo_active_optimizations())
    elif args.combinations:
        asyncio.run(demo_indicator_combinations())
    else:
        asyncio.run(run_all_demos())