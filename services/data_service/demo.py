#!/usr/bin/env python3
"""
Data Service Demo - Demonstration of data fetching, caching, and quality validation.

This script demonstrates:
- Basic data fetching from multiple sources
- Cache performance and management
- Data quality validation
- Fallback mechanisms
- Performance comparisons
"""

import asyncio
import json
import time
from datetime import datetime, timedelta

# Import service
from services.data_service.service import DataService, DataRequest, EventSystem


async def demo_basic_data_fetch():
    """Demonstrate basic data fetching."""
    print("📊 Running Basic Data Fetch Demo")
    print("-" * 40)

    event_system = EventSystem()
    service = DataService(event_system=event_system)

    # Fetch BTC/USDT data
    request = DataRequest(
        symbol="BTC/USDT",
        timeframe="1d",
        start_date="2024-01-01T00:00:00",
        end_date="2024-01-15T00:00:00",
        use_cache=True,
        validate_quality=True
    )

    print(f"Fetching {request.symbol} {request.timeframe} data...")
    print(f"Period: {request.start_date} to {request.end_date}")

    response = await service.fetch_data(request)

    if response.success:
        print("✅ Data fetch successful!")
        print(f"   Records: {len(response.data['ohlcv'])}")
        print(f"   Cache hit: {response.cache_hit}")
        print(".2f")

        if response.metadata:
            meta = response.metadata
            print(f"   Quality: {meta['quality_level']} ({meta['quality_score']:.2f})")
            print(f"   Source: {meta['source']} ({meta['provider']})")
            print(f"   Fallback used: {meta.get('fallback_used', False)}")

        # Show sample data
        ohlcv = response.data['ohlcv']
        sample_keys = list(ohlcv.keys())[:3]
        if sample_keys:
            print(f"   Sample data (first 3 records):")
            for key in sample_keys:
                data = ohlcv[key]
                print(f"     {key}: O={data.get('open', 'N/A'):.2f}, H={data.get('high', 'N/A'):.2f}, "
                      f"L={data.get('low', 'N/A'):.2f}, C={data.get('close', 'N/A'):.2f}, V={data.get('volume', 'N/A'):.0f}")

    else:
        print(f"❌ Data fetch failed: {response.error}")

    return response


async def demo_cache_performance():
    """Demonstrate cache performance benefits."""
    print("\n⚡ Running Cache Performance Demo")
    print("-" * 40)

    event_system = EventSystem()
    service = DataService(event_system=event_system)

    request = DataRequest(
        symbol="BTC/USDT",
        timeframe="1d",
        start_date="2024-01-01T00:00:00",
        end_date="2024-01-31T00:00:00"
    )

    print("Testing cache performance...")

    # First fetch (cache miss)
    print("1. First fetch (cache miss):")
    start_time = time.time()
    response1 = await service.fetch_data(request)
    first_fetch_time = time.time() - start_time

    if response1.success:
        print(".2f")
        print(f"   Cache hit: {response1.cache_hit}")

        # Second fetch (cache hit)
        print("2. Second fetch (cache hit):")
        start_time = time.time()
        response2 = await service.fetch_data(request)
        second_fetch_time = time.time() - start_time

        print(".2f")
        print(f"   Cache hit: {response2.cache_hit}")

        # Calculate speedup
        if second_fetch_time > 0:
            speedup = first_fetch_time / second_fetch_time
            print(".1f")

            if speedup > 2:
                print("   🎉 Significant performance improvement!")
            elif speedup > 1.5:
                print("   👍 Good performance improvement")
            else:
                print("   🤔 Moderate performance improvement")
        else:
            print("   ⚡ Instant cache retrieval!")

    else:
        print(f"❌ Fetch failed: {response1.error}")

    return response1, response2 if 'response2' in locals() else None


async def demo_data_quality_validation():
    """Demonstrate data quality validation."""
    print("\n🔍 Running Data Quality Validation Demo")
    print("-" * 40)

    event_system = EventSystem()
    service = DataService(event_system=event_system)

    symbols_and_periods = [
        ("BTC/USDT", "1d", "2024-01-01T00:00:00", "2024-01-15T00:00:00"),
        ("ETH/USDT", "1d", "2024-01-01T00:00:00", "2024-01-15T00:00:00"),
        ("ADA/USDT", "1h", "2024-01-01T00:00:00", "2024-01-02T00:00:00"),
    ]

    quality_results = []

    for symbol, timeframe, start_date, end_date in symbols_and_periods:
        print(f"Validating {symbol} {timeframe}...")

        request = DataRequest(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            validate_quality=True
        )

        response = await service.fetch_data(request)

        if response.success and response.metadata:
            meta = response.metadata
            quality_level = meta['quality_level']
            quality_score = meta['quality_score']
            record_count = meta['record_count']

            quality_results.append({
                'symbol': symbol,
                'timeframe': timeframe,
                'quality_level': quality_level,
                'quality_score': quality_score,
                'record_count': record_count
            })

            print(f"   ✅ Quality: {quality_level} ({quality_score:.2f}) - {record_count} records")
        else:
            print(f"   ❌ Failed: {response.error}")
            quality_results.append({
                'symbol': symbol,
                'timeframe': timeframe,
                'quality_level': 'failed',
                'quality_score': 0.0,
                'record_count': 0
            })

    # Summary
    print("\n📊 Quality Summary:")
    print("-" * 30)

    excellent = sum(1 for r in quality_results if r['quality_level'] == 'excellent')
    good = sum(1 for r in quality_results if r['quality_level'] == 'good')
    fair = sum(1 for r in quality_results if r['quality_level'] == 'fair')
    poor = sum(1 for r in quality_results if r['quality_level'] == 'poor')
    failed = sum(1 for r in quality_results if r['quality_level'] == 'failed')

    print(f"   Excellent: {excellent}")
    print(f"   Good: {good}")
    print(f"   Fair: {fair}")
    print(f"   Poor: {poor}")
    print(f"   Failed: {failed}")

    if excellent + good >= len(quality_results) * 0.8:
        print("   🎉 High quality data overall!")
    elif excellent + good >= len(quality_results) * 0.6:
        print("   👍 Acceptable data quality")
    else:
        print("   ⚠️  Data quality concerns detected")

    return quality_results


async def demo_fallback_mechanisms():
    """Demonstrate automatic fallback mechanisms."""
    print("\n🔄 Running Fallback Mechanisms Demo")
    print("-" * 40)

    event_system = EventSystem()
    service = DataService(event_system=event_system)

    # Test different source preferences
    test_cases = [
        {
            "name": "Prefer CCXT (Binance)",
            "symbol": "BTC/USDT",
            "source": "ccxt",
            "provider": "binance"
        },
        {
            "name": "Prefer Yahoo Finance",
            "symbol": "AAPL",
            "source": "yahoo",
            "provider": "yahoo"
        },
        {
            "name": "Auto-fallback",
            "symbol": "BTC/USDT",
            "source": None,
            "provider": None
        }
    ]

    fallback_results = []

    for test_case in test_cases:
        print(f"Testing: {test_case['name']}")

        request = DataRequest(
            symbol=test_case["symbol"],
            timeframe="1d",
            start_date="2024-01-01T00:00:00",
            end_date="2024-01-05T00:00:00",
            source=test_case["source"],
            provider=test_case["provider"]
        )

        response = await service.fetch_data(request)

        if response.success and response.metadata:
            meta = response.metadata
            source_used = meta['source']
            provider_used = meta['provider']
            fallback_used = meta.get('fallback_used', False)

            fallback_results.append({
                'test': test_case['name'],
                'source_used': source_used,
                'provider_used': provider_used,
                'fallback_used': fallback_used,
                'success': True
            })

            print(f"   ✅ Source: {source_used} ({provider_used}) - Fallback: {fallback_used}")
        else:
            print(f"   ❌ Failed: {response.error}")
            fallback_results.append({
                'test': test_case['name'],
                'source_used': None,
                'provider_used': None,
                'fallback_used': False,
                'success': False
            })

    # Summary
    print("\n🔄 Fallback Summary:")
    print("-" * 30)

    successful_fallbacks = sum(1 for r in fallback_results if r['fallback_used'])
    total_successful = sum(1 for r in fallback_results if r['success'])

    print(f"   Successful fetches: {total_successful}/{len(fallback_results)}")
    print(f"   Fallbacks used: {successful_fallbacks}")

    if successful_fallbacks > 0:
        print("   🎉 Fallback mechanisms working!")
    else:
        print("   ℹ️  No fallbacks needed (direct success)")

    return fallback_results


async def demo_concurrent_fetching():
    """Demonstrate concurrent data fetching."""
    print("\n🔥 Running Concurrent Fetching Demo")
    print("-" * 40)

    event_system = EventSystem()
    service = DataService(event_system=event_system)

    # Define multiple requests
    symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT", "DOT/USDT"]
    requests = []

    for symbol in symbols:
        request = DataRequest(
            symbol=symbol,
            timeframe="1d",
            start_date="2024-01-01T00:00:00",
            end_date="2024-01-10T00:00:00"
        )
        requests.append(request)

    print(f"Fetching data for {len(symbols)} symbols concurrently...")

    start_time = time.time()

    # Execute concurrently
    tasks = [service.fetch_data(req) for req in requests]
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    total_time = time.time() - start_time

    # Analyze results
    successful = 0
    total_records = 0

    print("\n📈 Results:")
    for i, (symbol, response) in enumerate(zip(symbols, responses)):
        if isinstance(response, Exception):
            print(f"   {symbol}: ❌ Exception - {str(response)}")
        elif response.success:
            records = len(response.data.get('ohlcv', {}))
            total_records += records
            successful += 1
            print(f"   {symbol}: ✅ {records} records")
        else:
            print(f"   {symbol}: ❌ Failed - {response.error}")

    print("\n⏱️  Performance:")
    print(".2f")
    print(".1f")
    print(".2f")

    if successful == len(symbols):
        print("   🎉 All concurrent fetches successful!")
    elif successful >= len(symbols) * 0.75:
        print("   👍 Most fetches successful")
    else:
        print("   ⚠️  Some fetches failed")

    return responses


async def demo_cache_management():
    """Demonstrate cache management operations."""
    print("\n💾 Running Cache Management Demo")
    print("-" * 40)

    event_system = EventSystem()
    service = DataService(event_system=event_system)

    # First, fetch some data to populate cache
    print("Populating cache...")
    request = DataRequest(
        symbol="BTC/USDT",
        timeframe="1d",
        start_date="2024-01-01T00:00:00",
        end_date="2024-01-31T00:00:00"
    )

    response = await service.fetch_data(request)
    if not response.success:
        print(f"❌ Could not populate cache: {response.error}")
        return

    print("✅ Cache populated")

    # Get cache stats
    print("\n📊 Cache Statistics:")
    try:
        # This would work with real Redis - for demo we'll show the concept
        info = await service.get_data_info("BTC/USDT", "1d")
        print(f"   Cache enabled: {info.get('cache_enabled', False)}")
        print("   Cache operations available: Clear, Stats")
    except Exception as e:
        print(f"   Cache stats not available: {e}")

    # Demonstrate cache clearing
    print("\n🧹 Cache Management:")
    print("   Clearing cache entries...")

    try:
        deleted = await service.clear_cache("data:*")
        print(f"   ✅ Cleared {deleted} cache entries")
    except Exception as e:
        print(f"   Cache clear failed: {e}")

    return True


async def demo_health_check():
    """Demonstrate health check functionality."""
    print("\n🏥 Running Health Check Demo")
    print("-" * 40)

    event_system = EventSystem()
    service = DataService(event_system=event_system)

    print("Performing health check...")
    health = await service.health_check()

    print(f"Overall Status: {health['status']}")
    print(f"Timestamp: {health['timestamp']}")

    print("\n🔧 Component Status:")
    for component, status in health['components'].items():
        status_icon = "✅" if status in ["available", "connected"] else "❌" if status in ["error", "not_available"] else "⚠️"
        print(f"   {component}: {status_icon} {status}")

    # Overall assessment
    components = health['components']
    all_healthy = all(
        status in ["available", "connected"]
        for status in components.values()
    )

    if all_healthy:
        print("\n🎉 All components healthy!")
    elif health['status'] == 'healthy':
        print("\n👍 Service is healthy (some optional components unavailable)")
    else:
        print("\n⚠️  Service has health issues")

    return health


async def run_all_demos():
    """Run all demo functions."""
    print("🚀 Data Service Demo Suite")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    demos = [
        ("Basic Data Fetch", demo_basic_data_fetch),
        ("Cache Performance", demo_cache_performance),
        ("Data Quality Validation", demo_data_quality_validation),
        ("Fallback Mechanisms", demo_fallback_mechanisms),
        ("Concurrent Fetching", demo_concurrent_fetching),
        ("Cache Management", demo_cache_management),
        ("Health Check", demo_health_check)
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
    elif successful >= total * 0.8:
        print("👍 Most demos completed successfully")
    else:
        print("⚠️  Some demos failed - check output above")

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Save results
    results_file = f"output/data_service_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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

    parser = argparse.ArgumentParser(description="Data Service Demo")
    parser.add_argument("--basic", action="store_true", help="Run only basic data fetch demo")
    parser.add_argument("--cache", action="store_true", help="Run only cache performance demo")
    parser.add_argument("--quality", action="store_true", help="Run only data quality demo")
    parser.add_argument("--fallback", action="store_true", help="Run only fallback demo")
    parser.add_argument("--concurrent", action="store_true", help="Run only concurrent fetching demo")
    parser.add_argument("--management", action="store_true", help="Run only cache management demo")
    parser.add_argument("--health", action="store_true", help="Run only health check demo")

    args = parser.parse_args()

    if args.basic:
        asyncio.run(demo_basic_data_fetch())
    elif args.cache:
        asyncio.run(demo_cache_performance())
    elif args.quality:
        asyncio.run(demo_data_quality_validation())
    elif args.fallback:
        asyncio.run(demo_fallback_mechanisms())
    elif args.concurrent:
        asyncio.run(demo_concurrent_fetching())
    elif args.management:
        asyncio.run(demo_cache_management())
    elif args.health:
        asyncio.run(demo_health_check())
    else:
        asyncio.run(run_all_demos())