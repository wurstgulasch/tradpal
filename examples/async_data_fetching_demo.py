#!/usr/bin/env python3
"""
Demo script for AsyncDataFetcher functionality.
Shows how to use async/await patterns for improved data fetching performance.
"""

import asyncio
import sys
import os
import time
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_fetcher import AsyncDataFetcher

async def demo_async_data_fetching():
    """Demonstrate async data fetching capabilities."""
    print("🚀 Async Data Fetching Demo")
    print("=" * 50)

    async with AsyncDataFetcher(max_concurrent_requests=3) as fetcher:
        # Demo 1: Fetch single symbol data
        print("\n📊 Demo 1: Fetch single symbol data asynchronously")
        start_time = time.time()

        try:
            df = await fetcher.fetch_data_async(
                limit=100,
                symbol="BTC/USDT",
                timeframe="1h"
            )
            elapsed = time.time() - start_time
            print(f"   ✅ Fetched in {elapsed:.2f}s")
            print(f"   Data shape: {df.shape}")

        except Exception as e:
            print(f"   Error: {e}")

        # Demo 2: Fetch multiple symbols concurrently
        print("\n📊 Demo 2: Fetch multiple symbols concurrently")
        symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT"]
        start_time = time.time()

        try:
            results = await fetcher.fetch_multiple_symbols_async(
                symbols=symbols,
                timeframe="1h",
                limit=50
            )
            elapsed = time.time() - start_time
            print(f"   ✅ Fetched in {elapsed:.2f}s")
            for symbol, df in results.items():
                print(f"   {symbol}: {df.shape}")

        except Exception as e:
            print(f"   Error: {e}")

        # Demo 3: Fetch multiple timeframes for one symbol
        print("\n📊 Demo 3: Fetch multiple timeframes for BTC/USDT")
        timeframes = ["1m", "5m", "15m", "1h"]
        start_date = datetime.now() - timedelta(days=1)
        start_time = time.time()

        try:
            results = await fetcher.fetch_historical_batch_async(
                symbol="BTC/USDT",
                timeframes=timeframes,
                start_date=start_date,
                limit=100
            )
            elapsed = time.time() - start_time
            print(f"   ✅ Fetched in {elapsed:.2f}s")
            for tf, df in results.items():
                print(f"   {tf}: {df.shape}")

        except Exception as e:
            print(f"   Error: {e}")

    print("\n✅ Async data fetching demo completed!")

async def compare_sync_vs_async():
    """Compare synchronous vs asynchronous performance."""
    print("\n🔍 Performance Comparison: Sync vs Async")
    print("=" * 50)

    symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT", "DOT/USDT", "LINK/USDT"]

    # Synchronous approach
    print("Synchronous fetching...")
    sync_start = time.time()
    sync_results = {}

    for symbol in symbols:
        try:
            from src.data_fetcher import fetch_data
            df = fetch_data(limit=50, symbol=symbol, timeframe="1h")
            sync_results[symbol] = df.shape[0] if not df.empty else 0
        except Exception as e:
            sync_results[symbol] = f"Error: {e}"
            continue

    sync_time = time.time() - sync_start

    # Asynchronous approach
    print("Asynchronous fetching...")
    async_start = time.time()

    async with AsyncDataFetcher(max_concurrent_requests=5) as fetcher:
        try:
            async_results = await fetcher.fetch_multiple_symbols_async(
                symbols=symbols,
                timeframe="1h",
                limit=50
            )
            async_results = {k: v.shape[0] if not v.empty else 0 for k, v in async_results.items()}
        except Exception as e:
            async_results = {symbol: f"Error: {e}" for symbol in symbols}

    async_time = time.time() - async_start

    # Results
    print("\nResults:")
    print(f"Sync time: {sync_time:.2f}s")
    print(f"Async time: {async_time:.2f}s")
    speedup = sync_time / async_time if async_time > 0 else float('inf')
    print(f"Speedup: {speedup:.1f}x")
    print("\nData points fetched:")
    for symbol in symbols:
        sync_count = sync_results.get(symbol, 0)
        async_count = async_results.get(symbol, 0)
        print(f"   {symbol}: Sync={sync_count}, Async={async_count}")

if __name__ == "__main__":
    # Run async demo
    asyncio.run(demo_async_data_fetching())

    # Run performance comparison
    asyncio.run(compare_sync_vs_async())