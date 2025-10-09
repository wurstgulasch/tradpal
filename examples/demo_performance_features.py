#!/usr/bin/env python
"""
Example script demonstrating the new performance enhancements.

This script shows how to use:
1. WebSocket data streaming for real-time data
2. Parallel backtesting for multiple symbols
3. Redis caching for improved performance
"""

import os
import sys
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.backtester import run_multi_symbol_backtest
from src.cache import HybridCache, get_cache_stats
from src.data_fetcher import fetch_data_realtime
from config.settings import WEBSOCKET_DATA_ENABLED, REDIS_ENABLED, PARALLEL_BACKTESTING_ENABLED


def demo_parallel_backtesting():
    """Demonstrate parallel backtesting."""
    print("\n" + "="*60)
    print("Parallel Backtesting Demo")
    print("="*60)
    
    if not PARALLEL_BACKTESTING_ENABLED:
        print("⚠️  Parallel backtesting is disabled.")
        return
    
    print("\nRunning backtests for multiple symbols...")
    
    symbols = ['BTC/USDT', 'ETH/USDT']
    
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Timeframe: 1h | Period: Last 7 days")
    
    try:
        start_time = datetime.now()
        
        results = run_multi_symbol_backtest(
            symbols=symbols,
            exchange='kraken',
            timeframe='1h',
            start_date=datetime.now() - timedelta(days=7),
            end_date=datetime.now(),
            initial_capital=10000,
            max_workers=2
        )
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        if 'error' in results:
            print(f"❌ Error: {results['error']}")
            return
        
        print(f"\n✓ Completed in {elapsed:.2f} seconds")
        
        summary = results.get('summary', {})
        print(f"\nSummary: {summary.get('successful', 0)}/{summary.get('total_symbols', 0)} successful")
        
        agg = results.get('aggregated_metrics', {})
        if agg:
            print(f"Avg Return: {agg.get('average_return_pct', 0):.2f}%")
            print(f"Avg Win Rate: {agg.get('average_win_rate', 0):.2f}%")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def demo_redis_caching():
    """Demonstrate Redis caching."""
    print("\n" + "="*60)
    print("Redis Caching Demo")
    print("="*60)
    
    status = "Enabled ✓" if REDIS_ENABLED else "Disabled (using file cache)"
    print(f"Status: {status}")
    
    try:
        cache = HybridCache(cache_dir="cache/demo", ttl_seconds=60)
        
        # Test caching
        test_data = {'price': 50000, 'volume': 1234567}
        cache.set('test_key', test_data)
        retrieved = cache.get('test_key')
        
        print(f"\n✓ Cache test: {test_data == retrieved}")
        
        stats = get_cache_stats()
        print(f"Cache entries: {sum(stats.values())}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Run demos."""
    print("\n" + "="*60)
    print("TradPal Indicator - Performance Enhancements Demo")
    print("="*60)
    
    print(f"\nWebSocket: {'ON' if WEBSOCKET_DATA_ENABLED else 'OFF'}")
    print(f"Parallel Backtesting: {'ON' if PARALLEL_BACKTESTING_ENABLED else 'OFF'}")
    print(f"Redis: {'ON' if REDIS_ENABLED else 'OFF'}")
    
    demo_parallel_backtesting()
    demo_redis_caching()
    
    print("\n" + "="*60)
    print("Demo Complete! See docs/PERFORMANCE_ENHANCEMENTS.md")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
