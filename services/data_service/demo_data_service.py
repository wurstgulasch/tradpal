#!/usr/bin/env python3
"""
Data Service Demo Script

Demonstrates the core functionality of the TradPal Data Service:
- Market data fetching with caching
- Data quality validation
- Alternative data collection
- Market regime detection
- Performance optimization features

Usage:
    python demo_data_service.py

Requirements:
    - Data Service running on localhost:8001
    - Redis running for caching
    - Optional: GPU support for performance demos
"""

import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
import pandas as pd

# Import the data service client
try:
    from services.data_service.client import DataServiceClient
    DATA_SERVICE_AVAILABLE = True
except ImportError:
    print("❌ DataServiceClient not available. Make sure the data service is properly installed.")
    DATA_SERVICE_AVAILABLE = False

# Fallback for demo purposes
class MockDataServiceClient:
    """Mock client for demonstration when service is not available"""

    async def fetch_market_data(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Mock market data fetching"""
        print(f"📊 Mock: Fetching market data for {symbol}")
        # Simulate some delay
        await asyncio.sleep(0.1)

        # Return mock OHLCV data
        return {
            "symbol": symbol,
            "data": [
                {"timestamp": "2025-01-01T00:00:00Z", "open": 50000, "high": 51000, "low": 49500, "close": 50500, "volume": 1000},
                {"timestamp": "2025-01-01T01:00:00Z", "open": 50500, "high": 52000, "low": 50000, "close": 51500, "volume": 1200},
                {"timestamp": "2025-01-01T02:00:00Z", "open": 51500, "high": 52500, "low": 51000, "close": 52200, "volume": 1100},
            ],
            "quality_score": 0.95,
            "source": "mock_data",
            "cached": False
        }

    async def get_data_quality(self, symbol: str) -> Dict[str, Any]:
        """Mock data quality check"""
        print(f"🔍 Mock: Checking data quality for {symbol}")
        await asyncio.sleep(0.05)
        return {
            "symbol": symbol,
            "quality_score": 0.92,
            "completeness": 0.98,
            "accuracy": 0.95,
            "timeliness": 0.89,
            "issues": []
        }

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Mock cache statistics"""
        print("📈 Mock: Getting cache statistics")
        await asyncio.sleep(0.02)
        return {
            "total_entries": 150,
            "hit_rate": 0.87,
            "miss_rate": 0.13,
            "memory_usage_mb": 45.2,
            "avg_response_time_ms": 12.5
        }

    async def fetch_alternative_data(self, data_type: str, symbol: str = None) -> Dict[str, Any]:
        """Mock alternative data fetching"""
        print(f"📊 Mock: Fetching {data_type} data for {symbol or 'global'}")
        await asyncio.sleep(0.15)

        if data_type == "sentiment":
            return {
                "symbol": symbol,
                "sentiment_score": 0.65,
                "confidence": 0.82,
                "sources": ["twitter", "reddit", "news"],
                "timestamp": datetime.now().isoformat()
            }
        elif data_type == "onchain":
            return {
                "symbol": symbol,
                "metrics": {
                    "hash_rate": 450.5,
                    "difficulty": 85.2,
                    "active_addresses": 1250000,
                    "transaction_volume": 450000
                },
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {"error": f"Unknown data type: {data_type}"}

    async def detect_market_regime(self, symbol: str) -> Dict[str, Any]:
        """Mock market regime detection"""
        print(f"🎯 Mock: Detecting market regime for {symbol}")
        await asyncio.sleep(0.1)
        return {
            "symbol": symbol,
            "regime": "bull_market",
            "confidence": 0.78,
            "features": {
                "volatility": 0.25,
                "trend_strength": 0.82,
                "momentum": 0.65
            },
            "timestamp": datetime.now().isoformat()
        }


async def demo_market_data_fetching(client):
    """Demonstrate market data fetching with caching"""
    print("\n" + "="*60)
    print("📊 MARKET DATA FETCHING DEMO")
    print("="*60)

    symbol = "BTC/USDT"

    # First fetch (cache miss)
    print(f"\n🔄 First fetch for {symbol} (should be cache miss)")
    start_time = time.time()
    result1 = await client.fetch_market_data(symbol, timeframe="1h", limit=10)
    first_fetch_time = time.time() - start_time

    print(f"✅ First fetch completed in {first_fetch_time:.3f}s")
    print(f"   Data points: {len(result1.get('data', []))}")
    print(f"   Quality score: {result1.get('quality_score', 'N/A')}")
    print(f"   Cached: {result1.get('cached', False)}")

    # Second fetch (cache hit)
    print(f"\n⚡ Second fetch for {symbol} (should be cache hit)")
    start_time = time.time()
    result2 = await client.fetch_market_data(symbol, timeframe="1h", limit=10)
    second_fetch_time = time.time() - start_time

    print(f"✅ Second fetch completed in {second_fetch_time:.3f}s")
    print(f"   Cached: {result2.get('cached', False)}")
    print(".2f")


async def demo_data_quality(client):
    """Demonstrate data quality validation"""
    print("\n" + "="*60)
    print("🔍 DATA QUALITY VALIDATION DEMO")
    print("="*60)

    symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT"]

    for symbol in symbols:
        print(f"\n🔍 Checking data quality for {symbol}")
        quality = await client.get_data_quality(symbol)

        print(f"   Overall Quality Score: {quality.get('quality_score', 'N/A'):.2f}")
        print(f"   Completeness: {quality.get('completeness', 'N/A'):.2f}")
        print(f"   Accuracy: {quality.get('accuracy', 'N/A'):.2f}")
        print(f"   Timeliness: {quality.get('timeliness', 'N/A'):.2f}")

        issues = quality.get('issues', [])
        if issues:
            print(f"   Issues found: {len(issues)}")
        else:
            print("   ✅ No quality issues detected")


async def demo_cache_performance(client):
    """Demonstrate cache performance monitoring"""
    print("\n" + "="*60)
    print("📈 CACHE PERFORMANCE DEMO")
    print("="*60)

    stats = await client.get_cache_stats()

    print("\n📊 Cache Statistics:")
    print(f"   Total entries: {stats.get('total_entries', 'N/A')}")
    print(f"   Hit rate: {stats.get('hit_rate', 'N/A'):.2%}")
    print(f"   Miss rate: {stats.get('miss_rate', 'N/A'):.2%}")
    print(f"   Memory usage: {stats.get('memory_usage_mb', 'N/A'):.1f} MB")
    print(f"   Avg response time: {stats.get('avg_response_time_ms', 'N/A'):.1f} ms")


async def demo_alternative_data(client):
    """Demonstrate alternative data collection"""
    print("\n" + "="*60)
    print("📊 ALTERNATIVE DATA COLLECTION DEMO")
    print("="*60)

    # Sentiment analysis
    print("\n😊 Sentiment Analysis for BTC/USDT")
    sentiment = await client.fetch_alternative_data("sentiment", "BTC/USDT")
    print(f"   Sentiment Score: {sentiment.get('sentiment_score', 'N/A'):.2f}")
    print(f"   Confidence: {sentiment.get('confidence', 'N/A'):.2f}")
    print(f"   Sources: {', '.join(sentiment.get('sources', []))}")

    # On-chain data
    print("\n⛓️  On-Chain Metrics for BTC")
    onchain = await client.fetch_alternative_data("onchain", "BTC")
    metrics = onchain.get('metrics', {})
    print(f"   Hash Rate: {metrics.get('hash_rate', 'N/A')}")
    print(f"   Difficulty: {metrics.get('difficulty', 'N/A')}")
    print(f"   Active Addresses: {metrics.get('active_addresses', 'N/A'):,}")
    print(f"   Transaction Volume: {metrics.get('transaction_volume', 'N/A'):,}")


async def demo_market_regime_detection(client):
    """Demonstrate market regime detection"""
    print("\n" + "="*60)
    print("🎯 MARKET REGIME DETECTION DEMO")
    print("="*60)

    symbols = ["BTC/USDT", "ETH/USDT"]

    for symbol in symbols:
        print(f"\n🎯 Analyzing market regime for {symbol}")
        regime = await client.detect_market_regime(symbol)

        print(f"   Current Regime: {regime.get('regime', 'unknown').replace('_', ' ').title()}")
        print(f"   Confidence: {regime.get('confidence', 'N/A'):.2f}")

        features = regime.get('features', {})
        print("   Key Features:")
        print(f"     - Volatility: {features.get('volatility', 'N/A'):.2f}")
        print(f"     - Trend Strength: {features.get('trend_strength', 'N/A'):.2f}")
        print(f"     - Momentum: {features.get('momentum', 'N/A'):.2f}")


async def demo_performance_optimization():
    """Demonstrate performance optimization features"""
    print("\n" + "="*60)
    print("⚡ PERFORMANCE OPTIMIZATION DEMO")
    print("="*60)

    print("\n🔧 Performance Features Available:")
    print("   ✅ Chunked Processing: Large datasets processed in configurable chunks")
    print("   ✅ Memory Mapping: Efficient storage for large time-series data")
    print("   ✅ GPU Acceleration: CUDA/PyTorch acceleration for matrix operations")
    print("   ✅ Async Processing: Non-blocking data operations")
    print("   ✅ Resource Management: Automatic cleanup and memory monitoring")

    print("\n💡 Performance Tips:")
    print("   - Use chunked processing for datasets > 100k rows")
    print("   - Enable memory mapping for frequently accessed historical data")
    print("   - GPU acceleration automatically detected and utilized")
    print("   - Monitor cache hit rates for optimal performance")


async def main():
    """Main demo function"""
    print("🚀 TradPal Data Service Demo")
    print("="*60)
    print("This demo showcases the core functionality of the Data Service")
    print("including market data fetching, quality validation, caching,")
    print("alternative data collection, and market regime detection.")
    print("="*60)

    # Initialize client
    if DATA_SERVICE_AVAILABLE:
        print("\n✅ Using real DataServiceClient")
        client = DataServiceClient()
    else:
        print("\n⚠️  Using MockDataServiceClient for demonstration")
        client = MockDataServiceClient()

    try:
        # Run all demos
        await demo_market_data_fetching(client)
        await demo_data_quality(client)
        await demo_cache_performance(client)
        await demo_alternative_data(client)
        await demo_market_regime_detection(client)
        await demo_performance_optimization()

        print("\n" + "="*60)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("The Data Service provides:")
        print("• High-performance market data fetching")
        print("• Comprehensive data quality assurance")
        print("• Intelligent caching and performance optimization")
        print("• Rich alternative data collection")
        print("• Advanced market regime detection")
        print("\nFor production use, ensure the service is running on localhost:8001")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Make sure the Data Service is running and accessible.")


if __name__ == "__main__":
    # Run the async demo
    asyncio.run(main())