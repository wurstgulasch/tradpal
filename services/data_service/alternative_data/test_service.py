#!/usr/bin/env python3
"""
Test Alternative Data Service

Comprehensive test suite for the Alternative Data Service components.
"""

import sys
import os

# Mock problematic dependencies BEFORE any other imports
sys.modules['transformers'] = type(sys)('transformers')
sys.modules['transformers.pipeline'] = type(sys)('pipeline')
sys.modules['textblob'] = type(sys)('textblob')

# Add the project root to the path to access config and other modules
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now we can use absolute imports
from services.alternative_data_service.sentiment_analyzer import SentimentAnalyzer
from services.alternative_data_service.onchain_collector import OnChainDataCollector
from services.alternative_data_service.economic_collector import EconomicDataCollector
from services.alternative_data_service.data_processor import AlternativeDataProcessor
from services.alternative_data_service import AlternativeDataPacket

async def test_sentiment_analyzer():
    """Test sentiment analysis functionality."""
    print("🧪 Testing Sentiment Analyzer...")

    try:
        analyzer = SentimentAnalyzer()
    except Exception as e:
        print(f"❌ Failed to import SentimentAnalyzer: {e}")
        return False

    try:
        # Test initialization
        await analyzer.initialize()
        print("✅ Sentiment analyzer initialized")

        # Test basic sentiment analysis
        test_text = "Bitcoin is going to the moon! This is amazing!"
        sentiment = await analyzer._analyze_text_sentiment(test_text)
        print(f"✅ Basic sentiment analysis: {sentiment}")

        # Test symbol sentiment analysis
        symbol_sentiment = await analyzer.analyze_symbol_sentiment("BTC/USDT", hours=1)
        print(f"✅ Symbol sentiment analysis: {len(symbol_sentiment)} data points")

        # Test Fear & Greed Index
        fear_greed = await analyzer.get_fear_greed_index()
        print(f"✅ Fear & Greed Index: {fear_greed}")

        # Test metrics
        metrics = await analyzer.get_metrics()
        print(f"✅ Metrics: {metrics}")

        return True

    except Exception as e:
        print(f"❌ Sentiment analyzer test failed: {e}")
        return False


async def test_onchain_collector():
    """Test on-chain data collection."""
    print("🧪 Testing On-Chain Data Collector...")

    try:
        collector = OnChainDataCollector()
    except Exception as e:
        print(f"❌ Failed to import OnChainDataCollector: {e}")
        return False

    try:
        # Test initialization
        await collector.initialize()
        print("✅ On-chain collector initialized")

        # Test metrics collection
        metrics = await collector.get_metrics("BTC")
        print(f"✅ On-chain metrics collected: {len(metrics)} metrics")

        # Test specific metrics
        nvt_data = await collector._get_glassnode_nvt_ratio("BTC")
        print(f"✅ NVT Ratio data: {nvt_data}")

        # Test metrics
        metrics_info = await collector.get_metrics()
        print(f"✅ Collector metrics: {metrics_info}")

        return True

    except Exception as e:
        print(f"❌ On-chain collector test failed: {e}")
        return False


async def test_economic_collector():
    """Test economic data collection."""
    print("🧪 Testing Economic Data Collector...")

    try:
        collector = EconomicDataCollector()
    except Exception as e:
        print(f"❌ Failed to import EconomicDataCollector: {e}")
        return False

    try:
        # Test initialization
        await collector.initialize()
        print("✅ Economic collector initialized")

        # Test indicators collection
        indicators = await collector.get_indicators()
        print(f"✅ Economic indicators collected: {len(indicators)} indicators")

        # Test specific indicators
        fed_rate = await collector._get_fed_funds_rate()
        print(f"✅ Fed Funds Rate: {fed_rate}")

        # Test metrics
        metrics = await collector.get_metrics()
        print(f"✅ Collector metrics: {metrics}")

        return True

    except Exception as e:
        print(f"❌ Economic collector test failed: {e}")
        return False


async def test_data_processor():
    """Test data processing pipeline."""
    print("🧪 Testing Data Processor...")

    try:
        processor = AlternativeDataProcessor()
    except Exception as e:
        print(f"❌ Failed to import DataProcessor: {e}")
        return False

    try:
        # Create test data packet
        test_packet = AlternativeDataPacket(
            symbol="BTC/USDT",
            sentiment_data=[],
            onchain_data=[],
            economic_data=[],
            fear_greed_index=65.0
        )

        # Test feature processing
        features = await processor.process_to_features(test_packet)
        print(f"✅ Feature processing: {len(features.composite_features)} features")

        # Test feature importance
        importance = await processor.get_feature_importance("BTC/USDT")
        print(f"✅ Feature importance: {len(importance)} features analyzed")

        # Test metrics
        metrics = await processor.get_metrics()
        print(f"✅ Processor metrics: {metrics}")

        return True

    except Exception as e:
        print(f"❌ Data processor test failed: {e}")
        return False


async def test_integration():
    """Test full integration of all components."""
    print("🧪 Testing Full Integration...")

    try:
        # Initialize all components
        sentiment_analyzer = SentimentAnalyzer()
        onchain_collector = OnChainDataCollector()
        economic_collector = EconomicDataCollector()
        data_processor = AlternativeDataProcessor()

        await sentiment_analyzer.initialize()
        await onchain_collector.initialize()
        await economic_collector.initialize()

        print("✅ All components initialized")

        # Collect data for BTC
        symbol = "BTC/USDT"

        sentiment_data = await sentiment_analyzer.analyze_symbol_sentiment(symbol)
        onchain_data = await onchain_collector.get_metrics("BTC")
        economic_data = await economic_collector.get_indicators()
        fear_greed = await sentiment_analyzer.get_fear_greed_index()

        print("✅ Data collection completed")

        # Create data packet
        packet = AlternativeDataPacket(
            symbol=symbol,
            sentiment_data=sentiment_data if isinstance(sentiment_data, list) else [sentiment_data],
            onchain_data=onchain_data if isinstance(onchain_data, list) else [onchain_data],
            economic_data=economic_data if isinstance(economic_data, list) else [economic_data],
            fear_greed_index=fear_greed.get('value') if fear_greed else None
        )

        print("✅ Data packet created")

        # Process features
        features = await data_processor.process_to_features(packet)
        print(f"✅ Features processed: {len(features.composite_features)} composite features")

        return True

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


async def run_all_tests():
    """Run all test suites."""
    print("🚀 Starting Alternative Data Service Tests")
    print("=" * 50)

    test_results = []

    # Run individual component tests
    test_results.append(await test_sentiment_analyzer())
    test_results.append(await test_onchain_collector())
    test_results.append(await test_economic_collector())
    test_results.append(await test_data_processor())

    # Run integration test
    test_results.append(await test_integration())

    print("=" * 50)

    passed = sum(test_results)
    total = len(test_results)

    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Alternative Data Service is ready.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    import asyncio
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)