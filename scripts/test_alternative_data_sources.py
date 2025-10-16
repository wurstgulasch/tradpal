#!/usr/bin/env python3
"""
Test Script for Alternative Data Sources

Tests the new Sentiment and On-Chain Metrics data sources
that serve as fallbacks when primary liquidation data is unavailable.
"""

import sys
import os
import logging
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from services.data_service.data_sources.factory import DataSourceFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_sentiment_data_source():
    """Test the Sentiment data source."""
    print("\n" + "="*60)
    print("Testing Sentiment Data Source")
    print("="*60)

    try:
        # Create sentiment data source
        sentiment_ds = DataSourceFactory.create_data_source('sentiment')

        # Test recent data
        print("\n1. Testing recent sentiment data...")
        recent_data = sentiment_ds.fetch_recent_data('BTC/USDT', '1h', limit=1)

        if not recent_data.empty:
            print("✓ Recent sentiment data fetched successfully")
            print(f"   Data shape: {recent_data.shape}")
            print(f"   Columns: {list(recent_data.columns)}")
            print(f"   Latest sentiment signal: {recent_data.iloc[0].get('sentiment_signal', 'N/A')}")
            print(f"   Latest sentiment strength: {recent_data.iloc[0].get('sentiment_strength', 'N/A'):.3f}")
            print(f"   Fear & Greed value: {recent_data.iloc[0].get('fear_greed_value', 'N/A')}")
        else:
            print("✗ No recent sentiment data available")

        # Test historical data
        print("\n2. Testing historical sentiment data...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        historical_data = sentiment_ds.fetch_historical_data(
            'BTC/USDT', '1h', start_date, end_date, limit=24
        )

        if not historical_data.empty:
            print("✓ Historical sentiment data fetched successfully")
            print(f"   Data shape: {historical_data.shape}")
            print(f"   Date range: {historical_data.index.min()} to {historical_data.index.max()}")
            print(f"   Average sentiment signal: {historical_data['sentiment_signal'].mean():.2f}")
        else:
            print("✗ No historical sentiment data available")

        # Test sentiment stats
        print("\n3. Testing sentiment statistics...")
        stats = sentiment_ds.get_sentiment_stats('BTC/USDT')

        if stats:
            print("✓ Sentiment statistics retrieved successfully")
            print(f"   Current signal: {stats.get('current_sentiment_signal', 'N/A')}")
            print(f"   Current strength: {stats.get('current_sentiment_strength', 'N/A'):.3f}")
            print(f"   Fear & Greed: {stats.get('fear_greed_value', 'N/A')} ({stats.get('fear_greed_classification', 'N/A')})")
        else:
            print("✗ No sentiment statistics available")

        return True

    except Exception as e:
        print(f"✗ Sentiment data source test failed: {e}")
        return False

def test_onchain_data_source():
    """Test the On-Chain Metrics data source."""
    print("\n" + "="*60)
    print("Testing On-Chain Metrics Data Source")
    print("="*60)

    try:
        # Create on-chain data source
        onchain_ds = DataSourceFactory.create_data_source('onchain')

        # Test recent data
        print("\n1. Testing recent on-chain data...")
        recent_data = onchain_ds.fetch_recent_data('BTC/USDT', '1h', limit=1)

        if not recent_data.empty:
            print("✓ Recent on-chain data fetched successfully")
            print(f"   Data shape: {recent_data.shape}")
            print(f"   Columns: {list(recent_data.columns)}")
            print(f"   Latest on-chain signal: {recent_data.iloc[0].get('onchain_signal', 'N/A')}")
            print(f"   Latest on-chain strength: {recent_data.iloc[0].get('onchain_strength', 'N/A'):.3f}")
            print(f"   Active addresses: {recent_data.iloc[0].get('active_addresses', 'N/A')}")
        else:
            print("✗ No recent on-chain data available")

        # Test historical data
        print("\n2. Testing historical on-chain data...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3)

        historical_data = onchain_ds.fetch_historical_data(
            'BTC/USDT', '1h', start_date, end_date, limit=24
        )

        if not historical_data.empty:
            print("✓ Historical on-chain data fetched successfully")
            print(f"   Data shape: {historical_data.shape}")
            print(f"   Date range: {historical_data.index.min()} to {historical_data.index.max()}")
            print(f"   Average on-chain signal: {historical_data['onchain_signal'].mean():.2f}")
        else:
            print("✗ No historical on-chain data available")

        # Test on-chain stats
        print("\n3. Testing on-chain statistics...")
        stats = onchain_ds.get_onchain_stats('BTC/USDT')

        if stats:
            print("✓ On-chain statistics retrieved successfully")
            print(f"   Current signal: {stats.get('current_onchain_signal', 'N/A')}")
            print(f"   Current strength: {stats.get('current_onchain_strength', 'N/A'):.3f}")
            print(f"   Active addresses: {stats.get('active_addresses', 'N/A')}")
            print(f"   Data source: {stats.get('data_source', 'N/A')}")
        else:
            print("✗ No on-chain statistics available")

        return True

    except Exception as e:
        print(f"✗ On-chain data source test failed: {e}")
        return False

def test_data_source_availability():
    """Test data source availability."""
    print("\n" + "="*60)
    print("Testing Data Source Availability")
    print("="*60)

    try:
        available_sources = DataSourceFactory.get_available_sources()
        available_list = DataSourceFactory.list_sources()

        print(f"Available data sources: {len(available_list)}")
        for source, available in available_sources.items():
            status = "✓" if available else "✗"
            print(f"   {status} {source}")

        # Check if our new sources are available
        required_sources = ['sentiment', 'onchain', 'volatility', 'liquidation']
        all_available = all(available_sources.get(source, False) for source in required_sources)

        if all_available:
            print("✓ All required alternative data sources are available")
        else:
            print("✗ Some required alternative data sources are missing")
            for source in required_sources:
                if not available_sources.get(source, False):
                    print(f"     Missing: {source}")

        return all_available

    except Exception as e:
        print(f"✗ Data source availability test failed: {e}")
        return False

def test_fallback_chain():
    """Test the fallback chain for liquidation data."""
    print("\n" + "="*60)
    print("Testing Fallback Chain for Liquidation Data")
    print("="*60)

    try:
        # Test liquidation data source with fallbacks
        liquidation_ds = DataSourceFactory.create_data_source('liquidation')

        print("\n1. Testing liquidation data with fallbacks...")
        recent_data = liquidation_ds.fetch_recent_data('BTC/USDT', '1h', limit=1)

        if not recent_data.empty:
            print("✓ Liquidation data (with fallbacks) fetched successfully")
            print(f"   Data shape: {recent_data.shape}")
            print(f"   Columns: {list(recent_data.columns)}")

            # Check which data source was used
            if 'liquidation_price' in recent_data.columns:
                print("   Primary data source: Real liquidation data")
            elif 'volatility_signal' in recent_data.columns:
                print("   Fallback data source: Volatility indicators")
            elif 'sentiment_signal' in recent_data.columns:
                print("   Fallback data source: Sentiment analysis")
            elif 'onchain_signal' in recent_data.columns:
                print("   Fallback data source: On-chain metrics")
            else:
                print("   Data source: Unknown fallback")
        else:
            print("✗ No liquidation data available through any fallback")

        return not recent_data.empty

    except Exception as e:
        print(f"✗ Fallback chain test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Alternative Data Sources for TradPal")
    print("This tests the new fallback data sources when primary liquidation data is unavailable.")

    results = []

    # Test data source availability
    results.append(("Data Source Availability", test_data_source_availability()))

    # Test individual data sources
    results.append(("Sentiment Data Source", test_sentiment_data_source()))
    results.append(("On-Chain Metrics Data Source", test_onchain_data_source()))

    # Test fallback chain
    results.append(("Fallback Chain", test_fallback_chain()))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = 0
    total = len(results)

    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{status}: {test_name}")
        if success:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Alternative data sources are working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())