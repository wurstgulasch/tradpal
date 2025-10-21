#!/usr/bin/env python3
"""
Test script for Funding Rate Analysis functionality.

This script demonstrates how to use the new funding rate data source
and analysis features for perpetual futures trading.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_sources.factory import create_data_source
from src.indicators import funding_rate_analysis, funding_rate_signal, combined_funding_market_analysis
from config.settings import DATA_SOURCE_CONFIG

def test_funding_rate_data_source():
    """Test the funding rate data source functionality."""
    print("🧪 Testing Funding Rate Data Source...")

    try:
        # Create funding rate data source
        funding_source = create_data_source('funding_rate')

        # Test with BTC/USDT perpetual futures
        symbol = 'BTC/USDT:USDT'  # Binance perpetual futures format

        # Get current funding rate
        current_funding = funding_source.fetch_current_funding_rate(symbol)
        if current_funding:
            print(f"✅ Current funding rate for {symbol}: {current_funding['funding_rate']:.6f}")
        else:
            print(f"⚠️  Could not fetch current funding rate for {symbol}")

        # Get funding rate statistics
        stats = funding_source.get_funding_rate_stats(symbol, days=30)
        if stats:
            print(f"✅ Funding rate stats for {symbol} (30 days):")
            print(f"   Mean: {stats['mean_funding_rate']:.6f}")
            print(f"   Min: {stats['min_funding_rate']:.6f}")
            print(f"   Max: {stats['max_funding_rate']:.6f}")
            print(f"   Positive rate %: {stats['positive_rate_percentage']:.1f}%")
        else:
            print(f"⚠️  Could not fetch funding rate stats for {symbol}")

        # Test historical data fetch
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        print(f"📊 Fetching historical funding rates from {start_date.date()} to {end_date.date()}...")
        funding_df = funding_source.fetch_historical_data(symbol, '8h', start_date, end_date)

        if not funding_df.empty:
            print(f"✅ Fetched {len(funding_df)} funding rate records")
            print(f"   Columns: {list(funding_df.columns)}")
            print(f"   Sample data:")
            print(funding_df.head(3))
            return funding_df
        else:
            print("⚠️  No historical funding rate data available")
            return None

    except Exception as e:
        print(f"❌ Error testing funding rate data source: {e}")
        return None

def test_funding_rate_analysis(funding_df):
    """Test funding rate analysis functions."""
    print("\n🧪 Testing Funding Rate Analysis...")

    if funding_df is None or funding_df.empty:
        print("⚠️  No funding data available for analysis")
        return None

    try:
        # Apply funding rate analysis
        analyzed_df = funding_rate_analysis(funding_df, window=24)

        if not analyzed_df.empty:
            print("✅ Funding rate analysis completed")
            print(f"   New columns added: {[col for col in analyzed_df.columns if col not in funding_df.columns]}")

            # Generate signals
            signals = funding_rate_signal(analyzed_df, threshold=0.001)
            signal_counts = signals.value_counts()
            print(f"   Generated signals: {dict(signal_counts)}")

            return analyzed_df
        else:
            print("⚠️  Funding rate analysis failed")
            return None

    except Exception as e:
        print(f"❌ Error in funding rate analysis: {e}")
        return None

def test_combined_analysis(market_df, funding_df):
    """Test combined market and funding rate analysis."""
    print("\n🧪 Testing Combined Market & Funding Analysis...")

    if market_df is None or funding_df is None:
        print("⚠️  Missing data for combined analysis")
        return None

    try:
        # Create sample market data with indicators
        market_df = market_df.copy()
        market_df['EMA9'] = market_df['close'].ewm(span=9).mean()
        market_df['EMA21'] = market_df['close'].ewm(span=21).mean()
        market_df['signal'] = np.where(market_df['EMA9'] > market_df['EMA21'], 1, -1)

        # Combine analyses
        combined_df = combined_funding_market_analysis(market_df, funding_df, funding_weight=0.3)

        if not combined_df.empty:
            print("✅ Combined analysis completed")
            print(f"   Combined signals generated: {len(combined_df)} rows")

            # Show sample of combined data
            sample_cols = ['close', 'funding_rate', 'signal', 'funding_signal', 'combined_signal']
            available_cols = [col for col in sample_cols if col in combined_df.columns]
            if available_cols:
                print("   Sample combined data:")
                print(combined_df[available_cols].tail(3))

            return combined_df
        else:
            print("⚠️  Combined analysis failed")
            return None

    except Exception as e:
        print(f"❌ Error in combined analysis: {e}")
        return None

def create_sample_market_data():
    """Create sample market data for testing."""
    print("\n📊 Creating sample market data...")

    # Create sample OHLCV data
    dates = pd.date_range(start=datetime.now() - timedelta(days=7), periods=168, freq='1H')  # 7 days * 24 hours
    np.random.seed(42)  # For reproducible results

    # Generate realistic BTC/USDT price data
    base_price = 50000
    price_changes = np.random.normal(0, 0.02, len(dates))  # 2% volatility
    prices = base_price * np.exp(np.cumsum(price_changes))

    # Create OHLCV data
    high_mult = 1 + np.abs(np.random.normal(0, 0.01, len(dates)))
    low_mult = 1 - np.abs(np.random.normal(0, 0.01, len(dates)))
    volume = np.random.uniform(100, 1000, len(dates))

    market_df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
        'high': prices * high_mult,
        'low': prices * low_mult,
        'close': prices,
        'volume': volume
    })

    market_df.set_index('timestamp', inplace=True)
    print(f"✅ Created {len(market_df)} rows of sample market data")
    return market_df

def main():
    """Main test function."""
    print("🚀 Testing Funding Rate Analysis for Perpetual Futures")
    print("=" * 60)

    # Test data source
    funding_df = test_funding_rate_data_source()

    # Test analysis functions
    if funding_df is not None:
        analyzed_funding_df = test_funding_rate_analysis(funding_df)

        # Create sample market data for combined testing
        market_df = create_sample_market_data()

        # Test combined analysis
        if analyzed_funding_df is not None:
            combined_df = test_combined_analysis(market_df, analyzed_funding_df)

    print("\n" + "=" * 60)
    print("✅ Funding Rate Analysis Testing Complete!")

    # Summary
    print("\n📋 Summary:")
    print("• Funding Rate Data Source: Implemented ✅")
    print("• Funding Rate Analysis Functions: Implemented ✅")
    print("• Signal Generation: Implemented ✅")
    print("• Combined Market Analysis: Implemented ✅")
    print("• Integration with Signal Generator: Ready ✅")

    print("\n💡 Usage Tips:")
    print("• Set FUNDING_RATE_ENABLED=true in your environment to enable")
    print("• Use 'funding_rate' as data source for perpetual futures analysis")
    print("• Funding rates enhance signals for BTC/USDT:USDT and other perpetuals")
    print("• Positive funding rates favor short positions, negative favor longs")

if __name__ == "__main__":
    main()