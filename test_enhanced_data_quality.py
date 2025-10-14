#!/usr/bin/env python3
"""
Test Script for Enhanced Data Quality Features

Demonstrates the new fallback system, indicator validation, and quality monitoring.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_fetcher import fetch_historical_data_with_fallback
from src.indicators import calculate_indicators_with_validation
from src.data_quality_monitor import get_data_quality_monitor
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)

def test_fallback_system():
    """Test the automatic fallback system."""
    print("🔄 **Testing Automatic Fallback System**")
    print("=" * 50)

    try:
        # Test with BTC/USDT
        df = fetch_historical_data_with_fallback(
            symbol='BTC/USDT',
            timeframe='1d',
            limit=100,
            show_progress=True
        )

        print(f"✅ Successfully fetched {len(df)} records")
        print(f"   Date range: {df.index.min()} to {df.index.max()}")
        print(f"   Columns: {list(df.columns)}")

        return df

    except Exception as e:
        print(f"❌ Fallback system failed: {e}")
        return None

def test_indicator_validation(df):
    """Test indicator validation system."""
    print("\n🔧 **Testing Indicator Validation**")
    print("=" * 50)

    if df is None or df.empty:
        print("❌ No data available for indicator testing")
        return

    try:
        # Calculate indicators with validation
        valid_indicators = calculate_indicators_with_validation(df)

        print(f"✅ Successfully calculated {len(valid_indicators)} valid indicators:")
        for name, indicator in valid_indicators.items():
            valid_ratio = indicator.notnull().mean() * 100
            print(".1f")

        if len(valid_indicators) < 5:  # We expect at least 5 indicators
            print("⚠️  Some indicators were excluded due to quality issues")

        return valid_indicators

    except Exception as e:
        print(f"❌ Indicator validation failed: {e}")
        return {}

def test_quality_monitoring(df):
    """Test data quality monitoring."""
    print("\n📊 **Testing Data Quality Monitoring**")
    print("=" * 50)

    if df is None or df.empty:
        print("❌ No data available for quality monitoring")
        return

    try:
        monitor = get_data_quality_monitor()
        quality_report = monitor.monitor_data_quality(df, "test_source")  # Use monitor_data_quality instead of get_quality_report
        report_text = monitor.get_quality_report(df, "test_source")

        print(report_text)

        return quality_report

    except Exception as e:
        print(f"❌ Quality monitoring failed: {e}")
        return None

def main():
    """Main test function."""
    print("🧪 **TradPal Enhanced Data Quality Test**")
    print("=" * 60)

    # Test fallback system
    df = test_fallback_system()

    # Test indicator validation
    indicators = test_indicator_validation(df)

    # Test quality monitoring
    quality_report = test_quality_monitoring(df)

    # Summary
    print("\n📋 **Test Summary**")
    print("=" * 30)

    success_count = 0
    total_tests = 3

    if df is not None and not df.empty:
        success_count += 1
        print("✅ Fallback System: PASSED")
    else:
        print("❌ Fallback System: FAILED")

    if indicators and len(indicators) >= 3:
        success_count += 1
        print("✅ Indicator Validation: PASSED")
    else:
        print("❌ Indicator Validation: FAILED")

    if quality_report and quality_report.get('quality_score', 0) > 70:
        success_count += 1
        print("✅ Quality Monitoring: PASSED")
    else:
        print("❌ Quality Monitoring: FAILED")

    print(f"\n🎯 **Overall Result: {success_count}/{total_tests} tests passed**")

    if success_count == total_tests:
        print("🎉 All enhanced data quality features are working correctly!")
    else:
        print("⚠️  Some features need attention")

if __name__ == "__main__":
    main()