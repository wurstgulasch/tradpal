#!/usr/bin/env python3
"""
Test script for Kelly Criterion position sizing.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from src.signal_generator import calculate_kelly_position_size, calculate_risk_management
from src.data_fetcher import fetch_historical_data
from src.indicators import calculate_indicators

def test_kelly_criterion():
    """Test Kelly Criterion position sizing."""
    print("Testing Kelly Criterion position sizing...")

def test_kelly_criterion():
    """Test Kelly Criterion position sizing."""
    print("Testing Kelly Criterion position sizing...")

    try:
        # Create sample data instead of fetching
        print("📊 Creating sample data...")
        dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
        np.random.seed(42)  # For reproducible results

        # Generate realistic BTC/USDT price data
        base_price = 30000
        prices = []
        current_price = base_price

        for i in range(200):
            # Add some trend and volatility
            trend = 0.001 * np.sin(i / 20)  # Slow trend
            noise = np.random.normal(0, 0.02)  # Daily volatility ~2%
            change = trend + noise
            current_price *= (1 + change)
            prices.append(current_price)

        # Create OHLCV data
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': [np.random.uniform(1000000, 5000000) for _ in range(200)]
        })
        df.set_index('timestamp', inplace=True)

        # Calculate indicators
        print("📈 Calculating technical indicators...")
        df = calculate_indicators(df)

        # Generate some signals for testing
        print("📊 Generating test signals...")
        df['Buy_Signal'] = np.where(df['RSI'] < 30, 1, 0)
        df['Sell_Signal'] = np.where(df['RSI'] > 70, 1, 0)

        # Test Kelly position sizing
        print("🎯 Testing Kelly Criterion...")
        from config.settings import CAPITAL, RISK_PER_TRADE
        kelly_sizes = calculate_kelly_position_size(df, CAPITAL, RISK_PER_TRADE)

        print(f"   Kelly sizes calculated for {len(kelly_sizes)} data points")
        print(".3f")
        print(".3f")
        print(".3f")

        # Test full risk management with Kelly
        print("🔄 Testing full risk management with Kelly...")
        df_risk = calculate_risk_management(df)

        # Show results for signals
        signal_rows = df_risk[(df_risk['Buy_Signal'] == 1) | (df_risk['Sell_Signal'] == 1)].head(5)

        print("\n📊 Sample Risk Management Results:")
        print("Signal | Kelly_Fraction | Position_Size_% | Stop_Loss | Take_Profit")
        print("-" * 70)

        for idx, row in signal_rows.iterrows():
            signal = "BUY" if row['Buy_Signal'] == 1 else "SELL"
            kelly_frac = row.get('Kelly_Fraction', RISK_PER_TRADE)
            pos_size_pct = row.get('Position_Size_Percent', 0)
            sl = row.get('Stop_Loss_Buy', 0)
            tp = row.get('Take_Profit_Buy', 0)

            print(".3f")

        print("\n✅ Kelly Criterion test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_kelly_criterion()
    sys.exit(0 if success else 1)