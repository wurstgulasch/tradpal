#!/usr/bin/env python3
"""
Test script for advanced ML signal enhancement integration.
Tests the integration of LSTM and Transformer models into the signal enhancement system.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.signal_generator import apply_ml_signal_enhancement
from config.settings import SYMBOL, TIMEFRAME

def create_test_data():
    """Create synthetic trading data for testing."""
    # Generate 100 data points
    dates = [datetime.now() - timedelta(minutes=i) for i in range(100, 0, -1)]

    # Create OHLCV data with some trends
    np.random.seed(42)
    base_price = 50000
    prices = []
    current_price = base_price

    for i in range(100):
        # Add some trend and volatility
        trend = 0.001 * np.sin(i / 10)  # Sine wave trend
        noise = np.random.normal(0, 0.01)  # Random noise
        change = trend + noise
        current_price *= (1 + change)
        prices.append(current_price)

    # Create OHLCV from prices
    data = []
    for i, price in enumerate(prices):
        high = price * (1 + abs(np.random.normal(0, 0.005)))
        low = price * (1 - abs(np.random.normal(0, 0.005)))
        open_price = prices[i-1] if i > 0 else price * (1 + np.random.normal(0, 0.002))
        volume = np.random.uniform(100, 1000)

        data.append({
            'timestamp': dates[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume,
            'symbol': SYMBOL
        })

    df = pd.DataFrame(data)
    return df

def add_technical_indicators(df):
    """Add basic technical indicators for testing."""
    # Simple moving averages
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()

    # RSI (simplified)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['BB_middle'] = df['close'].rolling(window=20).mean()
    df['BB_upper'] = df['BB_middle'] + 2 * df['close'].rolling(window=20).std()
    df['BB_lower'] = df['BB_middle'] - 2 * df['close'].rolling(window=20).std()

    # MACD (simplified)
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()

    # ATR (simplified)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()

    return df

def add_traditional_signals(df):
    """Add traditional trading signals for testing."""
    df['Buy_Signal'] = 0
    df['Sell_Signal'] = 0

    # Simple crossover signals
    df.loc[(df['SMA_20'] > df['SMA_50']) & (df['SMA_20'].shift() <= df['SMA_50'].shift()), 'Buy_Signal'] = 1
    df.loc[(df['SMA_20'] < df['SMA_50']) & (df['SMA_20'].shift() >= df['SMA_50'].shift()), 'Sell_Signal'] = 1

    # RSI signals
    df.loc[df['RSI'] < 30, 'Buy_Signal'] = 1
    df.loc[df['RSI'] > 70, 'Sell_Signal'] = 1

    return df

def test_ml_enhancement():
    """Test the ML signal enhancement with advanced models."""
    print("🧪 Testing Advanced ML Signal Enhancement Integration")
    print("=" * 60)

    try:
        # Create test data
        print("📊 Creating test data...")
        df = create_test_data()
        df = add_technical_indicators(df)
        df = add_traditional_signals(df)

        # Remove NaN values
        df = df.dropna().reset_index(drop=True)

        print(f"✅ Created {len(df)} test data points")
        print(f"📈 Traditional signals: {df['Buy_Signal'].sum()} BUY, {df['Sell_Signal'].sum()} SELL")

        # Apply ML enhancement
        print("\n🤖 Applying ML signal enhancement...")

        # Debug: Check ML availability
        from config.settings import ML_ENABLED
        from src.ml_predictor import is_ml_available, is_lstm_available, is_transformer_available
        from src.ml_predictor import get_ml_predictor, get_lstm_predictor, get_transformer_predictor

        print(f"ML_ENABLED: {ML_ENABLED}")
        print(f"ML available: {is_ml_available()}")
        print(f"LSTM available: {is_lstm_available()}")
        print(f"Transformer available: {is_transformer_available()}")

        # Check if predictors are trained
        ml_pred = get_ml_predictor()
        lstm_pred = get_lstm_predictor()
        transformer_pred = get_transformer_predictor()

        print(f"ML predictor trained: {ml_pred.is_trained if ml_pred else False}")
        print(f"LSTM predictor trained: {lstm_pred.is_trained if lstm_pred else False}")
        print(f"Transformer predictor trained: {transformer_pred.is_trained if transformer_pred else False}")

        # For testing purposes, manually mark models as trained
        # (in real usage, models would be trained separately)
        if ml_pred:
            ml_pred.is_trained = True
            print("✅ ML predictor marked as trained (mock)")
        if lstm_pred:
            lstm_pred.is_trained = True
            print("✅ LSTM predictor marked as trained (mock)")
        if transformer_pred:
            transformer_pred.is_trained = True
            print("✅ Transformer predictor marked as trained (mock)")

        enhanced_df = apply_ml_signal_enhancement(df.copy())

        print(f"Enhanced DataFrame columns: {list(enhanced_df.columns)}")

        # Analyze results
        print("\n📊 Enhancement Results:")
        print(f"Total signals: {len(enhanced_df)}")

        # Count signal sources
        signal_sources = enhanced_df['Signal_Source'].value_counts()
        print(f"Signal sources: {dict(signal_sources)}")

        # Count enhanced signals
        enhanced_signals = enhanced_df[enhanced_df['Signal_Source'] != 'TRADITIONAL']
        print(f"Enhanced signals: {len(enhanced_signals)}")

        # Show confidence distribution
        if 'ML_Confidence' in enhanced_df.columns:
            avg_confidence = enhanced_df['ML_Confidence'].mean()
            max_confidence = enhanced_df['ML_Confidence'].max()
            print(".3f")
            print(".3f")

        # Show ensemble votes
        if 'Ensemble_Vote' in enhanced_df.columns:
            ensemble_signals = enhanced_df[enhanced_df['Ensemble_Vote'].abs() > 0.1]
            print(f"Ensemble signals (>0.1): {len(ensemble_signals)}")

        # Show sample enhanced signals
        print("\n🔍 Sample Enhanced Signals:")
        sample_signals = enhanced_df[enhanced_df['Signal_Source'] != 'TRADITIONAL'].head(3)
        for idx, row in sample_signals.iterrows():
            print(f"  Row {idx}: {row['Signal_Source']} -> {row['Enhanced_Signal']} "
                  f"(Confidence: {row.get('ML_Confidence', 0):.3f}, "
                  f"Ensemble: {row.get('Ensemble_Vote', 0):.3f})")

        print("\n✅ ML Enhancement Integration Test Completed Successfully!")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ml_enhancement()
    sys.exit(0 if success else 1)