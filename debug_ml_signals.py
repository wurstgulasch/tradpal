#!/usr/bin/env python3
"""
Debug script to check ML signal generation and confidence levels.
"""
import sys
import os
sys.path.append('/Users/danielsadowski/VSCodeProjects/tradpal_indicator')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data_fetcher import fetch_historical_data
from src.indicators import calculate_indicators
from src.signal_generator import generate_signals
from src.ml_predictor import get_ml_predictor, is_ml_available

def debug_ml_signals():
    """Debug ML signal generation."""
    print("🔍 Debugging ML signal generation...")

    # Fetch some data
    symbol = 'BTC/USDT'
    timeframe = '1m'
    limit = 100  # Small sample for debugging

    print(f"📊 Fetching {limit} data points for {symbol} {timeframe}...")
    data = fetch_historical_data(symbol=symbol, timeframe=timeframe, limit=limit)

    if data.empty:
        print("❌ No data available")
        return

    print(f"✅ Fetched {len(data)} data points")

    # Calculate indicators (minimal config for debugging)
    config = {
        'ema': {'enabled': True, 'periods': [9, 21]},
        'rsi': {'enabled': True, 'period': 14, 'oversold': 30, 'overbought': 70},
        'bb': {'enabled': True, 'period': 20, 'std_dev': 2.0}
    }

    print("📈 Calculating indicators...")
    data = calculate_indicators(data, config=config)

    # Generate traditional signals
    print("📊 Generating traditional signals...")
    data = generate_signals(data, config=config)

    # Check traditional signals
    buy_signals = data['Buy_Signal'].sum()
    sell_signals = data['Sell_Signal'].sum()
    print(f"📊 Traditional signals: {buy_signals} BUY, {sell_signals} SELL")

    # Apply ML enhancement
    if is_ml_available():
        predictor = get_ml_predictor(symbol=symbol, timeframe=timeframe)
        if predictor and predictor.is_trained:
            print("🤖 Applying ML enhancement...")

            # Add ML columns
            data['ML_Signal'] = 'HOLD'
            data['ML_Confidence'] = 0.0
            data['Enhanced_Signal'] = 'HOLD'
            data['Signal_Source'] = 'TRADITIONAL'

            # Get predictions for first few rows
            sample_data = data.head(10).copy()
            print(f"🔬 Testing ML predictions on first {len(sample_data)} rows...")

            for idx, row in sample_data.iterrows():
                row_df = pd.DataFrame([row])
                try:
                    prediction = predictor.predict_signal(row_df, threshold=0.5)
                    print(f"Row {idx}: Signal={prediction['signal']}, Confidence={prediction['confidence']:.3f}")
                except Exception as e:
                    print(f"Row {idx}: Prediction failed - {e}")

            # Apply full enhancement
            from src.backtester import Backtester
            backtester = Backtester()
            enhanced_data = backtester._apply_ml_enhancement(data.copy(), predictor, 'ml_enhanced')

            # Check results
            ml_buy_signals = (enhanced_data['Buy_Signal'] == 1).sum()
            ml_sell_signals = (enhanced_data['Sell_Signal'] == 1).sum()
            ml_signals = enhanced_data['ML_Signal'].value_counts()
            confidences = enhanced_data['ML_Confidence'].describe()

            print("📊 ML Enhancement Results:")
            print(f"   Buy signals: {ml_buy_signals}")
            print(f"   Sell signals: {ml_sell_signals}")
            print(f"   ML signal distribution: {ml_signals.to_dict()}")
            print(f"   Confidence stats: mean={confidences['mean']:.3f}, min={confidences['min']:.3f}, max={confidences['max']:.3f}")
            print(f"   High confidence signals (>0.5): {(enhanced_data['ML_Confidence'] > 0.5).sum()}")

        else:
            print("⚠️ ML predictor not available or not trained")
    else:
        print("⚠️ ML not available")

if __name__ == "__main__":
    debug_ml_signals()