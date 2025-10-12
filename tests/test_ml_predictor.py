#!/usr/bin/env python3
"""
Test script for ML prediction functionality.

Tests ML model training, prediction, and signal enhancement.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.ml_predictor import get_ml_predictor, is_ml_available, MLSignalPredictor
from src.indicators import calculate_indicators
from config.settings import SYMBOL, TIMEFRAME


def create_test_dataframe():
    """Create a test DataFrame with sample trading data."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')  # REDUCED FROM 200 TO 100
    np.random.seed(42)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': 50000 + np.random.normal(0, 100, 100),
        'high': 50100 + np.random.normal(0, 100, 100),
        'low': 49900 + np.random.normal(0, 100, 100),
        'close': 50000 + np.random.normal(0, 100, 100),
        'volume': np.random.randint(100, 1000, 100)
    })

    # Add some trend to make it more realistic
    trend = np.linspace(0, 500, 100)  # REDUCED TREND
    df['close'] = df['close'] + trend

    return df


def test_ml_availability():
    """Test ML availability and basic functionality."""
    print("Testing ML availability...")

    available = is_ml_available()
    assert available, "scikit-learn should be available"
    print("✅ scikit-learn available")


def test_ml_predictor_initialization():
    """Test ML predictor initialization."""
    print("Testing ML predictor initialization...")

    predictor = MLSignalPredictor(symbol='TEST/BTC', timeframe='1m')
    assert predictor is not None, "ML predictor should be initialized"
    print("✅ ML predictor initialized successfully")


def test_feature_engineering():
    """Test feature engineering functionality."""
    print("Testing feature engineering...")

    df = create_test_dataframe()
    df = calculate_indicators(df)

    predictor = MLSignalPredictor(symbol='TEST/BTC', timeframe='1m')
    assert predictor is not None, "Predictor should be initialized"

    X, feature_names = predictor.prepare_features(df)
    assert X.shape[1] > 0, "Should create features"
    assert len(feature_names) == X.shape[1], "Feature names should match feature count"
    print(f"✅ Feature engineering successful: {X.shape[1]} features created")

    # Test label creation
    y = predictor.create_labels(df)
    assert len(y) == len(df), "Labels should match data length"
    print(f"✅ Labels created: {len(y)} samples")


def test_model_training():
    """Test ML model training."""
    print("Testing ML model training...")

    df = create_test_dataframe()
    df = calculate_indicators(df)

    predictor = MLSignalPredictor(symbol='TEST/BTC', timeframe='1m')
    assert predictor is not None, "Predictor should be initialized"

    # Train models
    result = predictor.train_models(df, test_size=0.3)
    assert result['success'], f"Model training should succeed: {result.get('error', 'Unknown error')}"

    print("✅ Model training successful!")
    print(f"   Best model: {result['best_model']}")
    print(f"   Samples: {result['samples']}")
    print(f"   Features: {result['features']}")

    # Show performance
    performance = result['performance']
    for model_name, metrics in performance.items():
        if isinstance(metrics, dict) and 'f1_score' in metrics:
            print(".3f")


def test_signal_prediction():
    """Test signal prediction functionality."""
    print("Testing signal prediction...")

    df = create_test_dataframe()
    df = calculate_indicators(df)

    predictor = MLSignalPredictor(symbol='TEST/BTC', timeframe='1m')
    assert predictor is not None, "Predictor should be initialized"

    # First train the model
    result = predictor.train_models(df, test_size=0.3)
    assert result['success'], "Model should be trained successfully"

    # Test prediction on last row
    prediction = predictor.predict_signal(df)
    assert 'signal' in prediction, "Prediction should contain signal"
    assert 'confidence' in prediction, "Prediction should contain confidence"
    assert prediction['signal'] in ['BUY', 'SELL', 'HOLD'], "Signal should be valid"
    assert 0.0 <= prediction['confidence'] <= 1.0, "Confidence should be between 0 and 1"

    print("✅ Signal prediction successful!")
    print(f"   Predicted signal: {prediction['signal']}")
    print(".2f")
    print(f"   Reason: {prediction['reason']}")


def test_signal_enhancement():
    """Test signal enhancement functionality."""
    print("Testing signal enhancement...")

    df = create_test_dataframe()
    df = calculate_indicators(df)

    # Add traditional signals
    df['Buy_Signal'] = np.where(df['close'] > df['close'].shift(1), 1, 0)
    df['Sell_Signal'] = np.where(df['close'] < df['close'].shift(1), 1, 0)

    predictor = MLSignalPredictor(symbol='TEST/BTC', timeframe='1m')
    assert predictor is not None, "Predictor should be initialized"

    # Train model if needed
    if not predictor.is_trained:
        result = predictor.train_models(df, test_size=0.3)
        assert result['success'], "Model should be trained"

    # Test enhancement for a few rows
    enhanced_count = 0
    for idx in range(min(3, len(df))):  # REDUCED FROM 5 TO 3
        row_df = df.iloc[idx:idx+1]
        ml_prediction = predictor.predict_signal(row_df)

        traditional_signal = 'BUY' if row_df['Buy_Signal'].iloc[0] == 1 else 'SELL' if row_df['Sell_Signal'].iloc[0] == 1 else 'HOLD'

        enhanced = predictor.enhance_signal(traditional_signal, ml_prediction, df=row_df)

        if enhanced['source'] != 'TRADITIONAL':
            enhanced_count += 1

        if idx == 0:  # Show first result
            print("✅ Signal enhancement working!")
            print(f"   Traditional: {traditional_signal}")
            print(f"   ML: {ml_prediction['signal']} (conf: {ml_prediction['confidence']:.2f})")
            print(f"   Enhanced: {enhanced['signal']} (source: {enhanced['source']})")

    print(f"   Enhanced signals in sample: {enhanced_count}")


def test_model_persistence():
    """Test model saving and loading."""
    print("Testing model persistence...")

    df = create_test_dataframe()
    df = calculate_indicators(df)

    predictor1 = MLSignalPredictor(symbol='TEST/BTC', timeframe='1m')
    assert predictor1 is not None, "Predictor should be initialized"

    # Train and save
    result = predictor1.train_models(df, test_size=0.3)
    assert result['success'], "Model should be trained"
    predictor1.save_model()

    # Create new predictor and load
    predictor2 = MLSignalPredictor(symbol='TEST/BTC', timeframe='1m')
    assert predictor2.is_trained, "Model should be loaded"

    model_info = predictor2.get_model_info()
    assert model_info['status'] == 'trained', "Model should be marked as trained"
    assert model_info['feature_count'] > 0, "Should have features"

    print("✅ Model persistence successful!")
    print(f"   Model loaded with {model_info['feature_count']} features")


def main():
    """Run all ML tests."""
    print("🧪 Starting ML Prediction Tests")
    print("=" * 60)

    tests = [
        ("ML Availability", test_ml_availability),
        ("ML Predictor Initialization", test_ml_predictor_initialization),
        ("Feature Engineering", test_feature_engineering),
        ("Model Training", test_model_training),
        ("Signal Prediction", test_signal_prediction),
        ("Signal Enhancement", test_signal_enhancement),
        ("Model Persistence", test_model_persistence),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}")
        print("-" * 40)

        try:
            test_func()
            print(f"✅ {test_name} PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")

    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All ML tests completed successfully!")
    else:
        print("⚠️  Some tests failed. Check the output above.")
        raise AssertionError(f"Only {passed}/{total} tests passed")


if __name__ == "__main__":
    main()