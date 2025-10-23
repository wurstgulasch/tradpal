#!/usr/bin/env python3
"""
Test script for Market Regime Detection integration.
Validates the advanced ML-based regime detection works correctly with the ML trainer.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Set up environment
os.environ.setdefault('PYTHONPATH', project_root)

from services.trading_service.trading_ai_service.ml_training.ml_trainer import MLTrainerService
from services.trading_service.trading_ai_service.ml_training.market_regime_detector import MarketRegimeDetector


def generate_test_data(n_samples: int = 500) -> pd.DataFrame:
    """Generate synthetic market data for testing."""
    np.random.seed(42)

    # Generate dates
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_samples)]

    # Generate price data with different regimes
    prices = []
    current_price = 100.0

    for i in range(n_samples):
        # Simulate different market regimes
        if i < 150:  # Bull market
            change = np.random.normal(0.001, 0.02)
        elif i < 300:  # Bear market
            change = np.random.normal(-0.001, 0.025)
        elif i < 400:  # High volatility
            change = np.random.normal(0.0005, 0.04)
        else:  # Sideways
            change = np.random.normal(0.0001, 0.015)

        current_price *= (1 + change)
        prices.append(current_price)

    # Create OHLC data
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_samples)
    })

    df.set_index('timestamp', inplace=True)

    # Add returns column
    df['returns'] = df['close'].pct_change().fillna(0)

    return df


def test_market_regime_detector():
    """Test the MarketRegimeDetector independently."""
    print("Testing MarketRegimeDetector...")

    # Generate test data with required technical indicators
    df = generate_test_data(500)

    # Add required technical indicators that the detector expects
    df['adx'] = 25  # Default ADX value
    df['volatility_20'] = df['returns'].rolling(20).std().fillna(0.02)
    df['rsi'] = 50  # Neutral RSI

    # Initialize detector
    detector = MarketRegimeDetector()

    # Test regime detection
    try:
        regimes_df = detector.detect_market_regimes(df)
        print(f"✓ Regime detection successful. Shape: {regimes_df.shape}")
        print(f"✓ Detected regimes: {regimes_df['market_regime'].unique()}")
        print(f"✓ Confidence range: {regimes_df['regime_confidence'].min():.3f} - {regimes_df['regime_confidence'].max():.3f}")

        # Check for required columns (use actual column names from the detector)
        required_cols = ['market_regime', 'regime_confidence', 'regime_label']
        missing_cols = [col for col in required_cols if col not in regimes_df.columns]
        if missing_cols:
            print(f"✗ Missing columns: {missing_cols}")
            print(f"Available columns: {list(regimes_df.columns)}")
            return False
        else:
            print("✓ All required columns present")

        return True

    except Exception as e:
        print(f"✗ Regime detection failed: {e}")
        return False


def test_ml_trainer_integration():
    """Test ML trainer integration with regime detection."""
    print("\nTesting ML Trainer integration...")

    # Generate test data
    df = generate_test_data(500)

    # Initialize trainer
    trainer = MLTrainerService()

    # Test regime indicator addition
    try:
        enhanced_df = trainer._add_market_regime_indicators(df)
        print(f"✓ ML trainer regime integration successful. Shape: {enhanced_df.shape}")

        # Check for regime columns
        regime_cols = [col for col in enhanced_df.columns if 'regime' in col.lower()]
        if regime_cols:
            print(f"✓ Found regime columns: {regime_cols}")
        else:
            print("✗ No regime columns found")
            return False

        # Check for numeric regime indicators
        numeric_cols = ['trend_strength_num', 'volatility_regime_num', 'momentum_regime_num', 'trend_direction_num']
        missing_numeric = [col for col in numeric_cols if col not in enhanced_df.columns]
        if missing_numeric:
            print(f"✗ Missing numeric regime columns: {missing_numeric}")
            return False
        else:
            print("✓ All numeric regime indicators present")

        return True

    except Exception as e:
        print(f"✗ ML trainer integration failed: {e}")
        return False


def test_fallback_mechanism():
    """Test that fallback mechanism works when advanced detection fails."""
    print("\nTesting fallback mechanism...")

    # Generate minimal data that might cause advanced detection to fail
    df = generate_test_data(50)  # Too small for ML training

    trainer = MLTrainerService()

    try:
        # Test fallback method directly
        enhanced_df = trainer._add_basic_regime_indicators(df)
        print(f"✓ Fallback mechanism successful. Shape: {enhanced_df.shape}")

        # Check that basic indicators are present
        basic_cols = ['trend_strength', 'volatility_regime', 'momentum_regime', 'trend_direction']
        missing_basic = [col for col in basic_cols if col not in enhanced_df.columns]
        if missing_basic:
            print(f"✗ Missing basic regime columns: {missing_basic}")
            return False
        else:
            print("✓ Basic regime indicators present")

        return True

    except Exception as e:
        print(f"✗ Fallback mechanism failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Market Regime Detection Integration Test")
    print("=" * 50)

    results = []

    # Test individual components
    results.append(("MarketRegimeDetector", test_market_regime_detector()))
    results.append(("ML Trainer Integration", test_ml_trainer_integration()))
    results.append(("Fallback Mechanism", test_fallback_mechanism()))

    # Summary
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\nPassed: {passed}/{len(results)} tests")

    if passed == len(results):
        print("🎉 All tests passed! Market Regime Detection is ready.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())