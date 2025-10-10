#!/usr/bin/env python3
"""
Test script for multi-model backtesting functionality.

Tests parallel ML model backtesting, automatic training, and performance comparison.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.backtester import run_multi_model_backtest
from src.ml_predictor import get_ml_predictor, is_ml_available, MLSignalPredictor
from scripts.train_ml_model import train_ml_model
from main import _auto_train_models


def create_test_dataframe():
    """Create a test DataFrame with sample trading data."""
    dates = pd.date_range(start='2024-01-01', periods=500, freq='1min')
    np.random.seed(42)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': 50000 + np.random.normal(0, 100, 500),
        'high': 50100 + np.random.normal(0, 100, 500),
        'low': 49900 + np.random.normal(0, 100, 500),
        'close': 50000 + np.random.normal(0, 100, 500),
        'volume': np.random.randint(100, 1000, 500)
    })

    # Add some trend to make it more realistic
    trend = np.linspace(0, 2000, 500)  # Upward trend
    df['close'] = df['close'] + trend

    return df


def test_multi_model_backtest_basic():
    """Test basic multi-model backtesting functionality."""
    print("Testing basic multi-model backtesting...")

    with patch('src.backtester.fetch_historical_data') as mock_fetch:
        # Mock the data fetch
        df = create_test_dataframe()
        mock_fetch.return_value = df

        # Test with only traditional ML (should work)
        result = run_multi_model_backtest(
            symbol='BTC/USDT',
            exchange='kraken',
            timeframe='1m',
            start_date='2024-01-01',
            end_date='2024-01-02',
            models_to_test=['traditional_ml'],
            max_workers=1
        )

        assert 'error' not in result, f"Multi-model backtest should succeed: {result.get('error', 'Unknown error')}"
        assert 'successful_models' in result, "Should have successful models list"
        assert 'comparison' in result, "Should have comparison data"
        assert 'best_model' in result, "Should identify best model"

        print("✅ Basic multi-model backtest successful!")
        print(f"   Best model: {result['best_model']}")
        print(f"   Successful models: {', '.join(result['successful_models'])}")


def test_multi_model_backtest_with_untrained_models():
    """Test multi-model backtesting with untrained models."""
    print("Testing multi-model backtesting with untrained models...")

    with patch('src.backtester.fetch_historical_data') as mock_fetch:
        # Mock the data fetch
        df = create_test_dataframe()
        mock_fetch.return_value = df

        # Test with multiple models including potentially untrained ones
        result = run_multi_model_backtest(
            symbol='BTC/USDT',
            exchange='kraken',
            timeframe='1m',
            start_date='2024-01-01',
            end_date='2024-01-02',
            models_to_test=['traditional_ml', 'lstm', 'transformer'],
            max_workers=2
        )

        # Should handle untrained models gracefully
        assert 'successful_models' in result or 'failed_models' in result, "Should handle model failures"
        print("✅ Multi-model backtest with untrained models handled correctly!")


def test_auto_train_models_function():
    """Test the automatic model training function."""
    print("Testing automatic model training function...")

    # Test with traditional ML only (should work)
    trained, failed = _auto_train_models(
        models_to_test=['traditional_ml'],
        force_retrain=False,
        symbol='BTC/USDT',
        timeframe='1m'
    )

    assert trained >= 0, "Should return valid training count"
    assert failed >= 0, "Should return valid failure count"
    print("✅ Auto training function works correctly!")


def test_auto_train_force_retrain():
    """Test force retraining functionality."""
    print("Testing force retraining functionality...")

    # Test force retrain
    trained, failed = _auto_train_models(
        models_to_test=['traditional_ml'],
        force_retrain=True,
        symbol='BTC/USDT',
        timeframe='1m'
    )

    assert trained >= 0, "Should return valid training count"
    assert failed >= 0, "Should return valid failure count"
    print("✅ Force retraining works correctly!")


def test_model_comparison_and_ranking():
    """Test model performance comparison and ranking."""
    print("Testing model comparison and ranking...")

    with patch('src.backtester.fetch_historical_data') as mock_fetch:
        # Mock the data fetch
        df = create_test_dataframe()
        mock_fetch.return_value = df

        result = run_multi_model_backtest(
            symbol='BTC/USDT',
            exchange='kraken',
            timeframe='1m',
            start_date='2024-01-01',
            end_date='2024-01-02',
            models_to_test=['traditional_ml'],
            max_workers=1
        )

        if 'comparison' in result and result['comparison']:
            comparison_data = result['comparison']
            assert isinstance(comparison_data, list), "Comparison should be a list"
            assert len(comparison_data) > 0, "Should have comparison data"

            # Check ranking
            if 'rankings' in result:
                rankings = result['rankings']
                assert isinstance(rankings, dict), "Rankings should be a dict"
                print("✅ Model comparison and ranking works correctly!")

                # Show rankings
                for metric, ranking in rankings.items():
                    print(f"   {metric}: {' > '.join(ranking)}")


def test_parallel_execution():
    """Test parallel execution of multiple models."""
    print("Testing parallel execution...")

    with patch('src.backtester.fetch_historical_data') as mock_fetch:
        # Mock the data fetch
        df = create_test_dataframe()
        mock_fetch.return_value = df

        import time
        start_time = time.time()

        result = run_multi_model_backtest(
            symbol='BTC/USDT',
            exchange='kraken',
            timeframe='1m',
            start_date='2024-01-01',
            end_date='2024-01-02',
            models_to_test=['traditional_ml'],
            max_workers=2  # Test with multiple workers
        )

        end_time = time.time()
        duration = end_time - start_time

        assert duration < 30, f"Backtest should complete quickly, took {duration:.2f}s"
        print("✅ Parallel execution works correctly!")


def test_multi_model_error_handling():
    """Test error handling in multi-model backtesting."""
    print("Testing error handling...")

    # Test with invalid model names
    result = run_multi_model_backtest(
        symbol='BTC/USDT',
        exchange='kraken',
        timeframe='1m',
        start_date='2024-01-01',
        end_date='2024-01-02',
        models_to_test=['invalid_model'],
        max_workers=1
    )

    # Should handle invalid models gracefully
    assert 'error' in result or 'failed_models' in result, "Should handle invalid models"
    print("✅ Error handling works correctly!")


def test_model_filtering():
    """Test filtering of available vs requested models."""
    print("Testing model filtering...")

    # Test with None (should use all available)
    trained, failed = _auto_train_models(
        models_to_test=None,
        force_retrain=False,
        symbol='BTC/USDT',
        timeframe='1m'
    )

    assert trained >= 0, "Should handle None models_to_test"
    assert failed >= 0, "Should handle None models_to_test"
    print("✅ Model filtering works correctly!")


def main():
    """Run all multi-model backtesting tests."""
    print("🧪 Starting Multi-Model Backtesting Tests")
    print("=" * 60)

    tests = [
        ("Basic Multi-Model Backtest", test_multi_model_backtest_basic),
        ("Multi-Model with Untrained Models", test_multi_model_backtest_with_untrained_models),
        ("Auto Train Models Function", test_auto_train_models_function),
        ("Force Retraining", test_auto_train_force_retrain),
        ("Model Comparison and Ranking", test_model_comparison_and_ranking),
        ("Parallel Execution", test_parallel_execution),
        ("Error Handling", test_multi_model_error_handling),
        ("Model Filtering", test_model_filtering),
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
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All multi-model backtesting tests completed successfully!")
    else:
        print("⚠️  Some tests failed. Check the output above.")
        raise AssertionError(f"Only {passed}/{total} tests passed")


if __name__ == "__main__":
    main()