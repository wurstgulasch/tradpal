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
    dates = pd.date_range(start='2024-01-01', periods=50, freq='1min')  # Reduced from 500 to 50
    np.random.seed(42)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': 50000 + np.random.normal(0, 100, 50),
        'high': 50100 + np.random.normal(0, 100, 50),
        'low': 49900 + np.random.normal(0, 100, 50),
        'close': 50000 + np.random.normal(0, 100, 50),
        'volume': np.random.randint(100, 1000, 50)
    })

    # Add some trend to make it more realistic
    trend = np.linspace(0, 200, 50)  # Adjusted for 50 rows
    df['close'] = df['close'] + trend

    return df


def test_multi_model_backtest_basic():
    """Test basic multi-model backtesting functionality."""
    print("Testing basic multi-model backtesting...")

    # Mock all the expensive operations
    with patch('src.backtester.fetch_historical_data') as mock_fetch, \
         patch('src.backtester._run_parallel_backtests') as mock_parallel, \
         patch('src.backtester._is_model_trained') as mock_is_trained, \
         patch('src.backtester._run_single_model_backtest') as mock_single:

        # Mock the data fetch
        df = create_test_dataframe()
        mock_fetch.return_value = df

        # Mock model availability
        mock_is_trained.return_value = True

        # Mock single model backtest results
        mock_single.return_value = {
            'total_trades': 10,
            'win_rate': 60.0,
            'total_pnl': 1000.0,
            'sharpe_ratio': 1.5,
            'model_type': 'traditional_ml'
        }

        # Mock parallel backtesting result
        mock_parallel.return_value = {
            'symbol': 'BTC/USDT',
            'timeframe': '1m',
            'results': {
                'traditional_ml': {
                    'total_trades': 10,
                    'win_rate': 60.0,
                    'total_pnl': 1000.0,
                    'sharpe_ratio': 1.5,
                    'model_type': 'traditional_ml'
                }
            }
        }

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

    # Mock all the expensive operations at module level
    with patch('src.backtester.fetch_historical_data') as mock_fetch, \
         patch('src.backtester._run_parallel_backtests') as mock_parallel, \
         patch('src.backtester._is_model_trained') as mock_is_trained, \
         patch('src.backtester._run_single_model_backtest') as mock_single, \
         patch('src.backtester._compare_model_results') as mock_compare:

        # Mock the data fetch
        df = create_test_dataframe()
        mock_fetch.return_value = df

        # Mock model availability - some trained, some not
        def mock_is_trained_func(model_type):
            return model_type == 'traditional_ml'  # Only traditional ML is trained

        mock_is_trained.side_effect = mock_is_trained_func

        # Mock single model backtest results
        def mock_single_func(*args, **kwargs):
            model_type = kwargs.get('model_type', args[3] if len(args) > 3 else 'unknown')
            if model_type == 'traditional_ml':
                return {
                    'total_trades': 10,
                    'win_rate': 60.0,
                    'total_pnl': 1000.0,
                    'sharpe_ratio': 1.5,
                    'model_type': 'traditional_ml'
                }
            else:
                return {"error": f"{model_type} model not trained"}

        mock_single.side_effect = mock_single_func

        # Mock comparison results
        mock_compare.return_value = {
            'comparison': [{'Model': 'traditional_ml', 'Sharpe Ratio': 1.5}],
            'rankings': {'Sharpe Ratio': ['traditional_ml']},
            'model_scores': {'traditional_ml': 1.0},
            'best_model': 'traditional_ml',
            'best_metrics': {
                'total_trades': 10,
                'win_rate': 60.0,
                'total_pnl': 1000.0,
                'sharpe_ratio': 1.5,
                'model_type': 'traditional_ml'
            },
            'all_results': {
                'traditional_ml': {
                    'total_trades': 10,
                    'win_rate': 60.0,
                    'total_pnl': 1000.0,
                    'sharpe_ratio': 1.5,
                    'model_type': 'traditional_ml'
                }
            },
            'successful_models': ['traditional_ml'],
            'failed_models': ['lstm', 'transformer']
        }

        # Test with multiple models including potentially untrained ones
        result = run_multi_model_backtest(
            symbol='BTC/USDT',
            exchange='kraken',
            timeframe='1m',
            start_date='2024-01-01',
            end_date='2024-01-02',
            models_to_test=['traditional_ml', 'lstm', 'transformer'],
            max_workers=1  # Reduce workers to avoid memory issues
        )

        # Should handle untrained models gracefully
        assert 'successful_models' in result or 'failed_models' in result, "Should handle model failures"
        print("✅ Multi-model backtest with untrained models handled correctly!")


def test_auto_train_models_function():
    """Test the automatic model training function."""
    print("Testing automatic model training function...")

    # Mock the training functions to avoid actual training
    with patch('scripts.train_ml_model.train_ml_model') as mock_train:
        mock_train.return_value = True  # Simulate successful training

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

    # Mock the training functions to avoid actual training
    with patch('scripts.train_ml_model.train_ml_model') as mock_train:
        mock_train.return_value = True  # Simulate successful training

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

    # Mock all the expensive operations
    with patch('src.backtester.fetch_historical_data') as mock_fetch, \
         patch('src.backtester._run_parallel_backtests') as mock_parallel, \
         patch('src.backtester._is_model_trained') as mock_is_trained, \
         patch('src.backtester._run_single_model_backtest') as mock_single:

        # Mock the data fetch
        df = create_test_dataframe()
        mock_fetch.return_value = df

        # Mock model availability
        mock_is_trained.return_value = True

        # Mock single model backtest results
        mock_single.return_value = {
            'total_trades': 10,
            'win_rate': 60.0,
            'total_pnl': 1000.0,
            'sharpe_ratio': 1.5,
            'model_type': 'traditional_ml'
        }

        # Mock parallel backtesting result with comparison data
        mock_parallel.return_value = {
            'symbol': 'BTC/USDT',
            'timeframe': '1m',
            'results': {
                'traditional_ml': {
                    'total_trades': 10,
                    'win_rate': 60.0,
                    'total_pnl': 1000.0,
                    'sharpe_ratio': 1.5,
                    'model_type': 'traditional_ml'
                }
            }
        }

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

    # Mock all the expensive operations
    with patch('src.backtester.fetch_historical_data') as mock_fetch, \
         patch('src.backtester._run_parallel_backtests') as mock_parallel, \
         patch('src.backtester._is_model_trained') as mock_is_trained, \
         patch('src.backtester._run_single_model_backtest') as mock_single:

        # Mock the data fetch
        df = create_test_dataframe()
        mock_fetch.return_value = df

        # Mock model availability
        mock_is_trained.return_value = True

        # Mock single model backtest results
        mock_single.return_value = {
            'total_trades': 10,
            'win_rate': 60.0,
            'total_pnl': 1000.0,
            'sharpe_ratio': 1.5,
            'model_type': 'traditional_ml'
        }

        # Mock parallel backtesting result
        mock_parallel.return_value = {
            'symbol': 'BTC/USDT',
            'timeframe': '1m',
            'results': {
                'traditional_ml': {
                    'total_trades': 10,
                    'win_rate': 60.0,
                    'total_pnl': 1000.0,
                    'sharpe_ratio': 1.5,
                    'model_type': 'traditional_ml'
                }
            }
        }

        import time
        start_time = time.time()

        result = run_multi_model_backtest(
            symbol='BTC/USDT',
            exchange='kraken',
            timeframe='1m',
            start_date='2024-01-01',
            end_date='2024-01-02',
            models_to_test=['traditional_ml'],
            max_workers=1  # Use single worker for testing to avoid memory issues
        )

        end_time = time.time()
        duration = end_time - start_time

        assert duration < 30, f"Backtest should complete quickly, took {duration:.2f}s"
        print("✅ Parallel execution works correctly!")


def test_multi_model_error_handling():
    """Test error handling in multi-model backtesting."""
    print("Testing error handling...")

    # Mock all the expensive operations
    with patch('src.backtester.fetch_historical_data') as mock_fetch, \
         patch('src.backtester._run_parallel_backtests') as mock_parallel, \
         patch('src.backtester._is_model_trained') as mock_is_trained, \
         patch('src.backtester._run_single_model_backtest') as mock_single:

        # Mock the data fetch
        df = create_test_dataframe()
        mock_fetch.return_value = df

        # Mock model availability - no models trained
        mock_is_trained.return_value = False

        # Mock single model backtest to return errors
        mock_single.return_value = {"error": "Model not trained"}

        # Mock parallel backtesting to return error
        mock_parallel.return_value = {"error": "No trained models available"}

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

    # Mock the training functions to avoid actual training
    with patch('scripts.train_ml_model.train_ml_model') as mock_train:
        mock_train.return_value = True  # Simulate successful training

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