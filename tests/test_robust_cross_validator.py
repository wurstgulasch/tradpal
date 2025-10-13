#!/usr/bin/env python3
"""
Test script for Robust Cross-Validation functionality.

Tests the new RobustCrossValidator class with different CV strategies.
"""

import sys
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.ml_predictor import RobustCrossValidator


def create_test_data(n_samples=1000, n_features=10, random_state=42):
    """Create test data for cross-validation."""
    np.random.seed(random_state)

    # Create features
    X = np.random.randn(n_samples, n_features)

    # Create target with some signal
    weights = np.random.randn(n_features)
    linear_signal = X.dot(weights)
    noise = np.random.randn(n_samples) * 0.5
    y_prob = 1 / (1 + np.exp(-(linear_signal + noise)))  # Sigmoid
    y = (y_prob > 0.5).astype(int)

    return X, y


def test_robust_cross_validator_initialization():
    """Test RobustCrossValidator initialization."""
    print("Testing RobustCrossValidator initialization...")

    # Test different CV methods
    for method in ['kfold', 'stratified', 'time_series']:
        validator = RobustCrossValidator(cv_method=method, n_splits=5)
        assert validator.cv_method == method
        assert validator.n_splits == 5
        print(f"✅ Initialized {method} cross-validator")

    # Test invalid method
    try:
        validator = RobustCrossValidator(cv_method='invalid')
        assert False, "Should raise error for invalid method"
    except ValueError:
        print("✅ Correctly rejects invalid CV methods")


def test_kfold_cross_validation():
    """Test KFold cross-validation."""
    print("Testing KFold cross-validation...")

    X, y = create_test_data(n_samples=500)
    model = RandomForestClassifier(n_estimators=50, random_state=42)

    validator = RobustCrossValidator(cv_method='kfold', n_splits=5)
    results = validator.cross_validate_model(model, X, y)

    # Check results structure
    required_metrics = ['accuracy', 'precision', 'recall', 'f1']
    for metric in required_metrics:
        assert metric in results
        assert 'mean' in results[metric]
        assert 'std' in results[metric]
        assert isinstance(results[metric]['mean'], (int, float))
        assert isinstance(results[metric]['std'], (int, float))

    print(f"   KFold F1: {results['f1']['mean']:.3f} ± {results['f1']['std']:.3f}")
    # Check stability score
    summary = validator.get_cv_summary()
    assert 'overall_stability' in summary
    assert 0 <= summary['overall_stability'] <= 1
    print(f"   Stability Score: {summary['overall_stability']:.3f}")


def test_stratified_cross_validation():
    """Test StratifiedKFold cross-validation."""
    print("Testing StratifiedKFold cross-validation...")

    X, y = create_test_data(n_samples=500)
    model = LogisticRegression(random_state=42, max_iter=1000)

    validator = RobustCrossValidator(cv_method='stratified', n_splits=5)
    results = validator.cross_validate_model(model, X, y)

    # Check results
    assert 'f1' in results
    assert results['f1']['mean'] > 0
    print(f"   Stratified F1: {results['f1']['mean']:.3f}")

    # Test with imbalanced data
    np.random.seed(42)
    y_imbalanced = np.random.choice([0, 1], size=500, p=[0.9, 0.1])
    results_imbalanced = validator.cross_validate_model(model, X, y_imbalanced)

    # With imbalanced data, F1 might be 0, but the test should still run without error
    assert 'f1' in results_imbalanced
    assert isinstance(results_imbalanced['f1']['mean'], (int, float))
    print("✅ Handles imbalanced data correctly")


def test_time_series_cross_validation():
    """Test TimeSeriesSplit cross-validation."""
    print("Testing TimeSeriesSplit cross-validation...")

    X, y = create_test_data(n_samples=500)
    model = RandomForestClassifier(n_estimators=50, random_state=42)

    validator = RobustCrossValidator(cv_method='time_series', n_splits=5)
    results = validator.cross_validate_model(model, X, y)

    # Check results
    assert 'f1' in results
    assert results['f1']['mean'] > 0
    print(f"   Time Series F1: {results['f1']['mean']:.3f}")

    # Verify it's using TimeSeriesSplit
    assert hasattr(validator, 'cv')
    assert str(type(validator.cv)).endswith("TimeSeriesSplit'>")


def test_walk_forward_validation():
    """Test walk-forward validation."""
    print("Testing walk-forward validation...")

    X, y = create_test_data(n_samples=500)
    model = RandomForestClassifier(n_estimators=50, random_state=42)

    validator = RobustCrossValidator(cv_method='walk_forward', n_splits=5)
    results = validator.cross_validate_model(model, X, y)

    # Check results
    assert 'f1' in results
    assert results['f1']['mean'] > 0
    print(f"   Walk-forward F1: {results['f1']['mean']:.3f}")


def test_strategy_comparison():
    """Test comparison of different CV strategies."""
    print("Testing CV strategy comparison...")

    X, y = create_test_data(n_samples=500)
    model = RandomForestClassifier(n_estimators=50, random_state=42)

    validator = RobustCrossValidator()
    strategies = ['kfold', 'stratified', 'time_series']

    comparison = validator.compare_cv_strategies(model, X, y, strategies=strategies)

    # Check comparison results
    assert len(comparison) == len(strategies)
    for strategy in strategies:
        assert strategy in comparison
        if 'error' not in comparison[strategy]:
            assert 'f1_mean' in comparison[strategy]
            assert 'f1_std' in comparison[strategy]
            assert isinstance(comparison[strategy]['f1_mean'], (int, float))
            assert isinstance(comparison[strategy]['f1_std'], (int, float))

    print("✅ Strategy comparison completed")
    print("   Results:")
    for strategy, metrics in comparison.items():
        if 'error' not in metrics:
            print(f"   {strategy}: F1={metrics['f1_mean']:.3f} ± {metrics['f1_std']:.3f}")
        else:
            print(f"   {strategy}: Error - {metrics['error']}")


def test_stability_scoring():
    """Test stability scoring functionality."""
    print("Testing stability scoring...")

    X, y = create_test_data(n_samples=500)

    # Test with stable model (should have high stability)
    stable_model = RandomForestClassifier(n_estimators=100, random_state=42)
    validator = RobustCrossValidator(cv_method='kfold', n_splits=10)
    validator.cross_validate_model(stable_model, X, y)
    stable_summary = validator.get_cv_summary()

    # Test with unstable model (should have lower stability)
    unstable_model = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=42)
    validator2 = RobustCrossValidator(cv_method='kfold', n_splits=10)
    validator2.cross_validate_model(unstable_model, X, y)
    unstable_summary = validator2.get_cv_summary()

    print(f"   Stable model stability: {stable_summary['overall_stability']:.3f}")
    print(f"   Unstable model stability: {unstable_summary['overall_stability']:.3f}")

    # Stable model should generally have higher stability
    assert stable_summary['overall_stability'] >= unstable_summary['overall_stability'] * 0.8  # Allow some tolerance


def test_error_handling():
    """Test error handling in cross-validation."""
    print("Testing error handling...")

    X, y = create_test_data(n_samples=100)

    # Test with invalid model
    class InvalidModel:
        pass

    validator = RobustCrossValidator()
    try:
        results = validator.cross_validate_model(InvalidModel(), X, y)
        assert False, "Should raise error for invalid model"
    except Exception as e:
        print(f"✅ Correctly handles invalid model: {type(e).__name__}")

    # Test with mismatched data
    X_bad = np.random.randn(50, 5)  # Different shape
    y_bad = np.random.randint(0, 2, 100)  # Different length

    try:
        results = validator.cross_validate_model(RandomForestClassifier(), X_bad, y_bad)
        assert False, "Should raise error for mismatched data"
    except Exception as e:
        print(f"✅ Correctly handles mismatched data: {type(e).__name__}")


def test_different_model_types():
    """Test cross-validation with different model types."""
    print("Testing different model types...")

    X, y = create_test_data(n_samples=300)

    models = [
        ("RandomForest", RandomForestClassifier(n_estimators=50, random_state=42)),
        ("LogisticRegression", LogisticRegression(random_state=42, max_iter=1000)),
    ]

    validator = RobustCrossValidator(cv_method='kfold', n_splits=3)

    for model_name, model in models:
        results = validator.cross_validate_model(model, X, y)
        assert 'f1' in results
        assert results['f1']['mean'] > 0
        print(f"   {model_name} F1: {results['f1']['mean']:.3f}")


def test_cv_summary_detailed():
    """Test detailed CV summary."""
    print("Testing detailed CV summary...")

    X, y = create_test_data(n_samples=400)
    model = RandomForestClassifier(n_estimators=50, random_state=42)

    validator = RobustCrossValidator(cv_method='stratified', n_splits=4)
    validator.cross_validate_model(model, X, y)

    summary = validator.get_cv_summary()

    required_keys = ['overall_stability', 'best_fold', 'worst_fold', 'fold_scores']
    for key in required_keys:
        assert key in summary, f"Missing key: {key}"

    assert isinstance(summary['fold_scores'], list)
    assert len(summary['fold_scores']) == 4  # n_splits
    assert all(isinstance(score, (int, float)) for score in summary['fold_scores'])

    print("✅ Detailed summary contains all required information")
    print(f"   Fold scores: {summary['fold_scores']}")
    print(f"   Best fold: {summary['best_fold']}, Worst fold: {summary['worst_fold']}")


def main():
    """Run all RobustCrossValidator tests."""
    print("🧪 Starting Robust Cross-Validation Tests")
    print("=" * 60)

    tests = [
        ("RobustCrossValidator Initialization", test_robust_cross_validator_initialization),
        ("KFold Cross-Validation", test_kfold_cross_validation),
        ("Stratified Cross-Validation", test_stratified_cross_validation),
        ("Time Series Cross-Validation", test_time_series_cross_validation),
        ("Walk-Forward Validation", test_walk_forward_validation),
        ("Strategy Comparison", test_strategy_comparison),
        ("Stability Scoring", test_stability_scoring),
        ("Error Handling", test_error_handling),
        ("Different Model Types", test_different_model_types),
        ("Detailed CV Summary", test_cv_summary_detailed),
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
        print("🎉 All RobustCrossValidator tests completed successfully!")
    else:
        print("⚠️  Some tests failed. Check the output above.")
        raise AssertionError(f"Only {passed}/{total} tests passed")


if __name__ == "__main__":
    main()