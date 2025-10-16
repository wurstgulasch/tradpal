"""
Test script for advanced ML and optimization features

Tests:
- Walk-forward optimizer with enhanced metrics
- Ensemble predictor
- Module availability checks
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_walk_forward_enhancements():
    """Test walk-forward optimizer enhancements."""
    print("=" * 70)
    print("Testing Walk-Forward Optimizer Enhancements")
    print("=" * 70)
    
    from services.walk_forward_optimizer import WalkForwardOptimizer
    
    optimizer = WalkForwardOptimizer(symbol="BTC/USD", timeframe="1h")
    
    # Test bias-variance calculation
    in_sample = [0.8, 0.9, 0.85, 0.88, 0.92]
    out_of_sample = [0.7, 0.75, 0.72, 0.68, 0.78]
    
    bias_variance = optimizer._calculate_bias_variance(in_sample, out_of_sample)
    
    print(f"\n📊 Bias-Variance Analysis:")
    print(f"  Bias: {bias_variance['bias']:.4f}")
    print(f"  Variance: {bias_variance['variance']:.4f}")
    print(f"  Total Error: {bias_variance['total_error']:.4f}")
    print(f"  Bias/Variance Ratio: {bias_variance['bias_variance_ratio']:.4f}")
    print(f"  Interpretation: {bias_variance['interpretation']}")
    
    print("\n✅ Walk-forward optimizer enhancements working!")
    return True


def test_ensemble_predictor():
    """Test ensemble predictor."""
    print("\n" + "=" * 70)
    print("Testing Ensemble Predictor")
    print("=" * 70)
    
    from services.ml_ensemble import get_ensemble_predictor
    
    ensemble = get_ensemble_predictor(symbol="BTC/USD", timeframe="1h")
    
    # Test predictions
    ml_prediction = {'signal': 'BUY', 'confidence': 0.8}
    ga_prediction = {'signal': 'BUY', 'confidence': 0.9}
    
    result = ensemble.predict(ml_prediction, ga_prediction)
    
    print(f"\n🎭 Ensemble Prediction:")
    print(f"  ML Signal: {ml_prediction['signal']} ({ml_prediction['confidence']:.2f})")
    print(f"  GA Signal: {ga_prediction['signal']} ({ga_prediction['confidence']:.2f})")
    print(f"  Ensemble Signal: {result['signal']} ({result['confidence']:.2f})")
    print(f"  Method: {result['method']}")
    
    # Test performance tracking
    ensemble.update_performance(1.0, ml_prediction, ga_prediction, result)
    
    stats = ensemble.get_performance_stats()
    print(f"\n📈 Performance Stats:")
    for component, stat in stats.items():
        if component != 'current_weights':
            print(f"  {component.upper()}: {stat}")
    
    print("\n✅ Ensemble predictor working!")
    return True


def test_module_availability():
    """Test module availability checks."""
    print("\n" + "=" * 70)
    print("Testing Module Availability")
    print("=" * 70)
    
    from services.ml_pytorch_models import is_pytorch_available
    from services.ml_automl import is_automl_available
    from services.ml_predictor import is_ml_available
    
    print(f"\n📦 Module Availability:")
    print(f"  scikit-learn: {is_ml_available()}")
    print(f"  PyTorch: {is_pytorch_available()}")
    print(f"  Optuna (AutoML): {is_automl_available()}")
    
    if not is_ml_available():
        print("\n⚠️  scikit-learn not available. Install with: pip install scikit-learn")
    if not is_pytorch_available():
        print("⚠️  PyTorch not available. Install with: pip install torch")
    if not is_automl_available():
        print("⚠️  Optuna not available. Install with: pip install optuna")
    
    print("\n✅ Module availability checks working!")
    return True


def test_config_settings():
    """Test configuration settings."""
    print("\n" + "=" * 70)
    print("Testing Configuration Settings")
    print("=" * 70)
    
    from config.settings import (
        ML_USE_PYTORCH, ML_PYTORCH_MODEL_TYPE, ML_USE_AUTOML,
        ML_USE_ENSEMBLE, ML_ENSEMBLE_VOTING, ML_ENSEMBLE_WEIGHTS
    )
    
    print(f"\n⚙️  Configuration:")
    print(f"  PyTorch Enabled: {ML_USE_PYTORCH}")
    print(f"  PyTorch Model Type: {ML_PYTORCH_MODEL_TYPE}")
    print(f"  AutoML Enabled: {ML_USE_AUTOML}")
    print(f"  Ensemble Enabled: {ML_USE_ENSEMBLE}")
    print(f"  Ensemble Voting: {ML_ENSEMBLE_VOTING}")
    print(f"  Ensemble Weights: {ML_ENSEMBLE_WEIGHTS}")
    
    print("\n✅ Configuration settings loaded!")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("TESTING ADVANCED ML AND OPTIMIZATION FEATURES")
    print("=" * 70)
    
    tests = [
        ('Module Availability', test_module_availability),
        ('Configuration Settings', test_config_settings),
        ('Walk-Forward Enhancements', test_walk_forward_enhancements),
        ('Ensemble Predictor', test_ensemble_predictor),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success, None))
        except Exception as e:
            print(f"\n❌ {test_name} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False, str(e)))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, error in results:
        status = "✅ PASS" if success else f"❌ FAIL"
        print(f"{status}: {test_name}")
        if error:
            print(f"     Error: {error}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
