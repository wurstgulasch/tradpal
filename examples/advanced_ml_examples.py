"""
Example Script: Using Advanced ML and Optimization Features

This script demonstrates:
1. Walk-forward optimization with enhanced metrics
2. Ensemble predictions combining GA and ML
3. PyTorch model training (if available)
4. AutoML hyperparameter optimization (if available)
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def example_walk_forward_optimization():
    """
    Example: Walk-forward optimization with enhanced metrics.
    
    Demonstrates:
    - Information Coefficient tracking
    - Bias-Variance tradeoff analysis
    - Overfitting detection
    """
    print("=" * 70)
    print("EXAMPLE 1: Walk-Forward Optimization with Enhanced Metrics")
    print("=" * 70)
    
    from src.walk_forward_optimizer import WalkForwardOptimizer
    import numpy as np
    
    optimizer = WalkForwardOptimizer(symbol="BTC/USDT", timeframe="1h")
    
    # Simulate walk-forward analysis results
    print("\n📊 Simulating walk-forward analysis...")
    
    # Mock data for demonstration
    in_sample_scores = [0.75, 0.80, 0.78, 0.82, 0.85]
    out_of_sample_scores = [0.65, 0.68, 0.70, 0.72, 0.69]
    
    # Calculate enhanced metrics
    bias_variance = optimizer._calculate_bias_variance(in_sample_scores, out_of_sample_scores)
    
    print("\n📈 Enhanced Metrics:")
    print(f"  Information Coefficient: {np.corrcoef(in_sample_scores, out_of_sample_scores)[0,1]:.4f}")
    print(f"  Bias: {bias_variance['bias']:.4f}")
    print(f"  Variance: {bias_variance['variance']:.4f}")
    print(f"  Overfitting Ratio: {(np.mean(in_sample_scores) - np.mean(out_of_sample_scores)) / np.mean(out_of_sample_scores):.4f}")
    print(f"\n💡 {bias_variance['interpretation']}")
    
    print("\n✅ Walk-forward optimization complete!")


def example_ensemble_predictions():
    """
    Example: Ensemble predictions combining GA and ML.
    
    Demonstrates:
    - Weighted voting
    - Adaptive weight adjustment
    - Performance tracking
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Ensemble Predictions (GA + ML)")
    print("=" * 70)
    
    from src.ml_ensemble import get_ensemble_predictor
    
    ensemble = get_ensemble_predictor(symbol="BTC/USDT", timeframe="1h")
    
    print("\n🎭 Initial Configuration:")
    print(f"  Voting Strategy: weighted")
    print(f"  ML Weight: {ensemble.weights['ml']:.2f}")
    print(f"  GA Weight: {ensemble.weights['ga']:.2f}")
    
    # Simulate several predictions
    scenarios = [
        {
            'ml': {'signal': 'BUY', 'confidence': 0.85},
            'ga': {'signal': 'BUY', 'confidence': 0.75},
            'actual': 1.0,
            'description': 'Both agree on BUY'
        },
        {
            'ml': {'signal': 'SELL', 'confidence': 0.70},
            'ga': {'signal': 'NEUTRAL', 'confidence': 0.50},
            'actual': -1.0,
            'description': 'ML suggests SELL, GA uncertain'
        },
        {
            'ml': {'signal': 'BUY', 'confidence': 0.90},
            'ga': {'signal': 'SELL', 'confidence': 0.80},
            'actual': 1.0,
            'description': 'Disagreement - ML correct'
        }
    ]
    
    print("\n📊 Running Ensemble Predictions:")
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n  Scenario {i}: {scenario['description']}")
        
        result = ensemble.predict(scenario['ml'], scenario['ga'])
        print(f"    Ensemble Decision: {result['signal']} (confidence: {result['confidence']:.2f})")
        
        # Update performance
        ensemble.update_performance(
            scenario['actual'],
            scenario['ml'],
            scenario['ga'],
            result
        )
    
    # Show updated weights
    stats = ensemble.get_performance_stats()
    print(f"\n📈 Updated Weights After Learning:")
    print(f"  ML Weight: {ensemble.weights['ml']:.2f}")
    print(f"  GA Weight: {ensemble.weights['ga']:.2f}")
    
    print(f"\n📊 Performance Summary:")
    for component in ['ml', 'ga', 'ensemble']:
        if component in stats:
            print(f"  {component.upper()}: {stats[component]['accuracy']:.2%} accuracy")
    
    print("\n✅ Ensemble predictions complete!")


def example_pytorch_training():
    """
    Example: PyTorch model training (if available).
    
    Demonstrates:
    - Model creation
    - Training configuration
    - Feature availability check
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: PyTorch Neural Network Training")
    print("=" * 70)
    
    from src.ml_pytorch_models import is_pytorch_available, get_pytorch_predictor
    
    if not is_pytorch_available():
        print("\n⚠️  PyTorch not available")
        print("   Install with: pip install torch")
        print("   This example would train LSTM, GRU, or Transformer models")
        print("   Features: GPU acceleration, attention mechanisms, early stopping")
        return
    
    print("\n🧠 PyTorch is available!")
    print("   Supported models: LSTM, GRU, Transformer")
    print("   Features: GPU acceleration, attention, residual connections")
    
    # Would train model here if we had real data
    print("\n💡 To train a model, use:")
    print("   cd services/ml-trainer")
    print("   python train_service.py --mode pytorch --model-type lstm")
    
    print("\n✅ PyTorch example complete!")


def example_automl_optimization():
    """
    Example: AutoML with Optuna (if available).
    
    Demonstrates:
    - Hyperparameter optimization
    - Study creation
    - Feature availability check
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: AutoML Hyperparameter Optimization")
    print("=" * 70)
    
    from src.ml_automl import is_automl_available, get_automl_optimizer
    
    if not is_automl_available():
        print("\n⚠️  Optuna not available")
        print("   Install with: pip install optuna")
        print("   This example would automatically optimize:")
        print("   - Model hyperparameters")
        print("   - Learning rates")
        print("   - Architecture parameters")
        print("   - Regularization settings")
        return
    
    print("\n🤖 Optuna is available!")
    print("   Supported samplers: TPE, Random, Grid")
    print("   Supported pruners: Median, Hyperband")
    print("   Features: Multi-objective, visualization, persistence")
    
    print("\n💡 To run AutoML optimization, use:")
    print("   cd services/ml-trainer")
    print("   python train_service.py --mode automl --model-type random_forest")
    
    print("\n✅ AutoML example complete!")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("ADVANCED ML AND OPTIMIZATION EXAMPLES")
    print("=" * 70)
    print("\nThis script demonstrates the new features:")
    print("  1. Enhanced walk-forward optimization metrics")
    print("  2. Ensemble predictions (GA + ML)")
    print("  3. PyTorch neural networks")
    print("  4. AutoML hyperparameter optimization")
    
    try:
        example_walk_forward_optimization()
        example_ensemble_predictions()
        example_pytorch_training()
        example_automl_optimization()
        
        print("\n" + "=" * 70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 70)
        
        print("\n💡 Next Steps:")
        print("  1. Install optional dependencies:")
        print("     pip install torch optuna")
        print("  2. Try the training services:")
        print("     python services/ml-trainer/train_service.py --help")
        print("  3. Run walk-forward optimization:")
        print("     python services/optimizer/optimize_service.py --help")
        print("  4. Enable features in config/settings.py:")
        print("     ML_USE_PYTORCH = True")
        print("     ML_USE_AUTOML = True")
        print("     ML_USE_ENSEMBLE = True")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
