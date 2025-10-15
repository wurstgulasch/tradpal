"""
ML Trainer Service

Advanced ML training service with support for:
- Traditional sklearn models
- PyTorch neural networks (LSTM, GRU, Transformer)
- AutoML with Optuna
- Ensemble model training
"""

import sys
import os
from pathlib import Path
import argparse
import json
from datetime import datetime, timedelta

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_fetcher import fetch_historical_data
from src.indicators import calculate_indicators
from config.settings import (
    SYMBOL, TIMEFRAME, LOOKBACK_DAYS, ML_MODEL_DIR,
    ML_USE_PYTORCH, ML_PYTORCH_MODEL_TYPE, ML_USE_AUTOML
)


def train_sklearn_model(symbol: str, timeframe: str, start_date: str, end_date: str,
                       model_type: str = 'random_forest', use_automl: bool = False):
    """
    Train sklearn model with optional AutoML optimization.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        start_date: Start date for training data
        end_date: End date for training data
        model_type: Type of model ('random_forest', 'gradient_boosting')
        use_automl: Whether to use AutoML optimization
    """
    print(f"📊 Training sklearn {model_type} model for {symbol} on {timeframe}")
    
    # Fetch and prepare data
    print(f"📈 Fetching data from {start_date} to {end_date}...")
    df = fetch_historical_data(
        symbol=symbol,
        timeframe=timeframe,
        since=start_date,
        limit=10000
    )
    
    if df is None or len(df) == 0:
        print("❌ Failed to fetch data")
        return None
    
    print(f"✅ Fetched {len(df)} candles")
    
    # Calculate indicators
    print("🔧 Calculating indicators...")
    df = calculate_indicators(df)
    
    # Train with AutoML if enabled
    if use_automl:
        from src.ml_automl import run_automl_optimization, is_automl_available
        
        if not is_automl_available():
            print("❌ Optuna not available for AutoML")
            return None
        
        print("🤖 Starting AutoML optimization...")
        results = run_automl_optimization(
            df=df,
            model_type=model_type,
            symbol=symbol,
            timeframe=timeframe,
            n_trials=50  # Can be configured
        )
        
        print(f"✅ AutoML complete! Best score: {results.get('best_score', 'N/A')}")
        return results
    
    # Standard training
    from src.ml_predictor import get_ml_predictor, is_ml_available
    
    if not is_ml_available():
        print("❌ scikit-learn not available")
        return None
    
    print("🚀 Training model...")
    predictor = get_ml_predictor(symbol=symbol, timeframe=timeframe)
    results = predictor.train_models(df)
    
    print(f"✅ Training complete!")
    return results


def train_pytorch_model(symbol: str, timeframe: str, start_date: str, end_date: str,
                       model_type: str = 'lstm'):
    """
    Train PyTorch neural network model.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        start_date: Start date for training data
        end_date: End date for training data
        model_type: Type of model ('lstm', 'gru', 'transformer')
    """
    print(f"🧠 Training PyTorch {model_type.upper()} model for {symbol} on {timeframe}")
    
    from src.ml_pytorch_models import get_pytorch_predictor, is_pytorch_available
    
    if not is_pytorch_available():
        print("❌ PyTorch not available")
        return None
    
    # Fetch and prepare data
    print(f"📈 Fetching data from {start_date} to {end_date}...")
    df = fetch_historical_data(
        symbol=symbol,
        timeframe=timeframe,
        since=start_date,
        limit=10000
    )
    
    if df is None or len(df) == 0:
        print("❌ Failed to fetch data")
        return None
    
    print(f"✅ Fetched {len(df)} candles")
    
    # Calculate indicators
    print("🔧 Calculating indicators...")
    df = calculate_indicators(df)
    
    # Train model
    print(f"🚀 Training {model_type.upper()} model...")
    predictor = get_pytorch_predictor(
        model_type=model_type,
        symbol=symbol,
        timeframe=timeframe
    )
    
    if predictor is None:
        print("❌ Failed to initialize predictor")
        return None
    
    results = predictor.train_model(df)
    
    print(f"✅ Training complete! Test accuracy: {results.get('final_test_accuracy', 'N/A'):.4f}")
    return results


def train_ensemble_model(symbol: str, timeframe: str, start_date: str, end_date: str):
    """
    Train ensemble model combining multiple approaches.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        start_date: Start date for training data
        end_date: End date for training data
    """
    print(f"🎭 Training ensemble model for {symbol} on {timeframe}")
    
    # Train both sklearn and PyTorch models
    print("\n=== Training sklearn model ===")
    sklearn_results = train_sklearn_model(symbol, timeframe, start_date, end_date)
    
    print("\n=== Training PyTorch model ===")
    pytorch_results = train_pytorch_model(symbol, timeframe, start_date, end_date)
    
    # Initialize ensemble predictor
    from src.ml_ensemble import get_ensemble_predictor
    
    ensemble = get_ensemble_predictor(symbol=symbol, timeframe=timeframe)
    
    results = {
        'sklearn': sklearn_results,
        'pytorch': pytorch_results,
        'ensemble_weights': ensemble.weights,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save results
    results_dir = Path("output/ml_training")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"ensemble_training_{symbol}_{timeframe}.json"
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Ensemble training complete! Results saved to {results_path}")
    return results


def main():
    """Main training service entry point."""
    parser = argparse.ArgumentParser(description="ML Trainer Service")
    parser.add_argument('--symbol', type=str, default=SYMBOL, help='Trading symbol')
    parser.add_argument('--timeframe', type=str, default=TIMEFRAME, help='Timeframe')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--lookback-days', type=int, default=LOOKBACK_DAYS,
                       help='Days of historical data')
    parser.add_argument('--mode', type=str, default='sklearn',
                       choices=['sklearn', 'pytorch', 'automl', 'ensemble'],
                       help='Training mode')
    parser.add_argument('--model-type', type=str, default='random_forest',
                       help='Model type (sklearn: random_forest, gradient_boosting; pytorch: lstm, gru, transformer)')
    
    args = parser.parse_args()
    
    # Calculate dates if not provided
    if not args.end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    else:
        end_date = args.end_date
    
    if not args.start_date:
        start_date = (datetime.now() - timedelta(days=args.lookback_days)).strftime('%Y-%m-%d')
    else:
        start_date = args.start_date
    
    print("=" * 70)
    print("ML TRAINER SERVICE")
    print("=" * 70)
    print(f"Symbol: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Training Period: {start_date} to {end_date}")
    print(f"Mode: {args.mode}")
    print(f"Model Type: {args.model_type}")
    print("=" * 70)
    print()
    
    # Run training based on mode
    try:
        if args.mode == 'sklearn':
            results = train_sklearn_model(
                symbol=args.symbol,
                timeframe=args.timeframe,
                start_date=start_date,
                end_date=end_date,
                model_type=args.model_type,
                use_automl=False
            )
        elif args.mode == 'automl':
            results = train_sklearn_model(
                symbol=args.symbol,
                timeframe=args.timeframe,
                start_date=start_date,
                end_date=end_date,
                model_type=args.model_type,
                use_automl=True
            )
        elif args.mode == 'pytorch':
            results = train_pytorch_model(
                symbol=args.symbol,
                timeframe=args.timeframe,
                start_date=start_date,
                end_date=end_date,
                model_type=args.model_type
            )
        elif args.mode == 'ensemble':
            results = train_ensemble_model(
                symbol=args.symbol,
                timeframe=args.timeframe,
                start_date=start_date,
                end_date=end_date
            )
        else:
            print(f"❌ Unknown mode: {args.mode}")
            return
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        
        if results:
            print("\n📊 Results Summary:")
            print(json.dumps(results, indent=2, default=str))
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
