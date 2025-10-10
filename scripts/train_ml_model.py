#!/usr/bin/env python3
"""
ML Model Training Script for Trading Signals

Trains machine learning models for signal prediction enhancement.
Can be run manually or as part of automated retraining.
"""

import sys
import os
import argparse
from datetime import datetime, timedelta
import numpy as np
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.ml_predictor import get_ml_predictor, is_ml_available
from src.data_fetcher import fetch_historical_data
from src.indicators import calculate_indicators
from config.settings import (
    SYMBOL, TIMEFRAME, LOOKBACK_DAYS, ML_ENABLED,
    ML_MIN_TRAINING_SAMPLES, ML_TRAINING_HORIZON
)


def train_ml_model(symbol: str = SYMBOL, timeframe: str = TIMEFRAME,
                  lookback_days: int = LOOKBACK_DAYS, force_retrain: bool = False):
    """
    Train ML model for signal prediction.

    Args:
        symbol: Trading symbol
        timeframe: Timeframe for training
        lookback_days: Historical data period
        force_retrain: Force retraining even if model exists
    """
    print(f"🤖 Training ML model for {symbol} on {timeframe} timeframe")
    print(f"   Historical data: {lookback_days} days")
    print(f"   Force retrain: {force_retrain}")
    print("-" * 60)

    # Check ML availability
    if not ML_ENABLED:
        print("❌ ML features are disabled in configuration")
        return False

    if not is_ml_available():
        print("❌ scikit-learn is not available. Install with: pip install scikit-learn")
        return False

    try:
        # Get ML predictor
        ml_predictor = get_ml_predictor(symbol=symbol, timeframe=timeframe)
        if ml_predictor is None:
            print("❌ Failed to initialize ML predictor")
            return False

        # Fetch historical data first (needed for feature count check)
        print(f"📊 Fetching historical data ({lookback_days} days)...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        df = fetch_historical_data(
            symbol=symbol,
            exchange_name='kraken',
            timeframe=timeframe,
            start_date=start_date,
            limit=5000  # Fetch up to 5000 candles for training
        )

        if df.empty:
            print("❌ No historical data available")
            return False

        print(f"✅ Loaded {len(df)} data points")

        # Check if model already exists and retraining not forced
        if ml_predictor.is_trained and not force_retrain:
            # Check if feature count matches current feature engineering
            model_info = ml_predictor.get_model_info()
            current_feature_count = len(ml_predictor.prepare_features(df.head(10))[1])  # Get feature count from sample
            
            if model_info['feature_count'] == current_feature_count:
                print("ℹ️  ML model already trained with matching features. Use --force to retrain.")
                print(f"   Status: {model_info['status']}")
                print(f"   Features: {model_info['feature_count']}")
                return True
            else:
                print(f"⚠️  ML model has {model_info['feature_count']} features, but current feature engineering generates {current_feature_count} features.")
                print("   Retraining model to match new feature set...")
                force_retrain = True  # Force retraining due to feature mismatch

        # Check minimum sample requirement
        min_samples = min(ML_MIN_TRAINING_SAMPLES, len(df) // 2)  # Allow training with at least half the available data
        if len(df) < min_samples:
            print(f"⚠️  Insufficient data: {len(df)} < {min_samples} required")
            print("   Using available data for demonstration purposes...")
            # Continue anyway for demonstration

        # Calculate indicators
        print("📈 Calculating technical indicators...")
        df = calculate_indicators(df)

        # Remove rows with NaN values
        df_clean = df.dropna()
        if len(df_clean) < min_samples * 0.5:  # Allow 50% data loss
            print(f"⚠️  Too much data lost during cleaning: {len(df_clean)} remaining")
            print("   Using available cleaned data for demonstration purposes...")
            # Continue with available data

        print(f"✅ Clean data: {len(df_clean)} samples")

        # Train the model
        print("🎯 Training ML models...")
        training_result = ml_predictor.train_models(
            df_clean,
            prediction_horizon=ML_TRAINING_HORIZON
        )

        if training_result['success']:
            print("✅ ML model training completed!")
            print(f"   Best model: {training_result['best_model']}")
            print(f"   Training samples: {training_result['samples']}")
            print(f"   Features used: {training_result['features']}")

            # Show performance summary
            performance = training_result['performance']
            for model_name, metrics in performance.items():
                if 'f1_score' in metrics:
                    cv_info = ""
                    if 'cv_f1_mean' in metrics:
                        cv_info = f", CV-F1={metrics['cv_f1_mean']:.3f}±{metrics['cv_f1_std']:.3f}"
                    print(f"   {model_name}: F1={metrics['f1_score']:.3f}{cv_info}")

            return True
        else:
            print(f"❌ ML training failed: {training_result.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        return False


def optimize_gradient_boosting_model(symbol: str = SYMBOL, timeframe: str = TIMEFRAME,
                                     lookback_days: int = LOOKBACK_DAYS):
    """
    Optimize Gradient Boosting model hyperparameters using Optuna.

    Args:
        symbol: Trading symbol
        timeframe: Timeframe for training
        lookback_days: Historical data period
    """
    print(f"🎯 Optimizing Gradient Boosting model for {symbol} on {timeframe} timeframe")
    print(f"   Historical data: {lookback_days} days")
    print("-" * 70)

    # Check ML availability
    if not ML_ENABLED:
        print("❌ ML features are disabled in configuration")
        return False

    if not is_ml_available():
        print("❌ scikit-learn is not available. Install with: pip install scikit-learn")
        return False

    try:
        # Get ML predictor
        ml_predictor = get_ml_predictor(symbol=symbol, timeframe=timeframe)
        if ml_predictor is None:
            print("❌ Failed to initialize ML predictor")
            return False

        # Fetch historical data
        print(f"📊 Fetching historical data ({lookback_days} days)...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        df = fetch_historical_data(
            symbol=symbol,
            exchange_name='kraken',
            timeframe=timeframe,
            start_date=start_date,
            limit=5000
        )

        if df.empty:
            print("❌ No historical data available")
            return False

        print(f"✅ Loaded {len(df)} data points")

        # Calculate indicators
        print("📈 Calculating technical indicators...")
        df = calculate_indicators(df)

        # Remove rows with NaN values
        df_clean = df.dropna()
        if len(df_clean) < 500:
            print(f"⚠️  Insufficient clean data: {len(df_clean)} samples")
            return False

        print(f"✅ Clean data: {len(df_clean)} samples")

        # Prepare features and labels
        X, feature_names = ml_predictor.prepare_features(df_clean)
        y = ml_predictor.create_labels(df_clean, prediction_horizon=ML_TRAINING_HORIZON)

        # Remove NaN
        valid_indices = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_indices]
        y = y[valid_indices]

        if len(X) < 300:
            print(f"⚠️  Insufficient training data: {len(X)} samples")
            return False

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"📊 Training data: {len(X_train)} train, {len(X_test)} test")

        # Run optimization
        optimization_result = ml_predictor.optimize_gradient_boosting(X_train, X_test, y_train, y_test)

        if optimization_result:
            print("✅ Gradient Boosting optimization completed!")
            print(f"   Best F1 Score: {optimization_result['best_score']:.4f}")
            print(f"   Best Parameters: {optimization_result['best_params']}")

            # Retrain with optimized parameters
            print("🔄 Retraining model with optimized parameters...")
            training_result = ml_predictor.train_models(df_clean)

            if training_result['success']:
                print("✅ Model retrained with optimized parameters!")
                return True
            else:
                print("❌ Failed to retrain model")
                return False
        else:
            print("❌ Optimization failed")
            return False

    except Exception as e:
        print(f"❌ Optimization failed with error: {e}")
        return False


def evaluate_ml_model(symbol: str = SYMBOL, timeframe: str = TIMEFRAME):
    """Evaluate the current ML model performance."""
    print(f"📊 Evaluating ML model for {symbol} on {timeframe}")

    if not is_ml_available():
        print("❌ scikit-learn not available")
        return

    ml_predictor = get_ml_predictor(symbol=symbol, timeframe=timeframe)
    if ml_predictor is None or not ml_predictor.is_trained:
        print("❌ No trained model available")
        return

    model_info = ml_predictor.get_model_info()
    print(f"Status: {model_info['status']}")
    print(f"Features: {model_info['feature_count']}")
    print(f"Feature names: {', '.join(model_info['features'][:5])}...")

    if 'performance' in model_info:
        print("\nModel Performance:")
        for model_name, metrics in model_info['performance'].items():
            if isinstance(metrics, dict) and 'f1_score' in metrics:
                print(f"  {model_name}:")
                print(".3f")
                print(".3f")
                print(".3f")
                print(".3f")


def main():
    """Main function for ML training script."""
    parser = argparse.ArgumentParser(description='ML Model Training for Trading Signals')
    parser.add_argument('--symbol', default=SYMBOL, help=f'Trading symbol (default: {SYMBOL})')
    parser.add_argument('--timeframe', default=TIMEFRAME, help=f'Timeframe (default: {TIMEFRAME})')
    parser.add_argument('--days', type=int, default=LOOKBACK_DAYS, help=f'Historical days (default: {LOOKBACK_DAYS})')
    parser.add_argument('--force', action='store_true', help='Force retraining even if model exists')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate existing model instead of training')
    parser.add_argument('--optimize', action='store_true', help='Optimize Gradient Boosting hyperparameters with Optuna')

    args = parser.parse_args()

    if args.evaluate:
        evaluate_ml_model(args.symbol, args.timeframe)
    elif args.optimize:
        success = optimize_gradient_boosting_model(args.symbol, args.timeframe, args.days)
        if success:
            print("\n🎉 Gradient Boosting optimization completed successfully!")
            print("The optimized model will now enhance trading signals with improved predictions.")
        else:
            print("\n❌ Gradient Boosting optimization failed.")
            sys.exit(1)
    else:
        success = train_ml_model(args.symbol, args.timeframe, args.days, args.force)
        if success:
            print("\n🎉 ML model training completed successfully!")
            print("The model will now enhance trading signals with ML predictions.")
        else:
            print("\n❌ ML model training failed.")
            sys.exit(1)


if __name__ == "__main__":
    main()