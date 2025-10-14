#!/usr/bin/env python3
"""
Advanced ML Training Script

Trains and evaluates advanced ML models including ensemble methods,
market regime detection, and feature engineering.
"""

import sys
import os
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from config.settings import (
    SYMBOL, TIMEFRAME, ML_ADVANCED_FEATURES_ENABLED,
    ML_ENSEMBLE_MODELS, ML_MARKET_REGIME_DETECTION
)

# Import data fetching and processing
from src.data_fetcher import fetch_historical_data
from src.indicators import calculate_indicators

# Import advanced ML predictor
from src.advanced_ml_predictor import (
    get_advanced_ml_predictor, is_advanced_ml_available,
    MarketRegimeDetector, AdvancedFeatureEngineer
)

# Import existing ML for comparison
from src.ml_predictor import get_ml_predictor

# Import audit logging
from src.audit_logger import audit_logger


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Advanced ML Models')

    parser.add_argument('--symbol', type=str, default=SYMBOL,
                       help=f'Trading symbol (default: {SYMBOL})')
    parser.add_argument('--timeframe', type=str, default=TIMEFRAME,
                       help=f'Timeframe (default: {TIMEFRAME})')
    parser.add_argument('--start-date', type=str,
                       help='Start date for training data (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str,
                       help='End date for training data (YYYY-MM-DD)')
    parser.add_argument('--days', type=int, default=365,
                       help='Number of days of historical data (default: 365)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    parser.add_argument('--prediction-horizon', type=int, default=5,
                       help='Prediction horizon in periods (default: 5)')
    parser.add_argument('--compare-models', action='store_true',
                       help='Compare advanced ML with traditional ML')
    parser.add_argument('--regime-analysis', action='store_true',
                       help='Perform market regime analysis')
    parser.add_argument('--feature-analysis', action='store_true',
                       help='Analyze engineered features')
    parser.add_argument('--save-results', action='store_true',
                       help='Save training results to file')

    return parser.parse_args()


def prepare_training_data(symbol: str, timeframe: str, start_date: str = None,
                         end_date: str = None, days: int = 365):
    """
    Prepare training data for ML models.

    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        start_date: Start date string
        end_date: End date string
        days: Number of days if dates not provided

    Returns:
        DataFrame with indicators
    """
    print(f"📊 Preparing training data for {symbol} {timeframe}...")

    # Determine date range
    if end_date is None:
        end_date = datetime.now()
    else:
        end_date = datetime.strptime(end_date, '%Y-%m-%d')

    if start_date is None:
        start_date = end_date - timedelta(days=days)
    else:
        start_date = datetime.strptime(start_date, '%Y-%m-%d')

    print(f"📅 Date range: {start_date.date()} to {end_date.date()}")

    # Fetch historical data
    try:
        df = fetch_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            limit=None
        )

        if df is None or df.empty:
            raise ValueError("No data retrieved")

        print(f"📈 Raw data: {len(df)} periods")

        # Calculate indicators
        df_with_indicators = calculate_indicators(df)

        print(f"🔧 Data with indicators: {len(df_with_indicators)} periods")
        print(f"📊 Available columns: {len(df_with_indicators.columns)}")

        return df_with_indicators

    except Exception as e:
        print(f"❌ Failed to prepare training data: {e}")
        return None


def train_advanced_ml_model(df, symbol: str, timeframe: str, test_size: float = 0.2,
                           prediction_horizon: int = 5):
    """
    Train the advanced ML model.

    Args:
        df: DataFrame with indicators
        symbol: Trading symbol
        timeframe: Timeframe
        test_size: Test set size
        prediction_horizon: Prediction horizon

    Returns:
        Training results
    """
    if not is_advanced_ml_available():
        print("❌ Advanced ML features not available")
        return None

    print("🚀 Training Advanced ML Ensemble...")

    try:
        # Get advanced ML predictor
        predictor = get_advanced_ml_predictor(symbol=symbol, timeframe=timeframe)

        # Train the model
        training_results = predictor.train_ensemble(
            df=df,
            test_size=test_size,
            prediction_horizon=prediction_horizon
        )

        if training_results and training_results.get('success', False):
            print("✅ Advanced ML training completed successfully")
            print(f"🎯 Models trained: {len(training_results.get('training_results', {}))}")
            print(f"⚖️  Ensemble weights: {training_results.get('ensemble_weights', {})}")

            # Log training completion
            audit_logger.log_system_event(
                event_type="ADVANCED_ML_TRAINING_SCRIPT_COMPLETED",
                message=f"Advanced ML training script completed for {symbol} {timeframe}",
                details={
                    'models_trained': len(training_results.get('training_results', {})),
                    'feature_count': training_results.get('feature_count', 0),
                    'test_size': test_size,
                    'prediction_horizon': prediction_horizon
                }
            )

            return training_results
        else:
            print("❌ Advanced ML training failed")
            return None

    except Exception as e:
        print(f"❌ Advanced ML training error: {e}")
        return None


def compare_with_traditional_ml(df, symbol: str, timeframe: str, test_size: float = 0.2):
    """
    Compare advanced ML with traditional ML models.

    Args:
        df: DataFrame with indicators
        symbol: Trading symbol
        timeframe: Timeframe
        test_size: Test set size

    Returns:
        Comparison results
    """
    print("🔄 Comparing Advanced ML with Traditional ML...")

    results = {
        'advanced_ml': None,
        'traditional_ml': None,
        'comparison': {}
    }

    try:
        # Test advanced ML
        advanced_predictor = get_advanced_ml_predictor(symbol=symbol, timeframe=timeframe)
        if advanced_predictor and advanced_predictor.is_trained:
            # Get last signal
            last_signal = advanced_predictor.predict_signal(df.tail(100))
            results['advanced_ml'] = {
                'signal': last_signal.get('signal'),
                'confidence': last_signal.get('confidence'),
                'regime': last_signal.get('regime'),
                'model_info': advanced_predictor.get_model_info()
            }

        # Test traditional ML
        traditional_predictor = get_ml_predictor(symbol=symbol, timeframe=timeframe)
        if traditional_predictor:
            # Get last signal
            last_signal = traditional_predictor.predict_signal(df.tail(100))
            results['traditional_ml'] = {
                'signal': last_signal.get('signal'),
                'confidence': last_signal.get('confidence'),
                'model_info': traditional_predictor.get_model_info()
            }

        # Compare results
        if results['advanced_ml'] and results['traditional_ml']:
            adv_conf = results['advanced_ml']['confidence']
            trad_conf = results['traditional_ml']['confidence']

            results['comparison'] = {
                'confidence_difference': adv_conf - trad_conf,
                'same_signal': results['advanced_ml']['signal'] == results['traditional_ml']['signal'],
                'advanced_better_confidence': adv_conf > trad_conf,
                'regime_adaptation': results['advanced_ml'].get('regime') is not None
            }

            print("📊 Comparison Results:")
            print(f"   Advanced ML - Signal: {results['advanced_ml']['signal']}, Confidence: {adv_conf:.3f}")
            print(f"   Traditional ML - Signal: {results['traditional_ml']['signal']}, Confidence: {trad_conf:.3f}")
            print(f"   Same signal: {results['comparison']['same_signal']}")
            print(f"   Advanced better: {results['comparison']['advanced_better_confidence']}")

        return results

    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        return results


def perform_regime_analysis(df, symbol: str, timeframe: str):
    """
    Perform market regime analysis.

    Args:
        df: DataFrame with indicators
        symbol: Trading symbol
        timeframe: Timeframe

    Returns:
        Regime analysis results
    """
    print("📈 Performing Market Regime Analysis...")

    try:
        regime_detector = MarketRegimeDetector()

        # Analyze entire dataset
        regimes = []
        for i in range(50, len(df), 10):  # Sample every 10 periods
            window_df = df.iloc[max(0, i-50):i+1]
            regime_info = regime_detector.detect_regime(window_df)
            regimes.append(regime_info)

        # Get statistics
        stats = regime_detector.get_regime_statistics()

        analysis_results = {
            'total_regimes_analyzed': len(regimes),
            'regime_statistics': stats,
            'recent_regime': regime_detector.detect_regime(df.tail(50)) if len(df) > 50 else None,
            'regime_distribution': {}
        }

        # Calculate distribution
        regime_counts = {}
        for regime in regimes:
            reg_type = regime['regime']
            regime_counts[reg_type] = regime_counts.get(reg_type, 0) + 1

        total_regimes = len(regimes)
        analysis_results['regime_distribution'] = {
            reg_type: {
                'count': count,
                'percentage': count / total_regimes * 100
            }
            for reg_type, count in regime_counts.items()
        }

        print("📊 Regime Analysis Results:")
        for reg_type, data in analysis_results['regime_distribution'].items():
            print(f"   {reg_type}: {data['count']} ({data['percentage']:.1f}%)")

        return analysis_results

    except Exception as e:
        print(f"❌ Regime analysis failed: {e}")
        return None


def analyze_features(df, symbol: str, timeframe: str):
    """
    Analyze engineered features.

    Args:
        df: DataFrame with indicators
        symbol: Trading symbol
        timeframe: Timeframe

    Returns:
        Feature analysis results
    """
    print("🔍 Analyzing Engineered Features...")

    try:
        feature_engineer = AdvancedFeatureEngineer()

        # Create features
        features, feature_names = feature_engineer.create_advanced_features(df)

        analysis_results = {
            'total_features': len(feature_names),
            'feature_names': feature_names[:20],  # First 20 features
            'feature_stats': {},
            'correlation_analysis': {}
        }

        # Basic statistics for key features
        if len(features) > 0:
            for i, name in enumerate(feature_names[:10]):  # Analyze first 10 features
                if i < features.shape[1]:
                    feature_data = features[:, i]
                    valid_data = feature_data[~np.isnan(feature_data)]

                    if len(valid_data) > 0:
                        analysis_results['feature_stats'][name] = {
                            'mean': float(np.mean(valid_data)),
                            'std': float(np.std(valid_data)),
                            'min': float(np.min(valid_data)),
                            'max': float(np.max(valid_data)),
                            'nan_count': int(np.sum(np.isnan(feature_data)))
                        }

        # Correlation with price (if available)
        if 'close' in df.columns and len(features) > 0:
            price_changes = df['close'].pct_change().fillna(0).values
            correlations = []

            for i in range(min(10, features.shape[1])):  # First 10 features
                feature_data = features[:, i]
                valid_mask = ~(np.isnan(feature_data) | np.isnan(price_changes))
                if np.sum(valid_mask) > 10:
                    corr = np.corrcoef(feature_data[valid_mask], price_changes[valid_mask])[0, 1]
                    correlations.append((feature_names[i], float(corr)))

            analysis_results['correlation_analysis'] = {
                'price_correlations': correlations
            }

        print("📊 Feature Analysis Results:")
        print(f"   Total features: {analysis_results['total_features']}")
        print(f"   Features with stats: {len(analysis_results['feature_stats'])}")

        return analysis_results

    except Exception as e:
        print(f"❌ Feature analysis failed: {e}")
        return None


def save_results(results: dict, symbol: str, timeframe: str):
    """
    Save training results to file.

    Args:
        results: Results dictionary
        symbol: Trading symbol
        timeframe: Timeframe
    """
    try:
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"advanced_ml_training_{symbol.replace('/', '_')}_{timeframe}_{timestamp}.json"

        with open(output_dir / filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"💾 Results saved to: output/{filename}")

    except Exception as e:
        print(f"❌ Failed to save results: {e}")


def main():
    """Main training function."""
    args = parse_arguments()

    print("🤖 Advanced ML Training Script")
    print("=" * 50)

    # Check if advanced ML is available
    if not ML_ADVANCED_FEATURES_ENABLED:
        print("⚠️  Advanced ML features are disabled in configuration")
        print("   Enable with: export ML_ADVANCED_FEATURES_ENABLED=true")
        return

    if not is_advanced_ml_available():
        print("❌ Advanced ML dependencies not available")
        print("   Required: PyTorch, TensorFlow, or scikit-learn")
        return

    # Prepare training data
    df = prepare_training_data(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
        days=args.days
    )

    if df is None or df.empty:
        print("❌ No training data available")
        return

    results = {
        'training_config': {
            'symbol': args.symbol,
            'timeframe': args.timeframe,
            'test_size': args.test_size,
            'prediction_horizon': args.prediction_horizon,
            'data_points': len(df)
        },
        'timestamp': datetime.now().isoformat()
    }

    # Train advanced ML model
    training_results = train_advanced_ml_model(
        df=df,
        symbol=args.symbol,
        timeframe=args.timeframe,
        test_size=args.test_size,
        prediction_horizon=args.prediction_horizon
    )

    if training_results:
        results['training_results'] = training_results

    # Compare with traditional ML
    if args.compare_models:
        comparison_results = compare_with_traditional_ml(
            df=df,
            symbol=args.symbol,
            timeframe=args.timeframe,
            test_size=args.test_size
        )
        if comparison_results:
            results['model_comparison'] = comparison_results

    # Perform regime analysis
    if args.regime_analysis:
        regime_results = perform_regime_analysis(
            df=df,
            symbol=args.symbol,
            timeframe=args.timeframe
        )
        if regime_results:
            results['regime_analysis'] = regime_results

    # Analyze features
    if args.feature_analysis:
        feature_results = analyze_features(
            df=df,
            symbol=args.symbol,
            timeframe=args.timeframe
        )
        if feature_results:
            results['feature_analysis'] = feature_results

    # Save results
    if args.save_results:
        save_results(results, args.symbol, args.timeframe)

    print("\n🎉 Advanced ML Training Script Completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()