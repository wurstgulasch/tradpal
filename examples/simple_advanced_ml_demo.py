#!/usr/bin/env python3
"""
Simple Advanced ML Trading Demo for TradPal

This is a simplified demo that shows the core advanced ML functionality
without external dependencies.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from services.mlops_service.advanced_ml_models import (
    ModelConfig, LSTMTradingModel, TransformerTradingModel,
    EnsembleTradingModel, AutoMLSelector, create_trading_model
)
from services.mlops_service.advanced_feature_engineering import (
    FeatureConfig, AdvancedFeatureEngineer, create_feature_engineer
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data(n_samples=500):
    """Create sample OHLCV trading data."""
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')

    # Generate synthetic price data with trends and volatility
    np.random.seed(42)
    base_price = 100
    prices = [base_price]

    for i in range(1, n_samples):
        # Add trend, seasonality, and random walk
        trend = 0.0005 * np.sin(i / 50)  # Seasonal trend
        random_walk = np.random.normal(0, 0.015)  # Daily volatility
        new_price = prices[-1] * (1 + trend + random_walk)
        prices.append(max(new_price, 1))  # Ensure positive prices

    # Create OHLCV data
    data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(100000, 1000000, n_samples)
    }, index=dates)

    return data

def create_sequences(data, sequence_length):
    """Create sequences for LSTM input."""
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequences.append(data[i:i + sequence_length])
    return np.array(sequences)

def demo_feature_engineering():
    """Demonstrate feature engineering."""
    logger.info("=== Feature Engineering Demo ===")

    # Create sample data
    data = create_sample_data(200)
    logger.info(f"Created {len(data)} data points")

    # Create feature engineer
    config = FeatureConfig(
        include_technical_indicators=True,
        include_statistical_features=True,
        include_microstructure_features=True,
        lookback_periods=[5, 10, 20]
    )

    feature_engineer = create_feature_engineer(config)

    # Create features
    logger.info("Creating features...")
    features = feature_engineer.create_all_features(data)
    logger.info(f"Created {features.shape[1]} features")

    # Show some statistics
    logger.info("Feature statistics:")
    logger.info(f"Total features: {features.shape[1]}")
    logger.info(f"Features with NaN: {features.isnull().any().sum()}")
    logger.info(f"Complete feature rows: {features.dropna().shape[0]}")

    # Plot price vs some features
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Price
    axes[0,0].plot(data['close'], label='Close Price')
    axes[0,0].set_title('Original Price Data')
    axes[0,0].legend()

    # RSI if available
    if 'rsi_14' in features.columns:
        axes[0,1].plot(features['rsi_14'], label='RSI')
        axes[0,1].axhline(y=70, color='r', linestyle='--', alpha=0.5)
        axes[0,1].axhline(y=30, color='g', linestyle='--', alpha=0.5)
        axes[0,1].set_title('RSI Indicator')
        axes[0,1].legend()

    # MACD if available
    if 'macd' in features.columns and 'macd_signal' in features.columns:
        axes[1,0].plot(features['macd'], label='MACD')
        axes[1,0].plot(features['macd_signal'], label='Signal')
        axes[1,0].set_title('MACD Indicator')
        axes[1,0].legend()

    # Volume if available
    if 'volume' in data.columns:
        axes[1,1].plot(data['volume'], label='Volume')
        axes[1,1].set_title('Volume Data')
        axes[1,1].legend()

    plt.tight_layout()
    plt.savefig('feature_demo.png', dpi=150, bbox_inches='tight')
    plt.show()

    return features

def demo_ml_models(features):
    """Demonstrate ML model training."""
    logger.info("\n=== ML Model Training Demo ===")

    # Prepare target (next day return prediction)
    data = create_sample_data(200)
    target = data['close'].pct_change().shift(-1).dropna()  # Next day return
    features = features.iloc[:-1]  # Remove last row to match target

    logger.info(f"Features shape: {features.shape}")
    logger.info(f"Target shape: {target.shape}")

    # Split data
    split_idx = int(len(features) * 0.7)
    X_train = features.iloc[:split_idx]
    y_train = target.iloc[:split_idx]
    X_test = features.iloc[split_idx:]
    y_test = target.iloc[split_idx:]

    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Test different models
    models_to_test = ['lstm', 'ensemble']
    results = {}

    for model_name in models_to_test:
        logger.info(f"\nTesting {model_name.upper()} model...")

        try:
            # Create model config
            config = ModelConfig(
                model_type=model_name,
                input_size=X_train.shape[1],
                output_size=1,
                hidden_size=32,
                epochs=5,  # Few epochs for demo
                batch_size=16,
                sequence_length=10  # For LSTM
            )

            model = create_trading_model(model_name, config)

            if model_name == 'lstm':
                # Prepare sequence data for LSTM
                X_train_seq = create_sequences(X_train.values, config.sequence_length)
                X_test_seq = create_sequences(X_test.values, config.sequence_length)
                y_train_seq = y_train.values[config.sequence_length:]
                y_test_seq = y_test.values[config.sequence_length:]

                logger.info(f"LSTM sequences - Train: {X_train_seq.shape}, Test: {X_test_seq.shape}")

                # Mock training for demo
                logger.info("Mock training (would train actual model in production)...")
                with patch.object(model, '_create_data_loaders') as mock_loaders:
                    mock_train_loader = Mock()
                    mock_val_loader = Mock()
                    mock_loaders.return_value = (mock_train_loader, mock_val_loader)

                    with patch('services.core.gpu_accelerator.train_gpu_model') as mock_train:
                        mock_train.return_value = {
                            'loss': np.random.exponential(0.1, 5).tolist(),
                            'val_loss': np.random.exponential(0.15, 5).tolist()
                        }

                        model.train(X_train_seq, y_train_seq.reshape(-1, 1))
                        model.is_trained = True

                # Make predictions
                predictions = model.predict(X_test_seq)

            elif model_name == 'ensemble':
                # Train ensemble model
                model.train(X_train.values, y_train.values.reshape(-1, 1))

                # Make predictions
                predictions = model.predict(X_test.values)

            # Calculate simple metrics
            mse = np.mean((predictions.flatten() - y_test.values[-len(predictions):]) ** 2)
            mae = np.mean(np.abs(predictions.flatten() - y_test.values[-len(predictions):]))

            results[model_name] = {
                'mse': mse,
                'mae': mae,
                'predictions': predictions.flatten()[:10]  # First 10 predictions
            }

            logger.info(".6f")
            logger.info(".6f")

        except Exception as e:
            logger.error(f"Error with {model_name}: {e}")
            results[model_name] = None

    # Plot comparison
    plt.figure(figsize=(12, 6))

    model_names = [m for m in results.keys() if results[m] is not None]
    mse_scores = [results[m]['mse'] for m in model_names]

    plt.bar(model_names, mse_scores)
    plt.title('Model MSE Comparison')
    plt.ylabel('MSE')
    plt.yscale('log')

    for i, v in enumerate(mse_scores):
        plt.text(i, v * 1.1, '.4f', ha='center')

    plt.tight_layout()
    plt.savefig('model_comparison_demo.png', dpi=150, bbox_inches='tight')
    plt.show()

    return results

def demo_automl():
    """Demonstrate AutoML selection."""
    logger.info("\n=== AutoML Demo ===")

    # Create AutoML selector
    selector = AutoMLSelector()

    # Add models
    base_config = ModelConfig(
        model_type='auto',  # Will be overridden
        input_size=50,  # Simplified
        output_size=1,
        hidden_size=16,
        epochs=3
    )

    selector.add_model('lstm', LSTMTradingModel(base_config))
    selector.add_model('ensemble', EnsembleTradingModel(base_config))

    logger.info(f"AutoML selector created with {len(selector.models)} models")
    logger.info("Models available: LSTM, Ensemble")
    logger.info("AutoML would automatically select the best performing model")

def main():
    """Main demo function."""
    logger.info("TradPal Advanced ML Trading Demo")
    logger.info("=" * 50)

    try:
        # Feature engineering demo
        features = demo_feature_engineering()

        # ML model demo
        model_results = demo_ml_models(features)

        # AutoML demo
        demo_automl()

        logger.info("\n" + "=" * 50)
        logger.info("Demo Complete!")
        logger.info("Generated files:")
        logger.info("- feature_demo.png")
        logger.info("- model_comparison_demo.png")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise

if __name__ == '__main__':
    from unittest.mock import Mock, patch
    main()