#!/usr/bin/env python3
"""
Advanced ML Trading Example for TradPal

This example demonstrates how to use the advanced ML models for trading predictions,
including LSTM, Transformer, Ensemble methods, and reinforcement learning.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from services.mlops_service.advanced_ml_models import (
    ModelConfig, LSTMTradingModel, TransformerTradingModel,
    EnsembleTradingModel, AutoMLSelector, create_trading_model
)
from services.mlops_service.advanced_feature_engineering import (
    FeatureConfig, AdvancedFeatureEngineer, create_feature_engineer
)
from services.mlops_service.reinforcement_learning import (
    RLConfig, create_trading_environment, create_rl_agent, create_rl_trainer
)
from services.data_service.data_service import DataService
from services.core.gpu_accelerator import get_gpu_accelerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_sample_data():
    """Load sample trading data."""
    # Create sample data (in practice, use real market data)
    dates = pd.date_range('2023-01-01', periods=500, freq='D')

    # Simulate price data with some trends and volatility
    np.random.seed(42)
    base_price = 100
    prices = [base_price]

    for i in range(1, 500):
        # Add trend and random walk
        trend = 0.001 * np.sin(i / 50)  # Seasonal trend
        random_walk = np.random.normal(0, 0.02)  # Daily volatility
        new_price = prices[-1] * (1 + trend + random_walk)
        prices.append(max(new_price, 1))  # Ensure positive prices

    # Create OHLCV data
    data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(100000, 1000000, 500)
    }, index=dates)

    return data

def demonstrate_feature_engineering():
    """Demonstrate advanced feature engineering."""
    logger.info("=== Advanced Feature Engineering Demo ===")

    # Load data
    data = load_sample_data()
    logger.info(f"Loaded {len(data)} data points")

    # Create feature engineer
    feature_config = FeatureConfig(
        include_technical_indicators=True,
        include_statistical_features=True,
        include_microstructure_features=True,
        lookback_periods=[5, 10, 20, 50]
    )

    feature_engineer = create_feature_engineer(feature_config)

    # Create features
    logger.info("Creating features...")
    features = feature_engineer.create_all_features(data)
    logger.info(f"Created {features.shape[1]} features")

    # Display some feature statistics
    logger.info("Feature statistics:")
    logger.info(features.describe())

    # Plot some features
    plt.figure(figsize=(15, 10))

    # Plot original price
    plt.subplot(2, 2, 1)
    plt.plot(data['close'], label='Close Price')
    plt.title('Original Price Data')
    plt.legend()

    # Plot some technical indicators
    plt.subplot(2, 2, 2)
    if 'rsi_14' in features.columns:
        plt.plot(features['rsi_14'], label='RSI')
        plt.axhline(y=70, color='r', linestyle='--', alpha=0.5)
        plt.axhline(y=30, color='g', linestyle='--', alpha=0.5)
        plt.title('RSI Indicator')
        plt.legend()

    # Plot MACD
    plt.subplot(2, 2, 3)
    if 'macd' in features.columns and 'macd_signal' in features.columns:
        plt.plot(features['macd'], label='MACD')
        plt.plot(features['macd_signal'], label='Signal')
        plt.title('MACD Indicator')
        plt.legend()

    # Plot statistical features
    plt.subplot(2, 2, 4)
    if 'close_mean_20' in features.columns and 'close_std_20' in features.columns:
        plt.plot(features['close_mean_20'], label='20-day MA')
        plt.fill_between(features.index,
                        features['close_mean_20'] - features['close_std_20'],
                        features['close_mean_20'] + features['close_std_20'],
                        alpha=0.3, label='±1 STD')
        plt.title('Statistical Features')
        plt.legend()

    plt.tight_layout()
    plt.savefig('feature_engineering_demo.png', dpi=300, bbox_inches='tight')
    plt.show()

    return features

def demonstrate_ml_models(features):
    """Demonstrate ML model training and prediction."""
    logger.info("\n=== ML Model Training Demo ===")

    # Prepare target (next day return prediction)
    data = load_sample_data()
    target = data['close'].pct_change().shift(-1).dropna()  # Next day return
    features = features.iloc[:-1]  # Remove last row to match target

    # Split data
    split_idx = int(len(features) * 0.7)
    X_train = features.iloc[:split_idx]
    y_train = target.iloc[:split_idx]
    X_val = features.iloc[split_idx:int(len(features) * 0.85)]
    y_val = target.iloc[split_idx:int(len(features) * 0.85)]
    X_test = features.iloc[int(len(features) * 0.85):]
    y_test = target.iloc[int(len(features) * 0.85):]

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Model configurations
    model_configs = {
        'lstm': ModelConfig(
            model_type='lstm',
            input_size=X_train.shape[1],
            output_size=1,
            hidden_size=64,
            num_layers=2,
            sequence_length=10,
            epochs=50,
            batch_size=32
        ),
        'transformer': ModelConfig(
            model_type='transformer',
            input_size=X_train.shape[1],
            output_size=1,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            sequence_length=10,
            epochs=50,
            batch_size=16  # Smaller batch for transformers
        ),
        'ensemble': ModelConfig(
            model_type='ensemble',
            input_size=X_train.shape[1],
            output_size=1,
            num_trees=50,
            max_depth=6
        )
    }

    results = {}

    # Train and evaluate each model
    for model_name, config in model_configs.items():
        logger.info(f"\nTraining {model_name.upper()} model...")

        try:
            model = create_trading_model(model_name, config)

            # Mock training for demo (real training would take time)
            logger.info("Mock training (would train actual model in production)...")

            # Simulate training results
            if model_name == 'lstm':
                # Mock LSTM training
                with patch.object(model, '_create_data_loaders') as mock_loaders:
                    mock_train_loader = Mock()
                    mock_val_loader = Mock()
                    mock_loaders.return_value = (mock_train_loader, mock_val_loader)

                    with patch('services.core.gpu_accelerator.train_gpu_model') as mock_train:
                        mock_train.return_value = {
                            'loss': np.random.exponential(0.1, 50).tolist(),
                            'val_loss': np.random.exponential(0.15, 50).tolist()
                        }

                        train_results = model.train(X_train.values, y_train.values.reshape(-1, 1),
                                                   validation_data=(X_val.values, y_val.values.reshape(-1, 1)))
                        model.is_trained = True

            elif model_name == 'transformer':
                # Mock Transformer training
                with patch.object(model, '_create_data_loaders') as mock_loaders:
                    mock_train_loader = Mock()
                    mock_val_loader = Mock()
                    mock_loaders.return_value = (mock_train_loader, mock_val_loader)

                    with patch('services.core.gpu_accelerator.train_gpu_model') as mock_train:
                        mock_train.return_value = {
                            'loss': np.random.exponential(0.12, 50).tolist(),
                            'val_loss': np.random.exponential(0.18, 50).tolist()
                        }

                        train_results = model.train(X_train.values, y_train.values.reshape(-1, 1),
                                                   validation_data=(X_val.values, y_val.values.reshape(-1, 1)))
                        model.is_trained = True

            else:  # ensemble
                train_results = model.train(X_train.values, y_train.values.reshape(-1, 1))

            # Evaluate model
            performance = model.evaluate(X_test.values, y_test.values.reshape(-1, 1))

            results[model_name] = {
                'performance': performance,
                'train_results': train_results
            }

            logger.info(f"{model_name.upper()} Results:")
            logger.info(".4f")
            logger.info(".4f")

        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            results[model_name] = None

    # Plot comparison
    plt.figure(figsize=(15, 5))

    models = [m for m in results.keys() if results[m] is not None]
    mse_scores = [results[m]['performance'].mse for m in models]
    r2_scores = [results[m]['performance'].r2_score for m in models]

    plt.subplot(1, 2, 1)
    plt.bar(models, mse_scores)
    plt.title('MSE Comparison')
    plt.ylabel('MSE')

    plt.subplot(1, 2, 2)
    plt.bar(models, r2_scores)
    plt.title('R² Score Comparison')
    plt.ylabel('R² Score')

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    return results

def demonstrate_automl(features):
    """Demonstrate AutoML model selection."""
    logger.info("\n=== AutoML Model Selection Demo ===")

    # Prepare data
    data = load_sample_data()
    target = data['close'].pct_change().shift(-1).dropna()
    features = features.iloc[:-1]

    # Create AutoML selector
    selector = AutoMLSelector()

    # Add models with appropriate configs
    base_config = ModelConfig(
        input_size=features.shape[1],
        output_size=1,
        hidden_size=32,  # Smaller for demo
        epochs=10  # Fewer epochs for demo
    )

    selector.add_model('lstm', LSTMTradingModel(base_config))
    selector.add_model('transformer', TransformerTradingModel(base_config))
    selector.add_model('ensemble', EnsembleTradingModel(base_config))

    logger.info("AutoML will select the best model...")
    logger.info("Note: This would perform actual training in production")

    # In a real scenario, you would call:
    # best_model_name = selector.select_best_model(features.values, target.values)
    # best_model = selector.get_best_model()

    logger.info("AutoML selector created with 3 models")
    logger.info("Best model selection would compare performance metrics")

def demonstrate_reinforcement_learning():
    """Demonstrate reinforcement learning for trading."""
    logger.info("\n=== Reinforcement Learning Demo ===")

    # Load data
    data = load_sample_data()

    # Create trading environment
    env = create_trading_environment(data, initial_balance=10000.0)

    # Create RL agent
    rl_config = RLConfig(
        algorithm='dqn',
        state_size=4,  # Simplified state: [price, position, cash, portfolio_value]
        action_size=3,  # Hold, Buy, Sell
        hidden_size=32,
        episodes=5,  # Few episodes for demo
        max_steps_per_episode=50
    )

    agent = create_rl_agent(rl_config)

    # Create trainer
    trainer = create_rl_trainer(agent, env, rl_config)

    logger.info("Training RL agent...")
    logger.info("Note: This would perform actual RL training in production")

    # In a real scenario, you would call:
    # results = trainer.train()
    # evaluation = trainer.evaluate(episodes=3)

    logger.info("RL setup complete - would train agent on trading environment")

def demonstrate_gpu_acceleration():
    """Demonstrate GPU acceleration capabilities."""
    logger.info("\n=== GPU Acceleration Demo ===")

    gpu_accelerator = get_gpu_accelerator()

    logger.info(f"GPU Available: {gpu_accelerator.is_gpu_available()}")
    logger.info(f"Device: {gpu_accelerator.get_optimal_device()}")

    if gpu_accelerator.is_gpu_available():
        logger.info("GPU acceleration is enabled for ML models")
        logger.info("Models will automatically use GPU for training and inference")
    else:
        logger.info("GPU not available - models will use CPU")
        logger.info("Consider installing CUDA and PyTorch GPU version for acceleration")

def main():
    """Main demonstration function."""
    logger.info("TradPal Advanced ML Trading Demonstration")
    logger.info("=" * 50)

    try:
        # Demonstrate GPU capabilities
        demonstrate_gpu_acceleration()

        # Feature engineering
        features = demonstrate_feature_engineering()

        # ML model training
        model_results = demonstrate_ml_models(features)

        # AutoML selection
        demonstrate_automl(features)

        # Reinforcement learning
        demonstrate_reinforcement_learning()

        logger.info("\n" + "=" * 50)
        logger.info("Advanced ML Demo Complete!")
        logger.info("Generated files:")
        logger.info("- feature_engineering_demo.png")
        logger.info("- model_comparison.png")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise

if __name__ == '__main__':
    main()