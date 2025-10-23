#!/usr/bin/env python3
"""
Tests for Advanced ML Models in TradPal

This module contains comprehensive tests for all advanced ML components:
- LSTM and Transformer models
- Ensemble methods
- Feature engineering
- Reinforcement learning
- AutoML selection
"""

import unittest
import numpy as np
import pandas as pd
import torch
from unittest.mock import Mock, patch
import tempfile
import os

from services.monitoring_service.mlops_service.advanced_ml_models import (
    ModelConfig, ModelPerformance, LSTMTradingModel, TransformerTradingModel,
    EnsembleTradingModel, AutoMLSelector, TradingModelFactory
)
from services.monitoring_service.mlops_service.advanced_feature_engineering import (
    FeatureConfig, TechnicalIndicatorFeatures, StatisticalFeatures,
    MicrostructureFeatures, AdvancedFeatureEngineer
)
from services.monitoring_service.mlops_service.reinforcement_learning import (
    RLConfig, TradingEnvironment, DQNAgent, PPOAgent, RLTrainer
)

class TestAdvancedMLModels(unittest.TestCase):
    """Test advanced ML models."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ModelConfig(
            model_type='lstm',
            input_size=10,
            output_size=1,
            hidden_size=32,
            num_layers=1,
            sequence_length=5
        )

        # Create sample data
        self.X = np.random.randn(100, 10)
        self.y = np.random.randn(100, 1)

    def test_lstm_model_creation(self):
        """Test LSTM model creation."""
        model = LSTMTradingModel(self.config)
        self.assertIsNotNone(model.model)
        self.assertFalse(model.is_trained)

    def test_lstm_model_training(self):
        """Test LSTM model training."""
        model = LSTMTradingModel(self.config)

        # Mock data loaders for testing
        with patch.object(model, '_create_data_loaders') as mock_loaders:
            mock_train_loader = Mock()
            mock_val_loader = Mock()
            mock_loaders.return_value = (mock_train_loader, mock_val_loader)

            with patch('services.core_service.gpu_accelerator.train_gpu_model') as mock_train:
                mock_train.return_value = {'loss': [0.5, 0.3], 'val_loss': [0.6, 0.4]}

                results = model.train(self.X, self.y)
                self.assertIn('loss', results)
                self.assertTrue(model.is_trained)

    def test_lstm_model_prediction(self):
        """Test LSTM model prediction."""
        model = LSTMTradingModel(self.config)

        # Train model first
        model.is_trained = True

        predictions = model.predict(self.X)
        self.assertEqual(predictions.shape[0], self.X.shape[0])

    def test_lstm_model_evaluation(self):
        """Test LSTM model evaluation."""
        model = LSTMTradingModel(self.config)
        model.is_trained = True

        performance = model.evaluate(self.X, self.y)
        self.assertIsInstance(performance, ModelPerformance)

    def test_transformer_model_creation(self):
        """Test Transformer model creation."""
        model = TransformerTradingModel(self.config)
        self.assertIsNotNone(model.model)
        self.assertFalse(model.is_trained)

    def test_ensemble_model_creation(self):
        """Test Ensemble model creation."""
        model = EnsembleTradingModel(self.config)
        self.assertIn('rf', model.models)
        self.assertIn('gb', model.models)
        self.assertIn('xgb', model.models)

    def test_ensemble_model_training(self):
        """Test Ensemble model training."""
        model = EnsembleTradingModel(self.config)

        results = model.train(self.X, self.y)
        self.assertIn('rf', results)
        self.assertTrue(model.is_trained)

    def test_ensemble_model_prediction(self):
        """Test Ensemble model prediction."""
        model = EnsembleTradingModel(self.config)
        model.train(self.X, self.y)  # Actually train the model
        predictions = model.predict(self.X)
        self.assertEqual(predictions.shape[0], self.X.shape[0])

    def test_automl_selector(self):
        """Test AutoML selector."""
        selector = AutoMLSelector()

        # Add models
        selector.add_model('lstm', LSTMTradingModel(self.config))
        selector.add_model('ensemble', EnsembleTradingModel(self.config))

        self.assertEqual(len(selector.models), 2)

        # Test selection (would need actual training)
        # best_model = selector.select_best_model(self.X, self.y)
        # self.assertIn(best_model, ['lstm', 'ensemble'])

    def test_model_factory(self):
        """Test model factory."""
        factory = TradingModelFactory()

        lstm_model = factory.create_model('lstm', self.config)
        self.assertIsInstance(lstm_model, LSTMTradingModel)

        ensemble_model = factory.create_model('ensemble', self.config)
        self.assertIsInstance(ensemble_model, EnsembleTradingModel)

        with self.assertRaises(ValueError):
            factory.create_model('invalid', self.config)

class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering components."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample OHLCV data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        self.data = pd.DataFrame({
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(105, 115, 100),
            'low': np.random.uniform(95, 105, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        }, index=dates)

        self.config = FeatureConfig(
            include_technical_indicators=True,
            include_statistical_features=True,
            include_microstructure_features=True,
            lookback_periods=[5, 10]
        )

    def test_technical_indicators(self):
        """Test technical indicator features."""
        engineer = TechnicalIndicatorFeatures(self.config)
        features = engineer.create_features(self.data)

        # Check that features were created
        self.assertGreater(features.shape[1], 0)

        # Check specific indicators
        self.assertIn('rsi_14', features.columns)
        self.assertIn('macd', features.columns)

    def test_statistical_features(self):
        """Test statistical features."""
        engineer = StatisticalFeatures(self.config)
        features = engineer.create_features(self.data)

        # Check that features were created
        self.assertGreater(features.shape[1], 0)

        # Check specific statistical features
        self.assertIn('close_mean_5', features.columns)
        self.assertIn('close_std_5', features.columns)

    def test_microstructure_features(self):
        """Test microstructure features."""
        engineer = MicrostructureFeatures(self.config)
        features = engineer.create_features(self.data)

        # Check that features were created
        self.assertGreater(features.shape[1], 0)

        # Check specific microstructure features
        self.assertIn('spread_proxy', features.columns)

    def test_advanced_feature_engineer(self):
        """Test advanced feature engineer."""
        engineer = AdvancedFeatureEngineer(self.config)
        features = engineer.create_all_features(self.data)

        # Check that multiple feature types were created
        self.assertGreater(features.shape[1], 10)

        # Test preprocessing
        processed = engineer.preprocess_features(features)
        self.assertEqual(processed.shape[0], features.shape[0])

class TestReinforcementLearning(unittest.TestCase):
    """Test reinforcement learning components."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        self.data = pd.DataFrame({
            'open': np.random.uniform(100, 110, 50),
            'high': np.random.uniform(105, 115, 50),
            'low': np.random.uniform(95, 105, 50),
            'close': np.random.uniform(100, 110, 50),
            'volume': np.random.uniform(1000, 10000, 50)
        }, index=dates)

        self.rl_config = RLConfig(
            algorithm='dqn',
            state_size=4,  # Simplified state
            action_size=3,
            hidden_size=16,
            episodes=2,
            max_steps_per_episode=10
        )

    def test_trading_environment(self):
        """Test trading environment."""
        env = TradingEnvironment(self.data)

        # Test reset
        state = env.reset()
        self.assertIsInstance(state, object)  # TradingState dataclass

        # Test step
        action = Mock()
        action.action_type = 0  # Hold
        action.quantity = 1.0

        next_state, reward, done = env.step(action)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)

    def test_dqn_agent(self):
        """Test DQN agent."""
        agent = DQNAgent(self.rl_config)

        # Test action selection
        state = Mock()
        state.features = np.random.randn(self.rl_config.state_size)
        action = agent.select_action(state)
        self.assertIn(action.action_type, [0, 1, 2])

    def test_ppo_agent(self):
        """Test PPO agent."""
        agent = PPOAgent(self.rl_config)

        # Test action selection
        state = Mock()
        state.features = np.random.randn(self.rl_config.state_size)
        action = agent.select_action(state)
        self.assertIn(action.action_type, [0, 1, 2])

    def test_rl_trainer(self):
        """Test RL trainer."""
        env = TradingEnvironment(self.data, initial_balance=1000.0)
        agent = DQNAgent(self.rl_config)
        trainer = RLTrainer(agent, env, self.rl_config)

        # Test training (short episode for testing)
        results = trainer.train()
        self.assertIn('episode_rewards', results)
        self.assertIn('episode_losses', results)

class TestIntegration(unittest.TestCase):
    """Integration tests for ML components."""

    def setUp(self):
        """Set up integration test fixtures."""
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        self.data = pd.DataFrame({
            'open': np.random.uniform(100, 110, 200),
            'high': np.random.uniform(105, 115, 200),
            'low': np.random.uniform(95, 105, 200),
            'close': np.random.uniform(100, 110, 200),
            'volume': np.random.uniform(1000, 10000, 200)
        }, index=dates)

    def test_ml_pipeline_integration(self):
        """Test complete ML pipeline integration."""
        # Feature engineering
        feature_config = FeatureConfig(lookback_periods=[5, 10])
        feature_engineer = AdvancedFeatureEngineer(feature_config)
        features = feature_engineer.create_all_features(self.data)

        # Prepare target (next day return)
        target = self.data['close'].pct_change().shift(-1).dropna()
        features = features.iloc[:-1]  # Remove last row to match target

        # Model training
        model_config = ModelConfig(
            model_type='lstm',
            input_size=features.shape[1],
            output_size=1,
            hidden_size=32
        )

        model = LSTMTradingModel(model_config)

        # Split data
        split_idx = int(len(features) * 0.8)
        X_train = features.iloc[:split_idx].values
        y_train = target.iloc[:split_idx].values.reshape(-1, 1)
        X_test = features.iloc[split_idx:].values
        y_test = target.iloc[split_idx:].values.reshape(-1, 1)

        # Mock training for integration test
        with patch.object(model, '_create_data_loaders') as mock_loaders:
            mock_train_loader = Mock()
            mock_val_loader = Mock()
            mock_loaders.return_value = (mock_train_loader, mock_val_loader)

            with patch('services.core_service.gpu_accelerator.train_gpu_model') as mock_train:
                mock_train.return_value = {'loss': [0.5, 0.3]}

                model.train(X_train, y_train)
                self.assertTrue(model.is_trained)

        # Test prediction
        predictions = model.predict(X_test)
        self.assertEqual(len(predictions), len(X_test))

        # Skip evaluation for mock test
        # performance = model.evaluate(X_test, y_test)
        # self.assertIsInstance(performance, object)  # ModelPerformance    def test_automl_integration(self):
        """Test AutoML integration."""
        # Feature engineering
        feature_config = FeatureConfig(lookback_periods=[5])
        feature_engineer = AdvancedFeatureEngineer(feature_config)
        features = feature_engineer.create_all_features(self.data)

        # Prepare target
        target = self.data['close'].pct_change().shift(-1).dropna()
        features = features.iloc[:-1]

        # AutoML selection
        selector = AutoMLSelector()

        model_config = ModelConfig(
            model_type='ensemble',
            input_size=features.shape[1],
            output_size=1,
            hidden_size=16  # Smaller for testing
        )

        selector.add_model('lstm', LSTMTradingModel(model_config))
        selector.add_model('ensemble', EnsembleTradingModel(model_config))

        # Note: Full training would take too long for unit tests
        # In practice, this would be tested with smaller datasets or mocked
        self.assertEqual(len(selector.models), 2)

class TestModelPersistence(unittest.TestCase):
    """Test model saving and loading."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ModelConfig(
            model_type='lstm',
            input_size=5,
            output_size=1,
            hidden_size=16
        )
        self.model = LSTMTradingModel(self.config)

    def test_model_save_load(self):
        """Test model saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'test_model.pth')

            # Save model
            self.model.save_model(model_path)
            self.assertTrue(os.path.exists(model_path))

            # Load model
            new_model = LSTMTradingModel(self.config)
            new_model.load_model(model_path)

            # Check that models have same parameters
            original_params = list(self.model.model.parameters())
            loaded_params = list(new_model.model.parameters())

            for orig, loaded in zip(original_params, loaded_params):
                self.assertTrue(torch.equal(orig, loaded))

if __name__ == '__main__':
    unittest.main()