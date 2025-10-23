#!/usr/bin/env python3
"""
Advanced ML Models for TradPal Trading System

This module provides advanced machine learning models for trading predictions,
including LSTM, Transformer, Ensemble methods, and reinforcement learning.
"""

import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from dataclasses import dataclass
from abc import ABC, abstractmethod

from services.core_service.gpu_accelerator import get_gpu_accelerator, is_gpu_available

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for ML models."""
    model_type: str
    input_size: int
    output_size: int
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    sequence_length: int = 60
    num_heads: int = 8
    num_trees: int = 100
    max_depth: int = 6

@dataclass
class ModelPerformance:
    """Model performance metrics."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    mse: float = 0.0
    mae: float = 0.0
    r2_score: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_return: float = 0.0

class BaseTradingModel(ABC):
    """Abstract base class for trading models."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.is_trained = False
        self.performance = ModelPerformance()

    @abstractmethod
    def build_model(self) -> None:
        """Build the model architecture."""
        pass

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray,
             validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, List[float]]:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass

    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> ModelPerformance:
        """Evaluate model performance."""
        pass

    def save_model(self, path: str) -> None:
        """Save model to disk."""
        if self.model is not None:
            torch.save(self.model.state_dict(), path)
            logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load model from disk."""
        if self.model is not None:
            self.model.load_state_dict(torch.load(path))
            self.model.eval()
            logger.info(f"Model loaded from {path}")

class LSTMTradingModel(BaseTradingModel):
    """LSTM-based trading prediction model."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.device = get_gpu_accelerator().get_optimal_device()
        self.build_model()

    def build_model(self) -> None:
        """Build LSTM model architecture."""
        from services.core_service.gpu_accelerator import create_gpu_lstm_model
        self.model = create_gpu_lstm_model(
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            output_size=self.config.output_size
        )

    def train(self, X: np.ndarray, y: np.ndarray,
             validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, List[float]]:
        """Train LSTM model."""
        from services.core_service.gpu_accelerator import train_gpu_model

        # Create data loaders
        train_loader, val_loader = self._create_data_loaders(X, y, validation_data)

        # Train model
        results = train_gpu_model(
            model=self.model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=self.config.epochs,
            learning_rate=self.config.learning_rate
        )

        self.is_trained = True
        return results

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with LSTM model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

            # Handle different input shapes
            if X_tensor.dim() == 2:
                # Single timestep: (batch_size, input_size) -> (batch_size, 1, input_size)
                X_tensor = X_tensor.unsqueeze(1)
            elif X_tensor.dim() == 3:
                # Already in sequence format: (batch_size, seq_len, input_size)
                pass
            else:
                raise ValueError(f"Input tensor must be 2D or 3D, got {X_tensor.dim()}D")

            predictions = self.model(X_tensor)
            return predictions.cpu().numpy()

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> ModelPerformance:
        """Evaluate LSTM model performance."""
        predictions = self.predict(X)

        # Calculate metrics
        mse = mean_squared_error(y, predictions)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)

        # For classification tasks
        if self.config.output_size == 1:
            # Regression metrics
            self.performance.mse = mse
            self.performance.mae = mae
            self.performance.r2_score = r2
        else:
            # Classification metrics
            y_pred_classes = np.argmax(predictions, axis=1)
            y_true_classes = np.argmax(y, axis=1) if y.ndim > 1 else y

            self.performance.accuracy = accuracy_score(y_true_classes, y_pred_classes)
            self.performance.precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
            self.performance.recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
            self.performance.f1_score = f1_score(y_true_classes, y_pred_classes, average='weighted')

        return self.performance

    def _create_data_loaders(self, X: np.ndarray, y: np.ndarray,
                           validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """Create PyTorch data loaders."""
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        val_loader = None
        if validation_data is not None:
            val_X, val_y = validation_data
            val_dataset = torch.utils.data.TensorDataset(
                torch.tensor(val_X, dtype=torch.float32),
                torch.tensor(val_y, dtype=torch.float32)
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False
            )

        return train_loader, val_loader

class TransformerTradingModel(BaseTradingModel):
    """Transformer-based trading prediction model."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.device = get_gpu_accelerator().get_optimal_device()
        self.build_model()

    def build_model(self) -> None:
        """Build Transformer model architecture."""
        from services.core_service.gpu_accelerator import create_gpu_transformer_model
        self.model = create_gpu_transformer_model(
            input_size=self.config.input_size,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            output_size=self.config.output_size
        )

    def train(self, X: np.ndarray, y: np.ndarray,
             validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, List[float]]:
        """Train Transformer model."""
        from services.core_service.gpu_accelerator import train_gpu_model

        # Create data loaders
        train_loader, val_loader = self._create_data_loaders(X, y, validation_data)

        # Train model (Transformers typically need lower learning rate)
        results = train_gpu_model(
            model=self.model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=self.config.epochs,
            learning_rate=self.config.learning_rate * 0.1  # Lower LR for transformers
        )

        self.is_trained = True
        return results

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with Transformer model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

            # Handle different input shapes
            if X_tensor.dim() == 2:
                # Single timestep: (batch_size, input_size) -> (batch_size, 1, input_size)
                X_tensor = X_tensor.unsqueeze(1)
            elif X_tensor.dim() == 3:
                # Already in sequence format: (batch_size, seq_len, input_size)
                pass
            else:
                raise ValueError(f"Input tensor must be 2D or 3D, got {X_tensor.dim()}D")

            predictions = self.model(X_tensor)
            return predictions.cpu().numpy()

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> ModelPerformance:
        """Evaluate Transformer model performance."""
        predictions = self.predict(X)

        # Calculate metrics (same as LSTM)
        mse = mean_squared_error(y, predictions)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)

        if self.config.output_size == 1:
            self.performance.mse = mse
            self.performance.mae = mae
            self.performance.r2_score = r2
        else:
            y_pred_classes = np.argmax(predictions, axis=1)
            y_true_classes = np.argmax(y, axis=1) if y.ndim > 1 else y

            self.performance.accuracy = accuracy_score(y_true_classes, y_pred_classes)
            self.performance.precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
            self.performance.recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
            self.performance.f1_score = f1_score(y_true_classes, y_pred_classes, average='weighted')

        return self.performance

    def _create_data_loaders(self, X: np.ndarray, y: np.ndarray,
                           validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """Create PyTorch data loaders for Transformer."""
        # Same as LSTM but with different batch size for transformers
        batch_size = min(self.config.batch_size, 16)  # Transformers need smaller batches

        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        val_loader = None
        if validation_data is not None:
            val_X, val_y = validation_data
            val_dataset = torch.utils.data.TensorDataset(
                torch.tensor(val_X, dtype=torch.float32),
                torch.tensor(val_y, dtype=torch.float32)
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False
            )

        return train_loader, val_loader

class EnsembleTradingModel(BaseTradingModel):
    """Ensemble model combining multiple algorithms."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.models = {}
        self.build_model()

    def build_model(self) -> None:
        """Build ensemble of models."""
        # Random Forest
        if self.config.output_size == 1:
            self.models['rf'] = RandomForestRegressor(
                n_estimators=self.config.num_trees,
                max_depth=self.config.max_depth,
                random_state=42
            )
            self.models['gb'] = GradientBoostingRegressor(
                n_estimators=self.config.num_trees,
                max_depth=self.config.max_depth,
                random_state=42
            )
        else:
            self.models['rf'] = RandomForestClassifier(
                n_estimators=self.config.num_trees,
                max_depth=self.config.max_depth,
                random_state=42
            )
            self.models['gb'] = GradientBoostingClassifier(
                n_estimators=self.config.num_trees,
                max_depth=self.config.max_depth,
                random_state=42
            )

        # XGBoost
        if self.config.output_size == 1:
            self.models['xgb'] = xgb.XGBRegressor(
                n_estimators=self.config.num_trees,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                random_state=42
            )
        else:
            self.models['xgb'] = xgb.XGBClassifier(
                n_estimators=self.config.num_trees,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                random_state=42
            )

        # LightGBM
        try:
            if self.config.output_size == 1:
                self.models['lgb'] = lgb.LGBMRegressor(
                    n_estimators=self.config.num_trees,
                    max_depth=self.config.max_depth,
                    learning_rate=self.config.learning_rate,
                    random_state=42,
                    verbose=-1
                )
            else:
                self.models['lgb'] = lgb.LGBMClassifier(
                    n_estimators=self.config.num_trees,
                    max_depth=self.config.max_depth,
                    learning_rate=self.config.learning_rate,
                    random_state=42,
                    verbose=-1
                )
        except ImportError:
            logger.warning("LightGBM not available, skipping")

    def train(self, X: np.ndarray, y: np.ndarray,
             validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, List[float]]:
        """Train ensemble models."""
        results = {}

        for name, model in self.models.items():
            logger.info(f"Training {name} model...")
            try:
                # Flatten target for sklearn models
                y_flat = y.flatten() if y.ndim > 1 and y.shape[1] == 1 else y

                # Use time series split for validation
                tscv = TimeSeriesSplit(n_splits=3)

                train_scores = []
                val_scores = []

                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y_flat[train_idx], y_flat[val_idx]

                    # Flatten X for sklearn
                    X_train_flat = X_train.reshape(X_train.shape[0], -1)
                    X_val_flat = X_val.reshape(X_val.shape[0], -1)

                    model.fit(X_train_flat, y_train)

                    train_pred = model.predict(X_train_flat)
                    val_pred = model.predict(X_val_flat)

                    if self.config.output_size == 1:
                        train_score = mean_squared_error(y_train, train_pred)
                        val_score = mean_squared_error(y_val, val_pred)
                    else:
                        train_score = accuracy_score(y_train, train_pred)
                        val_score = accuracy_score(y_val, val_pred)

                    train_scores.append(train_score)
                    val_scores.append(val_score)

                results[name] = {
                    'train_scores': train_scores,
                    'val_scores': val_scores
                }

            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                results[name] = {'train_scores': [], 'val_scores': []}

        self.is_trained = True
        return results

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Flatten X for sklearn models
        X_flat = X.reshape(X.shape[0], -1)

        predictions = []
        for name, model in self.models.items():
            try:
                # Check if sklearn model is fitted
                if hasattr(model, 'predict') and hasattr(model, 'fit'):
                    # For sklearn models, check if fitted
                    if not hasattr(model, 'estimators_') and not hasattr(model, 'n_features_in_'):
                        logger.warning(f"Model {name} is not fitted, skipping")
                        continue
                pred = model.predict(X_flat)
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Error predicting with {name}: {e}")
                continue

        if not predictions:
            raise ValueError("No models available for prediction")

        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)

        # Reshape for single output
        if self.config.output_size == 1:
            ensemble_pred = ensemble_pred.reshape(-1, 1)

        return ensemble_pred

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> ModelPerformance:
        """Evaluate ensemble model performance."""
        predictions = self.predict(X)

        # Calculate metrics
        mse = mean_squared_error(y, predictions)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)

        if self.config.output_size == 1:
            self.performance.mse = mse
            self.performance.mae = mae
            self.performance.r2_score = r2
        else:
            y_pred_classes = np.argmax(predictions, axis=1)
            y_true_classes = np.argmax(y, axis=1) if y.ndim > 1 else y

            self.performance.accuracy = accuracy_score(y_true_classes, y_pred_classes)
            self.performance.precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
            self.performance.recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
            self.performance.f1_score = f1_score(y_true_classes, y_pred_classes, average='weighted')

        return self.performance

class AutoMLSelector:
    """Automatic ML model selection and hyperparameter tuning."""

    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_score = -float('inf')

    def add_model(self, name: str, model: BaseTradingModel) -> None:
        """Add a model to the selection pool."""
        self.models[name] = model

    def select_best_model(self, X: np.ndarray, y: np.ndarray,
                         validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> str:
        """Select the best performing model."""
        results = {}

        for name, model in self.models.items():
            logger.info(f"Evaluating {name}...")
            try:
                # Train model
                model.train(X, y, validation_data)

                # Evaluate
                performance = model.evaluate(X, y)

                # Calculate composite score
                if model.config.output_size == 1:
                    score = performance.r2_score - performance.mse
                else:
                    score = performance.f1_score

                results[name] = {
                    'score': score,
                    'performance': performance
                }

                if score > self.best_score:
                    self.best_score = score
                    self.best_model = name

            except Exception as e:
                logger.error(f"Error evaluating {name}: {e}")
                results[name] = {'score': -float('inf'), 'performance': None}

        logger.info(f"Best model: {self.best_model} (score: {self.best_score:.4f})")
        return self.best_model

    def get_best_model(self) -> Optional[BaseTradingModel]:
        """Get the best performing model."""
        return self.models.get(self.best_model) if self.best_model else None

class TradingModelFactory:
    """Factory for creating trading models."""

    @staticmethod
    def create_model(model_type: str, config: ModelConfig) -> BaseTradingModel:
        """Create a trading model instance."""
        if model_type.lower() == 'lstm':
            return LSTMTradingModel(config)
        elif model_type.lower() == 'transformer':
            return TransformerTradingModel(config)
        elif model_type.lower() == 'ensemble':
            return EnsembleTradingModel(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def create_auto_ml_selector() -> AutoMLSelector:
        """Create an AutoML selector with default models."""
        selector = AutoMLSelector()

        # Default configurations
        base_config = ModelConfig(
            model_type='auto',
            input_size=50,
            output_size=1,
            hidden_size=64,
            num_layers=2,
            sequence_length=30
        )

        # Add different model types
        selector.add_model('lstm', LSTMTradingModel(base_config))
        selector.add_model('transformer', TransformerTradingModel(base_config))
        selector.add_model('ensemble', EnsembleTradingModel(base_config))

        return selector

# Global instances
model_factory = TradingModelFactory()

def create_trading_model(model_type: str, config: ModelConfig) -> BaseTradingModel:
    """Create a trading model."""
    return model_factory.create_model(model_type, config)

def create_auto_ml_selector() -> AutoMLSelector:
    """Create an AutoML selector."""
    return model_factory.create_auto_ml_selector()