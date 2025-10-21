#!/usr/bin/env python3
"""
ML Trainer Service - Machine learning model training and optimization.

Provides comprehensive ML training capabilities including:
- Model training with various algorithms
- Hyperparameter optimization with Optuna
- Model evaluation and validation
- Feature engineering and selection
- Model persistence and management
"""

import asyncio
import logging
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

from config.settings import (
    ML_MODELS_DIR,
    ML_FEATURES,
    ML_TARGET_HORIZON,
    ML_TRAINING_WINDOW,
    ML_VALIDATION_SPLIT,
    ML_RANDOM_STATE
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for trained ML models."""
    name: str
    symbol: str
    timeframe: str
    model_type: str
    created_at: datetime
    training_start: str
    training_end: str
    target_horizon: int
    features: List[str]
    hyperparameters: Dict[str, Any]
    performance: Dict[str, float]
    feature_importance: Dict[str, float]


@dataclass
class TrainingStatus:
    """Status of model training."""
    symbol: str
    status: str  # 'idle', 'training', 'completed', 'failed'
    progress: float
    message: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class EventSystem:
    """Simple event system for service communication."""

    def __init__(self):
        self._handlers: Dict[str, List[callable]] = {}

    def subscribe(self, event_type: str, handler: callable):
        """Subscribe to an event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    async def publish(self, event_type: str, data: Dict[str, Any]):
        """Publish an event."""
        if event_type in self._handlers:
            for handler in self._handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    logger.error(f"Event handler error: {e}")


class MLTrainerService:
    """ML model training and optimization service."""

    def __init__(self, event_system: Optional[EventSystem] = None):
        self.event_system = event_system or EventSystem()
        self.models_dir = Path(ML_MODELS_DIR)
        self.models_dir.mkdir(exist_ok=True)

        # Training status tracking
        self.training_status: Dict[str, TrainingStatus] = {}

        # Model cache
        self._model_cache: Dict[str, Any] = {}

        # Initialize scalers and selectors
        self.scalers: Dict[str, StandardScaler] = {}
        self.selectors: Dict[str, SelectKBest] = {}

        logger.info("ML Trainer Service initialized")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            "service": "ml_trainer",
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "models_count": len(await self.list_models()),
            "active_training": len([s for s in self.training_status.values() if s.status == "training"])
        }

    async def train_model(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        model_type: str = "random_forest",
        target_horizon: int = ML_TARGET_HORIZON,
        use_optuna: bool = False
    ) -> Dict[str, Any]:
        """
        Train an ML model for trading signals.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            start_date: Training start date
            end_date: Training end date
            model_type: Type of ML model
            target_horizon: Prediction horizon
            use_optuna: Whether to use Optuna for optimization

        Returns:
            Training results
        """
        model_name = f"{symbol}_{timeframe}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Update training status
        self.training_status[symbol] = TrainingStatus(
            symbol=symbol,
            status="training",
            progress=0.0,
            message="Starting training...",
            start_time=datetime.now()
        )

        try:
            # Fetch training data (placeholder - would integrate with data service)
            logger.info(f"Fetching training data for {symbol} {timeframe}")
            training_data = await self._fetch_training_data(symbol, timeframe, start_date, end_date)

            if training_data.empty:
                raise ValueError("No training data available")

            self.training_status[symbol].progress = 0.2
            self.training_status[symbol].message = "Data loaded, preparing features..."

            # Prepare features and target
            X, y, feature_names = self._prepare_features(training_data, target_horizon)

            if len(X) == 0:
                raise ValueError("Insufficient data for training")

            self.training_status[symbol].progress = 0.4
            self.training_status[symbol].message = "Features prepared, training model..."

            # Train model
            if use_optuna:
                model, best_params = await self._train_with_optuna(X, y, model_type)
            else:
                model = self._train_model(X, y, model_type)
                best_params = self._get_default_params(model_type)

            self.training_status[symbol].progress = 0.7
            self.training_status[symbol].message = "Model trained, evaluating..."

            # Evaluate model
            performance = self._evaluate_model(model, X, y)

            # Get feature importance
            feature_importance = self._get_feature_importance(model, feature_names)

            self.training_status[symbol].progress = 0.9
            self.training_status[symbol].message = "Saving model..."

            # Save model and metadata
            metadata = ModelMetadata(
                name=model_name,
                symbol=symbol,
                timeframe=timeframe,
                model_type=model_type,
                created_at=datetime.now(),
                training_start=start_date,
                training_end=end_date,
                target_horizon=target_horizon,
                features=feature_names,
                hyperparameters=best_params,
                performance=performance,
                feature_importance=feature_importance
            )

            await self._save_model(model_name, model, metadata)

            # Update status
            self.training_status[symbol].status = "completed"
            self.training_status[symbol].progress = 1.0
            self.training_status[symbol].message = "Training completed"
            self.training_status[symbol].end_time = datetime.now()

            # Publish event
            await self.event_system.publish("ml.training_completed", {
                "model_name": model_name,
                "symbol": symbol,
                "performance": performance
            })

            logger.info(f"Model training completed: {model_name}")

            return {
                "success": True,
                "model_name": model_name,
                "performance": performance,
                "feature_importance": feature_importance
            }

        except Exception as e:
            # Update status on failure
            self.training_status[symbol].status = "failed"
            self.training_status[symbol].message = str(e)
            self.training_status[symbol].end_time = datetime.now()

            logger.error(f"Model training failed: {e}")
            raise

    async def retrain_model(
        self,
        model_name: str,
        new_data_start: str,
        new_data_end: str
    ) -> Dict[str, Any]:
        """
        Retrain an existing model with new data.

        Args:
            model_name: Name of the model to retrain
            new_data_start: Start date for new data
            new_data_end: End date for new data

        Returns:
            Retraining results
        """
        # Load existing model and metadata
        model, metadata = await self._load_model(model_name)

        # Fetch new data
        new_data = await self._fetch_training_data(
            metadata.symbol,
            metadata.timeframe,
            new_data_start,
            new_data_end
        )

        # Combine with existing training data
        combined_data = await self._fetch_training_data(
            metadata.symbol,
            metadata.timeframe,
            metadata.training_start,
            new_data_end
        )

        # Prepare features
        X, y, _ = self._prepare_features(combined_data, metadata.target_horizon)

        # Retrain model
        model = self._train_model(X, y, metadata.model_type)

        # Evaluate
        performance = self._evaluate_model(model, X, y)

        # Update metadata
        metadata.performance = performance
        metadata.training_end = new_data_end

        # Save updated model
        await self._save_model(model_name, model, metadata)

        return {
            "success": True,
            "model_name": model_name,
            "performance": performance
        }

    async def evaluate_model(
        self,
        model_name: str,
        test_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Evaluate a trained model on test data.

        Args:
            model_name: Name of the model to evaluate
            test_data: Test data

        Returns:
            Performance metrics
        """
        model, metadata = await self._load_model(model_name)

        # Prepare test features
        X_test, y_test, _ = self._prepare_features(test_data, metadata.target_horizon)

        # Evaluate
        performance = self._evaluate_model(model, X_test, y_test)

        # Publish event
        await self.event_system.publish("ml.model_evaluated", {
            "model_name": model_name,
            "performance": performance
        })

        return performance

    async def list_models(self) -> List[str]:
        """List all trained models."""
        model_files = list(self.models_dir.glob("*.pkl"))
        return [f.stem for f in model_files]

    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        _, metadata = await self._load_model(model_name)
        return asdict(metadata)

    async def delete_model(self, model_name: str) -> bool:
        """Delete a trained model."""
        model_path = self.models_dir / f"{model_name}.pkl"
        metadata_path = self.models_dir / f"{model_name}_metadata.pkl"

        success = True
        if model_path.exists():
            model_path.unlink()
        else:
            success = False

        if metadata_path.exists():
            metadata_path.unlink()

        # Clear from cache
        if model_name in self._model_cache:
            del self._model_cache[model_name]

        return success

    async def get_feature_importance(self, symbol: str, timeframe: str) -> Dict[str, float]:
        """Get feature importance for recent models."""
        models = await self.list_models()
        symbol_models = [m for m in models if m.startswith(f"{symbol}_{timeframe}")]

        if not symbol_models:
            return {}

        # Get latest model
        latest_model = max(symbol_models)
        _, metadata = await self._load_model(latest_model)

        return metadata.feature_importance

    def get_hyperparameter_ranges(self, model_type: str) -> Dict[str, Any]:
        """Get hyperparameter ranges for a model type."""
        ranges = {
            "random_forest": {
                "n_estimators": {"type": "int", "low": 50, "high": 500},
                "max_depth": {"type": "int", "low": 5, "high": 50},
                "min_samples_split": {"type": "int", "low": 2, "high": 20},
                "min_samples_leaf": {"type": "int", "low": 1, "high": 10}
            },
            "gradient_boosting": {
                "n_estimators": {"type": "int", "low": 50, "high": 300},
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.3},
                "max_depth": {"type": "int", "low": 3, "high": 10},
                "subsample": {"type": "float", "low": 0.5, "high": 1.0}
            },
            "logistic_regression": {
                "C": {"type": "float", "low": 0.01, "high": 10.0},
                "penalty": {"type": "categorical", "choices": ["l1", "l2", "elasticnet"]}
            },
            "svm": {
                "C": {"type": "float", "low": 0.1, "high": 100.0},
                "gamma": {"type": "float", "low": 0.001, "high": 1.0},
                "kernel": {"type": "categorical", "choices": ["rbf", "linear", "poly"]}
            }
        }

        return ranges.get(model_type, {})

    async def get_training_status(self, symbol: str) -> Dict[str, Any]:
        """Get training status for a symbol."""
        status = self.training_status.get(symbol)
        if status:
            return asdict(status)
        return {"status": "idle", "message": "No training in progress"}

    async def _fetch_training_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Fetch training data (placeholder - integrate with data service)."""
        # This would normally call the data service
        # For now, return sample data structure
        logger.warning("Using placeholder training data - integrate with data service")

        # Generate sample OHLCV data
        dates = pd.date_range(start=start_date, end=end_date, freq='1H')
        np.random.seed(42)

        data = {
            'timestamp': dates,
            'open': 50000 + np.random.normal(0, 1000, len(dates)),
            'high': 50000 + np.random.normal(0, 1000, len(dates)) + 100,
            'low': 50000 + np.random.normal(0, 1000, len(dates)) - 100,
            'close': 50000 + np.random.normal(0, 1000, len(dates)),
            'volume': np.random.normal(100, 20, len(dates))
        }

        df = pd.DataFrame(data)
        df['close'] = df['close'].clip(lower=0)  # Ensure positive prices

        return df

    def _prepare_features(self, data: pd.DataFrame, target_horizon: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features and target for ML training."""
        # Calculate technical indicators
        df = data.copy()

        # Simple moving averages
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()

        # Bollinger Bands
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        df['bb_upper'] = sma_20 + (std_20 * 2)
        df['bb_lower'] = sma_20 - (std_20 * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma_20

        # ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift(1)).abs()
        low_close = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()

        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        # Price momentum
        df['returns'] = df['close'].pct_change()
        df['momentum'] = df['close'] / df['close'].shift(10) - 1

        # Target: future price movement
        df['future_return'] = df['close'].shift(-target_horizon) / df['close'] - 1
        df['target'] = (df['future_return'] > 0).astype(int)

        # Select features
        feature_cols = [
            'sma_20', 'sma_50', 'rsi', 'macd', 'macd_signal',
            'bb_upper', 'bb_lower', 'bb_width', 'atr',
            'volume_ratio', 'returns', 'momentum'
        ]

        # Drop NaN values
        df_clean = df.dropna(subset=feature_cols + ['target'])

        X = df_clean[feature_cols].values
        y = df_clean['target'].values

        return X, y, feature_cols

    def _train_model(self, X: np.ndarray, y: np.ndarray, model_type: str) -> Any:
        """Train a model with default parameters."""
        if model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=ML_RANDOM_STATE
            )
        elif model_type == "gradient_boosting":
            model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=ML_RANDOM_STATE
            )
        elif model_type == "logistic_regression":
            model = LogisticRegression(random_state=ML_RANDOM_STATE)
        elif model_type == "svm":
            model = SVC(probability=True, random_state=ML_RANDOM_STATE)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.fit(X, y)
        return model

    def _get_default_params(self, model_type: str) -> Dict[str, Any]:
        """Get default hyperparameters for a model type."""
        defaults = {
            "random_forest": {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 2,
                "min_samples_leaf": 1
            },
            "gradient_boosting": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 5,
                "subsample": 1.0
            },
            "logistic_regression": {
                "C": 1.0,
                "penalty": "l2"
            },
            "svm": {
                "C": 1.0,
                "gamma": "scale",
                "kernel": "rbf"
            }
        }

        return defaults.get(model_type, {})

    async def _train_with_optuna(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str
    ) -> Tuple[Any, Dict[str, Any]]:
        """Train a model with Optuna hyperparameter optimization."""

        def objective(trial):
            if model_type == "random_forest":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "max_depth": trial.suggest_int("max_depth", 5, 50),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10)
                }
                model = RandomForestClassifier(**params, random_state=ML_RANDOM_STATE)

            elif model_type == "gradient_boosting":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0)
                }
                model = GradientBoostingClassifier(**params, random_state=ML_RANDOM_STATE)

            else:
                # For other models, use default training
                model = self._train_model(X, y, model_type)
                return 0.5  # Dummy score

            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []

            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                scores.append(f1_score(y_val, y_pred))

            return np.mean(scores)

        # Run optimization
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50)

        # Train final model with best parameters
        best_params = study.best_params
        model = self._train_model(X, y, model_type)

        # Apply best parameters if applicable
        if hasattr(model, 'set_params'):
            model.set_params(**best_params)

        model.fit(X, y)

        return model, best_params

    def _evaluate_model(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        y_pred = model.predict(X)

        # Get prediction probabilities if available
        y_prob = None
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1_score": f1_score(y, y_pred, zero_division=0)
        }

        # Only calculate ROC AUC if we have both classes present
        if y_prob is not None and len(np.unique(y)) > 1:
            try:
                metrics["roc_auc"] = roc_auc_score(y, y_prob)
            except ValueError:
                # ROC AUC not defined for single class
                metrics["roc_auc"] = 0.5

        return metrics

    def _get_feature_importance(self, model: Any, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from model."""
        importance_dict = {}

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            for name, importance in zip(feature_names, importances):
                importance_dict[name] = float(importance)
        else:
            # For models without feature_importances_, assign equal weights
            for name in feature_names:
                importance_dict[name] = 1.0 / len(feature_names)

        return importance_dict

    async def _save_model(self, model_name: str, model: Any, metadata: ModelMetadata):
        """Save model and metadata to disk."""
        model_path = self.models_dir / f"{model_name}.pkl"
        metadata_path = self.models_dir / f"{model_name}_metadata.pkl"

        # Ensure directory exists
        model_path.parent.mkdir(parents=True, exist_ok=True)

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        # Cache the model
        self._model_cache[model_name] = (model, metadata)

    async def _load_model(self, model_name: str) -> Tuple[Any, ModelMetadata]:
        """Load model and metadata from disk or cache."""
        if model_name in self._model_cache:
            return self._model_cache[model_name]

        model_path = self.models_dir / f"{model_name}.pkl"
        metadata_path = self.models_dir / f"{model_name}_metadata.pkl"

        if not model_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Model {model_name} not found")

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        # Cache the model
        self._model_cache[model_name] = (model, metadata)

        return model, metadata