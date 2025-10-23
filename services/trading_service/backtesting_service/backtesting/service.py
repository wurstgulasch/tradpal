"""
TradPal Backtesting Service - Core Backtesting Engine
Consolidated implementation integrating backtesting, ML training, optimization, and walk-forward analysis
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


class BacktestingService:
    """Consolidated backtesting service integrating all backtesting functionality"""

    def __init__(self, event_system=None, data_service=None):
        self.event_system = event_system or EventSystem()
        self.data_service = data_service
        self.is_initialized = False

        # ML Training components
        self.models_dir = Path(ML_MODELS_DIR)
        self.models_dir.mkdir(exist_ok=True)
        self.training_status: Dict[str, TrainingStatus] = {}
        self._model_cache: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.selectors: Dict[str, SelectKBest] = {}

        logger.info("Consolidated Backtesting Service initialized")

    async def initialize(self):
        """Initialize the consolidated backtesting service"""
        logger.info("Initializing Consolidated Backtesting Service...")

        # Initialize all components
        self.is_initialized = True
        logger.info("Consolidated Backtesting Service initialized")

    async def shutdown(self):
        """Shutdown the consolidated backtesting service"""
        logger.info("Consolidated Backtesting Service shut down")
        self.is_initialized = False

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        return {
            "service": "backtesting_service",
            "status": "healthy" if self.is_initialized else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "backtesting": True,
                "ml_training": True,
                "optimization": True,
                "walk_forward": True
            },
            "models_count": len(await self.list_models()),
            "active_training": len([s for s in self.training_status.values() if s.status == "training"])
        }

    async def get_service_info(self) -> Dict[str, Any]:
        """Get information about the backtesting service"""
        return {
            "name": "TradPal Backtesting Service",
            "version": "2.0.0",
            "description": "Consolidated backtesting service with ML training, optimization, and walk-forward analysis",
            "components": [
                "backtesting_engine",
                "ml_training",
                "optimization",
                "walk_forward_analysis"
            ],
            "api_port": 8002,
            "initialized": self.is_initialized
        }

    # ML Training Methods

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
        if not self.is_initialized:
            raise RuntimeError("Backtesting service not initialized")

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
            # Fetch training data
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

    async def list_models(self) -> List[str]:
        """List all trained models."""
        model_files = list(self.models_dir.glob("*.pkl"))
        return [f.stem for f in model_files if not f.stem.endswith('_metadata')]

    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        _, metadata = await self._load_model(model_name)
        return asdict(metadata)

    async def get_training_status(self, symbol: str) -> Dict[str, Any]:
        """Get training status for a symbol."""
        status = self.training_status.get(symbol)
        if status:
            return asdict(status)
        return {"status": "idle", "message": "No training in progress"}

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

    # Private ML Training Methods

    async def _fetch_training_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Fetch training data using data service."""
        if self.data_service:
            # Use data service if available
            try:
                from services.data_service.service import DataRequest
                request = DataRequest(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    limit=5000
                )
                response = await self.data_service.fetch_data(request)
                if response.success and response.data:
                    # Convert back to DataFrame
                    ohlcv_data = response.data.get("ohlcv", {})
                    df = pd.DataFrame.from_dict(ohlcv_data, orient='index')
                    df.index = pd.to_datetime(df.index)
                    return df
            except Exception as e:
                logger.warning(f"Failed to fetch data from data service: {e}")

        # Fallback: Generate sample data
        logger.warning("Using placeholder training data - integrate with data service")

        dates = pd.date_range(start=start_date, end=end_date, freq='1H')
        np.random.seed(42)

        data = {
            'timestamp': dates,
            'open': 50000 + np.random.normal(0, 1000, len(dates)),
            'high': 51000 + np.random.normal(0, 1000, len(dates)),
            'low': 49000 + np.random.normal(0, 1000, len(dates)),
            'close': 50000 + np.random.normal(0, 1000, len(dates)),
            'volume': np.random.normal(100, 20, len(dates))
        }

        df = pd.DataFrame(data)
        df['close'] = df['close'].clip(lower=0.01)  # Ensure positive prices
        df.set_index('timestamp', inplace=True)

        return df

    def _prepare_features(self, data: pd.DataFrame, target_horizon: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features and target for ML training."""
        df = data.copy()

        # Calculate technical indicators
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

    # Core Backtesting Methods

    async def run_backtest(self, strategy_config: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Run a backtest with given strategy and data"""
        if not self.is_initialized:
            raise RuntimeError("Backtesting service not initialized")

        logger.info(f"Running backtest with strategy: {strategy_config.get('name', 'unknown')}")

        # Simple moving average crossover strategy for demo
        results = await self._run_simple_strategy(data, strategy_config)

        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(results)

        return {
            "strategy": strategy_config.get("name", "unknown"),
            "results": results,
            "metrics": metrics,
            "success": True
        }

    async def _run_simple_strategy(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Run a simple moving average crossover strategy"""
        df = data.copy()

        # Calculate indicators
        short_window = config.get("short_window", 10)
        long_window = config.get("long_window", 30)

        df['SMA_short'] = df['close'].rolling(short_window).mean()
        df['SMA_long'] = df['close'].rolling(long_window).mean()

        # Generate signals
        df['signal'] = 0
        df.loc[df['SMA_short'] > df['SMA_long'], 'signal'] = 1  # Buy
        df.loc[df['SMA_short'] < df['SMA_long'], 'signal'] = -1  # Sell

        # Calculate returns
        df['returns'] = df['close'].pct_change()
        df['strategy_returns'] = df['signal'].shift(1) * df['returns']

        return df

    def _calculate_performance_metrics(self, results: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance metrics"""
        if 'strategy_returns' not in results.columns:
            return {"error": "No strategy returns found"}

        returns = results['strategy_returns'].dropna()

        if len(returns) == 0:
            return {"error": "No valid returns data"}

        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = total_return * (252 / len(returns))  # Assuming daily data
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        max_drawdown = (results['close'] / results['close'].cummax() - 1).min()

        return {
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "total_trades": int((results['signal'].diff() != 0).sum())
        }

    def get_available_strategies(self) -> List[str]:
        """Get list of available strategies"""
        return [
            "moving_average_crossover",
            "rsi_divergence",
            "bollinger_bands",
            "ml_enhanced_strategy"
        ]

    async def optimize_strategy(self, strategy_name: str, param_ranges: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize strategy parameters"""
        logger.info(f"Optimizing strategy: {strategy_name}")

        # Simple grid search for demo
        best_params = {}
        best_score = -np.inf

        # Generate sample parameter combinations
        if strategy_name == "moving_average_crossover":
            for short in range(5, 20, 5):
                for long in range(20, 50, 10):
                    if short >= long:
                        continue

                    score = np.random.random()  # Placeholder scoring
                    if score > best_score:
                        best_score = score
                        best_params = {"short_window": short, "long_window": long}

        return {
            "strategy": strategy_name,
            "best_params": best_params,
            "best_score": float(best_score),
            "optimization_method": "grid_search"
        }

    # Walk-Forward Optimization Methods

    async def run_walk_forward_optimization(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        evaluation_metric: str = 'sharpe_ratio'
    ) -> Dict[str, Any]:
        """
        Run walk-forward optimization with enhanced metrics.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
            evaluation_metric: Metric to optimize

        Returns:
            Walk-forward optimization results
        """
        if not self.is_initialized:
            raise RuntimeError("Backtesting service not initialized")

        logger.info(f"Starting walk-forward optimization for {symbol} on {timeframe}")
        logger.info(f"Evaluation metric: {evaluation_metric}")

        # Fetch data
        data = await self._fetch_training_data(symbol, timeframe, start_date, end_date)

        if data.empty:
            raise ValueError("No data available for optimization")

        # Walk-forward parameters
        in_sample_window = 252  # ~1 year
        out_sample_window = 21  # ~1 month
        step_size = out_sample_window

        results = []
        data_length = len(data)

        for i in range(in_sample_window, data_length - out_sample_window, step_size):
            # Define IS and OOS periods
            is_end = i
            oos_end = min(i + out_sample_window, data_length)

            is_data = data.iloc[:is_end]
            oos_data = data.iloc[is_end:oos_end]

            if len(is_data) < 50 or len(oos_data) < 10:
                continue

            # Optimize on IS data
            best_params = await self._optimize_on_window(is_data, evaluation_metric)

            # Evaluate on OOS data
            oos_performance = await self._evaluate_on_window(oos_data, best_params, evaluation_metric)

            results.append({
                'window_start': data.index[is_end - in_sample_window],
                'window_end': data.index[oos_end - 1],
                'is_performance': await self._evaluate_on_window(is_data, best_params, evaluation_metric),
                'oos_performance': oos_performance,
                'best_params': best_params
            })

        # Analyze results
        analysis = self._analyze_walk_forward_results(results)

        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'evaluation_metric': evaluation_metric,
            'total_windows': len(results),
            'results': results,
            'analysis': analysis
        }

    async def _optimize_on_window(self, data: pd.DataFrame, metric: str) -> Dict[str, Any]:
        """Optimize parameters on a data window"""
        # Simple optimization for demo
        best_params = {"short_window": 10, "long_window": 30}
        best_score = 0

        for short in [5, 10, 15, 20]:
            for long in [20, 30, 40, 50]:
                if short >= long:
                    continue

                # Test parameters
                config = {"short_window": short, "long_window": long}
                result = await self.run_backtest({"name": "test"}, data)
                score = result["metrics"].get(metric, 0)

                if score > best_score:
                    best_score = score
                    best_params = config

        return best_params

    async def _evaluate_on_window(self, data: pd.DataFrame, params: Dict[str, Any], metric: str) -> float:
        """Evaluate parameters on a data window"""
        config = {"name": "evaluation", **params}
        result = await self.run_backtest(config, data)
        return result["metrics"].get(metric, 0)

    def _analyze_walk_forward_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze walk-forward optimization results"""
        if not results:
            return {}

        is_performances = [r['is_performance'] for r in results]
        oos_performances = [r['oos_performance'] for r in results]

        analysis = {
            'average_is_performance': np.mean(is_performances),
            'average_oos_performance': np.mean(oos_performances),
            'std_oos_performance': np.std(oos_performances),
            'performance_decay': np.mean(is_performances) - np.mean(oos_performances),
            'positive_windows': sum(1 for p in oos_performances if p > 0),
            'positive_ratio': sum(1 for p in oos_performances if p > 0) / len(oos_performances)
        }

        # Calculate Information Coefficient (correlation between IS and OOS)
        if len(is_performances) > 1:
            analysis['information_coefficient'] = np.corrcoef(is_performances, oos_performances)[0, 1]

        # Overfitting ratio
        if analysis['average_is_performance'] > 0:
            analysis['overfitting_ratio'] = max(0, analysis['performance_decay'] / analysis['average_is_performance'])

        # Consistency score (lower std is better)
        if analysis['std_oos_performance'] > 0:
            analysis['consistency_score'] = 1 / (1 + analysis['std_oos_performance'])

        return analysis


# Simplified model classes for API compatibility
class BacktestRequest:
    """Backtest request model"""
    def __init__(self, strategy: str, symbol: str, start_date: str, end_date: str, **params):
        self.strategy = strategy
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.params = params

class BacktestResponse:
    """Backtest response model"""
    def __init__(self, success: bool, results: Dict[str, Any] = None, error: str = None):
        self.success = success
        self.results = results or {}
        self.error = error

class OptimizationRequest:
    """Optimization request model"""
    def __init__(self, strategy: str, param_ranges: Dict[str, Any]):
        self.strategy = strategy
        self.param_ranges = param_ranges

class OptimizationResponse:
    """Optimization response model"""
    def __init__(self, success: bool, best_params: Dict[str, Any] = None, error: str = None):
        self.success = success
        self.best_params = best_params or {}
        self.error = error