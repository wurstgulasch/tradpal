#!/usr/bin/env python3
"""
ML Training Service for TradPal

Advanced ML model training and optimization with ensemble methods.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
from scipy.optimize import minimize

from config.core_settings import ML_RANDOM_STATE
from services.data_service.data_service.alternative_data.client import AlternativeDataService

logger = logging.getLogger(__name__)


class MLTrainerService:
    """ML model training and optimization service."""

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or Path("cache/ml_models")
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Initialize alternative data service
        self.alternative_data_service = AlternativeDataService()

        # Initialize ensemble trainer
        self.ensemble_trainer = EnsembleTrainer()

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            "status": "healthy",
            "service": "ml_trainer",
            "models_dir": str(self.models_dir),
            "models_count": len(list(self.models_dir.glob("*.pkl")))
        }

    async def list_models(self) -> List[str]:
        """List all trained models."""
        return [f.stem for f in self.models_dir.glob("*.pkl")]

    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        metadata_file = self.models_dir / f"{model_name}_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        return {"error": "Model not found"}

    async def delete_model(self, model_name: str) -> bool:
        """Delete a trained model."""
        model_file = self.models_dir / f"{model_name}.pkl"
        metadata_file = self.models_dir / f"{model_name}_metadata.json"

        deleted = False
        if model_file.exists():
            model_file.unlink()
            deleted = True
        if metadata_file.exists():
            metadata_file.unlink()
            deleted = True

        return deleted

    async def get_recent_feature_importance(self, limit: int = 5) -> Dict[str, Any]:
        """Get feature importance for recent models."""
        models = sorted(self.models_dir.glob("*_metadata.json"),
                       key=lambda x: x.stat().st_mtime, reverse=True)[:limit]

        importance_data = {}
        for metadata_file in models:
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                model_name = metadata_file.stem.replace('_metadata', '')
                if 'feature_importance' in metadata:
                    importance_data[model_name] = metadata['feature_importance']
            except Exception as e:
                logger.warning(f"Failed to load metadata for {metadata_file}: {e}")

        return importance_data

    async def get_hyperparameter_ranges(self, model_type: str) -> Dict[str, Any]:
        """Get hyperparameter ranges for a model type."""
        ranges = {
            "random_forest": {
                "n_estimators": [50, 200],
                "max_depth": [5, 20],
                "min_samples_split": [2, 10],
                "min_samples_leaf": [1, 5]
            },
            "gradient_boosting": {
                "n_estimators": [50, 200],
                "learning_rate": [0.01, 0.2],
                "max_depth": [3, 10],
                "subsample": [0.6, 1.0]
            },
            "xgboost": {
                "n_estimators": [50, 200],
                "learning_rate": [0.01, 0.2],
                "max_depth": [3, 10],
                "subsample": [0.6, 1.0],
                "colsample_bytree": [0.6, 1.0]
            }
        }
        return ranges.get(model_type, {})

    async def get_training_status(self, symbol: str) -> Dict[str, Any]:
        """Get training status for a symbol."""
        # Placeholder - implement actual status tracking
        return {
            "symbol": symbol,
            "status": "not_training",
            "last_trained": None,
            "performance": {}
        }

    async def train_model(self, symbol: str, model_type: str = "random_forest",
                         start_date: str = "2020-01-01", end_date: str = "2024-01-01") -> Dict[str, Any]:
        """Train a model with default parameters."""
        try:
            # Fetch training data
            df = await self._fetch_training_data(symbol, "1d", start_date, end_date)
            if df.empty:
                return {"error": "No training data available"}

            # Prepare features
            X, y = await self._prepare_features(df)

            # Get default hyperparameters
            params = await self._get_default_params(model_type)

            # Train model
            model, performance = await self._train_model(X, y, model_type, params)

            # Save model
            model_name = f"{symbol}_{model_type}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            await self._save_model(model, model_name, {
                "symbol": symbol,
                "model_type": model_type,
                "performance": performance,
                "features": list(df.columns),
                "training_date": pd.Timestamp.now().isoformat()
            })

            return {
                "model_name": model_name,
                "performance": performance,
                "status": "trained"
            }

        except Exception as e:
            logger.error(f"Failed to train model: {e}")
            return {"error": str(e)}

    async def _fetch_training_data(self, symbol: str, timeframe: str,
                                  start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch training data (placeholder - integrate with data service)."""
        # Placeholder implementation
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        np.random.seed(42)
        n = len(dates)

        df = pd.DataFrame({
            'open': 100 + np.random.randn(n).cumsum(),
            'high': 105 + np.random.randn(n).cumsum(),
            'low': 95 + np.random.randn(n).cumsum(),
            'close': 100 + np.random.randn(n).cumsum(),
            'volume': np.random.randint(1000, 10000, n)
        }, index=dates)

        return df

    async def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target for ML training."""
        # Simple feature engineering
        df = df.copy()
        df['returns'] = df['close'].pct_change()
        df['target'] = (df['returns'].shift(-1) > 0).astype(int)

        # Basic technical indicators
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['rsi'] = self._calculate_rsi(df['close'])
        df['macd'], df['macd_signal'] = self._calculate_macd(df['close'])

        # Drop NaN values
        df = df.dropna()

        # Features and target
        feature_cols = ['sma_20', 'sma_50', 'rsi', 'macd', 'macd_signal', 'volume']
        X = df[feature_cols].values
        y = df['target'].values

        return X, y

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26,
                        signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal

    async def _get_default_params(self, model_type: str) -> Dict[str, Any]:
        """Get default hyperparameters for a model type."""
        defaults = {
            "random_forest": {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": ML_RANDOM_STATE
            },
            "gradient_boosting": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 5,
                "random_state": ML_RANDOM_STATE
            },
            "xgboost": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
                "random_state": ML_RANDOM_STATE
            }
        }
        return defaults.get(model_type, {})

    async def _train_model(self, X: np.ndarray, y: np.ndarray, model_type: str,
                          params: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
        """Train a model and return performance."""
        if model_type == "random_forest":
            model = RandomForestClassifier(**params)
        elif model_type == "gradient_boosting":
            model = GradientBoostingClassifier(**params)
        elif model_type == "xgboost":
            model = xgb.XGBClassifier(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.fit(X, y)

        # Evaluate
        y_pred = model.predict(X)
        performance = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1_score": f1_score(y, y_pred, zero_division=0)
        }

        return model, performance

    async def _save_model(self, model: Any, model_name: str, metadata: Dict[str, Any]):
        """Save model and metadata to disk."""
        import joblib

        # Save model
        model_file = self.models_dir / f"{model_name}.pkl"
        joblib.dump(model, model_file)

        # Save metadata
        metadata_file = self.models_dir / f"{model_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    async def _load_model(self, model_name: str) -> Tuple[Any, Dict[str, Any]]:
        """Load model and metadata from disk or cache."""
        import joblib

        model_file = self.models_dir / f"{model_name}.pkl"
        metadata_file = self.models_dir / f"{model_name}_metadata.json"

        if not model_file.exists():
            raise FileNotFoundError(f"Model {model_name} not found")

        model = joblib.load(model_file)

        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

        return model, metadata

    async def train_ensemble(self, symbol: str, start_date: str = "2020-01-01",
                           end_date: str = "2024-01-01") -> Dict[str, Any]:
        """Train an ensemble model for benchmark outperformance."""
        try:
            # Fetch training data
            df = await self._fetch_training_data(symbol, "1d", start_date, end_date)
            if df.empty:
                return {"error": "No training data available"}

            # Prepare enhanced features
            X, y = await self._prepare_enhanced_features(df, symbol)

            # Train ensemble
            result = self.ensemble_trainer.train_ensemble(X, y)

            # Save ensemble model
            model_name = f"{symbol}_ensemble_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            await self._save_model(result['model'], model_name, {
                "symbol": symbol,
                "model_type": "ensemble",
                "performance": result['performance'],
                "base_models": list(result.get('all_models', {}).keys()),
                "ensemble_type": result.get('ensemble_type', 'unknown'),
                "training_date": pd.Timestamp.now().isoformat()
            })

            return {
                "model_name": model_name,
                "performance": result['performance'],
                "ensemble_type": result.get('ensemble_type', 'unknown'),
                "base_models": list(result.get('all_models', {}).keys()),
                "status": "trained"
            }

        except Exception as e:
            logger.error(f"Failed to train ensemble: {e}")
            return {"error": str(e)}

    async def _prepare_enhanced_features(self, df: pd.DataFrame, symbol: str = "BTCUSDT") -> Tuple[np.ndarray, np.ndarray]:
        """Prepare enhanced features for ML training with advanced indicators."""
        df = df.copy()

        # Add advanced technical indicators
        df = self._add_advanced_technical_indicators(df)

        # Add market regime indicators
        df = self._add_market_regime_indicators(df)

        # Add momentum and volatility indicators
        df = self._add_momentum_volatility_indicators(df)

        # Integrate alternative data
        df = await self._integrate_alternative_data(df, symbol)

        # Create target
        df['returns'] = df['close'].pct_change()
        df['target'] = (df['returns'].shift(-1) > 0).astype(int)

        # Drop NaN values
        df = df.dropna()

        # Features (exclude target and raw OHLCV)
        exclude_cols = ['open', 'high', 'low', 'close', 'target', 'returns']
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        X = df[feature_cols].values
        y = df['target'].values

        return X, y

    def _add_advanced_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced technical indicators for enhanced feature engineering."""
        df = df.copy()

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_upper'] = df['bb_middle'] + 2 * df['close'].rolling(20).std()
        df['bb_lower'] = df['bb_middle'] - 2 * df['close'].rolling(20).std()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        # Stochastic Oscillator
        df['stoch_k'], df['stoch_d'] = self._calculate_stochastic(df)

        # Williams %R
        df['williams_r'] = self._calculate_williams_r(df)

        # Commodity Channel Index
        df['cci'] = self._calculate_cci(df)

        # Average Directional Index
        df['adx'] = self._calculate_adx(df)

        # Chaikin Money Flow
        df['cmf'] = self._calculate_cmf(df)

        return df

    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14,
                            d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        low_min = df['low'].rolling(k_period).min()
        high_max = df['high'].rolling(k_period).max()
        k = 100 * (df['close'] - low_min) / (high_max - low_min)
        d = k.rolling(d_period).mean()
        return k, d

    def _calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R."""
        high_max = df['high'].rolling(period).max()
        low_min = df['low'].rolling(period).min()
        return -100 * (high_max - df['close']) / (high_max - low_min)

    def _calculate_cci(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma = typical_price.rolling(period).mean()
        mad = (typical_price - sma).abs().rolling(period).mean()
        return (typical_price - sma) / (0.015 * mad)

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index."""
        high_diff = df['high'].diff()
        low_diff = df['low'].diff()

        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

        tr = np.maximum(df['high'] - df['low'],
                       np.maximum(abs(df['high'] - df['close'].shift(1)),
                                abs(df['low'] - df['close'].shift(1))))

        atr = pd.Series(tr).rolling(period).mean()

        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()

        return adx

    def _calculate_cmf(self, df: pd.DataFrame, period: int = 21) -> pd.Series:
        """Calculate Chaikin Money Flow."""
        money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        money_flow_volume = money_flow_multiplier * df['volume']
        return money_flow_volume.rolling(period).sum() / df['volume'].rolling(period).sum()

    def _add_market_regime_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced market regime detection indicators using ML."""
        df = df.copy()

        # Volatility regime (high/low volatility)
        df['volatility'] = df['close'].pct_change().rolling(20).std()
        df['volatility_regime'] = (df['volatility'] > df['volatility'].rolling(100).mean()).astype(int)

        # Trend strength
        df['trend_strength'] = abs(df['close'].rolling(20).mean() - df['close'].rolling(50).mean())

        # Volume regime
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_regime'] = (df['volume'] > df['volume_sma']).astype(int)

        return df

    def _add_momentum_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum and volatility indicators."""
        df = df.copy()

        # Momentum indicators
        df['momentum_1d'] = df['close'].pct_change(1)
        df['momentum_5d'] = df['close'].pct_change(5)
        df['momentum_20d'] = df['close'].pct_change(20)

        # Volatility indicators
        df['volatility_5d'] = df['close'].pct_change().rolling(5).std()
        df['volatility_20d'] = df['close'].pct_change().rolling(20).std()
        df['volatility_ratio'] = df['volatility_5d'] / df['volatility_20d']

        # Rate of change
        df['roc_5d'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
        df['roc_20d'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)

        return df

    async def _integrate_alternative_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Integrate alternative data sources into the feature set."""
        try:
            df = df.copy()

            # Initialize alternative data service if needed
            await self.alternative_data_service.initialize()

            # Get date range for alternative data
            start_date = df.index.min().strftime('%Y-%m-%d')
            end_date = df.index.max().strftime('%Y-%m-%d')

            logger.info(f"Fetching alternative data for {symbol} from {start_date} to {end_date}")

            # Fetch sentiment data
            try:
                sentiment_data = await self.alternative_data_service.get_sentiment_data(
                    symbol=symbol,
                    timeframe="1d",
                    limit=len(df)
                )

                if sentiment_data and 'data' in sentiment_data:
                    sentiment_df = pd.DataFrame(sentiment_data['data'])
                    if not sentiment_df.empty and 'timestamp' in sentiment_df.columns:
                        sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'])
                        sentiment_df = sentiment_df.set_index('timestamp')

                        # Merge sentiment data
                        for col in ['sentiment_score', 'sentiment_volume', 'bullish_ratio']:
                            if col in sentiment_df.columns:
                                df[f'sentiment_{col}'] = sentiment_df[col]

                        logger.info("Sentiment data integrated successfully")
            except Exception as e:
                logger.warning(f"Failed to fetch sentiment data: {e}")

            # Fetch on-chain metrics
            try:
                onchain_data = await self.alternative_data_service.get_onchain_metrics(
                    symbol=symbol,
                    metrics=['active_addresses', 'transaction_count', 'hash_rate']
                )

                if onchain_data and 'data' in onchain_data:
                    onchain_df = pd.DataFrame(onchain_data['data'])
                    if not onchain_df.empty and 'timestamp' in onchain_df.columns:
                        onchain_df['timestamp'] = pd.to_datetime(onchain_df['timestamp'])
                        onchain_df = onchain_df.set_index('timestamp')

                        # Merge on-chain features
                        for col in ['active_addresses', 'transaction_count', 'hash_rate']:
                            if col in onchain_df.columns:
                                df[f'onchain_{col}'] = onchain_df[col]

                        logger.info("On-chain data integrated successfully")
            except Exception as e:
                logger.warning(f"Failed to fetch on-chain data: {e}")

            # Fetch economic indicators
            try:
                economic_data = await self.alternative_data_service.get_economic_indicators(
                    indicators=['interest_rate', 'inflation', 'gdp_growth']
                )

                if economic_data and 'data' in economic_data:
                    economic_df = pd.DataFrame(economic_data['data'])
                    if not economic_df.empty and 'timestamp' in economic_df.columns:
                        economic_df['timestamp'] = pd.to_datetime(economic_df['timestamp'])
                        economic_df = onchain_df.set_index('timestamp')

                        # Merge economic features
                        for col in ['interest_rate', 'inflation', 'gdp_growth']:
                            if col in economic_df.columns:
                                df[f'economic_{col}'] = economic_df[col]

                        logger.info("Economic data integrated successfully")
            except Exception as e:
                logger.warning(f"Failed to fetch economic data: {e}")

            # Fetch Fear & Greed Index
            try:
                fear_greed_data = await self.alternative_data_service.get_fear_greed_index()

                if fear_greed_data and 'data' in fear_greed_data:
                    fear_greed_df = pd.DataFrame(fear_greed_data['data'])
                    if not fear_greed_df.empty and 'timestamp' in fear_greed_df.columns:
                        fear_greed_df['timestamp'] = pd.to_datetime(fear_greed_df['timestamp'])
                        fear_greed_df = fear_greed_df.set_index('timestamp')

                        # Add fear & greed features
                        for col in ['fear_greed_value', 'fear_greed_classification']:
                            if col in fear_greed_df.columns:
                                df[f'fear_greed_{col}'] = fear_greed_df[col]

                        logger.info("Fear & Greed Index integrated successfully")
            except Exception as e:
                logger.warning(f"Failed to fetch Fear & Greed Index: {e}")

            # Fill missing alternative data with forward/backward fill
            alternative_cols = [col for col in df.columns if any(prefix in col for prefix in
                                                                ['sentiment_', 'onchain_', 'economic_', 'fear_greed_'])]
            if alternative_cols:
                df[alternative_cols] = df[alternative_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)

            logger.info(f"Alternative data integration completed. Added {len(alternative_cols)} features.")

            return df

        except Exception as e:
            logger.error(f"Failed to integrate alternative data: {e}")
            return df  # Return original dataframe if integration fails

    async def _get_composite_alternative_score(self, symbol: str) -> Dict[str, Any]:
        """Get composite alternative data score for a symbol."""
        try:
            await self.alternative_data_service.initialize()
            composite_data = await self.alternative_data_service.get_composite_score(symbol)

            if composite_data and 'composite_score' in composite_data:
                return {
                    'composite_score': composite_data['composite_score'],
                    'sentiment_contribution': composite_data.get('sentiment_contribution', 0),
                    'onchain_contribution': composite_data.get('onchain_contribution', 0),
                    'economic_contribution': composite_data.get('economic_contribution', 0),
                    'fear_greed_contribution': composite_data.get('fear_greed_contribution', 0),
                    'market_regime': composite_data.get('market_regime', 'unknown')
                }
            else:
                return {"error": "Failed to retrieve composite score"}

        except Exception as e:
            logger.error(f"Failed to get composite alternative score: {e}")
            return {"error": str(e)}

    async def explain_prediction(self, model_name: str, features: Dict[str, float]) -> Dict[str, Any]:
        """Explain a prediction using SHAP."""
        try:
            # Load model and metadata
            model, metadata = await self._load_model(model_name)

            # Initialize SHAP interpreter for this model
            success = self.shap_interpreter.load_model(
                str(self.models_dir / f"{model_name}.pkl"),
                metadata.features
            )

            if not success:
                return {"error": "Failed to load model for SHAP interpretation"}

            # Set background data (use a small sample for efficiency)
            background_size = min(50, len(metadata.features))
            background_data = np.random.randn(background_size, len(metadata.features))
            self.shap_interpreter.set_background_data(background_data)

            # Explain prediction
            explanation = self.shap_interpreter.explain_trading_decision(features, 0.0)

            return explanation

        except Exception as e:
            logger.error(f"Failed to explain prediction: {e}")
            return {"error": str(e)}

    async def get_model_interpretability(self, model_name: str) -> Dict[str, Any]:
        """Get interpretability report for a model.

        Args:
            model_name: Name of the model

        Returns:
            Interpretability report
        """
        try:
            # Try to load saved SHAP report
            report_file = self.models_dir / model_name / "shap_report.json"
            if report_file.exists():
                with open(report_file, 'r') as f:
                    report = json.load(f)
                return report

            # If no saved report, return basic info
            model, metadata = await self._load_model(model_name)
            return {
                "model_name": model_name,
                "features": metadata.features,
                "performance": metadata.performance,
                "shap_available": False,
                "message": "SHAP report not available. Install shap package and retrain model."
            }

        except Exception as e:
            logger.error(f"Failed to get model interpretability: {e}")
            return {"error": str(e)}


class EnsembleTrainer:
    """Advanced ensemble trainer for benchmark-outperforming ML models."""

    def __init__(self):
        self.models = {}
        self.ensemble_weights = {}
        self.feature_importance_ensemble = {}
        self.model_correlations = {}
        self.diversity_metrics = {}

    def create_stacking_ensemble(self, models: Dict[str, Any], cv_folds: int = 5) -> VotingClassifier:
        """Create a stacking ensemble with cross-validation."""
        estimators = [(name, model) for name, model in models.items()]

        # Use logistic regression as meta-learner
        from sklearn.linear_model import LogisticRegression
        meta_learner = LogisticRegression(random_state=ML_RANDOM_STATE)

        from sklearn.ensemble import StackingClassifier
        return StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=cv_folds,
            n_jobs=-1
        )

    def create_weighted_voting_ensemble(self, models: Dict[str, Any], X_val: np.ndarray, y_val: np.ndarray) -> VotingClassifier:
        """Create a weighted voting ensemble based on validation performance."""
        # Evaluate each model on validation set
        model_scores = {}
        for name, model in models.items():
            try:
                y_pred = model.predict(X_val)
                f1 = f1_score(y_val, y_pred, zero_division=0)
                model_scores[name] = max(f1, 0.001)  # Avoid zero weights
            except Exception as e:
                logger.warning(f"Failed to evaluate {name}: {e}")
                model_scores[name] = 0.001

        # Calculate weights based on performance
        total_score = sum(model_scores.values())
        weights = [model_scores[name] / total_score for name, _ in models.items()]

        # Normalize weights
        weights = np.array(weights) / np.sum(weights)

        logger.info(f"Ensemble weights: {dict(zip(models.keys(), weights))}")

        estimators = [(name, model) for name, model in models.items()]

        return VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=weights,
            n_jobs=-1
        )

    def optimize_ensemble_weights(self, models: Dict[str, Any], X_val: np.ndarray, y_val: np.ndarray) -> VotingClassifier:
        """Optimize ensemble weights using numerical optimization."""
        def objective(weights):
            """Objective function to maximize F1 score."""
            # Ensure weights sum to 1 and are non-negative
            weights = np.abs(weights)
            weights = weights / np.sum(weights)

            # Get predictions from each model
            predictions = []
            for model in models.values():
                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba(X_val)[:, 1]
                else:
                    pred_proba = model.predict(X_val).astype(float)
                predictions.append(pred_proba)

            # Weighted average prediction
            ensemble_pred_proba = np.average(predictions, axis=0, weights=weights)
            ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)

            # Calculate F1 score (negative because we minimize)
            f1 = f1_score(y_val, ensemble_pred, zero_division=0)
            return -f1

        # Initial weights (equal)
        n_models = len(models)
        initial_weights = np.ones(n_models) / n_models

        # Optimize weights
        bounds = [(0.01, 1.0) for _ in range(n_models)]  # Weight bounds
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights must sum to 1

        try:
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100}
            )

            optimal_weights = np.abs(result.x)
            optimal_weights = optimal_weights / np.sum(optimal_weights)

            logger.info(f"Optimized ensemble weights: {dict(zip(models.keys(), optimal_weights))}")

        except Exception as e:
            logger.warning(f"Weight optimization failed: {e}, using equal weights")
            optimal_weights = initial_weights

        estimators = [(name, model) for name, model in models.items()]

        return VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=optimal_weights,
            n_jobs=-1
        )

    def calculate_model_diversity(self, models: Dict[str, Any], X: np.ndarray) -> Dict[str, Any]:
        """Calculate diversity metrics for ensemble models."""
        diversity_metrics = {}

        # Get predictions from all models
        predictions = {}
        for name, model in models.items():
            try:
                pred = model.predict(X)
                predictions[name] = pred
            except Exception as e:
                logger.warning(f"Failed to get predictions from {name}: {e}")
                continue

        if len(predictions) < 2:
            return {'error': 'Need at least 2 models for diversity calculation'}

        model_names = list(predictions.keys())

        # Calculate pairwise disagreement
        pairwise_disagreement = {}
        for i, name1 in enumerate(model_names):
            for j, name2 in enumerate(model_names):
                if i < j:
                    disagree = np.mean(predictions[name1] != predictions[name2])
                    pairwise_disagreement[f"{name1}_{name2}"] = disagree

        # Calculate correlation matrix
        pred_matrix = np.column_stack([predictions[name] for name in model_names])
        correlation_matrix = np.corrcoef(pred_matrix.T)

        # Average disagreement
        avg_disagreement = np.mean(list(pairwise_disagreement.values()))

        # Diversity score (higher is more diverse)
        diversity_score = avg_disagreement

        diversity_metrics = {
            'pairwise_disagreement': pairwise_disagreement,
            'correlation_matrix': correlation_matrix.tolist(),
            'average_disagreement': avg_disagreement,
            'diversity_score': diversity_score,
            'correlation_mean': np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
        }

        self.diversity_metrics = diversity_metrics
        return diversity_metrics

    def select_optimal_ensemble(self, models: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Select the optimal ensemble method based on performance and diversity."""
        ensemble_options = {}

        # Calculate diversity
        diversity = self.calculate_model_diversity(models, X_val)

        # 1. Simple Voting Ensemble
        try:
            voting_ensemble = self.create_voting_ensemble(models)
            voting_ensemble.fit(X_train, y_train)
            voting_pred = voting_ensemble.predict(X_val)
            voting_f1 = f1_score(y_val, voting_pred, zero_division=0)
            ensemble_options['voting'] = {
                'model': voting_ensemble,
                'f1_score': voting_f1,
                'type': 'simple_voting'
            }
        except Exception as e:
            logger.warning(f"Voting ensemble failed: {e}")

        # 2. Weighted Voting Ensemble
        try:
            weighted_voting = self.create_weighted_voting_ensemble(models, X_val, y_val)
            weighted_voting.fit(X_train, y_train)
            weighted_pred = weighted_voting.predict(X_val)
            weighted_f1 = f1_score(y_val, weighted_pred, zero_division=0)
            ensemble_options['weighted_voting'] = {
                'model': weighted_voting,
                'f1_score': weighted_f1,
                'type': 'weighted_voting'
            }
        except Exception as e:
            logger.warning(f"Weighted voting ensemble failed: {e}")

        # 3. Optimized Weights Ensemble
        try:
            optimized_voting = self.optimize_ensemble_weights(models, X_val, y_val)
            optimized_voting.fit(X_train, y_train)
            optimized_pred = optimized_voting.predict(X_val)
            optimized_f1 = f1_score(y_val, optimized_pred, zero_division=0)
            ensemble_options['optimized_voting'] = {
                'model': optimized_voting,
                'f1_score': optimized_f1,
                'type': 'optimized_voting'
            }
        except Exception as e:
            logger.warning(f"Optimized voting ensemble failed: {e}")

        # 4. Stacking Ensemble
        try:
            stacking_ensemble = self.create_stacking_ensemble(models)
            stacking_ensemble.fit(X_train, y_train)
            stacking_pred = stacking_ensemble.predict(X_val)
            stacking_f1 = f1_score(y_val, stacking_pred, zero_division=0)
            ensemble_options['stacking'] = {
                'model': stacking_ensemble,
                'f1_score': stacking_f1,
                'type': 'stacking'
            }
        except Exception as e:
            logger.warning(f"Stacking ensemble failed: {e}")

        # Select best performing ensemble
        if ensemble_options:
            best_ensemble_name = max(ensemble_options.keys(),
                                   key=lambda x: ensemble_options[x]['f1_score'])
            best_ensemble = ensemble_options[best_ensemble_name]

            logger.info(f"Selected optimal ensemble: {best_ensemble_name} with F1={best_ensemble['f1_score']:.4f}")

            return {
                'model': best_ensemble['model'],
                'model_name': f"{best_ensemble_name}_ensemble",
                'performance': {'f1_score': best_ensemble['f1_score']},
                'ensemble_type': best_ensemble['type'],
                'all_options': ensemble_options,
                'diversity_metrics': diversity
            }
        else:
            # Fallback to best individual model
            logger.warning("All ensemble methods failed, falling back to best individual model")
            return self._select_best_individual_model(models, X_val, y_val)

    def _select_best_individual_model(self, models: Dict[str, Any], X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Select the best individual model as fallback."""
        model_scores = {}
        for name, model in models.items():
            try:
                y_pred = model.predict(X_val)
                f1 = f1_score(y_val, y_pred, zero_division=0)
                model_scores[name] = f1
            except Exception as e:
                logger.warning(f"Failed to evaluate {name}: {e}")
                model_scores[name] = 0

        best_model_name = max(model_scores.keys(), key=lambda x: model_scores[x])
        best_model = models[best_model_name]

        return {
            'model': best_model,
            'model_name': best_model_name,
            'performance': {'f1_score': model_scores[best_model_name]},
            'ensemble_type': 'individual_fallback'
        }

    def create_voting_ensemble(self, models: Dict[str, Any]) -> VotingClassifier:
        """Create a simple voting ensemble."""
        estimators = [(name, model) for name, model in models.items()]

        return VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )

    def train_ensemble(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train an ensemble of models for outperformance."""
        # Define base models for ensemble
        base_models = {
            'rf': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=ML_RANDOM_STATE
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                random_state=ML_RANDOM_STATE
            ),
            'xgb': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=ML_RANDOM_STATE
            ),
            'lgb': lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=ML_RANDOM_STATE
            )
        }

        # Train base models
        trained_models = {}
        for name, model in base_models.items():
            try:
                model.fit(X, y)
                trained_models[name] = model
                logger.info(f"Trained {name} model")
            except Exception as e:
                logger.warning(f"Failed to train {name}: {e}")

        if not trained_models:
            raise ValueError("No models could be trained")

        # Select optimal ensemble
        ensemble_result = self.select_optimal_ensemble(
            trained_models, X, y, X, y  # Using same data for simplicity
        )

        # Calculate feature importance for ensemble
        self._calculate_ensemble_feature_importance(trained_models, X)

        return {
            'model': ensemble_result['model'],
            'model_name': ensemble_result['model_name'],
            'performance': ensemble_result['performance'],
            'base_models': trained_models,
            'ensemble_type': ensemble_result['ensemble_type'],
            'diversity_metrics': ensemble_result.get('diversity_metrics', {})
        }

    def _calculate_ensemble_feature_importance(self, models: Dict[str, Any], X: np.ndarray):
        """Calculate feature importance for ensemble models."""
        feature_importances = {}

        for name, model in models.items():
            if hasattr(model, 'feature_importances_'):
                if name not in feature_importances:
                    feature_importances[name] = model.feature_importances_
                else:
                    feature_importances[name] = np.mean([feature_importances[name], model.feature_importances_], axis=0)

        # Average across all models
        if feature_importances:
            self.feature_importance_ensemble = np.mean(list(feature_importances.values()), axis=0)
        else:
            self.feature_importance_ensemble = np.zeros(X.shape[1])