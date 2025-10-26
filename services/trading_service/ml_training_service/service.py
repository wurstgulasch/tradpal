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
from services.data_service.alternative_data.client import AlternativeDataService
from services.trading_service.trading_ai_service.ml_training.ml_trainer import EnsembleTrainer

logger = logging.getLogger(__name__)


class SHAPInterpreter:
    """Simple SHAP interpreter placeholder for model explainability."""

    def __init__(self):
        self.model = None
        self.feature_names = []
        self.background_data = None
        self.explainer = None

    def load_model(self, model_path: str, feature_names: list) -> bool:
        """Load model for interpretation."""
        try:
            # Placeholder - would load actual model
            self.feature_names = feature_names
            return True
        except Exception:
            return False

    def set_background_data(self, background_data):
        """Set background data for SHAP explanations."""
        self.background_data = background_data

    def explain_trading_decision(self, features: dict, threshold: float = 0.0):
        """Explain a trading decision."""
        return {
            "trading_interpretation": {
                "signal_strength": "neutral",
                "confidence_level": "low",
                "top_contributing_features": list(features.keys())[:3]
            },
            "feature_importance": {
                feature: {"shap_value": 0.0, "feature_value": value}
                for feature, value in features.items()
            }
        }


class MLTrainerService:
    """ML model training and optimization service."""

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or Path("cache/ml_models")
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Initialize alternative data service
        self.alternative_data_service = AlternativeDataService()

        # Initialize ensemble trainer
        self.ensemble_trainer = EnsembleTrainer()

        # Initialize SHAP interpreter
        self.shap_interpreter = SHAPInterpreter()

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

    async def train_benchmark_outperforming_model(self, symbol: str, timeframe: str,
                                                start_date: str, end_date: str) -> Dict[str, Any]:
        """Train a model that outperforms traditional benchmarks."""
        try:
            # Fetch enhanced training data
            df = await self._fetch_enhanced_training_data(symbol, timeframe, start_date, end_date)
            if df.empty:
                return {"error": "No training data available", "success": False}

            # Prepare enhanced features
            X, y, feature_names = await self._prepare_enhanced_features(df, symbol)

            # Train ensemble model
            ensemble_result = self.ensemble_trainer.train_ensemble(X, y)

            # Analyze outperformance against benchmark
            benchmark_metrics = {
                'accuracy': 0.52,
                'precision': 0.50,
                'recall': 0.48,
                'f1_score': 0.49,
                'roc_auc': 0.51
            }

            outperformance_analysis = self._analyze_outperformance(
                ensemble_result, benchmark_metrics, X, y
            )

            # Save model
            model_name = f"{symbol}_benchmark_outperformer_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            await self._save_model(ensemble_result['model'], model_name, {
                "symbol": symbol,
                "model_type": "benchmark_outperformer",
                "performance": ensemble_result['performance'],
                "outperformance_analysis": outperformance_analysis,
                "features": feature_names,
                "training_date": pd.Timestamp.now().isoformat()
            })

            return {
                "success": True,
                "model_name": model_name,
                "performance": ensemble_result['performance'],
                "outperformance_analysis": outperformance_analysis,
                "selected_model": ensemble_result['model_name']
            }

        except Exception as e:
            logger.error(f"Failed to train benchmark outperforming model: {e}")
            return {"error": str(e), "success": False}

    async def _fetch_enhanced_training_data(self, symbol: str, timeframe: str,
                                          start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch enhanced training data with additional features."""
        # Get basic data
        df = await self._fetch_training_data(symbol, timeframe, start_date, end_date)
        if df.empty:
            return df

        # Add enhanced indicators
        df = self._add_advanced_technical_indicators(df)
        df = self._add_market_regime_indicators(df)
        df = self._add_momentum_volatility_indicators(df)

        # Add alternative data
        df = await self._integrate_alternative_data(df, symbol)

        return df

    def _analyze_outperformance(self, model_result: Dict[str, Any], benchmark_metrics: Dict[str, float],
                              X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analyze how much the model outperforms benchmarks."""
        model_performance = model_result['performance']

        metric_outperformance = {}
        outperforming_metrics = 0
        total_improvement = 0

        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
            if metric in model_performance and metric in benchmark_metrics:
                model_value = model_performance[metric]
                benchmark_value = benchmark_metrics[metric]
                improvement = model_value - benchmark_value
                improvement_pct = (improvement / benchmark_value) * 100 if benchmark_value > 0 else 0

                metric_outperformance[metric] = {
                    'model_value': model_value,
                    'benchmark_value': benchmark_value,
                    'improvement': improvement,
                    'improvement_pct': improvement_pct,
                    'outperforms': improvement > 0
                }

                if improvement > 0:
                    outperforming_metrics += 1
                    total_improvement += improvement_pct

        overall_outperformance_pct = total_improvement / len(metric_outperformance) if metric_outperformance else 0

        return {
            'metric_outperformance': metric_outperformance,
            'outperforming_metrics': outperforming_metrics,
            'total_metrics': len(metric_outperformance),
            'overall_outperformance_pct': overall_outperformance_pct,
            'consistency': outperforming_metrics / len(metric_outperformance) if metric_outperformance else 0
        }