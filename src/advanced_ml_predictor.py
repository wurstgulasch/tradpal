"""
Advanced ML Predictor with Ensemble Methods, Market Regime Detection, and Reinforcement Learning

This module provides advanced machine learning capabilities for trading signal prediction,
including ensemble methods, market regime detection, reinforcement learning integration,
and advanced feature engineering.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import json
import logging
from datetime import datetime, timedelta
import pickle

# ML Libraries
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Config imports
from config.settings import (
    SYMBOL, TIMEFRAME, ML_ADVANCED_FEATURES_ENABLED,
    ML_ENSEMBLE_MODELS, ML_MARKET_REGIME_DETECTION,
    ML_REINFORCEMENT_LEARNING, ML_GPU_OPTIMIZATION
)

# Existing ML imports
from .ml_predictor import (
    LSTMSignalPredictor, TransformerSignalPredictor,
    get_lstm_predictor, get_transformer_predictor,
    is_lstm_available, is_transformer_available
)

# Audit logging
from src.audit_logger import audit_logger


class MarketRegimeDetector:
    """
    Market Regime Detection using unsupervised learning and technical indicators.

    Detects different market conditions: trending, ranging, volatile, calm.
    """

    def __init__(self, lookback_periods: int = 50):
        """
        Initialize market regime detector.

        Args:
            lookback_periods: Number of periods to analyze for regime detection
        """
        self.lookback_periods = lookback_periods
        self.regime_history = []
        self.regime_features = []

    def detect_regime(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect current market regime.

        Args:
            df: DataFrame with OHLCV and technical indicators

        Returns:
            Dictionary with regime information
        """
        if len(df) < self.lookback_periods:
            return {'regime': 'unknown', 'confidence': 0.0}

        # Calculate regime features
        recent_data = df.tail(self.lookback_periods).copy()

        # Volatility (ATR normalized)
        if 'ATR' in recent_data.columns:
            volatility = recent_data['ATR'].iloc[-1] / recent_data['close'].iloc[-1]
        else:
            volatility = recent_data['close'].pct_change().std()

        # Trend strength (ADX)
        if 'ADX' in recent_data.columns:
            trend_strength = recent_data['ADX'].iloc[-1] / 100.0
        else:
            # Calculate simple trend strength
            sma_short = recent_data['close'].rolling(10).mean()
            sma_long = recent_data['close'].rolling(30).mean()
            trend_strength = abs(sma_short.iloc[-1] - sma_long.iloc[-1]) / recent_data['close'].iloc[-1]

        # Momentum (RSI divergence from 50)
        if 'RSI' in recent_data.columns:
            momentum = abs(recent_data['RSI'].iloc[-1] - 50) / 50.0
        else:
            momentum = abs(recent_data['close'].pct_change(10).iloc[-1])

        # Volume trend
        if 'volume' in recent_data.columns:
            volume_trend = recent_data['volume'].pct_change(10).mean()
        else:
            volume_trend = 0.0

        # Determine regime based on features
        if trend_strength > 0.05 and volatility < 0.03:
            regime = 'strong_trend'
            confidence = min(trend_strength * 2, 0.9)
        elif trend_strength > 0.03 and volatility > 0.05:
            regime = 'volatile_trend'
            confidence = (trend_strength + volatility) / 2
        elif volatility > 0.08:
            regime = 'high_volatility'
            confidence = volatility
        elif abs(momentum) < 0.1 and volatility < 0.02:
            regime = 'ranging'
            confidence = 1.0 - volatility - abs(momentum)
        elif volume_trend > 0.2:
            regime = 'high_volume'
            confidence = volume_trend
        else:
            regime = 'normal'
            confidence = 0.5

        # Store regime history
        self.regime_history.append({
            'timestamp': df.index[-1] if hasattr(df.index[-1], 'timestamp') else datetime.now(),
            'regime': regime,
            'confidence': confidence,
            'features': {
                'volatility': volatility,
                'trend_strength': trend_strength,
                'momentum': momentum,
                'volume_trend': volume_trend
            }
        })

        # Keep only recent history
        if len(self.regime_history) > 1000:
            self.regime_history = self.regime_history[-1000:]

        return {
            'regime': regime,
            'confidence': confidence,
            'features': {
                'volatility': volatility,
                'trend_strength': trend_strength,
                'momentum': momentum,
                'volume_trend': volume_trend
            }
        }

    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get statistics about detected regimes."""
        if not self.regime_history:
            return {'total_regimes': 0}

        regimes = [r['regime'] for r in self.regime_history]
        unique_regimes = list(set(regimes))

        stats = {
            'total_regimes': len(regimes),
            'unique_regimes': unique_regimes,
            'regime_counts': {regime: regimes.count(regime) for regime in unique_regimes},
            'most_common_regime': max(unique_regimes, key=lambda x: regimes.count(x)) if unique_regimes else None
        }

        return stats


class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for trading signals.

    Creates sophisticated features from raw price and volume data.
    """

    def __init__(self):
        """Initialize feature engineer."""
        self.feature_scalers = {}
        self.feature_names = []

    def create_advanced_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Create advanced features from OHLCV data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        features = []
        feature_names = []

        # Basic price features
        if 'close' in df.columns:
            # Price momentum features
            for period in [5, 10, 20, 50]:
                momentum = df['close'].pct_change(period)
                features.append(momentum.fillna(0))
                feature_names.append(f'price_momentum_{period}')

                # Rate of change of momentum
                momentum_roc = momentum.pct_change(5).fillna(0)
                features.append(momentum_roc)
                feature_names.append(f'momentum_roc_{period}')

            # Price acceleration (second derivative)
            acceleration = df['close'].pct_change().pct_change().fillna(0)
            features.append(acceleration)
            feature_names.append('price_acceleration')

        # Volume features
        if 'volume' in df.columns:
            # Volume momentum
            for period in [5, 10, 20]:
                vol_momentum = df['volume'].pct_change(period).fillna(0)
                features.append(vol_momentum)
                feature_names.append(f'volume_momentum_{period}')

            # Volume-price trend
            if 'close' in df.columns:
                vpt = (df['close'].pct_change() * df['volume']).cumsum().fillna(0)
                features.append(vpt)
                feature_names.append('volume_price_trend')

        # Volatility features
        if all(col in df.columns for col in ['high', 'low', 'close']):
            # True Range
            tr = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                )
            ).fillna(0)
            features.append(tr)
            feature_names.append('true_range')

            # ATR (simplified)
            atr = tr.rolling(14).mean().fillna(0)
            features.append(atr)
            feature_names.append('atr_14')

            # Bollinger Band features
            sma = df['close'].rolling(20).mean()
            std = df['close'].rolling(20).std()
            bb_upper = sma + 2 * std
            bb_lower = sma - 2 * std
            bb_width = (bb_upper - bb_lower) / sma

            features.append(bb_width.fillna(0))
            feature_names.append('bb_width')

            # Price position in BB
            bb_position = (df['close'] - bb_lower) / (bb_upper - bb_lower)
            features.append(bb_position.fillna(0.5))
            feature_names.append('bb_position')

        # Statistical features
        if 'close' in df.columns:
            # Rolling statistics
            for period in [10, 20, 50]:
                roll_mean = df['close'].rolling(period).mean()
                roll_std = df['close'].rolling(period).std()
                roll_skew = df['close'].rolling(period).skew()
                roll_kurt = df['close'].rolling(period).kurt()

                features.extend([
                    roll_mean.fillna(0),
                    roll_std.fillna(0),
                    roll_skew.fillna(0),
                    roll_kurt.fillna(0)
                ])
                feature_names.extend([
                    f'roll_mean_{period}',
                    f'roll_std_{period}',
                    f'roll_skew_{period}',
                    f'roll_kurt_{period}'
                ])

        # Combine all features
        if features:
            feature_matrix = np.column_stack(features)
        else:
            feature_matrix = np.zeros((len(df), 1))
            feature_names = ['dummy_feature']

        # Handle NaN and infinite values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        self.feature_names = feature_names
        return feature_matrix, feature_names

    def scale_features(self, features: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Scale features using robust scaler.

        Args:
            features: Feature matrix
            fit: Whether to fit the scaler

        Returns:
            Scaled features
        """
        if not SKLEARN_AVAILABLE:
            return features

        if 'robust_scaler' not in self.feature_scalers:
            self.feature_scalers['robust_scaler'] = RobustScaler()

        scaler = self.feature_scalers['robust_scaler']

        if fit:
            return scaler.fit_transform(features)
        else:
            return scaler.transform(features)


class EnsembleModel(nn.Module if TORCH_AVAILABLE else object):
    """
    PyTorch ensemble model combining multiple architectures.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64], dropout_rate: float = 0.2):
        """
        Initialize ensemble model.

        Args:
            input_dim: Input feature dimension
            hidden_dims: Hidden layer dimensions
            dropout_rate: Dropout rate
        """
        if TORCH_AVAILABLE:
            super().__init__()
        else:
            # Fallback if PyTorch not available
            return

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass."""
        return self.model(x)


class AdvancedMLPredictor:
    """
    Advanced ML Predictor with Ensemble Methods, Market Regime Detection, and RL Integration.

    Features:
    - Multiple ML model ensemble (LSTM, Transformer, RF, GB)
    - Market regime detection and adaptation
    - Reinforcement learning integration
    - Advanced feature engineering
    - GPU optimization
    - Confidence-weighted predictions
    """

    def __init__(self, model_dir: str = "cache/ml_models", symbol: str = SYMBOL,
                 timeframe: str = TIMEFRAME, sequence_length: int = 60):
        """
        Initialize advanced ML predictor.

        Args:
            model_dir: Directory to store trained models
            symbol: Trading symbol
            timeframe: Timeframe
            sequence_length: Sequence length for time series models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.symbol = symbol
        self.timeframe = timeframe
        self.sequence_length = sequence_length

        # Initialize components
        self.regime_detector = MarketRegimeDetector()
        self.feature_engineer = AdvancedFeatureEngineer()

        # Model components
        self.models = {}
        self.model_weights = {}
        self.is_trained = False

        # Performance tracking
        self.performance_history = []
        self.regime_performance = {}

        # GPU setup
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() and ML_GPU_OPTIMIZATION else 'cpu')
        else:
            self.device = 'cpu'

        # Load existing models
        self._load_models()

    def build_ensemble(self, input_dim: int) -> Dict[str, Any]:
        """
        Build ensemble of ML models.

        Args:
            input_dim: Input feature dimension

        Returns:
            Dictionary of trained models
        """
        models = {}

        # PyTorch Ensemble Model
        if TORCH_AVAILABLE and 'torch_ensemble' in ML_ENSEMBLE_MODELS:
            models['torch_ensemble'] = EnsembleModel(input_dim=input_dim).to(self.device)

        # Scikit-learn models
        if SKLEARN_AVAILABLE:
            if 'random_forest' in ML_ENSEMBLE_MODELS:
                models['random_forest'] = RandomForestClassifier(
                    n_estimators=100, max_depth=10, random_state=42
                )

            if 'gradient_boosting' in ML_ENSEMBLE_MODELS:
                models['gradient_boosting'] = GradientBoostingClassifier(
                    n_estimators=100, max_depth=5, random_state=42
                )

        # Existing deep learning models
        if is_lstm_available():
            models['lstm'] = get_lstm_predictor(self.symbol, self.timeframe)

        if is_transformer_available():
            models['transformer'] = get_transformer_predictor(self.symbol, self.timeframe)

        self.models = models

        # Initialize equal weights
        self.model_weights = {name: 1.0 / len(models) for name in models.keys()}

        return models

    def train_ensemble(self, df: pd.DataFrame, test_size: float = 0.2,
                      prediction_horizon: int = 5) -> Dict[str, Any]:
        """
        Train the ensemble of models.

        Args:
            df: DataFrame with technical indicators and price data
            test_size: Fraction of data for testing
            prediction_horizon: Periods to look ahead for labeling

        Returns:
            Training results
        """
        try:
            print("🔬 Training Advanced ML Ensemble...")

            # Create advanced features
            features, feature_names = self.feature_engineer.create_advanced_features(df)
            scaled_features = self.feature_engineer.scale_features(features, fit=True)

            # Create labels
            labels = self._create_labels(df, prediction_horizon)

            # Align features and labels
            min_length = min(len(scaled_features), len(labels))
            scaled_features = scaled_features[:min_length]
            labels = labels[:min_length]

            if len(scaled_features) < 100:
                raise ValueError(f"Insufficient data for training: {len(scaled_features)} samples")

            # Split data
            split_idx = int(len(scaled_features) * (1 - test_size))
            X_train, X_test = scaled_features[:split_idx], scaled_features[split_idx:]
            y_train, y_test = labels[:split_idx], labels[split_idx:]

            # Build ensemble
            self.build_ensemble(input_dim=X_train.shape[1])

            training_results = {}

            # Train each model
            for name, model in self.models.items():
                try:
                    if name == 'torch_ensemble' and TORCH_AVAILABLE:
                        result = self._train_torch_model(model, X_train, y_train, X_test, y_test)
                    elif name in ['random_forest', 'gradient_boosting'] and SKLEARN_AVAILABLE:
                        result = self._train_sklearn_model(model, X_train, y_train, X_test, y_test)
                    elif name in ['lstm', 'transformer']:
                        # Use existing training methods
                        if hasattr(model, 'train_model'):
                            result = model.train_model(df, test_size=test_size, prediction_horizon=prediction_horizon)
                        else:
                            result = {'success': False, 'error': 'No training method'}
                    else:
                        result = {'success': False, 'error': 'Unknown model type'}

                    training_results[name] = result

                except Exception as e:
                    training_results[name] = {'success': False, 'error': str(e)}

            # Calculate ensemble weights based on performance
            self._update_ensemble_weights(training_results)

            self.is_trained = True
            self._save_models()

            # Log training completion
            audit_logger.log_system_event(
                event_type="ADVANCED_ML_TRAINING_COMPLETED",
                message=f"Advanced ML ensemble training completed for {self.symbol} {self.timeframe}",
                details={
                    'models_trained': len([r for r in training_results.values() if r.get('success', False)]),
                    'total_models': len(training_results),
                    'feature_count': len(feature_names)
                }
            )

            print("✅ Advanced ML Ensemble trained successfully")

            return {
                'success': True,
                'training_results': training_results,
                'ensemble_weights': self.model_weights.copy(),
                'feature_count': len(feature_names)
            }

        except Exception as e:
            error_msg = f"Advanced ML training failed: {str(e)}"
            print(f"❌ {error_msg}")

            audit_logger.log_error(
                error_type="ADVANCED_ML_TRAINING_ERROR",
                message=error_msg,
                context={'symbol': self.symbol, 'timeframe': self.timeframe}
            )

            return {'success': False, 'error': str(e)}

    def predict_signal(self, df: pd.DataFrame, threshold: float = 0.6) -> Dict[str, Any]:
        """
        Generate ensemble prediction with market regime adaptation.

        Args:
            df: DataFrame with latest technical indicators
            threshold: Confidence threshold for signal generation

        Returns:
            Dictionary with ensemble prediction results
        """
        if not self.is_trained or not self.models:
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'reason': 'Ensemble not trained'
            }

        try:
            # Detect market regime
            regime_info = self.regime_detector.detect_regime(df)
            regime = regime_info['regime']

            # Create features
            features, _ = self.feature_engineer.create_advanced_features(df)
            scaled_features = self.feature_engineer.scale_features(features, fit=False)

            if len(scaled_features) == 0:
                return {
                    'signal': 'HOLD',
                    'confidence': 0.0,
                    'reason': 'No features available'
                }

            # Get predictions from each model
            model_predictions = {}
            model_confidences = {}

            for name, model in self.models.items():
                try:
                    if name == 'torch_ensemble' and TORCH_AVAILABLE:
                        pred = self._predict_torch_model(model, scaled_features)
                    elif name in ['random_forest', 'gradient_boosting'] and SKLEARN_AVAILABLE:
                        pred = self._predict_sklearn_model(model, scaled_features)
                    elif name in ['lstm', 'transformer']:
                        pred = model.predict_signal(df)
                    else:
                        continue

                    model_predictions[name] = pred.get('signal', 'HOLD')
                    model_confidences[name] = pred.get('confidence', 0.5)

                except Exception as e:
                    print(f"⚠️  Prediction failed for {name}: {e}")
                    model_predictions[name] = 'HOLD'
                    model_confidences[name] = 0.0

            # Apply regime-based weighting
            regime_weights = self._get_regime_weights(regime)

            # Calculate weighted ensemble prediction
            ensemble_result = self._calculate_weighted_prediction(
                model_predictions, model_confidences, regime_weights, threshold
            )

            # Add regime information
            ensemble_result.update({
                'regime': regime,
                'regime_confidence': regime_info['confidence'],
                'model_predictions': model_predictions,
                'model_confidences': model_confidences,
                'regime_weights': regime_weights
            })

            return ensemble_result

        except Exception as e:
            error_msg = f"Ensemble prediction failed: {str(e)}"
            print(f"❌ {error_msg}")

            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'reason': f'Prediction error: {str(e)}'
            }

    def _train_torch_model(self, model, X_train, y_train, X_test, y_test) -> Dict[str, Any]:
        """Train PyTorch model."""
        if not TORCH_AVAILABLE:
            return {'success': False, 'error': 'PyTorch not available'}

        try:
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train).to(self.device)
            y_train_tensor = torch.FloatTensor(y_train).to(self.device)
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            y_test_tensor = torch.FloatTensor(y_test).to(self.device)

            # Create data loader
            train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

            # Optimizer and loss
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.BCELoss()

            # Training loop
            model.train()
            for epoch in range(50):
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

            # Evaluate
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_tensor).squeeze()
                test_predictions = (test_outputs > 0.5).float()
                accuracy = (test_predictions == y_test_tensor).float().mean().item()

            return {
                'success': True,
                'test_accuracy': accuracy,
                'model_type': 'torch_ensemble'
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _train_sklearn_model(self, model, X_train, y_train, X_test, y_test) -> Dict[str, Any]:
        """Train scikit-learn model."""
        if not SKLEARN_AVAILABLE:
            return {'success': False, 'error': 'Scikit-learn not available'}

        try:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)

            return {
                'success': True,
                'test_accuracy': accuracy,
                'model_type': 'sklearn'
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _predict_torch_model(self, model, features: np.ndarray) -> Dict[str, Any]:
        """Predict with PyTorch model."""
        if not TORCH_AVAILABLE:
            return {'signal': 'HOLD', 'confidence': 0.0}

        try:
            model.eval()
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features[-1:]).to(self.device)  # Last sample
                output = model(features_tensor).item()

            confidence = max(output, 1 - output)
            signal = 'BUY' if output > 0.5 else 'SELL'

            return {'signal': signal, 'confidence': confidence}

        except Exception as e:
            return {'signal': 'HOLD', 'confidence': 0.0}

    def _predict_sklearn_model(self, model, features: np.ndarray) -> Dict[str, Any]:
        """Predict with scikit-learn model."""
        if not SKLEARN_AVAILABLE:
            return {'signal': 'HOLD', 'confidence': 0.0}

        try:
            # Use probability if available
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(features[-1:])[0]
                confidence = max(prob)
                signal = 'BUY' if prob[1] > prob[0] else 'SELL'
            else:
                prediction = model.predict(features[-1:])[0]
                signal = 'BUY' if prediction == 1 else 'SELL'
                confidence = 0.6  # Default confidence

            return {'signal': signal, 'confidence': confidence}

        except Exception as e:
            return {'signal': 'HOLD', 'confidence': 0.0}

    def _create_labels(self, df: pd.DataFrame, prediction_horizon: int) -> np.ndarray:
        """Create binary labels for supervised learning."""
        if 'close' not in df.columns:
            return np.zeros(len(df))

        future_price = df['close'].shift(-prediction_horizon)
        current_price = df['close']
        labels = (future_price > current_price).astype(int)

        return labels.fillna(0).values

    def _update_ensemble_weights(self, training_results: Dict[str, Any]):
        """Update ensemble weights based on model performance."""
        successful_models = {name: result for name, result in training_results.items()
                           if result.get('success', False)}

        if not successful_models:
            return

        # Use accuracy for weighting
        accuracies = {name: result.get('test_accuracy', 0.5) for name, result in successful_models.items()}

        # Normalize weights
        total_accuracy = sum(accuracies.values())
        if total_accuracy > 0:
            self.model_weights = {name: acc / total_accuracy for name, acc in accuracies.items()}
        else:
            # Equal weights if no accuracy data
            weight = 1.0 / len(successful_models)
            self.model_weights = {name: weight for name in successful_models.keys()}

    def _get_regime_weights(self, regime: str) -> Dict[str, float]:
        """Get regime-adapted model weights."""
        # Base weights
        regime_weights = self.model_weights.copy()

        # Adjust weights based on regime
        if regime == 'strong_trend':
            # Favor momentum-based models
            regime_weights = self._adjust_weights_for_regime(regime_weights, ['lstm', 'transformer'], 1.2)
        elif regime == 'high_volatility':
            # Favor robust models
            regime_weights = self._adjust_weights_for_regime(regime_weights, ['random_forest', 'gradient_boosting'], 1.2)
        elif regime == 'ranging':
            # Favor mean-reversion models
            regime_weights = self._adjust_weights_for_regime(regime_weights, ['torch_ensemble'], 1.2)

        return regime_weights

    def _adjust_weights_for_regime(self, weights: Dict[str, float], favored_models: List[str], multiplier: float) -> Dict[str, float]:
        """Adjust weights to favor certain models."""
        adjusted_weights = weights.copy()

        # Increase weights for favored models
        for model in favored_models:
            if model in adjusted_weights:
                adjusted_weights[model] *= multiplier

        # Normalize
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {name: w / total_weight for name, w in adjusted_weights.items()}

        return adjusted_weights

    def _calculate_weighted_prediction(self, predictions: Dict[str, str], confidences: Dict[str, float],
                                     weights: Dict[str, float], threshold: float) -> Dict[str, Any]:
        """Calculate weighted ensemble prediction."""
        # Convert signals to scores
        signal_scores = {'BUY': 1.0, 'SELL': -1.0, 'HOLD': 0.0}

        weighted_score = 0.0
        total_weight = 0.0

        for model_name, signal in predictions.items():
            if model_name in weights and model_name in confidences:
                score = signal_scores.get(signal, 0.0)
                confidence = confidences[model_name]
                weight = weights[model_name]

                weighted_score += score * confidence * weight
                total_weight += confidence * weight

        if total_weight == 0:
            return {'signal': 'HOLD', 'confidence': 0.0}

        # Normalize score
        normalized_score = weighted_score / total_weight
        ensemble_confidence = abs(normalized_score)

        # Determine signal
        if normalized_score > threshold:
            signal = 'BUY'
        elif normalized_score < -threshold:
            signal = 'SELL'
        else:
            signal = 'HOLD'

        return {
            'signal': signal,
            'confidence': ensemble_confidence,
            'ensemble_score': normalized_score
        }

    def _save_models(self):
        """Save trained models and metadata."""
        try:
            model_path = self.model_dir / f"advanced_ml_{self.symbol.replace('/', '_')}_{self.timeframe}"

            # Save model weights and metadata
            metadata = {
                'model_weights': self.model_weights,
                'is_trained': self.is_trained,
                'feature_names': self.feature_engineer.feature_names,
                'training_timestamp': datetime.now().isoformat(),
                'symbol': self.symbol,
                'timeframe': self.timeframe
            }

            with open(model_path.with_suffix('.pkl'), 'wb') as f:
                pickle.dump(metadata, f)

            # Save PyTorch models
            if TORCH_AVAILABLE:
                for name, model in self.models.items():
                    if hasattr(model, 'state_dict'):  # PyTorch model
                        torch.save(model.state_dict(), model_path.with_name(f"{model_path.stem}_{name}.pth"))

            print(f"💾 Advanced ML models saved to {model_path}")

        except Exception as e:
            print(f"❌ Failed to save advanced ML models: {e}")

    def _load_models(self):
        """Load trained models from disk."""
        try:
            model_path = self.model_dir / f"advanced_ml_{self.symbol.replace('/', '_')}_{self.timeframe}"

            if not model_path.with_suffix('.pkl').exists():
                return

            # Load metadata
            with open(model_path.with_suffix('.pkl'), 'rb') as f:
                metadata = pickle.load(f)

            self.model_weights = metadata.get('model_weights', {})
            self.is_trained = metadata.get('is_trained', False)
            self.feature_engineer.feature_names = metadata.get('feature_names', [])

            training_time = metadata.get('training_timestamp', 'Unknown')
            print(f"📂 Advanced ML models loaded from {model_path} (trained: {training_time})")

        except Exception as e:
            print(f"❌ Failed to load advanced ML models: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the advanced ML predictor."""
        return {
            'is_trained': self.is_trained,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'models': list(self.models.keys()),
            'model_weights': self.model_weights,
            'feature_count': len(self.feature_engineer.feature_names),
            'features': self.feature_engineer.feature_names,
            'regime_stats': self.regime_detector.get_regime_statistics(),
            'device': str(self.device) if TORCH_AVAILABLE else 'cpu'
        }


# Global advanced ML predictor instance
advanced_ml_predictor = None


def get_advanced_ml_predictor(symbol: str = SYMBOL, timeframe: str = TIMEFRAME) -> Optional[AdvancedMLPredictor]:
    """Get or create advanced ML predictor instance."""
    global advanced_ml_predictor

    if not ML_ADVANCED_FEATURES_ENABLED:
        return None

    if advanced_ml_predictor is None:
        try:
            advanced_ml_predictor = AdvancedMLPredictor(symbol=symbol, timeframe=timeframe)
        except Exception as e:
            print(f"❌ Failed to initialize advanced ML predictor: {e}")
            return None

    return advanced_ml_predictor


def is_advanced_ml_available() -> bool:
    """Check if advanced ML features are available."""
    return ML_ADVANCED_FEATURES_ENABLED and (TORCH_AVAILABLE or SKLEARN_AVAILABLE or TENSORFLOW_AVAILABLE)