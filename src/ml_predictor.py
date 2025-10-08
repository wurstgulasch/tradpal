"""
Modular ML Prediction System for Trading Signals

Provides machine learning enhanced signal prediction using scikit-learn.
Supports multiple ML algorithms with automatic model selection and training.
Features confidence scoring and signal enhancement capabilities.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  scikit-learn not available. ML features will be disabled.")
    print("   Install with: pip install scikit-learn")

from config.settings import SYMBOL, TIMEFRAME, LOOKBACK_DAYS
from src.audit_logger import audit_logger


class MLSignalPredictor:
    """
    Machine Learning enhanced signal predictor for trading indicators.

    Features:
    - Multiple ML algorithms (Random Forest, Gradient Boosting, SVM, Logistic Regression)
    - Automatic model selection based on performance
    - Feature engineering from technical indicators
    - Confidence scoring for signal enhancement
    - Model persistence and retraining capabilities
    - Cross-validation for robust evaluation
    """

    def __init__(self, model_dir: str = "cache/ml_models", symbol: str = SYMBOL, timeframe: str = TIMEFRAME):
        """
        Initialize the ML signal predictor.

        Args:
            model_dir: Directory to store trained models
            symbol: Trading symbol for model naming
            timeframe: Timeframe for model naming
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for ML features. Install with: pip install scikit-learn")

        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.symbol = symbol
        self.timeframe = timeframe

        # Model configurations
        self.models = {
            'random_forest': {
                'model': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1
                ),
                'name': 'Random Forest'
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                ),
                'name': 'Gradient Boosting'
            },
            'svm': {
                'model': Pipeline([
                    ('scaler', StandardScaler()),
                    ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42))
                ]),
                'name': 'Support Vector Machine'
            },
            'logistic_regression': {
                'model': Pipeline([
                    ('scaler', StandardScaler()),
                    ('lr', LogisticRegression(random_state=42, max_iter=1000))
                ]),
                'name': 'Logistic Regression'
            }
        }

        self.best_model = None
        self.feature_columns = []
        self.model_performance = {}
        self.is_trained = False

        # Load existing model if available
        self.load_model()

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare features from technical indicators for ML training/prediction.

        Args:
            df: DataFrame with technical indicators

        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        features = []

        # Basic price features
        if 'close' in df.columns:
            features.append(df['close'].pct_change().fillna(0))  # Price change
            features.append(df['close'].rolling(window=5).mean().fillna(0))  # Short MA
            features.append(df['close'].rolling(window=20).mean().fillna(0))  # Long MA

        # Technical indicators
        indicator_features = [
            'EMA9', 'EMA21', 'RSI', 'BB_upper', 'BB_lower', 'ATR', 'ADX',
            'MACD', 'MACD_signal', 'MACD_hist', 'Stoch_K', 'Stoch_D'
        ]

        for indicator in indicator_features:
            if indicator in df.columns:
                # Add the indicator value
                features.append(df[indicator].fillna(0))
                # Add rate of change
                features.append(df[indicator].pct_change(fill_method=None).fillna(0))

        # Volatility features
        if 'ATR' in df.columns and 'close' in df.columns:
            features.append((df['ATR'] / df['close']).fillna(0))  # Normalized ATR

        # Trend features
        if 'EMA9' in df.columns and 'EMA21' in df.columns:
            features.append((df['EMA9'] - df['EMA21']).fillna(0))  # EMA spread
            features.append(((df['EMA9'] - df['EMA21']) / df['close']).fillna(0))  # Normalized spread

        # Momentum features
        if 'RSI' in df.columns:
            features.append(df['RSI'].rolling(window=5).mean().fillna(50))  # RSI MA

        # Create feature names
        feature_names = []
        base_indicators = ['close_pct', 'close_ma5', 'close_ma20'] + indicator_features
        for name in base_indicators:
            feature_names.extend([f"{name}_value", f"{name}_roc"])

        # Additional features
        feature_names.extend(['atr_normalized', 'ema_spread', 'ema_spread_norm', 'rsi_ma5'])

        # Combine features
        if features:
            feature_matrix = np.column_stack(features)
        else:
            # Fallback if no features available
            feature_matrix = np.zeros((len(df), 1))

        return feature_matrix, feature_names[:feature_matrix.shape[1]]

    def create_labels(self, df: pd.DataFrame, prediction_horizon: int = 5) -> np.ndarray:
        """
        Create labels for supervised learning based on future price movements.

        Args:
            df: DataFrame with price data
            prediction_horizon: Number of periods to look ahead for labeling

        Returns:
            Binary labels (1 for price increase, 0 for decrease)
        """
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'close' column for label creation")

        # Future price change
        future_price = df['close'].shift(-prediction_horizon)
        current_price = df['close']

        # Label: 1 if price will increase, 0 if decrease
        labels = (future_price > current_price).astype(int)

        # Remove NaN values from shifting
        labels = labels.fillna(0)

        return labels.values

    def train_models(self, df: pd.DataFrame, test_size: float = 0.2, prediction_horizon: int = 5) -> Dict[str, Any]:
        """
        Train all ML models and select the best performing one.

        Args:
            df: DataFrame with technical indicators and price data
            test_size: Fraction of data to use for testing
            prediction_horizon: Periods to look ahead for labeling

        Returns:
            Dictionary with training results and best model info
        """
        try:
            # Prepare features and labels
            X, feature_names = self.prepare_features(df)
            y = self.create_labels(df, prediction_horizon)

            # Remove rows with NaN
            valid_indices = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[valid_indices]
            y = y[valid_indices]

            if len(X) < 100:
                raise ValueError(f"Insufficient data for training: {len(X)} samples")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            self.feature_columns = feature_names

            # Train and evaluate all models
            results = {}
            best_score = 0
            best_model_name = None

            for model_name, model_config in self.models.items():
                try:
                    model = model_config['model']

                    # Train model
                    model.fit(X_train, y_train)

                    # Evaluate on test set
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)

                    # Cross-validation score
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')

                    results[model_name] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'cv_f1_mean': cv_scores.mean(),
                        'cv_f1_std': cv_scores.std(),
                        'model': model
                    }

                    # Select best model based on F1 score
                    if f1 > best_score:
                        best_score = f1
                        best_model_name = model_name

                    print(f"✅ {model_config['name']}: F1={f1:.3f}, CV-F1={cv_scores.mean():.3f}±{cv_scores.std():.3f}")

                except Exception as e:
                    print(f"❌ Failed to train {model_config['name']}: {e}")
                    results[model_name] = {'error': str(e)}

            if best_model_name:
                self.best_model = results[best_model_name]['model']
                self.model_performance = results
                self.is_trained = True

                # Save the best model
                self.save_model()

                # Log training completion
                audit_logger.log_system_event(
                    event_type="ML_TRAINING_COMPLETED",
                    message=f"ML model training completed for {self.symbol} {self.timeframe}",
                    details={
                        'best_model': best_model_name,
                        'f1_score': best_score,
                        'samples': len(X),
                        'features': len(feature_names)
                    }
                )

                return {
                    'success': True,
                    'best_model': best_model_name,
                    'performance': results,
                    'samples': len(X),
                    'features': len(feature_names)
                }
            else:
                raise ValueError("No model could be trained successfully")

        except Exception as e:
            error_msg = f"ML training failed: {str(e)}"
            print(f"❌ {error_msg}")

            audit_logger.log_error(
                error_type="ML_TRAINING_ERROR",
                message=error_msg,
                context={'symbol': self.symbol, 'timeframe': self.timeframe}
            )

            return {'success': False, 'error': str(e)}

    def predict_signal(self, df: pd.DataFrame, threshold: float = 0.6) -> Dict[str, Any]:
        """
        Predict trading signal using the trained ML model.

        Args:
            df: DataFrame with latest technical indicators
            threshold: Confidence threshold for signal generation

        Returns:
            Dictionary with prediction results
        """
        if not self.is_trained or self.best_model is None:
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'reason': 'Model not trained'
            }

        try:
            # Prepare features for prediction (use last row)
            X, _ = self.prepare_features(df)
            if len(X) == 0:
                return {
                    'signal': 'HOLD',
                    'confidence': 0.0,
                    'reason': 'No features available'
                }

            last_features = X[-1:].reshape(1, -1)

            # Get prediction and probability
            prediction = self.best_model.predict(last_features)[0]

            if hasattr(self.best_model, 'predict_proba'):
                prob = self.best_model.predict_proba(last_features)[0]
                confidence = max(prob)  # Highest probability
                predicted_class_prob = prob[1] if len(prob) > 1 else prob[0]
            else:
                confidence = 0.5  # Default confidence for models without predict_proba
                predicted_class_prob = 0.5

            # Determine signal based on prediction and threshold
            if predicted_class_prob > threshold:
                signal = 'BUY'
            elif predicted_class_prob < (1 - threshold):
                signal = 'SELL'
            else:
                signal = 'HOLD'

            return {
                'signal': signal,
                'confidence': float(confidence),
                'predicted_prob': float(predicted_class_prob),
                'threshold': threshold,
                'reason': f'ML prediction with {confidence:.2f} confidence'
            }

        except Exception as e:
            error_msg = f"ML prediction failed: {str(e)}"
            print(f"❌ {error_msg}")

            audit_logger.log_error(
                error_type="ML_PREDICTION_ERROR",
                message=error_msg,
                context={'symbol': self.symbol, 'timeframe': self.timeframe}
            )

            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'reason': f'Prediction error: {str(e)}'
            }

    def enhance_signal(self, traditional_signal: str, ml_prediction: Dict[str, Any],
                      confidence_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Enhance traditional signals with ML predictions.

        Args:
            traditional_signal: Signal from traditional indicators ('BUY', 'SELL', 'HOLD')
            ml_prediction: ML prediction result
            confidence_threshold: Minimum confidence to override traditional signal

        Returns:
            Enhanced signal decision
        """
        ml_signal = ml_prediction['signal']
        ml_confidence = ml_prediction['confidence']

        # If ML has high confidence and differs from traditional signal, use ML
        if ml_confidence >= confidence_threshold and ml_signal != traditional_signal:
            return {
                'signal': ml_signal,
                'source': 'ML_ENHANCED',
                'confidence': ml_confidence,
                'traditional_signal': traditional_signal,
                'reason': f'ML override: {ml_confidence:.2f} confidence vs traditional {traditional_signal}'
            }

        # If signals agree, boost confidence
        elif ml_signal == traditional_signal and ml_signal != 'HOLD':
            boosted_confidence = min(1.0, ml_confidence + 0.2)  # Boost by 20%
            return {
                'signal': traditional_signal,
                'source': 'CONFIRMED',
                'confidence': boosted_confidence,
                'traditional_signal': traditional_signal,
                'reason': f'Signals confirmed: ML confidence {ml_confidence:.2f}'
            }

        # Default to traditional signal
        else:
            return {
                'signal': traditional_signal,
                'source': 'TRADITIONAL',
                'confidence': 0.5,
                'traditional_signal': traditional_signal,
                'ml_signal': ml_signal,
                'reason': f'Using traditional signal: {traditional_signal}'
            }

    def save_model(self):
        """Save the trained model and metadata to disk."""
        if not self.is_trained or self.best_model is None:
            return

        model_data = {
            'model': self.best_model,
            'feature_columns': self.feature_columns,
            'model_performance': self.model_performance,
            'training_timestamp': datetime.now().isoformat(),
            'symbol': self.symbol,
            'timeframe': self.timeframe
        }

        model_file = self.model_dir / f"ml_model_{self.symbol.replace('/', '_')}_{self.timeframe}.pkl"

        try:
            with open(model_file, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"💾 ML model saved to {model_file}")
        except Exception as e:
            print(f"❌ Failed to save ML model: {e}")

    def load_model(self):
        """Load a previously trained model from disk."""
        model_file = self.model_dir / f"ml_model_{self.symbol.replace('/', '_')}_{self.timeframe}.pkl"

        if not model_file.exists():
            return

        try:
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)

            self.best_model = model_data['model']
            self.feature_columns = model_data.get('feature_columns', [])
            self.model_performance = model_data.get('model_performance', {})
            self.is_trained = True

            training_time = model_data.get('training_timestamp', 'Unknown')
            print(f"📂 ML model loaded from {model_file} (trained: {training_time})")

        except Exception as e:
            print(f"❌ Failed to load ML model: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        if not self.is_trained:
            return {'status': 'not_trained'}

        return {
            'status': 'trained',
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'feature_count': len(self.feature_columns),
            'features': self.feature_columns,
            'performance': self.model_performance
        }

    def retrain_model(self, df: pd.DataFrame, force: bool = False) -> Dict[str, Any]:
        """
        Retrain the model with new data.

        Args:
            df: New DataFrame for training
            force: Force retraining even if model exists

        Returns:
            Training results
        """
        if not force and self.is_trained:
            return {'success': False, 'message': 'Model already trained. Use force=True to retrain.'}

        print(f"🔄 Retraining ML model for {self.symbol} {self.timeframe}...")
        return self.train_models(df)


# Global ML predictor instance
ml_predictor = None

def get_ml_predictor(symbol: str = SYMBOL, timeframe: str = TIMEFRAME) -> Optional[MLSignalPredictor]:
    """Get or create ML predictor instance."""
    global ml_predictor

    if ml_predictor is None and SKLEARN_AVAILABLE:
        try:
            ml_predictor = MLSignalPredictor(symbol=symbol, timeframe=timeframe)
        except Exception as e:
            print(f"❌ Failed to initialize ML predictor: {e}")
            return None

    return ml_predictor

def is_ml_available() -> bool:
    """Check if ML features are available."""
    return SKLEARN_AVAILABLE