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

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None
    print("⚠️  XGBoost not available. Enhanced ensemble methods will be limited.")
    print("   Install with: pip install xgboost")

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
            'xgboost': {
                'model': xgb.XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42,
                    n_jobs=-1
                ) if XGBOOST_AVAILABLE else None,
                'name': 'XGBoost'
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
                # Skip XGBoost if not available
                if model_name == 'xgboost' and not XGBOOST_AVAILABLE:
                    continue

                try:
                    model = model_config['model']
                    if model is None:
                        continue

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

                # Try to create stacking ensemble if multiple models are available
                self.stacking_ensemble = None
                if len([m for m in results.keys() if 'error' not in results[m]]) >= 3:
                    self.create_stacking_ensemble(X_train, X_test, y_train, y_test)

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

    def create_stacking_ensemble(self, X_train, X_test, y_train, y_test):
        """
        Create a stacking ensemble using the trained base models.

        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test labels
        """
        try:
            from sklearn.ensemble import StackingClassifier

            # Get successful base models
            base_models = []
            for model_name, result in self.model_performance.items():
                if 'error' not in result and 'model' in result:
                    base_models.append((model_name, result['model']))

            if len(base_models) < 3:
                return  # Not enough models for stacking

            # Create stacking ensemble
            self.stacking_ensemble = StackingClassifier(
                estimators=base_models,
                final_estimator=LogisticRegression(random_state=42, max_iter=1000),
                cv=5,
                n_jobs=-1
            )

            # Train stacking ensemble
            self.stacking_ensemble.fit(X_train, y_train)

            # Evaluate stacking ensemble
            stacking_pred = self.stacking_ensemble.predict(X_test)
            stacking_f1 = f1_score(y_test, stacking_pred, zero_division=0)

            # Add to performance results
            self.model_performance['stacking_ensemble'] = {
                'accuracy': accuracy_score(y_test, stacking_pred),
                'precision': precision_score(y_test, stacking_pred, zero_division=0),
                'recall': recall_score(y_test, stacking_pred, zero_division=0),
                'f1_score': stacking_f1,
                'model': self.stacking_ensemble
            }

            # Use stacking if it's better than the best individual model
            best_f1 = max([result.get('f1_score', 0) for result in self.model_performance.values()
                          if isinstance(result, dict) and 'f1_score' in result])

            if stacking_f1 > best_f1:
                self.best_model = self.stacking_ensemble
                print(f"✅ Stacking ensemble selected (F1={stacking_f1:.3f} > {best_f1:.3f})")
            else:
                print(f"ℹ️  Stacking ensemble trained (F1={stacking_f1:.3f}) but individual model kept")

        except Exception as e:
            print(f"⚠️  Stacking ensemble creation failed: {e}")
            self.stacking_ensemble = None

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
                'reason': f'ML prediction with {confidence:.2f} confidence',
                'explanation': self.explain_prediction(df) if SHAP_AVAILABLE else None
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
                      confidence_threshold: float = 0.7, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
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
            explanation = self.explain_prediction(df) if SHAP_AVAILABLE and df is not None else None
            return {
                'signal': traditional_signal,
                'source': 'TRADITIONAL',
                'confidence': 0.5,
                'traditional_signal': traditional_signal,
                'ml_signal': ml_signal,
                'reason': f'Using traditional signal: {traditional_signal}',
                'explanation': explanation
            }

    def explain_prediction(self, df: pd.DataFrame, background_samples: int = 100) -> Optional[Dict[str, Any]]:
        """
        Explain the ML model prediction using SHAP values.

        Args:
            df: DataFrame with technical indicators
            background_samples: Number of background samples for SHAP

        Returns:
            Dictionary with SHAP explanations or None if not available
        """
        if not SHAP_AVAILABLE or not self.is_trained or self.best_model is None:
            return None

        try:
            # Prepare features
            X, feature_names = self.prepare_features(df)

            if len(X) < background_samples:
                return None

            # Use recent data for background
            background_data = X[-background_samples:]

            # Create SHAP explainer - simplified approach
            try:
                if 'RandomForest' in str(type(self.best_model)) or 'GradientBoosting' in str(type(self.best_model)):
                    explainer = shap.TreeExplainer(self.best_model, background_data)
                elif 'XGB' in str(type(self.best_model)):
                    explainer = shap.TreeExplainer(self.best_model, background_data)
                else:
                    # For other models, use a simpler approach
                    background_subset = background_data[:min(50, len(background_data))]
                    explainer = shap.KernelExplainer(self.best_model.predict_proba, background_subset)
            except Exception:
                # Fallback: skip SHAP if explainer creation fails
                return None

            # Explain the last sample
            last_sample = X[-1:].reshape(1, -1)

            try:
                shap_values = explainer.shap_values(last_sample)
            except Exception:
                return None

            # Simplified feature importance extraction
            try:
                # Handle different SHAP output formats
                if isinstance(shap_values, list):
                    if len(shap_values) > 1:
                        # Multi-class, take positive class
                        shap_vals = shap_values[1]
                    else:
                        shap_vals = shap_values[0]
                else:
                    shap_vals = shap_values

                # Extract feature importance
                if hasattr(shap_vals, 'shape') and len(shap_vals.shape) > 0:
                    if len(shap_vals.shape) > 1:
                        importance_vals = np.abs(shap_vals).mean(axis=0)  # Average across samples if needed
                    else:
                        importance_vals = np.abs(shap_vals)

                    # Ensure we have the right number of features
                    if len(importance_vals) == len(feature_names):
                        importance_dict = dict(zip(feature_names, importance_vals))
                    else:
                        # Create generic names
                        importance_dict = {f'feature_{i}': float(val) for i, val in enumerate(importance_vals[:len(feature_names)])}
                else:
                    importance_dict = {'unknown': 0.0}

                # Get top features
                top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]

                return {
                    'feature_importance': importance_dict,
                    'top_features': top_features,
                    'shap_available': True
                }

            except Exception:
                # If feature extraction fails, return basic info
                return {
                    'feature_importance': {'shap_error': 0.0},
                    'top_features': [('shap_error', 0.0)],
                    'shap_available': False
                }

        except Exception as e:
            print(f"⚠️  SHAP explanation failed: {e}")
            return None

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
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    Sequential = None  # Define as None when not available
    print("⚠️  TensorFlow not available. LSTM features will be disabled.")
    print("   Install with: pip install tensorflow")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP not available. Model explanations will be disabled.")
    print("   Install with: pip install shap")


if TENSORFLOW_AVAILABLE:
    class LSTMSignalPredictor:
        """
        LSTM-based neural network predictor for trading signals.

        Features:
        - Bidirectional LSTM layers for sequence modeling
        - Time-series optimized feature engineering
        - Early stopping and model checkpointing
        - SHAP-based model interpretability
        - Confidence scoring and signal enhancement
        """

        def __init__(self, model_dir: str = "cache/ml_models", symbol: str = SYMBOL, timeframe: str = TIMEFRAME,
                     sequence_length: int = 60, lstm_units: int = 64, dropout_rate: float = 0.2):
            """
            Initialize the LSTM signal predictor.

            Args:
                model_dir: Directory to store trained models
                symbol: Trading symbol for model naming
                timeframe: Timeframe for model naming
                sequence_length: Number of time steps for LSTM input
                lstm_units: Number of LSTM units in each layer
                dropout_rate: Dropout rate for regularization
            """
            if not TENSORFLOW_AVAILABLE:
                raise ImportError("TensorFlow is required for LSTM features. Install with: pip install tensorflow")

            self.model_dir = Path(model_dir)
            self.model_dir.mkdir(parents=True, exist_ok=True)
            self.symbol = symbol
            self.timeframe = timeframe
            self.sequence_length = sequence_length
            self.lstm_units = lstm_units
            self.dropout_rate = dropout_rate

            self.model = None
            self.scaler = StandardScaler()
            self.feature_columns = []
            self.is_trained = False
            self.training_history = {}

            # Load existing model if available
            self.load_model()

        def prepare_sequences(self, df: pd.DataFrame, prediction_horizon: int = 5) -> Tuple[np.ndarray, np.ndarray]:
            """
            Prepare time sequences for LSTM training.

            Args:
                df: DataFrame with technical indicators
                prediction_horizon: Periods to look ahead for labeling

            Returns:
                Tuple of (X_sequences, y_labels)
            """
            # Prepare features
            features, feature_names = self.prepare_features(df)
            self.feature_columns = feature_names

            # Create labels
            labels = self.create_labels(df, prediction_horizon)

            # Create sequences
            sequences = []
            sequence_labels = []

            for i in range(self.sequence_length, len(features)):
                sequences.append(features[i-self.sequence_length:i])
                sequence_labels.append(labels[i])

            X = np.array(sequences)
            y = np.array(sequence_labels)

            # Remove NaN sequences
            valid_indices = ~(np.isnan(X).any(axis=(1, 2)) | np.isnan(y))
            X = X[valid_indices]
            y = y[valid_indices]

            return X, y

        def build_model(self, input_shape: Tuple[int, int]):
            """
            Build the LSTM neural network model.

            Args:
                input_shape: Shape of input sequences (sequence_length, n_features)

            Returns:
                Compiled Keras model
            """
            model = Sequential([
                Bidirectional(LSTM(self.lstm_units, return_sequences=True, input_shape=input_shape)),
                Dropout(self.dropout_rate),
                Bidirectional(LSTM(self.lstm_units // 2)),
                Dropout(self.dropout_rate),
                Dense(32, activation='relu'),
                Dropout(self.dropout_rate / 2),
                Dense(1, activation='sigmoid')
            ])

            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC()]
            )

            return model

        def train_model(self, df: pd.DataFrame, test_size: float = 0.2, prediction_horizon: int = 5,
                       epochs: int = 100, batch_size: int = 32, validation_split: float = 0.1) -> Dict[str, Any]:
            """
            Train the LSTM model.

            Args:
                df: DataFrame with technical indicators and price data
                test_size: Fraction of data for testing
                prediction_horizon: Periods to look ahead for labeling
                epochs: Maximum number of training epochs
                batch_size: Batch size for training
                validation_split: Fraction of training data for validation

            Returns:
                Dictionary with training results
            """
            try:
                # Prepare sequences
                X, y = self.prepare_sequences(df, prediction_horizon)

                if len(X) < 100:
                    raise ValueError(f"Insufficient data for LSTM training: {len(X)} sequences")

                # Split data
                split_idx = int(len(X) * (1 - test_size))
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]

                print(f"📊 LSTM Training Data: {len(X_train)} train, {len(X_test)} test sequences")

                # Build model
                self.model = self.build_model((self.sequence_length, X.shape[2]))

                # Callbacks
                callbacks = [
                    EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True,
                        verbose=1
                    ),
                    ModelCheckpoint(
                        filepath=str(self.model_dir / f"lstm_model_{self.symbol.replace('/', '_')}_{self.timeframe}.h5"),
                        monitor='val_loss',
                        save_best_only=True,
                        verbose=1
                    )
                ]

                # Train model
                history = self.model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=validation_split,
                    callbacks=callbacks,
                    verbose=1
                )

                # Evaluate on test set
                test_loss, test_accuracy, test_auc = self.model.evaluate(X_test, y_test, verbose=0)

                # Store training history
                self.training_history = {
                    'loss': history.history['loss'],
                    'val_loss': history.history['val_loss'],
                    'accuracy': history.history['accuracy'],
                    'val_accuracy': history.history['val_accuracy'],
                    'test_loss': test_loss,
                    'test_accuracy': test_accuracy,
                    'test_auc': test_auc
                }

                self.is_trained = True
                self.save_model()

                # Log training completion
                audit_logger.log_system_event(
                    event_type="LSTM_TRAINING_COMPLETED",
                    message=f"LSTM model training completed for {self.symbol} {self.timeframe}",
                    details={
                        'test_accuracy': test_accuracy,
                        'test_auc': test_auc,
                        'sequences': len(X),
                        'features': len(self.feature_columns),
                        'epochs_trained': len(history.history['loss'])
                    }
                )

                print(f"✅ LSTM Model trained: Test Accuracy={test_accuracy:.3f}, AUC={test_auc:.3f}")

                return {
                    'success': True,
                    'test_accuracy': test_accuracy,
                    'test_auc': test_auc,
                    'sequences': len(X),
                    'features': len(self.feature_columns),
                    'history': self.training_history
                }

            except Exception as e:
                error_msg = f"LSTM training failed: {str(e)}"
                print(f"❌ {error_msg}")

                audit_logger.log_error(
                    error_type="LSTM_TRAINING_ERROR",
                    message=error_msg,
                    context={'symbol': self.symbol, 'timeframe': self.timeframe}
                )

                return {'success': False, 'error': str(e)}

        def predict_signal(self, df: pd.DataFrame, threshold: float = 0.6) -> Dict[str, Any]:
            """
            Predict trading signal using the trained LSTM model.

            Args:
                df: DataFrame with latest technical indicators
                threshold: Confidence threshold for signal generation

            Returns:
                Dictionary with prediction results
            """
            if not self.is_trained or self.model is None:
                return {
                    'signal': 'HOLD',
                    'confidence': 0.0,
                    'reason': 'LSTM model not trained'
                }

            try:
                # Prepare features for the last sequence
                features, _ = self.prepare_features(df)

                if len(features) < self.sequence_length:
                    return {
                        'signal': 'HOLD',
                        'confidence': 0.0,
                        'reason': f'Insufficient data: need {self.sequence_length} periods, got {len(features)}'
                    }

                # Get the last sequence
                last_sequence = features[-self.sequence_length:].reshape(1, self.sequence_length, -1)

                # Make prediction
                prediction_prob = self.model.predict(last_sequence, verbose=0)[0][0]
                confidence = max(prediction_prob, 1 - prediction_prob)  # Highest probability

                # Determine signal
                if prediction_prob > threshold:
                    signal = 'BUY'
                elif prediction_prob < (1 - threshold):
                    signal = 'SELL'
                else:
                    signal = 'HOLD'

                return {
                    'signal': signal,
                    'confidence': float(confidence),
                    'predicted_prob': float(prediction_prob),
                    'threshold': threshold,
                    'reason': f'LSTM prediction with {confidence:.2f} confidence'
                }

            except Exception as e:
                error_msg = f"LSTM prediction failed: {str(e)}"
                print(f"❌ {error_msg}")

                audit_logger.log_error(
                    error_type="LSTM_PREDICTION_ERROR",
                    message=error_msg,
                    context={'symbol': self.symbol, 'timeframe': self.timeframe}
                )

                return {
                    'signal': 'HOLD',
                    'confidence': 0.0,
                    'reason': f'Prediction error: {str(e)}'
                }

        def explain_prediction(self, df: pd.DataFrame, background_samples: int = 100) -> Optional[Dict[str, Any]]:
            """
            Explain the LSTM prediction using SHAP values.

            Args:
                df: DataFrame with technical indicators
                background_samples: Number of background samples for SHAP

            Returns:
                Dictionary with SHAP explanations or None if not available
            """
            if not SHAP_AVAILABLE or not self.is_trained:
                return None

            try:
                # Prepare background data
                features, feature_names = self.prepare_features(df)

                if len(features) < background_samples + self.sequence_length:
                    return None

                # Use recent data for background
                background_data = features[-background_samples-self.sequence_length:-self.sequence_length]
                background_sequences = []

                for i in range(len(background_data) - self.sequence_length + 1):
                    background_sequences.append(background_data[i:i+self.sequence_length])

                background_sequences = np.array(background_sequences)

                # Create SHAP explainer
                def model_predict(sequences):
                    return self.model.predict(sequences, verbose=0).flatten()

                explainer = shap.DeepExplainer(model_predict, background_sequences)

                # Explain the last sequence
                last_sequence = features[-self.sequence_length:].reshape(1, self.sequence_length, -1)
                shap_values = explainer.shap_values(last_sequence)

                # Aggregate SHAP values across time steps for each feature
                feature_importance = np.abs(shap_values[0]).mean(axis=0)  # Mean absolute SHAP across time steps

                # Create feature importance dictionary
                importance_dict = dict(zip(feature_names, feature_importance))

                return {
                    'feature_importance': importance_dict,
                    'top_features': sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10],
                    'shap_values': shap_values[0].tolist()
                }

            except Exception as e:
                print(f"⚠️  SHAP explanation failed: {e}")
                return None

        def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
            """
            Prepare features from technical indicators for LSTM training/prediction.

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

        def save_model(self):
            """Save the trained LSTM model and metadata to disk."""
            if not self.is_trained or self.model is None:
                return

            model_file = self.model_dir / f"lstm_model_{self.symbol.replace('/', '_')}_{self.timeframe}"

            try:
                # Save Keras model
                self.model.save(model_file.with_suffix('.h5'))

                # Save metadata
                metadata = {
                    'scaler': self.scaler,
                    'feature_columns': self.feature_columns,
                    'training_history': self.training_history,
                    'sequence_length': self.sequence_length,
                    'lstm_units': self.lstm_units,
                    'dropout_rate': self.dropout_rate,
                    'training_timestamp': datetime.now().isoformat(),
                    'symbol': self.symbol,
                    'timeframe': self.timeframe
                }

                with open(model_file.with_suffix('.pkl'), 'wb') as f:
                    pickle.dump(metadata, f)

                print(f"💾 LSTM model saved to {model_file}")

            except Exception as e:
                print(f"❌ Failed to save LSTM model: {e}")

        def load_model(self):
            """Load a previously trained LSTM model from disk."""
            model_file = self.model_dir / f"lstm_model_{self.symbol.replace('/', '_')}_{self.timeframe}"

            if not model_file.with_suffix('.h5').exists():
                return

            try:
                # Load Keras model
                self.model = keras.models.load_model(model_file.with_suffix('.h5'))

                # Load metadata
                with open(model_file.with_suffix('.pkl'), 'rb') as f:
                    metadata = pickle.load(f)

                self.scaler = metadata.get('scaler', StandardScaler())
                self.feature_columns = metadata.get('feature_columns', [])
                self.training_history = metadata.get('training_history', {})
                self.sequence_length = metadata.get('sequence_length', 60)
                self.lstm_units = metadata.get('lstm_units', 64)
                self.dropout_rate = metadata.get('dropout_rate', 0.2)
                self.is_trained = True

                training_time = metadata.get('training_timestamp', 'Unknown')
                print(f"📂 LSTM model loaded from {model_file} (trained: {training_time})")

            except Exception as e:
                print(f"❌ Failed to load LSTM model: {e}")

        def get_model_info(self) -> Dict[str, Any]:
            """Get information about the current LSTM model."""
            if not self.is_trained:
                return {'status': 'not_trained'}

            return {
                'status': 'trained',
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'sequence_length': self.sequence_length,
                'lstm_units': self.lstm_units,
                'dropout_rate': self.dropout_rate,
                'feature_count': len(self.feature_columns),
                'features': self.feature_columns,
                'training_history': self.training_history
            }

if TENSORFLOW_AVAILABLE:
    class TransformerSignalPredictor:
        """
        Transformer-based neural network predictor for trading signals.

        Features:
        - Multi-head self-attention mechanism
        - Positional encoding for sequence modeling
        - Time-series optimized feature engineering
        - SHAP-based model interpretability
        - Confidence scoring and signal enhancement
        """

        def __init__(self, model_dir: str = "cache/ml_models", symbol: str = SYMBOL, timeframe: str = TIMEFRAME,
                     sequence_length: int = 60, num_heads: int = 8, ff_dim: int = 128, num_transformer_blocks: int = 4,
                     dropout_rate: float = 0.1):
            """
            Initialize the Transformer signal predictor.

            Args:
                model_dir: Directory to store trained models
                symbol: Trading symbol for model naming
                timeframe: Timeframe for model naming
                sequence_length: Number of time steps for input
                num_heads: Number of attention heads
                ff_dim: Feed-forward dimension
                num_transformer_blocks: Number of transformer blocks
                dropout_rate: Dropout rate for regularization
            """
            if not TENSORFLOW_AVAILABLE:
                raise ImportError("TensorFlow is required for Transformer features. Install with: pip install tensorflow")

            self.model_dir = Path(model_dir)
            self.model_dir.mkdir(parents=True, exist_ok=True)
            self.symbol = symbol
            self.timeframe = timeframe
            self.sequence_length = sequence_length
            self.num_heads = num_heads
            self.ff_dim = ff_dim
            self.num_transformer_blocks = num_transformer_blocks
            self.dropout_rate = dropout_rate

            self.model = None
            self.scaler = StandardScaler()
            self.feature_columns = []
            self.is_trained = False
            self.training_history = {}

            # Load existing model if available
            self.load_model()

        def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
            """
            Transformer encoder block.

            Args:
                inputs: Input tensor
                head_size: Size of attention heads
                num_heads: Number of attention heads
                ff_dim: Feed-forward dimension
                dropout: Dropout rate

            Returns:
                Encoded tensor
            """
            # Multi-head attention
            x = tf.keras.layers.MultiHeadAttention(
                key_dim=head_size, num_heads=num_heads, dropout=dropout
            )(inputs, inputs)
            x = tf.keras.layers.Dropout(dropout)(x)
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
            res = x + inputs

            # Feed-forward
            x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
            x = tf.keras.layers.Dropout(dropout)(x)
            x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

            return x + res

        def build_model(self, input_shape):
            """
            Build the Transformer neural network model.

            Args:
                input_shape: Shape of input sequences (sequence_length, n_features)

            Returns:
                Compiled Keras model
            """
            inputs = tf.keras.Input(shape=input_shape)

            # Positional encoding
            positions = tf.range(start=0, limit=input_shape[0], delta=1)
            positional_encoding = self.get_positional_encoding(input_shape[0], input_shape[1])
            x = inputs + positional_encoding

            # Transformer blocks
            for _ in range(self.num_transformer_blocks):
                x = self.transformer_encoder(x, self.ff_dim // self.num_heads, self.num_heads, self.ff_dim, self.dropout_rate)

            # Global average pooling
            x = tf.keras.layers.GlobalAveragePooling1D()(x)

            # Output layers
            x = tf.keras.layers.Dense(64, activation="relu")(x)
            x = tf.keras.layers.Dropout(self.dropout_rate)(x)
            outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

            model = tf.keras.Model(inputs=inputs, outputs=outputs)

            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC()]
            )

            return model

        def get_positional_encoding(self, seq_len, d_model):
            """
            Generate positional encoding for transformer.

            Args:
                seq_len: Sequence length
                d_model: Model dimension

            Returns:
                Positional encoding tensor
            """
            positions = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
            i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]

            angle_rates = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
            angle_rads = positions * angle_rates

            # Apply sin to even indices, cos to odd indices
            sines = tf.sin(angle_rads[:, 0::2])
            cosines = tf.cos(angle_rads[:, 1::2])

            pos_encoding = tf.concat([sines, cosines], axis=-1)
            pos_encoding = pos_encoding[tf.newaxis, ...]

            return pos_encoding

        def prepare_sequences(self, df: pd.DataFrame, prediction_horizon: int = 5) -> Tuple[np.ndarray, np.ndarray]:
            """
            Prepare time sequences for Transformer training.

            Args:
                df: DataFrame with technical indicators
                prediction_horizon: Periods to look ahead for labeling

            Returns:
                Tuple of (X_sequences, y_labels)
            """
            # Prepare features
            features, feature_names = self.prepare_features(df)
            self.feature_columns = feature_names

            # Create labels
            labels = self.create_labels(df, prediction_horizon)

            # Create sequences
            sequences = []
            sequence_labels = []

            for i in range(self.sequence_length, len(features)):
                sequences.append(features[i-self.sequence_length:i])
                sequence_labels.append(labels[i])

            X = np.array(sequences)
            y = np.array(sequence_labels)

            # Remove NaN sequences
            valid_indices = ~(np.isnan(X).any(axis=(1, 2)) | np.isnan(y))
            X = X[valid_indices]
            y = y[valid_indices]

            return X, y

        def train_model(self, df: pd.DataFrame, test_size: float = 0.2, prediction_horizon: int = 5,
                       epochs: int = 100, batch_size: int = 32, validation_split: float = 0.1) -> Dict[str, Any]:
            """
            Train the Transformer model.

            Args:
                df: DataFrame with technical indicators and price data
                test_size: Fraction of data for testing
                prediction_horizon: Periods to look ahead for labeling
                epochs: Maximum number of training epochs
                batch_size: Batch size for training
                validation_split: Fraction of training data for validation

            Returns:
                Dictionary with training results
            """
            try:
                # Prepare sequences
                X, y = self.prepare_sequences(df, prediction_horizon)

                if len(X) < 100:
                    raise ValueError(f"Insufficient data for Transformer training: {len(X)} sequences")

                # Split data
                split_idx = int(len(X) * (1 - test_size))
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]

                print(f"📊 Transformer Training Data: {len(X_train)} train, {len(X_test)} test sequences")

                # Build model
                self.model = self.build_model((self.sequence_length, X.shape[2]))

                # Callbacks
                callbacks = [
                    EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True,
                        verbose=1
                    ),
                    ModelCheckpoint(
                        filepath=str(self.model_dir / f"transformer_model_{self.symbol.replace('/', '_')}_{self.timeframe}.h5"),
                        monitor='val_loss',
                        save_best_only=True,
                        verbose=1
                    )
                ]

                # Train model
                history = self.model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=validation_split,
                    callbacks=callbacks,
                    verbose=1
                )

                # Evaluate on test set
                test_loss, test_accuracy, test_auc = self.model.evaluate(X_test, y_test, verbose=0)

                # Store training history
                self.training_history = {
                    'loss': history.history['loss'],
                    'val_loss': history.history['val_loss'],
                    'accuracy': history.history['accuracy'],
                    'val_accuracy': history.history['val_accuracy'],
                    'test_loss': test_loss,
                    'test_accuracy': test_accuracy,
                    'test_auc': test_auc
                }

                self.is_trained = True
                self.save_model()

                # Log training completion
                audit_logger.log_system_event(
                    event_type="TRANSFORMER_TRAINING_COMPLETED",
                    message=f"Transformer model training completed for {self.symbol} {self.timeframe}",
                    details={
                        'test_accuracy': test_accuracy,
                        'test_auc': test_auc,
                        'sequences': len(X),
                        'features': len(self.feature_columns),
                        'epochs_trained': len(history.history['loss'])
                    }
                )

                print(f"✅ Transformer Model trained: Test Accuracy={test_accuracy:.3f}, AUC={test_auc:.3f}")

                return {
                    'success': True,
                    'test_accuracy': test_accuracy,
                    'test_auc': test_auc,
                    'sequences': len(X),
                    'features': len(self.feature_columns),
                    'history': self.training_history
                }

            except Exception as e:
                error_msg = f"Transformer training failed: {str(e)}"
                print(f"❌ {error_msg}")

                audit_logger.log_error(
                    error_type="TRANSFORMER_TRAINING_ERROR",
                    message=error_msg,
                    context={'symbol': self.symbol, 'timeframe': self.timeframe}
                )

                return {'success': False, 'error': str(e)}

        def predict_signal(self, df: pd.DataFrame, threshold: float = 0.6) -> Dict[str, Any]:
            """
            Predict trading signal using the trained Transformer model.

            Args:
                df: DataFrame with latest technical indicators
                threshold: Confidence threshold for signal generation

            Returns:
                Dictionary with prediction results
            """
            if not self.is_trained or self.model is None:
                return {
                    'signal': 'HOLD',
                    'confidence': 0.0,
                    'reason': 'Transformer model not trained'
                }

            try:
                # Prepare features for the last sequence
                features, _ = self.prepare_features(df)

                if len(features) < self.sequence_length:
                    return {
                        'signal': 'HOLD',
                        'confidence': 0.0,
                        'reason': f'Insufficient data: need {self.sequence_length} periods, got {len(features)}'
                    }

                # Get the last sequence
                last_sequence = features[-self.sequence_length:].reshape(1, self.sequence_length, -1)

                # Make prediction
                prediction_prob = self.model.predict(last_sequence, verbose=0)[0][0]
                confidence = max(prediction_prob, 1 - prediction_prob)  # Highest probability

                # Determine signal
                if prediction_prob > threshold:
                    signal = 'BUY'
                elif prediction_prob < (1 - threshold):
                    signal = 'SELL'
                else:
                    signal = 'HOLD'

                return {
                    'signal': signal,
                    'confidence': float(confidence),
                    'predicted_prob': float(prediction_prob),
                    'threshold': threshold,
                    'reason': f'Transformer prediction with {confidence:.2f} confidence'
                }

            except Exception as e:
                error_msg = f"Transformer prediction failed: {str(e)}"
                print(f"❌ {error_msg}")

                audit_logger.log_error(
                    error_type="TRANSFORMER_PREDICTION_ERROR",
                    message=error_msg,
                    context={'symbol': self.symbol, 'timeframe': self.timeframe}
                )

                return {
                    'signal': 'HOLD',
                    'confidence': 0.0,
                    'reason': f'Prediction error: {str(e)}'
                }

        def explain_prediction(self, df: pd.DataFrame, background_samples: int = 100) -> Optional[Dict[str, Any]]:
            """
            Explain the Transformer prediction using SHAP values.

            Args:
                df: DataFrame with technical indicators
                background_samples: Number of background samples for SHAP

            Returns:
                Dictionary with SHAP explanations or None if not available
            """
            if not SHAP_AVAILABLE or not self.is_trained:
                return None

            try:
                # Prepare background data
                features, feature_names = self.prepare_features(df)

                if len(features) < background_samples + self.sequence_length:
                    return None

                # Use recent data for background
                background_data = features[-background_samples-self.sequence_length:-self.sequence_length]
                background_sequences = []

                for i in range(len(background_data) - self.sequence_length + 1):
                    background_sequences.append(background_data[i:i+self.sequence_length])

                background_sequences = np.array(background_sequences)

                # Create SHAP explainer
                def model_predict(sequences):
                    return self.model.predict(sequences, verbose=0).flatten()

                explainer = shap.DeepExplainer(model_predict, background_sequences)

                # Explain the last sequence
                last_sequence = features[-self.sequence_length:].reshape(1, self.sequence_length, -1)
                shap_values = explainer.shap_values(last_sequence)

                # Aggregate SHAP values across time steps for each feature
                feature_importance = np.abs(shap_values[0]).mean(axis=0)  # Mean absolute SHAP across time steps

                # Create feature importance dictionary
                importance_dict = dict(zip(feature_names, feature_importance))

                return {
                    'feature_importance': importance_dict,
                    'top_features': sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10],
                    'shap_values': shap_values[0].tolist()
                }

            except Exception as e:
                print(f"⚠️  SHAP explanation failed: {e}")
                return None

        def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
            """
            Prepare features from technical indicators for Transformer training/prediction.

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

        def save_model(self):
            """Save the trained Transformer model and metadata to disk."""
            if not self.is_trained or self.model is None:
                return

            model_file = self.model_dir / f"transformer_model_{self.symbol.replace('/', '_')}_{self.timeframe}"

            try:
                # Save Keras model
                self.model.save(model_file.with_suffix('.h5'))

                # Save metadata
                metadata = {
                    'scaler': self.scaler,
                    'feature_columns': self.feature_columns,
                    'training_history': self.training_history,
                    'sequence_length': self.sequence_length,
                    'num_heads': self.num_heads,
                    'ff_dim': self.ff_dim,
                    'num_transformer_blocks': self.num_transformer_blocks,
                    'dropout_rate': self.dropout_rate,
                    'training_timestamp': datetime.now().isoformat(),
                    'symbol': self.symbol,
                    'timeframe': self.timeframe
                }

                with open(model_file.with_suffix('.pkl'), 'wb') as f:
                    pickle.dump(metadata, f)

                print(f"💾 Transformer model saved to {model_file}")

            except Exception as e:
                print(f"❌ Failed to save Transformer model: {e}")

        def load_model(self):
            """Load a previously trained Transformer model from disk."""
            model_file = self.model_dir / f"transformer_model_{self.symbol.replace('/', '_')}_{self.timeframe}"

            if not model_file.with_suffix('.h5').exists():
                return

            try:
                # Load Keras model
                self.model = keras.models.load_model(model_file.with_suffix('.h5'))

                # Load metadata
                with open(model_file.with_suffix('.pkl'), 'rb') as f:
                    metadata = pickle.load(f)

                self.scaler = metadata.get('scaler', StandardScaler())
                self.feature_columns = metadata.get('feature_columns', [])
                self.training_history = metadata.get('training_history', {})
                self.sequence_length = metadata.get('sequence_length', 60)
                self.num_heads = metadata.get('num_heads', 8)
                self.ff_dim = metadata.get('ff_dim', 128)
                self.num_transformer_blocks = metadata.get('num_transformer_blocks', 4)
                self.dropout_rate = metadata.get('dropout_rate', 0.1)
                self.is_trained = True

                training_time = metadata.get('training_timestamp', 'Unknown')
                print(f"📂 Transformer model loaded from {model_file} (trained: {training_time})")

            except Exception as e:
                print(f"❌ Failed to load Transformer model: {e}")

        def get_model_info(self) -> Dict[str, Any]:
            """Get information about the current Transformer model."""
            if not self.is_trained:
                return {'status': 'not_trained'}

            return {
                'status': 'trained',
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'sequence_length': self.sequence_length,
                'num_heads': self.num_heads,
                'ff_dim': self.ff_dim,
                'num_transformer_blocks': self.num_transformer_blocks,
                'dropout_rate': self.dropout_rate,
                'feature_count': len(self.feature_columns),
                'features': self.feature_columns,
                'training_history': self.training_history
            }

else:
    # Placeholder class when TensorFlow is not available
    class TransformerSignalPredictor:
        def __init__(self, *args, **kwargs):
            raise ImportError('TensorFlow is not available. Install with: pip install tensorflow')

# Global LSTM predictor instance
lstm_predictor = None

if TENSORFLOW_AVAILABLE:
    def get_lstm_predictor(symbol: str = SYMBOL, timeframe: str = TIMEFRAME) -> Optional[LSTMSignalPredictor]:
        """Get or create LSTM predictor instance."""
        global lstm_predictor

        if lstm_predictor is None:
            try:
                lstm_predictor = LSTMSignalPredictor(symbol=symbol, timeframe=timeframe)
            except Exception as e:
                print(f"❌ Failed to initialize LSTM predictor: {e}")
                return None

        return lstm_predictor
else:
    def get_lstm_predictor(symbol: str = SYMBOL, timeframe: str = TIMEFRAME) -> Optional[Any]:
        """Get LSTM predictor instance (not available)."""
        return None

def is_lstm_available() -> bool:
    """Check if LSTM features are available."""
    return TENSORFLOW_AVAILABLE

# Global Transformer predictor instance
transformer_predictor = None

if TENSORFLOW_AVAILABLE:
    def get_transformer_predictor(symbol: str = SYMBOL, timeframe: str = TIMEFRAME) -> Optional[TransformerSignalPredictor]:
        """Get or create Transformer predictor instance."""
        global transformer_predictor

        if transformer_predictor is None:
            try:
                transformer_predictor = TransformerSignalPredictor(symbol=symbol, timeframe=timeframe)
            except Exception as e:
                print(f"❌ Failed to initialize Transformer predictor: {e}")
                return None

        return transformer_predictor
else:
    def get_transformer_predictor(symbol: str = SYMBOL, timeframe: str = TIMEFRAME) -> Optional[Any]:
        """Get Transformer predictor instance (not available)."""
        return None

def is_transformer_available() -> bool:
    """Check if Transformer features are available."""
    return TENSORFLOW_AVAILABLE

def is_shap_available() -> bool:
    """Check if SHAP explanations are available."""
    return SHAP_AVAILABLE
