"""
ML Ensemble Service - Ensemble methods for trading signal generation.

Provides advanced ensemble techniques combining multiple ML models.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score

from config.settings import ML_MODELS_DIR, ML_ENABLED

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """Ensemble predictor combining multiple ML models."""

    def __init__(self):
        self.models = {}
        self.is_trained = False
        self.ensemble_model = None
        self.models_dir = Path(ML_MODELS_DIR)
        self.models_dir.mkdir(exist_ok=True)

    def create_ensemble(self, model_configs: List[Dict[str, Any]] = None):
        """Create ensemble from individual models."""
        if model_configs is None:
            model_configs = [
                {'name': 'rf', 'model': RandomForestClassifier(n_estimators=100, random_state=42)},
                {'name': 'gb', 'model': GradientBoostingClassifier(n_estimators=100, random_state=42)},
                {'name': 'lr', 'model': LogisticRegression(random_state=42)},
                {'name': 'svm', 'model': SVC(probability=True, random_state=42)}
            ]

        estimators = []
        for config in model_configs:
            estimators.append((config['name'], config['model']))
            self.models[config['name']] = config['model']

        # Create voting classifier
        self.ensemble_model = VotingClassifier(
            estimators=estimators,
            voting='soft'  # Use probability-based voting
        )

    def train_ensemble(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the ensemble model."""
        try:
            if self.ensemble_model is None:
                self.create_ensemble()

            self.ensemble_model.fit(X_train, y_train)
            self.is_trained = True

            # Save trained models
            self._save_models()

            return True
        except Exception as e:
            logger.error(f"Ensemble training failed: {e}")
            return False

    def predict_signals(self, X: np.ndarray) -> np.ndarray:
        """Predict trading signals using ensemble."""
        if not self.is_trained or self.ensemble_model is None:
            return np.zeros(len(X))

        try:
            return self.ensemble_model.predict(X)
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            return np.zeros(len(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict signal probabilities using ensemble."""
        if not self.is_trained or self.ensemble_model is None:
            return np.full((len(X), 2), 0.5)

        try:
            return self.ensemble_model.predict_proba(X)
        except Exception as e:
            logger.error(f"Ensemble probability prediction failed: {e}")
            return np.full((len(X), 2), 0.5)

    def _save_models(self):
        """Save trained models to disk."""
        try:
            for name, model in self.models.items():
                model_path = self.models_dir / f"ensemble_{name}_model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)

            # Save ensemble
            ensemble_path = self.models_dir / "ensemble_model.pkl"
            with open(ensemble_path, 'wb') as f:
                pickle.dump(self.ensemble_model, f)

        except Exception as e:
            logger.error(f"Failed to save models: {e}")

    def load_models(self) -> bool:
        """Load trained models from disk."""
        try:
            ensemble_path = self.models_dir / "ensemble_model.pkl"
            if ensemble_path.exists():
                with open(ensemble_path, 'rb') as f:
                    self.ensemble_model = pickle.load(f)
                self.is_trained = True
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False

    def get_model_weights(self) -> Dict[str, float]:
        """Get weights/importance of individual models in ensemble."""
        if not hasattr(self.ensemble_model, 'estimators_'):
            return {}

        # For voting classifier, all models have equal weight by default
        weights = {}
        for name, _ in self.ensemble_model.estimators:
            weights[name] = 1.0 / len(self.ensemble_model.estimators)

        return weights


def create_ensemble_signal_generation(data: pd.DataFrame,
                                    feature_columns: List[str] = None) -> pd.DataFrame:
    """Generate signals using ensemble methods."""
    if not ML_ENABLED:
        return data

    try:
        if feature_columns is None:
            feature_columns = ['EMA9', 'EMA21', 'RSI', 'BB_upper', 'BB_lower', 'ATR']

        # Prepare features
        available_features = [col for col in feature_columns if col in data.columns]
        if not available_features:
            return data

        X = data[available_features].fillna(0).values

        # Create target (simplified: 1 if price goes up next period, 0 otherwise)
        if len(data) > 1:
            y = (data['close'].shift(-1) > data['close']).astype(int).fillna(0).values[:-1]
            X = X[:-1]  # Remove last sample since we don't have target
        else:
            return data

        # Train ensemble if not trained
        ensemble = EnsemblePredictor()
        if not ensemble.load_models():
            ensemble.train_ensemble(X, y)

        # Generate predictions
        predictions = ensemble.predict_signals(X)
        probabilities = ensemble.predict_proba(X)

        # Add to dataframe
        result_df = data.copy()
        result_df['ensemble_signal'] = 0
        result_df['ensemble_confidence'] = 0.5

        # Map predictions back (excluding last sample)
        if len(predictions) > 0:
            result_df.iloc[:-1, result_df.columns.get_loc('ensemble_signal')] = predictions
            result_df.iloc[:-1, result_df.columns.get_loc('ensemble_confidence')] = probabilities[:, 1]

        return result_df

    except Exception as e:
        logger.error(f"Ensemble signal generation failed: {e}")
        return data


def run_ensemble_backtest(data: pd.DataFrame) -> Dict[str, Any]:
    """Run backtest using ensemble signals."""
    try:
        # Generate ensemble signals
        signal_data = create_ensemble_signal_generation(data)

        # Simple signal generation based on ensemble predictions
        signal_data['Buy_Signal'] = (signal_data['ensemble_signal'] == 1).astype(int)
        signal_data['Sell_Signal'] = (signal_data['ensemble_signal'] == 0).astype(int)

        # Import here to avoid circular imports
        from services.backtester import run_backtest
        return run_backtest(signal_data, strategy='ensemble')

    except Exception as e:
        logger.error(f"Ensemble backtest failed: {e}")
        return {'error': str(e)}