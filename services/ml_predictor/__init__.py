"""
ML Predictor Service - Machine learning signal prediction.

Provides ML-based signal prediction and enhancement capabilities.
"""

import logging
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import os

from config.settings import ML_MODELS_DIR, ML_ENABLED

logger = logging.getLogger(__name__)


class MLPredictor:
    """ML-based signal predictor."""

    def __init__(self):
        self.is_trained = False
        self.model = None
        self.scaler = None
        self.feature_columns = []
        self.models_dir = Path(ML_MODELS_DIR)
        self.models_dir.mkdir(exist_ok=True)

    def load_model(self, model_name: str = "default") -> bool:
        """Load a trained model from disk."""
        try:
            model_path = self.models_dir / f"{model_name}_model.pkl"
            scaler_path = self.models_dir / f"{model_name}_scaler.pkl"
            config_path = self.models_dir / f"{model_name}_config.pkl"

            if not all(p.exists() for p in [model_path, scaler_path, config_path]):
                return False

            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            with open(config_path, 'rb') as f:
                config = pickle.load(f)
                self.feature_columns = config.get('feature_columns', [])
                self.is_trained = config.get('is_trained', False)

            return True
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False

    def predict_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict trading signals using ML model."""
        if not self.is_trained or self.model is None:
            # Return original data if no model available
            return data

        try:
            # Prepare features
            features = data[self.feature_columns].copy()
            features_scaled = self.scaler.transform(features)

            # Make predictions
            predictions = self.model.predict_proba(features_scaled)[:, 1]

            # Add predictions to data
            data = data.copy()
            data['ml_signal_confidence'] = predictions
            data['ml_buy_signal'] = (predictions > 0.6).astype(int)
            data['ml_sell_signal'] = (predictions < 0.4).astype(int)

            return data
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return data

    def enhance_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enhance existing signals with ML predictions."""
        if 'Buy_Signal' not in data.columns or 'Sell_Signal' not in data.columns:
            return data

        enhanced_data = self.predict_signals(data)

        # Combine traditional and ML signals
        enhanced_data['enhanced_buy_signal'] = (
            (enhanced_data.get('Buy_Signal', 0) == 1) |
            (enhanced_data.get('ml_buy_signal', 0) == 1)
        ).astype(int)

        enhanced_data['enhanced_sell_signal'] = (
            (enhanced_data.get('Sell_Signal', 0) == 1) |
            (enhanced_data.get('ml_sell_signal', 0) == 1)
        ).astype(int)

        # Add Signal_Source column to indicate signal origin
        def determine_signal_source(row):
            traditional_buy = row.get('Buy_Signal', 0) == 1
            traditional_sell = row.get('Sell_Signal', 0) == 1
            ml_buy = row.get('ml_buy_signal', 0) == 1
            ml_sell = row.get('ml_sell_signal', 0) == 1

            has_traditional = traditional_buy or traditional_sell
            has_ml = ml_buy or ml_sell

            if has_traditional and has_ml:
                return 'COMBINED'
            elif has_ml:
                return 'ML'
            else:
                return 'TRADITIONAL'

        enhanced_data['Signal_Source'] = enhanced_data.apply(determine_signal_source, axis=1)

        return enhanced_data


def get_ml_predictor() -> Optional[MLPredictor]:
    """Get ML predictor instance."""
    if not ML_ENABLED:
        return None

    predictor = MLPredictor()
    predictor.load_model()  # Try to load default model
    return predictor


def get_lstm_predictor() -> Optional[MLPredictor]:
    """Get LSTM predictor instance (placeholder for future LSTM implementation)."""
    # LSTM implementation would go here
    # For now, return None to indicate LSTM is not available
    return None


def is_lstm_available() -> bool:
    """Check if LSTM functionality is available."""
    return False  # LSTM not yet implemented


def is_ml_available() -> bool:
    """Check if ML functionality is available."""
    return ML_ENABLED