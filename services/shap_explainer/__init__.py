"""
SHAP Explainer Service - Model explainability for trading predictions.

Provides SHAP-based explainability for PyTorch models used in trading signal generation.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Union
import logging

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install with: pip install shap")

try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Install with: pip install torch")


class PyTorchSHAPExplainer:
    """SHAP explainer for PyTorch models used in trading."""

    def __init__(self, cache_manager=None):
        """Initialize SHAP explainer.

        Args:
            cache_manager: Optional cache manager for storing explanations
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for PyTorchSHAPExplainer. Install with: pip install shap")

        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required for PyTorchSHAPExplainer. Install with: pip install torch")

        self.cache_manager = cache_manager
        self.model = None
        self.explainer = None
        self.background_data = None
        self.feature_names = None
        self.is_initialized = False
        self.expected_value = None

    def initialize_explainer(self, model, background_data: np.ndarray,
                           feature_names: List[str]) -> bool:
        """Initialize the SHAP explainer with a trained model.

        Args:
            model: Trained PyTorch model
            background_data: Background dataset for SHAP (typically 50-100 samples)
            feature_names: Names of input features

        Returns:
            bool: True if initialization successful
        """
        try:
            self.model = model
            self.background_data = background_data
            self.feature_names = feature_names

            # Set model to evaluation mode
            if hasattr(model, 'eval'):
                model.eval()

            # Create SHAP explainer
            self.explainer = shap.DeepExplainer(model, background_data)
            self.expected_value = self.explainer.expected_value
            self.is_initialized = True

            logger.info(f"SHAP explainer initialized with {len(feature_names)} features")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            return False

    def explain_prediction(self, input_data: np.ndarray) -> Dict[str, Any]:
        """Explain a single prediction.

        Args:
            input_data: Input features for prediction (shape: [1, n_features])

        Returns:
            dict: Explanation containing SHAP values, predictions, etc.
        """
        if not self.is_initialized:
            raise ValueError("Explainer not initialized. Call initialize_explainer first.")

        try:
            # Ensure input is 2D
            if input_data.ndim == 1:
                input_data = input_data.reshape(1, -1)

            # Convert to torch tensor if needed
            if PYTORCH_AVAILABLE:
                # Always convert to tensor when PyTorch is available
                input_tensor = torch.tensor(input_data, dtype=torch.float32)
            else:
                # If PyTorch not available, work with numpy arrays directly
                input_tensor = input_data

            # Get SHAP values
            shap_values = self.explainer.shap_values(input_tensor)

            # Get model predictions
            if PYTORCH_AVAILABLE:
                with torch.no_grad():
                    predictions = self.model(input_tensor)
                    # Convert to numpy if it's a tensor
                    try:
                        if hasattr(predictions, 'cpu') and hasattr(predictions, 'numpy'):
                            predictions = predictions.cpu().numpy()
                    except (AttributeError, TypeError):
                        pass  # Keep as is if conversion fails
            else:
                # For non-PyTorch models, assume they have a predict method
                predictions = self.model.predict(input_tensor)

            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # Multi-output case
                shap_values = shap_values[0] if len(shap_values) == 1 else shap_values

            if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                # Shape: [n_samples, n_classes, n_features] -> [n_samples, n_features]
                shap_values = shap_values[:, 0, :] if shap_values.shape[1] == 1 else shap_values

            explanation = {
                'shap_values': shap_values,
                'predictions': predictions,
                'feature_names': self.feature_names,
                'expected_value': self.expected_value,
                'input_data': input_data
            }

            return explanation

        except Exception as e:
            logger.error(f"Failed to explain prediction: {e}")
            raise

    def explain_feature_importance(self, test_data: np.ndarray) -> Dict[str, Any]:
        """Calculate feature importance across multiple samples.

        Args:
            test_data: Test dataset (shape: [n_samples, n_features])

        Returns:
            dict: Feature importance metrics
        """
        if not self.is_initialized:
            raise ValueError("Explainer not initialized. Call initialize_explainer first.")

        try:
            # Get explanations for all samples
            explanations = []
            for i in range(min(len(test_data), 100)):  # Limit to 100 samples for performance
                exp = self.explain_prediction(test_data[i:i+1])
                explanations.append(exp)

            # Aggregate SHAP values
            all_shap_values = np.array([exp['shap_values'] for exp in explanations])

            # Calculate mean absolute SHAP values for each feature
            mean_shap_values = np.mean(np.abs(all_shap_values), axis=0)

            # Flatten if needed to handle different array shapes
            if mean_shap_values.ndim > 1:
                mean_shap_values = mean_shap_values.flatten()

            # Create feature importance ranking
            feature_importance = []
            for idx, feature_name in enumerate(self.feature_names):
                if idx < len(mean_shap_values):
                    importance = {
                        'feature': feature_name,
                        'mean_shap_value': float(mean_shap_values[idx]),
                        'rank': 0  # Will be set below
                    }
                    feature_importance.append(importance)

            # Sort by importance
            feature_importance.sort(key=lambda x: x['mean_shap_value'], reverse=True)

            # Set ranks
            for rank, feat in enumerate(feature_importance, 1):
                feat['rank'] = rank

            return {
                'feature_importance': feature_importance,
                'mean_shap_values': mean_shap_values,
                'samples_used': len(explanations)
            }

        except Exception as e:
            logger.error(f"Failed to calculate feature importance: {e}")
            raise

    def explain_trading_signal(self, input_data: np.ndarray,
                             signal_type: str) -> Dict[str, Any]:
        """Explain a trading signal prediction.

        Args:
            input_data: Input features
            signal_type: Type of signal (BUY/SELL/HOLD)

        Returns:
            dict: Signal explanation with key contributing features
        """
        if not self.is_initialized:
            raise ValueError("Explainer not initialized. Call initialize_explainer first.")

        try:
            explanation = self.explain_prediction(input_data)

            shap_values = explanation['shap_values'][0]  # First sample
            predictions = explanation['predictions'][0]

            # Ensure shap_values is 1D
            if isinstance(shap_values, np.ndarray) and shap_values.ndim > 1:
                shap_values = shap_values.flatten()

            # Get prediction confidence (assuming binary classification)
            if hasattr(predictions, '__len__') and len(predictions) > 1:
                confidence = float(max(predictions))
            else:
                confidence = float(abs(predictions))

            # Create feature contribution ranking
            feature_contributions = []
            for idx, feature_name in enumerate(self.feature_names):
                if idx < len(shap_values):
                    contribution = {
                        'feature': feature_name,
                        'shap_value': float(shap_values[idx]),
                        'abs_shap_value': float(abs(shap_values[idx]))
                    }
                    feature_contributions.append(contribution)

            # Sort by absolute contribution
            feature_contributions.sort(key=lambda x: x['abs_shap_value'], reverse=True)

            # Separate positive and negative contributions
            positive_features = [f for f in feature_contributions if f['shap_value'] > 0][:5]
            negative_features = [f for f in feature_contributions if f['shap_value'] < 0][:5]

            return {
                'signal_type': signal_type,
                'prediction_confidence': confidence,
                'top_positive_features': positive_features,
                'top_negative_features': negative_features,
                'all_feature_contributions': feature_contributions,
                'shap_summary': {
                    'total_positive_contribution': float(np.sum(shap_values[shap_values > 0])),
                    'total_negative_contribution': float(np.sum(shap_values[shap_values < 0])),
                    'expected_value': float(self.expected_value) if self.expected_value is not None else 0.0
                }
            }

        except Exception as e:
            logger.error(f"Failed to explain trading signal: {e}")
            raise

    def cache_explanation(self, key: str, explanation: Dict[str, Any]) -> bool:
        """Cache an explanation for later retrieval.

        Args:
            key: Cache key
            explanation: Explanation data to cache

        Returns:
            bool: True if caching successful
        """
        if not self.cache_manager:
            return False

        try:
            return self.cache_manager.set(key, explanation)
        except Exception as e:
            logger.error(f"Failed to cache explanation: {e}")
            return False

    def get_cached_explanation(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a cached explanation.

        Args:
            key: Cache key

        Returns:
            Cached explanation or None if not found
        """
        if not self.cache_manager:
            return None

        try:
            return self.cache_manager.get(key)
        except Exception as e:
            logger.error(f"Failed to retrieve cached explanation: {e}")
            return None


class SHAPManager:
    """Manager for multiple SHAP explainers."""

    def __init__(self, cache_manager=None):
        """Initialize SHAP manager.

        Args:
            cache_manager: Optional cache manager
        """
        self.cache_manager = cache_manager
        self.explainers: Dict[str, PyTorchSHAPExplainer] = {}

    def register_model(self, model_name: str, model,
                      background_data: np.ndarray,
                      feature_names: List[str]) -> bool:
        """Register a model with SHAP explainer.

        Args:
            model_name: Name identifier for the model
            model: Trained PyTorch model
            background_data: Background dataset for SHAP
            feature_names: Feature names

        Returns:
            bool: True if registration successful
        """
        try:
            explainer = PyTorchSHAPExplainer(cache_manager=self.cache_manager)
            success = explainer.initialize_explainer(model, background_data, feature_names)

            if success:
                self.explainers[model_name] = explainer
                logger.info(f"Model '{model_name}' registered with SHAP explainer")
                return True
            else:
                logger.error(f"Failed to initialize explainer for model '{model_name}'")
                return False

        except Exception as e:
            logger.error(f"Failed to register model '{model_name}': {e}")
            return False

    def explain_model_prediction(self, model_name: str,
                               input_data: np.ndarray) -> Dict[str, Any]:
        """Explain prediction for a registered model.

        Args:
            model_name: Name of registered model
            input_data: Input features

        Returns:
            dict: Prediction explanation
        """
        if model_name not in self.explainers:
            raise ValueError(f"Model '{model_name}' not registered")

        return self.explainers[model_name].explain_prediction(input_data)

    def get_model_feature_importance(self, model_name: str,
                                   test_data: np.ndarray) -> Dict[str, Any]:
        """Get feature importance for a registered model.

        Args:
            model_name: Name of registered model
            test_data: Test dataset

        Returns:
            dict: Feature importance analysis
        """
        if model_name not in self.explainers:
            raise ValueError(f"Model '{model_name}' not registered")

        return self.explainers[model_name].explain_feature_importance(test_data)

    def explain_model_signal(self, model_name: str, input_data: np.ndarray,
                           signal_type: str) -> Dict[str, Any]:
        """Explain trading signal for a registered model.

        Args:
            model_name: Name of registered model
            input_data: Input features
            signal_type: Signal type (BUY/SELL/HOLD)

        Returns:
            dict: Signal explanation
        """
        if model_name not in self.explainers:
            raise ValueError(f"Model '{model_name}' not registered")

        return self.explainers[model_name].explain_trading_signal(input_data, signal_type)