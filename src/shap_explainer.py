"""
SHAP Integration for PyTorch Models in TradPal Indicator

This module provides SHAP (SHapley Additive exPlanations) integration for PyTorch models,
enabling explainable AI for trading signal predictions. SHAP helps understand which features
are most important for model predictions and provides insights into model behavior.

Features:
- SHAP explanations for PyTorch neural networks
- Feature importance analysis
- Prediction explanations for individual samples
- Global and local interpretability
- Integration with LSTM and Transformer models
- Visualization support

Author: TradPal Indicator Team
Version: 2.5.0
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")
    logging.warning("SHAP integration will be disabled.")

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logging.warning("PyTorch not available. SHAP integration requires PyTorch.")
    # Define dummy classes for type hints
    class nn:
        class Module:
            pass
    torch = None

from src.cache import Cache

# Configure logging
logger = logging.getLogger(__name__)

class PyTorchSHAPExplainer:
    """
    SHAP explainer for PyTorch models used in trading predictions.

    Provides comprehensive explainability for neural network predictions,
    helping traders understand model decisions and build trust in AI signals.
    """

    def __init__(self,
                 model: Optional[nn.Module] = None,
                 background_data: Optional[np.ndarray] = None,
                 cache_manager: Optional[Cache] = None):
        """
        Initialize the SHAP explainer.

        Args:
            model: PyTorch model to explain
            background_data: Background dataset for SHAP (representative samples)
            cache_manager: Cache manager for storing explanations
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for explainability features. Install with: pip install shap")

        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required for model explanations.")

        self.model = model
        self.background_data = background_data
        self.cache_manager = cache_manager or Cache()
        self.explainer = None
        self.feature_names = None
        self.is_initialized = False

        logger.info("PyTorchSHAPExplainer initialized")

    def initialize_explainer(self,
                           model: nn.Module,
                           background_data: np.ndarray,
                           feature_names: Optional[List[str]] = None,
                           explainer_type: str = "deep") -> bool:
        """
        Initialize the SHAP explainer with model and background data.

        Args:
            model: PyTorch model to explain
            background_data: Background dataset (should be representative of training data)
            feature_names: Names of input features
            explainer_type: Type of SHAP explainer ('deep', 'gradient', 'deep_lift')

        Returns:
            True if initialization successful
        """
        try:
            self.model = model
            self.background_data = background_data
            self.feature_names = feature_names or [f"feature_{i}" for i in range(background_data.shape[1])]

            # Convert background data to torch tensor
            background_tensor = torch.tensor(background_data, dtype=torch.float32)

            # Set model to evaluation mode
            self.model.eval()

            # Create SHAP explainer based on type
            if explainer_type == "deep":
                self.explainer = shap.DeepExplainer(self.model, background_tensor)
            elif explainer_type == "gradient":
                self.explainer = shap.GradientExplainer(self.model, background_tensor)
            elif explainer_type == "deep_lift":
                # DeepLIFT explainer for more detailed attributions
                self.explainer = shap.DeepExplainer(self.model, background_tensor)
            else:
                raise ValueError(f"Unsupported explainer type: {explainer_type}")

            self.is_initialized = True

            logger.info(f"SHAP explainer initialized with {explainer_type} method")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            return False

    def explain_prediction(self,
                          input_data: Union[np.ndarray, Any],
                          max_evals: int = 1000) -> Dict[str, Any]:
        """
        Explain a single prediction or batch of predictions.

        Args:
            input_data: Input data to explain (single sample or batch)
            max_evals: Maximum evaluations for SHAP (for performance)

        Returns:
            Dictionary containing SHAP values and explanations
        """
        if not self.is_initialized:
            raise ValueError("Explainer not initialized. Call initialize_explainer() first.")

        try:
            # Ensure input is tensor
            if isinstance(input_data, np.ndarray):
                input_tensor = torch.tensor(input_data, dtype=torch.float32)
            else:
                input_tensor = input_data

            # Handle single sample vs batch
            if input_tensor.dim() == 1:
                input_tensor = input_tensor.unsqueeze(0)

            # Calculate SHAP values
            shap_values = self.explainer.shap_values(input_tensor, max_evals=max_evals)

            # Convert to numpy for easier handling
            if isinstance(shap_values, list):
                # Multi-output case
                shap_values = np.array(shap_values)
            else:
                # Single output case
                shap_values = np.array([shap_values])

            # Get predictions
            with torch.no_grad():
                predictions = self.model(input_tensor).cpu().numpy()

            # Create explanation dictionary
            explanation = {
                'shap_values': shap_values,
                'predictions': predictions,
                'input_data': input_tensor.cpu().numpy(),
                'feature_names': self.feature_names,
                'expected_value': self.explainer.expected_value,
                'timestamp': datetime.utcnow().isoformat()
            }

            return explanation

        except Exception as e:
            logger.error(f"Failed to explain prediction: {e}")
            return {}

    def explain_feature_importance(self,
                                 test_data: np.ndarray,
                                 max_samples: int = 1000) -> Dict[str, Any]:
        """
        Calculate global feature importance across a test dataset.

        Args:
            test_data: Test dataset for global explanations
            max_samples: Maximum samples to use for efficiency

        Returns:
            Dictionary with feature importance metrics
        """
        if not self.is_initialized:
            raise ValueError("Explainer not initialized. Call initialize_explainer() first.")

        try:
            # Limit samples for performance
            if len(test_data) > max_samples:
                indices = np.random.choice(len(test_data), max_samples, replace=False)
                test_data = test_data[indices]

            # Get SHAP values for all test samples
            explanations = self.explain_prediction(test_data)

            if not explanations:
                return {}

            shap_values = explanations['shap_values']

            # Calculate mean absolute SHAP values for each feature
            mean_shap = np.mean(np.abs(shap_values), axis=(0, 1))  # Average across samples and outputs

            # Calculate feature importance rankings
            feature_importance = list(zip(self.feature_names, mean_shap))
            feature_importance.sort(key=lambda x: x[1], reverse=True)

            importance_dict = {
                'feature_importance': feature_importance,
                'mean_shap_values': dict(zip(self.feature_names, mean_shap)),
                'samples_used': len(test_data),
                'timestamp': datetime.utcnow().isoformat()
            }

            return importance_dict

        except Exception as e:
            logger.error(f"Failed to calculate feature importance: {e}")
            return {}

    def explain_trading_signal(self,
                             signal_data: np.ndarray,
                             signal_type: str = "BUY") -> Dict[str, Any]:
        """
        Provide detailed explanation for a trading signal prediction.

        Args:
            signal_data: Input features for the signal
            signal_type: Type of signal (BUY/SELL/HOLD)

        Returns:
            Detailed signal explanation
        """
        try:
            explanation = self.explain_prediction(signal_data)

            if not explanation:
                return {}

            shap_values = explanation['shap_values'][0]  # First sample
            prediction = explanation['predictions'][0]

            # Identify most important features for this prediction
            feature_contributions = list(zip(self.feature_names, shap_values[0]))
            feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)

            # Categorize features by impact
            positive_features = [f for f in feature_contributions if f[1] > 0]
            negative_features = [f for f in feature_contributions if f[1] < 0]

            signal_explanation = {
                'signal_type': signal_type,
                'prediction_confidence': float(prediction[0]),
                'top_positive_features': positive_features[:5],  # Top 5 positive contributors
                'top_negative_features': negative_features[:5],  # Top 5 negative contributors
                'feature_contributions': feature_contributions,
                'shap_summary': explanation,
                'explanation_timestamp': datetime.utcnow().isoformat()
            }

            logger.info(f"Trading signal explained for {signal_type}")

            return signal_explanation

        except Exception as e:
            logger.error(f"Failed to explain trading signal: {e}")
            return {}

    def create_shap_summary_plot(self,
                               explanation: Dict[str, Any],
                               save_path: Optional[str] = None) -> bool:
        """
        Create SHAP summary plot for visualization.

        Args:
            explanation: SHAP explanation dictionary
            save_path: Path to save the plot (optional)

        Returns:
            True if plot created successfully
        """
        try:
            if not explanation or 'shap_values' not in explanation:
                return False

            shap_values = explanation['shap_values']
            feature_names = explanation.get('feature_names', self.feature_names)

            # Create summary plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, feature_names=feature_names, show=False)

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"SHAP summary plot saved to {save_path}")
            else:
                plt.show()

            plt.close()
            return True

        except Exception as e:
            logger.error(f"Failed to create SHAP summary plot: {e}")
            return False

    def create_waterfall_plot(self,
                            explanation: Dict[str, Any],
                            sample_index: int = 0,
                            save_path: Optional[str] = None) -> bool:
        """
        Create SHAP waterfall plot for individual prediction explanation.

        Args:
            explanation: SHAP explanation dictionary
            sample_index: Index of sample to explain
            save_path: Path to save the plot (optional)

        Returns:
            True if plot created successfully
        """
        try:
            if not explanation or 'shap_values' not in explanation:
                return False

            shap_values = explanation['shap_values'][sample_index]
            feature_names = explanation.get('feature_names', self.feature_names)
            expected_value = explanation.get('expected_value', 0)

            # Create waterfall plot
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(expected_value, shap_values, feature_names, show=False)

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"SHAP waterfall plot saved to {save_path}")
            else:
                plt.show()

            plt.close()
            return True

        except Exception as e:
            logger.error(f"Failed to create SHAP waterfall plot: {e}")
            return False

    def cache_explanation(self, key: str, explanation: Dict[str, Any]) -> bool:
        """
        Cache SHAP explanation for later retrieval.

        Args:
            key: Cache key for the explanation
            explanation: SHAP explanation dictionary

        Returns:
            True if cached successfully
        """
        try:
            cache_key = f"shap_explanation_{key}"
            self.cache_manager.set(cache_key, explanation)
            return True
        except Exception as e:
            logger.error(f"Failed to cache explanation: {e}")
            return False

    def get_cached_explanation(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached SHAP explanation.

        Args:
            key: Cache key for the explanation

        Returns:
            Cached explanation dictionary or None
        """
        try:
            cache_key = f"shap_explanation_{key}"
            return self.cache_manager.get(cache_key)
        except Exception as e:
            logger.error(f"Failed to retrieve cached explanation: {e}")
            return None

class SHAPManager:
    """
    High-level manager for SHAP explanations across multiple models.

    Provides unified interface for explainability across different PyTorch models
    used in the trading system.
    """

    def __init__(self, cache_manager: Optional[Cache] = None):
        """
        Initialize the SHAP manager.

        Args:
            cache_manager: Cache manager for storing explanations
        """
        self.cache_manager = cache_manager or Cache()
        self.explainers: Dict[str, PyTorchSHAPExplainer] = {}

        logger.info("SHAPManager initialized")

    def register_model(self,
                      model_name: str,
                      model: nn.Module,
                      background_data: np.ndarray,
                      feature_names: Optional[List[str]] = None,
                      explainer_type: str = "deep") -> bool:
        """
        Register a model for SHAP explanations.

        Args:
            model_name: Name identifier for the model
            model: PyTorch model
            background_data: Background dataset for SHAP
            feature_names: Feature names
            explainer_type: SHAP explainer type

        Returns:
            True if registration successful
        """
        try:
            explainer = PyTorchSHAPExplainer(cache_manager=self.cache_manager)
            success = explainer.initialize_explainer(
                model=model,
                background_data=background_data,
                feature_names=feature_names,
                explainer_type=explainer_type
            )

            if success:
                self.explainers[model_name] = explainer
                logger.info(f"Model '{model_name}' registered for SHAP explanations")
                return True
            else:
                logger.error(f"Failed to register model '{model_name}'")
                return False

        except Exception as e:
            logger.error(f"Error registering model '{model_name}': {e}")
            return False

    def explain_model_prediction(self,
                               model_name: str,
                               input_data: np.ndarray) -> Dict[str, Any]:
        """
        Explain a prediction from a registered model.

        Args:
            model_name: Name of the registered model
            input_data: Input data to explain

        Returns:
            SHAP explanation dictionary
        """
        if model_name not in self.explainers:
            raise ValueError(f"Model '{model_name}' not registered")

        return self.explainers[model_name].explain_prediction(input_data)

    def get_model_feature_importance(self,
                                   model_name: str,
                                   test_data: np.ndarray) -> Dict[str, Any]:
        """
        Get feature importance for a registered model.

        Args:
            model_name: Name of the registered model
            test_data: Test data for importance calculation

        Returns:
            Feature importance dictionary
        """
        if model_name not in self.explainers:
            raise ValueError(f"Model '{model_name}' not registered")

        return self.explainers[model_name].explain_feature_importance(test_data)

    def explain_trading_decision(self,
                               model_name: str,
                               signal_data: np.ndarray,
                               decision: str = "HOLD") -> Dict[str, Any]:
        """
        Provide detailed explanation for a trading decision.

        Args:
            model_name: Name of the registered model
            signal_data: Signal input data
            decision: Trading decision (BUY/SELL/HOLD)

        Returns:
            Detailed trading decision explanation
        """
        if model_name not in self.explainers:
            raise ValueError(f"Model '{model_name}' not registered")

        return self.explainers[model_name].explain_trading_signal(signal_data, decision)

# Utility functions
def create_shap_explainer_for_lstm(model: nn.Module,
                                  background_data: np.ndarray,
                                  feature_names: Optional[List[str]] = None) -> PyTorchSHAPExplainer:
    """
    Create SHAP explainer specifically configured for LSTM models.

    Args:
        model: LSTM model
        background_data: Background sequences
        feature_names: Feature names

    Returns:
        Configured SHAP explainer
    """
    explainer = PyTorchSHAPExplainer()
    explainer.initialize_explainer(
        model=model,
        background_data=background_data,
        feature_names=feature_names,
        explainer_type="deep"  # Deep explainer works well with LSTMs
    )
    return explainer

def create_shap_explainer_for_transformer(model: nn.Module,
                                        background_data: np.ndarray,
                                        feature_names: Optional[List[str]] = None) -> PyTorchSHAPExplainer:
    """
    Create SHAP explainer specifically configured for Transformer models.

    Args:
        model: Transformer model
        background_data: Background sequences
        feature_names: Feature names

    Returns:
        Configured SHAP explainer
    """
    explainer = PyTorchSHAPExplainer()
    explainer.initialize_explainer(
        model=model,
        background_data=background_data,
        feature_names=feature_names,
        explainer_type="gradient"  # Gradient explainer often works better with Transformers
    )
    return explainer

def generate_shap_report(explanation: Dict[str, Any],
                        model_name: str,
                        output_dir: str = "output/shap_reports") -> str:
    """
    Generate comprehensive SHAP report with visualizations.

    Args:
        explanation: SHAP explanation dictionary
        model_name: Name of the model
        output_dir: Directory to save report files

    Returns:
        Path to generated report
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f"shap_report_{model_name}_{timestamp}.txt")

        with open(report_path, 'w') as f:
            f.write(f"SHAP Explanation Report for {model_name}\n")
            f.write(f"Generated: {datetime.utcnow().isoformat()}\n\n")

            if 'feature_importance' in explanation:
                f.write("Feature Importance Ranking:\n")
                for i, (feature, importance) in enumerate(explanation['feature_importance'][:10]):
                    f.write(f"{i+1}. {feature}: {importance:.4f}\n")
                f.write("\n")

            if 'signal_explanation' in explanation:
                signal_exp = explanation['signal_explanation']
                f.write(f"Trading Signal: {signal_exp['signal_type']}\n")
                f.write(f"Prediction Confidence: {signal_exp['prediction_confidence']:.4f}\n\n")

                f.write("Top Positive Contributors:\n")
                for feature, contribution in signal_exp['top_positive_features'][:5]:
                    f.write(f"  {feature}: +{contribution:.4f}\n")

                f.write("\nTop Negative Contributors:\n")
                for feature, contribution in signal_exp['top_negative_features'][:5]:
                    f.write(f"  {feature}: {contribution:.4f}\n")

        logger.info(f"SHAP report generated: {report_path}")
        return report_path

    except Exception as e:
        logger.error(f"Failed to generate SHAP report: {e}")
        return ""

# Import matplotlib for plotting (optional)
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available. SHAP plots will be disabled.")

if __name__ == "__main__":
    print("🔍 Testing SHAP Integration...")

    if not SHAP_AVAILABLE:
        print("❌ SHAP not available. Install with: pip install shap")
        exit(1)

    if not PYTORCH_AVAILABLE:
        print("❌ PyTorch not available. SHAP requires PyTorch.")
        exit(1)

    try:
        # Create a simple test model
        class SimpleModel(nn.Module):
            def __init__(self, input_size=10, hidden_size=50, output_size=1):
                super().__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, output_size)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                return self.fc2(x)

        # Create test data
        input_size = 10
        background_data = np.random.randn(100, input_size)
        test_data = np.random.randn(5, input_size)

        feature_names = [f"feature_{i}" for i in range(input_size)]

        # Initialize model and explainer
        model = SimpleModel(input_size=input_size)
        explainer = PyTorchSHAPExplainer()

        print("✅ Initializing SHAP explainer...")
        success = explainer.initialize_explainer(
            model=model,
            background_data=background_data,
            feature_names=feature_names
        )

        if success:
            print("✅ SHAP explainer initialized successfully")

            # Test prediction explanation
            print("\n🔍 Explaining predictions...")
            explanation = explainer.explain_prediction(test_data[:1])
            if explanation:
                print(f"✅ Explained prediction with {len(explanation['shap_values'][0][0])} features")

            # Test feature importance
            print("\n📊 Calculating feature importance...")
            importance = explainer.explain_feature_importance(test_data)
            if importance:
                print("✅ Feature importance calculated")
                top_features = importance['feature_importance'][:3]
                print("Top 3 features:")
                for feature, score in top_features:
                    print(f"  {feature}: {score:.4f}")

            # Test trading signal explanation
            print("\n📈 Explaining trading signal...")
            signal_exp = explainer.explain_trading_signal(test_data[:1], "BUY")
            if signal_exp:
                print("✅ Trading signal explained")
                print(f"Signal: {signal_exp['signal_type']}")
                print(".4f")

            print("\n🎉 SHAP integration demo completed!")

        else:
            print("❌ Failed to initialize SHAP explainer")

    except Exception as e:
        print(f"❌ SHAP integration failed: {e}")
        import traceback
        traceback.print_exc()