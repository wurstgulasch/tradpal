#!/usr/bin/env python3
"""
SHAP Integration Service for ML Model Interpretability.

This service provides comprehensive model interpretability using SHAP (SHapley Additive exPlanations)
to explain ML model predictions in trading contexts. Key features:

- Global feature importance analysis
- Local prediction explanations
- Waterfall plots for individual predictions
- Summary plots for feature relationships
- Trading-specific interpretability metrics
- Confidence intervals for SHAP values
- Integration with ensemble models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import pickle
import json
from datetime import datetime

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")

from config.settings import ML_MODELS_DIR

logger = logging.getLogger(__name__)


class SHAPInterpreter:
    """SHAP-based model interpretability for trading ML models."""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize SHAP interpreter.

        Args:
            model_path: Path to saved ML model for interpretation
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for model interpretability. Install with: pip install shap")

        self.model_path = model_path
        self.model = None
        self.explainer = None
        self.feature_names = None
        self.background_data = None
        self.shap_values_cache = {}

        # Interpretability results
        self.global_importance = {}
        self.local_explanations = {}
        self.feature_interactions = {}

        logger.info("SHAP Interpreter initialized")

    def load_model(self, model_path: str, feature_names: List[str]) -> bool:
        """
        Load ML model for SHAP interpretation.

        Args:
            model_path: Path to saved model
            feature_names: List of feature names

        Returns:
            Success status
        """
        try:
            # Load model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)

            self.feature_names = feature_names
            self.model_path = model_path

            # Initialize SHAP explainer based on model type
            self._initialize_explainer()

            logger.info(f"Model loaded for SHAP interpretation: {model_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model for SHAP: {e}")
            return False

    def _initialize_explainer(self):
        """Initialize appropriate SHAP explainer based on model type."""
        if self.model is None:
            raise ValueError("Model not loaded")

        try:
            # Choose appropriate explainer based on model type
            if hasattr(self.model, 'predict_proba'):
                # For tree-based models, use TreeExplainer (fastest and most accurate)
                if hasattr(self.model, 'feature_importances_'):
                    self.explainer = shap.TreeExplainer(self.model)
                    logger.info("Using TreeExplainer for tree-based model")
                else:
                    # For other models, use KernelExplainer (slower but more general)
                    # We'll need background data for this
                    self.explainer = shap.KernelExplainer(self.model.predict_proba, self.background_data)
                    logger.info("Using KernelExplainer for general model")
            else:
                # For regression models
                if hasattr(self.model, 'feature_importances_'):
                    self.explainer = shap.TreeExplainer(self.model)
                else:
                    self.explainer = shap.KernelExplainer(self.model.predict, self.background_data)

        except Exception as e:
            logger.warning(f"Failed to initialize TreeExplainer, falling back to KernelExplainer: {e}")
            # Fallback to KernelExplainer
            if hasattr(self.model, 'predict_proba'):
                self.explainer = shap.KernelExplainer(self.model.predict_proba, self.background_data)
            else:
                self.explainer = shap.KernelExplainer(self.model.predict, self.background_data)

    def set_background_data(self, background_data: np.ndarray, max_samples: int = 100):
        """
        Set background data for SHAP explanations.

        Args:
            background_data: Representative sample of training data
            max_samples: Maximum samples to use for background
        """
        if len(background_data) > max_samples:
            # Sample representative background data
            indices = np.random.choice(len(background_data), max_samples, replace=False)
            self.background_data = background_data[indices]
        else:
            self.background_data = background_data

        logger.info(f"Background data set with {len(self.background_data)} samples")

    def explain_prediction(self, X: np.ndarray, instance_idx: Optional[int] = None) -> Dict[str, Any]:
        """
        Explain a single prediction using SHAP.

        Args:
            X: Feature matrix (single instance or batch)
            instance_idx: Index of specific instance to explain (for batch explanations)

        Returns:
            SHAP explanation results
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized")

        try:
            # Handle single instance vs batch
            if len(X.shape) == 1:
                X = X.reshape(1, -1)

            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X)

            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # Multi-class case
                if instance_idx is not None:
                    explanation = self._format_multiclass_explanation(shap_values, X, instance_idx)
                else:
                    explanation = self._format_multiclass_explanation(shap_values, X, 0)
            else:
                # Single output case
                explanation = self._format_single_explanation(shap_values, X, instance_idx or 0)

            return explanation

        except Exception as e:
            logger.error(f"Failed to explain prediction: {e}")
            return {"error": str(e)}

    def _format_single_explanation(self, shap_values: np.ndarray, X: np.ndarray, instance_idx: int) -> Dict[str, Any]:
        """Format SHAP explanation for single output."""
        instance_shap = shap_values[instance_idx]
        instance_X = X[instance_idx]

        # Create feature importance dictionary
        feature_importance = {}
        for i, feature_name in enumerate(self.feature_names):
            feature_importance[feature_name] = {
                'shap_value': float(instance_shap[i]),
                'feature_value': float(instance_X[i]),
                'impact': 'positive' if instance_shap[i] > 0 else 'negative'
            }

        # Sort by absolute SHAP value
        sorted_features = sorted(feature_importance.items(),
                               key=lambda x: abs(x[1]['shap_value']), reverse=True)

        # Calculate prediction confidence
        base_value = float(self.explainer.expected_value)
        prediction = base_value + np.sum(instance_shap)

        return {
            'prediction': float(prediction),
            'base_value': base_value,
            'total_shap': float(np.sum(instance_shap)),
            'feature_importance': dict(sorted_features),
            'top_positive_features': [f for f, v in sorted_features if v['shap_value'] > 0][:5],
            'top_negative_features': [f for f, v in sorted_features if v['shap_value'] < 0][:5],
            'explanation_type': 'single_output'
        }

    def _format_multiclass_explanation(self, shap_values: List[np.ndarray], X: np.ndarray, instance_idx: int) -> Dict[str, Any]:
        """Format SHAP explanation for multi-class output."""
        # For trading, we're typically interested in the positive class (buy signal)
        positive_class_shap = shap_values[1][instance_idx]  # Assuming binary classification
        instance_X = X[instance_idx]

        feature_importance = {}
        for i, feature_name in enumerate(self.feature_names):
            feature_importance[feature_name] = {
                'shap_value': float(positive_class_shap[i]),
                'feature_value': float(instance_X[i]),
                'impact': 'positive' if positive_class_shap[i] > 0 else 'negative'
            }

        sorted_features = sorted(feature_importance.items(),
                               key=lambda x: abs(x[1]['shap_value']), reverse=True)

        base_value = float(self.explainer.expected_value[1])
        prediction = base_value + np.sum(positive_class_shap)

        return {
            'prediction': float(prediction),
            'base_value': base_value,
            'total_shap': float(np.sum(positive_class_shap)),
            'feature_importance': dict(sorted_features),
            'top_positive_features': [f for f, v in sorted_features if v['shap_value'] > 0][:5],
            'top_negative_features': [f for f, v in sorted_features if v['shap_value'] < 0][:5],
            'explanation_type': 'multiclass_positive_class'
        }

    def explain_global_importance(self, X: np.ndarray, max_evals: int = 1000) -> Dict[str, Any]:
        """
        Calculate global feature importance using SHAP.

        Args:
            X: Feature matrix from training/validation data
            max_evals: Maximum evaluations for approximation

        Returns:
            Global feature importance analysis
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized")

        try:
            # Sample data for efficiency if too large
            if len(X) > max_evals:
                indices = np.random.choice(len(X), max_evals, replace=False)
                X_sample = X[indices]
            else:
                X_sample = X

            # Calculate SHAP values for global analysis
            shap_values = self.explainer.shap_values(X_sample)

            # Handle different output formats
            if isinstance(shap_values, list):
                # Multi-class case - use positive class
                shap_matrix = shap_values[1]
            else:
                shap_matrix = shap_values

            # Calculate mean absolute SHAP values for each feature
            mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)

            # Create global importance dictionary
            global_importance = {}
            for i, feature_name in enumerate(self.feature_names):
                global_importance[feature_name] = {
                    'mean_abs_shap': float(mean_abs_shap[i]),
                    'importance_rank': 0,  # Will be set below
                    'percentage': 0.0  # Will be set below
                }

            # Sort and rank features
            sorted_features = sorted(global_importance.items(),
                                   key=lambda x: x[1]['mean_abs_shap'], reverse=True)

            total_importance = sum(abs_shap for _, data in sorted_features for abs_shap in [data['mean_abs_shap']])

            for rank, (feature_name, data) in enumerate(sorted_features, 1):
                global_importance[feature_name]['importance_rank'] = rank
                global_importance[feature_name]['percentage'] = (data['mean_abs_shap'] / total_importance) * 100

            self.global_importance = global_importance

            return {
                'global_importance': global_importance,
                'top_features': [f for f, _ in sorted_features[:10]],
                'total_importance': total_importance,
                'sample_size': len(X_sample)
            }

        except Exception as e:
            logger.error(f"Failed to calculate global importance: {e}")
            return {"error": str(e)}

    def analyze_feature_interactions(self, X: np.ndarray, feature_pairs: List[Tuple[str, str]] = None) -> Dict[str, Any]:
        """
        Analyze feature interactions using SHAP.

        Args:
            X: Feature matrix
            feature_pairs: Specific feature pairs to analyze (optional)

        Returns:
            Feature interaction analysis
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized")

        try:
            # For tree models, we can use tree-specific interaction analysis
            if hasattr(self.explainer, 'shap_interaction_values'):
                shap_interaction = self.explainer.shap_interaction_values(X)

                # Analyze top feature interactions
                interactions = {}
                n_features = len(self.feature_names)

                for i in range(n_features):
                    for j in range(i+1, n_features):
                        interaction_strength = np.mean(np.abs(shap_interaction[:, i, j]))
                        feature_pair = (self.feature_names[i], self.feature_names[j])

                        interactions[feature_pair] = {
                            'interaction_strength': float(interaction_strength),
                            'feature_1': self.feature_names[i],
                            'feature_2': self.feature_names[j]
                        }

                # Sort by interaction strength
                sorted_interactions = sorted(interactions.items(),
                                           key=lambda x: x[1]['interaction_strength'], reverse=True)

                self.feature_interactions = dict(sorted_interactions[:20])  # Top 20 interactions

                return {
                    'feature_interactions': self.feature_interactions,
                    'top_interactions': [pair for pair, _ in sorted_interactions[:10]],
                    'analysis_type': 'tree_interaction'
                }
            else:
                return {
                    'message': 'Feature interaction analysis not available for this model type',
                    'analysis_type': 'not_available'
                }

        except Exception as e:
            logger.error(f"Failed to analyze feature interactions: {e}")
            return {"error": str(e)}

    def explain_trading_decision(self, features: Dict[str, float], prediction: float) -> Dict[str, Any]:
        """
        Provide trading-specific explanation for a model decision.

        Args:
            features: Feature values for the prediction
            prediction: Model prediction (probability or score)

        Returns:
            Trading-focused explanation
        """
        try:
            # Convert features to array
            feature_values = [features.get(name, 0.0) for name in self.feature_names]
            X = np.array(feature_values).reshape(1, -1)

            # Get SHAP explanation
            explanation = self.explain_prediction(X)

            if 'error' in explanation:
                return explanation

            # Add trading-specific interpretation
            trading_interpretation = self._interpret_trading_signals(explanation, features)

            explanation['trading_interpretation'] = trading_interpretation

            return explanation

        except Exception as e:
            logger.error(f"Failed to explain trading decision: {e}")
            return {"error": str(e)}

    def _interpret_trading_signals(self, explanation: Dict[str, Any], features: Dict[str, float]) -> Dict[str, Any]:
        """Interpret SHAP values in trading context."""
        feature_importance = explanation.get('feature_importance', {})

        # Categorize features by type
        technical_features = []
        momentum_features = []
        volatility_features = []
        volume_features = []

        for feature_name in feature_importance.keys():
            if any(term in feature_name.lower() for term in ['sma', 'ema', 'rsi', 'macd', 'stoch', 'williams', 'cci', 'adx']):
                technical_features.append(feature_name)
            elif any(term in feature_name.lower() for term in ['momentum', 'roc']):
                momentum_features.append(feature_name)
            elif any(term in feature_name.lower() for term in ['volatility', 'atr']):
                volatility_features.append(feature_name)
            elif any(term in feature_name.lower() for term in ['volume', 'obv', 'cmf']):
                volume_features.append(feature_name)

        # Calculate category contributions
        def category_contribution(feature_list):
            return sum(abs(feature_importance.get(f, {}).get('shap_value', 0)) for f in feature_list)

        total_contribution = sum(abs(v.get('shap_value', 0)) for v in feature_importance.values())

        interpretation = {
            'signal_strength': 'strong' if explanation.get('prediction', 0) > 0.7 else 'moderate' if explanation.get('prediction', 0) > 0.6 else 'weak',
            'confidence_level': 'high' if abs(explanation.get('total_shap', 0)) > 0.5 else 'medium' if abs(explanation.get('total_shap', 0)) > 0.2 else 'low',
            'feature_categories': {
                'technical_indicators': {
                    'contribution': category_contribution(technical_features),
                    'percentage': (category_contribution(technical_features) / total_contribution * 100) if total_contribution > 0 else 0,
                    'top_features': sorted(technical_features, key=lambda x: abs(feature_importance.get(x, {}).get('shap_value', 0)), reverse=True)[:3]
                },
                'momentum_signals': {
                    'contribution': category_contribution(momentum_features),
                    'percentage': (category_contribution(momentum_features) / total_contribution * 100) if total_contribution > 0 else 0,
                    'top_features': sorted(momentum_features, key=lambda x: abs(feature_importance.get(x, {}).get('shap_value', 0)), reverse=True)[:3]
                },
                'volatility_measures': {
                    'contribution': category_contribution(volatility_features),
                    'percentage': (category_contribution(volatility_features) / total_contribution * 100) if total_contribution > 0 else 0,
                    'top_features': sorted(volatility_features, key=lambda x: abs(feature_importance.get(x, {}).get('shap_value', 0)), reverse=True)[:3]
                },
                'volume_indicators': {
                    'contribution': category_contribution(volume_features),
                    'percentage': (category_contribution(volume_features) / total_contribution * 100) if total_contribution > 0 else 0,
                    'top_features': sorted(volume_features, key=lambda x: abs(feature_importance.get(x, {}).get('shap_value', 0)), reverse=True)[:3]
                }
            },
            'decision_factors': {
                'bullish_drivers': [f for f, v in feature_importance.items() if v.get('shap_value', 0) > 0][:3],
                'bearish_drivers': [f for f, v in feature_importance.items() if v.get('shap_value', 0) < 0][:3]
            }
        }

        return interpretation

    def generate_interpretability_report(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Generate comprehensive interpretability report for a model.

        Args:
            model_name: Name of the model
            X_test: Test feature matrix
            y_test: Test target values

        Returns:
            Complete interpretability report
        """
        try:
            report = {
                'model_name': model_name,
                'generated_at': datetime.now().isoformat(),
                'shap_available': SHAP_AVAILABLE,
                'report_sections': {}
            }

            if not SHAP_AVAILABLE:
                report['error'] = 'SHAP not available'
                return report

            # Global feature importance
            logger.info("Calculating global feature importance...")
            global_importance = self.explain_global_importance(X_test)
            report['report_sections']['global_importance'] = global_importance

            # Sample predictions explanation
            logger.info("Explaining sample predictions...")
            sample_indices = np.random.choice(len(X_test), min(10, len(X_test)), replace=False)
            sample_explanations = []

            for idx in sample_indices:
                explanation = self.explain_prediction(X_test[idx:idx+1], 0)
                explanation['actual_value'] = float(y_test[idx])
                explanation['prediction_error'] = abs(explanation.get('prediction', 0) - y_test[idx])
                sample_explanations.append(explanation)

            report['report_sections']['sample_explanations'] = sample_explanations

            # Feature interactions (if available)
            logger.info("Analyzing feature interactions...")
            interactions = self.analyze_feature_interactions(X_test[:min(1000, len(X_test))])
            report['report_sections']['feature_interactions'] = interactions

            # Model reliability assessment
            logger.info("Assessing model reliability...")
            reliability = self._assess_model_reliability(sample_explanations)
            report['report_sections']['reliability_assessment'] = reliability

            # Trading-specific insights
            logger.info("Generating trading insights...")
            trading_insights = self._generate_trading_insights(global_importance, sample_explanations)
            report['report_sections']['trading_insights'] = trading_insights

            logger.info(f"Interpretability report generated for {model_name}")
            return report

        except Exception as e:
            logger.error(f"Failed to generate interpretability report: {e}")
            return {
                'model_name': model_name,
                'error': str(e),
                'generated_at': datetime.now().isoformat()
            }

    def _assess_model_reliability(self, sample_explanations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess model reliability based on explanation consistency."""
        if not sample_explanations:
            return {'error': 'No explanations available'}

        # Calculate explanation stability metrics
        prediction_errors = [exp.get('prediction_error', 0) for exp in sample_explanations]
        shap_consistency = []

        # Check if top features are consistent across explanations
        top_features_list = [exp.get('top_positive_features', [])[:3] for exp in sample_explanations]
        feature_stability = self._calculate_feature_stability(top_features_list)

        return {
            'mean_prediction_error': float(np.mean(prediction_errors)),
            'prediction_error_std': float(np.std(prediction_errors)),
            'feature_stability_score': feature_stability,
            'explanation_consistency': 'high' if feature_stability > 0.7 else 'medium' if feature_stability > 0.5 else 'low',
            'sample_size': len(sample_explanations)
        }

    def _calculate_feature_stability(self, feature_lists: List[List[str]]) -> float:
        """Calculate stability score for feature importance across samples."""
        if not feature_lists:
            return 0.0

        # Count how often each feature appears in top positions
        feature_counts = {}
        total_positions = 0

        for feature_list in feature_lists:
            for feature in feature_list:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
                total_positions += 1

        if total_positions == 0:
            return 0.0

        # Calculate concentration (Herfindahl-Hirschman Index style)
        concentrations = [(count / total_positions) ** 2 for count in feature_counts.values()]
        stability_score = sum(concentrations)

        return min(stability_score * len(feature_counts), 1.0)  # Normalize

    def _generate_trading_insights(self, global_importance: Dict[str, Any], sample_explanations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate trading-specific insights from interpretability analysis."""
        insights = {
            'key_drivers': [],
            'risk_factors': [],
            'model_characteristics': {},
            'recommendations': []
        }

        # Extract key drivers from global importance
        if 'global_importance' in global_importance:
            importance_data = global_importance['global_importance']
            sorted_features = sorted(importance_data.items(),
                                   key=lambda x: x[1].get('percentage', 0), reverse=True)

            insights['key_drivers'] = [f for f, _ in sorted_features[:5]]

        # Analyze prediction patterns
        high_confidence_predictions = [exp for exp in sample_explanations
                                     if exp.get('prediction', 0) > 0.8 or exp.get('prediction', 0) < 0.2]

        if len(high_confidence_predictions) > len(sample_explanations) * 0.3:
            insights['model_characteristics']['decision_confidence'] = 'high'
            insights['recommendations'].append('Model shows strong conviction in predictions - suitable for high-confidence strategies')
        else:
            insights['model_characteristics']['decision_confidence'] = 'moderate'
            insights['recommendations'].append('Model predictions may need additional filtering for confidence')

        # Risk assessment
        if len(insights['key_drivers']) > 0:
            volatility_drivers = [d for d in insights['key_drivers'] if 'volatility' in d.lower()]
            if len(volatility_drivers) > len(insights['key_drivers']) * 0.4:
                insights['risk_factors'].append('Heavy reliance on volatility indicators - monitor for changing market conditions')

        return insights

    def save_interpretability_results(self, model_name: str, results: Dict[str, Any]) -> bool:
        """
        Save interpretability results to disk.

        Args:
            model_name: Name of the model
            results: Interpretability results to save

        Returns:
            Success status
        """
        try:
            results_dir = Path(ML_MODELS_DIR) / model_name
            results_dir.mkdir(exist_ok=True)

            results_file = results_dir / "shap_interpretability.json"

            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"Interpretability results saved: {results_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to save interpretability results: {e}")
            return False

    def load_interpretability_results(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Load saved interpretability results.

        Args:
            model_name: Name of the model

        Returns:
            Loaded results or None if not found
        """
        try:
            results_file = Path(ML_MODELS_DIR) / model_name / "shap_interpretability.json"

            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                logger.info(f"Interpretability results loaded: {results_file}")
                return results
            else:
                logger.warning(f"Interpretability results not found: {results_file}")
                return None

        except Exception as e:
            logger.error(f"Failed to load interpretability results: {e}")
            return None