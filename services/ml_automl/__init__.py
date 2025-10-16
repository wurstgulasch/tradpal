"""
ML AutoML Service - Automated machine learning for trading.

Provides automated ML pipeline for model selection and hyperparameter tuning.
"""

import logging
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Check if AutoML libraries are available
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available. AutoML features will be limited.")

try:
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available. Basic ML models disabled.")


class AutoMLPipeline:
    """Automated ML pipeline for trading models."""

    def __init__(self):
        self.best_model = None
        self.best_params = {}
        self.is_trained = False
        self.study = None

        # Available models for AutoML
        self.model_configs = {
            'rf': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'gb': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            },
            'lr': {
                'model': LogisticRegression(random_state=42),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2']
                }
            },
            'svm': {
                'model': SVC(probability=True, random_state=42),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': ['scale', 'auto']
                }
            }
        }

    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray,
                               model_type: str = 'rf', n_trials: int = 50) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, using default parameters")
            return self._get_default_params(model_type)

        try:
            def objective(trial):
                # Sample hyperparameters
                params = {}
                config = self.model_configs.get(model_type, self.model_configs['rf'])

                for param_name, param_values in config['params'].items():
                    if isinstance(param_values[0], int):
                        params[param_name] = trial.suggest_int(param_name, min(param_values), max(param_values))
                    elif isinstance(param_values[0], float):
                        params[param_name] = trial.suggest_float(param_name, min(param_values), max(param_values))
                    else:
                        params[param_name] = trial.suggest_categorical(param_name, param_values)

                # Create and evaluate model
                model = config['model'].__class__(**params, random_state=42)
                scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
                return scores.mean()

            # Run optimization
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)

            self.study = study
            self.best_params = study.best_params

            return {
                'best_params': study.best_params,
                'best_score': study.best_value,
                'n_trials': n_trials,
                'model_type': model_type
            }

        except Exception as e:
            logger.error(f"Hyperparameter optimization failed: {e}")
            return self._get_default_params(model_type)

    def _get_default_params(self, model_type: str) -> Dict[str, Any]:
        """Get default parameters for a model type."""
        defaults = {
            'rf': {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2},
            'gb': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3},
            'lr': {'C': 1.0, 'penalty': 'l2'},
            'svm': {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'}
        }
        return defaults.get(model_type, defaults['rf'])

    def train_best_model(self, X: np.ndarray, y: np.ndarray, model_type: str = 'rf') -> bool:
        """Train the best model found by optimization."""
        try:
            if not self.best_params:
                self.best_params = self._get_default_params(model_type)

            config = self.model_configs.get(model_type, self.model_configs['rf'])
            self.best_model = config['model'].__class__(**self.best_params, random_state=42)
            self.best_model.fit(X, y)
            self.is_trained = True

            return True

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return False

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the best model."""
        if not self.is_trained or self.best_model is None:
            return np.zeros(len(X))

        try:
            return self.best_model.predict(X)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return np.zeros(len(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities with the best model."""
        if not self.is_trained or self.best_model is None:
            return np.full((len(X), 2), 0.5)

        try:
            if hasattr(self.best_model, 'predict_proba'):
                return self.best_model.predict_proba(X)
            else:
                # For models without predict_proba, use decision function
                decision = self.best_model.decision_function(X)
                prob_pos = 1 / (1 + np.exp(-decision))
                prob_neg = 1 - prob_pos
                return np.column_stack([prob_neg, prob_pos])
        except Exception as e:
            logger.error(f"Probability prediction failed: {e}")
            return np.full((len(X), 2), 0.5)


def is_automl_available() -> bool:
    """Check if AutoML functionality is available."""
    return OPTUNA_AVAILABLE and SKLEARN_AVAILABLE


def create_automl_pipeline(model_type: str = 'rf') -> Optional[AutoMLPipeline]:
    """Create AutoML pipeline instance."""
    if not is_automl_available():
        return None

    try:
        return AutoMLPipeline()
    except Exception as e:
        logger.error(f"Failed to create AutoML pipeline: {e}")
        return None


def run_automl_optimization(X: np.ndarray, y: np.ndarray,
                          model_types: List[str] = None,
                          n_trials: int = 50) -> Dict[str, Any]:
    """Run AutoML optimization across multiple model types."""
    if model_types is None:
        model_types = ['rf', 'gb', 'lr']

    results = {}

    for model_type in model_types:
        try:
            pipeline = AutoMLPipeline()
            opt_result = pipeline.optimize_hyperparameters(X, y, model_type, n_trials)
            pipeline.train_best_model(X, y, model_type)

            results[model_type] = {
                'pipeline': pipeline,
                'optimization_result': opt_result,
                'best_score': opt_result.get('best_score', 0.0)
            }

        except Exception as e:
            logger.error(f"AutoML optimization failed for {model_type}: {e}")
            results[model_type] = {'error': str(e)}

    # Find best overall model
    best_model_type = max(results.keys(),
                         key=lambda k: results[k].get('best_score', 0.0))

    return {
        'results': results,
        'best_model_type': best_model_type,
        'best_pipeline': results[best_model_type].get('pipeline'),
        'best_score': results[best_model_type].get('best_score', 0.0)
    }