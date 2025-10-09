"""
AutoML with Optuna for Hyperparameter Optimization

Provides automated hyperparameter tuning for ML models using Optuna.
Supports both scikit-learn and PyTorch models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import logging

try:
    import optuna
    from optuna.pruners import MedianPruner, HyperbandPruner
    from optuna.samplers import TPESampler, RandomSampler, GridSampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️  Optuna not available. AutoML features will be disabled.")
    print("   Install with: pip install optuna")

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from config.settings import (
    SYMBOL, TIMEFRAME, ML_AUTOML_N_TRIALS, ML_AUTOML_TIMEOUT,
    ML_AUTOML_STUDY_NAME, ML_AUTOML_STORAGE, ML_AUTOML_SAMPLER,
    ML_AUTOML_PRUNER
)


class AutoMLOptimizer:
    """
    Automated hyperparameter optimization using Optuna.
    
    Features:
    - Multiple optimization strategies (TPE, Random, Grid)
    - Pruning for efficient search
    - Multi-objective optimization support
    - Visualization and analysis tools
    """
    
    def __init__(self, model_type: str = 'random_forest', symbol: str = SYMBOL,
                 timeframe: str = TIMEFRAME, results_dir: str = "output/automl"):
        """
        Initialize AutoML optimizer.
        
        Args:
            model_type: Type of model to optimize ('random_forest', 'gradient_boosting', 'pytorch')
            symbol: Trading symbol
            timeframe: Timeframe
            results_dir: Directory to store results
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required. Install with: pip install optuna")
        
        self.model_type = model_type
        self.symbol = symbol
        self.timeframe = timeframe
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.study = None
        self.best_params = None
        
        # Suppress Optuna logging for cleaner output
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def _get_sampler(self, sampler_type: str):
        """Get Optuna sampler based on configuration."""
        if sampler_type == 'tpe':
            return TPESampler(seed=42)
        elif sampler_type == 'random':
            return RandomSampler(seed=42)
        else:
            return TPESampler(seed=42)
    
    def _get_pruner(self, pruner_type: str):
        """Get Optuna pruner based on configuration."""
        if pruner_type == 'median':
            return MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        elif pruner_type == 'hyperband':
            return HyperbandPruner(min_resource=1, max_resource=100, reduction_factor=3)
        elif pruner_type == 'none':
            return None
        else:
            return MedianPruner()
    
    def optimize_random_forest(self, X: np.ndarray, y: np.ndarray,
                               n_trials: int = ML_AUTOML_N_TRIALS,
                               timeout: int = ML_AUTOML_TIMEOUT) -> Dict[str, Any]:
        """
        Optimize Random Forest hyperparameters.
        
        Args:
            X: Feature matrix
            y: Labels
            n_trials: Number of trials
            timeout: Maximum optimization time
            
        Returns:
            Dictionary with best parameters and results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required")
        
        def objective(trial):
            """Objective function for Optuna."""
            # Suggest hyperparameters
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'random_state': 42,
                'n_jobs': -1
            }
            
            # Create model
            model = RandomForestClassifier(**params)
            
            # Cross-validation score
            scores = cross_val_score(model, X, y, cv=5, scoring='f1', n_jobs=-1)
            
            return scores.mean()
        
        # Create study
        sampler = self._get_sampler(ML_AUTOML_SAMPLER)
        pruner = self._get_pruner(ML_AUTOML_PRUNER)
        
        self.study = optuna.create_study(
            study_name=f"{ML_AUTOML_STUDY_NAME}_rf_{self.symbol}_{self.timeframe}",
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            storage=ML_AUTOML_STORAGE,
            load_if_exists=True
        )
        
        print(f"🔍 Starting Random Forest optimization ({n_trials} trials)...")
        
        # Optimize
        self.study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
        
        self.best_params = self.study.best_params
        
        results = {
            'model_type': 'random_forest',
            'best_params': self.best_params,
            'best_score': self.study.best_value,
            'n_trials': len(self.study.trials),
            'best_trial': self.study.best_trial.number,
            'optimization_history': self._get_optimization_history()
        }
        
        print(f"✅ Optimization complete! Best F1 Score: {self.study.best_value:.4f}")
        print(f"📊 Best parameters: {self.best_params}")
        
        return results
    
    def optimize_gradient_boosting(self, X: np.ndarray, y: np.ndarray,
                                   n_trials: int = ML_AUTOML_N_TRIALS,
                                   timeout: int = ML_AUTOML_TIMEOUT) -> Dict[str, Any]:
        """
        Optimize Gradient Boosting hyperparameters.
        
        Args:
            X: Feature matrix
            y: Labels
            n_trials: Number of trials
            timeout: Maximum optimization time
            
        Returns:
            Dictionary with best parameters and results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required")
        
        def objective(trial):
            """Objective function for Optuna."""
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': 42
            }
            
            model = GradientBoostingClassifier(**params)
            scores = cross_val_score(model, X, y, cv=5, scoring='f1', n_jobs=-1)
            
            return scores.mean()
        
        # Create study
        sampler = self._get_sampler(ML_AUTOML_SAMPLER)
        pruner = self._get_pruner(ML_AUTOML_PRUNER)
        
        self.study = optuna.create_study(
            study_name=f"{ML_AUTOML_STUDY_NAME}_gb_{self.symbol}_{self.timeframe}",
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            storage=ML_AUTOML_STORAGE,
            load_if_exists=True
        )
        
        print(f"🔍 Starting Gradient Boosting optimization ({n_trials} trials)...")
        
        self.study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
        
        self.best_params = self.study.best_params
        
        results = {
            'model_type': 'gradient_boosting',
            'best_params': self.best_params,
            'best_score': self.study.best_value,
            'n_trials': len(self.study.trials),
            'best_trial': self.study.best_trial.number,
            'optimization_history': self._get_optimization_history()
        }
        
        print(f"✅ Optimization complete! Best F1 Score: {self.study.best_value:.4f}")
        print(f"📊 Best parameters: {self.best_params}")
        
        return results
    
    def optimize_pytorch_model(self, train_func: Callable, X: np.ndarray, y: np.ndarray,
                               n_trials: int = ML_AUTOML_N_TRIALS,
                               timeout: int = ML_AUTOML_TIMEOUT) -> Dict[str, Any]:
        """
        Optimize PyTorch model hyperparameters.
        
        Args:
            train_func: Function that trains and evaluates model
            X: Feature matrix
            y: Labels
            n_trials: Number of trials
            timeout: Maximum optimization time
            
        Returns:
            Dictionary with best parameters and results
        """
        def objective(trial):
            """Objective function for Optuna."""
            params = {
                'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256, 512]),
                'num_layers': trial.suggest_int('num_layers', 1, 4),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            }
            
            # Call user-provided training function
            score = train_func(X, y, params, trial)
            
            return score
        
        # Create study
        sampler = self._get_sampler(ML_AUTOML_SAMPLER)
        pruner = self._get_pruner(ML_AUTOML_PRUNER)
        
        self.study = optuna.create_study(
            study_name=f"{ML_AUTOML_STUDY_NAME}_pytorch_{self.symbol}_{self.timeframe}",
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            storage=ML_AUTOML_STORAGE,
            load_if_exists=True
        )
        
        print(f"🔍 Starting PyTorch model optimization ({n_trials} trials)...")
        
        self.study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
        
        self.best_params = self.study.best_params
        
        results = {
            'model_type': 'pytorch',
            'best_params': self.best_params,
            'best_score': self.study.best_value,
            'n_trials': len(self.study.trials),
            'best_trial': self.study.best_trial.number,
            'optimization_history': self._get_optimization_history()
        }
        
        print(f"✅ Optimization complete! Best Score: {self.study.best_value:.4f}")
        print(f"📊 Best parameters: {self.best_params}")
        
        return results
    
    def _get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history from study."""
        if self.study is None:
            return []
        
        history = []
        for trial in self.study.trials:
            history.append({
                'trial_number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name
            })
        
        return history
    
    def get_parameter_importance(self) -> Dict[str, float]:
        """
        Calculate parameter importance.
        
        Returns:
            Dictionary mapping parameter names to importance scores
        """
        if self.study is None or len(self.study.trials) < 10:
            return {}
        
        try:
            importance = optuna.importance.get_param_importances(self.study)
            return importance
        except Exception as e:
            print(f"⚠️  Failed to calculate importance: {e}")
            return {}
    
    def visualize_optimization(self, save_path: Optional[str] = None):
        """
        Create visualization of optimization results.
        
        Args:
            save_path: Path to save visualizations
        """
        if self.study is None:
            print("⚠️  No study available for visualization")
            return
        
        try:
            from optuna.visualization import (
                plot_optimization_history,
                plot_param_importances,
                plot_parallel_coordinate,
                plot_contour
            )
            import plotly.io as pio
            
            # Create visualizations
            figs = {
                'history': plot_optimization_history(self.study),
                'importance': plot_param_importances(self.study),
                'parallel': plot_parallel_coordinate(self.study),
                'contour': plot_contour(self.study)
            }
            
            if save_path:
                save_dir = Path(save_path)
                save_dir.mkdir(parents=True, exist_ok=True)
                
                for name, fig in figs.items():
                    fig_path = save_dir / f"{self.model_type}_{name}.html"
                    pio.write_html(fig, str(fig_path))
                    print(f"💾 Saved {name} plot to {fig_path}")
            else:
                # Display in notebook/browser
                for name, fig in figs.items():
                    fig.show()
                    
        except ImportError:
            print("⚠️  Plotly required for visualization: pip install plotly")
        except Exception as e:
            print(f"⚠️  Visualization failed: {e}")
    
    def save_study(self, path: Optional[str] = None):
        """Save study results to file."""
        if self.study is None:
            return
        
        if path is None:
            path = self.results_dir / f"{self.model_type}_{self.symbol}_{self.timeframe}_study.pkl"
        
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.study, f)
        
        print(f"💾 Study saved to {path}")
    
    def load_study(self, path: str):
        """Load study from file."""
        import pickle
        with open(path, 'rb') as f:
            self.study = pickle.load(f)
        
        self.best_params = self.study.best_params
        print(f"📂 Study loaded from {path}")


# Global optimizer instance
automl_optimizer = None


def get_automl_optimizer(model_type: str = 'random_forest', symbol: str = SYMBOL,
                         timeframe: str = TIMEFRAME) -> Optional[AutoMLOptimizer]:
    """Get or create AutoML optimizer instance."""
    global automl_optimizer
    
    if automl_optimizer is None and OPTUNA_AVAILABLE:
        try:
            automl_optimizer = AutoMLOptimizer(
                model_type=model_type,
                symbol=symbol,
                timeframe=timeframe
            )
        except Exception as e:
            print(f"❌ Failed to initialize AutoML optimizer: {e}")
            return None
    
    return automl_optimizer


def is_automl_available() -> bool:
    """Check if AutoML features are available."""
    return OPTUNA_AVAILABLE


def run_automl_optimization(df: pd.DataFrame, model_type: str = 'random_forest',
                           symbol: str = SYMBOL, timeframe: str = TIMEFRAME,
                           n_trials: int = ML_AUTOML_N_TRIALS) -> Dict[str, Any]:
    """
    Convenience function to run AutoML optimization.
    
    Args:
        df: DataFrame with features and labels
        model_type: Type of model to optimize
        symbol: Trading symbol
        timeframe: Timeframe
        n_trials: Number of optimization trials
        
    Returns:
        Optimization results
    """
    if not OPTUNA_AVAILABLE:
        return {'error': 'Optuna not available'}
    
    optimizer = get_automl_optimizer(model_type, symbol, timeframe)
    if optimizer is None:
        return {'error': 'Failed to initialize optimizer'}
    
    # Prepare data (assumes features are in df and labels are computed)
    from src.ml_predictor import MLSignalPredictor
    
    predictor = MLSignalPredictor(symbol=symbol, timeframe=timeframe)
    X, feature_names = predictor.prepare_features(df)
    y = predictor.create_labels(df)
    
    # Remove NaN
    valid_indices = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[valid_indices]
    y = y[valid_indices]
    
    # Run optimization
    if model_type == 'random_forest':
        results = optimizer.optimize_random_forest(X, y, n_trials=n_trials)
    elif model_type == 'gradient_boosting':
        results = optimizer.optimize_gradient_boosting(X, y, n_trials=n_trials)
    else:
        return {'error': f'Unknown model type: {model_type}'}
    
    # Save results
    results_path = optimizer.results_dir / f"{model_type}_{symbol}_{timeframe}_results.json"
    import json
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to {results_path}")
    
    return results
