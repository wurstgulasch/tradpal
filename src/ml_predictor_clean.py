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
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None
    XGBClassifier = None
    print("⚠️  XGBoost not available. Enhanced ensemble methods will be limited.")
    print("   Install with: pip install xgboost")

try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None
    LGBMClassifier = None
    print("⚠️  LightGBM not available. Enhanced ensemble methods will be limited.")
    print("   Install with: pip install lightgbm")

try:
    import optuna
    from optuna.pruners import MedianPruner, HyperbandPruner
    from optuna.samplers import TPESampler, RandomSampler, GridSampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None
    TPESampler = None
    RandomSampler = None
    GridSampler = None
    MedianPruner = None
    HyperbandPruner = None
    print("⚠️  Optuna not available. AutoML features will be disabled.")
    print("   Install with: pip install optuna")

from config.settings import (
    SYMBOL, TIMEFRAME, LOOKBACK_DAYS,
    ML_PREFERRED_MODEL, ML_MODEL_SELECTION_CRITERIA,
    ML_GRADIENT_BOOSTING_N_ESTIMATORS, ML_GRADIENT_BOOSTING_LEARNING_RATE,
    ML_GRADIENT_BOOSTING_MAX_DEPTH, ML_GRADIENT_BOOSTING_MIN_SAMPLES_SPLIT,
    ML_GRADIENT_BOOSTING_MIN_SAMPLES_LEAF, ML_GRADIENT_BOOSTING_SUBSAMPLE,
    ML_GRADIENT_BOOSTING_MAX_FEATURES,
    ML_XGBOOST_N_ESTIMATORS, ML_XGBOOST_LEARNING_RATE, ML_XGBOOST_MAX_DEPTH,
    ML_XGBOOST_MIN_CHILD_WEIGHT, ML_XGBOOST_SUBSAMPLE, ML_XGBOOST_COLSAMPLE_BYTREE,
    ML_XGBOOST_GAMMA,
    ML_RF_N_ESTIMATORS, ML_RF_MAX_DEPTH, ML_RF_MIN_SAMPLES_SPLIT,
    ML_RF_MIN_SAMPLES_LEAF, ML_RF_MAX_FEATURES, ML_RF_BOOTSTRAP,
    ML_SVM_C, ML_SVM_KERNEL, ML_SVM_GAMMA, ML_SVM_CLASS_WEIGHT,
    ML_LR_C, ML_LR_PENALTY, ML_LR_SOLVER, ML_LR_MAX_ITER,
    ML_USE_AUTOML, ML_AUTOML_N_TRIALS, ML_AUTOML_TIMEOUT, ML_AUTOML_STUDY_NAME,
    ML_AUTOML_STORAGE, ML_AUTOML_SAMPLER, ML_AUTOML_PRUNER
)
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

        # Model configurations with optimized hyperparameters
        self.models = {
            'random_forest': {
                'model': RandomForestClassifier(
                    n_estimators=ML_RF_N_ESTIMATORS,
                    max_depth=ML_RF_MAX_DEPTH,
                    min_samples_split=ML_RF_MIN_SAMPLES_SPLIT,
                    min_samples_leaf=ML_RF_MIN_SAMPLES_LEAF,
                    max_features=ML_RF_MAX_FEATURES,
                    bootstrap=ML_RF_BOOTSTRAP,
                    random_state=42,
                    n_jobs=-1
                ),
                'name': 'Random Forest'
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(
                    n_estimators=ML_GRADIENT_BOOSTING_N_ESTIMATORS,
                    learning_rate=ML_GRADIENT_BOOSTING_LEARNING_RATE,
                    max_depth=ML_GRADIENT_BOOSTING_MAX_DEPTH,
                    min_samples_split=ML_GRADIENT_BOOSTING_MIN_SAMPLES_SPLIT,
                    min_samples_leaf=ML_GRADIENT_BOOSTING_MIN_SAMPLES_LEAF,
                    subsample=ML_GRADIENT_BOOSTING_SUBSAMPLE,
                    max_features=ML_GRADIENT_BOOSTING_MAX_FEATURES,
                    random_state=42
                ),
                'name': 'Gradient Boosting'
            },
            'xgboost': {
                'model': xgb.XGBClassifier(
                    n_estimators=ML_XGBOOST_N_ESTIMATORS,
                    learning_rate=ML_XGBOOST_LEARNING_RATE,
                    max_depth=ML_XGBOOST_MAX_DEPTH,
                    min_child_weight=ML_XGBOOST_MIN_CHILD_WEIGHT,
                    subsample=ML_XGBOOST_SUBSAMPLE,
                    colsample_bytree=ML_XGBOOST_COLSAMPLE_BYTREE,
                    gamma=ML_XGBOOST_GAMMA,
                    random_state=42,
                    n_jobs=-1
                ) if XGBOOST_AVAILABLE else None,
                'name': 'XGBoost'
            },
            'lightgbm': {
                'model': LGBMClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=-1
                ) if LIGHTGBM_AVAILABLE else None,
                'name': 'LightGBM'
            },
            'svm': {
                'model': Pipeline([
                    ('scaler', StandardScaler()),
                    ('svm', SVC(
                        kernel=ML_SVM_KERNEL,
                        C=ML_SVM_C,
                        gamma=ML_SVM_GAMMA,
                        class_weight=ML_SVM_CLASS_WEIGHT,
                        probability=True,
                        random_state=42
                    ))
                ]),
                'name': 'Support Vector Machine'
            },
            'logistic_regression': {
                'model': Pipeline([
                    ('scaler', StandardScaler()),
                    ('lr', LogisticRegression(
                        C=ML_LR_C,
                        penalty=ML_LR_PENALTY,
                        solver=ML_LR_SOLVER,
                        max_iter=ML_LR_MAX_ITER,
                        random_state=42
                    ))
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

        # Expected features from trained models - maintain compatibility
        expected_features = [
            'close_pct_roc',      # Price change rate of change
            'close_ma20_roc',     # 20-period MA rate of change
            'close_ma50_value',   # 50-period MA value
            'close_dev_ma20_roc', # Price deviation from MA rate of change
            'EMA9_value',         # EMA9 value
            'BB_upper_value',     # Bollinger Band upper value
            'BB_lower_roc',       # Bollinger Band lower rate of change
            'MACD_hist_roc',      # MACD histogram rate of change
            'Stoch_K_roc',        # Stochastic K rate of change
            'Stoch_D_roc',        # Stochastic D rate of change
            'BB_upper_lag1'       # Bollinger Band upper lagged by 1 period
        ]

        # Build features in the exact order expected by trained models
        if 'close' in df.columns:
            # close_pct_roc: Price change rate of change
            price_change = df['close'].pct_change().fillna(0)
            features.append(price_change.pct_change().fillna(0))

            # close_ma20_roc: 20-period MA rate of change
            ma20 = df['close'].rolling(window=20).mean().fillna(0)
            features.append(ma20.pct_change().fillna(0))

            # close_ma50_value: 50-period MA value
            ma50 = df['close'].rolling(window=50).mean().fillna(0)
            features.append(ma50)

            # close_dev_ma20_roc: Price deviation from MA rate of change
            price_dev = (df['close'] - ma20).fillna(0)
            features.append(price_dev.pct_change().fillna(0))

        # EMA9_value: EMA9 value
        if 'EMA9' in df.columns:
            features.append(df['EMA9'].fillna(0))

        # BB_upper_value: Bollinger Band upper value
        if 'BB_upper' in df.columns:
            features.append(df['BB_upper'].fillna(0))

        # BB_lower_roc: Bollinger Band lower rate of change
        if 'BB_lower' in df.columns:
            features.append(df['BB_lower'].pct_change().fillna(0))

        # MACD_hist_roc: MACD histogram rate of change
        if 'MACD_hist' in df.columns:
            features.append(df['MACD_hist'].pct_change().fillna(0))

        # Stoch_K_roc: Stochastic K rate of change
        if 'Stoch_K' in df.columns:
            features.append(df['Stoch_K'].pct_change().fillna(0))

        # Stoch_D_roc: Stochastic D rate of change
        if 'Stoch_D' in df.columns:
            features.append(df['Stoch_D'].pct_change().fillna(0))

        # BB_upper_lag1: Bollinger Band upper lagged by 1 period
        if 'BB_upper' in df.columns:
            features.append(df['BB_upper'].shift(1).fillna(0))

        # Ensure we have exactly the expected number of features
        if len(features) != len(expected_features):
            # Fallback: create zero features if we don't have all expected features
            print(f"⚠️  Feature count mismatch: expected {len(expected_features)}, got {len(features)}")
            features = [np.zeros(len(df)) for _ in range(len(expected_features))]

        # Combine features
        if features:
            feature_matrix = np.column_stack(features)
        else:
            # Fallback if no features available
            feature_matrix = np.zeros((len(df), len(expected_features)))

        # Replace infinities and NaN with 0
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        return feature_matrix, expected_features

    def scale_features(self, X: np.ndarray) -> np.ndarray:
        """
        Scale features using StandardScaler.

        Args:
            X: Feature matrix

        Returns:
            Scaled feature matrix
        """
        if not hasattr(self, 'scaler') or self.scaler is None:
            self.scaler = StandardScaler()

        return self.scaler.fit_transform(X)

    def select_features_rfe_shap(self, X: np.ndarray, y: np.ndarray, feature_names: List[str],
                                max_features: int = 30) -> Tuple[np.ndarray, List[str]]:
        """
        Advanced feature selection combining RFE with SHAP analysis.

        Args:
            X: Scaled feature matrix
            y: Target labels
            feature_names: Original feature names
            max_features: Maximum number of features to select

        Returns:
            Tuple of (selected_features, selected_feature_names)
        """
        try:
            from sklearn.feature_selection import RFE
            from sklearn.linear_model import LogisticRegression

            n_features = X.shape[1]
            if n_features <= max_features:
                return X, feature_names

            print("🔍 Starting RFE + SHAP feature selection...")

            # Step 1: Initial statistical filtering (keep top 2x target features)
            from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
            stat_selector = SelectKBest(score_func=f_classif, k=min(max_features * 2, n_features))
            X_stat = stat_selector.fit_transform(X, y)
            stat_mask = stat_selector.get_support()

            # Get statistically significant features
            stat_features = X[:, stat_mask]
            stat_feature_names = [name for name, selected in zip(feature_names, stat_mask) if selected]

            # Step 2: RFE with cross-validation on statistically significant features
            if len(stat_features[0]) > max_features:
                rfe_model = LogisticRegression(random_state=42, max_iter=1000)

                # Use fewer features for RFE if we have many
                rfe_n_features = min(max_features * 2, len(stat_features[0]))
                rfe_selector = RFE(
                    estimator=rfe_model,
                    n_features_to_select=min(max_features, rfe_n_features),
                    step=1,
                    verbose=0
                )

                X_rfe = rfe_selector.fit_transform(stat_features, y)
                rfe_mask = rfe_selector.get_support()

                # Map back to statistical feature space
                rfe_selected_features = stat_features[:, rfe_mask]
                rfe_selected_names = [name for name, selected in zip(stat_feature_names, rfe_mask) if selected]

                # Step 3: SHAP analysis on RFE-selected features
                if SHAP_AVAILABLE and len(rfe_selected_features[0]) > 5:
                    shap_features, shap_names = self._shap_feature_ranking(
                        rfe_selected_features, y, rfe_selected_names, max_features
                    )
                else:
                    shap_features, shap_names = rfe_selected_features, rfe_selected_names
            else:
                # If statistical filtering already gave us target number, use SHAP directly
                if SHAP_AVAILABLE and len(stat_features[0]) > 5:
                    shap_features, shap_names = self._shap_feature_ranking(
                        stat_features, y, stat_feature_names, max_features
                    )
                else:
                    shap_features, shap_names = stat_features, stat_feature_names

            print(f"✅ RFE + SHAP selection: {n_features} -> {shap_features.shape[1]} features")
            if len(shap_names) > 0:
                print(f"   Top features: {', '.join(shap_names[:5])}")

            return shap_features, shap_names

        except Exception as e:
            print(f"⚠️  RFE + SHAP selection failed: {e}, using statistical selection")
            return self.select_features(X, y, feature_names, max_features)

    def _shap_feature_ranking(self, X: np.ndarray, y: np.ndarray, feature_names: List[str],
                             max_features: int) -> Tuple[np.ndarray, List[str]]:
        """
        Rank features using SHAP values and select top features.

        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Feature names
            max_features: Maximum number of features to select

        Returns:
            Tuple of (selected_features, selected_feature_names)
        """
        try:
            # Train a simple model for SHAP analysis
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            model.fit(X, y)

            # Create SHAP explainer
            if 'RandomForest' in str(type(model)):
                explainer = shap.TreeExplainer(model)
            else:
                # Fallback for other models
                background_sample = X[:min(50, len(X))]
                explainer = shap.KernelExplainer(model.predict_proba, background_sample)

            # Calculate SHAP values for a sample of data
            sample_size = min(100, len(X))
            X_sample = X[:sample_size]

            shap_values = explainer.shap_values(X_sample)

            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                if len(shap_values) > 1:
                    shap_vals = shap_values[1]  # Positive class for binary classification
                else:
                    shap_vals = shap_values[0]
            else:
                shap_vals = shap_values

            # Calculate mean absolute SHAP values for each feature
            feature_importance = np.abs(shap_vals).mean(axis=0)

            # Rank features by importance
            feature_ranks = np.argsort(feature_importance)[::-1]  # Descending order

            # Select top features
            top_indices = feature_ranks[:max_features]
            selected_features = X[:, top_indices]
            selected_names = [feature_names[i] for i in top_indices]

            return selected_features, selected_names

        except Exception as e:
            print(f"⚠️  SHAP ranking failed: {e}, using all features")
            return X, feature_names

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

            # Feature scaling and selection
            X_scaled = self.scale_features(X)
            X_selected, selected_feature_names = self.select_features_rfe_shap(X_scaled, y, feature_names)

            # Store feature selection info for prediction
            self.selected_feature_indices = None
            if len(selected_feature_names) < len(feature_names):
                self.selected_feature_indices = [feature_names.index(name) for name in selected_feature_names]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=test_size, random_state=42, stratify=y
            )

            self.feature_columns = selected_feature_names

            # Train and evaluate all models
            results = {}
            best_score = 0
            best_model_name = None

            # Optimize models if AutoML is enabled
            if ML_USE_AUTOML and OPTUNA_AVAILABLE:
                print("🤖 Running AutoML optimization for all models...")
                for model_name in self.models.keys():
                    if model_name == 'xgboost' and not XGBOOST_AVAILABLE:
                        continue
                    optimization_result = self.optimize_model_hyperparameters(model_name, X_train, X_test, y_train, y_test)
                    if optimization_result:
                        print(f"✅ {model_name} optimized: F1={optimization_result['best_score']:.4f}")

            for model_name, model_config in self.models.items():
                # Skip XGBoost and LightGBM if not available
                if model_name == 'xgboost' and not XGBOOST_AVAILABLE:
                    continue
                if model_name == 'lightgbm' and not LIGHTGBM_AVAILABLE:
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

                    # Select best model based on preferred criteria
                    score = {
                        'f1': f1,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'balanced_accuracy': (accuracy + recall) / 2  # Simple balanced accuracy
                    }.get(ML_MODEL_SELECTION_CRITERIA, f1)

                    if score > best_score:
                        best_score = score
                        best_model_name = model_name

                    print(f"✅ {model_config['name']}: F1={f1:.3f}, CV-F1={cv_scores.mean():.3f}±{cv_scores.std():.3f}")

                except Exception as e:
                    print(f"❌ Failed to train {model_config['name']}: {e}")
                    results[model_name] = {'error': str(e)}

            # If preferred model is available and performs reasonably well, use it
            if ML_PREFERRED_MODEL in results and 'error' not in results[ML_PREFERRED_MODEL]:
                preferred_score = results[ML_PREFERRED_MODEL]['f1_score']
                if preferred_score >= best_score * 0.9:  # Within 90% of best score
                    best_model_name = ML_PREFERRED_MODEL
                    best_score = preferred_score
                    print(f"✅ Using preferred model: {ML_PREFERRED_MODEL} (F1={preferred_score:.3f})")
                else:
                    print(f"ℹ️  Preferred model {ML_PREFERRED_MODEL} underperforms (F1={preferred_score:.3f} vs {best_score:.3f}), using best model instead")

            if best_model_name:
                self.best_model = results[best_model_name]['model']
                self.model_performance = results
                self.is_trained = True

                # Try to create stacking ensemble if multiple models are available
                self.stacking_ensemble = None
                self.weighted_ensemble = None
                if len([m for m in results.keys() if 'error' not in results[m]]) >= 3:
                    self.create_advanced_ensembles(X_train, X_test, y_train, y_test)

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

    def train_single_model(self, df: pd.DataFrame, model_name: str, test_size: float = 0.2, prediction_horizon: int = 5) -> Dict[str, Any]:
        """
        Train a single ML model and select it as the best model.

        Args:
            df: DataFrame with technical indicators and price data
            model_name: Name of the model to train
            test_size: Fraction of data to use for testing
            prediction_horizon: Periods to look ahead for labeling

        Returns:
            Dictionary with training results and best model info
        """
        try:
            # Check if model is available
            if model_name not in self.models:
                available_models = list(self.models.keys())
                raise ValueError(f"Model '{model_name}' not available. Available models: {available_models}")

            # Prepare features and labels
            X, feature_names = self.prepare_features(df)
            y = self.create_labels(df, prediction_horizon)

            # Remove rows with NaN
            valid_indices = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[valid_indices]
            y = y[valid_indices]

            if len(X) < 100:
                raise ValueError(f"Insufficient data for training: {len(X)} samples")

            # Feature scaling and selection
            X_scaled = self.scale_features(X)
            X_selected, selected_feature_names = self.select_features_rfe_shap(X_scaled, y, feature_names)

            # Store feature selection info for prediction
            self.selected_feature_indices = None
            if len(selected_feature_names) < len(feature_names):
                self.selected_feature_indices = [feature_names.index(name) for name in selected_feature_names]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=test_size, random_state=42, stratify=y
            )

            self.feature_columns = selected_feature_names

            # Get the model configuration
            model_config = self.models[model_name]
            if model_config['model'] is None:
                raise ValueError(f"Model '{model_name}' is not available (library not installed)")

            # Train the single model
            model = model_config['model']
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

            results = {
                model_name: {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'cv_f1_mean': cv_scores.mean(),
                    'cv_f1_std': cv_scores.std(),
                    'model': model
                }
            }

            # Set as best model
            self.best_model = model
            self.model_performance = results
            self.is_trained = True

            # Save the model
            self.save_model()

            # Log training completion
            audit_logger.log_system_event(
                event_type="ML_SINGLE_MODEL_TRAINING_COMPLETED",
                message=f"Single ML model training completed for {self.symbol} {self.timeframe}",
                details={
                    'model': model_name,
                    'f1_score': f1,
                    'samples': len(X),
                    'features': len(feature_names)
                }
            )

            print(f"✅ {model_config['name']} trained: F1={f1:.3f}, CV-F1={cv_scores.mean():.3f}±{cv_scores.std():.3f}")

            return {
                'success': True,
                'best_model': model_name,
                'performance': results,
                'samples': len(X),
                'features': len(feature_names)
            }

        except Exception as e:
            error_msg = f"Single model training failed: {str(e)}"
            print(f"❌ {error_msg}")

            audit_logger.log_error(
                error_type="ML_SINGLE_MODEL_TRAINING_ERROR",
                message=error_msg,
                context={'symbol': self.symbol, 'timeframe': self.timeframe, 'model': model_name}
            )

            return {'success': False, 'error': str(e)}

    def create_advanced_ensembles(self, X_train, X_test, y_train, y_test):
        """
        Create advanced ensemble methods: stacking and weighted voting.

        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test labels
        """
        try:
            from sklearn.ensemble import VotingClassifier, StackingClassifier

            # Get successful base models
            base_models = []
            model_weights = []
            for model_name, result in self.model_performance.items():
                if 'error' not in result and 'model' in result and 'f1_score' in result:
                    base_models.append((model_name, result['model']))
                    # Weight by F1 score
                    model_weights.append(max(result['f1_score'], 0.1))  # Minimum weight of 0.1

            if len(base_models) < 3:
                return  # Not enough models for advanced ensembles

            # Normalize weights
            total_weight = sum(model_weights)
            model_weights = [w / total_weight for w in model_weights]

            # Create weighted voting ensemble
            self.weighted_ensemble = VotingClassifier(
                estimators=base_models,
                voting='soft',  # Use probability-based voting
                weights=model_weights
            )

            # Train weighted ensemble
            self.weighted_ensemble.fit(X_train, y_train)

            # Evaluate weighted ensemble
            weighted_pred = self.weighted_ensemble.predict(X_test)
            weighted_f1 = f1_score(y_test, weighted_pred, zero_division=0)

            # Add to performance results
            self.model_performance['weighted_ensemble'] = {
                'accuracy': accuracy_score(y_test, weighted_pred),
                'precision': precision_score(y_test, weighted_pred, zero_division=0),
                'recall': recall_score(y_test, weighted_pred, zero_division=0),
                'f1_score': weighted_f1,
                'model': self.weighted_ensemble,
                'weights': dict(zip([name for name, _ in base_models], model_weights))
            }

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

            # Compare ensembles and select the best
            best_individual_f1 = max([result.get('f1_score', 0) for result in self.model_performance.values()
                                    if isinstance(result, dict) and 'f1_score' in result and 'ensemble' not in result.get('model', '').__class__.__name__.lower()])

            ensemble_scores = {
                'weighted': weighted_f1,
                'stacking': stacking_f1
            }

            best_ensemble = max(ensemble_scores, key=ensemble_scores.get)
            best_ensemble_score = ensemble_scores[best_ensemble]

            if best_ensemble_score > best_individual_f1:
                if best_ensemble == 'weighted':
                    self.best_model = self.weighted_ensemble
                else:
                    self.best_model = self.stacking_ensemble
                print(f"✅ {best_ensemble.title()} ensemble selected (F1={best_ensemble_score:.3f} > {best_individual_f1:.3f})")
            else:
                print(f"ℹ️  Ensembles trained (Weighted F1={weighted_f1:.3f}, Stacking F1={stacking_f1:.3f}) but individual model kept")

        except Exception as e:
            print(f"⚠️  Advanced ensemble creation failed: {e}")
            self.stacking_ensemble = None
            self.weighted_ensemble = None

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
            X, feature_names = self.prepare_features(df)
            if len(X) == 0:
                return {
                    'signal': 'HOLD',
                    'confidence': 0.0,
                    'reason': 'No features available'
                }

            # Apply scaling and feature selection
            X_scaled = self.scaler.transform(X) if hasattr(self, 'scaler') and self.scaler else X
            if hasattr(self, 'selected_feature_indices') and self.selected_feature_indices is not None:
                X_scaled = X_scaled[:, self.selected_feature_indices]
                # Update feature names if feature selection was applied
                if hasattr(self, 'feature_columns') and self.feature_columns:
                    feature_names = [self.feature_columns[i] for i in self.selected_feature_indices]

            last_features = X_scaled[-1:].reshape(1, -1)

            # Check feature count compatibility before prediction
            expected_features = getattr(self.best_model, 'n_features_in_', None)
            if expected_features is None and hasattr(self, 'expected_features'):
                expected_features = self.expected_features

            if expected_features is not None and last_features.shape[1] != expected_features:
                error_msg = f"Feature count mismatch: model expects {expected_features} features, got {last_features.shape[1]} features"
                print(f"❌ {error_msg}")

                audit_logger.log_error(
                    error_type="FEATURE_COUNT_MISMATCH",
                    message=error_msg,
                    context={
                        'symbol': self.symbol,
                        'timeframe': self.timeframe,
                        'expected_features': expected_features,
                        'actual_features': last_features.shape[1],
                        'stored_feature_columns': len(self.feature_columns)
                    }
                )

                return {
                    'signal': 'HOLD',
                    'confidence': 0.0,
                    'reason': f'Feature mismatch: expected {expected_features}, got {last_features.shape[1]}'
                }

            # Get prediction and probability
            # Handle LightGBM feature name warnings by using feature_name_or_number
            if 'LGBM' in str(type(self.best_model)):
                # For LightGBM, create a DataFrame with proper column names to avoid warnings
                import pandas as pd
                last_features_df = pd.DataFrame(last_features, columns=feature_names[:last_features.shape[1]])
                prediction = self.best_model.predict(last_features_df)[0]
                if hasattr(self.best_model, 'predict_proba'):
                    prob = self.best_model.predict_proba(last_features_df)[0]
                    confidence = max(prob)  # Highest probability
                    predicted_class_prob = prob[1] if len(prob) > 1 else prob[0]
                else:
                    confidence = 0.5  # Default confidence for models without predict_proba
                    predicted_class_prob = 0.5
            else:
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

        # Get expected feature count from the model
        expected_features = getattr(self.best_model, 'n_features_in_', None)

        model_data = {
            'model': self.best_model,
            'feature_columns': self.feature_columns,
            'expected_features': expected_features,
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
            self.expected_features = model_data.get('expected_features', None)
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

    def optimize_gradient_boosting(self, X_train, X_test, y_train, y_test) -> Dict[str, Any]:
        """
        Optimize Gradient Boosting hyperparameters using Optuna.

        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test labels

        Returns:
            Dictionary with optimized parameters and performance
        """
        if not OPTUNA_AVAILABLE:
            print("⚠️  Optuna not available, using default parameters")
            return {}

        def objective(trial):
            """Optuna objective function for Gradient Boosting optimization."""
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': 42
            }

            model = GradientBoostingClassifier(**params)
            model.fit(X_train, y_train)

            # Use F1 score as optimization target
            y_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            return f1

        # Set up Optuna study
        sampler = {
            'tpe': TPESampler(),
            'random': RandomSampler(),
            'grid': GridSampler({})
        }.get(ML_AUTOML_SAMPLER, TPESampler())

        pruner = {
            'median': MedianPruner(),
            'hyperband': HyperbandPruner()
        }.get(ML_AUTOML_PRUNER, MedianPruner())

        study = optuna.create_study(
            study_name=ML_AUTOML_STUDY_NAME,
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            storage=ML_AUTOML_STORAGE
        )

        print(f"🎯 Starting Optuna optimization with {ML_AUTOML_N_TRIALS} trials...")

        # Run optimization
        study.optimize(objective, n_trials=ML_AUTOML_N_TRIALS, timeout=ML_AUTOML_TIMEOUT)

        # Get best parameters
        best_params = study.best_params
        best_score = study.best_value

        print(f"✅ Optimization completed! Best F1: {best_score:.4f}")
        print(f"   Best parameters: {best_params}")

        # Update model with best parameters
        self.models['gradient_boosting']['model'] = GradientBoostingClassifier(
            **best_params,
            random_state=42
        )

        return {
            'best_params': best_params,
            'best_score': best_score,
            'study': study
        }

    def optimize_model_hyperparameters(self, model_name: str, X_train, X_test, y_train, y_test) -> Dict[str, Any]:
        """
        Optimize hyperparameters for any ML model using Optuna.

        Args:
            model_name: Name of the model to optimize
            X_train, X_test: Training and test features
            y_train, y_test: Training and test labels

        Returns:
            Dictionary with optimized parameters and performance
        """
        if not OPTUNA_AVAILABLE or model_name not in self.models:
            return {}

        def objective(trial):
            """Generic Optuna objective function."""
            if model_name == 'gradient_boosting':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'random_state': 42
                }
                model = GradientBoostingClassifier(**params)

            elif model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                    'random_state': 42
                }
                model = RandomForestClassifier(**params)

            elif model_name == 'xgboost' and XGBOOST_AVAILABLE:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 5),
                    'random_state': 42
                }
                model = XGBClassifier(**params)

            elif model_name == 'lightgbm' and LIGHTGBM_AVAILABLE:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                    'random_state': 42,
                    'verbosity': -1
                }
                model = LGBMClassifier(**params)

            elif model_name == 'svm':
                # SVM parameters - define all parameters statically to avoid Optuna dynamic issues
                kernel = trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly'])
                C = trial.suggest_float('C', 0.1, 100, log=True)
                gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
                degree = trial.suggest_int('degree', 2, 5)

                params = {
                    'C': C,
                    'kernel': kernel,
                    'gamma': gamma,
                    'degree': degree,
                    'random_state': 42
                }

                # Remove gamma if kernel is linear
                if params['kernel'] == 'linear':
                    params.pop('gamma', None)
                model = SVC(**params, probability=True)

            elif model_name == 'logistic_regression':
                params = {
                    'C': trial.suggest_float('C', 0.01, 100, log=True),
                    'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet', 'none']),
                    'solver': trial.suggest_categorical('solver', ['liblinear', 'saga', 'lbfgs']),
                    'max_iter': 1000,
                    'random_state': 42
                }
                # Adjust solver based on penalty
                if params['penalty'] == 'l1' and params['solver'] not in ['liblinear', 'saga']:
                    params['solver'] = 'liblinear'
                elif params['penalty'] == 'elasticnet' and params['solver'] != 'saga':
                    params['solver'] = 'saga'
                elif params['penalty'] == 'none':
                    params['solver'] = 'lbfgs'
                model = LogisticRegression(**params)

            else:
                return 0.0  # Skip unsupported models

            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                return f1
            except Exception:
                return 0.0

        # Set up Optuna study
        sampler = {
            'tpe': TPESampler(),
            'random': RandomSampler(),
            'grid': GridSampler({})
        }.get(ML_AUTOML_SAMPLER, TPESampler())

        pruner = {
            'median': MedianPruner(),
            'hyperband': HyperbandPruner()
        }.get(ML_AUTOML_PRUNER, MedianPruner())

        study_name = f'tradpal_{model_name}_optimization'
        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            storage=ML_AUTOML_STORAGE
        )

        print(f"🎯 Optimizing {model_name} with {ML_AUTOML_N_TRIALS} trials...")

        # Run optimization
        study.optimize(objective, n_trials=ML_AUTOML_N_TRIALS, timeout=ML_AUTOML_TIMEOUT)

        # Get best parameters
        best_params = study.best_params
        best_score = study.best_value

        print(f"✅ {model_name} optimization completed! Best F1: {best_score:.4f}")

        # Update model with best parameters
        if model_name == 'gradient_boosting':
            self.models[model_name]['model'] = GradientBoostingClassifier(**best_params, random_state=42)
        elif model_name == 'random_forest':
            self.models[model_name]['model'] = RandomForestClassifier(**best_params, random_state=42)
        elif model_name == 'xgboost' and XGBOOST_AVAILABLE:
            self.models[model_name]['model'] = XGBClassifier(**best_params, random_state=42)
        elif model_name == 'lightgbm' and LIGHTGBM_AVAILABLE:
            self.models[model_name]['model'] = LGBMClassifier(**best_params, verbosity=-1)
        elif model_name == 'svm':
            # Handle SVM parameters
            svm_params = best_params.copy()
            if svm_params.get('kernel') == 'linear':
                svm_params.pop('gamma', None)
            self.models[model_name]['model'] = SVC(**svm_params, probability=True, random_state=42)
        elif model_name == 'logistic_regression':
            # Handle LogisticRegression parameters
            lr_params = best_params.copy()
            self.models[model_name]['model'] = LogisticRegression(**lr_params, max_iter=1000, random_state=42)

        return {
            'best_params': best_params,
            'best_score': best_score,
            'study': study
        }


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
    shap = None
    print("⚠️  SHAP not available. Install with: pip install shap")
    print("   SHAP integration will be disabled.")


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

# Placeholder classes for TensorFlow features (when not available)
class LSTMSignalPredictor:
    def __init__(self, *args, **kwargs):
        raise ImportError('TensorFlow is not available. Install with: pip install tensorflow')

class TransformerSignalPredictor:
    def __init__(self, *args, **kwargs):
        raise ImportError('TensorFlow is not available. Install with: pip install tensorflow')

def get_lstm_predictor(symbol: str = SYMBOL, timeframe: str = TIMEFRAME) -> Optional[Any]:
    """Get LSTM predictor instance (not available)."""
    return None

def get_transformer_predictor(symbol: str = SYMBOL, timeframe: str = TIMEFRAME) -> Optional[Any]:
    """Get Transformer predictor instance (not available)."""
    return None

def is_lstm_available() -> bool:
    """Check if LSTM features are available."""
    return False

def is_transformer_available() -> bool:
    """Check if Transformer features are available."""
    return False

def is_shap_available() -> bool:
    """Check if SHAP explanations are available."""
    return SHAP_AVAILABLE
