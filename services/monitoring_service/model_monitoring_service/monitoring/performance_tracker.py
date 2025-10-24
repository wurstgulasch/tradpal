"""
Performance Tracking Module for Model Monitoring Service
Tracks model performance metrics and detects performance degradation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import pickle
import os
import logging
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """
    Tracks model performance metrics and detects degradation.
    """

    def __init__(self, metrics_dir: str = "models", window_size: int = 1000):
        """
        Initialize performance tracker.

        Args:
            metrics_dir: Directory to store performance metrics
            window_size: Size of rolling window for performance tracking
        """
        self.metrics_dir = metrics_dir
        self.window_size = window_size
        self.performance_history: Dict[str, deque] = {}
        self.baseline_performance: Dict[str, Dict[str, float]] = {}
        self.alert_thresholds: Dict[str, Dict[str, float]] = {}

        os.makedirs(metrics_dir, exist_ok=True)
        self._load_existing_metrics()

    def register_model(self, model_id: str, baseline_metrics: Dict[str, float],
                      alert_thresholds: Optional[Dict[str, float]] = None):
        """
        Register a new model for performance tracking.

        Args:
            model_id: Unique model identifier
            baseline_metrics: Baseline performance metrics
            alert_thresholds: Thresholds for performance alerts
        """
        logger.info(f"Registering model {model_id} for performance tracking")

        self.baseline_performance[model_id] = baseline_metrics.copy()
        self.baseline_performance[model_id]['registered_at'] = datetime.now().isoformat()

        # Initialize performance history
        self.performance_history[model_id] = deque(maxlen=self.window_size)

        # Set default alert thresholds if not provided
        if alert_thresholds is None:
            alert_thresholds = {
                'mse_degradation': 0.2,  # 20% increase in MSE
                'accuracy_drop': 0.05,   # 5% drop in accuracy
                'f1_drop': 0.05,         # 5% drop in F1 score
                'precision_drop': 0.05,  # 5% drop in precision
                'recall_drop': 0.05      # 5% drop in recall
            }

        self.alert_thresholds[model_id] = alert_thresholds

        # Save metrics
        self._save_metrics(model_id)

    def track_prediction(self, model_id: str, prediction: float, actual: float,
                        metadata: Optional[Dict[str, Any]] = None):
        """
        Track a single prediction for performance monitoring.

        Args:
            model_id: Model identifier
            prediction: Model prediction
            actual: Actual value
            metadata: Additional metadata (confidence, features, etc.)
        """
        if model_id not in self.performance_history:
            logger.warning(f"Model {model_id} not registered for performance tracking")
            return

        # Calculate error metrics
        error = prediction - actual
        abs_error = abs(error)
        squared_error = error ** 2

        # Create performance record
        record = {
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction,
            'actual': actual,
            'error': error,
            'abs_error': abs_error,
            'squared_error': squared_error,
            'metadata': metadata or {}
        }

        # Add to history
        self.performance_history[model_id].append(record)

        logger.debug(f"Tracked prediction for {model_id}: error={error:.4f}")

    def get_current_metrics(self, model_id: str) -> Dict[str, float]:
        """
        Calculate current performance metrics.

        Args:
            model_id: Model identifier

        Returns:
            Current performance metrics
        """
        if model_id not in self.performance_history or len(self.performance_history[model_id]) < 10:
            return {}

        history = list(self.performance_history[model_id])

        # Extract values
        predictions = np.array([r['prediction'] for r in history])
        actuals = np.array([r['actual'] for r in history])
        errors = np.array([r['error'] for r in history])
        squared_errors = np.array([r['squared_error'] for r in history])

        # Calculate metrics
        metrics = {
            'mse': float(np.mean(squared_errors)),
            'rmse': float(np.sqrt(np.mean(squared_errors))),
            'mae': float(np.mean(np.abs(errors))),
            'mean_error': float(np.mean(errors)),
            'std_error': float(np.std(errors)),
            'sample_size': len(history),
            'last_updated': history[-1]['timestamp']
        }

        # Calculate directional accuracy for trading models
        if len(predictions) > 1:
            pred_direction = np.sign(np.diff(predictions))
            actual_direction = np.sign(np.diff(actuals))
            directional_accuracy = np.mean(pred_direction == actual_direction)
            metrics['directional_accuracy'] = float(directional_accuracy)

        return metrics

    def get_performance_trend(self, model_id: str, periods: int = 5) -> List[Dict[str, float]]:
        """
        Get performance trend over recent periods.

        Args:
            model_id: Model identifier
            periods: Number of periods to analyze

        Returns:
            List of performance metrics per period
        """
        if model_id not in self.performance_history:
            return []

        history = list(self.performance_history[model_id])
        if len(history) < periods * 10:  # Need at least 10 samples per period
            return []

        period_size = len(history) // periods
        trends = []

        for i in range(periods):
            start_idx = i * period_size
            end_idx = (i + 1) * period_size if i < periods - 1 else len(history)

            period_data = history[start_idx:end_idx]
            predictions = np.array([r['prediction'] for r in period_data])
            actuals = np.array([r['actual'] for r in period_data])
            errors = np.array([r['error'] for r in period_data])
            squared_errors = np.array([r['squared_error'] for r in period_data])

            period_metrics = {
                'period': i + 1,
                'mse': float(np.mean(squared_errors)),
                'rmse': float(np.sqrt(np.mean(squared_errors))),
                'mae': float(np.mean(np.abs(errors))),
                'sample_size': len(period_data),
                'start_time': period_data[0]['timestamp'],
                'end_time': period_data[-1]['timestamp']
            }

            trends.append(period_metrics)

        return trends

    def detect_performance_degradation(self, model_id: str) -> List[str]:
        """
        Detect performance degradation based on configured thresholds.

        Args:
            model_id: Model identifier

        Returns:
            List of degradation alerts
        """
        if model_id not in self.baseline_performance or model_id not in self.performance_history:
            return []

        current_metrics = self.get_current_metrics(model_id)
        if not current_metrics:
            return []

        baseline = self.baseline_performance[model_id]
        thresholds = self.alert_thresholds[model_id]

        alerts = []

        # Check MSE degradation
        if 'mse' in current_metrics and 'mse' in baseline:
            mse_ratio = current_metrics['mse'] / baseline['mse']
            if mse_ratio > (1 + thresholds.get('mse_degradation', 0.2)):
                alerts.append(f"MSE increased by {((mse_ratio - 1) * 100):.1f}% "
                            f"(threshold: {thresholds['mse_degradation'] * 100:.1f}%)")

        # Check accuracy drop (if available)
        for metric in ['accuracy', 'f1', 'precision', 'recall']:
            if metric in current_metrics and metric in baseline:
                current_val = current_metrics[metric]
                baseline_val = baseline[metric]
                threshold = thresholds.get(f'{metric}_drop', 0.05)

                if current_val < baseline_val * (1 - threshold):
                    drop_pct = ((baseline_val - current_val) / baseline_val) * 100
                    alerts.append(f"{metric.title()} dropped by {drop_pct:.1f}% "
                                f"(threshold: {threshold * 100:.1f}%)")

        # Check directional accuracy degradation for trading models
        if 'directional_accuracy' in current_metrics and 'directional_accuracy' in baseline:
            current_da = current_metrics['directional_accuracy']
            baseline_da = baseline['directional_accuracy']
            threshold = thresholds.get('directional_accuracy_drop', 0.05)

            if current_da < baseline_da * (1 - threshold):
                drop_pct = ((baseline_da - current_da) / baseline_da) * 100
                alerts.append(f"Directional accuracy dropped by {drop_pct:.1f}% "
                            f"(threshold: {threshold * 100:.1f}%)")

        return alerts

    def get_baseline_metrics(self, model_id: str) -> Dict[str, float]:
        """
        Get baseline performance metrics for a model.

        Args:
            model_id: Model identifier

        Returns:
            Baseline metrics
        """
        return self.baseline_performance.get(model_id, {}).copy()

    def get_alert_thresholds(self, model_id: str) -> Dict[str, float]:
        """
        Get alert thresholds for a model.

        Args:
            model_id: Model identifier

        Returns:
            Alert thresholds
        """
        return self.alert_thresholds.get(model_id, {}).copy()

    def update_baseline(self, model_id: str, new_baseline: Dict[str, float]):
        """
        Update baseline performance metrics.

        Args:
            model_id: Model identifier
            new_baseline: New baseline metrics
        """
        if model_id in self.baseline_performance:
            self.baseline_performance[model_id].update(new_baseline)
            self.baseline_performance[model_id]['updated_at'] = datetime.now().isoformat()
            self._save_metrics(model_id)
            logger.info(f"Updated baseline metrics for {model_id}")

    def unregister_model(self, model_id: str):
        """
        Unregister a model from performance tracking.

        Args:
            model_id: Model identifier
        """
        if model_id in self.performance_history:
            del self.performance_history[model_id]

        if model_id in self.baseline_performance:
            del self.baseline_performance[model_id]

        if model_id in self.alert_thresholds:
            del self.alert_thresholds[model_id]

        # Remove metrics file
        metrics_file = os.path.join(self.metrics_dir, f"{model_id}_metrics.pkl")
        if os.path.exists(metrics_file):
            os.remove(metrics_file)

        logger.info(f"Unregistered model {model_id} from performance tracking")

    def _save_metrics(self, model_id: str):
        """Save performance metrics to disk."""
        metrics_file = os.path.join(self.metrics_dir, f"{model_id}_metrics.pkl")
        try:
            data = {
                'baseline': self.baseline_performance[model_id],
                'thresholds': self.alert_thresholds[model_id],
                'history_length': len(self.performance_history[model_id])
            }
            with open(metrics_file, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved metrics for model {model_id}")
        except Exception as e:
            logger.error(f"Failed to save metrics for {model_id}: {e}")

    def _load_existing_metrics(self):
        """Load existing metrics files on startup."""
        if not os.path.exists(self.metrics_dir):
            return

        for filename in os.listdir(self.metrics_dir):
            if filename.endswith('_metrics.pkl'):
                model_id = filename.replace('_metrics.pkl', '')
                metrics_file = os.path.join(self.metrics_dir, filename)

                try:
                    with open(metrics_file, 'rb') as f:
                        data = pickle.load(f)

                    self.baseline_performance[model_id] = data['baseline']
                    self.alert_thresholds[model_id] = data['thresholds']

                    # Initialize performance history
                    self.performance_history[model_id] = deque(maxlen=self.window_size)

                    logger.info(f"Loaded metrics for model {model_id}")

                except Exception as e:
                    logger.error(f"Failed to load metrics for {model_id}: {e}")

    def get_performance_summary(self, model_id: str) -> Dict[str, Any]:
        """
        Get comprehensive performance summary for a model.

        Args:
            model_id: Model identifier

        Returns:
            Performance summary
        """
        summary = {
            'model_id': model_id,
            'current_metrics': self.get_current_metrics(model_id),
            'baseline_metrics': self.get_baseline_metrics(model_id),
            'alert_thresholds': self.get_alert_thresholds(model_id),
            'performance_trend': self.get_performance_trend(model_id),
            'degradation_alerts': self.detect_performance_degradation(model_id),
            'sample_count': len(self.performance_history.get(model_id, []))
        }

        return summary