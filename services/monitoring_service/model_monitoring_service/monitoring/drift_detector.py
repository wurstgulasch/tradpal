"""
Drift Detection Module for Model Monitoring Service
Implements Population Stability Index (PSI) and other drift detection algorithms
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
from typing import Dict, List, Any, Optional, Tuple
import pickle
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Drift detection using Population Stability Index (PSI) and other statistical methods.
    """

    def __init__(self, baseline_dir: str = "models", window_size: int = 1000):
        """
        Initialize drift detector.

        Args:
            baseline_dir: Directory to store baseline statistics
            window_size: Size of rolling window for current statistics
        """
        self.baseline_dir = baseline_dir
        self.window_size = window_size
        self.baseline_stats: Dict[str, Dict[str, Any]] = {}
        self.current_stats: Dict[str, List[np.ndarray]] = {}
        self.thresholds: Dict[str, float] = {}

        os.makedirs(baseline_dir, exist_ok=True)
        self._load_existing_baselines()

    def register_model(self, model_id: str, baseline_features: Dict[str, List[float]],
                      threshold: float = 0.1):
        """
        Register a new model for drift detection.

        Args:
            model_id: Unique model identifier
            baseline_features: Baseline feature distributions
            threshold: Drift threshold for alerts
        """
        logger.info(f"Registering model {model_id} for drift detection")

        # Convert baseline features to numpy arrays
        baseline_data = {}
        for feature_name, values in baseline_features.items():
            baseline_data[feature_name] = np.array(values)

        # Calculate baseline statistics
        self.baseline_stats[model_id] = self._calculate_baseline_stats(baseline_data)
        self.baseline_stats[model_id]['created_at'] = datetime.now().isoformat()
        self.baseline_stats[model_id]['feature_names'] = list(baseline_features.keys())

        # Initialize current stats
        self.current_stats[model_id] = []
        self.thresholds[model_id] = threshold

        # Save baseline
        self._save_baseline(model_id)

    def calculate_drift(self, features: Dict[str, float], model_id: str) -> float:
        """
        Calculate drift score for current features.

        Args:
            features: Current feature values
            model_id: Model identifier

        Returns:
            Drift score (PSI value)
        """
        if model_id not in self.baseline_stats:
            logger.warning(f"Model {model_id} not registered for drift detection")
            return 0.0

        # Convert features to numpy array in correct order
        feature_names = self.baseline_stats[model_id]['feature_names']
        try:
            current_values = np.array([features[name] for name in feature_names])
        except KeyError as e:
            logger.error(f"Missing feature in drift calculation: {e}")
            return 0.0

        # Calculate PSI
        psi_score = self._calculate_psi(current_values, self.baseline_stats[model_id])

        # Update current statistics
        self._update_current_stats(current_values, model_id)

        logger.debug(f"Drift score for {model_id}: {psi_score:.4f}")
        return psi_score

    def _calculate_psi(self, current_data: np.ndarray, baseline_stats: Dict[str, Any]) -> float:
        """
        Calculate Population Stability Index (PSI).

        Args:
            current_data: Current feature values (1D array)
            baseline_stats: Baseline statistics

        Returns:
            PSI score
        """
        try:
            # Create histograms with same bins
            bins = np.linspace(
                min(np.min(current_data), baseline_stats['min']),
                max(np.max(current_data), baseline_stats['max']),
                11  # 10 bins
            )

            current_hist, _ = np.histogram(current_data, bins=bins, density=True)
            baseline_hist = baseline_stats['histogram']

            # Avoid division by zero and log(0)
            current_hist = np.where(current_hist == 0, 1e-10, current_hist)
            baseline_hist = np.where(np.array(baseline_hist) == 0, 1e-10, baseline_hist)

            # Calculate PSI
            psi = np.sum((current_hist - baseline_hist) * np.log(current_hist / baseline_hist))

            return float(psi)

        except Exception as e:
            logger.error(f"PSI calculation failed: {e}")
            return 0.0

    def _calculate_baseline_stats(self, baseline_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Calculate baseline statistics for drift detection.

        Args:
            baseline_data: Dictionary of baseline feature arrays

        Returns:
            Baseline statistics
        """
        # Combine all features into single distribution for PSI
        all_values = np.concatenate(list(baseline_data.values()))

        stats = {
            'mean': float(np.mean(all_values)),
            'std': float(np.std(all_values)),
            'min': float(np.min(all_values)),
            'max': float(np.max(all_values)),
            'median': float(np.median(all_values)),
            'histogram': np.histogram(all_values, bins=10, density=True)[0].tolist(),
            'sample_size': len(all_values)
        }

        # Kolmogorov-Smirnov test preparation
        stats['ks_baseline'] = all_values.tolist()

        return stats

    def _update_current_stats(self, data: np.ndarray, model_id: str):
        """
        Update rolling statistics for current window.

        Args:
            data: Current feature values
            model_id: Model identifier
        """
        if model_id not in self.current_stats:
            self.current_stats[model_id] = []

        self.current_stats[model_id].append(data)

        # Keep only recent samples
        if len(self.current_stats[model_id]) > self.window_size:
            self.current_stats[model_id] = self.current_stats[model_id][-self.window_size:]

    def get_drift_score(self, model_id: str) -> float:
        """
        Get current drift score for a model.

        Args:
            model_id: Model identifier

        Returns:
            Current drift score
        """
        if model_id not in self.baseline_stats:
            return 0.0

        if model_id not in self.current_stats or len(self.current_stats[model_id]) < 10:
            return 0.0  # Not enough data for reliable drift calculation

        # Calculate drift from recent samples
        recent_data = np.concatenate(self.current_stats[model_id][-10:])
        return self._calculate_psi(recent_data, self.baseline_stats[model_id])

    def get_threshold(self, model_id: str) -> float:
        """
        Get drift threshold for a model.

        Args:
            model_id: Model identifier

        Returns:
            Drift threshold
        """
        return self.thresholds.get(model_id, 0.1)

    def get_last_update(self, model_id: str) -> str:
        """
        Get timestamp of last baseline update.

        Args:
            model_id: Model identifier

        Returns:
            ISO timestamp string
        """
        if model_id in self.baseline_stats:
            return self.baseline_stats[model_id].get('created_at', 'unknown')
        return 'never'

    def unregister_model(self, model_id: str):
        """
        Unregister a model from drift detection.

        Args:
            model_id: Model identifier
        """
        if model_id in self.baseline_stats:
            del self.baseline_stats[model_id]

        if model_id in self.current_stats:
            del self.current_stats[model_id]

        if model_id in self.thresholds:
            del self.thresholds[model_id]

        # Remove baseline file
        baseline_file = os.path.join(self.baseline_dir, f"{model_id}_baseline.pkl")
        if os.path.exists(baseline_file):
            os.remove(baseline_file)

        logger.info(f"Unregistered model {model_id} from drift detection")

    def _save_baseline(self, model_id: str):
        """Save baseline statistics to disk."""
        baseline_file = os.path.join(self.baseline_dir, f"{model_id}_baseline.pkl")
        try:
            with open(baseline_file, 'wb') as f:
                pickle.dump(self.baseline_stats[model_id], f)
            logger.info(f"Saved baseline for model {model_id}")
        except Exception as e:
            logger.error(f"Failed to save baseline for {model_id}: {e}")

    def _load_existing_baselines(self):
        """Load existing baseline files on startup."""
        if not os.path.exists(self.baseline_dir):
            return

        for filename in os.listdir(self.baseline_dir):
            if filename.endswith('_baseline.pkl'):
                model_id = filename.replace('_baseline.pkl', '')
                baseline_file = os.path.join(self.baseline_dir, filename)

                try:
                    with open(baseline_file, 'rb') as f:
                        self.baseline_stats[model_id] = pickle.load(f)

                    # Initialize current stats and thresholds
                    self.current_stats[model_id] = []
                    self.thresholds[model_id] = 0.1  # Default threshold

                    logger.info(f"Loaded baseline for model {model_id}")

                except Exception as e:
                    logger.error(f"Failed to load baseline for {model_id}: {e}")

    def get_drift_trend(self, model_id: str, periods: int = 5) -> List[float]:
        """
        Get drift score trend over recent periods.

        Args:
            model_id: Model identifier
            periods: Number of recent periods to analyze

        Returns:
            List of drift scores over time
        """
        if model_id not in self.current_stats:
            return []

        scores = []
        window_size = max(10, len(self.current_stats[model_id]) // periods)

        for i in range(min(periods, len(self.current_stats[model_id]) // window_size)):
            start_idx = i * window_size
            end_idx = (i + 1) * window_size
            period_data = np.concatenate(self.current_stats[model_id][start_idx:end_idx])
            score = self._calculate_psi(period_data, self.baseline_stats[model_id])
            scores.append(score)

        return scores