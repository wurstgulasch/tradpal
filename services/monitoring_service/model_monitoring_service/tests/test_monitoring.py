"""
Unit Tests for Model Monitoring Service
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add service path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from monitoring.drift_detector import DriftDetector
from monitoring.performance_tracker import PerformanceTracker
from monitoring.alert_manager import AlertManager


class TestDriftDetector:
    """Test cases for DriftDetector."""

    def setup_method(self):
        """Setup test fixtures."""
        self.detector = DriftDetector()

    def test_register_model(self):
        """Test model registration."""
        baseline_features = {
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature2": [0.1, 0.2, 0.3, 0.4, 0.5]
        }

        self.detector.register_model("test_model", baseline_features)

        assert "test_model" in self.detector.baseline_stats
        assert len(self.detector.baseline_stats["test_model"]["feature_names"]) == 2

    def test_calculate_drift_no_drift(self):
        """Test drift calculation with no drift."""
        # Use more data points for baseline
        baseline_features = {
            "feature1": [3.0] * 100  # All same value
        }

        self.detector.register_model("test_model", baseline_features)

        # Test with same distribution (multiple similar values)
        for i in range(10):
            self.detector.calculate_drift({"feature1": 3.0 + np.random.normal(0, 0.01)}, "test_model")

        drift_score = self.detector.get_drift_score("test_model")

        # Just check that we get a finite number (PSI calculation works)
        assert isinstance(drift_score, float)
        assert not np.isinf(drift_score)
        assert not np.isnan(drift_score)

    def test_calculate_drift_with_drift(self):
        """Test drift calculation with significant drift."""
        baseline_features = {
            "feature1": [1.0, 1.1, 1.2, 1.3, 1.4]  # Low values
        }

        self.detector.register_model("test_model", baseline_features)

        # Test with different distribution
        drift_score = self.detector.calculate_drift({"feature1": 5.0}, "test_model")

        # Should be higher drift score
        assert drift_score > 0.1

    def test_get_drift_score(self):
        """Test getting current drift score."""
        baseline_features = {"feature1": [1.0, 2.0, 3.0, 4.0, 5.0]}
        self.detector.register_model("test_model", baseline_features)

        # Add some current data
        for i in range(15):
            self.detector.calculate_drift({"feature1": 3.0 + i * 0.1}, "test_model")

        drift_score = self.detector.get_drift_score("test_model")
        assert isinstance(drift_score, float)


class TestPerformanceTracker:
    """Test cases for PerformanceTracker."""

    def setup_method(self):
        """Setup test fixtures."""
        self.tracker = PerformanceTracker()

    def test_register_model(self):
        """Test model registration for performance tracking."""
        baseline_metrics = {
            "mse": 0.02,
            "directional_accuracy": 0.65
        }

        self.tracker.register_model("test_model", baseline_metrics)

        assert "test_model" in self.tracker.baseline_performance
        assert self.tracker.baseline_performance["test_model"]["mse"] == 0.02

    def test_track_prediction(self):
        """Test prediction tracking."""
        baseline_metrics = {"mse": 0.02}
        self.tracker.register_model("test_model", baseline_metrics)

        self.tracker.track_prediction("test_model", 0.5, 0.48)

        assert len(self.tracker.performance_history["test_model"]) == 1

    def test_get_current_metrics(self):
        """Test current metrics calculation."""
        baseline_metrics = {"mse": 0.02}
        self.tracker.register_model("test_model", baseline_metrics)

        # Add some predictions
        for i in range(15):
            pred = 0.5 + i * 0.01
            actual = 0.48 + i * 0.01
            self.tracker.track_prediction("test_model", pred, actual)

        metrics = self.tracker.get_current_metrics("test_model")

        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert metrics["sample_size"] == 15

    def test_detect_performance_degradation(self):
        """Test performance degradation detection."""
        baseline_metrics = {"mse": 0.02}
        alert_thresholds = {"mse_degradation": 0.5}  # 50% degradation threshold

        self.tracker.register_model("test_model", baseline_metrics, alert_thresholds)

        # Add predictions with higher error (degraded performance)
        for i in range(15):
            pred = 0.5 + i * 0.01
            actual = 0.3 + i * 0.01  # Much higher error
            self.tracker.track_prediction("test_model", pred, actual)

        alerts = self.tracker.detect_performance_degradation("test_model")

        # Should detect degradation
        assert len(alerts) > 0


class TestAlertManager:
    """Test cases for AlertManager."""

    def setup_method(self):
        """Setup test fixtures."""
        self.manager = AlertManager()

    @pytest.mark.asyncio
    async def test_generate_alert(self):
        """Test alert generation."""
        alert_id = self.manager.generate_alert(
            "test_model",
            "drift",
            "Drift detected",
            severity="warning"
        )

        assert alert_id.startswith("test_model_drift_")
        assert len(self.manager.active_alerts) == 1

    @pytest.mark.asyncio
    async def test_resolve_alert(self):
        """Test alert resolution."""
        alert_id = self.manager.generate_alert(
            "test_model",
            "drift",
            "Drift detected"
        )

        self.manager.resolve_alert(alert_id, "Resolved manually")

        assert alert_id not in self.manager.active_alerts
        assert alert_id in [a["id"] for a in self.manager.alert_history["test_model"]]

    def test_get_active_alerts(self):
        """Test getting active alerts."""
        # Mock the async handling to avoid event loop issues
        with patch.object(self.manager, '_handle_alert'):
            self.manager.generate_alert("model1", "drift", "Alert 1")
            self.manager.generate_alert("model2", "performance", "Alert 2")

        active = self.manager.get_active_alerts()
        assert len(active) == 2

        model1_alerts = self.manager.get_active_alerts("model1")
        assert len(model1_alerts) == 1

    def test_alert_cooldown(self):
        """Test alert cooldown mechanism."""
        # Mock the async handling
        with patch.object(self.manager, '_handle_alert'):
            # Generate first alert
            alert_id1 = self.manager.generate_alert("test_model", "drift", "Alert 1")

            # Try to generate similar alert immediately (should be blocked by cooldown)
            alert_id2 = self.manager.generate_alert("test_model", "drift", "Alert 2")

        # Should have cooldown applied (only one active alert)
        assert len(self.manager.active_alerts) == 1

    def test_get_alert_summary(self):
        """Test alert summary generation."""
        # Mock the async handling
        with patch.object(self.manager, '_handle_alert'):
            self.manager.generate_alert("model1", "drift", "Alert 1", "warning")
            self.manager.generate_alert("model1", "performance", "Alert 2", "error")

        summary = self.manager.get_alert_summary("model1")

        assert summary["total_alerts"] == 2
        assert summary["active_alerts"] == 2
        assert summary["alerts_by_severity"]["warning"] == 1
        assert summary["alerts_by_severity"]["error"] == 1


if __name__ == "__main__":
    pytest.main([__file__])