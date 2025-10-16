#!/usr/bin/env python3
"""
Test configuration validation for new features.
"""

import pytest
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'services'))

from config.settings import (
    ML_ENABLED, ML_MODEL_DIR, ML_CONFIDENCE_THRESHOLD,
    ML_MIN_TRAINING_SAMPLES, ML_TEST_SIZE, ML_CV_FOLDS
)


class TestConfiguration:
    """Test configuration settings for new features."""

    def test_ml_config_types(self):
        """Test that ML configuration values have correct types."""
        assert isinstance(ML_ENABLED, bool)
        assert isinstance(ML_MODEL_DIR, str)
        assert isinstance(ML_CONFIDENCE_THRESHOLD, (int, float))
        assert isinstance(ML_MIN_TRAINING_SAMPLES, int)
        assert isinstance(ML_TEST_SIZE, (int, float))
        assert isinstance(ML_CV_FOLDS, int)

    def test_ml_config_ranges(self):
        """Test that ML configuration values are in valid ranges."""
        assert 0.0 <= ML_CONFIDENCE_THRESHOLD <= 1.0, "Confidence threshold should be between 0 and 1"
        assert ML_MIN_TRAINING_SAMPLES > 0, "Min training samples should be positive"
        assert 0.0 < ML_TEST_SIZE < 1.0, "Test size should be between 0 and 1"
        assert ML_CV_FOLDS > 1, "CV folds should be greater than 1"

    def test_ml_model_dir_exists(self):
        """Test that ML model directory can be created."""
        model_dir = Path(ML_MODEL_DIR)
        # Should be able to create the directory
        model_dir.mkdir(parents=True, exist_ok=True)
        assert model_dir.exists()
        assert model_dir.is_dir()

    def test_import_safety(self):
        """Test that all new imports work safely."""
        try:
            from services.ml_predictor import is_ml_available, get_ml_predictor
            from services.audit_logger import audit_logger
            # These should not raise exceptions
            assert callable(is_ml_available)
            assert callable(get_ml_predictor)
            assert hasattr(audit_logger, 'log_system_event')
            assert hasattr(audit_logger, 'log_signal')
            assert hasattr(audit_logger, 'log_trade')
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")

    def test_ml_availability_function(self):
        """Test that ML availability function works."""
        from services.ml_predictor import is_ml_available
        result = is_ml_available()
        assert isinstance(result, bool)

    def test_audit_logger_available(self):
        """Test that audit logger is available."""
        from services.audit_logger import audit_logger
        assert hasattr(audit_logger, 'log_system_event')
        assert hasattr(audit_logger, 'log_signal')
        assert hasattr(audit_logger, 'log_trade')
        assert hasattr(audit_logger, 'log_risk_assessment')