"""
Tests for MLOps Service
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch

from service import (
    MLOpsService,
    MLOpsConfig,
    MLflowConfig,
    ExperimentResult,
    DriftAlert
)


class TestMLOpsService:
    """Unit tests for MLOps service."""

    @pytest.fixture
    def mlops_service(self):
        """Create MLOps service for testing."""
        config = MLOpsConfig(
            mlflow=MLflowConfig(
                tracking_uri="http://localhost:5000",
                experiment_name="test_experiment"
            )
        )
        service = MLOpsService(config)
        return service

    @pytest.mark.asyncio
    async def test_initialization(self, mlops_service):
        """Test service initialization."""
        assert not mlops_service.is_running
        assert mlops_service.config is not None
        assert mlops_service.drift_detectors == {}

    @pytest.mark.asyncio
    async def test_service_start_stop(self, mlops_service):
        """Test service start and stop."""
        await mlops_service.start()
        assert mlops_service.is_running

        await mlops_service.stop()
        assert not mlops_service.is_running

    @pytest.mark.asyncio
    async def test_health_check(self, mlops_service):
        """Test health check functionality."""
        await mlops_service.start()

        health = await mlops_service.health_check()

        assert health['service'] == 'mlops_service'
        assert health['status'] == 'healthy'
        assert 'timestamp' in health
        assert 'components' in health
        assert 'models_deployed' in health

        await mlops_service.stop()

        # Test stopped state
        health = await mlops_service.health_check()
        assert health['status'] == 'stopped'

    @pytest.mark.asyncio
    async def test_experiment_logging_mock(self, mlops_service):
        """Test experiment logging with mocked MLflow."""
        await mlops_service.start()

        with patch('mlflow.start_run') as mock_run, \
             patch('mlflow.log_param'), \
             patch('mlflow.log_metric'), \
             patch('mlflow.sklearn.log_model'), \
             patch('mlflow.active_run') as mock_active_run:

            # Mock the run context
            mock_run.return_value.__enter__.return_value = Mock()
            mock_run.return_value.__enter__.return_value.info = Mock()
            mock_run.return_value.__enter__.return_value.info.experiment_id = "test_exp"
            mock_run.return_value.__enter__.return_value.info.run_id = "test_run"

            # Mock active_run
            mock_active_run.return_value = Mock()
            mock_active_run.return_value.info = Mock()
            mock_active_run.return_value.info.experiment_id = "test_exp"
            mock_active_run.return_value.info.run_id = "test_run"

            result = await mlops_service.log_experiment(
                model_name="test_model",
                metrics={"accuracy": 0.85},
                parameters={"param1": "value1"},
                model=Mock(),
                framework="sklearn"
            )

            assert isinstance(result, ExperimentResult)
            assert result.model_name == "test_model"
            assert result.run_id == "test_run"

        await mlops_service.stop()

    @pytest.mark.asyncio
    async def test_drift_detector_creation(self, mlops_service):
        """Test drift detector creation."""
        await mlops_service.start()

        # Create test data
        reference_data = np.random.randn(100, 5)

        with patch('builtins.open', create=True) as mock_open, \
             patch('json.dump') as mock_json_dump:

            detector_id = await mlops_service.create_drift_detector(
                model_name="test_model",
                reference_data=reference_data
            )

            assert detector_id.startswith("test_model_drift_detector_")
            assert detector_id in mlops_service.drift_detectors

            # Verify detector config
            config = mlops_service.drift_detectors[detector_id]
            assert config["model_name"] == "test_model"
            assert config["detector_type"] == "statistical"
            assert "reference_mean" in config
            assert "reference_std" in config
            assert config["threshold"] == 3.0

        await mlops_service.stop()

    @pytest.mark.asyncio
    async def test_get_experiment_history_mock(self, mlops_service):
        """Test getting experiment history with mocked MLflow."""
        await mlops_service.start()

        with patch.object(mlops_service, 'mlflow_client') as mock_client:
            mock_client.search_runs.return_value = []

            history = await mlops_service.get_experiment_history(limit=5)
            assert isinstance(history, list)

        await mlops_service.stop()


class TestMLOpsServiceIntegration:
    """Integration tests requiring external dependencies."""

    @pytest.mark.asyncio
    async def test_service_lifecycle(self):
        """Test complete service lifecycle."""
        config = MLOpsConfig()
        service = MLOpsService(config)

        # Start service
        await service.start()
        assert service.is_running

        # Check health
        health = await service.health_check()
        assert health['status'] == 'healthy'

        # Stop service
        await service.stop()
        assert not service.is_running

        # Check stopped health
        health = await service.health_check()
        assert health['status'] == 'stopped'


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])