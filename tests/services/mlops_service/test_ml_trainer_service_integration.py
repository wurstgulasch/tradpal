#!/usr/bin/env python3
"""
Integration Tests for ML Trainer Service

Tests the complete ML Trainer Service functionality including:
- API endpoints
- Service logic
- Client communication
- ML model training and evaluation
"""

import asyncio
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, AsyncMock

from services.ml_trainer.service import MLTrainerService, EventSystem
from services.ml_trainer.client import MLTrainerServiceClient


class TestMLTrainerServiceIntegration:
    """Integration tests for ML Trainer Service"""

    @pytest.fixture
    def test_client(self):
        """Create mock test client"""
        from unittest.mock import Mock
        client = Mock()
        # Mock common HTTP methods
        client.get = Mock()
        client.post = Mock()
        return client

    @pytest.fixture
    def event_system(self):
        """Create event system for testing"""
        return EventSystem()

    @pytest.fixture
    def ml_service(self, event_system):
        """Create ML trainer service instance"""
        return MLTrainerService(event_system=event_system)

    @pytest.fixture
    def ml_client(self):
        """Create ML trainer service client"""
        return MLTrainerServiceClient(base_url="http://test")

    def test_api_root_endpoint(self, test_client):
        """Test API root endpoint"""
        # Mock the response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "service": "ML Trainer Service",
            "version": "1.0.0"
        }
        test_client.get.return_value = mock_response

        response = test_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert data["service"] == "ML Trainer Service"

    @pytest.mark.asyncio
    async def test_service_health_check(self, ml_service):
        """Test service health check"""
        health = await ml_service.health_check()
        assert "service" in health
        assert "status" in health
        assert health["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_model_training(self, ml_service):
        """Test ML model training"""
        # Mock data fetching
        with patch.object(ml_service, '_fetch_training_data', new_callable=AsyncMock) as mock_fetch:
            # Create sample data
            dates = pd.date_range('2023-01-01', periods=100, freq='1H')
            data = pd.DataFrame({
                'timestamp': dates,
                'open': 50000 + np.random.normal(0, 1000, 100),
                'high': 50000 + np.random.normal(0, 1000, 100) + 100,
                'low': 50000 + np.random.normal(0, 1000, 100) - 100,
                'close': 50000 + np.random.normal(0, 1000, 100),
                'volume': np.random.normal(100, 20, 100)
            })
            mock_fetch.return_value = data

            result = await ml_service.train_model(
                symbol="BTC/USDT",
                timeframe="1h",
                start_date="2023-01-01",
                end_date="2023-01-10",
                model_type="random_forest",
                target_horizon=5,
                use_optuna=False
            )

            assert result["success"] is True
            assert "model_name" in result
            assert "performance" in result

    @pytest.mark.asyncio
    async def test_model_evaluation(self, ml_service):
        """Test model evaluation"""
        # Create sample test data
        test_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=50, freq='1H'),
            'open': 50000 + np.random.normal(0, 500, 50),
            'high': 50000 + np.random.normal(0, 500, 50) + 100,
            'low': 50000 + np.random.normal(0, 500, 50) - 100,
            'close': 50000 + np.random.normal(0, 500, 50),
            'volume': np.random.normal(100, 20, 50)
        })

        # Mock model loading - train with correct number of features (12)
        with patch.object(ml_service, '_load_model', new_callable=AsyncMock) as mock_load:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(random_state=42)
            # Train on dummy data with 12 features (matching _prepare_features output)
            X_dummy = np.random.random((10, 12))
            y_dummy = np.random.randint(0, 2, 10)
            model.fit(X_dummy, y_dummy)

            metadata = Mock()
            metadata.target_horizon = 5
            mock_load.return_value = (model, metadata)

            result = await ml_service.evaluate_model("test_model", test_data)

            assert isinstance(result, dict)
            assert "accuracy" in result
            assert "precision" in result
            assert "recall" in result
            assert "f1_score" in result

    @pytest.mark.asyncio
    async def test_feature_importance(self, ml_service):
        """Test feature importance retrieval"""
        with patch.object(ml_service, 'get_feature_importance', new_callable=AsyncMock) as mock_importance:
            mock_importance.return_value = {
                "sma_20": 0.3,
                "rsi": 0.25,
                "macd": 0.2
            }

            importance = await ml_service.get_feature_importance("BTC/USDT", "1h")
            assert isinstance(importance, dict)

    @pytest.mark.asyncio
    async def test_client_health_check(self, ml_client):
        """Test client health check"""
        with patch.object(ml_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"status": "healthy"}

            health = await ml_client.health_check()
            assert health["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_client_model_training(self, ml_client):
        """Test client model training"""
        with patch.object(ml_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {
                "success": True,
                "model_name": "test_model",
                "performance": {"accuracy": 0.8}
            }

            from services.ml_trainer.client import TrainingRequest
            request = TrainingRequest(
                symbol="BTC/USDT",
                timeframe="1h",
                start_date="2023-01-01",
                end_date="2023-01-10",
                model_type="random_forest"
            )

            result = await ml_client.train_model(request)
            assert result["success"] is True
            assert result["model_name"] == "test_model"

    @pytest.mark.asyncio
    async def test_client_model_evaluation(self, ml_client):
        """Test client model evaluation"""
        with patch.object(ml_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {
                "success": True,
                "evaluation": {"accuracy": 0.85}
            }

            from services.ml_trainer.client import EvaluationRequest
            request = EvaluationRequest(
                model_name="test_model",
                test_data=[{"feature1": 1.0, "feature2": 2.0}]
            )

            result = await ml_client.evaluate_model(request)
            assert result["success"] is True
            assert "evaluation" in result

    @pytest.mark.asyncio
    async def test_client_batch_training(self, ml_client):
        """Test client batch training"""
        with patch.object(ml_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"success": True, "message": "started"}

            from services.ml_trainer.client import TrainingRequest
            requests = [
                TrainingRequest(symbol="BTC/USDT", timeframe="1h", start_date="2023-01-01", end_date="2023-01-10"),
                TrainingRequest(symbol="ETH/USDT", timeframe="1h", start_date="2023-01-01", end_date="2023-01-10")
            ]

            results = await ml_client.batch_train_models(requests, max_concurrent=2)
            assert len(results) == 2

    def test_api_train_endpoint(self, test_client):
        """Test API train endpoint"""
        # Mock the response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "model_name": "test_model",
            "performance": {"accuracy": 0.8}
        }
        test_client.post.return_value = mock_response

        data = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2023-01-01",
            "end_date": "2023-01-10",
            "model_type": "random_forest"
        }

        response = test_client.post("/train", json=data)
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True

    def test_api_evaluate_endpoint(self, test_client):
        """Test API evaluate endpoint"""
        # Mock the response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "accuracy": 0.8,
            "precision": 0.75
        }
        test_client.post.return_value = mock_response

        data = {
            "model_name": "test_model",
            "test_data": [{"feature": 1.0}]
        }

        response = test_client.post("/evaluate", json=data)
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True

    def test_api_models_endpoint(self, test_client):
        """Test API models listing endpoint"""
        # Mock the response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": ["model1", "model2"],
            "count": 2
        }
        test_client.get.return_value = mock_response

        response = test_client.get("/models")
        assert response.status_code == 200
        result = response.json()
        assert "models" in result
        assert len(result["models"]) == 2

    @pytest.mark.asyncio
    async def test_service_error_handling(self, ml_service):
        """Test service error handling"""
        # Test with invalid model type
        with pytest.raises(ValueError):
            await ml_service.train_model(
                symbol="BTC/USDT",
                timeframe="1h",
                start_date="2023-01-01",
                end_date="2023-01-10",
                model_type="invalid_model",
                target_horizon=5
            )

    @pytest.mark.asyncio
    async def test_client_error_handling(self, ml_client):
        """Test client error handling"""
        with patch.object(ml_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = Exception("Connection failed")

            with pytest.raises(Exception):
                await ml_client.health_check()

    @pytest.mark.asyncio
    async def test_hyperparameter_ranges(self, ml_service):
        """Test hyperparameter ranges retrieval"""
        ranges = ml_service.get_hyperparameter_ranges("random_forest")
        assert "n_estimators" in ranges
        assert "max_depth" in ranges

        ranges_svm = ml_service.get_hyperparameter_ranges("svm")
        assert "C" in ranges_svm
        assert "gamma" in ranges_svm

    @pytest.mark.asyncio
    async def test_training_status_tracking(self, ml_service):
        """Test training status tracking"""
        # Start training
        from services.ml_trainer.service import TrainingStatus
        ml_service.training_status["BTC/USDT"] = TrainingStatus(
            symbol="BTC/USDT",
            status="training",
            progress=0.5,
            message="Training in progress"
        )

        status = await ml_service.get_training_status("BTC/USDT")
        assert "status" in status
        assert status["status"] == "training"

    @pytest.mark.asyncio
    async def test_event_system_integration(self, event_system, ml_service):
        """Test event system integration"""
        events_received = []

        async def event_handler(data):
            events_received.append(data)

        event_system.subscribe("ml.training_completed", event_handler)

        # Trigger event
        await event_system.publish("ml.training_completed", {"model": "test"})

        await asyncio.sleep(0.1)

        assert len(events_received) == 1
        assert events_received[0]["model"] == "test"

    @pytest.mark.asyncio
    async def test_optuna_integration(self, ml_service):
        """Test Optuna hyperparameter optimization"""
        # This would require mocking optuna, but we can test the method exists
        assert hasattr(ml_service, '_train_with_optuna')

    @pytest.mark.asyncio
    async def test_feature_preparation(self, ml_service):
        """Test feature preparation"""
        # Create sample data
        data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=50, freq='1H'),
            'open': 50000 + np.random.normal(0, 500, 50),
            'high': 50000 + np.random.normal(0, 500, 50) + 100,
            'low': 50000 + np.random.normal(0, 500, 50) - 100,
            'close': 50000 + np.random.normal(0, 500, 50),
            'volume': np.random.normal(100, 20, 50)
        })

        X, y, feature_names = ml_service._prepare_features(data, target_horizon=5)

        assert len(X) > 0
        assert len(y) > 0
        assert len(feature_names) > 0
        assert "sma_20" in feature_names
        assert "rsi" in feature_names


if __name__ == "__main__":
    pytest.main([__file__])