#!/usr/bin/env python3
"""
ML Trainer Service Client - Async client for ML model training service.

Provides async HTTP client for:
- Model training and retraining
- Model evaluation and management
- Feature importance analysis
- Training status monitoring
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

import aiohttp
from aiohttp import ClientTimeout

from config.settings import ML_TRAINER_SERVICE_URL, REQUEST_TIMEOUT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingRequest:
    """Request data for model training."""
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    model_type: str = "random_forest"
    target_horizon: int = 5
    use_optuna: bool = False


@dataclass
class RetrainingRequest:
    """Request data for model retraining."""
    model_name: str
    new_data_start: str
    new_data_end: str


@dataclass
class EvaluationRequest:
    """Request data for model evaluation."""
    model_name: str
    test_data: List[Dict[str, Any]]


class MLTrainerServiceClient:
    """Async client for ML Trainer Service."""

    def __init__(self, base_url: str = ML_TRAINER_SERVICE_URL):
        self.base_url = base_url.rstrip('/')
        self.timeout = ClientTimeout(total=REQUEST_TIMEOUT)

        # HTTP session
        self._session: Optional[aiohttp.ClientSession] = None

        logger.info(f"ML Trainer Service Client initialized with URL: {self.base_url}")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def start(self):
        """Start the HTTP session."""
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=self.timeout)

    async def close(self):
        """Close the HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make HTTP request to service."""
        if not self._session:
            await self.start()

        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        try:
            async with self._session.request(method, url, json=data, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")

        except aiohttp.ClientError as e:
            logger.error(f"Request failed: {e}")
            raise Exception(f"Service request failed: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Check service health."""
        return await self._make_request("GET", "/health")

    async def train_model(self, request: TrainingRequest) -> Dict[str, Any]:
        """
        Start model training.

        Args:
            request: Training request data

        Returns:
            Training response
        """
        data = {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "model_type": request.model_type,
            "target_horizon": request.target_horizon,
            "use_optuna": request.use_optuna
        }

        response = await self._make_request("POST", "/train", data)

        logger.info(f"Training started for {request.symbol} with {request.model_type}")
        return response

    async def retrain_model(self, request: RetrainingRequest) -> Dict[str, Any]:
        """
        Retrain an existing model.

        Args:
            request: Retraining request data

        Returns:
            Retraining response
        """
        data = {
            "model_name": request.model_name,
            "new_data_start": request.new_data_start,
            "new_data_end": request.new_data_end
        }

        response = await self._make_request("POST", "/retrain", data)

        logger.info(f"Retraining started for {request.model_name}")
        return response

    async def evaluate_model(self, request: EvaluationRequest) -> Dict[str, Any]:
        """
        Evaluate a trained model.

        Args:
            request: Evaluation request data

        Returns:
            Evaluation results
        """
        data = {
            "model_name": request.model_name,
            "test_data": request.test_data
        }

        response = await self._make_request("POST", "/evaluate", data)

        logger.info(f"Model evaluation completed for {request.model_name}")
        return response

    async def list_models(self) -> List[str]:
        """List all trained models."""
        response = await self._make_request("GET", "/models")
        return response.get("models", [])

    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        return await self._make_request("GET", f"/models/{model_name}")

    async def delete_model(self, model_name: str) -> bool:
        """Delete a trained model."""
        response = await self._make_request("DELETE", f"/models/{model_name}")
        return response.get("success", False)

    async def get_feature_importance(self, symbol: str, timeframe: str = "1h") -> Dict[str, float]:
        """Get feature importance for a symbol and timeframe."""
        params = {"symbol": symbol, "timeframe": timeframe}
        response = await self._make_request("GET", "/features", params=params)
        return response

    async def get_hyperparameter_ranges(self, model_type: str) -> Dict[str, Any]:
        """Get hyperparameter ranges for a model type."""
        return await self._make_request("GET", f"/hyperparameters/{model_type}")

    async def get_training_status(self, symbol: str) -> Dict[str, Any]:
        """Get training status for a symbol."""
        return await self._make_request("GET", f"/status/{symbol}")

    async def wait_for_training_completion(
        self,
        symbol: str,
        timeout: int = 3600,
        poll_interval: int = 10
    ) -> Dict[str, Any]:
        """
        Wait for training completion.

        Args:
            symbol: Symbol being trained
            timeout: Maximum wait time in seconds
            poll_interval: Polling interval in seconds

        Returns:
            Final training status
        """
        start_time = asyncio.get_event_loop().time()

        while True:
            status = await self.get_training_status(symbol)

            if status["status"] in ["completed", "failed"]:
                return status

            if asyncio.get_event_loop().time() - start_time > timeout:
                raise TimeoutError(f"Training timeout after {timeout} seconds")

            await asyncio.sleep(poll_interval)

    async def train_and_wait(
        self,
        request: TrainingRequest,
        timeout: int = 3600
    ) -> Dict[str, Any]:
        """
        Train a model and wait for completion.

        Args:
            request: Training request
            timeout: Maximum wait time in seconds

        Returns:
            Training results
        """
        # Start training
        train_response = await self.train_model(request)

        # Wait for completion
        status = await self.wait_for_training_completion(request.symbol, timeout)

        if status["status"] == "failed":
            raise Exception(f"Training failed: {status['message']}")

        # Get final model info
        models = await self.list_models()
        latest_model = max([m for m in models if m.startswith(f"{request.symbol}_{request.timeframe}")])

        model_info = await self.get_model_info(latest_model)

        return {
            "training_response": train_response,
            "final_status": status,
            "model_info": model_info
        }

    async def batch_train_models(
        self,
        requests: List[TrainingRequest],
        max_concurrent: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Train multiple models concurrently.

        Args:
            requests: List of training requests
            max_concurrent: Maximum concurrent trainings

        Returns:
            List of training responses
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def train_with_semaphore(request):
            async with semaphore:
                return await self.train_model(request)

        tasks = [train_with_semaphore(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "error": str(result),
                    "symbol": requests[i].symbol
                })
            else:
                processed_results.append(result)

        return processed_results

    async def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        health = await self.health_check()
        models = await self.list_models()

        return {
            "service_health": health,
            "total_models": len(models),
            "models": models
        }