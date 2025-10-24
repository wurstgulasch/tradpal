"""
Async Client for Model Monitoring Service
Provides methods to interact with the model monitoring service.
"""

import aiohttp
import asyncio
from typing import Dict, List, Any, Optional
import logging
from contextlib import asynccontextmanager
import json

logger = logging.getLogger(__name__)


class ModelMonitoringClient:
    """
    Async client for the Model Monitoring Service.
    """

    def __init__(self, base_url: str = "http://localhost:8020", timeout: float = 30.0):
        """
        Initialize the client.

        Args:
            base_url: Base URL of the model monitoring service
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    @asynccontextmanager
    async def _get_session(self):
        """Get or create HTTP session with circuit breaker protection."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        try:
            yield self._session
        finally:
            pass  # Keep session alive for reuse

    async def authenticate(self) -> bool:
        """
        Authenticate with the service (placeholder for future auth implementation).

        Returns:
            True if authentication successful
        """
        # TODO: Implement JWT authentication when security service is available
        logger.info("Model Monitoring Service authentication placeholder")
        return True

    async def register_model(self, model_id: str, baseline_features: Dict[str, List[float]],
                           baseline_metrics: Optional[Dict[str, float]] = None,
                           drift_threshold: float = 0.1,
                           alert_thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Register a model for monitoring.

        Args:
            model_id: Unique model identifier
            baseline_features: Baseline feature distributions
            baseline_metrics: Baseline performance metrics
            drift_threshold: Drift detection threshold
            alert_thresholds: Performance alert thresholds

        Returns:
            Registration response
        """
        data = {
            "model_id": model_id,
            "baseline_features": baseline_features,
            "baseline_metrics": baseline_metrics,
            "drift_threshold": drift_threshold,
            "alert_thresholds": alert_thresholds
        }

        async with self._get_session() as session:
            try:
                async with session.post(f"{self.base_url}/register", json=data) as response:
                    response.raise_for_status()
                    result = await response.json()
                    logger.info(f"Registered model {model_id} for monitoring")
                    return result

            except aiohttp.ClientError as e:
                logger.error(f"Failed to register model {model_id}: {e}")
                raise

    async def monitor_prediction(self, model_id: str, prediction: float, actual: float,
                               features: Optional[Dict[str, float]] = None,
                               metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Monitor a model prediction.

        Args:
            model_id: Model identifier
            prediction: Model prediction
            actual: Actual value
            features: Input features for drift detection
            metadata: Additional metadata

        Returns:
            Monitoring response
        """
        data = {
            "model_id": model_id,
            "prediction": prediction,
            "actual": actual,
            "features": features,
            "metadata": metadata
        }

        async with self._get_session() as session:
            try:
                async with session.post(f"{self.base_url}/monitor", json=data) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return result

            except aiohttp.ClientError as e:
                logger.error(f"Failed to monitor prediction for {model_id}: {e}")
                raise

    async def get_model_status(self, model_id: str) -> Dict[str, Any]:
        """
        Get comprehensive status for a model.

        Args:
            model_id: Model identifier

        Returns:
            Model status data
        """
        async with self._get_session() as session:
            try:
                async with session.get(f"{self.base_url}/status/{model_id}") as response:
                    response.raise_for_status()
                    result = await response.json()
                    return result.get('data', {})

            except aiohttp.ClientError as e:
                logger.error(f"Failed to get status for {model_id}: {e}")
                raise

    async def get_alerts(self, model_id: Optional[str] = None, hours: int = 24) -> Dict[str, Any]:
        """
        Get alerts, optionally filtered by model.

        Args:
            model_id: Optional model identifier
            hours: Hours of history to retrieve

        Returns:
            Alerts data
        """
        params = {}
        if model_id:
            params['model_id'] = model_id
        if hours != 24:
            params['hours'] = hours

        async with self._get_session() as session:
            try:
                async with session.get(f"{self.base_url}/alerts", params=params) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return result.get('data', {})

            except aiohttp.ClientError as e:
                logger.error(f"Failed to get alerts: {e}")
                raise

    async def resolve_alert(self, alert_id: str, resolution: str = "Resolved via API") -> Dict[str, Any]:
        """
        Resolve an active alert.

        Args:
            alert_id: Alert identifier
            resolution: Resolution description

        Returns:
            Resolution response
        """
        data = {"resolution": resolution}

        async with self._get_session() as session:
            try:
                async with session.post(f"{self.base_url}/alerts/{alert_id}/resolve", json=data) as response:
                    response.raise_for_status()
                    result = await response.json()
                    logger.info(f"Resolved alert {alert_id}")
                    return result

            except aiohttp.ClientError as e:
                logger.error(f"Failed to resolve alert {alert_id}: {e}")
                raise

    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List all registered models.

        Returns:
            List of model information
        """
        async with self._get_session() as session:
            try:
                async with session.get(f"{self.base_url}/models") as response:
                    response.raise_for_status()
                    result = await response.json()
                    return result.get('data', {}).get('models', [])

            except aiohttp.ClientError as e:
                logger.error(f"Failed to list models: {e}")
                raise

    async def unregister_model(self, model_id: str) -> Dict[str, Any]:
        """
        Unregister a model from monitoring.

        Args:
            model_id: Model identifier

        Returns:
            Unregistration response
        """
        async with self._get_session() as session:
            try:
                async with session.delete(f"{self.base_url}/models/{model_id}") as response:
                    response.raise_for_status()
                    result = await response.json()
                    logger.info(f"Unregistered model {model_id}")
                    return result

            except aiohttp.ClientError as e:
                logger.error(f"Failed to unregister model {model_id}: {e}")
                raise

    async def get_service_health(self) -> Dict[str, Any]:
        """
        Get service health status.

        Returns:
            Health status data
        """
        async with self._get_session() as session:
            try:
                async with session.get(f"{self.base_url}/health") as response:
                    response.raise_for_status()
                    return await response.json()

            except aiohttp.ClientError as e:
                logger.error("Failed to get service health: {e}")
                raise

    async def cleanup_old_data(self, days: int = 30) -> Dict[str, Any]:
        """
        Clean up old monitoring data.

        Args:
            days: Days to keep data

        Returns:
            Cleanup response
        """
        data = {"days": days}

        async with self._get_session() as session:
            try:
                async with session.post(f"{self.base_url}/maintenance/cleanup", json=data) as response:
                    response.raise_for_status()
                    result = await response.json()
                    logger.info(f"Cleaned up data older than {days} days")
                    return result

            except aiohttp.ClientError as e:
                logger.error(f"Failed to cleanup old data: {e}")
                raise

    async def batch_monitor_predictions(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Monitor multiple predictions in batch.

        Args:
            predictions: List of prediction data dictionaries

        Returns:
            List of monitoring responses
        """
        tasks = []
        for pred_data in predictions:
            task = self.monitor_prediction(**pred_data)
            tasks.append(task)

        # Execute all monitoring requests concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch monitoring failed for prediction {i}: {result}")
                processed_results.append({"error": str(result)})
            else:
                processed_results.append(result)

        return processed_results

    async def get_performance_summary(self, model_id: str) -> Dict[str, Any]:
        """
        Get detailed performance summary for a model.

        Args:
            model_id: Model identifier

        Returns:
            Performance summary
        """
        status = await self.get_model_status(model_id)
        return status.get('performance', {})

    async def get_drift_analysis(self, model_id: str) -> Dict[str, Any]:
        """
        Get drift analysis for a model.

        Args:
            model_id: Model identifier

        Returns:
            Drift analysis data
        """
        status = await self.get_model_status(model_id)
        return status.get('drift', {})

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()