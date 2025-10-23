#!/usr/bin/env python3
"""
MLOps Service Client

Async client for communicating with the MLOps Service API.
Provides experiment tracking, model deployment, and monitoring.
"""

import asyncio
import aiohttp
import logging
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from config.settings import MLOPS_SERVICE_URL, API_KEY

logger = logging.getLogger(__name__)


class MLOpsServiceClient:
    """Async client for MLOps Service API"""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or MLOPS_SERVICE_URL or "http://localhost:8003"
        self.session: Optional[aiohttp.ClientSession] = None
        self.api_key = API_KEY

    @asynccontextmanager
    async def _get_session(self):
        """Get or create HTTP session"""
        if self.session is None:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            self.session = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=60)
            )

        try:
            yield self.session
        except Exception:
            if self.session:
                await self.session.close()
                self.session = None
            raise

    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None

    async def health_check(self) -> bool:
        """Check if the MLOps service is healthy"""
        try:
            async with self._get_session() as session:
                async with session.get(f"{self.base_url}/health") as response:
                    return response.status == 200
        except Exception as e:
            logger.warning(f"MLOps service health check failed: {e}")
            return False

    async def log_experiment(
        self,
        model_name: str,
        metrics: Dict[str, float],
        parameters: Dict[str, Any],
        framework: str = "sklearn",
        model_artifact: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Log an ML experiment.

        Args:
            model_name: Name of the model
            metrics: Model performance metrics
            parameters: Model hyperparameters
            framework: ML framework used
            model_artifact: Model artifact data

        Returns:
            Experiment logging result
        """
        payload = {
            "model_name": model_name,
            "metrics": metrics,
            "parameters": parameters,
            "framework": framework
        }

        if model_artifact:
            payload["model_artifact"] = model_artifact

        async with self._get_session() as session:
            async with session.post(
                f"{self.base_url}/experiments/log",
                json=payload
            ) as response:
                response.raise_for_status()
                return await response.json()

    async def deploy_model(
        self,
        model_name: str,
        version: str = "1.0.0",
        model_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Deploy a model using BentoML.

        Args:
            model_name: Name of the model to deploy
            version: Model version
            model_data: Model data for deployment

        Returns:
            Deployment result
        """
        payload = {
            "model_name": model_name,
            "version": version,
            "model_data": model_data or {}
        }

        async with self._get_session() as session:
            async with session.post(
                f"{self.base_url}/models/deploy",
                json=payload
            ) as response:
                response.raise_for_status()
                return await response.json()

    async def create_drift_detector(
        self,
        model_name: str,
        reference_data: List[List[float]]
    ) -> Dict[str, Any]:
        """
        Create a drift detector for a model.

        Args:
            model_name: Name of the model
            reference_data: Reference data for drift detection

        Returns:
            Drift detector creation result
        """
        payload = {
            "model_name": model_name,
            "reference_data": reference_data
        }

        async with self._get_session() as session:
            async with session.post(
                f"{self.base_url}/drift-detectors/create",
                json=payload
            ) as response:
                response.raise_for_status()
                return await response.json()

    async def get_experiment_history(
        self,
        model_name: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get experiment history.

        Args:
            model_name: Filter by model name (optional)
            limit: Maximum number of experiments to return

        Returns:
            List of experiment results
        """
        params = {"limit": limit}
        if model_name:
            params["model_name"] = model_name

        async with self._get_session() as session:
            async with session.get(
                f"{self.base_url}/experiments/history",
                params=params
            ) as response:
                response.raise_for_status()
                return await response.json()

    async def get_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """
        Get all versions of a model.

        Args:
            model_name: Name of the model

        Returns:
            List of model versions
        """
        async with self._get_session() as session:
            async with session.get(
                f"{self.base_url}/models/{model_name}/versions"
            ) as response:
                response.raise_for_status()
                return await response.json()

    async def list_drift_detectors(self) -> Dict[str, Any]:
        """List all drift detectors"""
        async with self._get_session() as session:
            async with session.get(f"{self.base_url}/drift-detectors") as response:
                response.raise_for_status()
                return await response.json()

    async def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        async with self._get_session() as session:
            async with session.get(f"{self.base_url}/stats") as response:
                response.raise_for_status()
                return await response.json()