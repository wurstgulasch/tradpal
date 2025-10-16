"""
FastAPI REST API for MLOps Service
Provides endpoints for experiment tracking, model deployment, and monitoring.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from .service import MLOpsService, MLOpsConfig, ExperimentResult, ModelMetadata, DriftAlert


# Pydantic models for API
class ExperimentLogRequest(BaseModel):
    """Request to log an experiment."""
    model_name: str = Field(..., description="Name of the model")
    metrics: Dict[str, float] = Field(..., description="Model performance metrics")
    parameters: Dict[str, Any] = Field(..., description="Model hyperparameters")
    framework: str = Field("sklearn", description="ML framework used")
    model_artifact: Optional[Dict[str, Any]] = Field(None, description="Model artifact data")


class ModelDeployRequest(BaseModel):
    """Request to deploy a model."""
    model_name: str = Field(..., description="Name of the model to deploy")
    version: str = Field("1.0.0", description="Model version")
    model_data: Dict[str, Any] = Field(..., description="Model data for deployment")


class DriftDetectorRequest(BaseModel):
    """Request to create a drift detector."""
    model_name: str = Field(..., description="Name of the model")
    reference_data: List[List[float]] = Field(..., description="Reference data for drift detection")


class HealthResponse(BaseModel):
    """Health check response."""
    service: str
    status: str
    timestamp: str
    components: Dict[str, bool]
    models_deployed: int
    experiments_logged: int


class MLOpsAPI:
    """FastAPI application for MLOps service."""

    def __init__(self, config: MLOpsConfig, mlops_service: MLOpsService):
        self.config = config
        self.mlops_service = mlops_service
        self.logger = logging.getLogger(__name__)

        # Create FastAPI app
        self.app = FastAPI(
            title="TradPal MLOps Service",
            description="MLOps API for experiment tracking, model deployment, and monitoring",
            version="1.0.0"
        )

        # Add CORS middleware
        if config.bentoml.enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

        # Register routes
        self._register_routes()

    def _register_routes(self):
        """Register API routes."""

        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Get service health status."""
            return await self.mlops_service.health_check()

        @self.app.post("/experiments/log", response_model=ExperimentResult)
        async def log_experiment(request: ExperimentLogRequest, background_tasks: BackgroundTasks):
            """Log an ML experiment."""
            try:
                # For now, we'll create a placeholder model object
                # In a real implementation, you'd receive the actual model
                model = {"placeholder": True}  # Placeholder

                result = await self.mlops_service.log_experiment(
                    model_name=request.model_name,
                    metrics=request.metrics,
                    parameters=request.parameters,
                    model=model,
                    framework=request.framework
                )
                return result
            except Exception as e:
                self.logger.error(f"Failed to log experiment: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/models/deploy")
        async def deploy_model(request: ModelDeployRequest):
            """Deploy a model using BentoML."""
            try:
                # Placeholder model deployment
                model = {"placeholder": True}  # Would be actual model

                deployment_path = await self.mlops_service.deploy_model(
                    model_name=request.model_name,
                    model=model,
                    version=request.version
                )
                return {"status": "success", "deployment_path": deployment_path}
            except Exception as e:
                self.logger.error(f"Failed to deploy model: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/drift-detectors/create")
        async def create_drift_detector(request: DriftDetectorRequest):
            """Create a drift detector for a model."""
            try:
                import numpy as np
                reference_data = np.array(request.reference_data)

                detector_id = await self.mlops_service.create_drift_detector(
                    model_name=request.model_name,
                    reference_data=reference_data
                )
                return {"status": "success", "detector_id": detector_id, "message": f"Drift detector created for {request.model_name}"}
            except Exception as e:
                self.logger.error(f"Failed to create drift detector: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/experiments/history")
        async def get_experiment_history(
            model_name: Optional[str] = None,
            limit: int = 10
        ) -> List[ExperimentResult]:
            """Get experiment history."""
            try:
                return await self.mlops_service.get_experiment_history(model_name, limit)
            except Exception as e:
                self.logger.error(f"Failed to get experiment history: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/models/{model_name}/versions")
        async def get_model_versions(model_name: str) -> List[ModelMetadata]:
            """Get all versions of a model."""
            try:
                return await self.mlops_service.get_model_versions(model_name)
            except Exception as e:
                self.logger.error(f"Failed to get model versions: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/drift-detectors")
        async def list_drift_detectors():
            """List all drift detectors."""
            return {
                "detectors": list(self.mlops_service.drift_detectors.keys()),
                "count": len(self.mlops_service.drift_detectors)
            }

        @self.app.get("/stats")
        async def get_service_stats():
            """Get service statistics."""
            return {
                "experiments_logged": 0,  # Would need to implement counting
                "models_deployed": len(self.mlops_service.drift_detectors),
                "drift_detectors_active": len(self.mlops_service.drift_detectors),
                "uptime": "unknown"  # Would need to track service start time
            }

    async def start_server(self, host: str = "0.0.0.0", port: int = 8001):
        """Start the FastAPI server."""
        self.logger.info(f"Starting MLOps API server on {host}:{port}")
        config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()

    def run_server(self, host: str = "0.0.0.0", port: int = 8001):
        """Run the server (blocking)."""
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info"
        )


# Global service instance for dependency injection
_mlops_service: Optional[MLOpsService] = None


def get_mlops_service() -> MLOpsService:
    """Dependency injection for MLOps service."""
    if _mlops_service is None:
        raise RuntimeError("MLOps service not initialized")
    return _mlops_service


def create_mlops_api(config: MLOpsConfig, mlops_service: MLOpsService) -> MLOpsAPI:
    """Create MLOps API instance."""
    global _mlops_service
    _mlops_service = mlops_service
    return MLOpsAPI(config, mlops_service)


if __name__ == "__main__":
    # For testing purposes
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    async def main():
        config = MLOpsConfig()
        service = MLOpsService(config)
        await service.start()

        api = create_mlops_api(config, service)
        await api.start_server()

    asyncio.run(main())