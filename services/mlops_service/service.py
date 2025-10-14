"""
MLOps Service for TradPal
Provides MLflow experiment tracking, BentoML model serving, and drift detection.
"""

from typing import Dict, Any, Optional, List
import asyncio
import logging
from datetime import datetime
from pathlib import Path
import json
import time

from pydantic import BaseModel, Field
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow.entities import Experiment
import bentoml
import numpy as np

from services.notification_service.service import NotificationService, NotificationType, NotificationPriority


class MLflowConfig(BaseModel):
    """Configuration for MLflow integration."""
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "tradpal_trading_models"
    artifact_location: Optional[str] = None
    enable_registry: bool = True


class BentoMLConfig(BaseModel):
    """Configuration for BentoML model serving."""
    service_name: str = "tradpal_trading_model"
    service_version: str = "1.0.0"
    api_port: int = 3000
    enable_cors: bool = True


class DriftDetectionConfig(BaseModel):
    """Configuration for drift detection."""
    enable_drift_detection: bool = True
    drift_threshold: float = 0.05
    reference_data_size: int = 1000
    check_interval_minutes: int = 60
    model_update_threshold: float = 0.1


class MLOpsConfig(BaseModel):
    """Main configuration for MLOps service."""
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    bentoml: BentoMLConfig = Field(default_factory=BentoMLConfig)
    drift_detection: DriftDetectionConfig = Field(default_factory=DriftDetectionConfig)
    max_workers: int = 4
    enable_notifications: bool = True


class ExperimentResult(BaseModel):
    """Result of an ML experiment."""
    experiment_id: str
    run_id: str
    model_name: str
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    artifacts: List[str]
    timestamp: datetime
    status: str  # "success", "failed", "running"


class ModelMetadata(BaseModel):
    """Metadata for a trained model."""
    model_name: str
    version: str
    framework: str  # "pytorch", "sklearn", "tensorflow"
    experiment_id: str
    run_id: str
    metrics: Dict[str, float]
    created_at: datetime
    deployed: bool = False
    drift_detected: bool = False


class DriftAlert(BaseModel):
    """Alert for model drift detection."""
    model_name: str
    model_version: str
    drift_score: float
    threshold: float
    timestamp: datetime
    features_affected: List[str]
    severity: str  # "low", "medium", "high", "critical"


class MLOpsService:
    """
    MLOps Service providing MLflow experiment tracking,
    BentoML model serving, and drift detection.
    """

    def __init__(self, config: MLOpsConfig, notification_service: Optional[NotificationService] = None):
        self.config = config
        self.notification_service = notification_service
        self.logger = logging.getLogger(__name__)

        # MLflow components
        self.mlflow_client: Optional[MlflowClient] = None
        self.current_experiment: Optional[Experiment] = None

        # BentoML components
        self.bento_service: Optional[Any] = None

        # Drift detection components (simplified implementation)
        self.drift_detectors: Dict[str, Dict[str, Any]] = {}
        self.reference_data: Dict[str, np.ndarray] = {}

        # Service state
        self.is_running = False
        self._drift_check_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the MLOps service."""
        self.logger.info("Starting MLOps Service...")

        # Initialize MLflow
        await self._initialize_mlflow()

        # Initialize BentoML
        await self._initialize_bentoml()

        # Initialize drift detection
        if self.config.drift_detection.enable_drift_detection:
            await self._initialize_drift_detection()

        self.is_running = True
        self.logger.info("MLOps Service started successfully")

        # Start drift monitoring if enabled
        if self.config.drift_detection.enable_drift_detection:
            self._start_drift_monitoring()

    async def stop(self) -> None:
        """Stop the MLOps service."""
        self.logger.info("Stopping MLOps Service...")

        # Stop drift monitoring
        if self._drift_check_task:
            self._drift_check_task.cancel()
            try:
                await self._drift_check_task
            except asyncio.CancelledError:
                pass

        self.is_running = False
        self.logger.info("MLOps Service stopped")

    async def _initialize_mlflow(self) -> None:
        """Initialize MLflow tracking."""
        try:
            mlflow.set_tracking_uri(self.config.mlflow.tracking_uri)

            # Create or get experiment
            try:
                experiment = mlflow.get_experiment_by_name(self.config.mlflow.experiment_name)
                if experiment is None:
                    experiment_id = mlflow.create_experiment(
                        self.config.mlflow.experiment_name,
                        artifact_location=self.config.mlflow.artifact_location
                    )
                    experiment = mlflow.get_experiment(experiment_id)
                self.current_experiment = experiment
            except Exception as e:
                self.logger.warning(f"Could not create/get MLflow experiment: {e}")

            self.mlflow_client = MlflowClient(self.config.mlflow.tracking_uri)
            self.logger.info(f"MLflow initialized with experiment: {self.config.mlflow.experiment_name}")

        except Exception as e:
            self.logger.error(f"Failed to initialize MLflow: {e}")
            raise

    async def _initialize_bentoml(self) -> None:
        """Initialize BentoML."""
        try:
            # BentoML initialization is mostly automatic
            # Service will be created when models are deployed
            self.logger.info("BentoML initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize BentoML: {e}")
            raise

    async def _initialize_drift_detection(self) -> None:
        """Initialize drift detection components."""
        try:
            # Load existing drift detectors if available
            detector_dir = Path("cache/ml_models/drift_detectors")
            detector_dir.mkdir(parents=True, exist_ok=True)

            # For now, we'll use a simple statistical approach
            # In a real implementation, you might load saved detectors
            self.logger.info("Drift detection initialized (simplified)")

        except Exception as e:
            self.logger.error(f"Failed to initialize drift detection: {e}")
            raise

    def _start_drift_monitoring(self) -> None:
        """Start periodic drift monitoring."""
        if self._drift_check_task is None:
            self._drift_check_task = asyncio.create_task(self._drift_monitoring_loop())

    async def _drift_monitoring_loop(self) -> None:
        """Periodic drift monitoring loop."""
        while self.is_running:
            try:
                await self._check_model_drift()
                await asyncio.sleep(self.config.drift_detection.check_interval_minutes * 60)
            except Exception as e:
                self.logger.error(f"Error in drift monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def _check_model_drift(self) -> None:
        """Check for model drift across all deployed models."""
        for detector_id, detector_config in self.drift_detectors.items():
            try:
                # Get recent prediction data (this would come from your prediction service)
                # For now, we'll simulate with random data
                recent_data = np.random.randn(100, 10)  # Placeholder

                # Simple statistical drift detection
                current_mean = np.mean(recent_data)
                current_std = np.std(recent_data)

                reference_mean = detector_config["reference_mean"]
                reference_std = detector_config["reference_std"]
                threshold = detector_config["threshold"]

                # Calculate drift score (z-score)
                drift_score = abs(current_mean - reference_mean) / (reference_std + 1e-6)

                if drift_score > threshold:
                    await self._handle_drift_detected(detector_id, drift_score, detector_config)

            except Exception as e:
                self.logger.error(f"Error checking drift for detector {detector_id}: {e}")

    async def _handle_drift_detected(self, detector_id: str, drift_score: float, detector_config: Dict[str, Any]) -> None:
        """Handle detected model drift."""
        threshold = self.config.drift_detection.drift_threshold

        # Determine severity
        if drift_score > threshold * 2:
            severity = "critical"
        elif drift_score > threshold * 1.5:
            severity = "high"
        elif drift_score > threshold:
            severity = "medium"
        else:
            severity = "low"

        alert = DriftAlert(
            model_name=detector_config["model_name"],
            model_version="current",  # Would need to track versions
            drift_score=drift_score,
            threshold=threshold,
            timestamp=datetime.now(),
            features_affected=[],  # Would need feature importance analysis
            severity=severity
        )

        self.logger.warning(f"Model drift detected: {alert}")

        # Send notification if service is available
        if self.notification_service and self.config.enable_notifications:
            await self.notification_service.send_notification(
                message=f"🚨 Model Drift Alert: {detector_config['model_name']} (Score: {drift_score:.3f})",
                title=f"Model Drift - {detector_config['model_name']}",
                notification_type=NotificationType.ALERT,
                priority=NotificationPriority.HIGH if severity in ["high", "critical"] else NotificationPriority.NORMAL
            )

    async def log_experiment(self, model_name: str, metrics: Dict[str, float],
                           parameters: Dict[str, Any], model: Any,
                           framework: str = "sklearn") -> ExperimentResult:
        """Log an ML experiment to MLflow."""
        if not self.mlflow_client:
            raise RuntimeError("MLflow not initialized")

        with mlflow.start_run(experiment_id=self.current_experiment.experiment_id if self.current_experiment else None) as run:
            # Log parameters
            for key, value in parameters.items():
                mlflow.log_param(key, value)

            # Log metrics
            for key, value in metrics.items():
                mlflow.log_metric(key, value)

            # Log model based on framework
            if framework == "sklearn":
                mlflow.sklearn.log_model(model, model_name)
            elif framework == "pytorch":
                mlflow.pytorch.log_model(model, model_name)
            else:
                # Generic model logging
                mlflow.log_artifact(str(model), model_name)

            # Get run info
            run_info = mlflow.active_run().info

            result = ExperimentResult(
                experiment_id=run_info.experiment_id,
                run_id=run_info.run_id,
                model_name=model_name,
                metrics=metrics,
                parameters=parameters,
                artifacts=[model_name],
                timestamp=datetime.now(),
                status="success"
            )

            self.logger.info(f"Logged experiment: {model_name} (Run ID: {run_info.run_id})")
            return result

    async def deploy_model(self, model_name: str, model: Any, version: str = "1.0.0") -> str:
        """Deploy a model using BentoML."""
        try:
            # For BentoML 1.x, we'll use a simpler approach
            # Save the model directly (placeholder for actual deployment)
            model_path = f"models/{model_name}/{version}"
            self.logger.info(f"Model {model_name} would be deployed to BentoML at: {model_path}")

            # In a real implementation, you would:
            # 1. Save the model using bentoml.save()
            # 2. Return the deployment path
            # For now, return a placeholder path
            return f"bentoml://{model_name}:{version}"

        except Exception as e:
            self.logger.error(f"Failed to deploy model {model_name}: {e}")
            raise

    async def create_drift_detector(self, model_name: str, reference_data: np.ndarray) -> str:
        """Create a drift detector for a model."""
        try:
            detector_id = f"{model_name}_drift_detector_{int(time.time())}"

            # Simple statistical drift detection using mean and std
            detector_config = {
                "model_name": model_name,
                "detector_type": "statistical",
                "reference_mean": float(np.mean(reference_data)),
                "reference_std": float(np.std(reference_data)),
                "threshold": 3.0,  # 3-sigma rule
                "created_at": datetime.now().isoformat()
            }

            # Save detector configuration
            detector_dir = Path("cache/ml_models/drift_detectors")
            detector_dir.mkdir(parents=True, exist_ok=True)
            detector_file = detector_dir / f"{detector_id}.json"

            with open(detector_file, 'w') as f:
                json.dump(detector_config, f, indent=2)

            self.drift_detectors[detector_id] = detector_config
            self.logger.info(f"Created drift detector: {detector_id}")

            return detector_id

        except Exception as e:
            self.logger.error(f"Failed to create drift detector for {model_name}: {e}")
            raise

    async def get_experiment_history(self, model_name: Optional[str] = None,
                                   limit: int = 10) -> List[ExperimentResult]:
        """Get experiment history from MLflow."""
        if not self.mlflow_client:
            raise RuntimeError("MLflow not initialized")

        try:
            # Get runs from current experiment
            experiment_id = self.current_experiment.experiment_id if self.current_experiment else None
            if experiment_id:
                runs = self.mlflow_client.search_runs([experiment_id], max_results=limit)
            else:
                runs = []

            results = []
            for run in runs:
                # Filter by model name if specified
                if model_name and run.data.tags.get('model_name') != model_name:
                    continue

                result = ExperimentResult(
                    experiment_id=run.info.experiment_id,
                    run_id=run.info.run_id,
                    model_name=run.data.tags.get('model_name', 'unknown'),
                    metrics=dict(run.data.metrics),
                    parameters=dict(run.data.params),
                    artifacts=[],  # Would need to fetch from artifacts
                    timestamp=datetime.fromtimestamp(run.info.start_time / 1000),
                    status="success" if run.info.status == "FINISHED" else "failed"
                )
                results.append(result)

            return results

        except Exception as e:
            self.logger.error(f"Failed to get experiment history: {e}")
            return []

    async def get_model_versions(self, model_name: str) -> List[ModelMetadata]:
        """Get all versions of a model."""
        if not self.mlflow_client:
            raise RuntimeError("MLflow not initialized")

        try:
            versions = self.mlflow_client.get_model_version(model_name)
            # This is simplified - would need to get all versions
            return []
        except Exception as e:
            self.logger.error(f"Failed to get model versions for {model_name}: {e}")
            return []

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of MLOps service."""
        health = {
            "service": "mlops_service",
            "status": "healthy" if self.is_running else "stopped",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "mlflow": self.mlflow_client is not None,
                "bentoml": True,  # BentoML is always available
                "drift_detection": len(self.drift_detectors) > 0
            },
            "models_deployed": len(self.drift_detectors),
            "experiments_logged": 0  # Would need to count from MLflow
        }

        return health