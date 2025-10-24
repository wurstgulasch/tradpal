"""
Model Monitoring Service
Provides comprehensive monitoring for ML models including drift detection, performance tracking, and alerting.
"""

import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import logging
import os
from datetime import datetime
import json

# Import monitoring modules
from .monitoring import DriftDetector, PerformanceTracker, AlertManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Model Monitoring Service",
    description="Comprehensive monitoring for ML models in TradPal trading system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize monitoring components
drift_detector = DriftDetector()
performance_tracker = PerformanceTracker()
alert_manager = AlertManager()

# Pydantic models for API
class ModelRegistration(BaseModel):
    model_id: str = Field(..., description="Unique model identifier")
    baseline_features: Dict[str, List[float]] = Field(..., description="Baseline feature distributions")
    baseline_metrics: Optional[Dict[str, float]] = Field(None, description="Baseline performance metrics")
    drift_threshold: Optional[float] = Field(0.1, description="Drift detection threshold")
    alert_thresholds: Optional[Dict[str, float]] = Field(None, description="Performance alert thresholds")

class PredictionData(BaseModel):
    model_id: str = Field(..., description="Model identifier")
    prediction: float = Field(..., description="Model prediction")
    actual: float = Field(..., description="Actual value")
    features: Optional[Dict[str, float]] = Field(None, description="Input features for drift detection")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class MonitoringResponse(BaseModel):
    status: str = Field(..., description="Response status")
    message: str = Field(..., description="Response message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    logger.info("Model Monitoring Service starting up")
    # Load existing models and baselines
    logger.info("Loaded existing monitoring data")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Model Monitoring Service shutting down")

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Model Monitoring Service", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "drift_detector": "active",
            "performance_tracker": "active",
            "alert_manager": "active"
        }
    }

@app.post("/register", response_model=MonitoringResponse)
async def register_model(registration: ModelRegistration):
    """
    Register a new model for monitoring.

    This endpoint registers a model with baseline data for drift detection
    and performance monitoring.
    """
    try:
        # Register for drift detection
        drift_detector.register_model(
            registration.model_id,
            registration.baseline_features,
            registration.drift_threshold
        )

        # Register for performance tracking
        if registration.baseline_metrics:
            performance_tracker.register_model(
                registration.model_id,
                registration.baseline_metrics,
                registration.alert_thresholds
            )

        logger.info(f"Registered model {registration.model_id} for monitoring")

        return MonitoringResponse(
            status="success",
            message=f"Model {registration.model_id} registered successfully",
            data={
                "model_id": registration.model_id,
                "drift_threshold": registration.drift_threshold,
                "baseline_features_count": len(registration.baseline_features)
            }
        )

    except Exception as e:
        logger.error(f"Failed to register model {registration.model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.post("/monitor", response_model=MonitoringResponse)
async def monitor_prediction(data: PredictionData, background_tasks: BackgroundTasks):
    """
    Monitor a model prediction.

    This endpoint processes a prediction for drift detection and performance tracking.
    """
    try:
        alerts_triggered = []

        # Track performance
        performance_tracker.track_prediction(
            data.model_id,
            data.prediction,
            data.actual,
            data.metadata
        )

        # Check for drift if features provided
        drift_score = 0.0
        if data.features:
            drift_score = drift_detector.calculate_drift(data.features, data.model_id)

            # Check drift threshold
            threshold = drift_detector.get_threshold(data.model_id)
            if drift_score > threshold:
                alert_id = alert_manager.generate_alert(
                    data.model_id,
                    "drift",
                    f"Drift detected: score {drift_score:.4f} exceeds threshold {threshold:.4f}",
                    severity="warning" if drift_score < threshold * 2 else "error",
                    metadata={
                        "drift_score": drift_score,
                        "threshold": threshold,
                        "features": data.features
                    }
                )
                alerts_triggered.append(alert_id)

        # Check for performance degradation
        degradation_alerts = performance_tracker.detect_performance_degradation(data.model_id)
        if degradation_alerts:
            for alert_msg in degradation_alerts:
                alert_id = alert_manager.generate_alert(
                    data.model_id,
                    "performance",
                    f"Performance degradation: {alert_msg}",
                    severity="warning",
                    metadata={
                        "degradation_details": degradation_alerts,
                        "current_metrics": performance_tracker.get_current_metrics(data.model_id)
                    }
                )
                alerts_triggered.append(alert_id)

        return MonitoringResponse(
            status="success",
            message=f"Prediction monitored for model {data.model_id}",
            data={
                "model_id": data.model_id,
                "drift_score": drift_score,
                "alerts_triggered": len(alerts_triggered),
                "alert_ids": alerts_triggered
            }
        )

    except Exception as e:
        logger.error(f"Failed to monitor prediction for {data.model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Monitoring failed: {str(e)}")

@app.get("/status/{model_id}", response_model=MonitoringResponse)
async def get_model_status(model_id: str):
    """
    Get comprehensive status for a model.

    Returns drift score, performance metrics, and active alerts.
    """
    try:
        # Get drift status
        drift_score = drift_detector.get_drift_score(model_id)
        drift_threshold = drift_detector.get_threshold(model_id)
        last_update = drift_detector.get_last_update(model_id)

        # Get performance status
        current_metrics = performance_tracker.get_current_metrics(model_id)
        baseline_metrics = performance_tracker.get_baseline_metrics(model_id)
        performance_trend = performance_tracker.get_performance_trend(model_id)

        # Get alerts
        active_alerts = alert_manager.get_active_alerts(model_id)
        alert_summary = alert_manager.get_alert_summary(model_id)

        return MonitoringResponse(
            status="success",
            message=f"Status retrieved for model {model_id}",
            data={
                "model_id": model_id,
                "drift": {
                    "current_score": drift_score,
                    "threshold": drift_threshold,
                    "last_update": last_update,
                    "drift_detected": drift_score > drift_threshold
                },
                "performance": {
                    "current_metrics": current_metrics,
                    "baseline_metrics": baseline_metrics,
                    "trend": performance_trend,
                    "degradation_alerts": performance_tracker.detect_performance_degradation(model_id)
                },
                "alerts": {
                    "active_count": len(active_alerts),
                    "active_alerts": active_alerts[:5],  # Last 5 active alerts
                    "summary": alert_summary
                }
            }
        )

    except Exception as e:
        logger.error(f"Failed to get status for {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")

@app.get("/alerts", response_model=MonitoringResponse)
async def get_alerts(model_id: Optional[str] = None, hours: int = 24):
    """
    Get alerts, optionally filtered by model.

    Returns active alerts and recent alert history.
    """
    try:
        active_alerts = alert_manager.get_active_alerts(model_id)
        alert_history = alert_manager.get_alert_history(model_id, hours)
        alert_summary = alert_manager.get_alert_summary(model_id)

        return MonitoringResponse(
            status="success",
            message="Alerts retrieved successfully",
            data={
                "active_alerts": active_alerts,
                "alert_history": alert_history[:50],  # Limit history
                "summary": alert_summary,
                "filters": {
                    "model_id": model_id,
                    "hours": hours
                }
            }
        )

    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Alert retrieval failed: {str(e)}")

@app.post("/alerts/{alert_id}/resolve", response_model=MonitoringResponse)
async def resolve_alert(alert_id: str, resolution: str = "Manually resolved"):
    """
    Resolve an active alert.
    """
    try:
        alert_manager.resolve_alert(alert_id, resolution)

        return MonitoringResponse(
            status="success",
            message=f"Alert {alert_id} resolved",
            data={
                "alert_id": alert_id,
                "resolution": resolution,
                "resolved_at": datetime.now().isoformat()
            }
        )

    except Exception as e:
        logger.error(f"Failed to resolve alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Alert resolution failed: {str(e)}")

@app.get("/models", response_model=MonitoringResponse)
async def list_models():
    """
    List all registered models.
    """
    try:
        models = []

        # Get models from drift detector
        for model_id in drift_detector.baseline_stats.keys():
            model_info = {
                "model_id": model_id,
                "drift_threshold": drift_detector.get_threshold(model_id),
                "last_update": drift_detector.get_last_update(model_id),
                "feature_count": len(drift_detector.baseline_stats[model_id].get('feature_names', [])),
                "has_performance_tracking": model_id in performance_tracker.baseline_performance
            }
            models.append(model_info)

        return MonitoringResponse(
            status="success",
            message=f"Found {len(models)} registered models",
            data={
                "models": models,
                "count": len(models)
            }
        )

    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=f"Model listing failed: {str(e)}")

@app.delete("/models/{model_id}", response_model=MonitoringResponse)
async def unregister_model(model_id: str):
    """
    Unregister a model from monitoring.
    """
    try:
        # Unregister from all components
        drift_detector.unregister_model(model_id)
        performance_tracker.unregister_model(model_id)

        # Resolve any active alerts
        active_alerts = alert_manager.get_active_alerts(model_id)
        for alert in active_alerts:
            alert_manager.resolve_alert(alert['id'], "Model unregistered")

        return MonitoringResponse(
            status="success",
            message=f"Model {model_id} unregistered successfully",
            data={
                "model_id": model_id,
                "alerts_resolved": len(active_alerts)
            }
        )

    except Exception as e:
        logger.error(f"Failed to unregister model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Unregistration failed: {str(e)}")

@app.post("/maintenance/cleanup", response_model=MonitoringResponse)
async def cleanup_old_data(days: int = 30):
    """
    Clean up old monitoring data and resolved alerts.
    """
    try:
        alert_manager.cleanup_old_alerts(days)

        return MonitoringResponse(
            status="success",
            message=f"Cleaned up data older than {days} days",
            data={
                "cleanup_days": days,
                "timestamp": datetime.now().isoformat()
            }
        )

    except Exception as e:
        logger.error(f"Failed to cleanup old data: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

if __name__ == "__main__":
    # Run the service
    port = int(os.getenv("MODEL_MONITORING_PORT", "8020"))
    uvicorn.run(app, host="0.0.0.0", port=port)