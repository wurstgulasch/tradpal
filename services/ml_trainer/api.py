#!/usr/bin/env python3
"""
ML Trainer Service API - FastAPI REST endpoints for ML model training.

Provides RESTful API endpoints for:
- ML model training and retraining
- Model evaluation and validation
- Feature engineering
- Hyperparameter optimization
"""

import asyncio
import logging
import uvicorn
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .service import MLTrainerService, EventSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global service instance
ml_trainer_service: Optional[MLTrainerService] = None
event_system = EventSystem()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global ml_trainer_service

    # Startup
    logger.info("Starting ML Trainer Service API")
    ml_trainer_service = MLTrainerService(event_system=event_system)

    yield

    # Shutdown
    logger.info("Shutting down ML Trainer Service API")


# Create FastAPI app
app = FastAPI(
    title="ML Trainer Service API",
    description="Machine learning model training and optimization",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TrainingRequest(BaseModel):
    """Request model for ML model training."""
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Timeframe")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    model_type: str = Field("random_forest", description="ML model type")
    target_horizon: int = Field(5, description="Prediction horizon in periods")
    use_optuna: bool = Field(False, description="Use Optuna for hyperparameter optimization")


class RetrainingRequest(BaseModel):
    """Request model for model retraining."""
    model_name: str = Field(..., description="Name of the model to retrain")
    new_data_start: str = Field(..., description="Start date for new data")
    new_data_end: str = Field(..., description="End date for new data")


class EvaluationRequest(BaseModel):
    """Request model for model evaluation."""
    model_name: str = Field(..., description="Name of the model to evaluate")
    test_data: List[Dict[str, Any]] = Field(..., description="Test data for evaluation")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "ML Trainer Service",
        "version": "1.0.0",
        "description": "Machine learning model training and optimization"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    if not ml_trainer_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    health_data = await ml_trainer_service.health_check()
    return health_data


@app.post("/train", response_model=Dict[str, Any])
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Train a new ML model for trading signals.

    Trains a machine learning model using historical data and technical indicators
    to predict trading signals.
    """
    if not ml_trainer_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Start training in background
        background_tasks.add_task(
            run_training_background,
            request.symbol,
            request.timeframe,
            request.start_date,
            request.end_date,
            request.model_type,
            request.target_horizon,
            request.use_optuna
        )

        return {
            "success": True,
            "message": "Training started in background",
            "symbol": request.symbol,
            "model_type": request.model_type
        }

    except Exception as e:
        logger.error(f"Training request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def run_training_background(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    model_type: str,
    target_horizon: int,
    use_optuna: bool
):
    """Background task to run model training."""
    try:
        await ml_trainer_service.train_model(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            model_type=model_type,
            target_horizon=target_horizon,
            use_optuna=use_optuna
        )

        logger.info(f"Training completed for {symbol} with {model_type}")

    except Exception as e:
        logger.error(f"Background training failed: {e}")


@app.post("/retrain", response_model=Dict[str, Any])
async def retrain_model(request: RetrainingRequest, background_tasks: BackgroundTasks):
    """
    Retrain an existing ML model with new data.

    Updates an existing model with new training data to maintain performance.
    """
    if not ml_trainer_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Start retraining in background
        background_tasks.add_task(
            run_retraining_background,
            request.model_name,
            request.new_data_start,
            request.new_data_end
        )

        return {
            "success": True,
            "message": "Retraining started in background",
            "model_name": request.model_name
        }

    except Exception as e:
        logger.error(f"Retraining request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def run_retraining_background(model_name: str, new_data_start: str, new_data_end: str):
    """Background task to run model retraining."""
    try:
        await ml_trainer_service.retrain_model(
            model_name=model_name,
            new_data_start=new_data_start,
            new_data_end=new_data_end
        )

        logger.info(f"Retraining completed for {model_name}")

    except Exception as e:
        logger.error(f"Background retraining failed: {e}")


@app.post("/evaluate", response_model=Dict[str, Any])
async def evaluate_model(request: EvaluationRequest):
    """
    Evaluate an ML model's performance.

    Tests the model on unseen data and returns performance metrics.
    """
    if not ml_trainer_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Convert test data to DataFrame
        import pandas as pd
        test_df = pd.DataFrame(request.test_data)

        # Evaluate model
        evaluation = await ml_trainer_service.evaluate_model(
            model_name=request.model_name,
            test_data=test_df
        )

        logger.info(f"Model evaluation completed for {request.model_name}")

        return {
            "success": True,
            "evaluation": evaluation,
            "model_name": request.model_name
        }

    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models():
    """List all trained ML models."""
    if not ml_trainer_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        models = await ml_trainer_service.list_models()
        return {
            "models": models,
            "count": len(models)
        }
    except Exception as e:
        logger.error(f"Model listing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/{model_name}")
async def get_model_info(model_name: str):
    """Get information about a specific model."""
    if not ml_trainer_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        model_info = await ml_trainer_service.get_model_info(model_name)
        return model_info
    except Exception as e:
        logger.error(f"Model info retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/models/{model_name}")
async def delete_model(model_name: str):
    """Delete a trained model."""
    if not ml_trainer_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        success = await ml_trainer_service.delete_model(model_name)

        if success:
            return {
                "success": True,
                "message": f"Model {model_name} deleted successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Model not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/features")
async def get_feature_importance(symbol: str, timeframe: str = "1h"):
    """Get feature importance for a symbol and timeframe."""
    if not ml_trainer_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        importance = await ml_trainer_service.get_feature_importance(symbol, timeframe)
        return importance
    except Exception as e:
        logger.error(f"Feature importance retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/hyperparameters/{model_type}")
async def get_hyperparameter_ranges(model_type: str):
    """Get hyperparameter ranges for a model type."""
    if not ml_trainer_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return ml_trainer_service.get_hyperparameter_ranges(model_type)


@app.get("/status/{symbol}")
async def get_training_status(symbol: str):
    """Get training status for a symbol."""
    if not ml_trainer_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        status = await ml_trainer_service.get_training_status(symbol)
        return status
    except Exception as e:
        logger.error(f"Training status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Event handlers for logging
async def handle_training_events(event_data: Dict[str, Any]):
    """Handle training completion events."""
    logger.info(f"Training event: {event_data}")

async def handle_evaluation_events(event_data: Dict[str, Any]):
    """Handle model evaluation events."""
    logger.info(f"Evaluation event: {event_data}")

# Subscribe to events
event_system.subscribe("ml.training_completed", handle_training_events)
event_system.subscribe("ml.model_evaluated", handle_evaluation_events)


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8003,
        reload=True,
        log_level="info"
    )