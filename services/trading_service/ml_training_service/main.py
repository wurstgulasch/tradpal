"""
TradPal ML Training Service API
FastAPI application for machine learning model training and ensemble methods
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import asyncio

from service import MLTrainerService

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="TradPal ML Training Service",
    description="Specialized service for machine learning model training and ensemble methods",
    version="1.0.0"
)

# Global service instance
ml_trainer = MLTrainerService()


# Pydantic models for API requests/responses
class TrainingConfig(BaseModel):
    model_type: str = Field(default="ensemble", description="Type of model to train")
    target_horizon: int = Field(default=24, description="Prediction horizon in hours")
    validation_split: float = Field(default=0.2, description="Validation data split ratio")
    random_state: int = Field(default=42, description="Random state for reproducibility")
    training_window: int = Field(default=1000, description="Training data window size")

class TrainingRequest(BaseModel):
    symbol: str = Field(..., description="Trading symbol")
    config: TrainingConfig = Field(default_factory=TrainingConfig, description="Training configuration")

class TrainingResponse(BaseModel):
    success: bool
    model_id: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    logger.info("Starting ML Training Service...")
    # Initialize service if needed
    pass


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down ML Training Service...")
    # Cleanup resources if needed
    pass


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "ml_training_service"}


@app.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Train a machine learning model"""
    try:
        logger.info(f"Starting model training for symbol: {request.symbol}")

        # Start training in background
        background_tasks.add_task(
            ml_trainer.train_model,
            request.symbol,
            request.config.dict()
        )

        return TrainingResponse(
            success=True,
            model_id=f"{request.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models():
    """List available trained models"""
    try:
        models = ml_trainer.list_available_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/{model_id}")
async def get_model_info(model_id: str):
    """Get information about a specific model"""
    try:
        info = ml_trainer.get_model_info(model_id)
        return info
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")


@app.delete("/model/{model_id}")
async def delete_model(model_id: str):
    """Delete a trained model"""
    try:
        success = ml_trainer.delete_model(model_id)
        return {"success": success}
    except Exception as e:
        logger.error(f"Failed to delete model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8012,  # Different port for ML training service
        reload=True
    )