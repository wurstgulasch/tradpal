"""
TradPal Backtesting Service API
FastAPI application for unified backtesting operations
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import uvicorn

from .orchestrator import BacktestingServiceOrchestrator

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="TradPal Backtesting Service",
    description="Unified API for backtesting, ML training, optimization, and walk-forward analysis",
    version="1.0.0"
)

# Global orchestrator instance
orchestrator = BacktestingServiceOrchestrator()


# Pydantic models for API requests/responses
class StrategyConfig(BaseModel):
    name: str = Field(..., description="Strategy name")
    type: str = Field(..., description="Strategy type (e.g., 'moving_average', 'ml_based')")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")

class BacktestRequest(BaseModel):
    strategy: StrategyConfig
    data: Dict[str, Any] = Field(..., description="OHLC data as dict")
    config: Dict[str, Any] = Field(default_factory=dict, description="Backtest configuration")

class BacktestResponse(BaseModel):
    success: bool
    results: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class OptimizationRequest(BaseModel):
    strategy_name: str
    param_ranges: Dict[str, List] = Field(..., description="Parameter ranges for optimization")
    data: Dict[str, Any] = Field(..., description="OHLC data as dict")
    config: Dict[str, Any] = Field(default_factory=dict, description="Optimization configuration")

class OptimizationResponse(BaseModel):
    success: bool
    results: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class MLTrainingRequest(BaseModel):
    strategy: StrategyConfig
    data: Dict[str, Any] = Field(..., description="OHLC data as dict")
    config: Dict[str, Any] = Field(default_factory=dict, description="ML training configuration")

class MLTrainingResponse(BaseModel):
    success: bool
    results: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class WalkForwardRequest(BaseModel):
    strategy_name: str
    param_ranges: Dict[str, List] = Field(..., description="Parameter ranges for analysis")
    data: Dict[str, Any] = Field(..., description="OHLC data as dict")
    config: Dict[str, Any] = Field(default_factory=dict, description="Walk-forward configuration")

class WalkForwardResponse(BaseModel):
    success: bool
    results: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class WorkflowRequest(BaseModel):
    strategy: StrategyConfig
    data: Dict[str, Any] = Field(..., description="OHLC data as dict")
    config: Dict[str, Any] = Field(default_factory=dict, description="Workflow configuration")

class WorkflowResponse(BaseModel):
    success: bool
    results: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# Helper functions
def convert_dict_to_dataframe(data_dict: Dict[str, Any]) -> pd.DataFrame:
    """Convert dictionary data to pandas DataFrame"""
    try:
        df = pd.DataFrame(data_dict)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        raise ValueError(f"Failed to convert data to DataFrame: {e}")

def validate_data_format(data: Dict[str, Any]) -> None:
    """Validate that data contains required OHLC columns"""
    required_columns = ['open', 'high', 'low', 'close']
    if not all(col in data for col in required_columns):
        raise ValueError(f"Data must contain OHLC columns: {required_columns}")


# API endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize the orchestrator on startup"""
    try:
        await orchestrator.initialize()
        logger.info("Backtesting Service API started successfully")
    except Exception as e:
        logger.error(f"Failed to start Backtesting Service API: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown the orchestrator on shutdown"""
    try:
        await orchestrator.shutdown()
        logger.info("Backtesting Service API shut down successfully")
    except Exception as e:
        logger.error(f"Error during Backtesting Service API shutdown: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "backtesting_service",
        "timestamp": datetime.now().isoformat(),
        "components": orchestrator.get_service_status()
    }

@app.post("/backtest", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest):
    """Run a quick backtest"""
    try:
        # Validate and convert data
        validate_data_format(request.data)
        data_df = convert_dict_to_dataframe(request.data)

        # Run backtest
        results = await orchestrator.run_quick_backtest(request.strategy.dict(), data_df)

        return BacktestResponse(success=True, results=results)

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize", response_model=OptimizationResponse)
async def optimize_strategy(request: OptimizationRequest):
    """Optimize strategy parameters"""
    try:
        # Validate and convert data
        validate_data_format(request.data)
        data_df = convert_dict_to_dataframe(request.data)

        # Run optimization
        results = await orchestrator.optimize_strategy(
            request.strategy_name, request.param_ranges, data_df
        )

        return OptimizationResponse(success=True, results=results)

    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ml/train", response_model=MLTrainingResponse)
async def train_ml_model(request: MLTrainingRequest):
    """Train ML model for strategy"""
    try:
        # Validate and convert data
        validate_data_format(request.data)
        data_df = convert_dict_to_dataframe(request.data)

        # Train model
        results = await orchestrator.train_ml_model(
            request.strategy.dict(), data_df, request.config
        )

        return MLTrainingResponse(success=True, results=results)

    except Exception as e:
        logger.error(f"ML training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/walk-forward", response_model=WalkForwardResponse)
async def run_walk_forward(request: WalkForwardRequest):
    """Run walk-forward analysis"""
    try:
        # Validate and convert data
        validate_data_format(request.data)
        data_df = convert_dict_to_dataframe(request.data)

        # Run walk-forward analysis
        results = await orchestrator.run_walk_forward_analysis(
            request.strategy_name, request.param_ranges, data_df, request.config
        )

        return WalkForwardResponse(success=True, results=results)

    except Exception as e:
        logger.error(f"Walk-forward analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/workflow", response_model=WorkflowResponse)
async def run_complete_workflow(request: WorkflowRequest, background_tasks: BackgroundTasks):
    """Run complete backtesting workflow"""
    try:
        # Validate and convert data
        validate_data_format(request.data)
        data_df = convert_dict_to_dataframe(request.data)

        # Run workflow (potentially in background for long-running tasks)
        results = await orchestrator.run_complete_backtesting_workflow(
            request.strategy.dict(), data_df, request.config
        )

        return WorkflowResponse(success=True, results=results)

    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_service_status():
    """Get detailed service status"""
    return orchestrator.get_service_status()

@app.get("/config/default")
async def get_default_config():
    """Get default workflow configuration"""
    return orchestrator.get_default_workflow_config()


# Main entry point for running the service
def main():
    """Run the Backtesting Service API"""
    import argparse

    parser = argparse.ArgumentParser(description="TradPal Backtesting Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    logger.info(f"Starting Backtesting Service API on {args.host}:{args.port}")

    uvicorn.run(
        "services.backtesting_service.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()