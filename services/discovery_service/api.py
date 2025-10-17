#!/usr/bin/env python3
"""
Discovery Service API - FastAPI REST endpoints for genetic algorithm optimization.

Provides RESTful API endpoints for:
- Starting optimization runs
- Monitoring optimization status
- Retrieving optimization results
- Managing active optimizations
"""

import asyncio
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import uvicorn

# Import service
from .service import DiscoveryService, EventSystem

# Create FastAPI app
app = FastAPI(
    title="Discovery Service API",
    description="Genetic Algorithm Optimization for Trading Indicators",
    version="1.0.0"
)

# Global service instance
discovery_service = DiscoveryService()

# Pydantic models for request/response
class OptimizationRequest(BaseModel):
    """Request model for starting an optimization."""
    symbol: str = Field(..., description="Trading symbol (e.g., 'BTC/USDT')")
    timeframe: str = Field(..., description="Timeframe (e.g., '1d', '1h')")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    population_size: int = Field(50, description="GA population size", ge=10, le=200)
    generations: int = Field(20, description="Number of GA generations", ge=5, le=100)
    use_walk_forward: bool = Field(True, description="Use walk-forward analysis")

class OptimizationResponse(BaseModel):
    """Response model for optimization results."""
    success: bool
    optimization_id: str
    best_fitness: Optional[float] = None
    best_config: Optional[Dict[str, Any]] = None
    total_evaluations: Optional[int] = None
    duration_seconds: Optional[float] = None
    top_configurations: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None

class StatusResponse(BaseModel):
    """Response model for optimization status."""
    optimization_id: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    population_size: int
    generations: int
    best_fitness: float
    best_config: Dict[str, Any]
    total_evaluations: int
    duration_seconds: float
    status: str
    error_message: Optional[str] = None
    top_configurations: Optional[List[Dict[str, Any]]] = None
    created_at: Optional[str] = None

class ActiveOptimizationsResponse(BaseModel):
    """Response model for active optimizations list."""
    optimizations: List[Dict[str, Any]]
    count: int

class OptimizeBotRequest(BaseModel):
    """Request model for optimizing bot configuration."""
    symbol: str = Field(..., description="Trading symbol (e.g., 'BTC/USDT')")
    timeframe: str = Field(..., description="Timeframe (e.g., '1d', '1h')")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    population_size: Optional[int] = Field(100, description="GA population size", ge=10, le=200)
    generations: Optional[int] = Field(30, description="Number of GA generations", ge=5, le=100)
    use_walk_forward: Optional[bool] = Field(True, description="Use walk-forward analysis")

class OptimizeBotResponse(BaseModel):
    """Response model for bot optimization results."""
    success: bool
    optimization_id: str
    best_fitness: Optional[float] = None
    best_config: Optional[Dict[str, Any]] = None
    total_evaluations: Optional[int] = None
    duration_seconds: Optional[float] = None
    top_configurations: Optional[List[Dict[str, Any]]] = None
    message: Optional[str] = None
    error: Optional[str] = None

# Background task storage
active_tasks = {}

@app.post("/api/v1/optimization/start", response_model=OptimizationResponse)
async def start_optimization(request: OptimizationRequest, background_tasks: BackgroundTasks):
    """
    Start a new optimization run.

    This endpoint initiates a genetic algorithm optimization for trading indicator
    configurations. The optimization runs asynchronously in the background.
    """
    try:
        # Generate unique optimization ID
        optimization_id = f"opt_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp())}"

        # Start optimization in background
        background_tasks.add_task(
            run_optimization_background,
            optimization_id,
            request.symbol,
            request.timeframe,
            request.start_date,
            request.end_date,
            request.population_size,
            request.generations,
            request.use_walk_forward
        )

        return OptimizationResponse(
            success=True,
            optimization_id=optimization_id,
            error=None
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start optimization: {str(e)}")

async def run_optimization_background(optimization_id: str, symbol: str, timeframe: str,
                                   start_date: str, end_date: str, population_size: int,
                                   generations: int, use_walk_forward: bool):
    """Background task to run optimization."""
    try:
        # Store task reference
        task = asyncio.current_task()
        active_tasks[optimization_id] = task

        # Run optimization
        result = await discovery_service.run_optimization_async(
            optimization_id=optimization_id,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            population_size=population_size,
            generations=generations,
            use_walk_forward=use_walk_forward
        )

        # Clean up
        if optimization_id in active_tasks:
            del active_tasks[optimization_id]

    except Exception as e:
        print(f"Background optimization failed: {e}")
        if optimization_id in active_tasks:
            del active_tasks[optimization_id]

@app.get("/api/v1/optimization/{optimization_id}/status", response_model=StatusResponse)
async def get_optimization_status(optimization_id: str):
    """Get the status of a specific optimization run."""
    try:
        status = await discovery_service.get_optimization_status(optimization_id)

        if "error" in status:
            raise HTTPException(status_code=404, detail=status["error"])

        return StatusResponse(**status)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@app.get("/api/v1/optimization/active", response_model=ActiveOptimizationsResponse)
async def list_active_optimizations():
    """List all currently active optimization runs."""
    try:
        active = await discovery_service.list_active_optimizations()
        return ActiveOptimizationsResponse(
            optimizations=active,
            count=len(active)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list active optimizations: {str(e)}")

@app.delete("/api/v1/optimization/{optimization_id}")
async def cancel_optimization(optimization_id: str):
    """Cancel an active optimization run."""
    try:
        result = await discovery_service.cancel_optimization(optimization_id)

        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])

        return {"success": True, "message": "Optimization cancelled", "optimization_id": optimization_id}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel optimization: {str(e)}")

@app.post("/api/v1/bot-optimization/start", response_model=OptimizeBotResponse)
async def start_bot_optimization(request: OptimizeBotRequest, background_tasks: BackgroundTasks):
    """
    Start a new bot optimization run.

    This endpoint initiates a genetic algorithm optimization for trading bot configurations,
    including technical indicators and risk parameters. The optimization runs asynchronously
    in the background.
    """
    try:
        # Validate request
        if not request.symbol or not request.timeframe:
            raise HTTPException(status_code=400, detail="Symbol and timeframe are required")

        # Generate optimization ID
        optimization_id = f"bot_opt_{request.symbol}_{request.timeframe}_{int(datetime.now().timestamp())}"

        # Start optimization in background
        background_tasks.add_task(
            run_bot_optimization_background,
            optimization_id,
            request.symbol,
            request.timeframe,
            request.start_date,
            request.end_date,
            request.population_size,
            request.generations,
            request.use_walk_forward
        )

        return OptimizeBotResponse(
            success=True,
            optimization_id=optimization_id,
            error=None
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start bot optimization: {str(e)}")

async def run_bot_optimization_background(optimization_id: str, symbol: str, timeframe: str,
                                        start_date: str, end_date: str, population_size: int,
                                        generations: int, use_walk_forward: bool):
    """Background task to run bot optimization."""
    try:
        # Store task reference
        task = asyncio.current_task()
        active_tasks[optimization_id] = task

        # Run optimization
        result = await discovery_service.optimize_bot_configuration(
            optimization_id=optimization_id,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            population_size=population_size,
            generations=generations,
            use_walk_forward=use_walk_forward
        )

        # Clean up
        if optimization_id in active_tasks:
            del active_tasks[optimization_id]

    except Exception as e:
        print(f"Background bot optimization failed: {e}")
        if optimization_id in active_tasks:
            del active_tasks[optimization_id]

@app.get("/api/v1/bot-optimization/{optimization_id}/status", response_model=StatusResponse)
async def get_bot_optimization_status(optimization_id: str):
    """Get the status of a specific bot optimization run."""
    try:
        status = await discovery_service.get_bot_optimization_status(optimization_id)

        if "error" in status:
            raise HTTPException(status_code=404, detail=status["error"])

        return StatusResponse(**status)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get bot optimization status: {str(e)}")

@app.get("/api/v1/bot-optimization/{optimization_id}/results")
async def get_bot_optimization_results(optimization_id: str):
    """Get detailed results of a completed bot optimization."""
    try:
        status = await discovery_service.get_bot_optimization_status(optimization_id)

        if "error" in status:
            raise HTTPException(status_code=404, detail=status["error"])

        if status["status"] != "completed":
            raise HTTPException(status_code=400, detail="Bot optimization not completed yet")

        # Return detailed results
        return {
            "optimization_id": optimization_id,
            "status": "completed",
            "results": {
                "best_fitness": status["best_fitness"],
                "best_config": status["best_config"],
                "total_evaluations": status["total_evaluations"],
                "duration_seconds": status["duration_seconds"],
                "top_configurations": status.get("top_configurations", [])
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get bot optimization results: {str(e)}")

# Event handlers for service integration
async def handle_optimization_started(event):
    """Handle optimization started event."""
    print(f"🎯 Optimization started: {event.data['optimization_id']}")

async def handle_optimization_completed(event):
    """Handle optimization completed event."""
    data = event.data
    print(f"✅ Optimization completed: {data['optimization_id']} "
          f"(fitness: {data['best_fitness']:.2f}, duration: {data['duration']:.1f}s)")

async def handle_optimization_failed(event):
    """Handle optimization failed event."""
    data = event.data
    print(f"❌ Optimization failed: {data['optimization_id']} - {data['error']}")

def setup_event_handlers():
    """Setup event handlers for the discovery service."""
    event_system = discovery_service.event_system
    event_system.subscribe("discovery.optimization.started", handle_optimization_started)
    event_system.subscribe("discovery.optimization.completed", handle_optimization_completed)
    event_system.subscribe("discovery.optimization.failed", handle_optimization_failed)

if __name__ == "__main__":
    # Setup event handlers
    setup_event_handlers()

    # Start server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )