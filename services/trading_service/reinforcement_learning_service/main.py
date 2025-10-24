"""
TradPal Reinforcement Learning Service API
FastAPI application for reinforcement learning-based trading strategies
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import asyncio

from service import ReinforcementLearningService

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="TradPal Reinforcement Learning Service",
    description="Specialized service for reinforcement learning-based trading strategies",
    version="1.0.0"
)

# Global service instance - initialized on startup
rl_service = None


# Pydantic models for API requests/responses
class RLConfig(BaseModel):
    algorithm: str = Field(default="ppo", description="RL algorithm (ppo, dqn, a2c)")
    total_timesteps: int = Field(default=100000, description="Total training timesteps")
    learning_rate: float = Field(default=3e-4, description="Learning rate")
    gamma: float = Field(default=0.99, description="Discount factor")
    policy: str = Field(default="MlpPolicy", description="Policy network type")

class TrainingRequest(BaseModel):
    symbol: str = Field(..., description="Trading symbol")
    config: RLConfig = Field(default_factory=RLConfig, description="RL training configuration")

class TrainingResponse(BaseModel):
    success: bool
    agent_id: Optional[str] = None
    status: str = "initialized"
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class ActionRequest(BaseModel):
    symbol: str = Field(..., description="Trading symbol")
    state: List[float] = Field(..., description="Current market state")
    agent_id: Optional[str] = None

class ActionResponse(BaseModel):
    action: int
    confidence: float
    agent_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    global rl_service
    logger.info("Starting Reinforcement Learning Service...")
    rl_service = ReinforcementLearningService()
    await rl_service.initialize()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Reinforcement Learning Service...")
    if rl_service:
        await rl_service.shutdown()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "reinforcement_learning_service"}


@app.post("/train", response_model=TrainingResponse)
async def train_agent(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Train a reinforcement learning agent"""
    try:
        logger.info(f"Starting RL training for symbol: {request.symbol}")

        # Start training in background
        background_tasks.add_task(
            rl_service.train_agent,
            request.symbol,
            request.config.dict()
        )

        agent_id = f"rl_{request.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return TrainingResponse(
            success=True,
            agent_id=agent_id,
            status="training_started"
        )

    except Exception as e:
        logger.error(f"RL training failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/action", response_model=ActionResponse)
async def get_action(request: ActionRequest):
    """Get trading action from RL agent"""
    try:
        action, confidence = await rl_service.get_action(
            request.symbol,
            request.state,
            request.agent_id
        )

        return ActionResponse(
            action=action,
            confidence=confidence,
            agent_id=request.agent_id or "default"
        )

    except Exception as e:
        logger.error(f"Failed to get action: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents")
async def list_agents():
    """List available trained agents"""
    try:
        agents = rl_service.list_agents()
        return {"agents": agents}
    except Exception as e:
        logger.error(f"Failed to list agents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agent/{agent_id}")
async def get_agent_info(agent_id: str):
    """Get information about a specific agent"""
    try:
        info = rl_service.get_agent_info(agent_id)
        return info
    except Exception as e:
        logger.error(f"Failed to get agent info: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")


@app.get("/status/{agent_id}")
async def get_training_status(agent_id: str):
    """Get training status of an agent"""
    try:
        status = rl_service.get_training_status(agent_id)
        return status
    except Exception as e:
        logger.error(f"Failed to get training status: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")


@app.delete("/agent/{agent_id}")
async def delete_agent(agent_id: str):
    """Delete a trained agent"""
    try:
        success = rl_service.delete_agent(agent_id)
        return {"success": success}
    except Exception as e:
        logger.error(f"Failed to delete agent: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8013,  # Different port for RL service
        reload=True
    )