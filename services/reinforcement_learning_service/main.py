"""
Reinforcement Learning Service - REST API
Provides RL-based trading decision making and model training.
"""
import asyncio
import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from .rl_agent import (
    RLAlgorithm, TradingAction, TradingState, QLearningAgent, RewardFunction
)
from .training_engine import TrainingConfig, AsyncRLTrainer

# Event system imports with fallback
try:
    from services.event_system import (
        publish_rl_model_update,
        publish_rl_action,
        EVENT_SYSTEM_AVAILABLE
    )
except ImportError:
    EVENT_SYSTEM_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Event system not available, running without event publishing")

    # Fallback functions
    async def publish_rl_model_update(*args, **kwargs):
        logger.debug("Event system not available - skipping RL model update event")

    async def publish_rl_action(*args, **kwargs):
        logger.debug("Event system not available - skipping RL action event")

logger = logging.getLogger(__name__)

# Global service instances
rl_agent = None
trainer = None
training_tasks: Dict[str, AsyncRLTrainer] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global rl_agent, trainer

    # Startup
    logger.info("Starting Reinforcement Learning Service")

    # Initialize default Q-learning agent
    actions = [TradingAction.BUY, TradingAction.SELL, TradingAction.HOLD,
              TradingAction.REDUCE, TradingAction.INCREASE]

    rl_agent = QLearningAgent(
        state_bins={'position': 5, 'price': 7, 'trend': 5},
        actions=actions
    )

    trainer = AsyncRLTrainer(TrainingConfig())

    yield

    # Shutdown
    logger.info("Shutting down Reinforcement Learning Service")


# FastAPI app
app = FastAPI(
    title="Reinforcement Learning Service",
    description="AI-powered trading decisions using reinforcement learning",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class TradingStateRequest(BaseModel):
    """Request model for trading state."""
    symbol: str = Field(..., description="Trading symbol")
    position_size: float = Field(..., description="Current position size (-1 to 1)")
    current_price: float = Field(..., description="Current market price")
    portfolio_value: float = Field(..., description="Current portfolio value")
    market_regime: str = Field("sideways", description="Current market regime")
    volatility_regime: str = Field("normal", description="Current volatility regime")
    trend_strength: float = Field(0.0, description="Trend strength (-1 to 1)")
    technical_indicators: Dict[str, float] = Field(default_factory=dict, description="Technical indicators")


class TrainingRequest(BaseModel):
    """Request model for training."""
    algorithm: str = Field("q_learning", description="RL algorithm to use")
    episodes: int = Field(1000, description="Number of training episodes")
    symbols: List[str] = Field(..., description="Symbols to train on")
    market_data: Dict[str, List[Dict[str, Any]]] = Field(..., description="Market data for training")
    config: Optional[Dict[str, Any]] = Field(None, description="Additional training configuration")


class ActionResponse(BaseModel):
    """Response model for action selection."""
    action: str
    confidence: float
    reasoning: str
    timestamp: datetime
    q_values: Optional[Dict[str, float]] = None


class TrainingStatusResponse(BaseModel):
    """Response model for training status."""
    status: str
    current_episode: int
    total_episodes: int
    progress: float
    metrics: Dict[str, Any]
    task_id: Optional[str] = None


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    algorithm: str
    state_bins: Dict[str, int]
    actions: List[str]
    training_metrics: Dict[str, Any]
    last_updated: Optional[datetime]
    model_size: int


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global rl_agent, trainer

    return {
        "status": "healthy",
        "service": "reinforcement_learning",
        "timestamp": datetime.now(),
        "event_system": EVENT_SYSTEM_AVAILABLE,
        "agent_loaded": rl_agent is not None,
        "trainer_ready": trainer is not None,
        "active_training_tasks": len(training_tasks)
    }


@app.post("/action", response_model=ActionResponse)
async def get_trading_action(request: TradingStateRequest):
    """
    Get trading action recommendation from RL agent.

    Analyzes current market state and returns recommended action.
    """
    try:
        global rl_agent

        if rl_agent is None:
            raise HTTPException(status_code=503, detail="RL agent not initialized")

        # Create trading state
        technical_indicators = request.technical_indicators.copy()
        # Ensure required indicators exist
        for indicator in ['rsi', 'macd', 'bb_position']:
            if indicator not in technical_indicators:
                technical_indicators[indicator] = 0.0

        state = TradingState(
            symbol=request.symbol,
            position_size=max(-1.0, min(1.0, request.position_size)),  # Clamp to valid range
            current_price=request.current_price,
            portfolio_value=request.portfolio_value,
            market_regime=request.market_regime,
            volatility_regime=request.volatility_regime,
            trend_strength=max(-1.0, min(1.0, request.trend_strength)),  # Clamp to valid range
            technical_indicators=technical_indicators,
            timestamp=datetime.now()
        )

        # Get action from agent
        action = rl_agent.get_action(state, training=False)

        # Calculate confidence (based on Q-value difference)
        discretized_state = state.discretize(rl_agent.state_bins)
        state_idx = rl_agent._state_to_index(discretized_state)
        q_values = rl_agent.q_table[state_idx]

        max_q = np.max(q_values)
        second_max_q = np.partition(q_values, -2)[-2] if len(q_values) > 1 else max_q
        confidence = (max_q - second_max_q) / (abs(max_q) + 1e-8)  # Normalized difference
        confidence = min(confidence, 1.0)  # Cap at 1.0

        # Generate reasoning
        reasoning = generate_action_reasoning(action, state, confidence)

        # Publish event
        if EVENT_SYSTEM_AVAILABLE:
            try:
                await publish_rl_action({
                    "symbol": request.symbol,
                    "action": action.value,
                    "confidence": confidence,
                    "state": {
                        "position_size": state.position_size,
                        "market_regime": state.market_regime,
                        "trend_strength": state.trend_strength
                    },
                    "q_values": {a.value: float(q) for a, q in zip(rl_agent.actions, q_values)},
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Failed to publish RL action event: {e}")

        # Prepare Q-values for response
        q_values_dict = {action.value: float(q_val) for action, q_val in zip(rl_agent.actions, q_values)}

        return ActionResponse(
            action=action.value,
            confidence=float(confidence),
            reasoning=reasoning,
            timestamp=datetime.now(),
            q_values=q_values_dict
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Action selection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def generate_action_reasoning(action: TradingAction, state: TradingState, confidence: float) -> str:
    """Generate human-readable reasoning for the selected action."""
    reasons = []

    # Position-based reasoning
    if abs(state.position_size) < 0.1:
        reasons.append("currently flat")
    elif state.position_size > 0.5:
        reasons.append("heavily long")
    elif state.position_size < -0.5:
        reasons.append("heavily short")
    elif state.position_size > 0:
        reasons.append("moderately long")
    elif state.position_size < 0:
        reasons.append("moderately short")

    # Regime-based reasoning
    if state.market_regime == "bull_market":
        if action == TradingAction.BUY:
            reasons.append("bull market favors buying")
        elif action == TradingAction.SELL:
            reasons.append("despite bull market, reducing exposure")
    elif state.market_regime == "bear_market":
        if action == TradingAction.SELL:
            reasons.append("bear market favors selling")
        elif action == TradingAction.BUY:
            reasons.append("despite bear market, seeing opportunity")

    # Trend-based reasoning
    if abs(state.trend_strength) > 0.5:
        trend_direction = "upward" if state.trend_strength > 0 else "downward"
        reasons.append(f"strong {trend_direction} trend detected")

    # Confidence reasoning
    if confidence > 0.8:
        reasons.append("high confidence in decision")
    elif confidence > 0.6:
        reasons.append("moderate confidence in decision")
    else:
        reasons.append("low confidence, consider holding")

    if not reasons:
        reasons.append("based on learned patterns")

    return f"Action {action.value} selected because {', '.join(reasons)}."


@app.post("/train", response_model=TrainingStatusResponse)
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Start RL model training.

    Trains the agent on provided market data and returns training status.
    """
    try:
        global trainer

        if trainer is None:
            raise HTTPException(status_code=503, detail="Trainer not initialized")

        # Convert data to DataFrames
        market_data = {}
        for symbol, data_list in request.market_data.items():
            try:
                df = pd.DataFrame(data_list)
                # Ensure required columns exist
                required_cols = ['close']
                if not all(col in df.columns for col in required_cols):
                    raise ValueError(f"Missing required columns for {symbol}")
                market_data[symbol] = df
            except Exception as e:
                logger.error(f"Failed to process data for {symbol}: {e}")
                continue

        if not market_data:
            raise HTTPException(status_code=400, detail="No valid market data provided")

        # Create training config
        config_dict = request.config or {}
        config_dict.update({
            'algorithm': RLAlgorithm(request.algorithm),
            'episodes': request.episodes,
            'symbols': request.symbols
        })

        training_config = TrainingConfig(**config_dict)
        trainer.config = training_config

        # Generate task ID
        task_id = str(uuid.uuid4())
        training_tasks[task_id] = trainer

        # Start training
        result = await trainer.train_async(market_data, progress_callback=None)

        if result['status'] == 'training_started':
            return TrainingStatusResponse(
                status="training_started",
                current_episode=0,
                total_episodes=request.episodes,
                progress=0.0,
                metrics={},
                task_id=task_id
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to start training")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Training request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/training/status/{task_id}", response_model=TrainingStatusResponse)
async def get_training_status(task_id: str):
    """Get training status for a specific task."""
    try:
        if task_id not in training_tasks:
            raise HTTPException(status_code=404, detail="Training task not found")

        trainer = training_tasks[task_id]
        status = trainer.get_training_status()

        # Check if training completed
        result = await trainer.get_training_result()
        if result:
            # Training completed
            del training_tasks[task_id]

            if result['status'] == 'completed':
                # Update global agent with trained model
                global rl_agent
                rl_agent = trainer.agent

                # Publish model update event
                if EVENT_SYSTEM_AVAILABLE:
                    try:
                        await publish_rl_model_update({
                            "algorithm": trainer.config.algorithm.value,
                            "episodes_trained": result['episodes_completed'],
                            "final_metrics": result['final_metrics'],
                            "training_summary": result['training_summary'],
                            "timestamp": datetime.now().isoformat()
                        })
                    except Exception as e:
                        logger.error(f"Failed to publish model update event: {e}")

            return TrainingStatusResponse(
                status=result['status'],
                current_episode=result.get('episodes_completed', 0),
                total_episodes=trainer.config.episodes,
                progress=1.0 if result['status'] == 'completed' else status['progress'],
                metrics=result.get('training_summary', {}),
                task_id=task_id
            )

        return TrainingStatusResponse(
            status="training" if status['is_training'] else "stopped",
            current_episode=status['current_episode'],
            total_episodes=status['total_episodes'],
            progress=status['progress'],
            metrics=status['metrics'],
            task_id=task_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/training/{task_id}")
async def stop_training(task_id: str):
    """Stop a training task."""
    try:
        if task_id not in training_tasks:
            raise HTTPException(status_code=404, detail="Training task not found")

        trainer = training_tasks[task_id]
        trainer.stop_training_async()

        return {"status": "stop_requested", "task_id": task_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the current RL model."""
    try:
        global rl_agent

        if rl_agent is None:
            raise HTTPException(status_code=503, detail="RL agent not loaded")

        metrics = rl_agent.get_training_metrics()

        return ModelInfoResponse(
            algorithm=rl_agent.config.get('algorithm', 'q_learning'),
            state_bins=rl_agent.state_bins,
            actions=[action.value for action in rl_agent.actions],
            training_metrics=metrics,
            last_updated=None,  # Would need to track this
            model_size=rl_agent.q_table.size
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/save")
async def save_model(filepath: str = "models/rl_model.pkl"):
    """Save the current RL model."""
    try:
        global rl_agent

        if rl_agent is None:
            raise HTTPException(status_code=503, detail="RL agent not loaded")

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        rl_agent.save_model(filepath)

        return {"status": "saved", "filepath": filepath}

    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/load")
async def load_model(filepath: str = "models/rl_model.pkl"):
    """Load an RL model."""
    try:
        global rl_agent

        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="Model file not found")

        if rl_agent is None:
            # Initialize agent first
            actions = [TradingAction.BUY, TradingAction.SELL, TradingAction.HOLD,
                      TradingAction.REDUCE, TradingAction.INCREASE]
            rl_agent = QLearningAgent(
                state_bins={'position': 5, 'price': 7, 'trend': 5},
                actions=actions
            )

        rl_agent.load_model(filepath)

        return {"status": "loaded", "filepath": filepath}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/algorithms")
async def list_algorithms():
    """List available RL algorithms."""
    return {
        "algorithms": [algo.value for algo in RLAlgorithm],
        "current_implemented": ["q_learning"],
        "descriptions": {
            "q_learning": "Q-Learning with discretized states",
            "sarsa": "SARSA algorithm (not yet implemented)",
            "dqn": "Deep Q-Network (not yet implemented)",
            "ppo": "Proximal Policy Optimization (not yet implemented)"
        }
    }


@app.get("/actions")
async def list_actions():
    """List available trading actions."""
    return {
        "actions": [action.value for action in TradingAction],
        "descriptions": {
            "buy": "Increase long position",
            "sell": "Increase short position or close long",
            "hold": "Maintain current position",
            "reduce": "Reduce position size",
            "increase": "Increase position size"
        }
    }


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Start server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8003,
        reload=True,
        log_level="info"
    )