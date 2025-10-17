"""
Reinforcement Learning Service
AI-powered trading decisions using reinforcement learning algorithms.
"""

from .rl_agent import (
    RLAlgorithm,
    TradingAction,
    TradingState,
    RLExperience,
    QLearningAgent,
    RewardFunction,
    TradingEnvironment
)

from .training_engine import (
    TrainingConfig,
    TrainingMetrics,
    RLTrainer,
    AsyncRLTrainer
)

from .client import ReinforcementLearningServiceClient

__version__ = "1.0.0"
__all__ = [
    "RLAlgorithm",
    "TradingAction",
    "TradingState",
    "RLExperience",
    "QLearningAgent",
    "RewardFunction",
    "TradingEnvironment",
    "TrainingConfig",
    "TrainingMetrics",
    "RLTrainer",
    "AsyncRLTrainer",
    "ReinforcementLearningServiceClient"
]