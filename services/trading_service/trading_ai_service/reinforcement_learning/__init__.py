"""
Reinforcement Learning Module for TradPal Trading AI Service

This module implements advanced reinforcement learning algorithms for automated trading:
- Q-Learning agents for discrete action spaces
- Deep RL agents (DQN, PPO) for complex trading strategies
- Custom reward functions optimized for trading performance
- Integration with market regime detection and risk management
- Walk-forward validation for RL model robustness

Key Components:
- RLAgent: Base RL agent interface
- QLearningAgent: Tabular Q-learning for trading
- DeepRLAgent: Neural network-based RL agents
- TradingEnvironment: Gym-like environment for trading simulation
- RewardFunction: Sophisticated reward calculation for trading actions
"""

from .rl_agent import (
    RLAlgorithm,
    TradingAction,
    TradingState,
    RLExperience,
    RewardFunction,
    QLearningAgent,
    DeepRLAgent,
    TradingEnvironment
)
from .service import ReinforcementLearningService

__all__ = [
    'RLAlgorithm',
    'TradingAction',
    'TradingState',
    'RLExperience',
    'RewardFunction',
    'QLearningAgent',
    'DeepRLAgent',
    'TradingEnvironment',
    'ReinforcementLearningService'
]