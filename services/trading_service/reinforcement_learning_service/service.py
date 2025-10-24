"""
TradPal Trading Service - Reinforcement Learning
Advanced RL implementation with ensemble integration and market regime awareness
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pickle
import os

from rl_agent import (
    RLAlgorithm, TradingAction, TradingState, RLExperience,
    RewardFunction, QLearningAgent, TradingEnvironment
)

logger = logging.getLogger(__name__)


class ReinforcementLearningService:
    """
    Advanced reinforcement learning service for trading decisions.

    Features:
    - Multiple RL algorithms (Q-Learning, DQN, PPO)
    - Integration with ensemble ML predictions
    - Market regime-aware decision making
    - Risk-adjusted reward functions
    - Walk-forward validation for robustness
    """

    def __init__(self, event_system=None, config: Optional[Dict[str, Any]] = None):
        self.event_system = event_system
        self.config = config or self._default_config()
        self.is_initialized = False

        # RL Components
        self.agents: Dict[str, Any] = {}
        self.environments: Dict[str, TradingEnvironment] = {}
        self.reward_functions: Dict[str, RewardFunction] = {}

        # Training state
        self.training_active = False
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Model storage
        self.model_dir = self.config.get('model_dir', '/tmp/rl_models')
        os.makedirs(self.model_dir, exist_ok=True)

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for RL service."""
        return {
            'model_dir': 'cache/rl_models',
            'training_episodes': 1000,
            'validation_episodes': 100,
            'learning_rate': 0.001,
            'discount_factor': 0.95,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'batch_size': 64,
            'experience_buffer_size': 50000,
            'target_update_freq': 100,
            'save_freq': 100,
            'algorithms': ['q_learning', 'dqn'],
            'reward_weights': {
                'pnl_weight': 0.4,
                'sharpe_weight': 0.2,
                'risk_penalty': 0.2,
                'regime_bonus': 0.1,
                'transaction_penalty': 0.1
            }
        }

    async def initialize(self):
        """Initialize the reinforcement learning service."""
        logger.info("Initializing Advanced Reinforcement Learning Service...")

        try:
            # Initialize reward functions
            self.reward_functions['default'] = RewardFunction(self.config['reward_weights'])
            self.reward_functions['conservative'] = RewardFunction({
                **self.config['reward_weights'],
                'pnl_weight': 0.3,
                'risk_penalty': 0.4,
                'transaction_penalty': 0.2
            })
            self.reward_functions['aggressive'] = RewardFunction({
                **self.config['reward_weights'],
                'pnl_weight': 0.6,
                'risk_penalty': 0.1,
                'transaction_penalty': 0.05
            })

            # Initialize agents for different algorithms
            for algorithm in self.config['algorithms']:
                await self._initialize_agent(algorithm)

            self.is_initialized = True
            logger.info("Reinforcement Learning Service initialized successfully")

            # Publish initialization event
            if self.event_system:
                await self.event_system.publish("rl.service_initialized", {
                    "algorithms": list(self.agents.keys()),
                    "reward_functions": list(self.reward_functions.keys()),
                    "timestamp": datetime.now().isoformat()
                })

        except Exception as e:
            logger.error(f"Failed to initialize RL service: {e}")
            raise

    async def _initialize_agent(self, algorithm: str):
        """Initialize RL agent for specific algorithm."""
        if algorithm == 'q_learning':
            # Q-Learning agent configuration
            state_bins = {'position': 5, 'price': 7, 'trend': 5}
            actions = [TradingAction.BUY, TradingAction.SELL, TradingAction.HOLD,
                      TradingAction.REDUCE, TradingAction.INCREASE]

            agent_config = {
                'learning_rate': self.config['learning_rate'],
                'discount_factor': self.config['discount_factor'],
                'epsilon': self.config['epsilon_start'],
                'epsilon_decay': self.config['epsilon_decay'],
                'min_epsilon': self.config['epsilon_end'],
                'experience_buffer_size': self.config['experience_buffer_size']
            }

            self.agents[algorithm] = QLearningAgent(state_bins, actions, agent_config)

        elif algorithm == 'dqn':
            # Deep Q-Network agent (placeholder for future implementation)
            logger.info("DQN agent initialization placeholder - not yet implemented")
            self.agents[algorithm] = None

        else:
            logger.warning(f"Unknown RL algorithm: {algorithm}")

    async def shutdown(self):
        """Shutdown the reinforcement learning service."""
        logger.info("Shutting down Reinforcement Learning Service...")

        # Save all trained models
        await self._save_all_models()

        # Shutdown executor
        self.executor.shutdown(wait=True)

        self.is_initialized = False
        logger.info("Reinforcement Learning Service shut down")

    async def _save_all_models(self):
        """Save all trained RL models."""
        for algorithm, agent in self.agents.items():
            if agent and hasattr(agent, 'save_model'):
                model_path = os.path.join(self.model_dir, f"{algorithm}_model.pkl")
                try:
                    agent.save_model(model_path)
                    logger.info(f"Saved {algorithm} model to {model_path}")
                except Exception as e:
                    logger.error(f"Failed to save {algorithm} model: {e}")

    async def train_rl_agent(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train reinforcement learning agent with market data.

        Args:
            training_data: Dictionary containing:
                - symbol: Trading symbol
                - market_data: DataFrame with OHLCV and indicators
                - episodes: Number of training episodes
                - algorithm: RL algorithm to use
                - reward_function: Reward function type

        Returns:
            Training results and metrics
        """
        if not self.is_initialized:
            raise RuntimeError("RL service not initialized")

        symbol = training_data.get('symbol', 'BTCUSDT')
        market_data = training_data.get('market_data')
        episodes = training_data.get('episodes', self.config['training_episodes'])
        algorithm = training_data.get('algorithm', 'q_learning')
        reward_type = training_data.get('reward_function', 'default')

        if not isinstance(market_data, pd.DataFrame):
            raise ValueError("market_data must be a pandas DataFrame")

        logger.info(f"Training RL agent for {symbol} using {algorithm} algorithm")

        # Initialize training environment
        env = TradingEnvironment(
            market_data=market_data,
            symbol=symbol,
            initial_balance=training_data.get('initial_balance', 10000.0),
            transaction_cost=training_data.get('transaction_cost', 0.001)
        )

        agent = self.agents.get(algorithm)
        reward_function = self.reward_functions.get(reward_type, self.reward_functions['default'])

        if not agent:
            raise ValueError(f"RL algorithm {algorithm} not available")

        # Training loop
        self.training_active = True
        episode_rewards = []
        episode_metrics = []

        try:
            for episode in range(episodes):
                if not self.training_active:
                    break

                episode_reward = 0
                state = env.reset()
                done = False
                step_count = 0

                while not done and step_count < len(market_data) - 1:
                    # Select action
                    action = agent.get_action(state, training=True)

                    # Execute action
                    next_state, reward, done, info = env.step(action)

                    # Store experience
                    experience = RLExperience(
                        state=state,
                        action=action,
                        reward=reward,
                        next_state=next_state,
                        done=done,
                        timestamp=datetime.now()
                    )

                    # Update agent
                    agent.update_q_value(experience)

                    episode_reward += reward
                    state = next_state
                    step_count += 1

                episode_rewards.append(episode_reward)

                # Log progress
                if (episode + 1) % 100 == 0:
                    avg_reward = np.mean(episode_rewards[-100:])
                    logger.info(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.4f}")

                    # Publish training progress
                    if self.event_system:
                        await self.event_system.publish("rl.training_progress", {
                            "episode": episode + 1,
                            "total_episodes": episodes,
                            "avg_reward": avg_reward,
                            "symbol": symbol,
                            "algorithm": algorithm
                        })

            # Calculate final performance metrics
            final_metrics = env.get_performance_metrics()
            training_metrics = agent.get_training_metrics()

            result = {
                "success": True,
                "symbol": symbol,
                "algorithm": algorithm,
                "episodes_trained": len(episode_rewards),
                "average_reward": float(np.mean(episode_rewards)),
                "best_reward": float(np.max(episode_rewards)),
                "final_performance": final_metrics,
                "training_metrics": training_metrics,
                "model_path": await self._save_trained_model(agent, algorithm, symbol),
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"RL training completed: {result['average_reward']:.4f} avg reward")

            # Publish training completion
            if self.event_system:
                await self.event_system.publish("rl.training_completed", result)

            return result

        except Exception as e:
            logger.error(f"RL training failed: {e}")
            raise
        finally:
            self.training_active = False

    async def _save_trained_model(self, agent: Any, algorithm: str, symbol: str) -> str:
        """Save trained RL model."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.model_dir, f"{algorithm}_{symbol}_{timestamp}.pkl")

        try:
            agent.save_model(model_path)
            return model_path
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return ""

    async def get_rl_signal(self, symbol: str, market_data: Dict[str, Any],
                           algorithm: str = 'q_learning') -> Dict[str, Any]:
        """
        Get trading signal from trained RL agent.

        Args:
            symbol: Trading symbol
            market_data: Current market data and indicators
            algorithm: RL algorithm to use

        Returns:
            Trading signal with confidence
        """
        if not self.is_initialized:
            raise RuntimeError("RL service not initialized")

        agent = self.agents.get(algorithm)
        if not agent:
            raise ValueError(f"RL algorithm {algorithm} not available")

        try:
            # Create current trading state
            state = self._create_trading_state(symbol, market_data)

            # Get action from agent
            action = agent.get_action(state, training=False)

            # Calculate confidence based on Q-values
            if hasattr(agent, 'q_table'):
                discretized_state = state.discretize(agent.state_bins)
                state_idx = agent._state_to_index(discretized_state)
                q_values = agent.q_table[state_idx]
                confidence = float(np.max(q_values) - np.min(q_values))  # Value spread as confidence
            else:
                confidence = 0.5  # Default confidence

            # Normalize confidence to 0-1 range
            confidence = min(max(confidence, 0.0), 1.0)

            signal = {
                "signal": action.value,
                "confidence": confidence,
                "algorithm": algorithm,
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "state_features": state.to_vector().tolist()
            }

            # Publish signal event
            if self.event_system:
                await self.event_system.publish("rl.signal_generated", signal)

            return signal

        except Exception as e:
            logger.error(f"Failed to generate RL signal: {e}")
            return {
                "signal": "hold",
                "confidence": 0.0,
                "error": str(e),
                "algorithm": algorithm,
                "symbol": symbol,
                "timestamp": datetime.now().isoformat()
            }

    def _create_trading_state(self, symbol: str, market_data: Dict[str, Any]) -> TradingState:
        """Create TradingState from market data dictionary."""
        # Extract position and portfolio info (would come from portfolio service)
        position_size = market_data.get('position_size', 0.0)
        portfolio_value = market_data.get('portfolio_value', 10000.0)

        # Extract technical indicators
        technical_indicators = {}
        for indicator in ['rsi', 'macd', 'bb_position', 'stoch_k', 'cci', 'adx']:
            technical_indicators[indicator] = market_data.get(indicator, 0.0)

        # Market regime information
        market_regime = market_data.get('market_regime', 'sideways')
        volatility_regime = market_data.get('volatility_regime', 'normal')

        # Calculate trend strength
        trend_strength = market_data.get('trend_strength', 0.0)

        return TradingState(
            symbol=symbol,
            position_size=position_size,
            current_price=market_data.get('close', 0.0),
            portfolio_value=portfolio_value,
            market_regime=market_regime,
            volatility_regime=volatility_regime,
            trend_strength=trend_strength,
            technical_indicators=technical_indicators,
            timestamp=datetime.now()
        )

    async def validate_rl_model(self, validation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate RL model using walk-forward methodology.

        Args:
            validation_data: Validation configuration and data

        Returns:
            Validation results and robustness metrics
        """
        if not self.is_initialized:
            raise RuntimeError("RL service not initialized")

        symbol = validation_data.get('symbol', 'BTCUSDT')
        market_data = validation_data.get('market_data')
        algorithm = validation_data.get('algorithm', 'q_learning')
        windows = validation_data.get('validation_windows', 5)

        logger.info(f"Validating RL model for {symbol} using {algorithm}")

        # Create validation environments
        validation_results = []

        # Split data into validation windows
        window_size = len(market_data) // windows

        for i in range(windows):
            start_idx = i * window_size
            end_idx = (i + 1) * window_size if i < windows - 1 else len(market_data)

            window_data = market_data.iloc[start_idx:end_idx]

            # Create environment for this window
            env = TradingEnvironment(
                market_data=window_data,
                symbol=symbol,
                initial_balance=10000.0
            )

            # Evaluate agent on this window
            agent = self.agents.get(algorithm)
            if not agent:
                continue

            # Run evaluation episodes
            window_rewards = []
            for episode in range(self.config['validation_episodes']):
                state = env.reset()
                episode_reward = 0
                done = False

                while not done:
                    action = agent.get_action(state, training=False)
                    next_state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    state = next_state

                window_rewards.append(episode_reward)

            window_metrics = env.get_performance_metrics()
            window_metrics['average_reward'] = np.mean(window_rewards)
            window_metrics['reward_std'] = np.std(window_rewards)

            validation_results.append({
                'window': i + 1,
                'start_date': window_data.index[0].isoformat() if hasattr(window_data.index[0], 'isoformat') else str(window_data.index[0]),
                'end_date': window_data.index[-1].isoformat() if hasattr(window_data.index[-1], 'isoformat') else str(window_data.index[-1]),
                'metrics': window_metrics
            })

        # Calculate overall validation metrics
        all_rewards = [w['metrics']['average_reward'] for w in validation_results]
        all_returns = [w['metrics']['total_return'] for w in validation_results]

        validation_summary = {
            'success': True,
            'total_windows': len(validation_results),
            'average_reward': float(np.mean(all_rewards)),
            'reward_stability': float(np.std(all_rewards)),
            'average_return': float(np.mean(all_returns)),
            'return_stability': float(np.std(all_returns)),
            'consistency_score': float(np.mean([1 if r > 0 else 0 for r in all_returns])),  # Win rate
            'validation_results': validation_results
        }

        logger.info(f"RL validation completed: {validation_summary['consistency_score']:.2f} consistency score")

        # Publish validation results
        if self.event_system:
            await self.event_system.publish("rl.validation_completed", {
                "symbol": symbol,
                "algorithm": algorithm,
                **validation_summary
            })

        return validation_summary

    async def get_rl_model_status(self) -> Dict[str, Any]:
        """Get status of all RL models and training state."""
        status = {
            "service_initialized": self.is_initialized,
            "training_active": self.training_active,
            "available_algorithms": list(self.agents.keys()),
            "available_reward_functions": list(self.reward_functions.keys()),
            "models": {}
        }

        for algorithm, agent in self.agents.items():
            if agent:
                status["models"][algorithm] = {
                    "type": type(agent).__name__,
                    "trained": hasattr(agent, 'episode_rewards') and len(agent.episode_rewards) > 0,
                    "training_episodes": len(agent.episode_rewards) if hasattr(agent, 'episode_rewards') else 0,
                    "current_epsilon": agent.config.get('epsilon', 0) if hasattr(agent, 'config') else 0
                }

        return status

    async def stop_training(self):
        """Stop active RL training."""
        logger.info("Stopping RL training...")
        self.training_active = False