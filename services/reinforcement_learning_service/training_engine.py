"""
Reinforcement Learning Training Engine
Handles training of RL agents for trading strategies.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os

from .rl_agent import (
    RLAlgorithm, TradingAction, TradingState, RLExperience,
    QLearningAgent, RewardFunction, TradingEnvironment
)

logger = logging.getLogger(__name__)


class TrainingConfig:
    """Configuration for RL training."""

    def __init__(self, **kwargs):
        self.algorithm = kwargs.get('algorithm', RLAlgorithm.Q_LEARNING)
        self.episodes = kwargs.get('episodes', 1000)
        self.max_steps_per_episode = kwargs.get('max_steps_per_episode', 1000)
        self.learning_rate = kwargs.get('learning_rate', 0.1)
        self.discount_factor = kwargs.get('discount_factor', 0.95)
        self.epsilon_start = kwargs.get('epsilon_start', 1.0)
        self.epsilon_end = kwargs.get('epsilon_end', 0.01)
        self.epsilon_decay = kwargs.get('epsilon_decay', 0.995)
        self.batch_size = kwargs.get('batch_size', 32)
        self.target_update_freq = kwargs.get('target_update_freq', 100)
        self.save_freq = kwargs.get('save_freq', 100)
        self.eval_freq = kwargs.get('eval_freq', 50)
        self.early_stopping_patience = kwargs.get('early_stopping_patience', 50)

        # State discretization for Q-learning
        self.state_bins = kwargs.get('state_bins', {
            'position': 5,   # -1, -0.5, 0, 0.5, 1
            'price': 7,      # Price levels
            'trend': 5       # Trend strength levels
        })

        # Training data
        self.symbols = kwargs.get('symbols', ['BTC/USDT'])
        self.start_date = kwargs.get('start_date', datetime.now() - timedelta(days=365))
        self.end_date = kwargs.get('end_date', datetime.now())
        self.timeframe = kwargs.get('timeframe', '1h')


class TrainingMetrics:
    """Tracks training performance metrics."""

    def __init__(self):
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.portfolio_values: List[float] = []
        self.win_rates: List[float] = []
        self.sharpe_ratios: List[float] = []
        self.max_drawdowns: List[float] = []
        self.epsilon_values: List[float] = []
        self.loss_values: List[float] = []

    def update_episode(self, episode: int, reward: float, length: int,
                      portfolio_value: float, epsilon: float):
        """Update metrics for an episode."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.portfolio_values.append(portfolio_value)
        self.epsilon_values.append(epsilon)

    def update_evaluation(self, win_rate: float, sharpe_ratio: float, max_drawdown: float):
        """Update evaluation metrics."""
        self.win_rates.append(win_rate)
        self.sharpe_ratios.append(sharpe_ratio)
        self.max_drawdowns.append(max_drawdown)

    def get_summary(self) -> Dict[str, Any]:
        """Get training summary statistics."""
        return {
            'total_episodes': len(self.episode_rewards),
            'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'avg_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'best_reward': max(self.episode_rewards) if self.episode_rewards else 0,
            'final_portfolio_value': self.portfolio_values[-1] if self.portfolio_values else 0,
            'avg_win_rate': np.mean(self.win_rates) if self.win_rates else 0,
            'avg_sharpe_ratio': np.mean(self.sharpe_ratios) if self.sharpe_ratios else 0,
            'min_max_drawdown': min(self.max_drawdowns) if self.max_drawdowns else 0,
            'final_epsilon': self.epsilon_values[-1] if self.epsilon_values else 1.0
        }


class RLTrainer:
    """Reinforcement Learning Trainer for trading agents."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.metrics = TrainingMetrics()
        self.agent = None
        self.reward_function = RewardFunction()
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Training state
        self.is_training = False
        self.current_episode = 0
        self.best_reward = float('-inf')

        logger.info(f"Initialized RL Trainer with algorithm: {config.algorithm.value}")

    def initialize_agent(self):
        """Initialize the RL agent based on configuration."""
        actions = [
            TradingAction.BUY,
            TradingAction.SELL,
            TradingAction.HOLD,
            TradingAction.REDUCE,
            TradingAction.INCREASE
        ]

        if self.config.algorithm == RLAlgorithm.Q_LEARNING:
            self.agent = QLearningAgent(
                state_bins=self.config.state_bins,
                actions=actions,
                config={
                    'learning_rate': self.config.learning_rate,
                    'discount_factor': self.config.discount_factor,
                    'epsilon': self.config.epsilon_start,
                    'epsilon_decay': self.config.epsilon_decay,
                    'min_epsilon': self.config.epsilon_end,
                    'experience_buffer_size': 10000,
                    'batch_size': self.config.batch_size
                }
            )
        else:
            raise ValueError(f"Algorithm {self.config.algorithm} not yet implemented")

    async def train(self, market_data: Dict[str, pd.DataFrame],
                   progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Train the RL agent on market data.

        Args:
            market_data: Dictionary of symbol -> DataFrame
            progress_callback: Optional callback for training progress

        Returns:
            Training results and metrics
        """
        if self.agent is None:
            self.initialize_agent()

        self.is_training = True
        self.current_episode = 0

        logger.info(f"Starting training for {self.config.episodes} episodes")

        try:
            for episode in range(self.config.episodes):
                if not self.is_training:
                    break

                self.current_episode = episode

                # Train on each symbol
                episode_reward = 0
                episode_length = 0

                for symbol, data in market_data.items():
                    if len(data) < 50:  # Skip if insufficient data
                        continue

                    # Create environment
                    env = TradingEnvironment(
                        market_data=data,
                        symbol=symbol,
                        initial_balance=10000.0
                    )

                    # Run episode
                    episode_reward_symbol, episode_length_symbol = await self._run_episode(env)
                    episode_reward += episode_reward_symbol
                    episode_length = max(episode_length, episode_length_symbol)

                # Update metrics
                portfolio_value = 10000 + episode_reward  # Simplified
                epsilon = self.agent.config['epsilon']

                self.metrics.update_episode(
                    episode, episode_reward, episode_length, portfolio_value, epsilon
                )

                # Periodic evaluation
                if episode % self.config.eval_freq == 0:
                    eval_metrics = await self._evaluate_agent(market_data)
                    self.metrics.update_evaluation(
                        eval_metrics['win_rate'],
                        eval_metrics['sharpe_ratio'],
                        eval_metrics['max_drawdown']
                    )

                # Save model periodically
                if episode % self.config.save_freq == 0:
                    await self._save_checkpoint(episode)

                # Early stopping check
                if self._check_early_stopping():
                    logger.info(f"Early stopping at episode {episode}")
                    break

                # Progress callback
                if progress_callback:
                    await progress_callback(episode, self.config.episodes, self.metrics.get_summary())

                # Log progress
                if episode % 50 == 0:
                    summary = self.metrics.get_summary()
                    logger.info(f"Episode {episode}: Reward={summary['avg_reward']:.2f}, "
                              f"Epsilon={summary['final_epsilon']:.3f}")

            # Final evaluation
            final_metrics = await self._evaluate_agent(market_data)

            # Save final model
            await self._save_checkpoint(self.current_episode, final=True)

            training_results = {
                'status': 'completed',
                'episodes_completed': self.current_episode + 1,
                'final_metrics': final_metrics,
                'training_summary': self.metrics.get_summary(),
                'config': self.config.__dict__
            }

            logger.info(f"Training completed. Final reward: {final_metrics.get('total_return', 0):.4f}")
            return training_results

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'episodes_completed': self.current_episode,
                'training_summary': self.metrics.get_summary()
            }
        finally:
            self.is_training = False

    async def _run_episode(self, env: TradingEnvironment) -> Tuple[float, int]:
        """Run a single training episode."""
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < self.config.max_steps_per_episode:
            # Select action
            action = self.agent.get_action(state, training=True)

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
            self.agent.update_q_value(experience)

            # Update state
            state = next_state
            total_reward += reward
            steps += 1

        return total_reward, steps

    async def _evaluate_agent(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Evaluate agent performance on validation data."""
        total_return = 0
        total_trades = 0
        winning_trades = 0
        returns_list = []

        for symbol, data in market_data.items():
            if len(data) < 50:
                continue

            # Create environment
            env = TradingEnvironment(market_data=data, symbol=symbol)

            # Run evaluation episode (greedy policy)
            state = env.reset()
            done = False
            episode_return = 0

            while not done:
                # Greedy action selection
                action = self.agent.get_action(state, training=False)
                next_state, reward, done, info = env.step(action)
                episode_return += reward
                state = next_state

            # Get performance metrics
            metrics = env.get_performance_metrics()
            total_return += metrics['total_return']
            total_trades += metrics['total_trades']
            winning_trades += metrics['win_rate'] * metrics['total_trades']

            if metrics['total_return'] != 0:
                returns_list.append(metrics['total_return'])

        # Aggregate metrics
        avg_return = total_return / len(market_data) if market_data else 0
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        sharpe_ratio = np.mean(returns_list) / (np.std(returns_list) + 1e-8) * np.sqrt(252) if returns_list else 0
        max_drawdown = 0  # Simplified

        return {
            'total_return': avg_return,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades
        }

    def _check_early_stopping(self) -> bool:
        """Check if training should stop early."""
        if len(self.metrics.episode_rewards) < self.config.early_stopping_patience:
            return False

        recent_rewards = self.metrics.episode_rewards[-self.config.early_stopping_patience:]
        best_recent = max(recent_rewards)

        if best_recent > self.best_reward:
            self.best_reward = best_recent
            return False

        # No improvement for patience episodes
        return True

    async def _save_checkpoint(self, episode: int, final: bool = False):
        """Save model checkpoint."""
        try:
            os.makedirs('models/rl_checkpoints', exist_ok=True)

            filename = f"rl_agent_{'final' if final else episode}.pkl"
            filepath = f"models/rl_checkpoints/{filename}"

            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, self.agent.save_model, filepath)

            logger.info(f"Saved checkpoint: {filepath}")

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def stop_training(self):
        """Stop ongoing training."""
        self.is_training = False
        logger.info("Training stop requested")

    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status."""
        return {
            'is_training': self.is_training,
            'current_episode': self.current_episode,
            'total_episodes': self.config.episodes,
            'progress': self.current_episode / self.config.episodes if self.config.episodes > 0 else 0,
            'metrics': self.metrics.get_summary()
        }


class AsyncRLTrainer(RLTrainer):
    """Asynchronous RL Trainer for parallel training."""

    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.training_task: Optional[asyncio.Task] = None

    async def train_async(self, market_data: Dict[str, pd.DataFrame],
                         progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """Train asynchronously."""
        if self.training_task and not self.training_task.done():
            return {'status': 'already_training'}

        self.training_task = asyncio.create_task(
            self.train(market_data, progress_callback)
        )

        return {'status': 'training_started', 'task_id': id(self.training_task)}

    async def get_training_result(self) -> Optional[Dict[str, Any]]:
        """Get training result if completed."""
        if self.training_task and self.training_task.done():
            try:
                return self.training_task.result()
            except Exception as e:
                return {'status': 'failed', 'error': str(e)}

        return None

    def stop_training_async(self):
        """Stop asynchronous training."""
        if self.training_task and not self.training_task.done():
            self.training_task.cancel()
        self.stop_training()