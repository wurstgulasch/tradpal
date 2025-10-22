"""
Reinforcement Learning Agent for Trading
Implements Q-Learning and Deep RL algorithms for automated trading decisions.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import pickle
import os
from abc import ABC, abstractmethod
from enum import Enum

logger = logging.getLogger(__name__)


class RLAlgorithm(Enum):
    """Available RL algorithms."""
    Q_LEARNING = "q_learning"
    SARSA = "sarsa"
    DQN = "dqn"
    PPO = "ppo"


class TradingAction(Enum):
    """Possible trading actions."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    REDUCE = "reduce_position"
    INCREASE = "increase_position"


@dataclass
class TradingState:
    """Represents the current trading state."""
    symbol: str
    position_size: float  # -1 (short) to 1 (long)
    current_price: float
    portfolio_value: float
    market_regime: str
    volatility_regime: str
    trend_strength: float
    technical_indicators: Dict[str, float]
    timestamp: datetime

    def to_vector(self) -> np.ndarray:
        """Convert state to numerical vector for RL algorithms."""
        # Normalize and create feature vector
        features = [
            self.position_size,  # -1 to 1
            np.log(self.current_price) / 10,  # Normalized log price
            np.log(self.portfolio_value) / 10,  # Normalized log portfolio
            self.trend_strength,  # -1 to 1
        ]

        # Add market regime one-hot encoding
        regime_features = self._encode_regime(self.market_regime)
        features.extend(regime_features)

        # Add volatility regime
        volatility_features = self._encode_regime(self.volatility_regime)
        features.extend(volatility_features)

        # Add technical indicators
        for indicator in ['rsi', 'macd', 'bb_position']:
            value = self.technical_indicators.get(indicator, 0.0)
            # Normalize common indicators
            if indicator == 'rsi':
                value = (value - 50) / 50  # Center around 0
            elif indicator == 'macd':
                value = value / 100  # Scale down
            features.append(value)

        return np.array(features, dtype=np.float32)

    def _encode_regime(self, regime: str) -> List[float]:
        """One-hot encode regime information."""
        regimes = ['bull_market', 'bear_market', 'sideways', 'high_volatility', 'low_volatility']
        encoding = [1.0 if r == regime else 0.0 for r in regimes]
        return encoding

    def discretize(self, bins: Dict[str, int]) -> Tuple:
        """Discretize continuous state variables for Q-learning."""
        discretized = []

        # Discretize position size
        pos_bins = np.linspace(-1, 1, bins.get('position', 5))
        discretized.append(np.digitize(self.position_size, pos_bins) - 1)

        # Discretize price (log scale)
        price_bins = np.linspace(8, 15, bins.get('price', 7))  # log(1000) to log(1e6)
        discretized.append(np.digitize(np.log(self.current_price), price_bins) - 1)

        # Discretize trend strength
        trend_bins = np.linspace(-1, 1, bins.get('trend', 5))
        discretized.append(np.digitize(self.trend_strength, trend_bins) - 1)

        # Add regime as categorical
        regime_map = {
            'bull_market': 0, 'bear_market': 1, 'sideways': 2,
            'high_volatility': 3, 'low_volatility': 4
        }
        discretized.append(regime_map.get(self.market_regime, 2))

        return tuple(discretized)


@dataclass
class RLExperience:
    """Experience tuple for RL training."""
    state: TradingState
    action: TradingAction
    reward: float
    next_state: TradingState
    done: bool
    timestamp: datetime


class RewardFunction:
    """Calculates rewards for trading actions."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        return {
            'pnl_weight': 0.4,
            'sharpe_weight': 0.2,
            'max_drawdown_penalty': 0.2,
            'regime_alignment_bonus': 0.1,
            'transaction_cost_penalty': 0.1,
            'risk_adjustment': 0.05,
            'time_decay': 0.01
        }

    def calculate_reward(self, action: TradingAction, state: TradingState,
                        next_state: TradingState, trade_pnl: float = 0.0) -> float:
        """
        Calculate reward for a trading action.

        Args:
            action: The action taken
            state: State before action
            next_state: State after action
            trade_pnl: Profit/loss from the trade

        Returns:
            Reward value
        """
        reward = 0.0

        # PnL reward
        reward += self.config['pnl_weight'] * trade_pnl

        # Sharpe ratio component (simplified)
        if next_state.portfolio_value > state.portfolio_value:
            reward += self.config['sharpe_weight'] * 0.1
        elif next_state.portfolio_value < state.portfolio_value:
            reward -= self.config['sharpe_weight'] * 0.1

        # Risk management
        position_change = abs(next_state.position_size - state.position_size)
        reward -= self.config['risk_adjustment'] * position_change

        # Regime alignment bonus
        regime_bonus = self._calculate_regime_alignment_bonus(action, state.market_regime)
        reward += self.config['regime_alignment_bonus'] * regime_bonus

        # Transaction cost penalty
        if action in [TradingAction.BUY, TradingAction.SELL]:
            reward -= self.config['transaction_cost_penalty']

        # Time decay (encourage action)
        reward -= self.config['time_decay']

        return reward

    def _calculate_regime_alignment_bonus(self, action: TradingAction, regime: str) -> float:
        """Calculate bonus for actions aligned with market regime."""
        alignment_matrix = {
            'bull_market': {
                TradingAction.BUY: 1.0,
                TradingAction.INCREASE: 0.5,
                TradingAction.HOLD: 0.2,
                TradingAction.SELL: -0.5,
                TradingAction.REDUCE: -0.2
            },
            'bear_market': {
                TradingAction.SELL: 1.0,
                TradingAction.REDUCE: 0.5,
                TradingAction.HOLD: 0.2,
                TradingAction.BUY: -0.5,
                TradingAction.INCREASE: -0.2
            },
            'sideways': {
                TradingAction.HOLD: 0.5,
                TradingAction.REDUCE: 0.3,
                TradingAction.BUY: -0.1,
                TradingAction.SELL: -0.1,
                TradingAction.INCREASE: -0.2
            },
            'high_volatility': {
                TradingAction.REDUCE: 0.8,
                TradingAction.HOLD: 0.4,
                TradingAction.SELL: 0.2,
                TradingAction.BUY: -0.3,
                TradingAction.INCREASE: -0.5
            },
            'low_volatility': {
                TradingAction.BUY: 0.3,
                TradingAction.INCREASE: 0.4,
                TradingAction.HOLD: 0.3,
                TradingAction.SELL: -0.1,
                TradingAction.REDUCE: -0.2
            }
        }

        return alignment_matrix.get(regime, {}).get(action, 0.0)


class QLearningAgent:
    """Q-Learning agent for trading decisions."""

    def __init__(self, state_bins: Dict[str, int], actions: List[TradingAction],
                 config: Optional[Dict[str, Any]] = None):
        self.state_bins = state_bins
        self.actions = actions
        self.config = config or self._default_config()

        # Initialize Q-table
        state_space_size = self._calculate_state_space_size()
        self.q_table = np.zeros((state_space_size, len(actions)))

        # Experience replay buffer
        self.experience_buffer: List[RLExperience] = []
        self.max_buffer_size = self.config['experience_buffer_size']

        # Training metrics
        self.episode_rewards: List[float] = []
        self.episode_losses: List[float] = []

        logger.info(f"Initialized Q-Learning agent with {state_space_size} states and {len(actions)} actions")

    def _default_config(self) -> Dict[str, Any]:
        return {
            'learning_rate': 0.1,
            'discount_factor': 0.95,
            'epsilon': 1.0,
            'epsilon_decay': 0.995,
            'min_epsilon': 0.01,
            'experience_buffer_size': 10000,
            'batch_size': 32
        }

    def _calculate_state_space_size(self) -> int:
        """Calculate total state space size for Q-table."""
        position_bins = self.state_bins.get('position', 5)
        price_bins = self.state_bins.get('price', 7)
        trend_bins = self.state_bins.get('trend', 5)
        regime_states = 5  # 5 market regimes

        return position_bins * price_bins * trend_bins * regime_states

    def get_action(self, state: TradingState, training: bool = True) -> TradingAction:
        """Select action using epsilon-greedy policy."""
        discretized_state = state.discretize(self.state_bins)
        state_idx = self._state_to_index(discretized_state)

        if training and np.random.random() < self.config['epsilon']:
            # Explore
            action_idx = np.random.randint(len(self.actions))
        else:
            # Exploit
            action_idx = np.argmax(self.q_table[state_idx])

        return self.actions[action_idx]

    def update_q_value(self, experience: RLExperience):
        """Update Q-value using Q-learning update rule."""
        state = experience.state.discretize(self.state_bins)
        next_state = experience.next_state.discretize(self.state_bins)

        state_idx = self._state_to_index(state)
        next_state_idx = self._state_to_index(next_state)
        action_idx = self.actions.index(experience.action)

        # Q-learning update
        current_q = self.q_table[state_idx, action_idx]
        max_next_q = np.max(self.q_table[next_state_idx])

        new_q = current_q + self.config['learning_rate'] * (
            experience.reward + self.config['discount_factor'] * max_next_q - current_q
        )

        self.q_table[state_idx, action_idx] = new_q

        # Store experience
        self.experience_buffer.append(experience)
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)

        # Decay epsilon
        self.config['epsilon'] = max(
            self.config['min_epsilon'],
            self.config['epsilon'] * self.config['epsilon_decay']
        )

    def _state_to_index(self, discretized_state: Tuple) -> int:
        """Convert discretized state tuple to linear index."""
        position, price, trend, regime = discretized_state

        position_bins = self.state_bins.get('position', 5)
        price_bins = self.state_bins.get('price', 7)
        trend_bins = self.state_bins.get('trend', 5)

        index = (
            position +
            price * position_bins +
            trend * position_bins * price_bins +
            regime * position_bins * price_bins * trend_bins
        )

        return min(index, self.q_table.shape[0] - 1)

    def train_on_experience_buffer(self, reward_function: RewardFunction):
        """Train on experiences from buffer."""
        if len(self.experience_buffer) < self.config['batch_size']:
            return

        # Sample batch
        batch_indices = np.random.choice(
            len(self.experience_buffer),
            size=min(self.config['batch_size'], len(self.experience_buffer)),
            replace=False
        )

        total_loss = 0
        for idx in batch_indices:
            experience = self.experience_buffer[idx]
            self.update_q_value(experience)
            total_loss += abs(experience.reward)  # Simplified loss

        self.episode_losses.append(total_loss / len(batch_indices))

    def save_model(self, filepath: str):
        """Save Q-table and configuration."""
        model_data = {
            'q_table': self.q_table,
            'config': self.config,
            'state_bins': self.state_bins,
            'actions': [action.value for action in self.actions],
            'episode_rewards': self.episode_rewards,
            'episode_losses': self.episode_losses
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load Q-table and configuration."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.q_table = model_data['q_table']
        self.config = model_data['config']
        self.state_bins = model_data['state_bins']
        self.actions = [TradingAction(action) for action in model_data['actions']]
        self.episode_rewards = model_data.get('episode_rewards', [])
        self.episode_losses = model_data.get('episode_losses', [])

        logger.info(f"Model loaded from {filepath}")

    def get_training_metrics(self) -> Dict[str, Any]:
        """Get training performance metrics."""
        return {
            'total_episodes': len(self.episode_rewards),
            'average_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'average_loss': np.mean(self.episode_losses) if self.episode_losses else 0,
            'current_epsilon': self.config['epsilon'],
            'q_table_sparsity': np.count_nonzero(self.q_table) / self.q_table.size
        }


class DeepRLAgent(ABC):
    """Abstract base class for deep reinforcement learning agents."""

    def __init__(self, state_dim: int, action_dim: int, config: Optional[Dict[str, Any]] = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or self._default_config()

    @abstractmethod
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for the agent."""
        pass

    @abstractmethod
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action given state."""
        pass

    @abstractmethod
    def update(self, experiences: List[RLExperience]):
        """Update agent parameters using experiences."""
        pass

    @abstractmethod
    def save_model(self, filepath: str):
        """Save model parameters."""
        pass

    @abstractmethod
    def load_model(self, filepath: str):
        """Load model parameters."""
        pass


class TradingEnvironment:
    """Trading environment for RL training."""

    def __init__(self, market_data: pd.DataFrame, symbol: str,
                 initial_balance: float = 10000.0, transaction_cost: float = 0.001):
        self.market_data = market_data.copy()
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost

        # Environment state
        self.current_step = 0
        self.balance = initial_balance
        self.position_size = 0.0  # -1 to 1 scale
        self.portfolio_value = initial_balance
        self.trades = []

        # Market regime information (would be provided by regime service)
        self.market_regime = "sideways"
        self.volatility_regime = "normal"

    def reset(self) -> TradingState:
        """Reset environment to initial state."""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position_size = 0.0
        self.portfolio_value = self.initial_balance
        self.trades = []

        return self._get_current_state()

    def step(self, action: TradingAction) -> Tuple[TradingState, float, bool, Dict[str, Any]]:
        """
        Execute action and return next state, reward, done flag, and info.

        Returns:
            next_state, reward, done, info
        """
        current_state = self._get_current_state()
        current_price = self.market_data.iloc[self.current_step]['close']

        # Execute action
        trade_pnl = 0.0
        if action == TradingAction.BUY and self.position_size < 1.0:
            # Increase long position
            position_change = min(0.2, 1.0 - self.position_size)  # Max 20% position change
            trade_value = self.portfolio_value * position_change
            trade_cost = trade_value * self.transaction_cost

            self.position_size += position_change
            self.balance -= trade_cost
            self.trades.append({
                'step': self.current_step,
                'action': 'buy',
                'size': position_change,
                'price': current_price,
                'cost': trade_cost
            })

        elif action == TradingAction.SELL and self.position_size > -1.0:
            # Increase short position or reduce long
            if self.position_size > 0:
                # Close long position
                position_value = self.portfolio_value * self.position_size
                trade_pnl = position_value * (current_price / self.market_data.iloc[self.current_step - 1]['close'] - 1)
                trade_cost = abs(position_value) * self.transaction_cost

                self.balance += position_value + trade_pnl - trade_cost
                self.position_size = 0.0
            else:
                # Increase short position
                position_change = min(0.2, 1.0 + self.position_size)
                trade_value = self.portfolio_value * position_change
                trade_cost = trade_value * self.transaction_cost

                self.position_size -= position_change
                self.balance -= trade_cost

            self.trades.append({
                'step': self.current_step,
                'action': 'sell',
                'pnl': trade_pnl,
                'cost': trade_cost
            })

        elif action == TradingAction.REDUCE and abs(self.position_size) > 0:
            # Reduce position size
            reduce_amount = min(0.1, abs(self.position_size))
            if self.position_size > 0:
                self.position_size -= reduce_amount
            else:
                self.position_size += reduce_amount

        elif action == TradingAction.INCREASE and abs(self.position_size) < 1.0:
            # Increase position size
            increase_amount = min(0.1, 1.0 - abs(self.position_size))
            if self.position_size >= 0:
                self.position_size += increase_amount
            else:
                self.position_size -= increase_amount

        # Update portfolio value
        position_value = self.portfolio_value * abs(self.position_size)
        if self.position_size > 0:
            # Long position
            price_change = current_price / self.market_data.iloc[max(0, self.current_step - 1)]['close']
            self.portfolio_value = self.balance + position_value * price_change
        elif self.position_size < 0:
            # Short position
            price_change = self.market_data.iloc[max(0, self.current_step - 1)]['close'] / current_price
            self.portfolio_value = self.balance + position_value * price_change
        else:
            self.portfolio_value = self.balance

        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.market_data) - 1

        # Get next state
        next_state = self._get_current_state()

        # Calculate reward
        reward_function = RewardFunction()
        reward = reward_function.calculate_reward(action, current_state, next_state, trade_pnl)

        info = {
            'portfolio_value': self.portfolio_value,
            'position_size': self.position_size,
            'balance': self.balance,
            'trade_pnl': trade_pnl,
            'step': self.current_step
        }

        return next_state, reward, done, info

    def _get_current_state(self) -> TradingState:
        """Get current trading state."""
        if self.current_step >= len(self.market_data):
            self.current_step = len(self.market_data) - 1

        row = self.market_data.iloc[self.current_step]

        # Extract technical indicators
        technical_indicators = {}
        for col in ['rsi', 'macd', 'bb_position']:
            if col in row.index:
                technical_indicators[col] = row[col]
            else:
                technical_indicators[col] = 0.0

        # Calculate trend strength (simplified)
        if self.current_step > 10:
            recent_prices = self.market_data.iloc[self.current_step-10:self.current_step+1]['close']
            trend_strength = np.polyfit(range(len(recent_prices)), recent_prices.values, 1)[0]
            trend_strength = np.tanh(trend_strength * 100)  # Normalize to -1, 1
        else:
            trend_strength = 0.0

        return TradingState(
            symbol=self.symbol,
            position_size=self.position_size,
            current_price=row['close'],
            portfolio_value=self.portfolio_value,
            market_regime=self.market_regime,
            volatility_regime=self.volatility_regime,
            trend_strength=trend_strength,
            technical_indicators=technical_indicators,
            timestamp=datetime.now()
        )

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calculate trading performance metrics."""
        if not self.trades:
            return {'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'win_rate': 0}

        # Calculate returns
        returns = []
        portfolio_values = [self.initial_balance]

        for trade in self.trades:
            if 'pnl' in trade:
                returns.append(trade['pnl'] / self.initial_balance)

        total_return = (self.portfolio_value - self.initial_balance) / self.initial_balance

        # Sharpe ratio (simplified)
        if returns:
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0

        # Max drawdown (simplified)
        max_drawdown = 0

        # Win rate
        winning_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(self.trades)
        }