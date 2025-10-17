#!/usr/bin/env python3
"""
Reinforcement Learning for TradPal Trading System

This module implements reinforcement learning algorithms for trading,
including DQN, PPO, and custom trading-specific RL agents.
"""

import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import deque
import random
from dataclasses import dataclass
from abc import ABC, abstractmethod

from services.core.gpu_accelerator import get_gpu_accelerator, is_gpu_available

logger = logging.getLogger(__name__)

@dataclass
class RLConfig:
    """Configuration for reinforcement learning."""
    algorithm: str = 'dqn'  # 'dqn', 'ppo', 'a2c'
    state_size: int = 50
    action_size: int = 3  # buy, hold, sell
    hidden_size: int = 128
    num_layers: int = 2
    learning_rate: float = 0.001
    gamma: float = 0.99  # discount factor
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    batch_size: int = 64
    memory_size: int = 10000
    target_update_freq: int = 100
    episodes: int = 1000
    max_steps_per_episode: int = 1000
    reward_function: str = 'sharpe'  # 'sharpe', 'returns', 'sortino', 'custom'

@dataclass
class TradingState:
    """Trading environment state."""
    portfolio_value: float
    position: int  # -1: short, 0: neutral, 1: long
    cash: float
    holdings: float
    current_price: float
    features: np.ndarray

@dataclass
class TradingAction:
    """Trading action."""
    action_type: int  # 0: hold, 1: buy, 2: sell
    quantity: float = 1.0  # fraction of available cash/holdings

@dataclass
class Experience:
    """RL experience tuple."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool

class TradingEnvironment:
    """Trading environment for reinforcement learning."""

    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000.0,
                 transaction_cost: float = 0.001):
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.current_step = 0
        self.max_steps = len(data) - 1

        # Portfolio state
        self.balance = initial_balance
        self.position = 0  # shares held
        self.portfolio_value = initial_balance

        # Track performance
        self.portfolio_history = [initial_balance]
        self.action_history = []

    def reset(self) -> TradingState:
        """Reset environment to initial state."""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.portfolio_value = self.initial_balance
        self.portfolio_history = [self.initial_balance]
        self.action_history = []

        return self._get_state()

    def step(self, action: TradingAction) -> Tuple[TradingState, float, bool]:
        """Execute action and return next state, reward, done."""
        current_price = self.data.iloc[self.current_step]['close']

        # Execute action
        reward = self._execute_action(action, current_price)

        # Update portfolio value
        self.portfolio_value = self.balance + (self.position * current_price)

        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.max_steps

        # Get next state
        next_state = self._get_state()

        # Update history
        self.portfolio_history.append(self.portfolio_value)
        self.action_history.append(action.action_type)

        return next_state, reward, done

    def _execute_action(self, action: TradingAction, current_price: float) -> float:
        """Execute trading action and return reward."""
        old_portfolio_value = self.portfolio_value

        if action.action_type == 0:  # Hold
            pass  # No action

        elif action.action_type == 1:  # Buy
            # Calculate affordable quantity
            available_cash = self.balance * action.quantity
            shares_to_buy = available_cash / current_price
            cost = shares_to_buy * current_price * (1 + self.transaction_cost)

            if cost <= self.balance:
                self.balance -= cost
                self.position += shares_to_buy

        elif action.action_type == 2:  # Sell
            # Calculate shares to sell
            shares_to_sell = self.position * action.quantity
            proceeds = shares_to_sell * current_price * (1 - self.transaction_cost)

            if shares_to_sell > 0:
                self.balance += proceeds
                self.position -= shares_to_sell

        # Calculate reward based on portfolio change
        new_portfolio_value = self.balance + (self.position * current_price)
        reward = self._calculate_reward(old_portfolio_value, new_portfolio_value)

        return reward

    def _calculate_reward(self, old_value: float, new_value: float) -> float:
        """Calculate reward based on portfolio performance."""
        if old_value == 0:
            return 0.0

        # Simple return-based reward
        return_rate = (new_value - old_value) / old_value

        # Add penalty for large position changes (encourage stability)
        position_change_penalty = abs(self.position) * 0.0001

        return return_rate - position_change_penalty

    def _get_state(self) -> TradingState:
        """Get current trading state."""
        current_price = self.data.iloc[self.current_step]['close']

        # Create feature vector (simplified - would use actual features)
        features = np.array([
            current_price,
            self.position,
            self.balance,
            self.portfolio_value
        ])

        return TradingState(
            portfolio_value=self.portfolio_value,
            position=self.position,
            cash=self.balance,
            holdings=self.position,
            current_price=current_price,
            features=features
        )

    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics."""
        returns = pd.Series(self.portfolio_history).pct_change().dropna()

        if len(returns) == 0:
            return {'total_return': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0}

        total_return = (self.portfolio_value - self.initial_balance) / self.initial_balance

        # Sharpe ratio (annualized)
        if returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_portfolio_value': self.portfolio_value
        }

class ReplayBuffer:
    """Experience replay buffer for RL."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, experience: Experience) -> None:
        """Add experience to buffer."""
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch of experiences."""
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)

class BaseRLAgent(ABC):
    """Abstract base class for RL agents."""

    def __init__(self, config: RLConfig):
        self.config = config
        self.device = get_gpu_accelerator().get_optimal_device()

    @abstractmethod
    def select_action(self, state: TradingState) -> TradingAction:
        """Select action based on current state."""
        pass

    @abstractmethod
    def train(self, experiences: List[Experience]) -> float:
        """Train agent with experiences."""
        pass

    @abstractmethod
    def update_target_network(self) -> None:
        """Update target network (for DQN)."""
        pass

    def save_model(self, path: str) -> None:
        """Save model to disk."""
        if hasattr(self, 'policy_net') and self.policy_net is not None:
            torch.save(self.policy_net.state_dict(), path)
            logger.info(f"Policy network saved to {path}")
        elif hasattr(self, 'actor') and self.actor is not None:
            # For PPO agents
            torch.save({
                'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict()
            }, path)
            logger.info(f"PPO networks saved to {path}")

    def load_model(self, path: str) -> None:
        """Load model from disk."""
        if hasattr(self, 'policy_net') and self.policy_net is not None:
            self.policy_net.load_state_dict(torch.load(path))
            self.policy_net.eval()
            logger.info(f"Policy network loaded from {path}")
        elif hasattr(self, 'actor') and self.actor is not None:
            # For PPO agents
            checkpoint = torch.load(path)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.actor.eval()
            self.critic.eval()
            logger.info(f"PPO networks loaded from {path}")

class DQNAgent(BaseRLAgent):
    """Deep Q-Network agent."""

    def __init__(self, config: RLConfig):
        super().__init__(config)

        # Networks
        self.policy_net = self._build_network()
        self.target_net = self._build_network()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)

        # Replay buffer
        self.memory = ReplayBuffer(config.memory_size)

        # Exploration
        self.epsilon = config.epsilon_start

    def _build_network(self) -> nn.Module:
        """Build neural network."""
        return nn.Sequential(
            nn.Linear(self.config.state_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, self.config.action_size)
        ).to(self.device)

    def select_action(self, state: TradingState) -> TradingAction:
        """Select action using epsilon-greedy policy."""
        if random.random() > self.epsilon:
            # Exploitation
            with torch.no_grad():
                state_tensor = torch.tensor(state.features, dtype=torch.float32).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                action_idx = q_values.argmax().item()
        else:
            # Exploration
            action_idx = random.randint(0, self.config.action_size - 1)

        # Decay epsilon
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

        return TradingAction(action_type=action_idx)

    def train(self, experiences: List[Experience]) -> float:
        """Train DQN with batch of experiences."""
        if len(experiences) < self.config.batch_size:
            return 0.0

        # Prepare batch
        states = torch.tensor([e.state for e in experiences], dtype=torch.float32).to(self.device)
        actions = torch.tensor([e.action for e in experiences], dtype=torch.long).to(self.device)
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32).to(self.device)
        next_states = torch.tensor([e.next_state for e in experiences], dtype=torch.float32).to(self.device)
        dones = torch.tensor([e.done for e in experiences], dtype=torch.float32).to(self.device)

        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (self.config.gamma * next_q_values * (1 - dones))

        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self) -> None:
        """Update target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

class PPOAgent(BaseRLAgent):
    """Proximal Policy Optimization agent."""

    def __init__(self, config: RLConfig):
        super().__init__(config)

        # Actor-Critic networks
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.old_actor = self._build_actor()
        self.old_actor.load_state_dict(self.actor.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.learning_rate)

        # PPO parameters
        self.clip_param = 0.2
        self.ppo_epochs = 10
        self.mini_batch_size = 64

    def _build_actor(self) -> nn.Module:
        """Build actor network."""
        return nn.Sequential(
            nn.Linear(self.config.state_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, self.config.action_size),
            nn.Softmax(dim=-1)
        ).to(self.device)

    def _build_critic(self) -> nn.Module:
        """Build critic network."""
        return nn.Sequential(
            nn.Linear(self.config.state_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, 1)
        ).to(self.device)

    def select_action(self, state: TradingState) -> TradingAction:
        """Select action using current policy."""
        with torch.no_grad():
            state_tensor = torch.tensor(state.features, dtype=torch.float32).unsqueeze(0).to(self.device)
            action_probs = self.actor(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action_idx = action_dist.sample().item()

        return TradingAction(action_type=action_idx)

    def train(self, experiences: List[Experience]) -> float:
        """Train PPO with experiences."""
        # Convert experiences to tensors
        states = torch.tensor([e.state for e in experiences], dtype=torch.float32).to(self.device)
        actions = torch.tensor([e.action for e in experiences], dtype=torch.long).to(self.device)
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32).to(self.device)
        old_log_probs = torch.tensor([self._get_log_prob(e.state, e.action) for e in experiences],
                                   dtype=torch.float32).to(self.device)

        # Calculate advantages and returns
        advantages, returns = self._calculate_advantages(rewards, states)

        # PPO training loop
        total_loss = 0
        for _ in range(self.ppo_epochs):
            # Create mini-batches
            indices = torch.randperm(len(experiences))
            for start in range(0, len(experiences), self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]

                # Actor loss
                new_log_probs = self._get_log_prob_batch(batch_states, batch_actions)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic loss
                values = self.critic(batch_states).squeeze()
                critic_loss = nn.MSELoss()(values, batch_returns)

                # Total loss
                loss = actor_loss + 0.5 * critic_loss

                # Update networks
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                total_loss += loss.item()

        # Update old actor
        self.old_actor.load_state_dict(self.actor.state_dict())

        return total_loss / (self.ppo_epochs * (len(experiences) // self.mini_batch_size))

    def _get_log_prob(self, state: np.ndarray, action: int) -> float:
        """Get log probability of action under old policy."""
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            action_probs = self.old_actor(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            return action_dist.log_prob(torch.tensor(action).to(self.device)).item()

    def _get_log_prob_batch(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Get log probabilities for batch."""
        action_probs = self.actor(states)
        action_dist = torch.distributions.Categorical(action_probs)
        return action_dist.log_prob(actions)

    def _calculate_advantages(self, rewards: torch.Tensor, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate advantages and returns."""
        with torch.no_grad():
            values = self.critic(states).squeeze()
            next_values = torch.cat([values[1:], torch.tensor([0.0]).to(self.device)])

        # Simple advantage calculation (could be improved with GAE)
        advantages = rewards + self.config.gamma * next_values - values
        returns = rewards + self.config.gamma * next_values

        return advantages, returns

    def update_target_network(self) -> None:
        """No target network for PPO."""
        pass

class RLTrainer:
    """Reinforcement learning trainer."""

    def __init__(self, agent: BaseRLAgent, environment: TradingEnvironment, config: RLConfig):
        self.agent = agent
        self.env = environment
        self.config = config

    def train(self) -> Dict[str, List[float]]:
        """Train RL agent."""
        episode_rewards = []
        episode_losses = []
        best_reward = -float('inf')

        for episode in range(self.config.episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_loss = 0
            experiences = []

            for step in range(self.config.max_steps_per_episode):
                # Select action
                action = self.agent.select_action(state)

                # Execute action
                next_state, reward, done = self.env.step(action)

                # Store experience
                experience = Experience(
                    state=state.features,
                    action=action.action_type,
                    reward=reward,
                    next_state=next_state.features,
                    done=done
                )
                experiences.append(experience)

                # Accumulate reward
                episode_reward += reward
                state = next_state

                if done:
                    break

            # Train agent
            if hasattr(self.agent, 'memory'):
                # DQN-style training
                for exp in experiences:
                    self.agent.memory.push(exp)
                if len(self.agent.memory) >= self.config.batch_size:
                    batch = self.agent.memory.sample(self.config.batch_size)
                    loss = self.agent.train(batch)
                    episode_loss = loss
            else:
                # PPO-style training
                loss = self.agent.train(experiences)
                episode_loss = loss

            # Update target network (for DQN)
            if episode % self.config.target_update_freq == 0:
                self.agent.update_target_network()

            episode_rewards.append(episode_reward)
            episode_losses.append(episode_loss)

            # Logging
            if episode % 10 == 0:
                logger.info(f"Episode {episode}: Reward = {episode_reward:.4f}, Loss = {episode_loss:.4f}")

                # Save best model
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    self.agent.save_model(f"best_rl_model_{episode}.pth")

        return {
            'episode_rewards': episode_rewards,
            'episode_losses': episode_losses
        }

    def evaluate(self, episodes: int = 10) -> Dict[str, float]:
        """Evaluate trained agent."""
        total_rewards = []
        portfolio_values = []

        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0

            for step in range(self.config.max_steps_per_episode):
                action = self.agent.select_action(state)
                next_state, reward, done = self.env.step(action)
                episode_reward += reward
                state = next_state

                if done:
                    break

            total_rewards.append(episode_reward)
            portfolio_values.append(self.env.portfolio_value)

        return {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'mean_portfolio_value': np.mean(portfolio_values),
            'std_portfolio_value': np.std(portfolio_values)
        }

# Factory functions
def create_rl_agent(config: RLConfig) -> BaseRLAgent:
    """Create RL agent."""
    if config.algorithm.lower() == 'dqn':
        return DQNAgent(config)
    elif config.algorithm.lower() == 'ppo':
        return PPOAgent(config)
    else:
        raise ValueError(f"Unknown RL algorithm: {config.algorithm}")

def create_trading_environment(data: pd.DataFrame, initial_balance: float = 10000.0) -> TradingEnvironment:
    """Create trading environment."""
    return TradingEnvironment(data, initial_balance)

def create_rl_trainer(agent: BaseRLAgent, environment: TradingEnvironment, config: RLConfig) -> RLTrainer:
    """Create RL trainer."""
    return RLTrainer(agent, environment, config)