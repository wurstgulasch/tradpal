"""
Unit tests for Reinforcement Learning Service components.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, AsyncMock
import asyncio
from datetime import datetime, timedelta

from services.reinforcement_learning_service.rl_agent import (
    RLAlgorithm,
    TradingAction,
    TradingState,
    RLExperience,
    QLearningAgent,
    RewardFunction,
    TradingEnvironment
)

from services.reinforcement_learning_service.training_engine import (
    TrainingConfig,
    TrainingMetrics,
    RLTrainer,
    AsyncRLTrainer
)


class TestRLAgent:
    """Test cases for RL Agent components."""

    def setup_method(self):
        """Setup test fixtures."""
        self.state_bins = {'position': 5, 'price': 7, 'trend': 5}
        self.actions = [TradingAction.BUY, TradingAction.HOLD, TradingAction.SELL]
        
        self.agent = QLearningAgent(
            state_bins=self.state_bins,
            actions=self.actions,
            config={
                'learning_rate': 0.1,
                'discount_factor': 0.9,
                'epsilon': 0.1,
                'epsilon_decay': 0.995,
                'min_epsilon': 0.01,
                'experience_buffer_size': 1000
            }
        )

        self.reward_function = RewardFunction(
            config={
                'pnl_weight': 0.4,
                'sharpe_weight': 0.2,
                'max_drawdown_penalty': 0.2,
                'regime_alignment_bonus': 0.1,
                'transaction_cost_penalty': 0.1,
                'risk_adjustment': 0.05,
                'time_decay': 0.01
            }
        )

        # Create sample market data for environment
        dates = pd.date_range('2023-01-01', periods=100, freq='1H')
        market_data = pd.DataFrame({
            'close': [50000 + i * 10 for i in range(100)],
            'rsi': [50 + (i % 20 - 10) for i in range(100)],
            'macd': [0.0 + (i % 10 - 5) * 0.1 for i in range(100)],
            'bb_position': [0.5 + (i % 10 - 5) * 0.02 for i in range(100)]
        }, index=dates)
        
        self.environment = TradingEnvironment(
            market_data=market_data,
            symbol="BTC/USDT",
            initial_balance=10000.0,
            transaction_cost=0.001
        )

    def test_q_learning_agent_initialization(self):
        """Test Q-Learning agent initialization."""
        assert self.agent.state_bins == self.state_bins
        assert self.agent.actions == self.actions
        assert self.agent.config['learning_rate'] == 0.1
        assert self.agent.config['discount_factor'] == 0.9
        assert self.agent.config['epsilon'] == 0.1
        expected_states = 5 * 7 * 5 * 5  # position * price * trend * regimes
        assert self.agent.q_table.shape == (expected_states, 3)

    def test_state_discretization(self):
        """Test state discretization."""
        # Create sample trading state
        state = TradingState(
            symbol="BTC/USDT",
            position_size=0.5,
            current_price=50000.0,
            portfolio_value=10500.0,
            market_regime="bull_market",
            volatility_regime="normal",
            trend_strength=0.8,
            technical_indicators={'rsi': 65.0, 'macd': 0.5, 'bb_position': 0.8},
            timestamp=datetime.now()
        )

        discretized = state.discretize(self.state_bins)
        assert isinstance(discretized, tuple)
        assert len(discretized) == 4  # position, price, trend, regime

    def test_epsilon_greedy_action_selection(self):
        """Test epsilon-greedy action selection."""
        state = TradingState(
            symbol="BTC/USDT",
            position_size=0.0,
            current_price=50000.0,
            portfolio_value=10000.0,
            market_regime="bull_market",
            volatility_regime="normal",
            trend_strength=0.0,
            technical_indicators={'rsi': 50.0, 'macd': 0.0, 'bb_position': 0.5},
            timestamp=datetime.now()
        )

        # Mock random choice to return specific action
        with patch('numpy.random.random', return_value=0.05):  # Below epsilon
            with patch('numpy.random.randint', return_value=1):  # Random action
                action = self.agent.get_action(state, training=True)
                assert action == TradingAction.HOLD

        # Test greedy action selection
        with patch('numpy.random.random', return_value=0.15):  # Above epsilon
            # Set high Q-value for action 2 (SELL)
            state_idx = self.agent._state_to_index(state.discretize(self.state_bins))
            self.agent.q_table[state_idx, 2] = 10.0
            action = self.agent.get_action(state, training=True)
            assert action == TradingAction.SELL

    def test_q_table_update(self):
        """Test Q-table update."""
        state = TradingState(
            symbol="BTC/USDT",
            position_size=0.0,
            current_price=50000.0,
            portfolio_value=10000.0,
            market_regime="bull_market",
            volatility_regime="normal",
            trend_strength=0.0,
            technical_indicators={'rsi': 50.0, 'macd': 0.0, 'bb_position': 0.5},
            timestamp=datetime.now()
        )
        
        next_state = TradingState(
            symbol="BTC/USDT",
            position_size=0.2,
            current_price=51000.0,
            portfolio_value=10200.0,
            market_regime="bull_market",
            volatility_regime="normal",
            trend_strength=0.1,
            technical_indicators={'rsi': 55.0, 'macd': 0.1, 'bb_position': 0.6},
            timestamp=datetime.now()
        )
        
        experience = RLExperience(
            state=state,
            action=TradingAction.BUY,
            reward=1.0,
            next_state=next_state,
            done=False,
            timestamp=datetime.now()
        )

        old_q_value = self.agent.q_table[self.agent._state_to_index(state.discretize(self.state_bins)), 0]
        self.agent.update_q_value(experience)

        new_q_value = self.agent.q_table[self.agent._state_to_index(state.discretize(self.state_bins)), 0]
        expected_q = old_q_value + 0.1 * (1.0 + 0.9 * np.max(self.agent.q_table[self.agent._state_to_index(next_state.discretize(self.state_bins))]) - old_q_value)

        assert abs(new_q_value - expected_q) < 1e-2  # Allow for small numerical differences

    def test_reward_function_calculation(self):
        """Test reward function calculation."""
        state = TradingState(
            symbol="BTC/USDT",
            position_size=0.0,
            current_price=50000.0,
            portfolio_value=10000.0,
            market_regime="bull_market",
            volatility_regime="normal",
            trend_strength=0.0,
            technical_indicators={'rsi': 50.0, 'macd': 0.0, 'bb_position': 0.5},
            timestamp=datetime.now()
        )
        
        next_state = TradingState(
            symbol="BTC/USDT",
            position_size=0.2,
            current_price=51000.0,
            portfolio_value=10200.0,
            market_regime="bull_market",
            volatility_regime="normal",
            trend_strength=0.1,
            technical_indicators={'rsi': 55.0, 'macd': 0.1, 'bb_position': 0.6},
            timestamp=datetime.now()
        )

        reward = self.reward_function.calculate_reward(TradingAction.BUY, state, next_state, trade_pnl=200.0)
        
        # Expected reward calculation based on config
        expected_reward = (
            0.4 * 200.0 +  # pnl_weight * trade_pnl
            0.2 * 0.1 -    # sharpe_weight (portfolio increased)
            0.05 * 0.2 -   # risk_adjustment * position_change
            0.1 +          # transaction_cost_penalty
            0.1 * 1.0 -    # regime_alignment_bonus (BUY in bull market)
            0.01           # time_decay
        )
        
        assert abs(reward - expected_reward) < 1e-6

    def test_trading_environment_step(self):
        """Test trading environment step."""
        # Create sample market data
        dates = pd.date_range('2023-01-01', periods=10, freq='1H')
        market_data = pd.DataFrame({
            'close': [50000 + i * 100 for i in range(10)],
            'rsi': [50 + i for i in range(10)],
            'macd': [0.0 + i * 0.1 for i in range(10)],
            'bb_position': [0.5 + i * 0.01 for i in range(10)]
        }, index=dates)
        
        environment = TradingEnvironment(
            market_data=market_data,
            symbol="BTC/USDT",
            initial_balance=10000.0,
            transaction_cost=0.001
        )

        initial_balance = environment.balance

        # Execute BUY action
        action = TradingAction.BUY
        next_state, reward, done, info = environment.step(action)

        # Balance should decrease due to transaction cost
        assert environment.balance < initial_balance
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert 'portfolio_value' in info
        assert 'position_size' in info


class TestTrainingEngine:
    """Test cases for Training Engine components."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = TrainingConfig(
            episodes=2,
            max_steps_per_episode=5,
            learning_rate=0.1,
            discount_factor=0.9,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            batch_size=32,
            target_update_freq=10
        )

        self.trainer = RLTrainer(self.config)

    def test_training_config_validation(self):
        """Test training configuration validation."""
        assert self.config.episodes == 2
        assert self.config.max_steps_per_episode == 5
        assert self.config.learning_rate == 0.1
        assert self.config.epsilon_start == 1.0
        assert self.config.epsilon_end == 0.01

    def test_training_metrics_initialization(self):
        """Test training metrics initialization."""
        metrics = TrainingMetrics()

        assert metrics.episode_rewards == []
        assert metrics.episode_lengths == []
        assert metrics.epsilon_values == []

    def test_single_episode_execution(self):
        """Test single episode execution."""
        # Create sample market data
        dates = pd.date_range('2023-01-01', periods=10, freq='1H')
        market_data = pd.DataFrame({
            'close': [50000 + i * 100 for i in range(10)],
            'rsi': [50 + i for i in range(10)],
            'macd': [0.0 + i * 0.1 for i in range(10)],
            'bb_position': [0.5 + i * 0.01 for i in range(10)]
        }, index=dates)
        
        market_data_dict = {"BTC/USDT": market_data}
        
        # Initialize agent
        self.trainer.initialize_agent()
        
        # Mock the _run_episode method to return test values
        with patch.object(self.trainer, '_run_episode', new_callable=AsyncMock) as mock_episode:
            mock_episode.return_value = (1.0, 5)
            
            with patch.object(self.trainer, '_evaluate_agent', new_callable=AsyncMock) as mock_eval:
                mock_eval.return_value = {'win_rate': 0.5, 'sharpe_ratio': 1.0, 'max_drawdown': 0.1}
                
                with patch.object(self.trainer, '_save_checkpoint', new_callable=AsyncMock):
                    result = asyncio.run(self.trainer.train(market_data_dict))
                    
                    assert result['status'] == 'completed'
                    assert result['episodes_completed'] == 2

    @pytest.mark.asyncio
    async def test_async_trainer_initialization(self):
        """Test async trainer initialization."""
        assert self.trainer.config == self.config
        assert self.trainer.agent is None  # Not initialized yet
        assert isinstance(self.trainer.metrics, TrainingMetrics)

    @pytest.mark.asyncio
    async def test_async_training_execution(self):
        """Test async training execution."""
        # Create sample market data
        dates = pd.date_range('2023-01-01', periods=10, freq='1H')
        market_data = pd.DataFrame({
            'close': [50000 + i * 100 for i in range(10)],
            'rsi': [50 + i for i in range(10)],
            'macd': [0.0 + i * 0.1 for i in range(10)],
            'bb_position': [0.5 + i * 0.01 for i in range(10)]
        }, index=dates)
        
        market_data_dict = {"BTC/USDT": market_data}
        
        # Mock training methods
        with patch.object(self.trainer, '_run_episode', new_callable=AsyncMock) as mock_episode:
            mock_episode.return_value = (1.0, 5)
            
            with patch.object(self.trainer, '_evaluate_agent', new_callable=AsyncMock) as mock_eval:
                mock_eval.return_value = {'win_rate': 0.5, 'sharpe_ratio': 1.0, 'max_drawdown': 0.1}
                
                with patch.object(self.trainer, '_save_checkpoint', new_callable=AsyncMock):
                    metrics = await self.trainer.train(market_data_dict)

                    assert isinstance(metrics, dict)
                    assert metrics['status'] == 'completed'


class TestIntegration:
    """Integration tests for RL Service components."""

    def test_agent_environment_integration(self):
        """Test integration between agent and environment."""
        state_bins = {'position': 5, 'price': 7, 'trend': 5}
        actions = [TradingAction.BUY, TradingAction.HOLD, TradingAction.SELL]
        
        agent = QLearningAgent(
            state_bins=state_bins,
            actions=actions
        )
        
        # Create sample market data
        dates = pd.date_range('2023-01-01', periods=10, freq='1H')
        market_data = pd.DataFrame({
            'close': [50000 + i * 1000 for i in range(10)],
            'rsi': [50 + i * 2 for i in range(10)],
            'macd': [0.0 + i * 0.1 for i in range(10)],
            'bb_position': [0.5 + i * 0.01 for i in range(10)]
        }, index=dates)
        
        environment = TradingEnvironment(
            market_data=market_data,
            symbol="BTC/USDT",
            initial_balance=1000.0,
            transaction_cost=0.001
        )

        # Simulate a few steps
        for step in range(5):
            state = environment._get_current_state()
            
            # Agent selects action
            action = agent.get_action(state)
            
            # Environment executes action
            next_state, reward, done, info = environment.step(action)

            # Create experience and update agent
            experience = RLExperience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                timestamp=datetime.now()
            )
            
            agent.update_q_value(experience)

            assert isinstance(reward, float)
            assert isinstance(done, bool)

    def test_training_pipeline_integration(self):
        """Test complete training pipeline integration."""
        config = TrainingConfig(episodes=2, max_steps_per_episode=5)
        
        trainer = RLTrainer(config)
        
        # Create sample market data
        dates = pd.date_range('2023-01-01', periods=15, freq='1H')
        market_data = pd.DataFrame({
            'close': [50000 + i * 500 for i in range(15)],
            'rsi': [50 + (i % 10 - 5) for i in range(15)],
            'macd': [0.0 + (i % 5 - 2) * 0.1 for i in range(15)],
            'bb_position': [0.5 + (i % 5 - 2) * 0.02 for i in range(15)]
        }, index=dates)
        
        market_data_dict = {"BTC/USDT": market_data}
        
        # Mock training methods
        with patch.object(trainer, '_run_episode', new_callable=AsyncMock) as mock_episode:
            mock_episode.return_value = (1.0, 5)
            
            with patch.object(trainer, '_evaluate_agent', new_callable=AsyncMock) as mock_eval:
                mock_eval.return_value = {'win_rate': 0.5, 'sharpe_ratio': 1.0, 'max_drawdown': 0.1}
                
                with patch.object(trainer, '_save_checkpoint', new_callable=AsyncMock):
                    metrics = asyncio.run(trainer.train(market_data_dict))

                    assert isinstance(metrics, dict)
                    assert metrics['status'] == 'completed'
                    assert metrics['episodes_completed'] == 2


if __name__ == "__main__":
    pytest.main([__file__])