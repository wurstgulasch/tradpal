"""
Unit tests for the isolated Backtesting Service
Tests the core backtesting logic and service functionality.
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from services.trading_service.backtesting_service.service import BacktestingService


class TestBacktestingService:
    """Test cases for BacktestingService."""

    @pytest.fixture
    def backtesting_service(self):
        """Create a BacktestingService instance for testing."""
        return BacktestingService()

    @pytest.fixture
    def sample_market_data(self):
        """Generate sample market data for testing."""
        dates = pd.date_range('2023-01-01', periods=100, freq='1H')
        np.random.seed(42)

        # Generate realistic OHLC data
        close_prices = np.random.uniform(40000, 60000, 100)
        highs = close_prices * np.random.uniform(1.001, 1.02, 100)
        lows = close_prices * np.random.uniform(0.98, 0.999, 100)
        opens = close_prices * np.random.uniform(0.999, 1.001, 100)
        volumes = np.random.uniform(100, 1000, 100)

        return {
            'timestamp': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': close_prices,
            'volume': volumes
        }

    @pytest.fixture
    def sample_strategy_config(self):
        """Sample strategy configuration."""
        return {
            'name': 'SMA Crossover',
            'type': 'technical',
            'parameters': {
                'fast_period': 10,
                'slow_period': 20,
                'stop_loss': 0.02,
                'take_profit': 0.05
            }
        }

    @pytest.mark.asyncio
    async def test_service_initialization(self, backtesting_service):
        """Test service initialization."""
        await backtesting_service.initialize()

        # Check that service is initialized
        assert backtesting_service.initialized
        assert backtesting_service.metrics is not None

    @pytest.mark.asyncio
    async def test_run_backtest_basic(self, backtesting_service, sample_market_data, sample_strategy_config):
        """Test basic backtest execution."""
        await backtesting_service.initialize()

        config = {
            'initial_capital': 10000,
            'commission': 0.001,
            'position_size': 0.1
        }

        result = await backtesting_service.run_backtest(
            sample_strategy_config,
            sample_market_data,
            config
        )

        # Check result structure
        assert 'performance' in result
        assert 'trades' in result
        assert 'risk_metrics' in result
        assert 'execution_time' in result

        # Check performance metrics
        perf = result['performance']
        assert 'total_return' in perf
        assert 'sharpe_ratio' in perf
        assert 'max_drawdown' in perf
        assert 'win_rate' in perf

        # Check that trades exist
        assert isinstance(result['trades'], list)

    @pytest.mark.asyncio
    async def test_run_backtest_with_invalid_data(self, backtesting_service):
        """Test backtest with invalid data."""
        await backtesting_service.initialize()

        strategy_config = {'name': 'Test Strategy'}
        invalid_data = {}  # Empty data
        config = {'initial_capital': 10000}

        with pytest.raises(ValueError):
            await backtesting_service.run_backtest(strategy_config, invalid_data, config)

    @pytest.mark.asyncio
    async def test_optimize_strategy(self, backtesting_service, sample_market_data):
        """Test strategy optimization."""
        await backtesting_service.initialize()

        strategy_name = 'SMA Crossover'
        param_ranges = {
            'fast_period': [5, 15],
            'slow_period': [15, 25]
        }

        config = {
            'optimization_method': 'grid',
            'max_evaluations': 10
        }

        result = await backtesting_service.optimize_strategy(
            strategy_name,
            param_ranges,
            sample_market_data,
            config
        )

        # Check optimization result structure
        assert 'best_parameters' in result
        assert 'best_score' in result
        assert 'optimization_history' in result
        assert 'trials_run' in result

        # Check best parameters
        best_params = result['best_parameters']
        assert 'fast_period' in best_params
        assert 'slow_period' in best_params

    @pytest.mark.asyncio
    async def test_compare_strategies(self, backtesting_service, sample_market_data):
        """Test strategy comparison."""
        await backtesting_service.initialize()

        strategies = [
            {
                'name': 'Strategy A',
                'type': 'technical',
                'parameters': {'fast_period': 10, 'slow_period': 20}
            },
            {
                'name': 'Strategy B',
                'type': 'technical',
                'parameters': {'fast_period': 5, 'slow_period': 15}
            }
        ]

        config = {'initial_capital': 10000}

        result = await backtesting_service.compare_strategies(
            strategies,
            sample_market_data,
            config
        )

        # Check comparison result structure
        assert 'strategies' in result
        assert 'comparison_metrics' in result
        assert 'rankings' in result

        # Check that all strategies are included
        assert len(result['strategies']) == 2

    @pytest.mark.asyncio
    async def test_performance_metrics_calculation(self, backtesting_service):
        """Test performance metrics calculation."""
        # Create mock trades data
        trades = [
            {'pnl': 100, 'entry_time': datetime.now(), 'exit_time': datetime.now() + timedelta(hours=1)},
            {'pnl': -50, 'entry_time': datetime.now(), 'exit_time': datetime.now() + timedelta(hours=2)},
            {'pnl': 200, 'entry_time': datetime.now(), 'exit_time': datetime.now() + timedelta(hours=3)},
        ]

        capital_history = [10000, 10100, 10050, 10250]

        metrics = backtesting_service._calculate_performance_metrics(trades, capital_history, {})

        # Check required metrics are present
        required_metrics = [
            'total_return', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
            'max_drawdown', 'win_rate', 'profit_factor', 'total_trades'
        ]

        for metric in required_metrics:
            assert metric in metrics

        # Check metric values are reasonable
        assert metrics['total_trades'] == 3
        assert metrics['win_rate'] == 2/3  # 2 winning trades out of 3
        assert metrics['total_return'] > 0

    @pytest.mark.asyncio
    async def test_risk_metrics_calculation(self, backtesting_service):
        """Test risk metrics calculation."""
        # Create mock trades data
        trades = [
            {'pnl': 100, 'entry_time': datetime.now(), 'exit_time': datetime.now() + timedelta(hours=1)},
            {'pnl': -50, 'entry_time': datetime.now(), 'exit_time': datetime.now() + timedelta(hours=2)},
            {'pnl': 200, 'entry_time': datetime.now(), 'exit_time': datetime.now() + timedelta(hours=3)},
        ]

        risk_metrics = backtesting_service._calculate_risk_metrics(pd.DataFrame(), trades, {'initial_capital': 10000})

        # Check required risk metrics
        required_metrics = ['volatility', 'var_95', 'cvar_95', 'max_drawdown', 'beta']

        for metric in required_metrics:
            assert metric in risk_metrics

        # Check values are reasonable
        assert risk_metrics['volatility'] > 0
        assert risk_metrics['var_95'] < 0  # VaR should be negative
        assert risk_metrics['max_drawdown'] >= 0  # Max drawdown should be positive

    @pytest.mark.asyncio
    async def test_get_health_status(self, backtesting_service):
        """Test health status retrieval."""
        await backtesting_service.initialize()

        health = await backtesting_service.get_health_status()

        assert 'status' in health
        assert 'timestamp' in health
        assert 'active_backtests' in health
        assert 'memory_usage' in health

    @pytest.mark.asyncio
    async def test_get_metrics(self, backtesting_service):
        """Test metrics retrieval."""
        await backtesting_service.initialize()

        metrics = await backtesting_service.get_metrics()

        assert 'backtests_completed' in metrics
        assert 'active_backtests' in metrics
        assert 'average_execution_time' in metrics
        assert 'memory_usage' in metrics

    @pytest.mark.asyncio
    async def test_cleanup_old_results(self, backtesting_service):
        """Test cleanup of old results."""
        await backtesting_service.initialize()

        # This should not raise an exception
        await backtesting_service.cleanup_old_results(days=30)

    @pytest.mark.asyncio
    async def test_service_cleanup(self, backtesting_service):
        """Test service cleanup."""
        await backtesting_service.initialize()

        # This should not raise an exception
        await backtesting_service.cleanup()

    @pytest.mark.asyncio
    async def test_concurrent_backtests(self, backtesting_service, sample_market_data, sample_strategy_config):
        """Test running multiple backtests concurrently."""
        await backtesting_service.initialize()

        config = {'initial_capital': 10000}

        # Run multiple backtests concurrently
        tasks = []
        for i in range(3):
            task = backtesting_service.run_backtest(
                sample_strategy_config,
                sample_market_data,
                config
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # Check that all backtests completed
        assert len(results) == 3
        for result in results:
            assert 'performance' in result
            assert 'trades' in result

    @pytest.mark.asyncio
    async def test_walk_forward_optimization(self, backtesting_service, sample_market_data):
        """Test walk-forward optimization."""
        await backtesting_service.initialize()

        strategy_name = 'SMA Crossover'
        param_ranges = {
            'fast_period': [5, 10],
            'slow_period': [15, 20]
        }

        config = {
            'optimization_method': 'walk_forward',
            'train_window': 50,
            'test_window': 25,
            'step_size': 10
        }

        result = await backtesting_service.optimize_strategy(
            strategy_name,
            param_ranges,
            sample_market_data,
            config
        )

        # Check walk-forward specific results
        assert 'optimization_method' in result
        assert result['optimization_method'] == 'grid_search'  # Currently only grid search is implemented

    def test_strategy_validation(self, backtesting_service):
        """Test strategy configuration validation."""
        # Valid strategy
        valid_strategy = {
            'name': 'Test Strategy',
            'type': 'technical',
            'parameters': {'param1': 10}
        }
        assert backtesting_service._validate_strategy_config(valid_strategy)

        # Invalid strategy - missing name
        invalid_strategy = {'type': 'technical'}
        assert not backtesting_service._validate_strategy_config(invalid_strategy)

        # Invalid strategy - invalid type
        invalid_strategy = {'name': 'Test', 'type': 'invalid'}
        assert not backtesting_service._validate_strategy_config(invalid_strategy)

    def test_data_validation(self, backtesting_service, sample_market_data):
        """Test market data validation."""
        # Valid data
        assert backtesting_service._validate_market_data(sample_market_data)

        # Invalid data - missing required fields
        invalid_data = {'timestamp': [], 'close': []}  # Missing open, high, low
        assert not backtesting_service._validate_market_data(invalid_data)

        # Invalid data - empty arrays
        invalid_data = {
            'timestamp': [], 'open': [], 'high': [], 'low': [], 'close': [], 'volume': []
        }
        assert not backtesting_service._validate_market_data(invalid_data)

    @pytest.mark.asyncio
    async def test_error_handling(self, backtesting_service):
        """Test error handling in backtest execution."""
        await backtesting_service.initialize()

        # Test with invalid strategy
        invalid_strategy = {'invalid': 'config'}
        invalid_data = {'timestamp': [1, 2, 3]}

        with pytest.raises(Exception):
            await backtesting_service.run_backtest(invalid_strategy, invalid_data)

    @pytest.mark.asyncio
    async def test_memory_management(self, backtesting_service, sample_market_data, sample_strategy_config):
        """Test memory management during backtesting."""
        await backtesting_service.initialize()

        # Run a backtest and check memory usage doesn't grow excessively
        initial_memory = backtesting_service.metrics.get('memory_usage', 0)

        result = await backtesting_service.run_backtest(
            sample_strategy_config,
            sample_market_data,
            {'initial_capital': 10000}
        )

        final_memory = backtesting_service.metrics.get('memory_usage', 0)

        # Memory should not increase by more than 50MB (reasonable for backtesting)
        memory_increase = final_memory - initial_memory
        assert memory_increase < 50 * 1024 * 1024  # 50MB in bytes