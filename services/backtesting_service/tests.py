"""
TradPal Backtesting Service Tests
Unit and integration tests for the unified backtesting service
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from services.trading_service.backtesting_service.orchestrator import BacktestingServiceOrchestrator


@pytest.fixture
def sample_data():
    """Generate sample OHLC data for testing"""
    dates = pd.date_range(start='2020-01-01', end='2023-01-01', freq='D')
    np.random.seed(42)

    # Generate realistic OHLC data
    n = len(dates)
    close_prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, n)))
    highs = close_prices * (1 + np.random.uniform(0, 0.03, n))
    lows = close_prices * (1 - np.random.uniform(0, 0.03, n))
    opens = close_prices + np.random.normal(0, close_prices * 0.01, n)

    return pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, n)
    }, index=dates)


@pytest.fixture
def sample_strategy():
    """Sample strategy configuration"""
    return {
        "name": "moving_average_crossover",
        "type": "technical",
        "parameters": {
            "fast_period": 10,
            "slow_period": 20,
            "stop_loss": 0.02,
            "take_profit": 0.05
        }
    }


@pytest.fixture
async def orchestrator():
    """Initialize backtesting orchestrator for testing"""
    orch = BacktestingServiceOrchestrator()
    await orch.initialize()
    try:
        yield orch
    finally:
        await orch.shutdown()


@pytest.fixture
def sample_data_sync():
    """Generate sample OHLC data for testing (sync version)"""
    dates = pd.date_range(start='2020-01-01', end='2023-01-01', freq='D')
    np.random.seed(42)

    # Generate realistic OHLC data
    n = len(dates)
    close_prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, n)))
    highs = close_prices * (1 + np.random.uniform(0, 0.03, n))
    lows = close_prices * (1 - np.random.uniform(0, 0.03, n))
    opens = close_prices + np.random.normal(0, close_prices * 0.01, n)

    return pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, n)
    }, index=dates)


@pytest.fixture
def sample_strategy_sync():
    """Sample strategy configuration (sync version)"""
    return {
        "name": "moving_average_crossover",
        "type": "technical",
        "parameters": {
            "short_window": 10,
            "long_window": 20
        }
    }


class TestBacktestingServiceOrchestrator:
    """Test the backtesting service orchestrator"""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test orchestrator initialization"""
        orchestrator = BacktestingServiceOrchestrator()
        await orchestrator.initialize()
        try:
            status = orchestrator.get_service_status()
            assert status["orchestrator_initialized"] is True
            assert status["backtesting_service"] is True
        finally:
            await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_quick_backtest(self, sample_data_sync, sample_strategy_sync):
        """Test quick backtest functionality"""
        orchestrator = BacktestingServiceOrchestrator()
        await orchestrator.initialize()
        try:
            results = await orchestrator.run_quick_backtest(sample_strategy_sync, sample_data_sync)

            assert results["success"] is True
            assert "metrics" in results
            assert "results" in results

            # Check metrics structure
            metrics = results["metrics"]
            assert "total_return" in metrics
            assert "sharpe_ratio" in metrics
            assert "max_drawdown" in metrics
            assert "total_trades" in metrics
        finally:
            await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_strategy_optimization(self, sample_data_sync):
        """Test strategy parameter optimization"""
        orchestrator = BacktestingServiceOrchestrator()
        await orchestrator.initialize()
        try:
            param_ranges = {
                "short_window": [5, 10, 15, 20],
                "long_window": [20, 30, 40, 50]
            }

            results = await orchestrator.optimize_strategy(
                "moving_average_crossover", param_ranges, sample_data_sync
            )

            assert results["strategy"] == "moving_average_crossover"
            assert "best_params" in results
            assert "best_score" in results
            assert "optimization_method" in results
        finally:
            await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_ml_training(self, sample_data_sync, sample_strategy_sync):
        """Test ML model training"""
        orchestrator = BacktestingServiceOrchestrator()
        await orchestrator.initialize()
        try:
            ml_config = {
                "model_type": "random_forest",
                "optimize_hyperparams": False
            }

            results = await orchestrator.train_ml_model(sample_strategy_sync, sample_data_sync, ml_config)

            assert results["success"] is True
            assert "model_name" in results
            assert "performance" in results
            assert "feature_importance" in results
        finally:
            await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_walk_forward_analysis(self, sample_data_sync):
        """Test walk-forward analysis"""
        orchestrator = BacktestingServiceOrchestrator()
        await orchestrator.initialize()
        try:
            param_ranges = {
                "short_window": [5, 10, 15],
                "long_window": [20, 30, 40]
            }

            config = {
                "in_sample_window": 252,  # ~1 year
                "out_sample_window": 21,  # ~1 month
                "step_size": 21
            }

            results = await orchestrator.run_walk_forward_analysis(
                "moving_average_crossover", param_ranges, sample_data_sync, config
            )

            assert "symbol" in results
            assert "timeframe" in results
            assert "total_windows" in results
            assert "results" in results
            assert "analysis" in results
        finally:
            await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_complete_workflow(self, sample_data_sync, sample_strategy_sync):
        """Test complete backtesting workflow"""
        orchestrator = BacktestingServiceOrchestrator()
        await orchestrator.initialize()
        try:
            workflow_config = {
                "enable_ml": False,
                "enable_optimization": True,
                "enable_walk_forward": True,
                "param_ranges": {
                    "short_window": [5, 10, 15],
                    "long_window": [20, 30, 40]
                },
                "walk_forward_config": {
                    "in_sample_window": 252,
                    "out_sample_window": 21,
                    "step_size": 21
                }
            }

            results = await orchestrator.run_complete_backtesting_workflow(
                sample_strategy_sync, sample_data_sync, workflow_config
            )

            assert results["success"] is True
            assert "phases" in results
            assert "final_recommendation" in results

            # Check phases
            phases = results["phases"]
            assert "initial_backtest" in phases
            assert "optimization" in phases
            assert "walk_forward" in phases
        finally:
            await orchestrator.shutdown()


class TestBacktestingService:
    """Test individual backtesting service components"""

    @pytest.mark.asyncio
    async def test_backtesting_service_basic(self, sample_data_sync, sample_strategy_sync):
        """Test basic backtesting functionality"""
        from services.trading_service.backtesting_service.service import BacktestingService

        service = BacktestingService()
        await service.initialize()

        try:
            results = await service.run_backtest(sample_strategy_sync, sample_data_sync)

            assert results["success"] is True
            assert "metrics" in results
            assert "results" in results
        finally:
            await service.shutdown()

    @pytest.mark.asyncio
    async def test_optimization_service(self, sample_data_sync):
        """Test optimization service"""
        from services.trading_service.backtesting_service.service import BacktestingService

        service = BacktestingService()
        await service.initialize()

        try:
            param_ranges = {
                "short_window": [5, 10, 15],
                "long_window": [20, 30, 40]
            }

            results = await service.optimize_strategy(
                "moving_average_crossover", param_ranges
            )

            assert results["strategy"] == "moving_average_crossover"
            assert "best_params" in results
            assert "best_score" in results
        finally:
            await service.shutdown()


# Integration test for the complete service
@pytest.mark.integration
@pytest.mark.asyncio
async def test_service_integration(sample_data_sync, sample_strategy_sync):
    """Integration test for the complete backtesting service"""
    orchestrator = BacktestingServiceOrchestrator()

    try:
        await orchestrator.initialize()

        # Run complete workflow
        workflow_config = orchestrator.get_default_workflow_config()
        workflow_config["param_ranges"] = {
            "short_window": [5, 10],
            "long_window": [20, 30]
        }

        results = await orchestrator.run_complete_backtesting_workflow(
            sample_strategy_sync, sample_data_sync, workflow_config
        )

        assert results["success"] is True
        assert len(results["phases"]) >= 2  # At least initial backtest and optimization

        print(f"Integration test completed successfully")
        print(f"Strategy: {results['strategy']}")
        print(f"Final recommendation: {results['final_recommendation']['overall_rating']}")

    finally:
        await orchestrator.shutdown()


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])