"""
TradPal Backtesting Service Tests
Unit and integration tests for the unified backtesting service
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from services.backtesting_service.orchestrator import BacktestingServiceOrchestrator


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
    yield orch
    await orch.shutdown()


class TestBacktestingServiceOrchestrator:
    """Test the backtesting service orchestrator"""

    @pytest.mark.asyncio
    async def test_initialization(self, orchestrator):
        """Test orchestrator initialization"""
        status = orchestrator.get_service_status()
        assert status["orchestrator_initialized"] is True
        assert status["backtesting_service"] is True
        assert status["optimization_service"] is True

    @pytest.mark.asyncio
    async def test_quick_backtest(self, orchestrator, sample_data, sample_strategy):
        """Test quick backtest functionality"""
        results = await orchestrator.run_quick_backtest(sample_strategy, sample_data)

        assert results["success"] is True
        assert "metrics" in results
        assert "trades" in results
        assert "performance" in results

        # Check metrics structure
        metrics = results["metrics"]
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "win_rate" in metrics

    @pytest.mark.asyncio
    async def test_strategy_optimization(self, orchestrator, sample_data):
        """Test strategy parameter optimization"""
        param_ranges = {
            "fast_period": [5, 10, 15, 20],
            "slow_period": [20, 30, 40, 50],
            "stop_loss": [0.01, 0.02, 0.03],
            "take_profit": [0.03, 0.05, 0.07]
        }

        results = await orchestrator.optimize_strategy(
            "moving_average_crossover", param_ranges, sample_data
        )

        assert results["success"] is True
        assert "best_params" in results
        assert "best_score" in results
        assert "optimization_history" in results

        # Check best parameters
        best_params = results["best_params"]
        assert "fast_period" in best_params
        assert "slow_period" in best_params
        assert best_params["fast_period"] < best_params["slow_period"]  # Fast should be less than slow

    @pytest.mark.asyncio
    async def test_ml_training(self, orchestrator, sample_data, sample_strategy):
        """Test ML model training"""
        ml_config = {
            "model_type": "random_forest",
            "train_split": 0.7,
            "validation_split": 0.2
        }

        results = await orchestrator.train_ml_model(sample_strategy, sample_data, ml_config)

        assert results["success"] is True
        assert "model_info" in results
        assert "training_metrics" in results
        assert "feature_importance" in results

    @pytest.mark.asyncio
    async def test_walk_forward_analysis(self, orchestrator, sample_data):
        """Test walk-forward analysis"""
        param_ranges = {
            "fast_period": [5, 10, 15],
            "slow_period": [20, 30, 40],
            "stop_loss": [0.01, 0.02],
            "take_profit": [0.03, 0.05]
        }

        config = {
            "in_sample_window": 252,  # ~1 year
            "out_sample_window": 21,  # ~1 month
            "step_size": 21
        }

        results = await orchestrator.run_walk_forward_analysis(
            "moving_average_crossover", param_ranges, sample_data, config
        )

        assert results["success"] is True
        assert "total_windows" in results
        assert "average_oos_score" in results
        assert "all_windows" in results
        assert len(results["all_windows"]) > 0

    @pytest.mark.asyncio
    async def test_complete_workflow(self, orchestrator, sample_data, sample_strategy):
        """Test complete backtesting workflow"""
        workflow_config = {
            "enable_ml": False,
            "enable_optimization": True,
            "enable_walk_forward": True,
            "param_ranges": {
                "fast_period": [5, 10, 15],
                "slow_period": [20, 30, 40],
                "stop_loss": [0.01, 0.02],
                "take_profit": [0.03, 0.05]
            },
            "walk_forward_config": {
                "in_sample_window": 252,
                "out_sample_window": 21,
                "step_size": 21
            }
        }

        results = await orchestrator.run_complete_backtesting_workflow(
            sample_strategy, sample_data, workflow_config
        )

        assert results["success"] is True
        assert "phases" in results
        assert "final_recommendation" in results

        # Check phases
        phases = results["phases"]
        assert "initial_backtest" in phases
        assert "optimization" in phases
        assert "walk_forward" in phases

        # Check recommendation
        recommendation = results["final_recommendation"]
        assert "overall_rating" in recommendation
        assert "confidence_level" in recommendation
        assert "recommendations" in recommendation


class TestBacktestingService:
    """Test individual backtesting service components"""

    @pytest.mark.asyncio
    async def test_backtesting_service_basic(self, orchestrator, sample_data, sample_strategy):
        """Test basic backtesting functionality"""
        from services.backtesting_service.backtesting.service import BacktestingService

        service = BacktestingService()
        await service.initialize()

        results = await service.run_backtest(sample_strategy, sample_data)

        assert results["success"] is True
        assert "metrics" in results
        assert "trades" in results

        await service.shutdown()

    @pytest.mark.asyncio
    async def test_optimization_service(self, orchestrator, sample_data):
        """Test optimization service"""
        from services.backtesting_service.optimization.service import OptimizationService

        service = OptimizationService()
        await service.initialize()

        param_ranges = {
            "fast_period": [5, 10, 15],
            "slow_period": [20, 30, 40]
        }

        results = await service.optimize_strategy(
            "moving_average_crossover", param_ranges, sample_data
        )

        assert results["success"] is True
        assert "best_params" in results
        assert "best_score" in results

        await service.shutdown()


# Integration test for the complete service
@pytest.mark.integration
@pytest.mark.asyncio
async def test_service_integration(sample_data, sample_strategy):
    """Integration test for the complete backtesting service"""
    orchestrator = BacktestingServiceOrchestrator()

    try:
        await orchestrator.initialize()

        # Run complete workflow
        workflow_config = orchestrator.get_default_workflow_config()
        workflow_config["param_ranges"] = {
            "fast_period": [5, 10],
            "slow_period": [20, 30],
            "stop_loss": [0.01, 0.02],
            "take_profit": [0.03, 0.05]
        }

        results = await orchestrator.run_complete_backtesting_workflow(
            sample_strategy, sample_data, workflow_config
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