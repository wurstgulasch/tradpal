"""
Tests for Backtesting Service.

Comprehensive test suite covering:
- Single backtest execution
- Multi-symbol backtesting
- Multi-model comparison
- Walk-forward optimization
- Event integration
- Error handling
"""

import asyncio
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from services.backtesting_service import BacktestingService, AsyncBacktester
# Event system - using mock implementation for now
EVENT_SYSTEM_AVAILABLE = False

class Event:
    def __init__(self, type: str, data: dict):
        self.type = type
        self.data = data

class EventSystem:
    def __init__(self):
        self.handlers = {}

    def subscribe(self, event_type: str, handler):
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)

    async def publish(self, event: Event):
        # Mock implementation - just call handlers synchronously
        if event.type in self.handlers:
            for handler in self.handlers[event.type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    print(f"Error in event handler: {e}")


class TestBacktestingService:
    """Test cases for BacktestingService class."""

    @pytest.fixture
    def event_system(self):
        """Mock event system for testing."""
        return Mock(spec=EventSystem)

    @pytest.fixture
    def cache(self):
        """Mock cache for testing."""
        return Mock()

class TestBacktestingService:
    """Test cases for BacktestingService class."""

    @pytest.fixture
    def event_system(self):
        """Mock event system for testing."""
        return Mock()

    @pytest.fixture
    def service(self, event_system):
        """Create BacktestingService instance for testing."""
        cache = Mock()
        return BacktestingService(event_system=event_system, cache=cache)

    @pytest.fixture
    def backtester(self):
        """Create AsyncBacktester instance for testing."""
        return AsyncBacktester(
            symbol="BTC/USDT",
            exchange="kraken",
            timeframe="1d",
            start_date="2024-01-01",
            end_date="2024-04-09",
            initial_capital=10000.0
        )

    @pytest.mark.asyncio
    async def test_initialization(self, service, event_system):
        """Test service initialization."""
        assert service.event_system == event_system
        assert service.active_backtests == {}
        assert service.backtest_results == {}

        # Check that event handlers are registered
        event_system.subscribe.assert_any_call("backtest.request", service._handle_backtest_request)
        event_system.subscribe.assert_any_call("backtest.multi_symbol.request", service._handle_multi_symbol_request)
        event_system.subscribe.assert_any_call("backtest.multi_model.request", service._handle_multi_model_request)
        event_system.subscribe.assert_any_call("backtest.walk_forward.request", service._handle_walk_forward_request)

    @pytest.mark.asyncio
    async def test_run_backtest_success(self, backtester):
        """Test successful single backtest execution."""
        # Mock data fetching and signal preparation
        with patch.object(backtester, '_fetch_data_async', return_value=pd.DataFrame({
            'close': [100, 101, 102],
            'Buy_Signal': [1, 0, 0],
            'Sell_Signal': [0, 0, 1],
            'Position_Size_Absolute': [1000, 1000, 1000],
            'Stop_Loss_Buy': [98, 99, 100],
            'Stop_Loss_Sell': [102, 103, 104],
            'Take_Profit_Buy': [102, 103, 104],
            'Take_Profit_Sell': [98, 99, 100],
            'ATR': [1, 1, 1]
        })):
            with patch.object(backtester, '_prepare_traditional_signals_async') as mock_prepare:
                mock_prepare.return_value = pd.DataFrame({
                    'close': [100, 101, 102],
                    'Buy_Signal': [1, 0, 0],
                    'Sell_Signal': [0, 0, 1],
                    'Position_Size_Absolute': [1000, 1000, 1000],
                    'Stop_Loss_Buy': [98, 99, 100],
                    'Stop_Loss_Sell': [102, 103, 104],
                    'Take_Profit_Buy': [102, 103, 104],
                    'Take_Profit_Sell': [98, 99, 100],
                    'ATR': [1, 1, 1]
                })

                result = await backtester.run_backtest_async(strategy="traditional")

                assert result["success"] is True
                assert "metrics" in result
                assert "trades" in result
                assert "trades_count" in result
                assert result["trades_count"] > 0

    @pytest.mark.asyncio
    async def test_run_backtest_failure(self, backtester):
        """Test backtest execution failure."""
        with patch.object(backtester, '_fetch_data_async', return_value=pd.DataFrame()):
            result = await backtester.run_backtest_async()

            assert result["success"] is False
            assert "No data available" in result["error"]

    @pytest.mark.asyncio
    async def test_multi_symbol_backtest(self, service):
        """Test multi-symbol backtest execution."""
        symbols = ["BTC/USDT", "ETH/USDT"]

        # Mock individual backtests
        with patch.object(service, 'run_backtest_async') as mock_run_backtest:
            mock_run_backtest.side_effect = [
                {
                    "success": True,
                    "metrics": {"total_pnl": 1000.0, "win_rate": 60.0},
                    "trades_count": 5
                },
                {
                    "success": True,
                    "metrics": {"total_pnl": 500.0, "win_rate": 55.0},
                    "trades_count": 3
                }
            ]

            result = await service.run_multi_symbol_backtest_async(
                symbols=symbols,
                backtest_id="test_multi_symbol"
            )

            assert result["backtest_id"] == "test_multi_symbol"
            assert result["symbols_tested"] == symbols
            assert len(result["successful_backtests"]) == 2
            assert "aggregated_metrics" in result

            # Verify individual backtests were called
            assert mock_run_backtest.call_count == 2

    @pytest.mark.asyncio
    async def test_multi_model_backtest(self, service):
        """Test multi-model backtest execution."""
        models = ["traditional_ml", "lstm"]

        with patch.object(service, 'run_backtest_async') as mock_run_backtest:
            mock_run_backtest.side_effect = [
                {
                    "success": True,
                    "metrics": {"sharpe_ratio": 1.5, "total_pnl": 1000.0}
                },
                {
                    "success": True,
                    "metrics": {"sharpe_ratio": 1.8, "total_pnl": 1200.0}
                }
            ]

            result = await service.run_multi_model_backtest_async(
                symbol="BTC/USDT",
                models_to_test=models,
                backtest_id="test_multi_model"
            )

            assert result["backtest_id"] == "test_multi_model"
            assert result["models_tested"] == models
            assert "comparison" in result
            assert "best_model" in result["comparison"]

    @pytest.mark.asyncio
    async def test_walk_forward_optimization(self, service):
        """Test walk-forward optimization."""
        parameter_grid = {"ema_short": [5, 9, 12]}

        with patch('src.walk_forward_optimizer.WalkForwardOptimizer') as mock_optimizer_class:
            mock_optimizer = Mock()
            mock_optimizer.optimize_strategy_parameters = Mock(return_value={
                "success": True,
                "best_parameters": {"ema_short": 9},
                "best_score": 1.5
            })
            mock_optimizer_class.return_value = mock_optimizer

            with patch.object(service, 'run_backtest_async') as mock_run_backtest:
                mock_run_backtest.return_value = {"success": True, "metrics": {"total_pnl": 1000.0}}

                result = await service.run_walk_forward_backtest_async(
                    parameter_grid=parameter_grid,
                    backtest_id="test_walk_forward"
                )

                assert "optimization_results" in result
                assert "final_backtest" in result
                assert result["optimization_results"]["best_parameters"]["ema_short"] == 9

    @pytest.mark.asyncio
    async def test_get_backtest_status(self, service):
        """Test getting backtest status."""
        # Test active backtest
        service.active_backtests["test_active"] = {
            "status": "running",
            "start_time": datetime.now()
        }

        status = await service.get_backtest_status("test_active")
        assert status["status"] == "running"

        # Test completed backtest
        service.backtest_results["test_completed"] = {"success": True, "metrics": {"pnl": 100.0}}

        status = await service.get_backtest_status("test_completed")
        assert status["status"] == "completed"
        assert status["result"]["success"] is True

        # Test not found
        status = await service.get_backtest_status("not_found")
        assert status["status"] == "not_found"

    @pytest.mark.asyncio
    async def test_cleanup_completed_backtests(self, service):
        """Test cleanup of old completed backtests."""
        old_time = datetime.now() - timedelta(hours=25)

        service.active_backtests = {
            "old_completed": {
                "status": "completed",
                "end_time": old_time
            },
            "recent_completed": {
                "status": "completed",
                "end_time": datetime.now()
            },
            "still_running": {
                "status": "running",
                "start_time": datetime.now()
            }
        }

        await service.cleanup_completed_backtests(max_age_hours=24)

        # Old completed backtest should be removed
        assert "old_completed" not in service.active_backtests
        # Recent completed should remain
        assert "recent_completed" in service.active_backtests
        # Running should remain
        assert "still_running" in service.active_backtests


class TestAsyncBacktester:
    """Test cases for AsyncBacktester class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1d')
        np.random.seed(42)

        data = pd.DataFrame({
            'timestamp': dates,
            'open': 50000 + np.random.normal(0, 1000, 100),
            'high': 51000 + np.random.normal(0, 1000, 100),
            'low': 49000 + np.random.normal(0, 1000, 100),
            'close': 50000 + np.random.normal(0, 1000, 100),
            'volume': np.random.randint(1000, 10000, 100)
        })

        # Add trend
        trend = np.linspace(0, 5000, 100)
        data['close'] = data['close'] + trend
        data.set_index('timestamp', inplace=True)

        return data

    @pytest.fixture
    def backtester(self):
        """Create AsyncBacktester instance for testing."""
        return AsyncBacktester(
            symbol="BTC/USDT",
            exchange="kraken",
            timeframe="1d",
            start_date="2024-01-01",
            end_date="2024-04-09",
            initial_capital=10000.0
        )

    @pytest.mark.asyncio
    async def test_initialization(self, backtester):
        """Test backtester initialization."""
        assert backtester.symbol == "BTC/USDT"
        assert backtester.exchange == "kraken"
        assert backtester.timeframe == "1d"
        assert backtester.initial_capital == 10000.0
        assert backtester.current_capital == 10000.0
        assert backtester.commission == 0.001

    @pytest.mark.asyncio
    async def test_run_backtest_traditional(self, backtester, sample_data):
        """Test running traditional strategy backtest."""
        # Mock data fetching
        with patch.object(backtester, '_fetch_data_async', return_value=sample_data):
            # Mock signal preparation
            with patch.object(backtester, '_prepare_traditional_signals_async') as mock_prepare:
                mock_prepare.return_value = sample_data.copy()

                # Add required columns for trade simulation
                test_data = sample_data.copy()
                test_data['Buy_Signal'] = np.random.choice([0, 1], len(test_data))
                test_data['Sell_Signal'] = np.random.choice([0, 1], len(test_data))
                test_data['Position_Size_Absolute'] = 1000.0
                test_data['Stop_Loss_Buy'] = test_data['close'] * 0.98
                test_data['Stop_Loss_Sell'] = test_data['close'] * 1.02
                test_data['Take_Profit_Buy'] = test_data['close'] * 1.02
                test_data['Take_Profit_Sell'] = test_data['close'] * 0.98
                test_data['ATR'] = 1000.0  # Mock ATR

                mock_prepare.return_value = test_data

                result = await backtester.run_backtest_async(strategy="traditional")

                assert "success" in result
                assert "metrics" in result
                assert "trades" in result
                assert "trades_count" in result

    @pytest.mark.asyncio
    async def test_run_backtest_ml_enhanced(self, backtester, sample_data):
        """Test running ML-enhanced strategy backtest."""
        with patch.object(backtester, '_fetch_data_async', return_value=sample_data):
            with patch.object(backtester, '_prepare_ml_enhanced_signals_async') as mock_prepare:
                test_data = sample_data.copy()
                test_data['Buy_Signal'] = np.random.choice([0, 1], len(test_data))
                test_data['Sell_Signal'] = np.random.choice([0, 1], len(test_data))
                test_data['Position_Size_Absolute'] = 1000.0
                test_data['ATR'] = 1000.0

                mock_prepare.return_value = test_data

                result = await backtester.run_backtest_async(strategy="ml_enhanced")

                assert "success" in result
                assert result["success"] is True

    @pytest.mark.asyncio
    async def test_run_backtest_no_data(self, backtester):
        """Test backtest with no data available."""
        with patch.object(backtester, '_fetch_data_async', return_value=pd.DataFrame()):
            result = await backtester.run_backtest_async()

            assert result["success"] is False
            assert "No data available" in result["error"]

    @pytest.mark.asyncio
    async def test_calculate_metrics_no_trades(self, backtester):
        """Test metrics calculation with no trades."""
        backtester.trades = []

        metrics = await backtester._calculate_metrics_async()

        assert "error" in metrics
        assert "No trades executed" in metrics["error"]

    @pytest.mark.asyncio
    async def test_calculate_metrics_with_trades(self, backtester):
        """Test metrics calculation with trades."""
        # Create sample trades
        backtester.trades = [
            {
                'pnl': 100.0,
                'entry_commission': 5.0,
                'exit_commission': 5.0,
                'status': 'closed'
            },
            {
                'pnl': -50.0,
                'entry_commission': 5.0,
                'exit_commission': 5.0,
                'status': 'closed'
            },
            {
                'pnl': 200.0,
                'entry_commission': 5.0,
                'exit_commission': 5.0,
                'status': 'closed'
            }
        ]
        backtester.initial_capital = 10000.0
        backtester.current_capital = 10250.0
        backtester.start_date = pd.to_datetime("2024-01-01")
        backtester.end_date = pd.to_datetime("2024-04-01")

        metrics = await backtester._calculate_metrics_async()

        assert 'total_trades' in metrics
        assert 'win_rate' in metrics
        assert 'total_pnl' in metrics
        assert 'sharpe_ratio' in metrics
        assert metrics['total_trades'] == 3
        assert metrics['winning_trades'] == 2
        assert metrics['losing_trades'] == 1


class TestEventHandling:
    """Test event handling in BacktestingService."""

    @pytest.fixture
    def service(self):
        """Create service for event testing."""
        event_system = EventSystem()
        cache = Mock()
        return BacktestingService(event_system=event_system, cache=cache)

    @pytest.mark.asyncio
    async def test_handle_backtest_request(self, service):
        """Test handling backtest request event."""
        with patch.object(service, 'run_backtest_async') as mock_run:
            mock_run.return_value = {"success": True, "metrics": {"pnl": 100.0}}

            event = Event(
                type="backtest.request",
                data={
                    "backtest_id": "event_test",
                    "symbol": "BTC/USDT",
                    "timeframe": "1d"
                }
            )

            await service._handle_backtest_request(event)

            # Verify backtest was called
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert call_args[1]["symbol"] == "BTC/USDT"
            assert call_args[1]["timeframe"] == "1d"
            assert call_args[1]["backtest_id"] == "event_test"

    @pytest.mark.asyncio
    async def test_handle_multi_symbol_request(self, service):
        """Test handling multi-symbol backtest request event."""
        with patch.object(service, 'run_multi_symbol_backtest_async') as mock_run:
            mock_run.return_value = {"success": True}

            event = Event(
                type="backtest.multi_symbol.request",
                data={
                    "backtest_id": "multi_symbol_event",
                    "symbols": ["BTC/USDT", "ETH/USDT"],
                    "timeframe": "1h"
                }
            )

            await service._handle_multi_symbol_request(event)

            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert call_args[1]["symbols"] == ["BTC/USDT", "ETH/USDT"]
            assert call_args[1]["timeframe"] == "1h"