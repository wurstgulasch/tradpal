#!/usr/bin/env python3
"""
Discovery Service Tests - Unit tests for genetic algorithm optimization.

Tests cover:
- Service initialization
- Optimization execution
- Configuration conversion
- Event system integration
- Error handling
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Import service
from services.discovery_service.service import (
    DiscoveryService,
    OptimizationResult,
    IndividualResult,
    EventSystem,
    Event
)

class TestDiscoveryService:
    """Test cases for DiscoveryService."""

    @pytest.fixture
    def event_system(self):
        """Create mock event system."""
        return EventSystem()

    @pytest.fixture
    def discovery_service(self, event_system):
        """Create discovery service instance."""
        with patch('services.discovery_service.service.DEAP_AVAILABLE', True):
            service = DiscoveryService(event_system=event_system)
            return service

    def test_service_initialization(self, discovery_service):
        """Test service initializes correctly."""
        assert discovery_service is not None
        assert discovery_service.event_system is not None
        assert hasattr(discovery_service, 'INDICATOR_COMBINATIONS')
        assert len(discovery_service.INDICATOR_COMBINATIONS) > 0

    def test_individual_to_config_conversion(self, discovery_service):
        """Test conversion from GA individual to config dict."""
        # Create test individual
        individual = [0, 9, 21, 14, 30, 70, 20, 2.0, 14, 12, 26, 9, 14, 3, 14, 20]

        config = discovery_service._individual_to_config(individual)

        assert 'ema' in config
        assert 'rsi' in config
        assert 'bb' in config
        assert config['ema']['periods'] == [9, 21]
        assert config['rsi']['period'] == 14
        assert config['bb']['period'] == 20

    @pytest.mark.asyncio
    async def test_run_optimization_async_basic(self, discovery_service):
        """Test basic optimization run."""
        with patch.object(discovery_service, '_run_optimization_sync') as mock_run:
            # Mock optimization results
            mock_result = IndividualResult(
                config={'test': 'config'},
                fitness=0.85,
                pnl=15.5,
                win_rate=65.0,
                sharpe_ratio=1.2,
                max_drawdown=8.5,
                total_trades=150,
                evaluation_time=2.5,
                backtest_duration_days=365
            )
            mock_run.return_value = [mock_result]

            result = await discovery_service.run_optimization_async(
                optimization_id="test_opt_001",
                symbol="BTC/USDT",
                timeframe="1d",
                start_date="2024-01-01",
                end_date="2024-12-31",
                population_size=20,
                generations=5
            )

            assert result['success'] is True
            assert result['optimization_id'] == "test_opt_001"
            assert result['best_fitness'] == 0.85
            assert 'best_config' in result

    @pytest.mark.asyncio
    async def test_get_optimization_status(self, discovery_service):
        """Test getting optimization status."""
        # Create a test result
        test_result = OptimizationResult(
            optimization_id="test_opt_001",
            symbol="BTC/USDT",
            timeframe="1d",
            start_date="2024-01-01",
            end_date="2024-12-31",
            population_size=50,
            generations=20,
            best_fitness=0.85,
            best_config={'test': 'config'},
            total_evaluations=1000,
            duration_seconds=120.5,
            status='completed'
        )

        discovery_service.optimization_results["test_opt_001"] = test_result

        status = await discovery_service.get_optimization_status("test_opt_001")

        assert status['optimization_id'] == "test_opt_001"
        assert status['status'] == 'completed'
        assert status['best_fitness'] == 0.85

    @pytest.mark.asyncio
    async def test_get_optimization_status_not_found(self, discovery_service):
        """Test getting status for non-existent optimization."""
        status = await discovery_service.get_optimization_status("non_existent")

        assert 'error' in status
        assert status['error'] == "Optimization not found"

    @pytest.mark.asyncio
    async def test_list_active_optimizations(self, discovery_service):
        """Test listing active optimizations."""
        # Create running optimization
        running_result = OptimizationResult(
            optimization_id="running_opt",
            symbol="BTC/USDT",
            timeframe="1d",
            start_date="2024-01-01",
            end_date="2024-12-31",
            population_size=50,
            generations=20,
            best_fitness=0.0,
            best_config={},
            total_evaluations=0,
            duration_seconds=0.0,
            status='running'
        )

        # Create completed optimization
        completed_result = OptimizationResult(
            optimization_id="completed_opt",
            symbol="BTC/USDT",
            timeframe="1d",
            start_date="2024-01-01",
            end_date="2024-12-31",
            population_size=50,
            generations=20,
            best_fitness=0.85,
            best_config={'test': 'config'},
            total_evaluations=1000,
            duration_seconds=120.5,
            status='completed'
        )

        discovery_service.optimization_results["running_opt"] = running_result
        discovery_service.optimization_results["completed_opt"] = completed_result

        active = await discovery_service.list_active_optimizations()

        assert len(active) == 1
        assert active[0]['optimization_id'] == "running_opt"
        assert active[0]['status'] == 'running'

    @pytest.mark.asyncio
    async def test_cancel_optimization(self, discovery_service):
        """Test cancelling an optimization."""
        # Create mock task
        mock_task = Mock()
        discovery_service.active_optimizations["test_opt"] = mock_task

        # Create optimization result
        test_result = OptimizationResult(
            optimization_id="test_opt",
            symbol="BTC/USDT",
            timeframe="1d",
            start_date="2024-01-01",
            end_date="2024-12-31",
            population_size=50,
            generations=20,
            best_fitness=0.0,
            best_config={},
            total_evaluations=0,
            duration_seconds=0.0,
            status='running'
        )
        discovery_service.optimization_results["test_opt"] = test_result

        result = await discovery_service.cancel_optimization("test_opt")

        assert result['success'] is True
        assert result['optimization_id'] == "test_opt"
        assert mock_task.cancel.called
        assert test_result.status == 'cancelled'

    def test_remove_duplicate_configs(self, discovery_service):
        """Test removing duplicate configurations."""
        # Create test results with some duplicates
        results = [
            IndividualResult(
                config={'ema': {'periods': [9, 21]}, 'rsi': {'period': 14}},
                fitness=0.8
            ),
            IndividualResult(
                config={'ema': {'periods': [9, 21]}, 'rsi': {'period': 14}},  # Duplicate
                fitness=0.75
            ),
            IndividualResult(
                config={'ema': {'periods': [12, 26]}, 'rsi': {'period': 14}},
                fitness=0.85
            )
        ]

        unique_results = discovery_service._remove_duplicate_configs(results, max_results=10)

        assert len(unique_results) == 2  # Should remove one duplicate
        # Should keep the one with higher fitness for duplicates
        assert unique_results[0].fitness == 0.85  # Best overall
        assert unique_results[1].fitness == 0.8   # Best of duplicates

    @pytest.mark.asyncio
    async def test_event_system_integration(self, discovery_service):
        """Test event system integration."""
        events_received = []

        async def event_handler(event):
            events_received.append(event)

        # Subscribe to events
        discovery_service.event_system.subscribe("discovery.optimization.started", event_handler)
        discovery_service.event_system.subscribe("discovery.optimization.completed", event_handler)

        # Mock the optimization to trigger events
        with patch.object(discovery_service, '_run_optimization_sync') as mock_run:
            mock_result = IndividualResult(
                config={'test': 'config'},
                fitness=0.85,
                pnl=15.5,
                win_rate=65.0,
                sharpe_ratio=1.2,
                max_drawdown=8.5,
                total_trades=150,
                evaluation_time=2.5,
                backtest_duration_days=365
            )
            mock_run.return_value = [mock_result]

            await discovery_service.run_optimization_async(
                optimization_id="event_test_opt",
                symbol="BTC/USDT",
                timeframe="1d",
                start_date="2024-01-01",
                end_date="2024-12-31"
            )

            # Wait a bit for events to be processed
            await asyncio.sleep(0.1)

            # Should have received start and completion events
            assert len(events_received) >= 2
            event_types = [event.type for event in events_received]
            assert "discovery.optimization.started" in event_types
            assert "discovery.optimization.completed" in event_types

class TestEventSystem:
    """Test cases for EventSystem."""

    @pytest.fixture
    def event_system(self):
        return EventSystem()

    @pytest.mark.asyncio
    async def test_event_subscription_and_publishing(self, event_system):
        """Test event subscription and publishing."""
        events_received = []

        async def handler(event):
            events_received.append(event)

        # Subscribe to event
        event_system.subscribe("test.event", handler)

        # Publish event
        test_event = Event(type="test.event", data={"test": "data"})
        await event_system.publish(test_event)

        # Wait for async processing
        await asyncio.sleep(0.1)

        assert len(events_received) == 1
        assert events_received[0].type == "test.event"
        assert events_received[0].data == {"test": "data"}

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self, event_system):
        """Test multiple subscribers to the same event."""
        events_received_1 = []
        events_received_2 = []

        async def handler1(event):
            events_received_1.append(event)

        async def handler2(event):
            events_received_2.append(event)

        # Subscribe both handlers
        event_system.subscribe("test.event", handler1)
        event_system.subscribe("test.event", handler2)

        # Publish event
        test_event = Event(type="test.event", data={"test": "data"})
        await event_system.publish(test_event)

        # Wait for async processing
        await asyncio.sleep(0.1)

        assert len(events_received_1) == 1
        assert len(events_received_2) == 1

class TestOptimizationResult:
    """Test cases for OptimizationResult dataclass."""

    def test_optimization_result_creation(self):
        """Test creating an OptimizationResult."""
        result = OptimizationResult(
            optimization_id="test_opt",
            symbol="BTC/USDT",
            timeframe="1d",
            start_date="2024-01-01",
            end_date="2024-12-31",
            population_size=50,
            generations=20,
            best_fitness=0.85,
            best_config={'ema': {'periods': [9, 21]}},
            total_evaluations=1000,
            duration_seconds=120.5,
            status='completed'
        )

        assert result.optimization_id == "test_opt"
        assert result.best_fitness == 0.85
        assert result.status == 'completed'

    def test_optimization_result_to_dict(self):
        """Test converting OptimizationResult to dict."""
        result = OptimizationResult(
            optimization_id="test_opt",
            symbol="BTC/USDT",
            timeframe="1d",
            start_date="2024-01-01",
            end_date="2024-12-31",
            population_size=50,
            generations=20,
            best_fitness=0.85,
            best_config={'ema': {'periods': [9, 21]}},
            total_evaluations=1000,
            duration_seconds=120.5,
            status='completed'
        )

        result_dict = result.to_dict()

        assert result_dict['optimization_id'] == "test_opt"
        assert result_dict['best_fitness'] == 0.85
        assert 'created_at' in result_dict

class TestIndividualResult:
    """Test cases for IndividualResult dataclass."""

    def test_individual_result_creation(self):
        """Test creating an IndividualResult."""
        result = IndividualResult(
            config={'ema': {'periods': [9, 21]}},
            fitness=0.85,
            pnl=15.5,
            win_rate=65.0,
            sharpe_ratio=1.2,
            max_drawdown=8.5,
            total_trades=150,
            evaluation_time=2.5,
            backtest_duration_days=365
        )

        assert result.fitness == 0.85
        assert result.pnl == 15.5
        assert result.total_trades == 150

# Integration tests
class TestDiscoveryServiceIntegration:
    """Integration tests for DiscoveryService."""

    @pytest.fixture
    def discovery_service(self):
        """Create discovery service for integration tests."""
        with patch('services.discovery_service.service.DEAP_AVAILABLE', True):
            service = DiscoveryService()
            return service

    @pytest.mark.asyncio
    async def test_full_optimization_workflow(self, discovery_service):
        """Test complete optimization workflow."""
        # This is a more comprehensive integration test
        # Mock the data loading and backtesting
        with patch.object(discovery_service, '_load_historical_data') as mock_load, \
             patch.object(discovery_service, '_evaluate_config_single') as mock_evaluate:

            # Mock data loading
            mock_data = Mock()
            mock_data.empty = False
            mock_data.index.min.return_value = "2024-01-01"
            mock_data.index.max.return_value = "2024-12-31"
            mock_load.return_value = mock_data

            # Mock evaluation to return good fitness for first config
            mock_evaluate.return_value = (0.85, {
                'total_pnl': 15.5,
                'win_rate': 65.0,
                'sharpe_ratio': 1.2,
                'max_drawdown': 8.5,
                'total_trades': 150
            })

            # Run optimization
            result = await discovery_service.run_optimization_async(
                optimization_id="integration_test_opt",
                symbol="BTC/USDT",
                timeframe="1d",
                start_date="2024-01-01",
                end_date="2024-12-31",
                population_size=10,  # Small for testing
                generations=2       # Small for testing
            )

            assert result['success'] is True
            assert result['optimization_id'] == "integration_test_opt"
            assert 'best_fitness' in result
            assert 'best_config' in result

if __name__ == "__main__":
    pytest.main([__file__])