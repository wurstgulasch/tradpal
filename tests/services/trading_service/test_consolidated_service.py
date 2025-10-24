"""
Test script for Consolidated Trading Service functionality.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from services.trading_service.orchestrator import TradingServiceOrchestrator


@pytest.mark.asyncio
async def test_trading_service_orchestrator_initialization():
    """Test that TradingServiceOrchestrator can be initialized."""
    orchestrator = TradingServiceOrchestrator()

    # Mock the service clients to avoid network calls
    mock_ai_client = AsyncMock()
    mock_ai_client.authenticate = AsyncMock(return_value=True)

    mock_backtesting_client = AsyncMock()
    mock_backtesting_client.initialize = AsyncMock(return_value=True)
    mock_backtesting_client.authenticate = AsyncMock(return_value=True)

    mock_live_client = AsyncMock()

    # Patch the client classes to return our mocks
    with patch('services.trading_service.orchestrator.MLTrainingServiceClient', return_value=mock_ai_client), \
         patch('services.trading_service.orchestrator.BacktestingServiceClient', return_value=mock_backtesting_client), \
         patch('services.trading_service.orchestrator.TradingBotLiveServiceClient', return_value=mock_live_client), \
         patch('services.trading_service.orchestrator.ReinforcementLearningServiceClient', return_value=AsyncMock()), \
         patch('services.trading_service.orchestrator.MarketRegimeServiceClient', return_value=AsyncMock()), \
         patch('services.trading_service.orchestrator.RiskManagementServiceClient', return_value=AsyncMock()), \
         patch('services.trading_service.orchestrator.TradingExecutionServiceClient', return_value=AsyncMock()):

        # Test initialization
        await orchestrator.initialize()

        assert orchestrator.is_initialized is True
        assert 'ml_training' in orchestrator.services
        assert 'backtesting' in orchestrator.services
        assert 'live' in orchestrator.services

        # Verify service methods were called
        mock_ai_client.authenticate.assert_called_once()
        mock_backtesting_client.initialize.assert_called_once()
        mock_backtesting_client.authenticate.assert_called_once()


@pytest.mark.asyncio
async def test_trading_service_orchestrator_health_check():
    """Test health check functionality."""
    orchestrator = TradingServiceOrchestrator()

    # Mock services
    mock_ai_client = AsyncMock()
    mock_ai_client.health_check = AsyncMock(return_value={'status': 'healthy'})

    mock_backtesting_client = AsyncMock()
    mock_backtesting_client.health_check = AsyncMock(return_value={'status': 'healthy'})

    mock_live_client = AsyncMock()
    mock_live_client.health_check = AsyncMock(return_value={'status': 'healthy'})

    orchestrator.services['ai'] = mock_ai_client
    orchestrator.services['backtesting'] = mock_backtesting_client
    orchestrator.services['live'] = mock_live_client
    orchestrator.is_initialized = True

    # Test health check
    health_status = await orchestrator.health_check()

    assert health_status['status'] == 'healthy'
    assert 'services' in health_status
    assert len(health_status['services']) == 3


def test_trading_service_orchestrator_creation():
    """Test that TradingServiceOrchestrator can be created."""
    orchestrator = TradingServiceOrchestrator()

    assert orchestrator.services == {}
    assert orchestrator.is_initialized is False
    assert orchestrator.active_sessions == {}
