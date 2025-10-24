"""
Test script for Trading Service Orchestrator functionality.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from services.trading_service.orchestrator import TradingServiceOrchestrator


@pytest.mark.asyncio
async def test_orchestrator_start_trading_session():
    """Test starting a trading session through the orchestrator."""
    orchestrator = TradingServiceOrchestrator()

    # Mock services
    mock_market_regime = AsyncMock()
    mock_market_regime.initialize = AsyncMock(return_value=True)

    mock_risk_management = AsyncMock()
    mock_risk_management.initialize = AsyncMock(return_value=True)

    mock_trading_execution = AsyncMock()
    mock_trading_execution.initialize = AsyncMock(return_value=True)

    mock_live_client = AsyncMock()
    mock_live_client.start_trading_session = AsyncMock(return_value={'status': 'started'})

    orchestrator.services['ml_training'] = AsyncMock()
    orchestrator.services['backtesting'] = AsyncMock()
    orchestrator.services['live'] = mock_live_client
    orchestrator.services['market_regime'] = mock_market_regime
    orchestrator.services['risk_management'] = mock_risk_management
    orchestrator.services['trading_execution'] = mock_trading_execution
    orchestrator.services['reinforcement_learning'] = AsyncMock()
    orchestrator.is_initialized = True

    # Test starting trading
    result = await orchestrator.start_trading_session({
        'symbol': 'BTC/USDT',
        'strategy': 'ai_trading',
        'capital': 10000
    })

    assert 'session_id' in result
    assert result['symbol'] == 'BTC/USDT'
    mock_market_regime.initialize.assert_called_once()
    mock_risk_management.initialize.assert_called_once()
    mock_trading_execution.initialize.assert_called_once()


@pytest.mark.asyncio
async def test_orchestrator_get_trading_status():
    """Test getting trading status from orchestrator."""
    orchestrator = TradingServiceOrchestrator()

    # Mock services
    mock_ai_client = AsyncMock()
    mock_ai_client.get_status = AsyncMock(return_value={
        'active_sessions': 1,
        'total_pnl': 150.50
    })

    orchestrator.services['ai'] = mock_ai_client
    orchestrator.services['backtesting'] = AsyncMock()
    orchestrator.services['live'] = AsyncMock()
    orchestrator.is_initialized = True

    # Test getting status
    status = await orchestrator.get_trading_status()

    assert 'orchestrator' in status
    assert 'services' in status
    assert status['orchestrator']['active_sessions'] == 0  # No active sessions in test
    mock_ai_client.get_status.assert_called_once()


@pytest.mark.asyncio
async def test_orchestrator_run_backtest():
    """Test running backtest through orchestrator."""
    orchestrator = TradingServiceOrchestrator()

    # Mock services
    mock_backtesting_client = AsyncMock()
    mock_backtesting_client.run_backtest = AsyncMock(return_value={
        'sharpe_ratio': 1.8,
        'total_return': 25.5,
        'max_drawdown': -8.2
    })

    orchestrator.services['ai'] = AsyncMock()
    orchestrator.services['backtesting'] = mock_backtesting_client
    orchestrator.services['live'] = AsyncMock()
    orchestrator.is_initialized = True

    # Test running backtest
    result = await orchestrator.run_backtest({
        'symbol': 'BTC/USDT',
        'start_date': '2023-01-01',
        'end_date': '2023-12-31'
    })

    assert result['sharpe_ratio'] == 1.8
    assert result['total_return'] == 25.5
    mock_backtesting_client.run_backtest.assert_called_once()


@pytest.mark.asyncio
async def test_orchestrator_uninitialized_error():
    """Test that orchestrator raises error when not initialized."""
    orchestrator = TradingServiceOrchestrator()

    with pytest.raises(RuntimeError, match="Trading Service Orchestrator not initialized"):
        await orchestrator.start_trading_session({})


@pytest.mark.asyncio
async def test_orchestrator_service_initialization_failure():
    """Test handling of service initialization failures."""
    orchestrator = TradingServiceOrchestrator()

    # Mock AI service to fail
    mock_ai_client = AsyncMock()
    mock_ai_client.authenticate = AsyncMock(side_effect=Exception("Connection failed"))

    # Patch the client classes
    with patch('services.trading_service.orchestrator.MLTrainingServiceClient', return_value=mock_ai_client), \
         patch('services.trading_service.orchestrator.BacktestingServiceClient', return_value=AsyncMock()), \
         patch('services.trading_service.orchestrator.TradingBotLiveServiceClient', return_value=AsyncMock()), \
         patch('services.trading_service.orchestrator.ReinforcementLearningServiceClient', return_value=AsyncMock()), \
         patch('services.trading_service.orchestrator.MarketRegimeServiceClient', return_value=AsyncMock()), \
         patch('services.trading_service.orchestrator.RiskManagementServiceClient', return_value=AsyncMock()), \
         patch('services.trading_service.orchestrator.TradingExecutionServiceClient', return_value=AsyncMock()):

        # Test that initialization handles failures gracefully
        with pytest.raises(Exception):
            await orchestrator.initialize()
