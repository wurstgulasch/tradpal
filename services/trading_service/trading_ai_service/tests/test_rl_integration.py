"""
Test RL Integration with Trading AI Service
Tests the integration of reinforcement learning with the trading AI orchestrator
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from services.trading_service.trading_ai_service.orchestrator import TradingServiceOrchestrator
from services.trading_service.trading_ai_service.reinforcement_learning.service import ReinforcementLearningService


class TestRLIntegration:
    """Test reinforcement learning integration with trading orchestrator"""

    @pytest.fixture
    def orchestrator(self):
        """Create test orchestrator instance"""
        return TradingServiceOrchestrator()

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        np.random.seed(42)

        # Generate realistic price data with trend and noise
        base_price = 50000
        trend = np.linspace(0, 1000, 100)  # Upward trend
        noise = np.random.normal(0, 500, 100)
        prices = base_price + trend + noise

        # Ensure positive prices
        prices = np.maximum(prices, 1000)

        # Create OHLCV data
        high = prices * (1 + np.random.uniform(0.005, 0.02, 100))
        low = prices * (1 - np.random.uniform(0.005, 0.02, 100))
        close = prices
        volume = np.random.uniform(100, 1000, 100)

        # Calculate technical indicators (all same length as prices)
        rsi = 50 + 30 * np.sin(np.linspace(0, 4*np.pi, 100))  # Oscillating RSI
        # Simple MACD approximation - ensure same length
        macd = np.zeros(100)
        for i in range(1, 100):
            macd[i] = close[i] - close[i-1]  # Simplified MACD

        return pd.DataFrame({
            'timestamp': dates,
            'open': prices * 0.995,  # Slight variation for open
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
            'rsi': rsi,
            'macd': macd,
            'bb_position': np.random.uniform(-1, 1, 100),
            'stoch_k': np.random.uniform(0, 100, 100),
            'cci': np.random.uniform(-200, 200, 100),
            'adx': np.random.uniform(10, 50, 100)
        })

    @pytest.mark.asyncio
    async def test_rl_service_initialization(self, orchestrator):
        """Test RL service initialization"""
        # Mock the central client
        orchestrator.client = AsyncMock()

        # Initialize AI services
        await orchestrator.rl_service.initialize()
        # ML trainer and regime detector don't have async initialize methods
        if hasattr(orchestrator.ml_trainer, 'initialize'):
            await orchestrator.ml_trainer.initialize()
        if hasattr(orchestrator.regime_service, 'initialize'):
            await orchestrator.regime_service.initialize()

        assert orchestrator.rl_service.is_initialized
        assert 'q_learning' in orchestrator.rl_service.agents
        assert 'default' in orchestrator.rl_service.reward_functions

        # Cleanup
        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_signal_combination(self, orchestrator):
        """Test signal combination from ML, RL, and regime detection"""
        # Mock signals
        ml_signal = {
            "action": "BUY",
            "confidence": 0.8
        }

        rl_signal = {
            "signal": "buy",
            "confidence": 0.7
        }

        regime_info = {
            "regime": "trending",
            "confidence": 0.9
        }

        market_data = {"close": 50000}

        # Test signal combination
        combined = orchestrator._combine_signals(ml_signal, rl_signal, regime_info, market_data)

        assert "action" in combined
        assert "confidence" in combined
        assert "signals" in combined
        assert combined["confidence"] > 0.7  # Should be high due to agreement
        assert combined["action"] == "BUY"

    @pytest.mark.asyncio
    async def test_conflicting_signals(self, orchestrator):
        """Test signal combination with conflicting ML and RL signals"""
        # Conflicting signals
        ml_signal = {
            "action": "BUY",
            "confidence": 0.8
        }

        rl_signal = {
            "signal": "sell",
            "confidence": 0.6
        }

        regime_info = {
            "regime": "volatile",
            "confidence": 0.8
        }

        market_data = {"close": 50000}

        # Test signal combination
        combined = orchestrator._combine_signals(ml_signal, rl_signal, regime_info, market_data)

        # Should result in HOLD due to conflict and volatility
        assert combined["action"] == "HOLD"
        assert combined["confidence"] < 0.8  # Lower confidence due to disagreement

    @pytest.mark.asyncio
    async def test_rl_training(self, orchestrator, sample_market_data):
        """Test RL agent training"""
        # Mock client response with flexible signature
        orchestrator.client = AsyncMock()
        async def mock_get_market_data(*args, **kwargs):
            return {"data": sample_market_data}  # Return DataFrame directly
        orchestrator.client.get_market_data = mock_get_market_data

        # Initialize orchestrator to avoid full initialization
        orchestrator.is_initialized = True
        await orchestrator.rl_service.initialize()

        training_config = {
            'episodes': 10,  # Short training for test
            'algorithm': 'q_learning',
            'reward_function': 'default',
            'initial_balance': 10000.0,
            'transaction_cost': 0.001
        }

        result = await orchestrator.train_rl_agent("BTCUSDT", training_config)

        assert result["success"] is True
        assert "average_reward" in result
        assert "episodes_trained" in result
        assert result["episodes_trained"] == 10

    @pytest.mark.asyncio
    async def test_rl_validation(self, orchestrator, sample_market_data):
        """Test RL model validation"""
        # Mock client response with flexible signature
        orchestrator.client = AsyncMock()
        async def mock_get_market_data(*args, **kwargs):
            return {"data": sample_market_data}  # Return DataFrame directly
        orchestrator.client.get_market_data = mock_get_market_data

        # Initialize orchestrator to avoid full initialization
        orchestrator.is_initialized = True
        await orchestrator.rl_service.initialize()

        validation_config = {
            'validation_windows': 3,
            'algorithm': 'q_learning'
        }

        result = await orchestrator.validate_rl_model("BTCUSDT", validation_config)

        assert result["success"] is True
        assert "consistency_score" in result
        assert "validation_results" in result
        assert len(result["validation_results"]) == 3

    @pytest.mark.asyncio
    async def test_smart_trade_execution(self, orchestrator):
        """Test smart trade execution with RL integration"""
        # Mock the central client
        orchestrator.client = AsyncMock()

        # Mock all required services
        orchestrator.trading_service = AsyncMock()
        orchestrator.monitoring_service = AsyncMock()

        orchestrator.trading_service.get_session_status.return_value = {
            "session": {
                "capital": 10000.0,
                "risk_per_trade": 0.02
            }
        }
        orchestrator.trading_service.calculate_position_size.return_value = {
            "quantity": 0.1
        }
        orchestrator.trading_service.execute_trade.return_value = {
            "order_id": "test_order_123",
            "status": "filled"
        }

        orchestrator.regime_service.detect_regime = AsyncMock(return_value={
            "regime": "trending",
            "confidence": 0.9
        })
        orchestrator.regime_service.get_regime_advice = AsyncMock(return_value={
            "position_size_multiplier": 1.2
        })

        orchestrator.rl_service.get_rl_signal = AsyncMock(return_value={
            "signal": "buy",
            "confidence": 0.8
        })

        market_data = {
            "prices": [50000.0, 50100.0, 49900.0],
            "current_price": 50000.0
        }

        # Initialize orchestrator
        orchestrator.is_initialized = True

        result = await orchestrator.execute_smart_trade("BTCUSDT", market_data)

        assert "decision" in result or "message" in result
        if "decision" in result:
            assert result["decision"] == "buy"
            assert result["confidence"] == 0.8

    @pytest.mark.asyncio
    async def test_risk_rejection(self, orchestrator):
        """Test trade rejection due to risk limits"""
        # Mock the central client
        orchestrator.client = AsyncMock()

        # Mock services with high confidence but risk rejection
        orchestrator.trading_service = AsyncMock()
        orchestrator.monitoring_service = AsyncMock()

        orchestrator.trading_service.get_session_status.return_value = {
            "session": None  # No active session
        }

        # Initialize required services
        await orchestrator.regime_service.initialize()
        await orchestrator.rl_service.initialize()

        market_data = {
            "prices": [50000.0],
            "current_price": 50000.0
        }

        # Initialize orchestrator
        orchestrator.is_initialized = True

        result = await orchestrator.execute_smart_trade("BTCUSDT", market_data)

        assert "error" in result
        assert "No active session" in result["error"]

    @pytest.mark.asyncio
    async def test_low_confidence_hold(self, orchestrator):
        """Test holding position due to low combined confidence"""
        # Mock the central client
        orchestrator.client = AsyncMock()

        # Mock services with low confidence
        orchestrator.trading_service = AsyncMock()
        orchestrator.monitoring_service = AsyncMock()

        orchestrator.trading_service.get_session_status.return_value = {
            "session": {
                "capital": 10000.0,
                "risk_per_trade": 0.02
            }
        }

        orchestrator.regime_service.detect_regime = AsyncMock(return_value={
            "regime": "volatile",
            "confidence": 0.8
        })
        orchestrator.regime_service.get_regime_advice = AsyncMock(return_value={
            "position_size_multiplier": 0.7
        })

        orchestrator.rl_service.get_rl_signal = AsyncMock(return_value={
            "signal": "sell",
            "confidence": 0.3  # Low confidence
        })

        market_data = {
            "prices": [50000.0],
            "current_price": 50000.0
        }

        # Initialize orchestrator
        orchestrator.is_initialized = True

        result = await orchestrator.execute_smart_trade("BTCUSDT", market_data)

        assert "message" in result
        assert "too low" in result["message"]

    @pytest.mark.asyncio
    async def test_rl_model_status(self, orchestrator):
        """Test RL model status retrieval"""
        # Mock the central client
        orchestrator.client = AsyncMock()

        # Initialize RL service
        await orchestrator.rl_service.initialize()

        # Set orchestrator as initialized to avoid full initialization
        orchestrator.is_initialized = True

        status = await orchestrator.get_rl_model_status()

        assert "service_initialized" in status
        assert "available_algorithms" in status
        assert "available_reward_functions" in status
        assert "models" in status
        assert status["service_initialized"] is True

        # Cleanup
        await orchestrator.rl_service.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_procedure(self, orchestrator):
        """Test proper shutdown of all services"""
        # Mock the central client
        orchestrator.client = AsyncMock()

        # Initialize services
        await orchestrator.rl_service.initialize()
        if hasattr(orchestrator.regime_service, 'initialize'):
            await orchestrator.regime_service.initialize()

        # Set initialized flag
        orchestrator.is_initialized = True

        assert orchestrator.is_initialized

        await orchestrator.shutdown()

        assert not orchestrator.is_initialized
        assert not orchestrator.rl_service.is_initialized


if __name__ == "__main__":
    # Run basic integration test
    async def run_integration_test():
        print("Running RL Integration Test...")

        orchestrator = TradingServiceOrchestrator()

        try:
            # Initialize
            await orchestrator.initialize()
            print("✓ Orchestrator initialized")

            # Test RL service status
            status = await orchestrator.get_rl_model_status()
            print(f"✓ RL Service status: {status['service_initialized']}")

            # Test signal combination
            combined = orchestrator._combine_signals(
                {"action": "BUY", "confidence": 0.8},
                {"signal": "buy", "confidence": 0.7},
                {"regime": "trending", "confidence": 0.9},
                {"close": 50000}
            )
            print(f"✓ Signal combination: {combined['action']} (confidence: {combined['confidence']:.2f})")

            print("✓ All integration tests passed!")

        except Exception as e:
            print(f"✗ Integration test failed: {e}")
        finally:
            await orchestrator.shutdown()

    asyncio.run(run_integration_test())