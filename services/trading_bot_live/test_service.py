"""
Tests for Trading Bot Live Service
"""

import pytest
import asyncio
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch

from services.trading_bot_live.service import TradingBotLiveService


class TestTradingBotLiveService:
    """Test cases for Trading Bot Live Service"""

    @pytest.fixture
    def service(self):
        """Create a test service instance"""
        return TradingBotLiveService('BTC/USDT', '1h')

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing"""
        return pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [103, 104, 105],
            'volume': [1000, 1100, 1200],
            'EMA9': [101, 102, 103],
            'EMA21': [100, 101, 102],
            'RSI': [25, 28, 32],
            'BB_lower': [98, 99, 100],
            'Buy_Signal': [0, 0, 1],
            'Sell_Signal': [0, 0, 0],
            'Position_Size_Percent': [2.0, 2.0, 2.0],
            'Stop_Loss_Buy': [95, 96, 97],
            'Take_Profit_Buy': [110, 111, 112]
        }, index=pd.date_range('2023-01-01', periods=3))

    @pytest.mark.asyncio
    async def test_initialization(self, service):
        """Test service initialization"""
        await service.initialize()

        assert service.symbol == 'BTC/USDT'
        assert service.timeframe == '1h'
        assert service.state.capital == 10000.0  # INITIAL_CAPITAL
        assert not service.is_running

        await service.shutdown()

    def test_generate_signal_reasoning_buy(self, service, sample_market_data):
        """Test signal reasoning generation for buy signals"""
        latest_data = sample_market_data.iloc[-1]

        reasoning = service._generate_signal_reasoning('BUY', latest_data)

        assert isinstance(reasoning, str)
        assert len(reasoning) > 0

    def test_generate_signal_reasoning_sell(self, service, sample_market_data):
        """Test signal reasoning generation for sell signals"""
        # Create sell signal data
        sell_data = sample_market_data.iloc[-1].copy()
        sell_data['Buy_Signal'] = 0
        sell_data['Sell_Signal'] = 1
        sell_data['EMA9'] = 100  # Below EMA21
        sell_data['RSI'] = 75   # Overbought

        reasoning = service._generate_signal_reasoning('SELL', sell_data)

        assert isinstance(reasoning, str)
        assert len(reasoning) > 0

    def test_get_status(self, service):
        """Test status reporting"""
        status = service.get_status()

        assert status['service'] == 'trading_bot_live'
        assert status['symbol'] == 'BTC/USDT'
        assert status['timeframe'] == '1h'
        assert 'is_running' in status
        assert 'capital' in status
        assert 'performance_metrics' in status

    @pytest.mark.asyncio
    async def test_risk_limits_check(self, service):
        """Test risk limits checking"""
        # Test normal conditions
        result = await service._check_risk_limits()
        assert result is True

        # Test drawdown limit exceeded
        service.state.capital = 5000  # Below 50% of initial capital
        service.max_drawdown = 0.3    # 30% max drawdown

        result = await service._check_risk_limits()
        assert result is False

        # Reset for other tests
        service.state.capital = 10000.0

    @pytest.mark.asyncio
    @patch('services.trading_bot_live.service.fetch_data')
    async def test_fetch_market_data(self, mock_fetch, service, sample_market_data):
        """Test market data fetching"""
        mock_fetch.return_value = sample_market_data

        data = await service._fetch_market_data()

        assert not data.empty
        assert len(data) == 3
        mock_fetch.assert_called_once()

    @pytest.mark.asyncio
    @patch('services.trading_bot_live.service.calculate_indicators')
    @patch('services.trading_bot_live.service.generate_signals')
    @patch('services.trading_bot_live.service.calculate_risk_management')
    async def test_process_market_data(self, mock_risk, mock_signals, mock_indicators, service, sample_market_data):
        """Test market data processing"""
        # Setup mocks
        mock_indicators.return_value = sample_market_data
        mock_signals.return_value = sample_market_data
        mock_risk.return_value = sample_market_data

        result = await service._process_market_data(sample_market_data)

        assert not result.empty
        mock_indicators.assert_called_once_with(sample_market_data)
        mock_signals.assert_called_once()
        mock_risk.assert_called_once()


if __name__ == "__main__":
    # Run basic smoke test
    async def smoke_test():
        print("🧪 Running Trading Bot Live Service smoke test...")

        service = TradingBotLiveService('BTC/USDT', '1h')

        # Test initialization
        await service.initialize()
        print("✅ Service initialized")

        # Test status
        status = service.get_status()
        print(f"✅ Status: {status['service']} - {status['symbol']}")

        # Test shutdown
        await service.shutdown()
        print("✅ Service shutdown")

        print("🎉 Smoke test passed!")

    asyncio.run(smoke_test())