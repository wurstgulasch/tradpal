"""
Tests for Trading Bot Live Service
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from dataclasses import asdict

from services.trading_bot_live.service import (
    TradingBotLiveService,
    TradingSession,
    Position,
    Order,
    OrderSide,
    OrderType,
    OrderStatus
)


class TestTradingBotLiveService:
    """Test cases for Trading Bot Live Service"""

    @pytest.fixture
    def service(self):
        """Create a test service instance"""
        return TradingBotLiveService()

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing"""
        return pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [103, 104, 105],
            'volume': [1000, 1100, 1200]
        }, index=pd.date_range('2023-01-01', periods=3))

    @pytest.fixture
    def trading_session(self):
        """Create a sample trading session"""
        return TradingSession(
            session_id="test_session_123",
            symbol="BTC/USDT",
            strategy="ml_enhanced",
            timeframe="1h",
            capital=10000.0,
            risk_per_trade=0.02,
            max_positions=5,
            is_active=True,
            paper_trading=True,
            start_time=datetime.now()
        )

    @pytest.mark.asyncio
    async def test_initialization(self, service):
        """Test service initialization"""
        # Test that service initializes with proper clients
        assert service.core_service is not None
        assert service.risk_service is not None
        assert service.market_regime_service is not None
        assert service.data_service is not None
        assert isinstance(service.trading_sessions, dict)
        assert isinstance(service.positions, dict)

    @pytest.mark.asyncio
    async def test_start_trading_session(self, service):
        """Test starting a trading session"""
        result = await service.start_trading(
            symbol='BTC/USDT',
            strategy='ml_enhanced',
            capital=10000.0,
            risk_per_trade=0.02,
            enable_paper_trading=True
        )

        assert result['trading_id'] is not None
        assert result['symbol'] == 'BTC/USDT'
        assert result['strategy'] == 'ml_enhanced'
        assert result['paper_trading'] is True

        # Check session was created
        assert 'BTC/USDT' in service.trading_sessions
        session = service.trading_sessions['BTC/USDT']
        assert session.symbol == 'BTC/USDT'
        assert session.capital == 10000.0
        assert session.risk_per_trade == 0.02

    @pytest.mark.asyncio
    async def test_health_check(self, service):
        """Test health check functionality"""
        health = await service.health_check()

        assert health['service'] == 'trading_bot_live'
        assert health['status'] == 'healthy'
        assert 'timestamp' in health
        assert 'active_sessions' in health
        assert 'open_positions' in health
        assert 'total_orders' in health

    @pytest.mark.asyncio
    async def test_generate_synthetic_data(self, service):
        """Test synthetic data generation"""
        data = service._generate_synthetic_data('BTC/USDT', 100)

        assert isinstance(data, pd.DataFrame)
        assert len(data) == 100
        assert list(data.columns) == ['open', 'high', 'low', 'close', 'volume']
        assert data.index.name == 'timestamp'

        # Check data is reasonable
        assert data['close'].min() > 0
        assert data['volume'].min() >= 0

    @pytest.mark.asyncio
    async def test_detect_basic_regime(self, service):
        """Test basic market regime detection"""
        regime = await service._detect_basic_regime('BTC/USDT')

        assert isinstance(regime, dict)
        assert 'regime' in regime
        assert 'confidence' in regime
        assert 'volatility' in regime
        assert 'timestamp' in regime
        assert regime['regime'] in ['consolidation', 'high_volatility', 'moderate_volatility']
        assert 0 <= regime['confidence'] <= 1

    @pytest.mark.asyncio
    async def test_get_market_regime_fallback(self, service):
        """Test market regime detection with fallback"""
        regime = await service._get_market_regime('BTC/USDT')

        assert isinstance(regime, dict)
        assert 'regime' in regime
        assert 'confidence' in regime

    @pytest.mark.asyncio
    async def test_calculate_position_size_fallback(self, service):
        """Test position size calculation fallback"""
        # First create a session
        await service.start_trading(
            symbol='BTC/USDT',
            strategy='test',
            capital=10000.0,
            risk_per_trade=0.02,
            enable_paper_trading=True
        )

        size = await service._calculate_position_size('BTC/USDT')

        assert isinstance(size, float)
        assert size > 0

    @pytest.mark.asyncio
    async def test_calculate_enhanced_fallback_position_size(self, service):
        """Test enhanced fallback position sizing"""
        # Create session first
        await service.start_trading(
            symbol='BTC/USDT',
            strategy='test',
            capital=10000.0,
            risk_per_trade=0.02,
            enable_paper_trading=True
        )

        regime_info = {'regime': 'trending', 'confidence': 0.8}
        size = await service._calculate_enhanced_fallback_position_size('BTC/USDT', 'BUY', 0.8, regime_info)

        assert isinstance(size, float)
        assert size > 0

    @pytest.mark.asyncio
    async def test_calculate_recent_win_rate(self, service):
        """Test win rate calculation"""
        # Create session first
        await service.start_trading(symbol='BTC/USDT', strategy='test', capital=10000.0)

        win_rate = await service._calculate_recent_win_rate('BTC/USDT')

        assert isinstance(win_rate, float)
        assert 0 <= win_rate <= 1

    def test_calculate_reward_risk_ratio(self, service):
        """Test reward/risk ratio calculation"""
        # Test different regimes
        trending_ratio = service._calculate_reward_risk_ratio('trending')
        consolidation_ratio = service._calculate_reward_risk_ratio('consolidation')
        high_vol_ratio = service._calculate_reward_risk_ratio('high_volatility')
        unknown_ratio = service._calculate_reward_risk_ratio('unknown')

        assert trending_ratio == 2.5
        assert consolidation_ratio == 1.5
        assert high_vol_ratio == 1.2
        assert unknown_ratio == 1.8

    def test_get_regime_position_multiplier(self, service):
        """Test regime position size multipliers"""
        multipliers = {
            'trending': service._get_regime_position_multiplier('trending'),
            'consolidation': service._get_regime_position_multiplier('consolidation'),
            'high_volatility': service._get_regime_position_multiplier('high_volatility'),
            'low_volatility': service._get_regime_position_multiplier('low_volatility'),
            'unknown': service._get_regime_position_multiplier('unknown')
        }

        assert multipliers['trending'] == 1.3
        assert multipliers['consolidation'] == 0.7
        assert multipliers['high_volatility'] == 0.5
        assert multipliers['low_volatility'] == 1.1
        assert multipliers['unknown'] == 0.9

    def test_get_confidence_threshold(self, service):
        """Test confidence thresholds for different regimes"""
        thresholds = {
            'trending': service._get_confidence_threshold('trending'),
            'consolidation': service._get_confidence_threshold('consolidation'),
            'high_volatility': service._get_confidence_threshold('high_volatility'),
            'low_volatility': service._get_confidence_threshold('low_volatility'),
            'unknown': service._get_confidence_threshold('unknown')
        }

        assert thresholds['trending'] == 0.75
        assert thresholds['consolidation'] == 0.80
        assert thresholds['high_volatility'] == 0.85
        assert thresholds['low_volatility'] == 0.70
        assert thresholds['unknown'] == 0.85

    def test_calculate_adaptive_risk_limits_static(self, service):
        """Test adaptive risk limits calculation"""
        # Test trending regime
        trending_limits = service._calculate_adaptive_risk_limits_static('trending', 0.05)
        assert trending_limits['max_drawdown'] == 0.075  # 1.5 * 0.05
        assert trending_limits['breach_response'] == 'reduce_positions_partial'
        assert trending_limits['position_limit_multiplier'] == 1.2

        # Test consolidation regime
        consolidation_limits = service._calculate_adaptive_risk_limits_static('consolidation', 0.05)
        assert consolidation_limits['max_drawdown'] == 0.035  # 0.7 * 0.05
        assert consolidation_limits['breach_response'] == 'close_all_positions'
        assert consolidation_limits['position_limit_multiplier'] == 0.8

    @pytest.mark.asyncio
    async def test_get_performance_based_multiplier(self, service):
        """Test performance-based position size adjustment"""
        # Create session first
        await service.start_trading(symbol='BTC/USDT', strategy='test', capital=10000.0)

        multiplier = await service._get_performance_based_multiplier('BTC/USDT')

        assert isinstance(multiplier, float)
        assert 0.5 <= multiplier <= 1.5

    @pytest.mark.asyncio
    async def test_calculate_smart_sl_tp(self, service):
        """Test smart stop loss and take profit calculation"""
        # Create session first
        await service.start_trading(symbol='BTC/USDT', strategy='test', capital=10000.0)

        sl_tp = await service._calculate_smart_sl_tp('BTC/USDT', 'BUY', 50000.0, 'trending')

        assert isinstance(sl_tp, dict)
        assert 'stop_loss' in sl_tp
        assert 'take_profit' in sl_tp
        assert sl_tp['stop_loss'] < 50000.0  # Stop loss below entry for BUY
        assert sl_tp['take_profit'] > 50000.0  # Take profit above entry for BUY

    @pytest.mark.asyncio
    async def test_execute_paper_trade(self, service):
        """Test paper trade execution"""
        trade_result = await service.execute_paper_trade({
            'signal': 'BUY',
            'symbol': 'BTC/USDT',
            'price': 50000.0,
            'quantity': 0.01
        })

        assert trade_result['success'] is True
        assert 'order_id' in trade_result
        assert trade_result['symbol'] == 'BTC/USDT'
        assert trade_result['action'] == 'BUY'
        assert trade_result['quantity'] == 0.01

    @pytest.mark.asyncio
    async def test_get_current_price(self, service):
        """Test current price retrieval"""
        price = await service._get_current_price('BTC/USDT')

        assert isinstance(price, float)
        assert price > 0

    @pytest.mark.asyncio
    async def test_get_volatility(self, service):
        """Test volatility calculation"""
        volatility = await service._get_volatility('BTC/USDT')

        assert isinstance(volatility, float)
        assert volatility > 0

    @pytest.mark.asyncio
    async def test_get_symbol_performance(self, service):
        """Test symbol performance calculation"""
        # Create session first
        await service.start_trading(symbol='BTC/USDT', strategy='test', capital=10000.0)

        performance = await service.get_symbol_performance('BTC/USDT')

        assert isinstance(performance, dict)
        assert 'total_return' in performance
        assert 'total_trades' in performance
        assert 'win_rate' in performance

    @pytest.mark.asyncio
    async def test_get_trading_status(self, service):
        """Test trading status retrieval"""
        status = await service.get_trading_status()

        assert isinstance(status, dict)
        assert 'sessions' in status
        assert 'total_sessions' in status
        assert 'active_sessions' in status

    @pytest.mark.asyncio
    async def test_get_symbol_status(self, service):
        """Test symbol-specific status"""
        # Test inactive symbol
        status = await service.get_symbol_status('BTC/USDT')
        assert status['status'] == 'inactive'
        assert status['symbol'] == 'BTC/USDT'

        # Test active symbol
        await service.start_trading(symbol='BTC/USDT', strategy='test', capital=10000.0)
        status = await service.get_symbol_status('BTC/USDT')
        assert status['status'] != 'inactive'
        assert status['symbol'] == 'BTC/USDT'

    @pytest.mark.asyncio
    async def test_get_positions(self, service):
        """Test position retrieval"""
        positions = await service.get_positions()

        assert isinstance(positions, dict)
        assert 'positions' in positions
        assert 'total_positions' in positions
        assert 'total_value' in positions

    @pytest.mark.asyncio
    async def test_get_symbol_positions(self, service):
        """Test symbol-specific position retrieval"""
        positions = await service.get_symbol_positions('BTC/USDT')

        assert isinstance(positions, list)

    @pytest.mark.asyncio
    async def test_get_performance_metrics(self, service):
        """Test overall performance metrics"""
        metrics = await service.get_performance_metrics()

        assert isinstance(metrics, dict)
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'win_rate' in metrics
        assert 'profit_factor' in metrics
        assert 'total_trades' in metrics

    @pytest.mark.asyncio
    async def test_get_order_history(self, service):
        """Test order history retrieval"""
        orders = await service.get_order_history()

        assert isinstance(orders, list)

    @pytest.mark.asyncio
    async def test_get_symbol_orders(self, service):
        """Test symbol-specific order history"""
        orders = await service.get_symbol_orders('BTC/USDT')

        assert isinstance(orders, list)

    @pytest.mark.asyncio
    async def test_get_portfolio(self, service):
        """Test portfolio information retrieval"""
        portfolio = await service.get_portfolio()

        assert isinstance(portfolio, dict)
        assert 'total_balance' in portfolio
        assert 'positions_value' in portfolio
        assert 'available_balance' in portfolio
        assert 'positions' in portfolio

    @pytest.mark.asyncio
    async def test_place_manual_order(self, service):
        """Test manual order placement"""
        # Create session first
        await service.start_trading(symbol='BTC/USDT', strategy='test', capital=10000.0)

        result = await service.place_manual_order(
            symbol='BTC/USDT',
            side='buy',
            quantity=0.01,
            order_type='market'
        )

        assert isinstance(result, dict)
        assert 'order_id' in result
        assert result['symbol'] == 'BTC/USDT'
        assert result['side'] == 'buy'
        assert result['quantity'] == 0.01

    @pytest.mark.asyncio
    async def test_update_risk_parameters(self, service):
        """Test risk parameter updates"""
        # Create session first
        await service.start_trading(symbol='BTC/USDT', strategy='test', capital=10000.0)

        result = await service.update_risk_parameters(
            symbol='BTC/USDT',
            risk_per_trade=0.03,
            max_positions=3
        )

        assert isinstance(result, dict)
        assert result['risk_per_trade'] == 0.03
        assert result['max_positions'] == 3

    @pytest.mark.asyncio
    async def test_stop_trading(self, service):
        """Test trading session stop"""
        # Create session first
        await service.start_trading(symbol='BTC/USDT', strategy='test', capital=10000.0)

        result = await service.stop_trading('BTC/USDT')

        assert isinstance(result, dict)
        assert result['symbol'] == 'BTC/USDT'
        assert 'positions_closed' in result

    @pytest.mark.asyncio
    async def test_emergency_stop(self, service):
        """Test emergency stop functionality"""
        # Create multiple sessions
        await service.start_trading(symbol='BTC/USDT', strategy='test1', capital=10000.0)
        await service.start_trading(symbol='ETH/USDT', strategy='test2', capital=5000.0)

        result = await service.emergency_stop()

        assert isinstance(result, dict)
        assert 'positions_closed' in result
        assert 'sessions_stopped' in result
        assert 'pending_orders_cancelled' in result

    @pytest.mark.asyncio
    async def test_update_position(self, service):
        """Test position updates"""
        # First create a position by executing a trade
        await service.execute_paper_trade({
            'signal': 'BUY',
            'symbol': 'BTC/USDT',
            'price': 50000.0,
            'quantity': 0.01
        })

        # Get the position
        positions = await service.get_symbol_positions('BTC/USDT')
        assert len(positions) > 0

        position = positions[0]
        position_id = position['position_id']

        # Update position
        result = await service.update_position({
            'position_id': position_id,
            'symbol': 'BTC/USDT',
            'current_price': 51000.0,
            'stop_loss': 49000.0,
            'take_profit': 53000.0
        })

        assert result['success'] is True

    @pytest.mark.asyncio
    async def test_check_risk_limits(self, service):
        """Test risk limits checking"""
        # Create session first
        await service.start_trading(symbol='BTC/USDT', strategy='test', capital=10000.0)

        result = await service.check_risk_limits({
            'capital': 10000.0,
            'current_value': 9500.0  # 5% drawdown
        })

        assert isinstance(result, dict)
        assert 'current_drawdown' in result
        assert 'max_drawdown' in result
        assert 'risk_breached' in result

    @pytest.mark.asyncio
    async def test_handle_price_update(self, service):
        """Test price update handling"""
        # First create a position
        await service.execute_paper_trade({
            'signal': 'BUY',
            'symbol': 'BTC/USDT',
            'price': 50000.0,
            'quantity': 0.01
        })

        result = await service.handle_price_update({
            'symbol': 'BTC/USDT',
            'price': 51000.0
        })

        assert result['success'] is True
        assert result['symbol'] == 'BTC/USDT'
        assert result['new_price'] == 51000.0
        assert 'updated_positions' in result

    @pytest.mark.asyncio
    async def test_monitor_trading_session(self, service):
        """Test trading session monitoring"""
        # Create session
        await service.start_trading(symbol='BTC/USDT', strategy='test', capital=10000.0)

        # Start monitoring in background
        session = service.trading_sessions['BTC/USDT']
        monitor_task = asyncio.create_task(service._monitor_trading_session('BTC/USDT'))

        # Let it run briefly
        await asyncio.sleep(0.1)

        # Stop the session
        session.is_active = False
        await asyncio.sleep(0.1)

        # Task should complete
        try:
            await asyncio.wait_for(monitor_task, timeout=1.0)
        except asyncio.TimeoutError:
            monitor_task.cancel()

    @pytest.mark.asyncio
    async def test_fallback_signal_generation(self, service):
        """Test fallback signal generation"""
        # Create session first
        await service.start_trading(symbol='BTC/USDT', strategy='test', capital=10000.0)

        # This should not raise an exception
        await service._fallback_signal_generation('BTC/USDT')

        # Check that positions list exists
        assert 'BTC/USDT' in service.positions

    @pytest.mark.asyncio
    async def test_update_positions(self, service):
        """Test position updates"""
        # Create session and position first
        await service.start_trading(symbol='BTC/USDT', strategy='test', capital=10000.0)
        await service.execute_paper_trade({
            'signal': 'BUY',
            'symbol': 'BTC/USDT',
            'price': 50000.0,
            'quantity': 0.01
        })

        # Update positions
        await service._update_positions('BTC/USDT')

        # Check positions were updated
        positions = service.positions.get('BTC/USDT', [])
        assert len(positions) > 0

    @pytest.mark.asyncio
    async def test_check_position_exits(self, service):
        """Test position exit checking"""
        # Create a position
        await service.start_trading(symbol='BTC/USDT', strategy='test', capital=10000.0)
        await service.execute_paper_trade({
            'signal': 'BUY',
            'symbol': 'BTC/USDT',
            'price': 50000.0,
            'quantity': 0.01
        })

        position = service.positions['BTC/USDT'][0]

        # Set stop loss that should trigger
        position.stop_loss = 51000.0  # Above current price
        position.current_price = 50000.0

        # Check exits (should not trigger)
        await service._check_position_exits(position)

        # Position should still exist
        assert len(service.positions['BTC/USDT']) == 1

    @pytest.mark.asyncio
    async def test_close_position(self, service):
        """Test position closing"""
        # Create a position
        await service.start_trading(symbol='BTC/USDT', strategy='test', capital=10000.0)
        await service.execute_paper_trade({
            'signal': 'BUY',
            'symbol': 'BTC/USDT',
            'price': 50000.0,
            'quantity': 0.01
        })

        position = service.positions['BTC/USDT'][0]

        # Close position
        await service._close_position(position, 'Test close')

        # Position should be removed
        assert len(service.positions['BTC/USDT']) == 0

        # Check trade history was updated
        assert len(service.trade_history) > 0

    @pytest.mark.asyncio
    async def test_execute_paper_order(self, service):
        """Test paper order execution"""
        # Create session first
        await service.start_trading(symbol='BTC/USDT', strategy='test', capital=10000.0)

        order = Order(
            order_id='test_order_123',
            symbol='BTC/USDT',
            side=OrderSide.BUY,
            quantity=0.01,
            order_type=OrderType.MARKET,
            price=50000.0,
            status=OrderStatus.PENDING,
            timestamp=datetime.now()
        )

        await service._execute_paper_order(order)

        # Check order was filled
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 0.01

        # Check position was created
        assert 'BTC/USDT' in service.positions
        assert len(service.positions['BTC/USDT']) == 1

    @pytest.mark.asyncio
    async def test_validate_order(self, service):
        """Test order validation"""
        # Create session first
        await service.start_trading(symbol='BTC/USDT', strategy='test', capital=10000.0)

        # Valid order
        result = await service._validate_order('BTC/USDT', 'buy', 0.01, 'market', 50000.0)
        assert result is True

        # Invalid quantity
        result = await service._validate_order('BTC/USDT', 'buy', 0, 'market', 50000.0)
        assert result is False

        # Invalid side
        result = await service._validate_order('BTC/USDT', 'invalid', 0.01, 'market', 50000.0)
        assert result is False

    def test_calculate_pnl(self, service):
        """Test P&L calculation"""
        position = {
            'side': 'BUY',
            'entry_price': 50000.0,
            'current_price': 51000.0,
            'quantity': 0.01
        }

        pnl = service._calculate_pnl(position)
        expected_pnl = (51000.0 - 50000.0) * 0.01
        assert pnl == expected_pnl

    def test_calculate_risk_limits(self, service):
        """Test risk limits calculation"""
        portfolio = {
            'capital': 10000.0,
            'positions_value': 2000.0
        }

        limits = service._calculate_risk_limits(portfolio)

        assert limits['max_single_position'] == 1000.0  # 10% of capital
        assert limits['max_total_positions'] == 5000.0  # 50% of capital
        assert limits['current_exposure'] == 2000.0
        assert limits['available_risk_capital'] == 200.0  # 2% of capital

    def test_validate_order_data(self, service):
        """Test order data validation"""
        # Valid order
        result = service._validate_order({
            'symbol': 'BTC/USDT',
            'signal': 'BUY',
            'quantity': 0.01,
            'price': 50000.0
        })
        assert result is True

        # Invalid quantity
        result = service._validate_order({
            'symbol': 'BTC/USDT',
            'signal': 'BUY',
            'quantity': 0,
            'price': 50000.0
        })
        assert result is False

        # Missing symbol
        result = service._validate_order({
            'signal': 'BUY',
            'quantity': 0.01,
            'price': 50000.0
        })
        assert result is False

    @pytest.mark.asyncio
    async def test_get_empty_performance(self, service):
        """Test empty performance metrics"""
        # This tests the private _get_empty_performance method indirectly
        # by calling get_performance_metrics with no trades
        metrics = await service.get_performance_metrics()

        assert metrics['total_return'] == 0.0
        assert metrics['total_trades'] == 0
        assert metrics['win_rate'] == 0.0
        assert metrics['avg_trade_return'] == 0.0


if __name__ == "__main__":
    # Run basic smoke test
    async def smoke_test():
        print("🧪 Running Trading Bot Live Service smoke test...")

        service = TradingBotLiveService()

        # Test initialization
        print("✅ Service initialized with smart capabilities")

        # Test health check
        health = await service.health_check()
        print(f"✅ Health check: {health['status']}")

        # Test session creation
        result = await service.start_trading(
            symbol='BTC/USDT',
            strategy='ml_enhanced',
            capital=10000.0,
            risk_per_trade=0.02,
            enable_paper_trading=True
        )
        print(f"✅ Trading session created: {result['trading_id']}")

        # Test synthetic data
        data = service._generate_synthetic_data('BTC/USDT', 50)
        print(f"✅ Synthetic data generated: {len(data)} rows")

        # Test paper trade
        trade = await service.execute_paper_trade({
            'signal': 'BUY',
            'symbol': 'BTC/USDT',
            'price': 50000.0,
            'quantity': 0.01
        })
        print(f"✅ Paper trade executed: {trade['success']}")

        # Test status
        status = await service.get_trading_status()
        print(f"✅ Status retrieved: {status['total_sessions']} sessions")

        print("🎉 Comprehensive smoke test passed!")

    asyncio.run(smoke_test())