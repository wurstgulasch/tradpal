"""
Tests for Paper Trading Engine

Tests the paper trading functionality for simulated trading.
"""

import pytest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Try to import paper trading components
try:
    from paper_trading import PaperTradingEngine, Portfolio, OrderType, PAPER_TRADING_ENABLED
    from config.settings import PAPER_TRADING_INITIAL_BALANCE, PAPER_TRADING_FEE_RATE
    PAPER_TRADING_AVAILABLE = True
except ImportError:
    PAPER_TRADING_AVAILABLE = False
    pytest.skip("Paper trading module not available", allow_module_level=True)


class TestPaperTradingPortfolio:
    """Test suite for paper trading portfolio management."""

    def setup_method(self):
        """Skip test if paper trading is not available."""
        if not PAPER_TRADING_AVAILABLE:
            pytest.skip("Paper trading module not available")

    def test_portfolio_initialization(self):
        """Test portfolio initialization."""
        portfolio = Portfolio(initial_balance=10000)

        assert portfolio.balance == 10000
        assert portfolio.initial_balance == 10000
        assert portfolio.positions == {}
        assert portfolio.trades == []

    def test_portfolio_buy_order(self):
        """Test buying in portfolio."""
        portfolio = Portfolio(initial_balance=10000)

        # Buy 0.1 BTC at $50000 (costs $5000)
        success = portfolio.execute_order('BTC/USDT', OrderType.BUY, 0.1, 50000)

        assert success
        assert portfolio.balance == 10000 - (0.1 * 50000)  # Balance reduced by $5000
        assert 'BTC/USDT' in portfolio.positions
        position = portfolio.positions['BTC/USDT']
        assert position.quantity == 0.1
        assert position.entry_price == 50000

    def test_portfolio_sell_order(self):
        """Test selling from portfolio."""
        portfolio = Portfolio(initial_balance=10000)

        # First buy
        portfolio.execute_order('BTC/USDT', OrderType.BUY, 0.1, 50000)

        # Then sell at higher price
        success = portfolio.execute_order('BTC/USDT', OrderType.SELL, 0.1, 55000)

        assert success
        assert portfolio.balance > 10000  # Should have profit
        assert 'BTC/USDT' not in portfolio.positions  # Position should be closed

    def test_portfolio_insufficient_balance(self):
        """Test order rejection due to insufficient balance."""
        portfolio = Portfolio(initial_balance=1000)

        # Try to buy more than balance allows
        success = portfolio.execute_order('BTC/USDT', OrderType.BUY, 1.0, 2000)

        assert not success
        assert portfolio.balance == 1000  # Balance unchanged
        assert len(portfolio.positions) == 0

    def test_portfolio_insufficient_position(self):
        """Test sell order rejection due to insufficient position."""
        portfolio = Portfolio(initial_balance=10000)

        # Try to sell without owning
        success = portfolio.execute_order('BTC/USDT', OrderType.SELL, 0.1, 50000)

        assert not success
        assert portfolio.balance == 10000  # Balance unchanged


class TestPaperTradingEngine:
    """Test suite for paper trading engine."""

    def setup_method(self):
        """Skip test if paper trading is not available."""
        if not PAPER_TRADING_AVAILABLE:
            pytest.skip("Paper trading module not available")

    def test_engine_initialization(self):
        """Test paper trading engine initialization."""
        engine = PaperTradingEngine()

        assert engine.portfolio is not None
        assert isinstance(engine.portfolio, Portfolio)
        assert hasattr(engine, 'execute_trade')
        assert hasattr(engine, 'get_portfolio_status')

    def test_execute_trade_buy(self):
        """Test executing a buy trade."""
        with patch('paper_trading.PAPER_TRADING_ENABLED', True), \
             patch('paper_trading.PAPER_TRADING_SAVE_TRADES', False), \
             patch('paper_trading.PAPER_TRADING_MAX_TRADES_PER_DAY', 100), \
             patch('paper_trading.PAPER_TRADING_MAX_POSITION_SIZE', 1.0), \
             patch('paper_trading.PAPER_TRADING_MAX_DRAWDOWN', 1.0), \
             patch.object(PaperTradingEngine, '_load_state', lambda self: None):  # Prevent loading old state
            engine = PaperTradingEngine()

            # Execute buy trade
            result = engine.execute_trade('BTC/USDT', OrderType.BUY, 0.1, 50000)

            assert result['success'] is True
            assert 'order_id' in result
            assert result['symbol'] == 'BTC/USDT'
            assert result['side'] == 'BUY'
            assert result['quantity'] == 0.1
            # Note: price includes slippage, so check it's close to original price
            assert abs(result['price'] - 50000) < 1000  # Allow for slippage

    def test_execute_trade_sell(self):
        """Test executing a sell trade."""
        with patch('paper_trading.PAPER_TRADING_ENABLED', True), \
             patch('paper_trading.PAPER_TRADING_MAX_POSITION_SIZE', 1.0), \
             patch('paper_trading.PAPER_TRADING_SAVE_TRADES', False), \
             patch.object(PaperTradingEngine, '_load_state', lambda self: None):
            engine = PaperTradingEngine()

            # First buy
            engine.execute_trade('BTC/USDT', OrderType.BUY, 0.1, 50000)

            # Then sell
            result = engine.execute_trade('BTC/USDT', OrderType.SELL, 0.1, 55000)

            assert result['success'] is True
            assert result['side'] == 'SELL'
            assert result['quantity'] == 0.1

    def test_portfolio_status(self):
        """Test getting portfolio status."""
        with patch.object(PaperTradingEngine, '_load_state', lambda self: None):
            engine = PaperTradingEngine()

            status = engine.get_portfolio_status()

            assert isinstance(status, dict)
            assert 'balance' in status
            assert 'positions' in status
            assert 'total_value' in status
            assert 'pnl' in status

    def test_risk_management(self):
        """Test risk management features."""
        with patch('paper_trading.PAPER_TRADING_ENABLED', True), \
             patch('paper_trading.PAPER_TRADING_STOP_LOSS_ENABLED', True), \
             patch('paper_trading.PAPER_TRADING_TAKE_PROFIT_ENABLED', True), \
             patch('paper_trading.PAPER_TRADING_MAX_POSITION_SIZE', 1.0), \
             patch.object(PaperTradingEngine, '_load_state', lambda self: None):
            engine = PaperTradingEngine()

            # Buy position with stop loss
            order = engine.place_order('BTC/USDT', OrderType.BUY, 0.1, 50000, stop_loss=45000)
            assert order is not None
            assert order.status.name == 'FILLED'

            # Test stop loss trigger (price drops below threshold)
            stop_loss_price = 45000  # 10% drop
            result = engine.check_stop_loss('BTC/USDT', stop_loss_price)

            # Should trigger stop loss
            assert result is True

    def test_max_position_size_limit(self):
        """Test maximum position size limits."""
        with patch('paper_trading.PAPER_TRADING_ENABLED', True), \
             patch('paper_trading.PAPER_TRADING_MAX_POSITION_SIZE', 0.05), \
             patch.object(PaperTradingEngine, '_load_state', lambda self: None):  # 5% max
            engine = PaperTradingEngine()

            # Try to buy large position (should be limited)
            result = engine.execute_trade('BTC/USDT', OrderType.BUY, 1.0, 50000)

            # Check that position size was limited
            if 'BTC/USDT' in engine.portfolio.positions:
                position = engine.portfolio.positions['BTC/USDT']
                position_value = position.quantity * 50000
                portfolio_value = engine.portfolio.balance + position_value
                position_pct = position_value / portfolio_value
                assert position_pct <= 0.05  # Should not exceed 5%

    def test_daily_trade_limit(self):
        """Test daily trade limits."""
        with patch('paper_trading.PAPER_TRADING_ENABLED', True), \
             patch('paper_trading.PAPER_TRADING_MAX_POSITION_SIZE', 1.0), \
             patch('paper_trading.PAPER_TRADING_MAX_TRADES_PER_DAY', 3), \
             patch.object(PaperTradingEngine, '_load_state', lambda self: None):
            engine = PaperTradingEngine()

            # Execute multiple trades
            for i in range(5):
                engine.execute_trade('BTC/USDT', OrderType.BUY, 0.01, 50000)

            # Check that only allowed number of trades were executed
            assert len(engine.portfolio.trades) <= 3

    def test_fee_calculation(self):
        """Test trading fee calculations."""
        with patch('paper_trading.PAPER_TRADING_ENABLED', True), \
             patch('paper_trading.PAPER_TRADING_MAX_POSITION_SIZE', 1.0), \
             patch('paper_trading.PAPER_TRADING_FEE_RATE', 0.001), \
             patch.object(PaperTradingEngine, '_load_state', lambda self: None):  # 0.1% fee
            engine = PaperTradingEngine()

            initial_balance = engine.portfolio.balance

            # Execute trade
            engine.execute_trade('BTC/USDT', OrderType.BUY, 0.1, 50000)

            # Check that fee was deducted
            expected_fee = 0.1 * 50000 * 0.001  # quantity * price * fee_rate
            final_balance = engine.portfolio.balance

            assert initial_balance - final_balance > 0.1 * 50000  # More than just position cost


class TestPaperTradingIntegration:
    """Integration tests for paper trading."""

    def setup_method(self):
        """Skip test if paper trading is not available."""
        if not PAPER_TRADING_AVAILABLE:
            pytest.skip("Paper trading module not available")

    def test_complete_trading_scenario(self):
        """Test a complete trading scenario."""
        with patch('paper_trading.PAPER_TRADING_ENABLED', True), \
             patch('paper_trading.PAPER_TRADING_MAX_POSITION_SIZE', 1.0), \
             patch.object(PaperTradingEngine, '_load_state', lambda self: None):
            engine = PaperTradingEngine()

            initial_balance = engine.portfolio.balance

            # Execute a series of trades
            trades = [
                ('BTC/USDT', OrderType.BUY, 0.01, 50000),  # Smaller position
                ('BTC/USDT', OrderType.SELL, 0.005, 55000),  # Partial sell at profit
                ('ETH/USDT', OrderType.BUY, 0.1, 3000),     # Smaller ETH position
                ('BTC/USDT', OrderType.SELL, 0.005, 52000),  # Sell remaining at smaller profit
                ('ETH/USDT', OrderType.SELL, 0.1, 2800),    # Sell at loss
            ]

            for symbol, side, quantity, price in trades:
                result = engine.execute_trade(symbol, side, quantity, price)
                assert result['success'] is True

            # Check final state
            final_status = engine.get_portfolio_status()

            assert final_status['balance'] != initial_balance  # Balance changed
            assert len(final_status['positions']) == 0  # All positions closed
            # Note: Only sell orders create trades, so we expect 2 trades (BTC sell and ETH sell)
            assert len(engine.portfolio.trades) == 2

    def test_performance_tracking(self):
        """Test performance tracking over time."""
        with patch('paper_trading.PAPER_TRADING_ENABLED', True), \
             patch('paper_trading.PAPER_TRADING_MAX_POSITION_SIZE', 1.0), \
             patch.object(PaperTradingEngine, '_load_state', lambda self: None):
            engine = PaperTradingEngine()

            # Execute several trades
            for i in range(10):
                price = 50000 + i * 100
                engine.execute_trade('BTC/USDT', OrderType.BUY, 0.01, price)
                engine.execute_trade('BTC/USDT', OrderType.SELL, 0.01, price + 200)

            # Check performance metrics
            status = engine.get_portfolio_status()

            assert 'pnl' in status
            assert 'total_value' in status
            assert isinstance(status['pnl'], (int, float))
            assert isinstance(status['total_value'], (int, float))