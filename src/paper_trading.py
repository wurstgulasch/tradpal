"""
Paper Trading Module for TradPal Trading System.

This module provides paper trading capabilities for risk-free strategy testing
and validation. It simulates real trading conditions including fees, slippage,
and position management.

Features:
- Realistic order execution simulation
- Portfolio management with position tracking
- Performance metrics and reporting
- Risk management integration
- Trade logging and analysis

Author: TradPal Team
Version: 2.5.0
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import logging
from enum import Enum

from config.settings import (
    PAPER_TRADING_ENABLED, PAPER_TRADING_INITIAL_BALANCE, PAPER_TRADING_FEE_RATE,
    PAPER_TRADING_SLIPPAGE, PAPER_TRADING_MAX_POSITION_SIZE, PAPER_TRADING_DATA_SOURCE,
    PAPER_TRADING_SAVE_TRADES, PAPER_TRADING_TRADE_LOG_FILE, PAPER_TRADING_PERFORMANCE_LOG_FILE,
    PAPER_TRADING_STOP_LOSS_ENABLED, PAPER_TRADING_TAKE_PROFIT_ENABLED,
    PAPER_TRADING_MAX_DRAWDOWN, PAPER_TRADING_MAX_TRADES_PER_DAY, SYMBOL
)

logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Order types for paper trading."""
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = None

    def update_pnl(self, current_price: float) -> float:
        """Update position P&L with current price."""
        self.current_price = current_price
        self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        return self.unrealized_pnl

    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary."""
        data = asdict(self)
        data['entry_time'] = self.entry_time.isoformat()
        return data

@dataclass
class Order:
    """Represents a trading order."""
    order_id: str
    symbol: str
    order_type: OrderType
    quantity: float
    price: float
    timestamp: datetime
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    fees: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary."""
        data = asdict(self)
        data['order_type'] = self.order_type.value
        data['status'] = self.status.value
        data['timestamp'] = self.timestamp.isoformat()
        return data

@dataclass
class Trade:
    """Represents a completed trade."""
    trade_id: str
    symbol: str
    order_type: OrderType
    quantity: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    fees: float
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary."""
        data = asdict(self)
        data['order_type'] = self.order_type.value
        data['entry_time'] = self.entry_time.isoformat()
        data['exit_time'] = self.exit_time.isoformat()
        return data

@dataclass
class Portfolio:
    """Portfolio management for paper trading."""
    balance: float
    positions: Dict[str, Position]
    trades: List[Trade]
    initial_balance: float

    def __init__(self, initial_balance: float = PAPER_TRADING_INITIAL_BALANCE):
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.positions = {}
        self.trades = []

    @property
    def total_value(self) -> float:
        """Calculate total portfolio value."""
        position_value = sum(pos.unrealized_pnl for pos in self.positions.values())
        return self.balance + position_value

    @property
    def total_pnl(self) -> float:
        """Calculate total P&L."""
        return self.total_value - self.initial_balance

    @property
    def pnl_percentage(self) -> float:
        """Calculate P&L as percentage."""
        return (self.total_pnl / self.initial_balance) * 100

    @property
    def max_drawdown(self) -> float:
        """Calculate maximum drawdown from peak."""
        if not self.trades:
            return 0.0

        # Calculate cumulative P&L
        cumulative_pnl = [0.0]
        for trade in self.trades:
            cumulative_pnl.append(cumulative_pnl[-1] + trade.pnl)

        # Find peak and maximum drawdown
        peak = cumulative_pnl[0]
        max_dd = 0.0

        for pnl in cumulative_pnl:
            if pnl > peak:
                peak = pnl
            dd = peak - pnl
            max_dd = max(max_dd, dd)

        return max_dd

    def add_position(self, position: Position) -> None:
        """Add a position to the portfolio."""
        self.positions[position.symbol] = position

    def remove_position(self, symbol: str) -> Optional[Position]:
        """Remove and return a position from the portfolio."""
        return self.positions.pop(symbol, None)

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self.positions.get(symbol)

    def update_positions(self, prices: Dict[str, float]) -> None:
        """Update all positions with current prices."""
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.update_pnl(prices[symbol])

    def to_dict(self) -> Dict[str, Any]:
        """Convert portfolio to dictionary."""
        return {
            'balance': self.balance,
            'total_value': self.total_value,
            'total_pnl': self.total_pnl,
            'pnl_percentage': self.pnl_percentage,
            'max_drawdown': self.max_drawdown,
            'positions': {symbol: pos.to_dict() for symbol, pos in self.positions.items()},
            'trades_count': len(self.trades),
            'initial_balance': self.initial_balance
        }

    def execute_order(self, symbol: str, order_type: OrderType, quantity: float, price: float) -> bool:
        """
        Execute an order on the portfolio.
        
        Args:
            symbol: Trading symbol
            order_type: BUY or SELL
            quantity: Order quantity
            price: Order price
            
        Returns:
            True if order executed successfully, False otherwise
        """
        try:
            if order_type == OrderType.BUY:
                cost = quantity * price
                if cost > self.balance:
                    return False
                self.balance -= cost
                
                # Create position
                position = Position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=price,
                    entry_time=datetime.now(),
                    current_price=price
                )
                self.add_position(position)
                return True
                
            elif order_type == OrderType.SELL:
                position = self.get_position(symbol)
                if not position or position.quantity < quantity:
                    return False
                
                # Calculate P&L
                pnl = (price - position.entry_price) * quantity
                self.balance += quantity * price
                
                # Create trade record
                trade = Trade(
                    trade_id=f"trade_{len(self.trades) + 1}",
                    symbol=symbol,
                    order_type=OrderType.SELL,
                    quantity=quantity,
                    entry_price=position.entry_price,
                    exit_price=price,
                    entry_time=position.entry_time,
                    exit_time=datetime.now(),
                    pnl=pnl,
                    fees=0.0
                )
                self.trades.append(trade)
                
                # Update or remove position
                if position.quantity == quantity:
                    self.remove_position(symbol)
                else:
                    position.quantity -= quantity
                    
                return True
                
        except Exception as e:
            logger.error(f"Failed to execute order: {e}")
            return False
            
        return False

class PaperTradingEngine:
    """
    Main paper trading engine for simulating trades.

    Provides realistic trading simulation with fees, slippage, and risk management.
    """

    def __init__(self):
        self.portfolio = Portfolio()
        self.pending_orders: List[Order] = []
        self.completed_orders: List[Order] = []
        self.order_counter = 0
        self.trade_counter = 0
        self.daily_trade_count = 0
        self.last_reset_date = datetime.now().date()

        # Load existing data if available
        self._load_state()

        logger.info("PaperTradingEngine initialized")

    def _load_state(self) -> None:
        """Load previous trading state from files."""
        try:
            # Load portfolio state
            if os.path.exists(PAPER_TRADING_PERFORMANCE_LOG_FILE):
                with open(PAPER_TRADING_PERFORMANCE_LOG_FILE, 'r') as f:
                    data = json.load(f)
                    self.portfolio.balance = data.get('balance', PAPER_TRADING_INITIAL_BALANCE)
                    self.portfolio.initial_balance = data.get('initial_balance', PAPER_TRADING_INITIAL_BALANCE)

            # Load trades
            if os.path.exists(PAPER_TRADING_TRADE_LOG_FILE):
                with open(PAPER_TRADING_TRADE_LOG_FILE, 'r') as f:
                    trades_data = json.load(f)
                    for trade_data in trades_data:
                        trade = Trade(**trade_data)
                        self.portfolio.trades.append(trade)
                        self.trade_counter = max(self.trade_counter, int(trade.trade_id.split('_')[1]))

        except Exception as e:
            logger.warning(f"Failed to load paper trading state: {e}")

    def _save_state(self) -> None:
        """Save current trading state to files."""
        if not PAPER_TRADING_SAVE_TRADES:
            return

        try:
            # Save portfolio performance
            os.makedirs(os.path.dirname(PAPER_TRADING_PERFORMANCE_LOG_FILE), exist_ok=True)
            with open(PAPER_TRADING_PERFORMANCE_LOG_FILE, 'w') as f:
                json.dump(self.portfolio.to_dict(), f, indent=2)

            # Save trades
            trades_data = [trade.to_dict() for trade in self.portfolio.trades]
            with open(PAPER_TRADING_TRADE_LOG_FILE, 'w') as f:
                json.dump(trades_data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save paper trading state: {e}")

    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        self.order_counter += 1
        return f"order_{self.order_counter}"

    def _generate_trade_id(self) -> str:
        """Generate unique trade ID."""
        self.trade_counter += 1
        return f"trade_{self.trade_counter}"

    def _apply_slippage(self, price: float, order_type: OrderType) -> float:
        """Apply slippage to order price."""
        slippage = price * PAPER_TRADING_SLIPPAGE
        if order_type == OrderType.BUY:
            return price + slippage  # Pay more for buys
        else:
            return price - slippage  # Get less for sells

    def _calculate_fees(self, quantity: float, price: float) -> float:
        """Calculate trading fees."""
        return quantity * price * PAPER_TRADING_FEE_RATE

    def _check_daily_trade_limit(self) -> bool:
        """Check if daily trade limit is exceeded."""
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.daily_trade_count = 0
            self.last_reset_date = today

        return self.daily_trade_count < PAPER_TRADING_MAX_TRADES_PER_DAY

    def _check_max_drawdown(self) -> bool:
        """Check if maximum drawdown limit is exceeded."""
        if PAPER_TRADING_MAX_DRAWDOWN <= 0:
            return True

        current_drawdown = self.portfolio.max_drawdown
        max_allowed_drawdown = self.portfolio.initial_balance * PAPER_TRADING_MAX_DRAWDOWN

        return current_drawdown <= max_allowed_drawdown

    def place_order(self, symbol: str, order_type: OrderType, quantity: float,
                   price: float, stop_loss: Optional[float] = None,
                   take_profit: Optional[float] = None) -> Optional[Order]:
        """
        Place a paper trading order.

        Args:
            symbol: Trading symbol
            order_type: BUY or SELL
            quantity: Order quantity
            price: Order price
            stop_loss: Stop loss price
            take_profit: Take profit price

        Returns:
            Order object if successful, None otherwise
        """
        if not PAPER_TRADING_ENABLED:
            return None

        # Check daily trade limit
        if not self._check_daily_trade_limit():
            logger.warning("Daily trade limit exceeded")
            return None

        # Check max drawdown
        if not self._check_max_drawdown():
            logger.warning("Maximum drawdown limit exceeded")
            return None

        # Check position size limit (only for BUY orders)
        if order_type == OrderType.BUY:
            max_position_value = self.portfolio.balance * PAPER_TRADING_MAX_POSITION_SIZE
            order_value = quantity * price

            if order_value > max_position_value:
                logger.warning(f"Order value {order_value} exceeds max position size {max_position_value}")
                return None

        # Create order
        order = Order(
            order_id=self._generate_order_id(),
            symbol=symbol,
            order_type=order_type,
            quantity=quantity,
            price=price,
            timestamp=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit
        )

        # Execute order immediately (paper trading)
        executed_order = self._execute_order(order)
        if executed_order:
            self.completed_orders.append(executed_order)
            self.daily_trade_count += 1

        return executed_order

    def _execute_order(self, order: Order) -> Optional[Order]:
        """
        Execute a paper trading order with realistic simulation.

        Args:
            order: Order to execute

        Returns:
            Executed order or None if failed
        """
        try:
            # Apply slippage
            execution_price = self._apply_slippage(order.price, order.order_type)

            # Calculate fees
            fees = self._calculate_fees(order.quantity, execution_price)

            # Check if we have enough balance for buy orders
            if order.order_type == OrderType.BUY:
                total_cost = (order.quantity * execution_price) + fees
                if total_cost > self.portfolio.balance:
                    logger.warning(f"Insufficient balance for order: {total_cost} > {self.portfolio.balance}")
                    order.status = OrderStatus.REJECTED
                    return order

            # Update order status
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.filled_price = execution_price
            order.fees = fees

            # Update portfolio
            if order.order_type == OrderType.BUY:
                # Create new position
                position = Position(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    entry_price=execution_price,
                    entry_time=order.timestamp,
                    current_price=execution_price,
                    stop_loss=order.stop_loss,
                    take_profit=order.take_profit,
                    metadata={'order_id': order.order_id}
                )
                self.portfolio.add_position(position)
                self.portfolio.balance -= (order.quantity * execution_price) + fees

            else:  # SELL
                # Close existing position
                position = self.portfolio.get_position(order.symbol)
                if position:
                    # Calculate P&L
                    pnl = (execution_price - position.entry_price) * position.quantity
                    fees_total = fees

                    # Create trade record
                    trade = Trade(
                        trade_id=self._generate_trade_id(),
                        symbol=order.symbol,
                        order_type=OrderType.SELL,
                        quantity=position.quantity,
                        entry_price=position.entry_price,
                        exit_price=execution_price,
                        entry_time=position.entry_time,
                        exit_time=order.timestamp,
                        pnl=pnl - fees_total,
                        fees=fees_total,
                        metadata={'order_id': order.order_id}
                    )

                    self.portfolio.trades.append(trade)
                    self.portfolio.balance += (order.quantity * execution_price) - fees
                    self.portfolio.remove_position(order.symbol)

            # Save state
            self._save_state()

            logger.info(f"Executed paper order: {order.order_type.value} {order.quantity} {order.symbol} @ {execution_price}")
            return order

        except Exception as e:
            logger.error(f"Failed to execute order {order.order_id}: {e}")
            order.status = OrderStatus.REJECTED
            return order

    def update_portfolio(self, current_prices: Dict[str, float]) -> None:
        """
        Update portfolio with current market prices.

        Args:
            current_prices: Dictionary of symbol -> price
        """
        self.portfolio.update_positions(current_prices)

        # Check stop loss and take profit
        if PAPER_TRADING_STOP_LOSS_ENABLED or PAPER_TRADING_TAKE_PROFIT_ENABLED:
            self._check_stop_conditions(current_prices)

    def _check_stop_conditions(self, current_prices: Dict[str, float]) -> None:
        """Check and execute stop loss and take profit orders."""
        positions_to_close = []

        for symbol, position in self.portfolio.positions.items():
            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]

            # Check stop loss
            if PAPER_TRADING_STOP_LOSS_ENABLED and position.stop_loss:
                if position.entry_price > position.stop_loss:  # Long position
                    if current_price <= position.stop_loss:
                        positions_to_close.append((symbol, current_price, "STOP_LOSS"))
                else:  # Short position
                    if current_price >= position.stop_loss:
                        positions_to_close.append((symbol, current_price, "STOP_LOSS"))

            # Check take profit
            if PAPER_TRADING_TAKE_PROFIT_ENABLED and position.take_profit:
                if position.entry_price < position.take_profit:  # Long position
                    if current_price >= position.take_profit:
                        positions_to_close.append((symbol, current_price, "TAKE_PROFIT"))
                else:  # Short position
                    if current_price <= position.take_profit:
                        positions_to_close.append((symbol, current_price, "TAKE_PROFIT"))

        # Execute stop orders
        for symbol, price, reason in positions_to_close:
            self._execute_stop_order(symbol, price, reason)

    def _execute_stop_order(self, symbol: str, price: float, reason: str) -> None:
        """Execute a stop loss or take profit order."""
        position = self.portfolio.get_position(symbol)
        if not position:
            return

        # Create sell order for stop
        order = Order(
            order_id=self._generate_order_id(),
            symbol=symbol,
            order_type=OrderType.SELL,
            quantity=position.quantity,
            price=price,
            timestamp=datetime.now(),
            metadata={'stop_reason': reason}
        )

        # Execute the order
        executed_order = self._execute_order(order)
        if executed_order:
            logger.info(f"Executed {reason} for {symbol} @ {price}")

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary with performance metrics."""
        return self.portfolio.to_dict()

    def get_trading_statistics(self) -> Dict[str, Any]:
        """Get detailed trading statistics."""
        if not self.portfolio.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0
            }

        trades = self.portfolio.trades
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]

        win_rate = len(winning_trades) / len(trades)

        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = abs(np.mean([t.pnl for t in losing_trades])) if losing_trades else 0

        profit_factor = (sum(t.pnl for t in winning_trades) /
                        abs(sum(t.pnl for t in losing_trades))) if losing_trades else float('inf')

        # Calculate Sharpe ratio (simplified)
        returns = [t.pnl / self.portfolio.initial_balance for t in trades]
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(365)  # Annualized
        else:
            sharpe_ratio = 0.0

        return {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio
        }

    def reset_portfolio(self) -> None:
        """Reset portfolio to initial state."""
        self.portfolio = Portfolio()
        self.pending_orders = []
        self.completed_orders = []
        self.daily_trade_count = 0
        self._save_state()
        logger.info("Portfolio reset to initial state")

    def execute_trade(self, symbol: str, order_type: OrderType, quantity: float, price: float) -> Dict[str, Any]:
        """
        Execute a trade through the paper trading engine.
        
        Args:
            symbol: Trading symbol
            order_type: BUY or SELL
            quantity: Trade quantity
            price: Trade price
            
        Returns:
            Dictionary with trade result
        """
        order = self.place_order(symbol, order_type, quantity, price)
        if order and order.status == OrderStatus.FILLED:
            return {
                'success': True,
                'order_id': order.order_id,
                'symbol': symbol,
                'side': order_type.value,
                'quantity': quantity,
                'price': order.filled_price,
                'fees': order.fees
            }
        else:
            return {'success': False, 'error': 'Order failed'}

    def get_portfolio_status(self) -> Dict[str, Any]:
        """
        Get current portfolio status.
        
        Returns:
            Dictionary with portfolio information
        """
        summary = self.get_portfolio_summary()
        return {
            'balance': summary['balance'],
            'positions': summary['positions'],
            'total_value': summary['total_value'],
            'pnl': summary['total_pnl'],
            'pnl_percentage': summary['pnl_percentage'],
            'max_drawdown': summary['max_drawdown'],
            'trades_count': summary['trades_count']
        }

    def check_stop_loss(self, symbol: str, current_price: float) -> bool:
        """
        Check if stop loss should be triggered for a position.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            
        Returns:
            True if stop loss was triggered
        """
        position = self.portfolio.get_position(symbol)
        if not position or not position.stop_loss:
            return False
            
        # Check if stop loss is triggered
        if position.entry_price > position.stop_loss:  # Long position
            if current_price <= position.stop_loss:
                self._execute_stop_order(symbol, current_price, "STOP_LOSS")
                return True
        else:  # Short position
            if current_price >= position.stop_loss:
                self._execute_stop_order(symbol, current_price, "STOP_LOSS")
                return True
                
        return False

# Global paper trading engine instance
paper_trading_engine = PaperTradingEngine()

def execute_paper_trade(symbol: str, side: str, price: float, size: float,
                       timestamp: Optional[datetime] = None, reason: str = "",
                       signal_data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Execute a paper trade with the given parameters.

    Args:
        symbol: Trading symbol
        side: 'BUY' or 'SELL'
        price: Trade price
        size: Trade size
        timestamp: Trade timestamp
        reason: Reason for the trade
        signal_data: Additional signal data

    Returns:
        Trade result dictionary or None if failed
    """
    if not PAPER_TRADING_ENABLED:
        return None

    try:
        order_type = OrderType.BUY if side.upper() == 'BUY' else OrderType.SELL

        # Extract stop loss and take profit from signal data if available
        stop_loss = None
        take_profit = None
        if signal_data:
            stop_loss = signal_data.get('stop_loss')
            take_profit = signal_data.get('take_profit')

        # Place the order
        order = paper_trading_engine.place_order(
            symbol=symbol,
            order_type=order_type,
            quantity=size,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

        if order and order.status == OrderStatus.FILLED:
            return {
                'order_id': order.order_id,
                'symbol': order.symbol,
                'side': order.order_type.value,
                'price': order.filled_price,
                'size': order.filled_quantity,
                'fees': order.fees,
                'timestamp': order.timestamp.isoformat(),
                'reason': reason
            }

        return None

    except Exception as e:
        logger.error(f"Failed to execute paper trade: {e}")
        return None

def get_paper_trading_status() -> Dict[str, Any]:
    """
    Get current paper trading status and performance.

    Returns:
        Dictionary with portfolio and performance data
    """
    if not PAPER_TRADING_ENABLED:
        return {'enabled': False}

    portfolio_summary = paper_trading_engine.get_portfolio_summary()
    trading_stats = paper_trading_engine.get_trading_statistics()

    return {
        'enabled': True,
        'portfolio': portfolio_summary,
        'statistics': trading_stats,
        'last_update': datetime.now().isoformat()
    }

def get_paper_portfolio_summary() -> Dict[str, Any]:
    """Get summary of the paper trading portfolio."""
    if not PAPER_TRADING_ENABLED:
        return {}
    return paper_trading_engine.get_portfolio_summary()

def reset_paper_portfolio() -> None:
    """Reset the paper trading portfolio to initial state."""
    if PAPER_TRADING_ENABLED:
        paper_trading_engine.reset_portfolio()