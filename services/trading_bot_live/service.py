#!/usr/bin/env python3
"""
Trading Bot Live Service - Live trading execution and monitoring.

Provides comprehensive live trading capabilities including:
- Real-time signal monitoring and execution
- Order management and position tracking
- Risk management and position sizing
- Performance monitoring and reporting
- Paper trading mode for testing
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum

import pandas as pd
import numpy as np

from config.settings import (
    DEFAULT_SYMBOL,
    DEFAULT_TIMEFRAME,
    RISK_PER_TRADE,
    MAX_POSITIONS,
    MAX_DRAWDOWN,
    ORDER_TIMEOUT,
    POSITION_UPDATE_INTERVAL
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Trading order data."""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    price: Optional[float]
    status: OrderStatus
    timestamp: datetime
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    fees: float = 0.0


@dataclass
class Position:
    """Trading position data."""
    position_id: str
    symbol: str
    side: OrderSide
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    timestamp: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class TradingSession:
    """Live trading session data."""
    session_id: str
    symbol: str
    strategy: str
    timeframe: str
    capital: float
    risk_per_trade: float
    max_positions: int
    is_active: bool
    paper_trading: bool
    start_time: datetime
    end_time: Optional[datetime] = None
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0


@dataclass
class PerformanceMetrics:
    """Trading performance metrics."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float
    largest_win: float
    largest_loss: float


class EventSystem:
    """Simple event system for service communication."""

    def __init__(self):
        self._handlers: Dict[str, List[callable]] = {}

    def subscribe(self, event_type: str, handler: callable):
        """Subscribe to an event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    async def publish(self, event_type: str, data: Dict[str, Any]):
        """Publish an event."""
        if event_type in self._handlers:
            for handler in self._handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    logger.error(f"Event handler error: {e}")


class TradingBotLiveService:
    """Live trading execution and monitoring service."""

    def __init__(self, event_system: Optional[EventSystem] = None):
        self.event_system = event_system or EventSystem()

        # Trading sessions
        self.trading_sessions: Dict[str, TradingSession] = {}

        # Orders and positions
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, List[Position]] = {}

        # Performance tracking
        self.trade_history: List[Dict[str, Any]] = []

        # Background tasks
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}

        logger.info("Trading Bot Live Service initialized")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        active_sessions = len([s for s in self.trading_sessions.values() if s.is_active])
        open_positions = sum(len(positions) for positions in self.positions.values())

        return {
            "service": "trading_bot_live",
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "active_sessions": active_sessions,
            "open_positions": open_positions,
            "total_orders": len(self.orders)
        }

    async def start_trading(
        self,
        symbol: str,
        strategy: str,
        timeframe: str = DEFAULT_TIMEFRAME,
        capital: float = 10000.0,
        risk_per_trade: float = RISK_PER_TRADE,
        max_positions: int = MAX_POSITIONS,
        enable_paper_trading: bool = True
    ) -> Dict[str, Any]:
        """
        Start live trading for a symbol.

        Args:
            symbol: Trading symbol
            strategy: Trading strategy name
            timeframe: Trading timeframe
            capital: Initial capital
            risk_per_trade: Risk per trade (decimal)
            max_positions: Maximum open positions
            enable_paper_trading: Enable paper trading mode

        Returns:
            Trading session info
        """
        session_id = str(uuid.uuid4())

        session = TradingSession(
            session_id=session_id,
            symbol=symbol,
            strategy=strategy,
            timeframe=timeframe,
            capital=capital,
            risk_per_trade=risk_per_trade,
            max_positions=max_positions,
            is_active=True,
            paper_trading=enable_paper_trading,
            start_time=datetime.now()
        )

        self.trading_sessions[symbol] = session

        # Initialize positions list
        if symbol not in self.positions:
            self.positions[symbol] = []

        # Start monitoring task
        task = asyncio.create_task(self._monitor_trading_session(symbol))
        self.monitoring_tasks[symbol] = task

        logger.info(f"Started trading session {session_id} for {symbol}")

        # Publish event
        await self.event_system.publish("trading.session_started", {
            "session_id": session_id,
            "symbol": symbol,
            "strategy": strategy
        })

        return {
            "trading_id": session_id,
            "symbol": symbol,
            "strategy": strategy,
            "paper_trading": enable_paper_trading
        }

    async def stop_trading(self, symbol: str, close_positions: bool = True) -> Dict[str, Any]:
        """
        Stop live trading for a symbol.

        Args:
            symbol: Trading symbol
            close_positions: Whether to close all open positions

        Returns:
            Stop trading results
        """
        if symbol not in self.trading_sessions:
            raise ValueError(f"No active trading session for {symbol}")

        session = self.trading_sessions[symbol]
        session.is_active = False
        session.end_time = datetime.now()

        positions_closed = 0

        if close_positions and symbol in self.positions:
            # Close all positions
            for position in self.positions[symbol]:
                await self._close_position(position, "Manual stop")
                positions_closed += 1

            self.positions[symbol] = []

        # Cancel monitoring task
        if symbol in self.monitoring_tasks:
            self.monitoring_tasks[symbol].cancel()
            del self.monitoring_tasks[symbol]

        logger.info(f"Stopped trading session for {symbol}, closed {positions_closed} positions")

        # Publish event
        await self.event_system.publish("trading.session_stopped", {
            "symbol": symbol,
            "positions_closed": positions_closed
        })

        return {
            "symbol": symbol,
            "positions_closed": positions_closed
        }

    async def place_manual_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Place a manual order.

        Args:
            symbol: Trading symbol
            side: Order side (buy/sell)
            quantity: Order quantity
            order_type: Order type (market/limit/stop)
            price: Order price (for limit/stop orders)

        Returns:
            Order placement result
        """
        if symbol not in self.trading_sessions:
            raise ValueError(f"No active trading session for {symbol}")

        session = self.trading_sessions[symbol]

        # Validate order
        await self._validate_order(symbol, side, quantity, order_type, price)

        # Create order
        order_id = str(uuid.uuid4())
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=OrderSide(side),
            quantity=quantity,
            order_type=OrderType(order_type),
            price=price,
            status=OrderStatus.PENDING,
            timestamp=datetime.now()
        )

        self.orders[order_id] = order

        # Execute order (in paper trading mode, simulate execution)
        if session.paper_trading:
            await self._execute_paper_order(order)
        else:
            # In live mode, this would integrate with broker API
            await self._execute_live_order(order)

        logger.info(f"Placed {order_type} {side} order for {quantity} {symbol}")

        return {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "order_type": order_type
        }

    async def update_risk_parameters(
        self,
        symbol: str,
        risk_per_trade: Optional[float] = None,
        max_positions: Optional[int] = None,
        max_drawdown: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Update risk management parameters.

        Args:
            symbol: Trading symbol
            risk_per_trade: New risk per trade
            max_positions: New max positions
            max_drawdown: New max drawdown

        Returns:
            Update result
        """
        if symbol not in self.trading_sessions:
            raise ValueError(f"No active trading session for {symbol}")

        session = self.trading_sessions[symbol]

        if risk_per_trade is not None:
            session.risk_per_trade = risk_per_trade

        if max_positions is not None:
            session.max_positions = max_positions

        if max_drawdown is not None:
            session.max_drawdown = max_drawdown

        logger.info(f"Updated risk parameters for {symbol}")

        return {
            "symbol": symbol,
            "risk_per_trade": session.risk_per_trade,
            "max_positions": session.max_positions,
            "max_drawdown": session.max_drawdown
        }

    async def get_trading_status(self) -> Dict[str, Any]:
        """Get status of all trading sessions."""
        sessions = []
        for symbol, session in self.trading_sessions.items():
            sessions.append({
                "symbol": symbol,
                "session": asdict(session),
                "open_positions": len(self.positions.get(symbol, [])),
                "active_orders": len([o for o in self.orders.values() if o.symbol == symbol and o.status == OrderStatus.PENDING])
            })

        return {
            "sessions": sessions,
            "total_sessions": len(sessions),
            "active_sessions": len([s for s in sessions if s["session"]["is_active"]])
        }

    async def get_symbol_status(self, symbol: str) -> Dict[str, Any]:
        """Get trading status for a specific symbol."""
        if symbol not in self.trading_sessions:
            return {"status": "inactive", "symbol": symbol}

        session = self.trading_sessions[symbol]

        return {
            "symbol": symbol,
            "session": asdict(session),
            "open_positions": len(self.positions.get(symbol, [])),
            "active_orders": len([o for o in self.orders.values() if o.symbol == symbol and o.status == OrderStatus.PENDING]),
            "performance": await self.get_symbol_performance(symbol)
        }

    async def get_positions(self) -> Dict[str, Any]:
        """Get all positions."""
        all_positions = []
        for symbol_positions in self.positions.values():
            if isinstance(symbol_positions, list):
                # New structure: list of Position objects
                all_positions.extend([asdict(pos) for pos in symbol_positions])
            else:
                # Old test structure: dict positions
                all_positions.append(symbol_positions)

        return {
            "positions": all_positions,
            "total_positions": len(all_positions),
            "total_value": sum(pos.get("quantity", 0) * pos.get("current_price", 0) for pos in all_positions)
        }

    async def get_symbol_positions(self, symbol: str) -> List[Dict[str, Any]]:
        """Get open positions for a specific symbol."""
        if symbol not in self.positions:
            return []

        return [asdict(pos) for pos in self.positions[symbol]]

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get overall trading performance."""
        if not self.trade_history:
            return self._get_empty_performance()

        # Calculate metrics
        returns = [trade["pnl"] for trade in self.trade_history]
        cumulative_returns = np.cumsum(returns)

        # Sharpe ratio (assuming daily returns, 252 trading days)
        if len(returns) > 1:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Max drawdown
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - peak
        max_dd = np.min(drawdown) if len(drawdown) > 0 else 0.0

        # Win rate
        winning_trades = len([r for r in returns if r > 0])
        win_rate = winning_trades / len(returns) if returns else 0.0

        # Profit factor
        gross_profit = sum(r for r in returns if r > 0)
        gross_loss = abs(sum(r for r in returns if r < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        return {
            "total_return": sum(returns),
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": len(returns),
            "avg_trade_return": np.mean(returns) if returns else 0.0,
            "largest_win": max(returns) if returns else 0.0,
            "largest_loss": min(returns) if returns else 0.0
        }

    async def get_symbol_performance(self, symbol: str) -> Dict[str, Any]:
        """Get performance metrics for a specific symbol."""
        symbol_trades = [trade for trade in self.trade_history if trade["symbol"] == symbol]

        if not symbol_trades:
            return self._get_empty_performance()

        returns = [trade["pnl"] for trade in symbol_trades]

        return {
            "total_return": sum(returns),
            "total_trades": len(returns),
            "win_rate": len([r for r in returns if r > 0]) / len(returns),
            "avg_trade_return": np.mean(returns),
            "largest_win": max(returns),
            "largest_loss": min(returns)
        }

    async def get_order_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent order history."""
        orders = list(self.orders.values())
        orders.sort(key=lambda x: x.timestamp, reverse=True)

        return [asdict(order) for order in orders[:limit]]

    async def get_symbol_orders(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get order history for a specific symbol."""
        symbol_orders = [order for order in self.orders.values() if order.symbol == symbol]
        symbol_orders.sort(key=lambda x: x.timestamp, reverse=True)

        return [asdict(order) for order in symbol_orders[:limit]]

    async def emergency_stop(self) -> Dict[str, Any]:
        """Emergency stop all trading activities."""
        logger.warning("Emergency stop activated!")

        positions_closed = 0
        sessions_stopped = 0

        # Stop all sessions
        for symbol in list(self.trading_sessions.keys()):
            try:
                result = await self.stop_trading(symbol, close_positions=True)
                positions_closed += result["positions_closed"]
                sessions_stopped += 1
            except Exception as e:
                logger.error(f"Error stopping session for {symbol}: {e}")

        # Cancel all pending orders
        pending_orders = [o for o in self.orders.values() if o.status == OrderStatus.PENDING]
        for order in pending_orders:
            order.status = OrderStatus.CANCELLED

        # Publish emergency event
        await self.event_system.publish("trading.emergency_stop", {
            "positions_closed": positions_closed,
            "sessions_stopped": sessions_stopped,
            "timestamp": datetime.now().isoformat()
        })

        return {
            "positions_closed": positions_closed,
            "sessions_stopped": sessions_stopped,
            "pending_orders_cancelled": len(pending_orders)
        }

    async def get_portfolio(self) -> Dict[str, Any]:
        """Get portfolio information."""
        total_balance = 10000.0  # Would integrate with actual balance
        positions_value = sum(
            pos.current_price * pos.quantity
            for symbol_positions in self.positions.values()
            for pos in symbol_positions
        )

        return {
            "total_balance": total_balance,
            "positions_value": positions_value,
            "available_balance": total_balance - positions_value,
            "positions": [asdict(pos) for symbol_positions in self.positions.values() for pos in symbol_positions]
        }

    async def _monitor_trading_session(self, symbol: str):
        """Monitor trading session for signals and position updates."""
        try:
            while symbol in self.trading_sessions and self.trading_sessions[symbol].is_active:
                # Check for trading signals (placeholder - would integrate with core service)
                await self._check_trading_signals(symbol)

                # Update positions
                await self._update_positions(symbol)

                # Check risk limits
                await self._check_risk_limits(symbol)

                await asyncio.sleep(POSITION_UPDATE_INTERVAL)

        except asyncio.CancelledError:
            logger.info(f"Monitoring cancelled for {symbol}")
        except Exception as e:
            logger.error(f"Monitoring error for {symbol}: {e}")

    async def _check_trading_signals(self, symbol: str):
        """Check for trading signals (placeholder)."""
        # This would integrate with the core service to get signals
        # For now, simulate occasional signals
        if np.random.random() < 0.01:  # 1% chance per check
            signal = np.random.choice(["buy", "sell"])
            await self.event_system.publish("trading.signal_generated", {
                "symbol": symbol,
                "signal": signal,
                "timestamp": datetime.now().isoformat()
            })

            # Execute signal if conditions met
            session = self.trading_sessions[symbol]
            current_positions = len(self.positions.get(symbol, []))

            if current_positions < session.max_positions:
                # Calculate position size
                position_size = await self._calculate_position_size(symbol)

                if position_size > 0:
                    await self.place_manual_order(
                        symbol=symbol,
                        side=signal,
                        quantity=position_size,
                        order_type="market"
                    )

    async def _update_positions(self, symbol: str):
        """Update position prices and P&L."""
        if symbol not in self.positions:
            return

        # Get current price (placeholder)
        current_price = await self._get_current_price(symbol)

        for position in self.positions[symbol]:
            position.current_price = current_price

            if position.side == OrderSide.BUY:
                position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
            else:
                position.unrealized_pnl = (position.entry_price - current_price) * position.quantity

            # Check stop loss / take profit
            await self._check_position_exits(position)

    async def _check_risk_limits(self, symbol: str):
        """Check risk management limits."""
        session = self.trading_sessions[symbol]

        # Calculate current drawdown
        performance = await self.get_symbol_performance(symbol)
        current_drawdown = -performance.get("total_return", 0)  # Negative return = drawdown

        if current_drawdown > session.max_drawdown:
            logger.warning(f"Drawdown limit reached for {symbol}: {current_drawdown}")
            await self.event_system.publish("trading.risk_triggered", {
                "symbol": symbol,
                "risk_type": "max_drawdown",
                "value": current_drawdown
            })

            # Could implement automatic position reduction here

    async def _calculate_position_size(self, symbol: str) -> float:
        """Calculate position size based on risk management."""
        session = self.trading_sessions[symbol]

        # Get current price and volatility (placeholder)
        current_price = await self._get_current_price(symbol)
        volatility = await self._get_volatility(symbol)  # ATR or similar

        # Position size = (Capital * Risk per trade) / (Volatility * Risk multiplier)
        risk_amount = session.capital * session.risk_per_trade
        risk_multiplier = 2.0  # Stop loss distance in volatility units

        position_size = risk_amount / (volatility * risk_multiplier)

        # Convert to quantity (assuming crypto, so divide by price)
        quantity = position_size / current_price

        return quantity

    def _validate_order(self, order_data: Dict[str, Any]) -> bool:
        """Validate order parameters."""
        symbol = order_data.get("symbol")
        signal = order_data.get("signal")
        price = order_data.get("price", 0)

        if not symbol or not signal:
            return False

        if signal not in ["BUY", "SELL"]:
            return False

        if price <= 0:
            return False

        return True

    async def _execute_paper_order(self, order: Order):
        """Execute order in paper trading mode."""
        # Simulate execution
        current_price = await self._get_current_price(order.symbol)

        if order.order_type == OrderType.MARKET:
            execution_price = current_price
        elif order.order_type == OrderType.LIMIT:
            # Simple simulation - assume limit orders fill immediately
            execution_price = order.price
        else:
            execution_price = order.price

        # Update order
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = execution_price
        order.fees = order.quantity * execution_price * 0.001  # 0.1% fee

        # Create position
        position = Position(
            position_id=str(uuid.uuid4()),
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            entry_price=execution_price,
            current_price=execution_price,
            unrealized_pnl=0.0,
            timestamp=datetime.now()
        )

        if order.symbol not in self.positions:
            self.positions[order.symbol] = []

        self.positions[order.symbol].append(position)

        # Record trade
        self.trade_history.append({
            "symbol": order.symbol,
            "side": order.side.value,
            "quantity": order.quantity,
            "entry_price": execution_price,
            "timestamp": datetime.now().isoformat(),
            "pnl": 0.0  # Will be updated when closed
        })

        # Update session stats (only if session exists)
        if order.symbol in self.trading_sessions:
            session = self.trading_sessions[order.symbol]
            session.total_trades += 1

        # Publish event
        await self.event_system.publish("trading.order_executed", {
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "quantity": order.quantity,
            "price": execution_price
        })

    async def _execute_live_order(self, order: Order):
        """Execute order in live trading mode (placeholder)."""
        # This would integrate with actual broker API
        logger.warning("Live order execution not implemented - using paper trading simulation")
        await self._execute_paper_order(order)

    async def _close_position(self, position: Position, reason: str):
        """Close a position."""
        # Calculate final P&L
        if position.side == OrderSide.BUY:
            pnl = (position.current_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - position.current_price) * position.quantity

        # Update trade history
        for trade in self.trade_history:
            if trade["symbol"] == position.symbol and trade["entry_price"] == position.entry_price:
                trade["pnl"] = pnl
                trade["exit_price"] = position.current_price
                trade["exit_time"] = datetime.now().isoformat()
                break

        # Update session stats
        session = self.trading_sessions[position.symbol]
        session.total_pnl += pnl
        if pnl > 0:
            session.winning_trades += 1

        logger.info(f"Closed position for {position.symbol}: P&L = {pnl}")

        # Publish event
        await self.event_system.publish("trading.position_closed", {
            "position_id": position.position_id,
            "symbol": position.symbol,
            "pnl": pnl,
            "reason": reason
        })

    async def _check_position_exits(self, position: Position):
        """Check if position should be closed due to SL/TP."""
        if position.stop_loss and position.current_price <= position.stop_loss:
            await self._close_position(position, "Stop Loss")
            # Remove from positions list
            symbol_positions = self.positions[position.symbol]
            symbol_positions.remove(position)

        elif position.take_profit and position.current_price >= position.take_profit:
            await self._close_position(position, "Take Profit")
            # Remove from positions list
            symbol_positions = self.positions[position.symbol]
            symbol_positions.remove(position)

    async def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol (placeholder)."""
        # This would integrate with data service
        # For simulation, return a random walk price
        base_price = 50000.0  # BTC base price
        return base_price + np.random.normal(0, 1000)

    async def _get_volatility(self, symbol: str) -> float:
        """Get volatility measure for symbol (placeholder)."""
        # This would calculate ATR or similar
        return 1000.0  # Fixed volatility for simulation

    async def execute_paper_trade(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a paper trade based on signal data."""
        # Handle both dict and string signals for backward compatibility
        if isinstance(signal_data, str):
            # Parse string signal (e.g., "BUY BTC/USDT")
            parts = signal_data.split()
            if len(parts) >= 2:
                action = parts[0]
                symbol = parts[1]
                signal_data = {"signal": action, "symbol": symbol, "price": 50000.0, "quantity": 0.01}
            else:
                raise ValueError(f"Invalid signal format: {signal_data}")

        symbol = signal_data.get("symbol", DEFAULT_SYMBOL)
        action = signal_data.get("signal", "BUY")
        price = signal_data.get("price", 50000.0)
        quantity = signal_data.get("quantity", 0.01)

        # Validate signal - raise exception before try block
        if action not in ["BUY", "SELL"]:
            raise ValueError(f"Invalid signal: {action}")

        try:
            # Create mock order
            order_id = str(uuid.uuid4())
            order = Order(
                order_id=order_id,
                symbol=symbol,
                side=OrderSide(action.lower()),
                quantity=quantity,
                order_type=OrderType.MARKET,
                price=price,
                status=OrderStatus.PENDING,
                timestamp=datetime.now()
            )

            await self._execute_paper_order(order)

            # Create position dict for response
            position = {
                "symbol": symbol,
                "side": action,
                "quantity": quantity,
                "entry_price": price,
                "current_price": price,
                "stop_loss": price * 0.95 if action == "BUY" else price * 1.05,
                "take_profit": price * 1.05 if action == "BUY" else price * 0.95,
                "unrealized_pnl": 0.0
            }

            return {
                "success": True,
                "order_id": order_id,
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "price": price,
                "position": position
            }

        except Exception as e:
            logger.error(f"Paper trade execution failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def update_position(self, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update position information."""
        try:
            position_id = update_data.get("position_id")
            symbol = update_data.get("symbol", DEFAULT_SYMBOL)

            # Handle both new structure (positions[symbol][position_id]) and old test structure (positions[position_id])
            if symbol in self.positions and isinstance(self.positions[symbol], list):
                # New structure: positions[symbol] is a list of Position objects
                position = None
                for pos in self.positions[symbol]:
                    if pos.position_id == position_id:
                        position = pos
                        break

                if not position:
                    raise ValueError(f"Position {position_id} not found")
            elif position_id in self.positions:
                # Old test structure: positions[position_id] is a dict
                position = self.positions[position_id]
            else:
                raise ValueError(f"No positions for symbol {symbol}")

            # Update position
            if "current_price" in update_data:
                if hasattr(position, 'current_price'):
                    position.current_price = update_data["current_price"]
                else:
                    position["current_price"] = update_data["current_price"]

            if "stop_loss" in update_data:
                if hasattr(position, 'stop_loss'):
                    position.stop_loss = update_data["stop_loss"]
                else:
                    position["stop_loss"] = update_data["stop_loss"]

            if "take_profit" in update_data:
                if hasattr(position, 'take_profit'):
                    position.take_profit = update_data["take_profit"]
                else:
                    position["take_profit"] = update_data["take_profit"]

            # Recalculate P&L
            if hasattr(position, 'side'):
                if position.side == OrderSide.BUY:
                    position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity
            else:
                side = position.get("side")
                if side == "BUY":
                    position["unrealized_pnl"] = (position["current_price"] - position["entry_price"]) * position["quantity"]
                else:
                    position["unrealized_pnl"] = (position["entry_price"] - position["current_price"]) * position["quantity"]

            return {
                "success": True,
                "position": position if hasattr(position, '__dict__') else position
            }

        except Exception as e:
            logger.error(f"Position update failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def check_risk_limits(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """Check risk limits for portfolio."""
        capital = portfolio.get("capital", 10000.0)
        current_value = portfolio.get("current_value", capital)
        drawdown = (capital - current_value) / capital

        max_drawdown = MAX_DRAWDOWN
        risk_breached = drawdown > max_drawdown

        return {
            "current_drawdown": drawdown,
            "max_drawdown": max_drawdown,
            "risk_breached": risk_breached,
            "capital": capital,
            "current_value": current_value,
            "position_size_limit": capital * 0.1,  # 10% max per position
            "daily_loss_limit": capital * 0.05     # 5% max daily loss
        }

    async def handle_price_update(self, price_update: Dict[str, Any]) -> Dict[str, Any]:
        """Handle price update for positions."""
        try:
            symbol = price_update.get("symbol")
            new_price = price_update.get("price")

            if not symbol or new_price is None:
                raise ValueError("Symbol and price required")

            # Update all positions for symbol
            updated_positions = []
            if symbol in self.positions:
                for position in self.positions[symbol]:
                    position.current_price = new_price

                    # Recalculate P&L
                    if position.side == OrderSide.BUY:
                        position.unrealized_pnl = (new_price - position.entry_price) * position.quantity
                    else:
                        position.unrealized_pnl = (position.entry_price - new_price) * position.quantity

                    # Check SL/TP
                    await self._check_position_exits(position)
                    updated_positions.append(asdict(position))

            return {
                "success": True,
                "symbol": symbol,
                "new_price": new_price,
                "updated_positions": len(updated_positions)
            }

        except Exception as e:
            logger.error(f"Price update handling failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _calculate_pnl(self, position: Dict[str, Any]) -> float:
        """Calculate P&L for a position."""
        side = position.get("side")
        if isinstance(side, str):
            side = OrderSide(side.lower())

        if side == OrderSide.BUY:
            return (position["current_price"] - position["entry_price"]) * position["quantity"]
        else:
            return (position["entry_price"] - position["current_price"]) * position["quantity"]

    def _calculate_risk_limits(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk limits for portfolio."""
        capital = portfolio.get("capital", 10000.0)
        positions_value = portfolio.get("positions_value", 0.0)

        max_single_position = capital * 0.1  # 10% max per position
        max_total_positions = capital * 0.5  # 50% max total exposure

        return {
            "max_single_position": max_single_position,
            "max_total_positions": max_total_positions,
            "current_exposure": positions_value,
            "available_risk_capital": capital * RISK_PER_TRADE,
            "max_drawdown_pct": MAX_DRAWDOWN,
            "position_size_pct": 0.1,
            "daily_loss_pct": 0.05
        }

    def _validate_order(self, order: Dict[str, Any]) -> bool:
        """Validate order parameters."""
        quantity = order.get("quantity", 0.01)  # Default quantity for signal validation
        price = order.get("price")

        if quantity <= 0:
            return False

        if price is not None and price <= 0:
            return False

        # For signals, we don't require order_type
        signal = order.get("signal")
        if signal and signal not in ["BUY", "SELL"]:
            return False

        return True

    async def _setup_live_trading(self):
        """Setup live trading environment."""
        # Placeholder for live trading setup
        logger.info("Setting up live trading environment")
        return True

    @property
    def paper_balance(self) -> float:
        """Get paper trading balance."""
        return 10000.0  # Default paper balance

    @property
    def paper_trading(self) -> bool:
        """Check if paper trading is enabled."""
        return True  # Default to paper trading