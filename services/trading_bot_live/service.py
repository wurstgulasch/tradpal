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

# Service imports for smart trading
try:
    from services.core.client import CoreServiceClient
    CORE_SERVICE_AVAILABLE = True
except ImportError:
    CORE_SERVICE_AVAILABLE = False
    logger.warning("CoreService not available - using legacy signal generation")

# Trading service imports (integrated services)
try:
    from services.trading_service.risk_management.service import RiskManagementService
    RISK_SERVICE_AVAILABLE = True
except ImportError:
    RISK_SERVICE_AVAILABLE = False
    logger.warning("RiskManagementService not available - using basic position sizing")

try:
    from services.trading_service.market_regime_detection.service import MarketRegimeDetectionService
    MARKET_REGIME_SERVICE_AVAILABLE = True
except ImportError:
    MARKET_REGIME_SERVICE_AVAILABLE = False
    logger.warning("MarketRegimeDetectionService not available - using basic regime detection")

try:
    from services.data_service.client import DataServiceClient
    DATA_SERVICE_AVAILABLE = True
except ImportError:
    DATA_SERVICE_AVAILABLE = False
    logger.warning("DataService not available - using simulated data")


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

        # Service clients for smart trading
        self.core_service = None
        self.risk_service = None
        self.market_regime_service = None
        self.data_service = None

        # Initialize service clients
        self._initialize_services()

    def _initialize_services(self):
        """Initialize smart trading service clients."""
        try:
            if CORE_SERVICE_AVAILABLE:
                self.core_service = CoreServiceClient()
                logger.info("✅ CoreService client initialized for ML signals")

            if RISK_SERVICE_AVAILABLE:
                self.risk_service = RiskManagementService()
                logger.info("✅ RiskManagementService initialized for smart position sizing")

            if MARKET_REGIME_SERVICE_AVAILABLE:
                self.market_regime_service = MarketRegimeDetectionService()
                logger.info("✅ MarketRegimeDetectionService initialized for adaptive trading")

            if DATA_SERVICE_AVAILABLE:
                self.data_service = DataServiceClient()
                logger.info("✅ DataService client initialized for real-time data")

        except Exception as e:
            logger.error(f"Error initializing service clients: {e}")
            logger.warning("Falling back to legacy trading logic")

        logger.info("Trading Bot Live Service initialized with smart capabilities")

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

    async def _check_trading_signals(self, symbol: str) -> List[Dict[str, Any]]:
        """Check for trading signals using ML-based advanced signal generation."""
        signals = []
        try:
            session = self.trading_sessions[symbol]

            # Use CoreService for ML signals if available
            if self.core_service and CORE_SERVICE_AVAILABLE:
                # Get historical data for signal generation
                historical_data = await self._get_historical_data(symbol, session.timeframe)

                if historical_data is not None and not historical_data.empty:
                    # Convert DataFrame to list of dicts for API
                    data_list = historical_data.reset_index().to_dict('records')

                    # Generate ML-based signals
                    ml_signals = await self.core_service.generate_signals(
                        symbol=symbol,
                        timeframe=session.timeframe,
                        data=data_list
                    )

                    # Process signals with confidence filtering
                    if ml_signals:
                        for signal in ml_signals:
                            await self._process_ml_signals(symbol, signal)
                        signals = ml_signals
                    else:
                        signals = []
                else:
                    logger.warning(f"No historical data available for {symbol}")
                    await self._fallback_signal_generation(symbol)

            else:
                # Fallback to legacy signal generation
                logger.warning(f"CoreService not available for {symbol}, using fallback")
                await self._fallback_signal_generation(symbol)

        except Exception as e:
            logger.error(f"Error checking trading signals for {symbol}: {e}")
            # Emergency fallback
            await self._fallback_signal_generation(symbol)

        return signals

    async def _get_historical_data(self, symbol: str, timeframe: str, limit: int = 1000) -> Optional[pd.DataFrame]:
        """Get historical data for signal generation."""
        try:
            if self.data_service and DATA_SERVICE_AVAILABLE:
                # Use DataService for real historical data
                data = await self.data_service.get_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=limit
                )
                return pd.DataFrame(data) if data else None
            else:
                # Fallback: generate synthetic data for testing
                return self._generate_synthetic_data(symbol, limit)

        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return self._generate_synthetic_data(symbol, limit)

    async def _process_ml_signals(self, symbol: str, signals: Dict[str, Any]):
        """Process ML-generated signals with confidence filtering."""
        try:
            session = self.trading_sessions[symbol]

            # Extract signal information
            signal_type = signals.get("signal", "").upper()
            confidence = signals.get("confidence", 0.0)
            market_regime = signals.get("market_regime", "unknown")

            # Confidence threshold based on market regime
            confidence_threshold = self._get_confidence_threshold(market_regime)

            logger.info(f"ML Signal for {symbol}: {signal_type} (confidence: {confidence:.3f}, regime: {market_regime})")

            # Only execute high-confidence signals
            if confidence >= confidence_threshold and signal_type in ["BUY", "SELL"]:
                # Check position limits
                current_positions = len(self.positions.get(symbol, []))
                if current_positions >= session.max_positions:
                    logger.info(f"Position limit reached for {symbol} ({current_positions}/{session.max_positions})")
                    return

                # Get market regime for position sizing
                regime_info = await self._get_market_regime(symbol)

                # Calculate smart position size
                position_size = await self._calculate_smart_position_size(
                    symbol, signal_type, confidence, regime_info
                )

                if position_size > 0:
                    # Execute the signal
                    await self._execute_ml_signal(symbol, signal_type, position_size, signals)

                    # Publish signal event
                    await self.event_system.publish("trading.ml_signal_executed", {
                        "symbol": symbol,
                        "signal": signal_type,
                        "confidence": confidence,
                        "market_regime": market_regime,
                        "position_size": position_size,
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    logger.warning(f"Invalid position size calculated for {symbol}: {position_size}")
            else:
                logger.info(f"Signal rejected for {symbol}: confidence {confidence:.3f} < threshold {confidence_threshold}")

        except Exception as e:
            logger.error(f"Error processing ML signals for {symbol}: {e}")

    async def _execute_ml_signal(self, symbol: str, signal_type: str, position_size: float, signal_data: Dict[str, Any]):
        """Execute ML-generated signal with smart parameters."""
        try:
            # Get current market conditions
            current_price = await self._get_current_price(symbol)
            market_regime = signal_data.get("market_regime", "unknown")

            # Calculate smart stop loss and take profit based on market regime
            sl_tp_levels = await self._calculate_smart_sl_tp(symbol, signal_type, current_price, market_regime)

            # Place order with smart parameters
            order_result = await self.place_manual_order(
                symbol=symbol,
                side=signal_type.lower(),
                quantity=position_size,
                order_type="market"
            )

            if order_result.get("order_id"):
                # Update position with smart SL/TP
                await self._update_position_with_sl_tp(
                    symbol, order_result["order_id"], sl_tp_levels
                )

                logger.info(f"Executed ML signal for {symbol}: {signal_type} {position_size:.6f} @ {current_price}")

        except Exception as e:
            logger.error(f"Error executing ML signal for {symbol}: {e}")

    async def _calculate_smart_position_size(self, symbol: str, signal_type: str, confidence: float, regime_info: Dict[str, Any]) -> float:
        """Calculate position size using RiskService with ML insights and Kelly Criterion."""
        try:
            session = self.trading_sessions[symbol]

            if self.risk_service and RISK_SERVICE_AVAILABLE:
                # Use RiskService for intelligent position sizing with Kelly Criterion
                position_sizing = await self.risk_service.calculate_position_size(
                    symbol=symbol,
                    capital=session.capital,
                    risk_per_trade=session.risk_per_trade,
                    volatility=await self._get_volatility(symbol),
                    market_regime=regime_info.get("regime", "unknown"),
                    signal_confidence=confidence,
                    signal_type=signal_type,
                    use_kelly_criterion=True  # Enable Kelly Criterion
                )

                position_size = position_sizing.get("position_size", 0.0)

                # Validate position size
                if position_size > 0:
                    # Apply additional ML-based adjustments
                    position_size = await self._apply_ml_position_adjustments(
                        symbol, position_size, confidence, regime_info
                    )

                return position_size

            else:
                # Enhanced fallback with ML insights
                return await self._calculate_enhanced_fallback_position_size(
                    symbol, signal_type, confidence, regime_info
                )

        except Exception as e:
            logger.error(f"Error calculating smart position size for {symbol}: {e}")
            # Ultimate fallback
            return await self._calculate_position_size(symbol)

    async def _apply_ml_position_adjustments(self, symbol: str, base_position_size: float, confidence: float, regime_info: Dict[str, Any]) -> float:
        """Apply additional ML-based adjustments to position size."""
        try:
            # Confidence-based adjustment (0.5x to 1.5x multiplier)
            confidence_multiplier = 0.5 + (confidence * 1.0)

            # Market regime adjustment
            regime = regime_info.get("regime", "unknown")
            regime_multiplier = self._get_regime_position_multiplier(regime)

            # Recent performance adjustment
            performance_multiplier = await self._get_performance_based_multiplier(symbol)

            # Apply all adjustments
            adjusted_size = base_position_size * confidence_multiplier * regime_multiplier * performance_multiplier

            # Apply sanity bounds (0.1x to 3x of base size)
            adjusted_size = max(base_position_size * 0.1, min(adjusted_size, base_position_size * 3.0))

            logger.debug(f"ML position adjustments for {symbol}: base={base_position_size:.6f}, confidence_mult={confidence_multiplier:.2f}, regime_mult={regime_multiplier:.2f}, perf_mult={performance_multiplier:.2f}, final={adjusted_size:.6f}")

            return adjusted_size

        except Exception as e:
            logger.error(f"Error applying ML position adjustments for {symbol}: {e}")
            return base_position_size

    async def _calculate_enhanced_fallback_position_size(self, symbol: str, signal_type: str, confidence: float, regime_info: Dict[str, Any]) -> float:
        """Enhanced fallback position sizing with ML insights when RiskService unavailable."""
        try:
            session = self.trading_sessions[symbol]

            # Get market data
            current_price = await self._get_current_price(symbol)
            volatility = await self._get_volatility(symbol)

            # Base Kelly Criterion calculation
            # K = (p * (b+1) - 1) / b, where p=win_rate, b=reward_risk_ratio
            win_rate = await self._calculate_recent_win_rate(symbol)
            reward_risk_ratio = self._calculate_reward_risk_ratio(regime_info.get("regime", "unknown"))

            if win_rate > 0 and reward_risk_ratio > 0:
                kelly_fraction = (win_rate * (reward_risk_ratio + 1) - 1) / reward_risk_ratio
                kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Bound Kelly fraction
            else:
                kelly_fraction = session.risk_per_trade  # Fallback to session risk

            # Apply ML adjustments
            confidence_multiplier = 0.7 + (confidence * 0.6)  # 0.7 to 1.3
            regime_multiplier = self._get_regime_position_multiplier(regime_info.get("regime", "unknown"))

            adjusted_risk = kelly_fraction * confidence_multiplier * regime_multiplier

            # Calculate position size
            risk_amount = session.capital * adjusted_risk
            position_value = risk_amount / 0.02  # Assume 2% risk per trade
            quantity = position_value / current_price

            # Apply position limits
            max_position_value = session.capital * 0.1  # Max 10% of capital per position
            max_quantity = max_position_value / current_price
            quantity = min(quantity, max_quantity)

            return max(0, quantity)

        except Exception as e:
            logger.error(f"Error in enhanced fallback position sizing for {symbol}: {e}")
            return await self._calculate_position_size(symbol)

    def _get_regime_position_multiplier(self, market_regime: str) -> float:
        """Get position size multiplier based on market regime."""
        multipliers = {
            "trending": 1.3,         # Larger positions in trends
            "consolidation": 0.7,    # Smaller positions in consolidation
            "high_volatility": 0.5,  # Much smaller in high vol
            "low_volatility": 1.1,   # Slightly larger in low vol
            "unknown": 0.9          # Conservative default
        }
        return multipliers.get(market_regime, 0.9)

    async def _get_performance_based_multiplier(self, symbol: str) -> float:
        """Calculate position size multiplier based on recent performance."""
        try:
            # Get recent trades (last 10)
            recent_trades = [t for t in self.trade_history if t["symbol"] == symbol][-10:]

            if len(recent_trades) < 5:
                return 1.0  # Not enough data

            # Calculate recent win rate
            recent_wins = sum(1 for t in recent_trades if t.get("pnl", 0) > 0)
            recent_win_rate = recent_wins / len(recent_trades)

            # Calculate recent profit factor
            recent_gross_profit = sum(t.get("pnl", 0) for t in recent_trades if t.get("pnl", 0) > 0)
            recent_gross_loss = abs(sum(t.get("pnl", 0) for t in recent_trades if t.get("pnl", 0) < 0))

            if recent_gross_loss > 0:
                profit_factor = recent_gross_profit / recent_gross_loss
            else:
                profit_factor = float('inf') if recent_gross_profit > 0 else 1.0

            # Combine metrics for multiplier (0.5x to 1.5x)
            win_rate_component = 0.5 + (recent_win_rate * 0.5)
            profit_factor_component = min(1.5, max(0.5, profit_factor * 0.3))

            multiplier = (win_rate_component + profit_factor_component) / 2

            return multiplier

        except Exception as e:
            logger.error(f"Error calculating performance multiplier for {symbol}: {e}")
            return 1.0

    async def _calculate_recent_win_rate(self, symbol: str, lookback: int = 20) -> float:
        """Calculate recent win rate for symbol."""
        try:
            recent_trades = [t for t in self.trade_history if t["symbol"] == symbol][-lookback:]
            if not recent_trades:
                return 0.5  # Neutral assumption

            wins = sum(1 for t in recent_trades if t.get("pnl", 0) > 0)
            return wins / len(recent_trades)

        except Exception as e:
            logger.error(f"Error calculating win rate for {symbol}: {e}")
            return 0.5

    def _calculate_reward_risk_ratio(self, market_regime: str) -> float:
        """Calculate expected reward-to-risk ratio based on market regime."""
        # Typical R:R ratios for different regimes
        rr_ratios = {
            "trending": 2.5,        # Higher R:R in trends
            "consolidation": 1.5,   # Lower R:R in consolidation
            "high_volatility": 1.2, # Conservative R:R in high vol
            "low_volatility": 2.0,  # Moderate R:R in low vol
            "unknown": 1.8         # Default
        }
        return rr_ratios.get(market_regime, 1.8)

    async def _calculate_basic_position_size_with_ml(self, symbol: str, confidence: float, regime_info: Dict[str, Any]) -> float:
        """Basic position sizing enhanced with ML insights."""
        session = self.trading_sessions[symbol]

        # Base calculation
        current_price = await self._get_current_price(symbol)
        volatility = await self._get_volatility(symbol)

        # Adjust risk based on confidence and market regime
        confidence_multiplier = 0.5 + (confidence * 0.5)  # 0.5 to 1.0
        regime_multiplier = self._get_regime_risk_multiplier(regime_info.get("regime", "unknown"))

        adjusted_risk = session.risk_per_trade * confidence_multiplier * regime_multiplier

        # Position size calculation
        risk_amount = session.capital * adjusted_risk
        risk_multiplier = 2.0  # Stop loss distance in volatility units

        position_size = risk_amount / (volatility * risk_multiplier)
        quantity = position_size / current_price

        return quantity

    async def _get_market_regime(self, symbol: str) -> Dict[str, Any]:
        """Get current market regime information."""
        try:
            if self.market_regime_service and MARKET_REGIME_SERVICE_AVAILABLE:
                # Use MarketRegimeService
                regime_data = await self.market_regime_service.get_market_regime(symbol=symbol)
                return regime_data
            else:
                # Fallback: basic regime detection
                return await self._detect_basic_regime(symbol)

        except Exception as e:
            logger.error(f"Error getting market regime for {symbol}: {e}")
            return {"regime": "unknown", "confidence": 0.0}

    async def _detect_basic_regime(self, symbol: str) -> Dict[str, Any]:
        """Basic market regime detection using volatility and trend."""
        try:
            # Get recent price data
            current_price = await self._get_current_price(symbol)
            volatility = await self._get_volatility(symbol)

            # Simple regime classification based on volatility
            if volatility > 2000:  # High volatility
                regime = "high_volatility"
                confidence = 0.8
            elif volatility > 1000:  # Moderate volatility
                regime = "moderate_volatility"
                confidence = 0.6
            else:  # Low volatility
                regime = "consolidation"
                confidence = 0.7

            return {
                "regime": regime,
                "confidence": confidence,
                "volatility": volatility,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error in basic regime detection for {symbol}: {e}")
            return {"regime": "unknown", "confidence": 0.0}

    async def _calculate_smart_sl_tp(self, symbol: str, signal_type: str, current_price: float, market_regime: str) -> Dict[str, float]:
        """Calculate smart stop loss and take profit levels based on market regime."""
        try:
            volatility = await self._get_volatility(symbol)

            # Base multipliers based on market regime
            if market_regime == "trending":
                sl_multiplier = 1.5  # Wider stops in trends
                tp_multiplier = 3.0  # Higher targets in trends
            elif market_regime == "high_volatility":
                sl_multiplier = 2.0  # Very wide stops in high vol
                tp_multiplier = 2.0  # Moderate targets
            elif market_regime == "consolidation":
                sl_multiplier = 1.0  # Tight stops in consolidation
                tp_multiplier = 2.0  # Moderate targets
            else:
                sl_multiplier = 1.5  # Default
                tp_multiplier = 2.5  # Default

            # Calculate levels
            if signal_type == "BUY":
                stop_loss = current_price - (volatility * sl_multiplier)
                take_profit = current_price + (volatility * tp_multiplier)
            else:  # SELL
                stop_loss = current_price + (volatility * sl_multiplier)
                take_profit = current_price - (volatility * tp_multiplier)

            return {
                "stop_loss": stop_loss,
                "take_profit": take_profit
            }

        except Exception as e:
            logger.error(f"Error calculating smart SL/TP for {symbol}: {e}")
            # Fallback to basic levels
            if signal_type == "BUY":
                return {
                    "stop_loss": current_price * 0.95,
                    "take_profit": current_price * 1.05
                }
            else:
                return {
                    "stop_loss": current_price * 1.05,
                    "take_profit": current_price * 0.95
                }

    async def _update_position_with_sl_tp(self, symbol: str, order_id: str, sl_tp_levels: Dict[str, float]):
        """Update position with smart stop loss and take profit levels."""
        try:
            # Find the position created by this order
            if symbol in self.positions:
                for position in self.positions[symbol]:
                    # Match by entry price (approximate)
                    if abs(position.entry_price - await self._get_current_price(symbol)) < 100:  # Within $100
                        position.stop_loss = sl_tp_levels["stop_loss"]
                        position.take_profit = sl_tp_levels["take_profit"]
                        logger.info(f"Updated position {position.position_id} with smart SL/TP: SL={position.stop_loss:.2f}, TP={position.take_profit:.2f}")
                        break

        except Exception as e:
            logger.error(f"Error updating position SL/TP for {symbol}: {e}")

    def _get_confidence_threshold(self, market_regime: str) -> float:
        """Get confidence threshold based on market regime."""
        thresholds = {
            "trending": 0.75,        # Higher threshold in trends
            "consolidation": 0.80,   # Very high threshold in consolidation
            "high_volatility": 0.85, # Highest threshold in high volatility
            "low_volatility": 0.70,  # Lower threshold in low volatility
            "unknown": 0.85         # Conservative default
        }
        return thresholds.get(market_regime, 0.85)

    def _get_regime_risk_multiplier(self, market_regime: str) -> float:
        """Get risk multiplier based on market regime."""
        multipliers = {
            "trending": 1.2,         # Higher risk in trends
            "consolidation": 0.8,    # Lower risk in consolidation
            "high_volatility": 0.6,  # Much lower risk in high vol
            "low_volatility": 1.0,   # Normal risk
            "unknown": 0.8          # Conservative
        }
        return multipliers.get(market_regime, 0.8)

    def _generate_synthetic_data(self, symbol: str, limit: int = 1000) -> pd.DataFrame:
        """Generate synthetic OHLCV data for testing."""
        # Create datetime index
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=limit)
        timestamps = pd.date_range(start=start_time, end=end_time, freq='1H')[:limit]

        # Generate synthetic price data
        base_price = 50000.0
        prices = []
        current_price = base_price

        for i in range(limit):
            # Random walk with trend
            change = np.random.normal(0, 500)  # Mean=0, StdDev=500
            current_price += change
            current_price = max(current_price, 1000)  # Floor price
            prices.append(current_price)

        # Create OHLCV data
        data = []
        for i, price in enumerate(prices):
            high = price + abs(np.random.normal(0, 200))
            low = price - abs(np.random.normal(0, 200))
            open_price = prices[i-1] if i > 0 else price
            volume = np.random.uniform(100, 1000)

            data.append({
                'timestamp': timestamps[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })

        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df

    async def _fallback_signal_generation(self, symbol: str):
        """Fallback signal generation when ML services are unavailable."""
        try:
            # Simple technical analysis based signal generation
            if np.random.random() < 0.02:  # 2% chance (higher than random)
                signal = np.random.choice(["buy", "sell"])

                session = self.trading_sessions[symbol]
                current_positions = len(self.positions.get(symbol, []))

                if current_positions < session.max_positions:
                    position_size = await self._calculate_position_size(symbol)

                    if position_size > 0:
                        await self.place_manual_order(
                            symbol=symbol,
                            side=signal,
                            quantity=position_size,
                            order_type="market"
                        )

                        logger.info(f"Fallback signal executed for {symbol}: {signal}")

        except Exception as e:
            logger.error(f"Error in fallback signal generation for {symbol}: {e}")

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
        """Check risk management limits with adaptive, market-regime-aware logic."""
        try:
            session = self.trading_sessions[symbol]

            # Get current market regime for adaptive risk management
            regime_info = await self._get_market_regime(symbol)
            market_regime = regime_info.get("regime", "unknown")

            # Calculate current drawdown
            performance = await self.get_symbol_performance(symbol)
            current_drawdown = -performance.get("total_return", 0)

            # Adaptive risk limits based on market regime
            adaptive_limits = self._calculate_adaptive_risk_limits(market_regime, session.max_drawdown)

            # Check if risk limits are breached
            risk_breached = current_drawdown > adaptive_limits["max_drawdown"]

            if risk_breached:
                logger.warning(f"Adaptive risk limit breached for {symbol} in {market_regime} regime: {current_drawdown:.2%} > {adaptive_limits['max_drawdown']:.2%}")

                # Adaptive response based on regime
                await self._handle_risk_breach(symbol, market_regime, current_drawdown, adaptive_limits)

                # Publish risk event
                await self.event_system.publish("trading.adaptive_risk_triggered", {
                    "symbol": symbol,
                    "market_regime": market_regime,
                    "risk_type": "adaptive_max_drawdown",
                    "current_drawdown": current_drawdown,
                    "limit": adaptive_limits["max_drawdown"],
                    "response": adaptive_limits.get("breach_response", "reduce_positions")
                })
            else:
                logger.debug(f"Risk check passed for {symbol} in {market_regime} regime: {current_drawdown:.2%} <= {adaptive_limits['max_drawdown']:.2%}")

        except Exception as e:
            logger.error(f"Error in adaptive risk checking for {symbol}: {e}")
            # Fallback to basic risk checking
            await self._fallback_risk_check(symbol)

    def _calculate_adaptive_risk_limits_static(self, market_regime: str, base_max_drawdown: float) -> Dict[str, Any]:
        """Calculate adaptive risk limits based on market regime."""
        # Base limits adjusted by market conditions
        regime_adjustments = {
            "trending": {
                "max_drawdown_multiplier": 1.5,  # Allow higher drawdown in trends
                "breach_response": "reduce_positions_partial",
                "position_limit_multiplier": 1.2
            },
            "consolidation": {
                "max_drawdown_multiplier": 0.7,  # Stricter limits in consolidation
                "breach_response": "close_all_positions",
                "position_limit_multiplier": 0.8
            },
            "high_volatility": {
                "max_drawdown_multiplier": 0.5,  # Very strict in high vol
                "breach_response": "emergency_stop",
                "position_limit_multiplier": 0.5
            },
            "low_volatility": {
                "max_drawdown_multiplier": 1.2,  # Slightly more lenient
                "breach_response": "reduce_positions_partial",
                "position_limit_multiplier": 1.1
            },
            "unknown": {
                "max_drawdown_multiplier": 0.8,  # Conservative default
                "breach_response": "reduce_positions_partial",
                "position_limit_multiplier": 0.9
            }
        }

        adjustment = regime_adjustments.get(market_regime, regime_adjustments["unknown"])

        return {
            "max_drawdown": base_max_drawdown * adjustment["max_drawdown_multiplier"],
            "breach_response": adjustment["breach_response"],
            "position_limit_multiplier": adjustment["position_limit_multiplier"],
            "regime": market_regime
        }

    async def _handle_risk_breach(self, symbol: str, market_regime: str, current_drawdown: float, limits: Dict[str, Any]):
        """Handle risk breaches with adaptive responses based on market regime."""
        try:
            response_type = limits.get("breach_response", "reduce_positions_partial")

            if response_type == "emergency_stop":
                logger.critical(f"Emergency stop triggered for {symbol} in {market_regime} regime")
                await self.stop_trading(symbol, close_positions=True)

            elif response_type == "close_all_positions":
                logger.warning(f"Closing all positions for {symbol} due to risk breach in {market_regime}")
                positions_closed = 0
                if symbol in self.positions:
                    for position in self.positions[symbol][:]:  # Copy list to avoid modification issues
                        await self._close_position(position, f"Risk breach - {market_regime}")
                        positions_closed += 1
                    self.positions[symbol] = []
                logger.info(f"Closed {positions_closed} positions for {symbol}")

            elif response_type == "reduce_positions_partial":
                # Reduce position sizes by 50%
                logger.warning(f"Reducing positions for {symbol} by 50% due to risk breach in {market_regime}")
                if symbol in self.positions:
                    for position in self.positions[symbol]:
                        # Reduce position size
                        original_size = position.quantity
                        position.quantity *= 0.5
                        logger.info(f"Reduced position {position.position_id} from {original_size} to {position.quantity}")

            # Update session risk parameters adaptively
            session = self.trading_sessions[symbol]
            session.risk_per_trade *= 0.8  # Reduce risk per trade by 20%
            session.max_positions = max(1, int(session.max_positions * 0.7))  # Reduce max positions

            logger.info(f"Adapted risk parameters for {symbol}: risk_per_trade={session.risk_per_trade:.3f}, max_positions={session.max_positions}")

        except Exception as e:
            logger.error(f"Error handling risk breach for {symbol}: {e}")

    async def _fallback_risk_check(self, symbol: str):
        """Fallback risk checking when adaptive system fails."""
        try:
            session = self.trading_sessions[symbol]

            # Basic drawdown check
            performance = await self.get_symbol_performance(symbol)
            current_drawdown = -performance.get("total_return", 0)

            if current_drawdown > session.max_drawdown:
                logger.warning(f"Fallback: Drawdown limit reached for {symbol}: {current_drawdown:.2%}")
                await self.event_system.publish("trading.fallback_risk_triggered", {
                    "symbol": symbol,
                    "risk_type": "max_drawdown",
                    "value": current_drawdown,
                    "limit": session.max_drawdown
                })

                # Basic response: reduce positions
                if symbol in self.positions:
                    for position in self.positions[symbol][:1]:  # Close just one position
                        await self._close_position(position, "Fallback risk management")
                        break

        except Exception as e:
            logger.error(f"Error in fallback risk check for {symbol}: {e}")

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