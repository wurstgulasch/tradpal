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

from config.service_settings import (
    DEFAULT_TIMEFRAME,
    MAX_POSITIONS,
    MAX_DRAWDOWN,
    ORDER_TIMEOUT,
    POSITION_UPDATE_INTERVAL
)

# Use central service client instead of direct imports
from services.trading_service.central_service_client import get_central_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
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
    total_pnl: float = 0.0
    winning_trades: int = 0
    max_drawdown: float = MAX_DRAWDOWN


class TradingBotLiveService:
    """Live trading execution and monitoring service using central client."""

    def __init__(self):
        self.client = None
        self.trading_sessions: Dict[str, TradingSession] = {}
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, List[Position]] = {}
        self.trade_history: List[Dict[str, Any]] = []
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}

    async def initialize(self):
        """Initialize service with central client"""
        self.client = await get_central_client()
        logger.info("TradingBotLiveService initialized with central client")

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
        risk_per_trade: float = 0.02,
        max_positions: int = MAX_POSITIONS,
        enable_paper_trading: bool = True
    ) -> Dict[str, Any]:
        """Start live trading for a symbol."""
        if not self.client:
            await self.initialize()

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
        self.positions[symbol] = []

        # Start monitoring task
        task = asyncio.create_task(self._monitor_trading_session(symbol))
        self.monitoring_tasks[symbol] = task

        logger.info(f"Started trading session {session_id} for {symbol}")

        # Publish event
        await self.client.publish_event("trading_session_started", {
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
        """Stop live trading for a symbol."""
        if symbol not in self.trading_sessions:
            raise ValueError(f"No active trading session for {symbol}")

        session = self.trading_sessions[symbol]
        session.is_active = False
        session.end_time = datetime.now()

        positions_closed = 0

        if close_positions and symbol in self.positions:
            for position in self.positions[symbol]:
                await self._close_position(position, "Manual stop")
                positions_closed += 1
            self.positions[symbol] = []

        if symbol in self.monitoring_tasks:
            self.monitoring_tasks[symbol].cancel()
            del self.monitoring_tasks[symbol]

        logger.info(f"Stopped trading session for {symbol}, closed {positions_closed} positions")

        await self.client.publish_event("trading_session_stopped", {
            "symbol": symbol,
            "positions_closed": positions_closed
        })

        return {"symbol": symbol, "positions_closed": positions_closed}

    async def execute_trade(self, trade_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trade using central client"""
        try:
            result = await self.client.execute_trade(trade_config)

            # Record in local history
            self.trade_history.append({
                "symbol": trade_config.get("symbol"),
                "action": trade_config.get("action"),
                "quantity": trade_config.get("quantity", 0),
                "price": trade_config.get("price", 0),
                "timestamp": datetime.now().isoformat(),
                "result": result
            })

            return result
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return {"success": False, "error": str(e)}

    async def get_positions(self) -> Dict[str, Any]:
        """Get all positions using central client"""
        try:
            return await self.client.get_positions()
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return {"positions": [], "error": str(e)}

    async def get_trading_status(self) -> Dict[str, Any]:
        """Get trading status using central client"""
        try:
            return await self.client.get_trading_status()
        except Exception as e:
            logger.error(f"Failed to get trading status: {e}")
            return {"error": str(e)}

    async def _monitor_trading_session(self, symbol: str):
        """Monitor trading session for signals and position updates."""
        try:
            while symbol in self.trading_sessions and self.trading_sessions[symbol].is_active:
                # Get trading signal
                market_data = await self.client.get_market_data(symbol=symbol, limit=100)
                if market_data.get("data"):
                    latest_data = market_data["data"][-1]
                    signal = await self.client.get_trading_signal(
                        market_data=latest_data,
                        symbol=symbol
                    )

                    # Execute trade if signal is strong enough
                    if signal.get("action") in ["BUY", "SELL"] and signal.get("confidence", 0) > 0.8:
                        await self.execute_trade({
                            "symbol": symbol,
                            "action": signal["action"],
                            "confidence": signal["confidence"],
                            "market_data": latest_data
                        })

                await asyncio.sleep(POSITION_UPDATE_INTERVAL)

        except asyncio.CancelledError:
            logger.info(f"Monitoring cancelled for {symbol}")
        except Exception as e:
            logger.error(f"Monitoring error for {symbol}: {e}")

    async def _close_position(self, position: Position, reason: str):
        """Close a position."""
        # Calculate final P&L
        if position.side == OrderSide.BUY:
            pnl = (position.current_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - position.current_price) * position.quantity

        # Update trade history
        for trade in self.trade_history:
            if trade["symbol"] == position.symbol and trade.get("entry_price") == position.entry_price:
                trade["pnl"] = pnl
                trade["exit_price"] = position.current_price
                trade["exit_time"] = datetime.now().isoformat()
                break

        logger.info(f"Closed position for {position.symbol}: P&L = {pnl}")

        await self.client.publish_event("position_closed", {
            "position_id": position.position_id,
            "symbol": position.symbol,
            "pnl": pnl,
            "reason": reason
        })

    async def emergency_stop(self) -> Dict[str, Any]:
        """Emergency stop all trading activities."""
        logger.warning("Emergency stop activated!")

        stopped_sessions = []
        for symbol in list(self.trading_sessions.keys()):
            try:
                result = await self.stop_trading(symbol, close_positions=True)
                stopped_sessions.append(symbol)
            except Exception as e:
                logger.error(f"Error stopping session for {symbol}: {e}")

        await self.client.publish_event("emergency_stop", {
            "stopped_sessions": stopped_sessions,
            "timestamp": datetime.now().isoformat()
        })

        return {
            "stopped_sessions": stopped_sessions,
            "total_stopped": len(stopped_sessions)
        }

    async def cleanup(self):
        """Cleanup service resources"""
        if self.client:
            await self.client.close()
        logger.info("TradingBotLiveService cleanup completed")