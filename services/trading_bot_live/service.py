"""
Trading Bot Live Service

This microservice handles live trading operations including:
- Real-time market data processing
- Signal generation and validation
- Risk management and position sizing
- Order execution (simulated or live)
- Performance monitoring and reporting
- Event-driven communication with other services

The service is designed to be horizontally scalable and fault-tolerant.
"""

import asyncio
import sys
import os
import time
import signal
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data_fetcher import fetch_data
from src.indicators import calculate_indicators
from src.signal_generator import generate_signals, calculate_risk_management
from src.events import (
    initialize_event_system, shutdown_event_system,
    create_market_data_event, create_signal_event,
    create_risk_event, create_trade_event,
    publish_market_data_event, publish_signal_event,
    publish_risk_event, publish_trade_event,
    start_event_processing, EventType, event_bus
)
from src.audit_logger import audit_logger
from config.settings import (
    SYMBOL, TIMEFRAME, INITIAL_CAPITAL, RISK_PER_TRADE,
    LIVE_TRADING_MAX_DRAWDOWN, LIVE_TRADING_MAX_TRADES_PER_DAY,
    LIVE_TRADING_MIN_SIGNAL_CONFIDENCE, LIVE_TRADING_AUTO_EXECUTE,
    LIVE_TRADING_CONFIRMATION_REQUIRED, LIVE_TRADING_MONITOR_ENABLED,
    LIVE_TRADING_POSITION_UPDATE_INTERVAL, LIVE_TRADING_PNL_LOG_FILE,
    LIVE_TRADING_TRADE_LOG_FILE, ADAPTIVE_OPTIMIZATION_ENABLED_LIVE
)


@dataclass
class TradingPosition:
    """Represents a trading position"""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    size: float
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    current_pnl: float = 0.0
    order_id: Optional[str] = None


@dataclass
class TradingState:
    """Current state of the trading bot"""
    capital: float
    positions: Dict[str, TradingPosition]
    trades_today: int
    last_reset_date: str
    total_pnl: float
    max_drawdown: float
    peak_value: float
    is_active: bool = True


class TradingBotLiveService:
    """
    Live Trading Bot Service

    Handles real-time trading operations with event-driven architecture.
    """

    def __init__(self, symbol: str = SYMBOL, timeframe: str = TIMEFRAME):
        self.symbol = symbol
        self.timeframe = timeframe
        self.state = TradingState(
            capital=INITIAL_CAPITAL,
            positions={},
            trades_today=0,
            last_reset_date=datetime.now().date().isoformat(),
            total_pnl=0.0,
            max_drawdown=0.0,
            peak_value=INITIAL_CAPITAL
        )

        # Service configuration
        self.max_drawdown = LIVE_TRADING_MAX_DRAWDOWN
        self.max_trades_per_day = LIVE_TRADING_MAX_TRADES_PER_DAY
        self.min_signal_confidence = LIVE_TRADING_MIN_SIGNAL_CONFIDENCE
        self.auto_execute = LIVE_TRADING_AUTO_EXECUTE
        self.confirmation_required = LIVE_TRADING_CONFIRMATION_REQUIRED
        self.monitor_enabled = LIVE_TRADING_MONITOR_ENABLED
        self.update_interval = LIVE_TRADING_POSITION_UPDATE_INTERVAL

        # Operational state
        self.is_running = False
        self.last_signal_time = 0
        self.signal_cooldown = 60  # seconds
        self.event_processing_task: Optional[asyncio.Task] = None

        # Performance tracking
        self.performance_metrics = {
            'signals_processed': 0,
            'trades_executed': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'start_time': datetime.now(),
            'last_update': datetime.now()
        }

    async def initialize(self):
        """Initialize the trading bot service"""
        print("🚀 Initializing Trading Bot Live Service...")

        # Initialize event system
        await initialize_event_system()

        # Load existing trading state if available
        await self._load_trading_state()

        # Start event processing
        self.event_processing_task = asyncio.create_task(
            start_event_processing(self.symbol)
        )

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        print("✅ Trading Bot Live Service initialized")
        print(f"   Symbol: {self.symbol}, Timeframe: {self.timeframe}")
        print(f"   Initial Capital: ${self.state.capital:.2f}")
        print(f"   Risk per Trade: {RISK_PER_TRADE*100:.1f}%")

        audit_logger.log_system_event(
            event_type="TRADING_BOT_STARTED",
            message="Trading Bot Live Service started",
            details={
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'initial_capital': self.state.capital,
                'max_drawdown': self.max_drawdown,
                'max_trades_per_day': self.max_trades_per_day
            }
        )

    async def shutdown(self):
        """Shutdown the trading bot service"""
        print("🛑 Shutting down Trading Bot Live Service...")

        self.is_running = False

        # Cancel event processing
        if self.event_processing_task:
            self.event_processing_task.cancel()
            try:
                await self.event_processing_task
            except asyncio.CancelledError:
                pass

        # Save final state
        await self._save_trading_state()

        # Shutdown event system
        await shutdown_event_system()

        print("✅ Trading Bot Live Service shutdown complete")

        audit_logger.log_system_event(
            event_type="TRADING_BOT_STOPPED",
            message="Trading Bot Live Service stopped",
            details=self.performance_metrics
        )

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n📡 Received signal {signum}, initiating shutdown...")
        self.is_running = False

    async def start_trading(self):
        """Start the main trading loop"""
        self.is_running = True
        print("🔄 Starting trading loop...")

        try:
            while self.is_running:
                try:
                    await self._trading_cycle()
                except Exception as e:
                    print(f"❌ Error in trading cycle: {e}")
                    audit_logger.log_error(
                        error_type="TRADING_CYCLE_ERROR",
                        message=f"Trading cycle error: {str(e)}",
                        context={'symbol': self.symbol, 'timeframe': self.timeframe}
                    )
                    await asyncio.sleep(30)  # Wait before retry

        except Exception as e:
            print(f"❌ Fatal error in trading loop: {e}")
            audit_logger.log_error(
                error_type="TRADING_LOOP_ERROR",
                message=f"Fatal trading loop error: {str(e)}",
                context={'symbol': self.symbol, 'timeframe': self.timeframe}
            )

    async def _trading_cycle(self):
        """Execute one trading cycle"""
        # Reset daily counters if needed
        await self._reset_daily_counters()

        # Check risk limits
        if not await self._check_risk_limits():
            await asyncio.sleep(self.update_interval)
            return

        # Fetch latest market data
        data = await self._fetch_market_data()
        if data.empty:
            await asyncio.sleep(30)
            return

        # Process market data through indicators and signals
        processed_data = await self._process_market_data(data)

        # Check for trading signals
        await self._check_trading_signals(processed_data)

        # Update position monitoring
        if self.monitor_enabled:
            await self._update_position_monitoring()

        # Update performance metrics
        self._update_performance_metrics()

        # Wait before next cycle
        await asyncio.sleep(self.update_interval)

    async def _fetch_market_data(self) -> pd.DataFrame:
        """Fetch latest market data"""
        try:
            data = fetch_data(limit=100, symbol=self.symbol, timeframe=self.timeframe)
            if not data.empty:
                # Publish market data event
                await publish_market_data_event(self.symbol, self.timeframe, data)
            return data
        except Exception as e:
            print(f"⚠️  Failed to fetch market data: {e}")
            return pd.DataFrame()

    async def _process_market_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process market data through indicators and signals"""
        try:
            # Calculate indicators
            data = calculate_indicators(data)

            # Generate signals
            data = generate_signals(data)

            # Calculate risk management
            data = calculate_risk_management(data)

            return data

        except Exception as e:
            print(f"⚠️  Failed to process market data: {e}")
            return data

    async def _check_trading_signals(self, data: pd.DataFrame):
        """Check for trading signals and execute trades"""
        if data.empty:
            return

        latest = data.iloc[-1]
        current_time = time.time()

        # Check cooldown
        if current_time - self.last_signal_time < self.signal_cooldown:
            return

        signal_detected = False
        signal_type = None
        confidence = 0.0

        # Determine signal type and confidence
        if latest.get('Buy_Signal', 0) == 1:
            signal_type = 'BUY'
            confidence = latest.get('ML_Confidence', 0.8) if 'ML_Confidence' in latest else 0.8
            signal_detected = True
        elif latest.get('Sell_Signal', 0) == 1:
            signal_type = 'SELL'
            confidence = latest.get('ML_Confidence', 0.8) if 'ML_Confidence' in latest else 0.8
            signal_detected = True

        if signal_detected and confidence >= self.min_signal_confidence:
            await self._handle_trading_signal(signal_type, confidence, latest, current_time)

    async def _handle_trading_signal(self, signal_type: str, confidence: float,
                                   market_data: pd.Series, signal_time: float):
        """Handle a detected trading signal"""
        print(f"🎯 {signal_type} SIGNAL detected at {datetime.fromtimestamp(signal_time).strftime('%H:%M:%S')}")
        print(f"   Price: {market_data['close']:.5f}, Confidence: {confidence:.2f}")

        # Publish signal event
        indicators = {col: market_data.get(col, 0) for col in market_data.index
                     if col not in ['open', 'high', 'low', 'close', 'volume', 'Buy_Signal', 'Sell_Signal']}

        reasoning = self._generate_signal_reasoning(signal_type, market_data)
        await publish_signal_event(
            self.symbol, self.timeframe, signal_type, confidence,
            indicators, reasoning, 'ensemble'
        )

        # Check if we should execute the trade
        if self.auto_execute:
            await self._execute_trade(signal_type, market_data)
        elif self.confirmation_required:
            # In a real implementation, this would prompt for confirmation
            print("📋 Trade signal logged (confirmation required)")

        self.last_signal_time = signal_time
        self.performance_metrics['signals_processed'] += 1

    async def _execute_trade(self, signal_type: str, market_data: pd.Series):
        """Execute a trade"""
        try:
            # Calculate position size based on risk management
            position_size_pct = market_data.get('Position_Size_Percent', RISK_PER_TRADE)
            position_size = self.state.capital * position_size_pct / 100

            # For now, simulate trade execution
            # In production, this would integrate with broker APIs
            order_id = f"sim_{int(time.time())}_{signal_type.lower()}"

            # Create simulated trade record
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'type': signal_type,
                'price': market_data['close'],
                'size': position_size,
                'order_id': order_id,
                'status': 'filled'
            }

            # Update trading state
            if signal_type == 'BUY':
                # Open long position
                position = TradingPosition(
                    symbol=self.symbol,
                    side='BUY',
                    size=position_size,
                    entry_price=market_data['close'],
                    entry_time=datetime.now(),
                    stop_loss=market_data.get('Stop_Loss_Buy', market_data['close'] * 0.95),
                    take_profit=market_data.get('Take_Profit_Buy', market_data['close'] * 1.05)
                )
                self.state.positions[self.symbol] = position

            # Update daily trade count
            self.state.trades_today += 1

            # Publish trade event
            await publish_trade_event(
                self.symbol, self.timeframe, signal_type.lower(),
                position_size, market_data['close'], order_id,
                f"Signal-based {signal_type.lower()} trade"
            )

            print(f"✅ Trade executed: {signal_type} {position_size:.4f} @ ${market_data['close']:.2f}")

            # Update performance metrics
            self.performance_metrics['trades_executed'] += 1

            # Save state
            await self._save_trading_state()

        except Exception as e:
            print(f"❌ Trade execution failed: {e}")
            audit_logger.log_error(
                error_type="TRADE_EXECUTION_ERROR",
                message=f"Trade execution failed: {str(e)}",
                context={'signal_type': signal_type, 'price': market_data['close']}
            )

    def _generate_signal_reasoning(self, signal_type: str, data: pd.Series) -> str:
        """Generate human-readable reasoning for signal"""
        reasons = []

        if signal_type == 'BUY':
            if data.get('EMA9', 0) > data.get('EMA21', 0):
                reasons.append("EMA9 > EMA21")
            if data.get('RSI', 50) < 30:
                reasons.append(f"RSI oversold ({data['RSI']:.1f})")
            if data['close'] > data.get('BB_lower', data['close']):
                reasons.append("Price above BB lower")

        elif signal_type == 'SELL':
            if data.get('EMA9', 0) < data.get('EMA21', 0):
                reasons.append("EMA9 < EMA21")
            if data.get('RSI', 50) > 70:
                reasons.append(f"RSI overbought ({data['RSI']:.1f})")
            if data['close'] < data.get('BB_upper', data['close']):
                reasons.append("Price below BB upper")

        return " | ".join(reasons) if reasons else "Signal generated"

    async def _check_risk_limits(self) -> bool:
        """Check if trading is allowed based on risk limits"""
        # Check drawdown limit
        current_value = self.state.capital + sum(
            pos.current_pnl for pos in self.state.positions.values()
        )

        if current_value < INITIAL_CAPITAL * (1 - self.max_drawdown):
            print(f"🚨 EMERGENCY STOP: Drawdown limit reached ({self.max_drawdown*100}%)")
            self.is_running = False
            return False

        # Check daily trade limit
        if self.state.trades_today >= self.max_trades_per_day:
            print(f"📊 Daily trade limit reached ({self.max_trades_per_day})")
            return False

        return True

    async def _reset_daily_counters(self):
        """Reset daily counters if date changed"""
        today = datetime.now().date().isoformat()
        if today != self.state.last_reset_date:
            self.state.trades_today = 0
            self.state.last_reset_date = today
            print(f"📅 Daily reset: {today}")

    async def _update_position_monitoring(self):
        """Update position monitoring and P&L"""
        try:
            # Fetch current price for P&L calculation
            current_data = fetch_data(limit=1, symbol=self.symbol, timeframe=self.timeframe)
            if current_data.empty:
                return

            current_price = current_data.iloc[-1]['close']

            # Update positions
            for symbol, position in self.state.positions.items():
                if position.side == 'BUY':
                    position.current_pnl = (current_price - position.entry_price) * position.size
                else:
                    position.current_pnl = (position.entry_price - current_price) * position.size

                # Check stop loss / take profit
                if position.side == 'BUY':
                    if current_price <= position.stop_loss:
                        await self._close_position(symbol, current_price, "Stop Loss")
                    elif current_price >= position.take_profit:
                        await self._close_position(symbol, current_price, "Take Profit")

        except Exception as e:
            print(f"⚠️  Position monitoring failed: {e}")

    async def _close_position(self, symbol: str, exit_price: float, reason: str):
        """Close a position"""
        if symbol not in self.state.positions:
            return

        position = self.state.positions[symbol]

        # Calculate P&L
        if position.side == 'BUY':
            pnl = (exit_price - position.entry_price) * position.size
        else:
            pnl = (position.entry_price - exit_price) * position.size

        # Update capital and total P&L
        self.state.capital += pnl
        self.state.total_pnl += pnl

        # Track winning trades
        if pnl > 0:
            self.performance_metrics['winning_trades'] += 1

        # Create trade record
        trade_record = {
            'timestamp': datetime.now().isoformat(),
            'type': 'SELL' if position.side == 'BUY' else 'BUY',
            'price': exit_price,
            'size': position.size,
            'pnl': pnl,
            'order_id': f"close_{int(time.time())}",
            'reason': reason
        }

        # Publish trade event
        await publish_trade_event(
            symbol, self.timeframe, trade_record['type'].lower(),
            position.size, exit_price, trade_record['order_id'], reason
        )

        print(f"🔴 Position closed: {reason} | P&L: ${pnl:.2f}")

        # Remove position
        del self.state.positions[symbol]

        # Update drawdown tracking
        current_value = self.state.capital
        self.state.peak_value = max(self.state.peak_value, current_value)
        current_drawdown = (self.state.peak_value - current_value) / self.state.peak_value
        self.state.max_drawdown = max(self.state.max_drawdown, current_drawdown)

    def _update_performance_metrics(self):
        """Update performance metrics"""
        self.performance_metrics['total_pnl'] = self.state.total_pnl
        self.performance_metrics['last_update'] = datetime.now()

    async def _load_trading_state(self):
        """Load trading state from file"""
        try:
            if os.path.exists(LIVE_TRADING_PNL_LOG_FILE):
                with open(LIVE_TRADING_PNL_LOG_FILE, 'r') as f:
                    state_data = json.load(f)
                    # Update state (simplified - would need proper deserialization)
                    self.state.capital = state_data.get('capital', INITIAL_CAPITAL)
                    self.state.total_pnl = state_data.get('total_pnl', 0.0)
                print("📊 Loaded existing trading state")
        except Exception as e:
            print(f"⚠️  Failed to load trading state: {e}")

    async def _save_trading_state(self):
        """Save trading state to file"""
        try:
            state_data = asdict(self.state)
            state_data['last_reset_date'] = self.state.last_reset_date

            with open(LIVE_TRADING_PNL_LOG_FILE, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)

            # Also save trade log
            trade_log = []
            # In a real implementation, you'd collect trade history
            with open(LIVE_TRADING_TRADE_LOG_FILE, 'w') as f:
                json.dump(trade_log, f, indent=2, default=str)

        except Exception as e:
            print(f"⚠️  Failed to save trading state: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current service status"""
        return {
            'service': 'trading_bot_live',
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'is_running': self.is_running,
            'capital': self.state.capital,
            'total_pnl': self.state.total_pnl,
            'active_positions': len(self.state.positions),
            'trades_today': self.state.trades_today,
            'performance_metrics': self.performance_metrics,
            'last_update': datetime.now().isoformat()
        }


async def main():
    """Main entry point for Trading Bot Live Service"""
    import argparse

    parser = argparse.ArgumentParser(description='Trading Bot Live Service')
    parser.add_argument('--symbol', default=SYMBOL, help='Trading symbol')
    parser.add_argument('--timeframe', default=TIMEFRAME, help='Timeframe')
    parser.add_argument('--mode', choices=['live', 'paper'], default='paper',
                       help='Trading mode (live or paper)')

    args = parser.parse_args()

    # Create and initialize service
    service = TradingBotLiveService(args.symbol, args.timeframe)
    await service.initialize()

    try:
        # Start trading
        await service.start_trading()
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user")
    finally:
        await service.shutdown()


if __name__ == "__main__":
    asyncio.run(main())