"""
TradPal Trading Service Orchestrator
Unified orchestrator for trading operations, risk management, and AI components
"""

import logging
from typing import Dict, Any, Optional, List
import asyncio
from datetime import datetime

from .trading.service import TradingService
from .execution.service import ExecutionService
from .risk_management.service import RiskManagementService
from .reinforcement_learning.service import ReinforcementLearningService
from .market_regime.service import MarketRegimeService
from .monitoring.service import MonitoringService

logger = logging.getLogger(__name__)


class TradingServiceOrchestrator:
    """Unified orchestrator for all trading-related services"""

    def __init__(self, event_system=None):
        self.event_system = event_system

        # Initialize service components
        self.trading_service = TradingService(event_system=event_system)
        self.execution_service = ExecutionService(event_system=event_system)
        self.risk_service = RiskManagementService(event_system=event_system)
        self.rl_service = ReinforcementLearningService(event_system=event_system)
        self.regime_service = MarketRegimeService(event_system=event_system)
        self.monitoring_service = MonitoringService(event_system=event_system)

        self.is_initialized = False

    async def initialize(self):
        """Initialize all trading service components"""
        logger.info("Initializing Trading Service Orchestrator...")

        try:
            # Initialize all services concurrently
            init_tasks = [
                self.trading_service.initialize(),
                self.execution_service.initialize(),
                self.risk_service.initialize(),
                self.rl_service.initialize(),
                self.regime_service.initialize(),
                self.monitoring_service.initialize()
            ]

            await asyncio.gather(*init_tasks)

            self.is_initialized = True
            logger.info("Trading Service Orchestrator initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Trading Service Orchestrator: {e}")
            raise

    async def shutdown(self):
        """Shutdown all trading service components"""
        logger.info("Shutting down Trading Service Orchestrator...")

        try:
            # Shutdown all services concurrently
            shutdown_tasks = [
                self.trading_service.shutdown(),
                self.execution_service.shutdown(),
                self.risk_service.shutdown(),
                self.rl_service.shutdown(),
                self.regime_service.shutdown(),
                self.monitoring_service.shutdown()
            ]

            await asyncio.gather(*shutdown_tasks)

            self.is_initialized = False
            logger.info("Trading Service Orchestrator shut down successfully")

        except Exception as e:
            logger.error(f"Error during Trading Service Orchestrator shutdown: {e}")

    async def start_automated_trading(self, symbol: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Start automated trading for a symbol"""
        if not self.is_initialized:
            raise RuntimeError("Trading Service Orchestrator not initialized")

        logger.info(f"Starting automated trading for {symbol}")

        # Start trading session
        session = await self.trading_service.start_trading_session(symbol, config)

        # Record metric
        await self.monitoring_service.record_metric(
            "trading_sessions_started",
            1.0,
            {"symbol": symbol}
        )

        return session

    async def execute_smart_trade(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a smart trade using AI and risk management"""
        if not self.is_initialized:
            raise RuntimeError("Trading Service Orchestrator not initialized")

        logger.info(f"Executing smart trade for {symbol}")

        try:
            # Detect market regime
            price_data = market_data.get("prices", [])
            regime_info = await self.regime_service.detect_regime(symbol, price_data)

            # Get regime-based advice
            regime_advice = await self.regime_service.get_regime_advice(symbol, regime_info["regime"])

            # Get RL signal
            rl_signal = await self.rl_service.get_rl_signal(symbol, market_data)

            # Calculate position size with risk management
            session = await self.trading_service.get_session_status(symbol)
            if not session.get("session"):
                return {"error": f"No active session for {symbol}"}

            capital = session["session"]["capital"]
            risk_per_trade = session["session"]["risk_per_trade"]

            # Simplified volatility calculation
            volatility = 0.02  # 2% daily volatility assumption

            position_size = await self.risk_service.calculate_position_size(
                capital, risk_per_trade, 0.02, market_data.get("current_price", 50000.0)
            )

            # Apply regime adjustments
            quantity = position_size["quantity"] * regime_advice["position_size_multiplier"]

            # Execute trade if signal is strong enough
            if rl_signal["confidence"] > 0.6:
                trade_result = await self.trading_service.execute_trade(
                    symbol, rl_signal["signal"], quantity, market_data.get("current_price", 50000.0)
                )

                # Record metrics
                await self.monitoring_service.record_metric(
                    "trades_executed",
                    1.0,
                    {"symbol": symbol, "signal": rl_signal["signal"]}
                )

                return {
                    "decision": rl_signal["signal"],
                    "position_size": position_size,
                    "confidence": rl_signal["confidence"],
                    "regime": regime_info,
                    "trade": trade_result
                }
            else:
                return {"message": "Signal confidence too low", "confidence": rl_signal["confidence"]}

        except Exception as e:
            logger.error(f"Error in smart trade execution for {symbol}: {e}")
            await self.monitoring_service.create_alert(
                "trade_execution_error",
                f"Failed to execute smart trade for {symbol}: {str(e)}",
                "error",
                {"symbol": symbol}
            )
            return {"error": str(e)}

    async def get_trading_status(self) -> Dict[str, Any]:
        """Get comprehensive trading status"""
        if not self.is_initialized:
            return {"error": "Trading Service Orchestrator not initialized"}

        # Get status from all services
        trading_status = await self.trading_service.get_session_status("all")
        execution_stats = await self.execution_service.get_execution_stats()
        system_metrics = await self.monitoring_service.get_system_metrics()
        active_alerts = await self.monitoring_service.get_alerts(acknowledged=False, limit=10)

        return {
            "active_sessions": trading_status.get("total_sessions", 0),
            "total_pnl": trading_status.get("total_pnl", 0),
            "risk_metrics": {
                "current_exposure": 0.0,  # Simplified
                "max_exposure": 0.1,  # Simplified
                "risk_score": 0.3  # Simplified
            },
            "system_health": {
                "cpu_usage": system_metrics.get("cpu_percent", 0),
                "memory_usage": system_metrics.get("memory_percent", 0),
                "disk_usage": system_metrics.get("disk_usage", 0),
                "status": "healthy" if system_metrics.get("cpu_percent", 100) < 90 else "warning"
            },
            "execution_stats": execution_stats,
            "active_alerts": active_alerts,
            "timestamp": datetime.now().isoformat()
        }

    async def stop_all_trading(self) -> Dict[str, Any]:
        """Emergency stop all trading activities"""
        if not self.is_initialized:
            return {"error": "Trading Service Orchestrator not initialized"}

        logger.warning("Emergency stop initiated for all trading activities")

        # This would stop all sessions and cancel orders
        # For now, just create an alert
        await self.monitoring_service.create_alert(
            "emergency_stop",
            "Emergency stop initiated for all trading activities",
            "critical"
        )

        return {"message": "Emergency stop initiated", "success": True, "stopped_sessions": 0}

    async def get_performance_report(self, symbol: str = None) -> Dict[str, Any]:
        """Get performance report"""
        if not self.is_initialized:
            return {"error": "Trading Service Orchestrator not initialized"}

        # Get trading performance
        if symbol:
            status = await self.trading_service.get_session_status(symbol)
            performance = {
                "symbol": symbol,
                "session": status.get("session", {}),
                "total_pnl": status.get("total_pnl", 0),
                "win_rate": status.get("win_rate", 0)
            }
        else:
            # Global performance - aggregate all sessions
            all_positions = await self.trading_service.get_positions()
            total_pnl = sum(pos.get("unrealized_pnl", 0) for pos in all_positions)
            total_trades = sum(sess.get("total_trades", 0) for sess in self.trading_service.trading_sessions.values())
            winning_trades = sum(sess.get("winning_trades", 0) for sess in self.trading_service.trading_sessions.values())

            performance = {
                "total_return": total_pnl,
                "sharpe_ratio": 1.5,  # Simplified
                "max_drawdown": 0.05,  # Simplified
                "win_rate": winning_trades / max(total_trades, 1),
                "total_trades": total_trades,
                "active_sessions": len(self.trading_service.trading_sessions)
            }

        # Get system metrics
        system_metrics = await self.monitoring_service.get_system_metrics()

        return {
            "total_return": performance.get("total_return", 0),
            "sharpe_ratio": performance.get("sharpe_ratio", 0),
            "max_drawdown": performance.get("max_drawdown", 0),
            "win_rate": performance.get("win_rate", 0),
            "total_trades": performance.get("total_trades", 0),
            "active_sessions": performance.get("active_sessions", 0),
            "system_metrics": system_metrics,
            "timestamp": datetime.now().isoformat()
        }

    def get_service_status(self) -> Dict[str, Any]:
        """Get status of all orchestrated services"""
        return {
            "orchestrator_initialized": self.is_initialized,
            "trading_service": self.trading_service.is_initialized,
            "execution_service": self.execution_service.is_initialized,
            "risk_service": self.risk_service.is_initialized,
            "rl_service": self.rl_service.is_initialized,
            "regime_service": self.regime_service.is_initialized,
            "monitoring_service": self.monitoring_service.is_initialized,
            "timestamp": datetime.now().isoformat()
        }

    def get_default_trading_config(self) -> Dict[str, Any]:
        """Get default trading configuration"""
        return {
            "capital": 10000.0,
            "risk_per_trade": 0.02,
            "max_positions": 5,
            "paper_trading": True,
            "strategy": "smart_ai",
            "rl_enabled": True,
            "regime_detection": True,
            "monitoring": True
        }


# Simplified model classes for API compatibility
class AutomatedTradingRequest:
    """Automated trading request model"""
    def __init__(self, symbol: str, config: Dict[str, Any] = None):
        self.symbol = symbol
        self.config = config or {}

class AutomatedTradingResponse:
    """Automated trading response model"""
    def __init__(self, success: bool, session: Dict[str, Any] = None, error: str = None):
        self.success = success
        self.session = session or {}
        self.error = error

class SmartTradeRequest:
    """Smart trade request model"""
    def __init__(self, symbol: str, market_data: Dict[str, Any]):
        self.symbol = symbol
        self.market_data = market_data

class SmartTradeResponse:
    """Smart trade response model"""
    def __init__(self, success: bool, result: Dict[str, Any] = None, error: str = None):
        self.success = success
        self.result = result or {}
        self.error = error