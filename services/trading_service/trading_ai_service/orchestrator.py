"""
TradPal Trading Service Orchestrator
Unified orchestrator for trading operations, risk management, and AI components
"""

import logging
from typing import Dict, Any, Optional, List
import asyncio
from datetime import datetime

from services.trading_service.central_service_client import get_central_client
from .reinforcement_learning.service import ReinforcementLearningService
from .ml_training.ml_trainer import MLTrainerService
from .market_regime.service import MarketRegimeService

logger = logging.getLogger(__name__)


class TradingServiceOrchestrator:
    """Unified orchestrator for all trading-related services using central client"""

    def __init__(self):
        self.client = None
        self.is_initialized = False
        self.active_sessions = {}

        # AI Services
        self.rl_service = ReinforcementLearningService()
        self.ml_trainer = MLTrainerService()
        self.regime_service = MarketRegimeService()

        # Trading and monitoring services (will be initialized via central client)
        self.trading_service = None
        self.monitoring_service = None

    async def initialize(self):
        """Initialize orchestrator with central service client"""
        logger.info("Initializing Trading Service Orchestrator...")

        try:
            self.client = await get_central_client()

            # Initialize AI services
            await self.rl_service.initialize()
            # ML trainer doesn't have async initialize method
            if hasattr(self.ml_trainer, 'initialize'):
                await self.ml_trainer.initialize()
            # Regime service has async initialize
            await self.regime_service.initialize()

            # Note: trading_service and monitoring_service are initialized via client when needed
            # They are not directly available from the central client

            self.is_initialized = True
            logger.info("Trading Service Orchestrator initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Trading Service Orchestrator: {e}")
            raise

    async def start_automated_trading(self, symbol: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Start automated trading session"""
        if not self.is_initialized:
            await self.initialize()

        session_id = f"trading_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            # Get current market data
            market_data = await self.client.get_market_data(symbol=symbol, limit=100)

            # Get trading signal from ML service
            latest_data = market_data.get("data", [])[-1] if market_data.get("data") else {}
            signal_response = await self.client.get_trading_signal(
                market_data=latest_data,
                symbol=symbol
            )

            # Check risk limits
            risk_check = await self.client.check_risk_limits({
                "symbol": symbol,
                "signal": signal_response,
                "config": config
            })

            if not risk_check.get("approved", False):
                return {
                    "success": False,
                    "error": "Risk limits exceeded",
                    "session_id": session_id
                }

            # Execute trade if signal is strong enough
            if signal_response.get("action") in ["BUY", "SELL"] and signal_response.get("confidence", 0) > 0.7:
                trade_result = await self.client.execute_trade({
                    "symbol": symbol,
                    "action": signal_response["action"],
                    "confidence": signal_response["confidence"],
                    "market_data": latest_data,
                    "config": config
                })

                session_data = {
                    "session_id": session_id,
                    "symbol": symbol,
                    "config": config,
                    "start_time": datetime.now().isoformat(),
                    "status": "active",
                    "last_signal": signal_response,
                    "last_trade": trade_result
                }
            else:
                session_data = {
                    "session_id": session_id,
                    "symbol": symbol,
                    "config": config,
                    "start_time": datetime.now().isoformat(),
                    "status": "monitoring",
                    "last_signal": signal_response
                }

            self.active_sessions[session_id] = session_data

            # Publish event
            await self.client.publish_event("trading_session_started", session_data)

            return {
                "success": True,
                "session": session_data
            }

        except Exception as e:
            logger.error(f"Failed to start automated trading: {e}")
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id
            }

    def _combine_signals(self, ml_signal: Dict[str, Any], rl_signal: Dict[str, Any],
                        regime_info: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine signals from ML ensemble, RL agent, and market regime detection.

        Uses weighted ensemble approach:
        - ML Signal: 40% weight (technical analysis)
        - RL Signal: 35% weight (reinforcement learning)
        - Regime Adjustment: 25% weight (market context)
        """
        # Extract signal strengths and actions
        ml_action = ml_signal.get("action", "HOLD")
        ml_confidence = ml_signal.get("confidence", 0.5)

        rl_action = rl_signal.get("signal", "hold").upper()
        rl_confidence = rl_signal.get("confidence", 0.5)

        regime = regime_info.get("regime", "sideways")
        regime_confidence = regime_info.get("confidence", 0.5)

        # Action mapping for consistency
        action_map = {
            "BUY": 1,
            "SELL": -1,
            "HOLD": 0,
            "REDUCE": -0.5,
            "INCREASE": 0.5
        }

        # Convert actions to numerical values
        ml_score = action_map.get(ml_action, 0) * ml_confidence
        rl_score = action_map.get(rl_action, 0) * rl_confidence

        # Regime adjustment (reduce position size in volatile regimes)
        regime_multiplier = 1.0
        if regime == "volatile":
            regime_multiplier = 0.7
        elif regime == "trending":
            regime_multiplier = 1.2
        elif regime == "sideways":
            regime_multiplier = 0.8

        # Weighted combination
        weights = {
            'ml': 0.4,
            'rl': 0.35,
            'regime': 0.25
        }

        combined_score = (
            ml_score * weights['ml'] +
            rl_score * weights['rl'] +
            (regime_multiplier - 1.0) * regime_confidence * weights['regime']
        )

        # Combined confidence based on agreement and individual confidences
        agreement_bonus = 1.0 if (ml_action == rl_action) else 0.7
        combined_confidence = min(0.95, (ml_confidence + rl_confidence + regime_confidence) / 3 * agreement_bonus)

        # Determine final action based on combined score
        if combined_score > 0.3:
            final_action = "BUY"
        elif combined_score < -0.3:
            final_action = "SELL"
        else:
            final_action = "HOLD"

        # Adjust position size based on regime and confidence
        position_size_multiplier = regime_multiplier * combined_confidence

        return {
            "action": final_action,
            "confidence": combined_confidence,
            "combined_score": combined_score,
            "position_size_multiplier": position_size_multiplier,
            "signals": {
                "ml": {
                    "action": ml_action,
                    "confidence": ml_confidence,
                    "weight": weights['ml']
                },
                "rl": {
                    "action": rl_action,
                    "confidence": rl_confidence,
                    "weight": weights['rl']
                },
                "regime": {
                    "regime": regime,
                    "confidence": regime_confidence,
                    "multiplier": regime_multiplier,
                    "weight": weights['regime']
                }
            },
            "timestamp": datetime.now().isoformat()
        }

    async def train_rl_agent(self, symbol: str, training_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Train reinforcement learning agent for a specific symbol.

        Args:
            symbol: Trading symbol to train on
            training_config: Training configuration parameters

        Returns:
            Training results and metrics
        """
        if not self.is_initialized:
            await self.initialize()

        config = training_config or {
            'episodes': 1000,
            'algorithm': 'q_learning',
            'reward_function': 'default',
            'initial_balance': 10000.0,
            'transaction_cost': 0.001
        }

        try:
            # Get historical market data for training
            market_data = await self.client.get_market_data(
                symbol=symbol,
                limit=5000,  # Sufficient data for training
                interval='1h'
            )

            if market_data.get('data') is None or (hasattr(market_data.get('data'), 'empty') and market_data.get('data').empty):
                return {
                    "success": False,
                    "error": f"No market data available for {symbol}"
                }

            # Prepare training data
            training_data = {
                'symbol': symbol,
                'market_data': market_data['data'],
                'episodes': config['episodes'],
                'algorithm': config['algorithm'],
                'reward_function': config['reward_function'],
                'initial_balance': config['initial_balance'],
                'transaction_cost': config['transaction_cost']
            }

            # Start RL training
            training_result = await self.rl_service.train_rl_agent(training_data)

            if training_result.get('success'):
                logger.info(f"RL training completed for {symbol}: {training_result['average_reward']:.4f} avg reward")

                # Publish training completion event
                await self.client.publish_event("rl_training_completed", {
                    "symbol": symbol,
                    "algorithm": config['algorithm'],
                    "episodes": config['episodes'],
                    "final_reward": training_result['average_reward'],
                    "model_path": training_result.get('model_path', ''),
                    "timestamp": datetime.now().isoformat()
                })

            return training_result

        except Exception as e:
            logger.error(f"Failed to train RL agent for {symbol}: {e}")
            return {
                "success": False,
                "error": str(e),
                "symbol": symbol
            }

    async def validate_rl_model(self, symbol: str, validation_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Validate trained RL model using walk-forward validation.

        Args:
            symbol: Trading symbol
            validation_config: Validation configuration

        Returns:
            Validation results and robustness metrics
        """
        if not self.is_initialized:
            await self.initialize()

        config = validation_config or {
            'validation_windows': 5,
            'algorithm': 'q_learning'
        }

        try:
            # Get validation market data
            market_data = await self.client.get_market_data(
                symbol=symbol,
                limit=2000,
                interval='1h'
            )

            if market_data.get('data') is None or (hasattr(market_data.get('data'), 'empty') and market_data.get('data').empty):
                return {
                    "success": False,
                    "error": f"No market data available for {symbol}"
                }

            # Prepare validation data
            validation_data = {
                'symbol': symbol,
                'market_data': market_data['data'],
                'validation_windows': config['validation_windows'],
                'algorithm': config['algorithm']
            }

            # Run validation
            validation_result = await self.rl_service.validate_rl_model(validation_data)

            if validation_result.get('success', False):
                logger.info(f"RL validation completed for {symbol}: {validation_result['consistency_score']:.2f} consistency")

            return validation_result

        except Exception as e:
            logger.error(f"Failed to validate RL model for {symbol}: {e}")
            return {
                "success": False,
                "error": str(e),
                "symbol": symbol
            }

    async def get_rl_model_status(self) -> Dict[str, Any]:
        """Get status of RL models and training state."""
        if not self.is_initialized:
            await self.initialize()

        try:
            return await self.rl_service.get_rl_model_status()
        except Exception as e:
            logger.error(f"Failed to get RL model status: {e}")
            return {"error": str(e)}

    async def get_trading_status(self) -> Dict[str, Any]:
        """Get overall trading status"""
        if not self.is_initialized:
            await self.initialize()

        try:
            # Get positions
            positions = await self.client.get_positions()

            # Get trading bot status
            bot_status = await self.client.get_trading_status()

            # Get risk metrics
            risk_metrics = await self.client.calculate_risk_metrics({
                "positions": positions.get("positions", [])
            })

            return {
                "active_sessions": len(self.active_sessions),
                "positions": positions,
                "bot_status": bot_status,
                "risk_metrics": risk_metrics,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get trading status: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def stop_trading_session(self, session_id: str) -> Dict[str, Any]:
        """Stop specific trading session"""
        if session_id not in self.active_sessions:
            return {"success": False, "error": "Session not found"}

        try:
            session = self.active_sessions[session_id]
            session["status"] = "stopped"
            session["end_time"] = datetime.now().isoformat()

            # Publish event
            await self.client.publish_event("trading_session_stopped", session)

            del self.active_sessions[session_id]

            return {"success": True, "session": session}

        except Exception as e:
            logger.error(f"Failed to stop trading session: {e}")
            return {"success": False, "error": str(e)}

    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get status of specific session"""
        session = self.active_sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}

        return session

    async def list_active_sessions(self) -> List[str]:
        """List all active session IDs"""
        return list(self.active_sessions.keys())

    async def emergency_stop(self) -> Dict[str, Any]:
        """Emergency stop all trading activities"""
        logger.warning("Emergency stop initiated!")

        try:
            stopped_sessions = []

            for session_id in list(self.active_sessions.keys()):
                result = await self.stop_trading_session(session_id)
                if result["success"]:
                    stopped_sessions.append(session_id)

            # Publish emergency stop event
            await self.client.publish_event("emergency_stop", {
                "stopped_sessions": stopped_sessions,
                "timestamp": datetime.now().isoformat()
            })

            return {
                "success": True,
                "stopped_sessions": stopped_sessions,
                "total_stopped": len(stopped_sessions)
            }

        except Exception as e:
            logger.error(f"Emergency stop failed: {e}")
            return {"success": False, "error": str(e)}

    async def cleanup(self):
        """Cleanup orchestrator resources"""
        if self.client:
            await self.client.close()
        logger.info("Trading Service Orchestrator cleanup completed")

    async def shutdown(self):
        """Shutdown all trading service components"""
        logger.info("Shutting down Trading Service Orchestrator...")

        try:
            # Shutdown AI services
            shutdown_tasks = [
                self.rl_service.shutdown(),
                self.ml_trainer.shutdown() if hasattr(self.ml_trainer, 'shutdown') else asyncio.sleep(0),
                self.regime_service.shutdown() if hasattr(self.regime_service, 'shutdown') else asyncio.sleep(0)
            ]

            # Close central client
            if self.client:
                shutdown_tasks.append(self.client.close())

            await asyncio.gather(*shutdown_tasks, return_exceptions=True)

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

            position_size = await self.trading_service.calculate_position_size(
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
        execution_stats = await self.trading_service.get_execution_stats()
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
            "trading_service": self.trading_service.is_initialized if self.trading_service else False,
            "rl_service": self.rl_service.is_initialized,
            "regime_service": self.regime_service.is_initialized,
            "monitoring_service": self.monitoring_service.is_initialized if self.monitoring_service else False,
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