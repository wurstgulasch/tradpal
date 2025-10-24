"""
Trading Service Orchestrator
Coordinates trading operations across specialized microservices
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd

from .ml_training_service.client import MLTrainingServiceClient
from .reinforcement_learning_service.client import ReinforcementLearningServiceClient
from .market_regime_service.client import MarketRegimeServiceClient
from .risk_management_service.client import RiskManagementServiceClient
from .trading_execution_service.client import TradingExecutionServiceClient
from .backtesting_service.client import BacktestingServiceClient
from .trading_bot_live_service.client import TradingBotLiveServiceClient

logger = logging.getLogger(__name__)


class TradingServiceOrchestrator:
    """Orchestrator for consolidated trading operations using specialized microservices"""

    def __init__(self):
        self.services = {}
        self.is_initialized = False
        self.active_sessions = {}

    async def initialize(self):
        """Initialize all trading services"""
        logger.info("Initializing Trading Service Orchestrator...")

        try:
            # Initialize specialized AI services
            self.services['ml_training'] = MLTrainingServiceClient()
            await self.services['ml_training'].authenticate()
            logger.info("✅ ML Training service initialized")

            self.services['reinforcement_learning'] = ReinforcementLearningServiceClient()
            await self.services['reinforcement_learning'].authenticate()
            logger.info("✅ Reinforcement Learning service initialized")

            self.services['market_regime'] = MarketRegimeServiceClient()
            await self.services['market_regime'].authenticate()
            logger.info("✅ Market Regime service initialized")

            self.services['risk_management'] = RiskManagementServiceClient()
            await self.services['risk_management'].authenticate()
            logger.info("✅ Risk Management service initialized")

            self.services['trading_execution'] = TradingExecutionServiceClient()
            await self.services['trading_execution'].authenticate()
            logger.info("✅ Trading Execution service initialized")

            # Initialize existing services
            self.services['backtesting'] = BacktestingServiceClient()
            await self.services['backtesting'].initialize()
            await self.services['backtesting'].authenticate()
            logger.info("✅ Backtesting service initialized")

            self.services['live'] = TradingBotLiveServiceClient()
            logger.info("✅ Live trading service initialized")

            self.is_initialized = True
            logger.info("✅ Trading Service Orchestrator initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Trading Service Orchestrator: {e}")
            raise

    async def shutdown(self):
        """Shutdown all trading services"""
        logger.info("Shutting down Trading Service Orchestrator...")

        for name, service in self.services.items():
            try:
                if hasattr(service, 'close'):
                    await service.close()
                logger.info(f"✅ {name} service shut down")
            except Exception as e:
                logger.error(f"❌ {name} service shutdown failed: {e}")

        self.is_initialized = False
        logger.info("✅ Trading Service Orchestrator shut down successfully")

    async def start_automated_trading(self, symbol: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Start automated trading session"""
        if not self.is_initialized:
            raise RuntimeError("Trading Service Orchestrator not initialized")

        session_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            # Create trading session
            session = {
                'session_id': session_id,
                'symbol': symbol,
                'config': config,
                'start_time': datetime.now(),
                'status': 'active',
                'performance': {}
            }

            # Start live trading if not paper trading
            if not config.get('paper_trading', True):
                await self.services['live'].start_trading(symbol, config)

            self.active_sessions[session_id] = session
            logger.info(f"✅ Started automated trading session {session_id} for {symbol}")

            return session

        except Exception as e:
            logger.error(f"❌ Failed to start automated trading for {symbol}: {e}")
            raise

    async def execute_smart_trade(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute smart trade using specialized AI services and risk management"""
        if not self.is_initialized:
            raise RuntimeError("Trading Service Orchestrator not initialized")

        try:
            # Get market regime
            price_data = market_data.get('close', []) if isinstance(market_data.get('close'), list) else [market_data.get('close', 0)]
            regime_result = await self.services['market_regime'].detect_regime(symbol, price_data)

            # Get ML predictions (if available)
            ml_signal = None
            try:
                # This would use trained ML models
                ml_signal = {"signal": "hold", "confidence": 0.5}  # Placeholder
            except Exception as e:
                logger.warning(f"ML prediction failed: {e}")

            # Get RL signal
            rl_signal = await self.services['reinforcement_learning'].get_rl_signal(symbol, market_data)

            # Combine signals (simple ensemble)
            final_signal = await self._combine_signals(ml_signal, rl_signal, regime_result)

            # Apply risk management
            risk_adjusted_signal = await self.services['risk_management'].check_risk_limits(
                market_data.get('portfolio_value', 10000),
                market_data.get('daily_loss', 0),
                0.05  # 5% max daily loss
            )

            # Execute trade if signal is valid and risk limits allow
            if (final_signal.get('action') in ['BUY', 'SELL'] and
                risk_adjusted_signal.get('within_limits', True)):

                # Calculate position size
                position_size = await self.services['risk_management'].calculate_position_size(
                    market_data.get('portfolio_value', 10000),
                    0.02,  # 2% risk per trade
                    0.02,  # 2% stop loss
                    market_data.get('close', 50000)
                )

                trade_order = {
                    "symbol": symbol,
                    "side": final_signal['action'].lower(),
                    "quantity": position_size.get('quantity', 0.01),
                    "order_type": "market"
                }

                trade_result = await self.services['trading_execution'].submit_order(trade_order)
                logger.info(f"✅ Smart trade executed: {trade_result}")
                return trade_result
            else:
                return {'action': 'HOLD', 'reason': 'No valid signal or risk limits exceeded'}

        except Exception as e:
            logger.error(f"❌ Failed to execute smart trade for {symbol}: {e}")
            raise

    async def _combine_signals(self, ml_signal: Optional[Dict], rl_signal: Dict,
                              regime_result: Dict) -> Dict[str, Any]:
        """Combine signals from different AI services"""
        # Simple signal combination logic
        signals = []

        if ml_signal and ml_signal.get('confidence', 0) > 0.6:
            signals.append(ml_signal)

        if rl_signal and rl_signal.get('confidence', 0) > 0.6:
            signals.append(rl_signal)

        if not signals:
            return {'action': 'HOLD', 'confidence': 0.0}

        # Regime-aware signal filtering
        regime = regime_result.get('regime', 'sideways')

        # In high volatility, prefer HOLD
        if regime == 'high_volatility':
            return {'action': 'HOLD', 'confidence': 0.8, 'reason': 'High volatility regime'}

        # Simple majority vote
        buy_signals = sum(1 for s in signals if s.get('signal', '').upper() == 'BUY')
        sell_signals = sum(1 for s in signals if s.get('signal', '').upper() == 'SELL')

        if buy_signals > sell_signals:
            return {'action': 'BUY', 'confidence': 0.7}
        elif sell_signals > buy_signals:
            return {'action': 'SELL', 'confidence': 0.7}
        else:
            return {'action': 'HOLD', 'confidence': 0.5}

    async def get_trading_status(self) -> Dict[str, Any]:
        """Get comprehensive trading status"""
        if not self.is_initialized:
            raise RuntimeError("Trading Service Orchestrator not initialized")

        try:
            status = {
                'orchestrator': {
                    'initialized': self.is_initialized,
                    'active_sessions': len(self.active_sessions)
                },
                'services': {},
                'sessions': list(self.active_sessions.values())
            }

            # Get status from each service
            for name, service in self.services.items():
                try:
                    if hasattr(service, 'get_status'):
                        status['services'][name] = await service.get_status()
                    else:
                        status['services'][name] = {'status': 'unknown'}
                except Exception as e:
                    status['services'][name] = {'status': 'error', 'error': str(e)}

            return status

        except Exception as e:
            logger.error(f"❌ Failed to get trading status: {e}")
            raise

    async def stop_all_trading(self) -> Dict[str, Any]:
        """Emergency stop all trading activities"""
        logger.info("🛑 Emergency stop: Stopping all trading activities")

        try:
            results = {}

            # Stop all active sessions
            for session_id, session in self.active_sessions.items():
                try:
                    symbol = session['symbol']
                    await self.services['live'].stop_trading(symbol)
                    session['status'] = 'stopped'
                    session['end_time'] = datetime.now()
                    results[session_id] = {'status': 'stopped'}
                except Exception as e:
                    results[session_id] = {'status': 'error', 'error': str(e)}

            # Clear active sessions
            self.active_sessions.clear()

            logger.info("✅ All trading activities stopped")
            return {'success': True, 'results': results}

        except Exception as e:
            logger.error(f"❌ Failed to stop trading: {e}")
            raise

    async def get_performance_report(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get performance report"""
        if not self.is_initialized:
            raise RuntimeError("Trading Service Orchestrator not initialized")

        try:
            # Get performance from backtesting service
            if symbol:
                # Get specific symbol performance
                report = await self.services['backtesting'].get_performance_report(symbol)
            else:
                # Get overall performance
                report = await self.services['backtesting'].get_overall_performance()

            return report

        except Exception as e:
            logger.error(f"❌ Failed to get performance report: {e}")
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        if not self.is_initialized:
            return {
                'status': 'not_initialized',
                'services': {},
                'timestamp': datetime.now().isoformat()
            }

        try:
            health_status = {
                'status': 'healthy',
                'services': {},
                'timestamp': datetime.now().isoformat()
            }

            # Check each service
            for name, service in self.services.items():
                try:
                    if hasattr(service, 'health_check'):
                        health_status['services'][name] = await service.health_check()
                    else:
                        health_status['services'][name] = {'status': 'unknown'}
                except Exception as e:
                    health_status['services'][name] = {'status': 'error', 'error': str(e)}
                    health_status['status'] = 'degraded'

            return health_status

        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    async def start_trading_session(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Start a new trading session using specialized services"""
        if not self.is_initialized:
            raise RuntimeError("Trading Service Orchestrator not initialized")

        symbol = config.get('symbol', 'BTC/USDT')
        session_id = f"session_{symbol.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            # Start trading session with specialized services
            session_config = config.copy()
            session_config['session_id'] = session_id

            # Initialize market regime detection
            await self.services['market_regime'].initialize()

            # Initialize risk management
            await self.services['risk_management'].initialize()

            # Initialize trading execution
            await self.services['trading_execution'].initialize()

            # Start live trading if not paper trading
            if not config.get('paper_trading', True):
                live_result = await self.services['live'].start_trading_session(session_config)
            else:
                live_result = {'status': 'paper_trading'}

            session = {
                'session_id': session_id,
                'symbol': symbol,
                'config': config,
                'services_initialized': ['market_regime', 'risk_management', 'trading_execution'],
                'live_session': live_result,
                'start_time': datetime.now(),
                'status': 'active'
            }

            self.active_sessions[session_id] = session

            logger.info(f"✅ Started trading session {session_id} for {symbol}")
            return session

        except Exception as e:
            logger.error(f"❌ Failed to start trading session: {e}")
            raise

    async def run_backtest(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run backtest using backtesting service"""
        if not self.is_initialized:
            raise RuntimeError("Trading Service Orchestrator not initialized")

        try:
            # Run backtest through backtesting service
            result = await self.services['backtesting'].run_backtest(config)

            logger.info("✅ Backtest completed successfully")
            return result

        except Exception as e:
            logger.error(f"❌ Backtest failed: {e}")
            raise

    def get_service_status(self) -> Dict[str, Any]:
        """Get orchestrator service status"""
        return {
            'orchestrator': {
                'initialized': self.is_initialized,
                'active_sessions': len(self.active_sessions),
                'services_count': len(self.services)
            },
            'timestamp': datetime.now().isoformat()
        }

    def get_default_trading_config(self) -> Dict[str, Any]:
        """Get default trading configuration"""
        return {
            'capital': 10000.0,
            'risk_per_trade': 0.02,
            'max_positions': 5,
            'paper_trading': True,
            'strategy': 'smart_ai',
            'rl_enabled': True,
            'regime_detection': True,
            'indicators': ['ema', 'rsi', 'bb', 'atr'],
            'timeframe': '1h'
        }
