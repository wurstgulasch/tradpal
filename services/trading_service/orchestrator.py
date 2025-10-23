"""
Trading Service Orchestrator
Coordinates trading operations across AI, backtesting, and live trading services
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd

from .trading_ai_service.client import TradingAIServiceClient
from .backtesting_service.client import BacktestingServiceClient
from .trading_bot_live_service.client import TradingBotLiveServiceClient

logger = logging.getLogger(__name__)


class TradingServiceOrchestrator:
    """Orchestrator for consolidated trading operations"""

    def __init__(self):
        self.services = {}
        self.is_initialized = False
        self.active_sessions = {}

    async def initialize(self):
        """Initialize all trading services"""
        logger.info("Initializing Trading Service Orchestrator...")

        try:
            # Initialize AI service
            self.services['ai'] = TradingAIServiceClient()
            await self.services['ai'].authenticate()
            logger.info("✅ Trading AI service initialized")

            # Initialize backtesting service
            self.services['backtesting'] = BacktestingServiceClient()
            await self.services['backtesting'].initialize()
            await self.services['backtesting'].authenticate()
            logger.info("✅ Backtesting service initialized")

            # Initialize live trading service
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
        """Execute smart trade using AI and risk management"""
        if not self.is_initialized:
            raise RuntimeError("Trading Service Orchestrator not initialized")

        try:
            # Get AI signal
            ai_signal = await self.services['ai'].get_signal(symbol, market_data)

            # Apply risk management
            risk_adjusted_signal = await self.services['live'].apply_risk_management(
                symbol, ai_signal, market_data
            )

            # Execute trade if signal is valid
            if risk_adjusted_signal.get('action') in ['BUY', 'SELL']:
                trade_result = await self.services['live'].execute_trade(risk_adjusted_signal)
                logger.info(f"✅ Smart trade executed: {trade_result}")
                return trade_result
            else:
                return {'action': 'HOLD', 'reason': 'No valid signal'}

        except Exception as e:
            logger.error(f"❌ Failed to execute smart trade for {symbol}: {e}")
            raise

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
