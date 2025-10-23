#!/usr/bin/env python3
"""
TradPal Trading System - Hybrid Orchestrator

Version 3.0.0 - Microservices Migration in Progress
This orchestrator uses microservices where available, falls back to legacy modules.

Available modes:
- live: Live trading with microservices
- backtest: Historical backtesting
- discovery: Parameter optimization
- ml-train: Machine learning training
- multi-timeframe: MTA analysis
- paper: Paper trading simulation
- web-ui: Start web interface
"""

import asyncio
import argparse
import logging
import signal
import sys
import os
import time
from typing import Dict, Any, Optional, List
from pathlib import Path

# Add pandas and numpy imports
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import configuration
from config.settings import (
    SYMBOL, TIMEFRAME, EXCHANGE, API_KEY, API_SECRET,
    ENABLE_ML, ENABLE_DISCOVERY, ENABLE_BACKTEST,
    ENABLE_LIVE_TRADING, ENABLE_NOTIFICATIONS
)

# Legacy imports (REMOVED - migration complete)
# sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
# try:
#     from src.data_fetcher import fetch_data, fetch_historical_data
#     from src.indicators import calculate_indicators
#     from src.signal_generator import generate_signals, calculate_risk_management
#     from src.output import save_signals_to_json, get_latest_signals
#     from src.logging_config import logger as legacy_logger, log_signal, log_error, log_system_status
#     from src.backtester import run_backtest
#     from src.cache import clear_all_caches, get_cache_stats
#     from src.config_validation import validate_configuration_at_startup
#     from src.audit_logger import audit_logger
#     from src.performance import PerformanceMonitor
#     from src.discovery import run_discovery, load_adaptive_config, save_adaptive_config, apply_adaptive_config
#     LEGACY_AVAILABLE = True
# except ImportError as e:
#     logger.warning(f"Legacy modules not available: {e}")
#     LEGACY_AVAILABLE = False

LEGACY_AVAILABLE = False  # Migration complete - legacy modules removed

# Service imports (primary architecture)
try:
    from services.core_service.client import CoreServiceClient
    CORE_SERVICE_AVAILABLE = True
except ImportError:
    CORE_SERVICE_AVAILABLE = False

try:
    from services.trading_service.trading_ai_service.ml_training.ml_trainer import MLTrainerServiceClient
    ML_TRAINER_SERVICE_AVAILABLE = True
except ImportError:
    ML_TRAINER_SERVICE_AVAILABLE = False

try:
    from services.trading_service.trading_bot_live_service.client import TradingBotLiveServiceClient
    TRADING_BOT_LIVE_SERVICE_AVAILABLE = True
except ImportError:
    TRADING_BOT_LIVE_SERVICE_AVAILABLE = False

try:
    from services.ui_service.web_ui_service.client import WebUIServiceClient
    WEB_UI_SERVICE_AVAILABLE = True
except ImportError:
    WEB_UI_SERVICE_AVAILABLE = False

try:
    from services.trading_service.backtesting_service.client import BacktestingServiceClient
    BACKTESTING_SERVICE_AVAILABLE = True
except ImportError:
    BACKTESTING_SERVICE_AVAILABLE = False

try:
    from services.data_service.data_service.client import DataService
    DATA_SERVICE_AVAILABLE = True
except ImportError:
    DATA_SERVICE_AVAILABLE = False

try:
    from services.monitoring_service.discovery_service.client import DiscoveryServiceClient
    DISCOVERY_SERVICE_AVAILABLE = True
except ImportError:
    DISCOVERY_SERVICE_AVAILABLE = False

try:
    from services.monitoring_service.notification_service.client import NotificationServiceClient
    NOTIFICATION_SERVICE_AVAILABLE = True
except ImportError:
    NOTIFICATION_SERVICE_AVAILABLE = False

try:
    from services.data_service.data_service.alternative_data.client import AlternativeDataService
    ALTERNATIVE_DATA_SERVICE_AVAILABLE = True
except ImportError:
    ALTERNATIVE_DATA_SERVICE_AVAILABLE = False

# Import market regime analysis functions
try:
    from services.monitoring_service.mlops_service.market_regime_analysis import (
        detect_market_regime, analyze_multi_timeframe, get_adaptive_strategy_config, MarketRegime
    )
    MARKET_REGIME_ANALYSIS_AVAILABLE = True
except ImportError:
    MARKET_REGIME_ANALYSIS_AVAILABLE = False


class TradPalOrchestrator:
    """Main orchestrator with hybrid architecture"""

    def __init__(self):
        self.services = {}
        self.legacy_mode = not any([
            CORE_SERVICE_AVAILABLE,
            ML_TRAINER_SERVICE_AVAILABLE,
            TRADING_BOT_LIVE_SERVICE_AVAILABLE,
            WEB_UI_SERVICE_AVAILABLE
        ])
        self.running = False

        if self.legacy_mode:
            logger.info("🔄 Using legacy modules (microservices migration in progress)")
        else:
            logger.info("🚀 Using microservices architecture")

    async def initialize_services(self) -> bool:
        """Initialize available services"""
        try:
            logger.info("🚀 Initializing TradPal Services...")

            # Initialize available microservices
            if CORE_SERVICE_AVAILABLE:
                self.services['core'] = CoreServiceClient()
                await self.services['core'].authenticate()
                logger.info("✅ Core service initialized with Zero Trust")

            if ML_TRAINER_SERVICE_AVAILABLE:
                self.services['ml_trainer'] = MLTrainerServiceClient()
                logger.info("✅ ML Trainer service initialized")

            if TRADING_BOT_LIVE_SERVICE_AVAILABLE:
                self.services['trading_bot_live'] = TradingBotLiveServiceClient()
                logger.info("✅ Trading Bot Live service initialized")

            if WEB_UI_SERVICE_AVAILABLE:
                self.services['web_ui'] = WebUIServiceClient()
                logger.info("✅ Web UI service initialized")

            if BACKTESTING_SERVICE_AVAILABLE:
                self.services['backtesting'] = BacktestingServiceClient()
                # Authenticate with security service
                await self.services['backtesting'].initialize()
                await self.services['backtesting'].authenticate()
                logger.info("✅ Backtesting service initialized with Zero Trust")

            if DATA_SERVICE_AVAILABLE:
                self.services['data'] = DataService()
                await self.services['data'].initialize()
                await self.services['data'].authenticate()
                logger.info("✅ Data service initialized with Zero Trust")

            if DISCOVERY_SERVICE_AVAILABLE:
                self.services['discovery'] = DiscoveryServiceClient()
                await self.services['discovery'].authenticate()
                logger.info("✅ Discovery service initialized with Zero Trust")

            if NOTIFICATION_SERVICE_AVAILABLE:
                self.services['notification'] = NotificationServiceClient()
                logger.info("✅ Notification service initialized")

            if ALTERNATIVE_DATA_SERVICE_AVAILABLE:
                self.services['alternative_data'] = AlternativeDataService()
                await self.services['alternative_data'].initialize()
                logger.info("✅ Alternative Data service initialized")

            logger.info("✅ Services initialized successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Service initialization failed: {e}")
            return False

    async def run_live_trading(self, **kwargs) -> None:
        """Run live trading mode"""
        logger.info("📈 Starting Live Trading Mode...")

        try:
            while self.running:
                try:
                    # Fetch data using data service
                    if DATA_SERVICE_AVAILABLE and 'data' in self.services:
                        data_response = await self.services['data'].fetch_realtime_data(
                            symbol=SYMBOL,
                            timeframe=TIMEFRAME,
                            exchange=EXCHANGE
                        )
                        data = data_response.get('data', [])
                    else:
                        logger.error("No data service available for live trading")
                        await asyncio.sleep(60)
                        continue

                    # Get alternative data for enhanced decision making
                    alternative_data = {}
                    if ALTERNATIVE_DATA_SERVICE_AVAILABLE and 'alternative_data' in self.services:
                        try:
                            sentiment_data = await self.services['alternative_data'].get_sentiment_data(
                                symbol=SYMBOL, timeframe=TIMEFRAME
                            )
                            onchain_data = await self.services['alternative_data'].get_onchain_metrics(SYMBOL)
                            composite_score = await self.services['alternative_data'].get_composite_score(SYMBOL)

                            alternative_data = {
                                'sentiment': sentiment_data,
                                'onchain': onchain_data,
                                'composite_score': composite_score
                            }
                            logger.debug(f"Alternative data collected: {len(alternative_data)} sources")
                        except Exception as e:
                            logger.warning(f"Failed to get alternative data: {e}")

                                        # Get market regime information using new analysis
                    market_regime_data = {}
                    if MARKET_REGIME_ANALYSIS_AVAILABLE and data:
                        try:
                            # Convert data to DataFrame for analysis
                            df_data = pd.DataFrame(data)
                            if len(df_data) > 50:  # Need minimum data for analysis
                                # Detect market regime
                                regime_series = detect_market_regime(df_data)
                                current_regime = regime_series.iloc[-1] if len(regime_series) > 0 else MarketRegime.SIDEWAYS

                                # Get multi-timeframe analysis if we have data for multiple timeframes
                                mtf_data = {TIMEFRAME: df_data}
                                mtf_analysis = analyze_multi_timeframe(mtf_data)

                                # Get adaptive strategy configuration
                                adaptive_config = get_adaptive_strategy_config(mtf_data)

                                market_regime_data = {
                                    'current_regime': current_regime,
                                    'regime_series': regime_series,
                                    'mtf_analysis': mtf_analysis,
                                    'adaptive_config': adaptive_config,
                                    'confidence_score': adaptive_config.get('confidence_score', 0.5)
                                }

                                logger.debug(f"Market regime analysis: {current_regime.value} (confidence: {market_regime_data['confidence_score']:.2f})")
                            else:
                                logger.debug("Insufficient data for market regime analysis")
                        except Exception as e:
                            logger.warning(f"Market regime analysis failed: {e}")

                    # Get alternative data for enhanced decision making
                    alternative_data = {}
                    if ALTERNATIVE_DATA_SERVICE_AVAILABLE and 'alternative_data' in self.services:
                        try:
                            sentiment_data = await self.services['alternative_data'].get_sentiment_data(
                                symbol=SYMBOL, timeframe=TIMEFRAME
                            )
                            onchain_data = await self.services['alternative_data'].get_onchain_metrics(SYMBOL)
                            composite_score = await self.services['alternative_data'].get_composite_score(SYMBOL)

                            alternative_data = {
                                'sentiment': sentiment_data,
                                'onchain': onchain_data,
                                'composite_score': composite_score
                            }
                            logger.debug(f"Alternative data collected: {len(alternative_data)} sources")
                        except Exception as e:
                            logger.warning(f"Failed to get alternative data: {e}")

                    # Get market regime information from service (legacy)
                    market_regime = {}
                    # Market regime analysis now integrated into core service

                    # Calculate indicators and signals using core service
                    if CORE_SERVICE_AVAILABLE and 'core' in self.services:
                        # Calculate indicators
                        indicators_response = await self.services['core'].calculate_indicators(
                            symbol=SYMBOL,
                            timeframe=TIMEFRAME,
                            data=data,
                            indicators=['ema', 'rsi', 'bb', 'atr']
                        )
                        indicators = indicators_response

                        # Generate signals
                        signals_response = await self.services['core'].generate_signals(
                            symbol=SYMBOL,
                            timeframe=TIMEFRAME,
                            data=data
                        )
                        signals = signals_response

                        # Enhance signals with AI services
                        enhanced_signals = await self._enhance_signals_with_ai(
                            signals, indicators, alternative_data, market_regime_data, data
                        )

                        # Execute strategy if enhanced signals found
                        if enhanced_signals and enhanced_signals.get('signals'):
                            for signal in enhanced_signals['signals']:
                                if signal.get('action') in ['BUY', 'SELL']:
                                    strategy_result = await self.services['core'].execute_strategy(
                                        symbol=SYMBOL,
                                        timeframe=TIMEFRAME,
                                        signal=signal,
                                        capital=10000.0,  # Default capital
                                        risk_config={'risk_per_trade': 0.01}
                                    )
                                    logger.info(f"AI-enhanced strategy executed: {strategy_result}")

                        # Log via notification service if available
                        if NOTIFICATION_SERVICE_AVAILABLE and 'notification' in self.services:
                            await self.services['notification'].log_signals(enhanced_signals)

                    await asyncio.sleep(60)  # 1-minute intervals

                except Exception as e:
                    logger.error(f"Live trading error: {e}")
                    await asyncio.sleep(10)

        except Exception as e:
            logger.error(f"Live trading failed: {e}")

    async def _enhance_signals_with_ai(self, signals: Dict[str, Any], indicators: Dict[str, Any],
                                     alternative_data: Dict[str, Any], market_regime_data: Dict[str, Any],
                                     market_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Enhance trading signals using AI services.

        Args:
            signals: Base signals from core service
            indicators: Technical indicators
            alternative_data: Alternative data sources
            market_regime_data: Market regime information from new analysis
            market_data: Raw market data

        Returns:
            Enhanced signals with AI insights
        """
        enhanced_signals = signals.copy()

        try:
            # Prepare market state for RL agent
            if market_data and len(market_data) > 0:
                latest_data = market_data[-1]

                # Build comprehensive market state
                market_state = {
                    "symbol": SYMBOL,
                    "position_size": 0.0,  # Assume neutral position for decision making
                    "current_price": latest_data.get('close', 0),
                    "portfolio_value": 10000.0,  # Default portfolio value
                    "market_regime": market_regime_data.get('current_regime', MarketRegime.SIDEWAYS).value if market_regime_data else 'sideways',
                    "volatility_regime": "normal",  # Placeholder
                    "trend_strength": market_regime_data.get('mtf_analysis', {}).get('timeframe_strength', {}).get(TIMEFRAME, 0.5) if market_regime_data else 0.5,
                    "technical_indicators": {
                        "rsi": indicators.get('rsi', 50.0),
                        "macd": indicators.get('macd', 0.0),
                        "bb_position": indicators.get('bb_position', 0.5)
                    }
                }

                # Add alternative data if available
                if alternative_data.get('composite_score'):
                    market_state["alternative_score"] = alternative_data['composite_score'].get('score', 0.5)

                # Get RL-based action recommendation
                # RL functionality now integrated into trading service
                rl_action = None                # Add market regime context to signals
                if market_regime_data:
                    enhanced_signals['market_regime_context'] = {
                        'current_regime': market_regime_data['current_regime'].value,
                        'confidence_score': market_regime_data['confidence_score'],
                        'adaptive_config': market_regime_data['adaptive_config']
                    }
                    logger.debug(f"Added market regime context: {market_regime_data['current_regime'].value}")

                # Add alternative data insights
                if alternative_data:
                    enhanced_signals['alternative_data'] = alternative_data
                    logger.debug(f"Added alternative data: {len(alternative_data)} sources")

        except Exception as e:
            logger.error(f"AI signal enhancement failed: {e}")
            # Return original signals if enhancement fails
            return signals

        return enhanced_signals

    async def run_backtesting(self, **kwargs) -> Dict[str, Any]:
        """Run backtesting mode with market regime analysis"""
        logger.info("📊 Starting Backtesting Mode with Market Regime Analysis...")

        try:
            # Use backtesting service
            if BACKTESTING_SERVICE_AVAILABLE and 'backtesting' in self.services:
                # Get adaptive strategy config based on market regime analysis
                adaptive_config = {}
                if MARKET_REGIME_ANALYSIS_AVAILABLE:
                    try:
                        # Create sample data for regime analysis (simplified)
                        # In production, this would use actual historical data
                        dates = pd.date_range(start=kwargs.get('start_date', '2023-01-01'),
                                            end=kwargs.get('end_date', '2024-01-01'),
                                            freq='D')
                        sample_prices = 100 + np.random.normal(0, 1, len(dates)).cumsum()
                        sample_data = pd.DataFrame({
                            'open': sample_prices,
                            'high': sample_prices * 1.01,
                            'low': sample_prices * 0.99,
                            'close': sample_prices,
                            'volume': np.random.uniform(100000, 1000000, len(dates))
                        }, index=dates)

                        mtf_data = {kwargs.get('timeframe', TIMEFRAME): sample_data}
                        adaptive_config = get_adaptive_strategy_config(mtf_data)

                        logger.info(f"Using adaptive strategy: {adaptive_config.get('model_type', 'default')} "
                                  f"(regime: {adaptive_config.get('current_regime', 'unknown').value if hasattr(adaptive_config.get('current_regime', 'unknown'), 'value') else adaptive_config.get('current_regime', 'unknown')})")
                    except Exception as e:
                        logger.warning(f"Failed to get adaptive config: {e}")

                # Enhanced strategy config with market regime insights
                strategy_config = {
                    'indicators': ['ema', 'rsi', 'bb', 'atr'],
                    'adaptive_config': {
                        'model_type': adaptive_config.get('model_type', 'default'),
                        'current_regime': adaptive_config.get('current_regime', 'unknown').value if hasattr(adaptive_config.get('current_regime', 'unknown'), 'value') else adaptive_config.get('current_regime', 'unknown'),
                        'confidence_score': adaptive_config.get('confidence_score', 0.5)
                    },
                    'use_market_regime': MARKET_REGIME_ANALYSIS_AVAILABLE
                }

                results = await self.services['backtesting'].run_backtest(
                    symbol=kwargs.get('symbol', SYMBOL),
                    timeframe=kwargs.get('timeframe', TIMEFRAME),
                    start_date=kwargs.get('start_date'),
                    end_date=kwargs.get('end_date'),
                    initial_capital=10000.0,
                    strategy_config=strategy_config,
                    risk_config={'risk_per_trade': 0.01},
                    data_source=kwargs.get('data_source', 'kaggle')  # Use Kaggle for better data
                )
            else:
                raise RuntimeError("No backtesting service available")

            logger.info(f"✅ Backtest completed: {results.get('summary', 'N/A')}")
            return results

        except Exception as e:
            logger.error(f"Backtesting failed: {e}")
            raise

    async def run_discovery(self, **kwargs) -> Dict[str, Any]:
        """Run parameter discovery/optimization"""
        logger.info("🔍 Starting Discovery Mode...")

        try:
            # Use discovery service
            if DISCOVERY_SERVICE_AVAILABLE and 'discovery' in self.services:
                results = await self.services['discovery'].run_parameter_discovery(
                    symbol=kwargs.get('symbol', SYMBOL),
                    timeframe=kwargs.get('timeframe', TIMEFRAME),
                    start_date=kwargs.get('start_date'),
                    end_date=kwargs.get('end_date'),
                    population_size=kwargs.get('population_size', 50),
                    generations=kwargs.get('generations', 20),
                    use_walk_forward=kwargs.get('use_walk_forward', True)
                )
            else:
                raise RuntimeError("Discovery service not available")

            logger.info(f"✅ Discovery completed: {results.get('best_fitness', 'N/A')}")
            return results

        except Exception as e:
            logger.error(f"Discovery failed: {e}")
            raise

    async def run_multi_timeframe(self, **kwargs) -> None:
        """Run multi-timeframe analysis"""
        logger.info("📊 Starting Multi-Timeframe Analysis...")

        try:
            timeframes = kwargs.get('timeframes', ['1m', '5m', '1h', '1d'])

            while self.running:
                analyses = {}

                # Analyze each timeframe
                for tf in timeframes:
                    # Fetch data using data service
                    if DATA_SERVICE_AVAILABLE and 'data' in self.services:
                        data_response = await self.services['data'].fetch_historical_data(
                            symbol=SYMBOL,
                            timeframe=tf,
                            start_date='2024-01-01',  # Default start date
                            exchange=EXCHANGE
                        )
                        data = data_response.get('data', [])
                    else:
                        continue

                    # Calculate indicators and signals using core service
                    if CORE_SERVICE_AVAILABLE and 'core' in self.services:
                        indicators_response = await self.services['core'].calculate_indicators(
                            symbol=SYMBOL,
                            timeframe=tf,
                            data=data,
                            indicators=['ema', 'rsi', 'bb']
                        )
                        indicators = indicators_response

                        signals_response = await self.services['core'].generate_signals(
                            symbol=SYMBOL,
                            timeframe=tf,
                            data=data
                        )
                        signals = signals_response

                        analyses[tf] = {
                            'data': data,
                            'indicators': indicators,
                            'signals': signals.get('signals', [])
                        }

                # Log multi-timeframe analysis
                if analyses:
                    logger.info(f"Multi-timeframe analysis completed for {len(analyses)} timeframes")

                await asyncio.sleep(300)  # 5-minute intervals for MTA

        except Exception as e:
            logger.error(f"Multi-timeframe analysis failed: {e}")

    async def run_paper_trading(self, **kwargs) -> None:
        """Run paper trading simulation"""
        logger.info("📝 Starting Paper Trading Mode...")

        try:
            capital = kwargs.get('capital', 10000)
            logger.info(f"Initial capital: ${capital}")

            while self.running:
                # Fetch data using data service
                if DATA_SERVICE_AVAILABLE and 'data' in self.services:
                    data_response = await self.services['data'].fetch_realtime_data(
                        symbol=SYMBOL,
                        timeframe=TIMEFRAME,
                        exchange=EXCHANGE
                    )
                    data = data_response.get('data', [])
                else:
                    await asyncio.sleep(60)
                    continue

                # Calculate indicators and signals using core service
                if CORE_SERVICE_AVAILABLE and 'core' in self.services:
                    indicators_response = await self.services['core'].calculate_indicators(
                        symbol=SYMBOL,
                        timeframe=TIMEFRAME,
                        data=data,
                        indicators=['ema', 'rsi', 'bb', 'atr']
                    )
                    indicators = indicators_response

                    signals_response = await self.services['core'].generate_signals(
                        symbol=SYMBOL,
                        timeframe=TIMEFRAME,
                        data=data
                    )
                    signals = signals_response

                    # Simulate paper trades using trading bot live service
                    if TRADING_BOT_LIVE_SERVICE_AVAILABLE and 'trading_bot_live' in self.services and signals.get('signals'):
                        for signal in signals['signals']:
                            if signal.get('action') in ['BUY', 'SELL']:
                                trade_result = await self.services['trading_bot_live'].execute_paper_trade(signal)
                                logger.info(f"Paper trade executed: {trade_result}")

                await asyncio.sleep(60)

        except Exception as e:
            logger.error(f"Paper trading failed: {e}")

    async def run_ml_training(self, **kwargs) -> Dict[str, Any]:
        """Run machine learning training"""
        logger.info("🤖 Starting ML Training Mode...")

        try:
            # ML training now integrated into trading service
            logger.info("ML training functionality moved to trading service")
            return {"status": "integrated", "message": "ML training now available through trading service"}

        except Exception as e:
            logger.error(f"ML training failed: {e}")
            raise

    async def shutdown(self) -> None:
        """Graceful shutdown"""
        logger.info("🛑 Shutting down TradPal Orchestrator...")

        self.running = False

        # Shutdown services
        for name, service in self.services.items():
            try:
                if hasattr(service, 'close'):
                    await service.close()
                logger.info(f"✅ {name} service shut down")
            except Exception as e:
                logger.error(f"❌ {name} service shutdown failed: {e}")

        logger.info("✅ TradPal Orchestrator shut down complete")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="TradPal Trading System v3.0.0")
    parser.add_argument(
        "mode",
        choices=["live", "backtest", "discovery", "ml-train", "multi-timeframe", "paper", "web-ui"],
        help="Operation mode"
    )
    parser.add_argument("--symbol", default=SYMBOL, help="Trading symbol")
    parser.add_argument("--timeframe", default=TIMEFRAME, help="Timeframe")
    parser.add_argument("--start-date", help="Start date for backtesting/training")
    parser.add_argument("--end-date", help="End date for backtesting/training")
    parser.add_argument("--capital", type=float, default=10000, help="Initial capital for paper trading")
    parser.add_argument("--iterations", type=int, default=100, help="Max iterations for discovery")
    parser.add_argument("--population-size", type=int, default=50, help="GA population size")
    parser.add_argument("--generations", type=int, default=20, help="GA generations")
    parser.add_argument("--use-walk-forward", action="store_true", default=True, help="Use walk-forward analysis")
    parser.add_argument("--model-type", default="ensemble", help="ML model type")
    parser.add_argument("--data-source", choices=["kaggle", "ccxt", "yahoo"], default="kaggle", help="Data source for backtesting")

    args = parser.parse_args()

    # Initialize orchestrator
    orchestrator = TradPalOrchestrator()

    # Signal handlers
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(orchestrator.shutdown())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Initialize services
        if not await orchestrator.initialize_services():
            logger.error("❌ Service initialization failed")
            return 1

        orchestrator.running = True

        # Execute requested mode
        if args.mode == "live":
            await orchestrator.run_live_trading()
        elif args.mode == "backtest":
            results = await orchestrator.run_backtesting(
                symbol=args.symbol,
                timeframe=args.timeframe,
                start_date=args.start_date,
                end_date=args.end_date,
                data_source=getattr(args, 'data_source', 'kaggle')  # Use Kaggle for better historical data
            )
            print(f"Backtest Results: {results}")
        elif args.mode == "discovery":
            results = await orchestrator.run_discovery(
                symbol=args.symbol,
                timeframe=args.timeframe,
                start_date=args.start_date,
                end_date=args.end_date,
                population_size=args.population_size,
                generations=args.generations,
                use_walk_forward=args.use_walk_forward
            )
            print(f"Discovery Results: {results}")
        elif args.mode == "multi-timeframe":
            await orchestrator.run_multi_timeframe()
        elif args.mode == "paper":
            await orchestrator.run_paper_trading(capital=args.capital)
        elif args.mode == "ml-train":
            results = await orchestrator.run_ml_training(
                symbol=args.symbol,
                timeframe=args.timeframe,
                start_date=args.start_date,
                end_date=args.end_date
            )
            print(f"ML Training Results: {results}")
        elif args.mode == "ml-train":
            print("ML Training not yet migrated to microservices")

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        return 1
    finally:
        await orchestrator.shutdown()

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
