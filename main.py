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
from typing import Dict, Any, Optional
from pathlib import Path

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
    from services.core.client import CoreServiceClient
    CORE_SERVICE_AVAILABLE = True
except ImportError:
    CORE_SERVICE_AVAILABLE = False

try:
    from services.ml_trainer.client import MLTrainerServiceClient
    ML_TRAINER_SERVICE_AVAILABLE = True
except ImportError:
    ML_TRAINER_SERVICE_AVAILABLE = False

try:
    from services.trading_bot_live.client import TradingBotLiveServiceClient
    TRADING_BOT_LIVE_SERVICE_AVAILABLE = True
except ImportError:
    TRADING_BOT_LIVE_SERVICE_AVAILABLE = False

try:
    from services.web_ui.client import WebUIServiceClient
    WEB_UI_SERVICE_AVAILABLE = True
except ImportError:
    WEB_UI_SERVICE_AVAILABLE = False

try:
    from services.backtesting_service.client import BacktestingServiceClient
    BACKTESTING_SERVICE_AVAILABLE = True
except ImportError:
    BACKTESTING_SERVICE_AVAILABLE = False

try:
    from services.data_service.client import DataServiceClient
    DATA_SERVICE_AVAILABLE = True
except ImportError:
    DATA_SERVICE_AVAILABLE = False

try:
    from services.discovery_service.client import DiscoveryServiceClient
    DISCOVERY_SERVICE_AVAILABLE = True
except ImportError:
    DISCOVERY_SERVICE_AVAILABLE = False

try:
    from services.notification_service.client import NotificationServiceClient
    NOTIFICATION_SERVICE_AVAILABLE = True
except ImportError:
    NOTIFICATION_SERVICE_AVAILABLE = False

try:
    from services.risk_service.client import RiskServiceClient
    RISK_SERVICE_AVAILABLE = True
except ImportError:
    RISK_SERVICE_AVAILABLE = False


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
                self.services['data'] = DataServiceClient()
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

            if RISK_SERVICE_AVAILABLE:
                self.services['risk'] = RiskServiceClient()
                logger.info("✅ Risk service initialized")

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

                        # Execute strategy if signals found
                        if signals and signals.get('signals'):
                            for signal in signals['signals']:
                                if signal.get('action') in ['BUY', 'SELL']:
                                    strategy_result = await self.services['core'].execute_strategy(
                                        symbol=SYMBOL,
                                        timeframe=TIMEFRAME,
                                        signal=signal,
                                        capital=10000.0,  # Default capital
                                        risk_config={'risk_per_trade': 0.01}
                                    )
                                    logger.info(f"Strategy executed: {strategy_result}")

                        # Log via notification service if available
                        if NOTIFICATION_SERVICE_AVAILABLE and 'notification' in self.services:
                            await self.services['notification'].log_signals(signals)

                    await asyncio.sleep(60)  # 1-minute intervals

                except Exception as e:
                    logger.error(f"Live trading error: {e}")
                    await asyncio.sleep(10)

        except Exception as e:
            logger.error(f"Live trading failed: {e}")

    async def run_backtesting(self, **kwargs) -> Dict[str, Any]:
        """Run backtesting mode"""
        logger.info("📊 Starting Backtesting Mode...")

        try:
            # Use backtesting service
            if BACKTESTING_SERVICE_AVAILABLE and 'backtesting' in self.services:
                results = await self.services['backtesting'].run_backtest(
                    symbol=kwargs.get('symbol', SYMBOL),
                    timeframe=kwargs.get('timeframe', TIMEFRAME),
                    start_date=kwargs.get('start_date'),
                    end_date=kwargs.get('end_date'),
                    initial_capital=10000.0,
                    strategy_config={'indicators': ['ema', 'rsi', 'bb']},
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
            # Use ML trainer service
            if ML_TRAINER_SERVICE_AVAILABLE and 'ml_trainer' in self.services:
                from services.ml_trainer.client import TrainingRequest

                request = TrainingRequest(
                    symbol=kwargs.get('symbol', SYMBOL),
                    timeframe=kwargs.get('timeframe', TIMEFRAME),
                    start_date=kwargs.get('start_date'),
                    end_date=kwargs.get('end_date'),
                    model_type=kwargs.get('model_type', 'random_forest'),
                    target_horizon=5,
                    use_optuna=True
                )

                training_result = await self.services['ml_trainer'].train_model(request)

                logger.info(f"ML training completed: {training_result}")
                return training_result

            elif LEGACY_AVAILABLE:
                # Legacy ML training would go here
                logger.info("Legacy ML training not implemented")
                return {"status": "legacy", "message": "Legacy ML training placeholder"}

            else:
                raise RuntimeError("No ML training service available")

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
    parser.add_argument("--profile", choices=["light", "heavy"], default="heavy", help="Performance profile")
    parser.add_argument("--data-source", choices=["kaggle", "ccxt", "yahoo"], default="kaggle", help="Data source for backtesting")

    args = parser.parse_args()

    # Load profile
    load_profile(args.profile)

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


def load_profile(profile_name: str) -> bool:
    """Load environment profile"""
    from dotenv import load_dotenv

    profile_files = {
        'light': '.env.light',
        'heavy': '.env.heavy'
    }

    if profile_name in profile_files:
        env_file = profile_files[profile_name]
        if os.path.exists(env_file):
            print(f"🔧 Loading profile: {profile_name} ({env_file})")
            load_dotenv(env_file)
            return True
        else:
            print("⚠️  Profile not found, using default .env")
            load_dotenv()
            return False
    else:
        print("⚠️  Unknown profile, using default .env")
        load_dotenv()
        return False


def validate_profile_config(profile_name: str) -> bool:
    """Validate profile configuration against requirements"""
    from config.settings import (
        ML_ENABLED, ADAPTIVE_OPTIMIZATION_ENABLED_LIVE,
        MONITORING_STACK_ENABLED, PERFORMANCE_MONITORING_ENABLED
    )

    if profile_name == 'light':
        # Light profile requirements: minimal features
        if ML_ENABLED or ADAPTIVE_OPTIMIZATION_ENABLED_LIVE or MONITORING_STACK_ENABLED or PERFORMANCE_MONITORING_ENABLED:
            return False
        return True
    elif profile_name == 'heavy':
        # Heavy profile: no restrictions, all features allowed
        return True
    else:
        # Unknown profiles are allowed (no validation)
        return True


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
