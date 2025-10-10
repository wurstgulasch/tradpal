#!/usr/bin/env python3
"""
Trading Indicator System - Modular Version
Based on EMA, RSI, Bollinger Bands and ATR for 1-minute charts.
Continuous monitoring version that only outputs signals when they occur.
"""

import sys
import os
import time
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def load_profile(profile_name):
    """
    Load the appropriate .env file based on the profile name.
    Must be called before importing config.settings.
    """
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
            print(f"⚠️  Profile '{profile_name}' not found, using default .env")
            load_dotenv()  # Fallback to default
            return False
    else:
        print(f"⚠️  Unknown profile '{profile_name}', using default .env")
        load_dotenv()  # Fallback to default
        return False

# Parse profile argument first (before other imports)
if __name__ == "__main__":
    # Quick pre-parse for profile argument
    import sys
    profile = 'default'
    for i, arg in enumerate(sys.argv):
        if arg == '--profile' and i + 1 < len(sys.argv):
            profile = sys.argv[i + 1]
            break

    # Load profile before importing settings
    if profile != 'default':
        load_profile(profile)
    else:
        # Load default .env if no profile specified
        from dotenv import load_dotenv
        load_dotenv()

# Now import everything else after profile is loaded
from src.data_fetcher import fetch_data, fetch_historical_data
from src.indicators import calculate_indicators
from src.signal_generator import generate_signals, calculate_risk_management
from src.output import save_signals_to_json, get_latest_signals
from src.logging_config import logger, log_signal, log_error, log_system_status
from src.backtester import run_backtest
from src.cache import clear_all_caches, get_cache_stats
from src.config_validation import validate_configuration_at_startup
from src.audit_logger import audit_logger
from config.settings import SYMBOL

# Optional imports for new production features
try:
    from src.secrets_manager import initialize_secrets_manager, get_secret
    SECRETS_AVAILABLE = True
except ImportError:
    print("⚠️  Secrets manager not available (optional dependencies not installed)")
    print("   Run: pip install hvac boto3")
    SECRETS_AVAILABLE = False
    def initialize_secrets_manager(*args, **kwargs):
        pass
    def get_secret(*args, **kwargs):
        return None

try:
    from src.performance import PerformanceMonitor
    PERFORMANCE_AVAILABLE = True
except ImportError:
    print("⚠️  Performance monitoring not available (optional dependencies not installed)")
    print("   Run: pip install prometheus-client psutil")
    PERFORMANCE_AVAILABLE = False
    class PerformanceMonitor:
        def __init__(self, *args, **kwargs):
            pass
        def start_monitoring(self):
            pass
        def record_signal(self, *args, **kwargs):
            pass
        def record_trade(self, *args, **kwargs):
            pass

# Optional imports for discovery functionality
try:
    from src.discovery import run_discovery, load_adaptive_config, save_adaptive_config, apply_adaptive_config
    DISCOVERY_AVAILABLE = True
except ImportError:
    print("⚠️  Discovery module not available (deap not installed)")
    print("   Run: pip install deap")
    DISCOVERY_AVAILABLE = False
    # Define dummy functions to prevent errors
    def run_discovery(*args, **kwargs):
        raise ImportError("Discovery module requires 'deap' package. Install with: pip install deap")
    def load_adaptive_config(*args, **kwargs):
        return None
    def save_adaptive_config(*args, **kwargs):
        pass
    def apply_adaptive_config(*args, **kwargs):
        return None

class AdaptiveOptimizer:
    """Manages periodic optimization during live trading."""

    def __init__(self):
        from config.settings import (
            ADAPTIVE_OPTIMIZATION_ENABLED_LIVE,
            ADAPTIVE_OPTIMIZATION_INTERVAL_HOURS_LIVE,
            ADAPTIVE_OPTIMIZATION_POPULATION,
            ADAPTIVE_OPTIMIZATION_GENERATIONS,
            ADAPTIVE_OPTIMIZATION_LOOKBACK_DAYS,
            ADAPTIVE_AUTO_APPLY_BEST_LIVE,
            ADAPTIVE_MIN_PERFORMANCE_THRESHOLD_LIVE,
            SYMBOL,
            TIMEFRAME,
            get_current_indicator_config
        )

        self.enabled = ADAPTIVE_OPTIMIZATION_ENABLED_LIVE
        self.interval_hours = ADAPTIVE_OPTIMIZATION_INTERVAL_HOURS_LIVE
        self.population = ADAPTIVE_OPTIMIZATION_POPULATION
        self.generations = ADAPTIVE_OPTIMIZATION_GENERATIONS
        self.lookback_days = ADAPTIVE_OPTIMIZATION_LOOKBACK_DAYS
        self.auto_apply = ADAPTIVE_AUTO_APPLY_BEST_LIVE
        self.min_threshold = ADAPTIVE_MIN_PERFORMANCE_THRESHOLD_LIVE
        self.symbol = SYMBOL
        self.timeframe = TIMEFRAME

        self.last_optimization = 0
        self.current_config = get_current_indicator_config()  # Use current config function

        # Load existing adaptive config if available
        self.load_existing_config()

        logger.info(f"Adaptive optimizer initialized (enabled: {self.enabled})")

    def load_existing_config(self):
        """Load existing adaptive configuration."""
        from config.settings import ADAPTIVE_CONFIG_FILE, get_current_indicator_config
        self.current_config = load_adaptive_config(ADAPTIVE_CONFIG_FILE)
        if self.current_config is None:
            # Fallback to current indicator config
            self.current_config = get_current_indicator_config()

    def should_run_optimization(self) -> bool:
        """Check if optimization should be run based on time interval."""
        if not self.enabled:
            return False

        current_time = time.time()
        time_since_last = current_time - self.last_optimization
        interval_seconds = self.interval_hours * 3600

        return time_since_last >= interval_seconds

    def run_optimization(self):
        """Run discovery optimization and potentially apply results."""
        if not self.enabled:
            return

        try:
            logger.info("🔄 Starting adaptive optimization...")

            # Calculate date range for optimization
            from datetime import datetime, timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days)

            print(f"🧬 Running adaptive optimization for {self.symbol}...")
            print(f"   Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            print(f"   Population: {self.population}, Generations: {self.generations}")

            # Run discovery
            results = run_discovery(
                symbol=self.symbol,
                exchange='kraken',
                timeframe=self.timeframe,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                population_size=self.population,
                generations=self.generations
            )

            if results and len(results) > 0:
                best_result = results[0]

                print(f"✅ Adaptive optimization completed!")
                print(f"   Best Fitness: {best_result.fitness:.2f}")
                print(f"   Win Rate: {best_result.win_rate:.1%}")
                print(f"   Total P&L: {best_result.pnl:.2f}%")

                # Save the best configuration
                from config.settings import ADAPTIVE_CONFIG_FILE
                save_adaptive_config(best_result.config, best_result.fitness, ADAPTIVE_CONFIG_FILE)

                # Apply if auto-apply is enabled and meets threshold
                if self.auto_apply and best_result.fitness >= self.min_threshold:
                    self.current_config = apply_adaptive_config(best_result.config)
                    print(f"🔄 Applied new optimized configuration (fitness: {best_result.fitness:.2f})")
                    logger.info(f"Applied adaptive configuration with fitness {best_result.fitness}")
                elif self.auto_apply and best_result.fitness < self.min_threshold:
                    print(f"⚠️ Best configuration below threshold ({best_result.fitness:.2f} < {self.min_threshold}), not applied")
                    logger.warning(f"Best configuration below threshold: {best_result.fitness} < {self.min_threshold}")

                self.last_optimization = time.time()

            else:
                logger.warning("Adaptive optimization returned no results")

        except Exception as e:
            logger.error(f"Adaptive optimization failed: {e}")
            print(f"❌ Adaptive optimization failed: {e}")

    def get_current_config(self):
        """Get current adaptive configuration for indicator calculation."""
        return self.current_config

def run_live_monitoring(performance_monitor=None):
    """Run continuous live monitoring mode."""
    print("Starting TradPal Indicator - Continuous Monitoring Mode...")
    print("Press Ctrl+C to stop monitoring\n")

    log_system_status("Live monitoring mode started")

    # Initialize adaptive optimizer only if enabled and config file exists
    from config.settings import ADAPTIVE_OPTIMIZATION_ENABLED_LIVE, ADAPTIVE_CONFIG_FILE_LIVE
    import os
    if ADAPTIVE_OPTIMIZATION_ENABLED_LIVE and os.path.exists(ADAPTIVE_CONFIG_FILE_LIVE):
        adaptive_optimizer = AdaptiveOptimizer()
    else:
        adaptive_optimizer = None

    last_signal_time = 0
    signal_cooldown = 60  # Minimum seconds between signals

    while True:
        try:
            # Check if adaptive optimization should run
            if adaptive_optimizer and adaptive_optimizer.should_run_optimization():
                adaptive_optimizer.run_optimization()

            # Fetch latest data
            data = fetch_data()

            if data.empty:
                print("No data available, retrying in 30 seconds...")
                time.sleep(30)
                continue

            # Calculate indicators (use adaptive config if available)
            if adaptive_optimizer:
                adaptive_config = adaptive_optimizer.get_current_config()
                if adaptive_config:
                    data = calculate_indicators(data, config=adaptive_config)
                else:
                    data = calculate_indicators(data)
            else:
                data = calculate_indicators(data)

            # Generate signals
            data = generate_signals(data)

            # Calculate risk management
            data = calculate_risk_management(data)

            # Check for new signals
            latest = data.iloc[-1]  # Get most recent data point

            current_time = time.time()
            has_new_signal = False

            if latest['Buy_Signal'] == 1 and (current_time - last_signal_time) > signal_cooldown:
                print(f"🟢 BUY SIGNAL at {time.strftime('%H:%M:%S')}")
                print(f"   Price: {latest['close']:.5f}")
                print(f"   RSI: {latest['RSI']:.2f}")
                print(f"   EMA9: {latest['EMA9']:.5f}, EMA21: {latest['EMA21']:.5f}")
                print(f"   Position Size: {latest['Position_Size_Percent']:.2f}% of portfolio")
                print(f"   Stop Loss: {latest['Stop_Loss_Buy']:.5f}")
                print(f"   Take Profit: {latest['Take_Profit_Buy']:.5f}")
                print(f"   Leverage: {latest['Leverage']}x")
                print()

                # Log signal for audit trail
                log_signal(
                    signal_type="BUY",
                    price=latest['close'],
                    rsi=latest['RSI'],
                    ema9=latest['EMA9'],
                    ema21=latest['EMA21'],
                    position_size_pct=latest['Position_Size_Percent'],
                    stop_loss=latest['Stop_Loss_Buy'],
                    take_profit=latest['Take_Profit_Buy'],
                    leverage=latest['Leverage']
                )

                # Record signal in performance monitor
                if performance_monitor:
                    performance_monitor.record_signal(
                        signal_type="BUY",
                        price=latest['close'],
                        rsi=latest['RSI'],
                        position_size_pct=latest['Position_Size_Percent']
                    )

                last_signal_time = current_time
                has_new_signal = True

            elif latest['Sell_Signal'] == 1 and (current_time - last_signal_time) > signal_cooldown:
                print(f"🔴 SELL SIGNAL at {time.strftime('%H:%M:%S')}")
                print(f"   Price: {latest['close']:.5f}")
                print(f"   RSI: {latest['RSI']:.2f}")
                print(f"   EMA9: {latest['EMA9']:.5f}, EMA21: {latest['EMA21']:.5f}")
                print(f"   Position Size: {latest['Position_Size_Percent']:.2f}% of portfolio")
                print(f"   Stop Loss: {latest['Stop_Loss_Buy']:.5f}")
                print(f"   Take Profit: {latest['Take_Profit_Buy']:.5f}")
                print(f"   Leverage: {latest['Leverage']}x")
                print()

                # Log signal for audit trail
                log_signal(
                    signal_type="SELL",
                    price=latest['close'],
                    rsi=latest['RSI'],
                    ema9=latest['EMA9'],
                    ema21=latest['EMA21'],
                    position_size_pct=latest['Position_Size_Percent'],
                    stop_loss=latest['Stop_Loss_Buy'],
                    take_profit=latest['Take_Profit_Buy'],
                    leverage=latest['Leverage']
                )

                # Record signal in performance monitor
                if performance_monitor:
                    performance_monitor.record_signal(
                        signal_type="SELL",
                        price=latest['close'],
                        rsi=latest['RSI'],
                        position_size_pct=latest['Position_Size_Percent']
                    )

                last_signal_time = current_time
                has_new_signal = True

            # Save signals to JSON only when there are actual signals
            if has_new_signal:
                save_signals_to_json(data)

            # Wait before next check (30 seconds for 1-minute charts)
            time.sleep(30)

        except KeyboardInterrupt:
            print("\nStopping TradPal Indicator...")
            log_system_status("Live monitoring mode stopped by user")
            break
        except Exception as e:
            print(f"Error occurred: {e}")
            log_error(f"Live monitoring error: {e}")
            print("Retrying in 60 seconds...")
            time.sleep(60)

def run_backtest_mode(args, performance_monitor=None):
    """Run backtesting mode."""
    from config.settings import LOOKBACK_DAYS
    from datetime import datetime, timedelta
    
    # Set default dates if not provided
    start_date = args.start_date
    end_date = args.end_date
    if not start_date:
        end_date_dt = datetime.now()
        start_date_dt = end_date_dt - timedelta(days=LOOKBACK_DAYS)
        start_date = start_date_dt.strftime('%Y-%m-%d')
        end_date = end_date_dt.strftime('%Y-%m-%d')
    
    print(f"Running backtest for {args.symbol} on {args.timeframe} timeframe")
    print(f"Period: {start_date} to {end_date}")
    log_system_status(f"Backtest mode started for {args.symbol} {args.timeframe}")

    try:
        results = run_backtest(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=start_date,
            end_date=end_date
        )

        print("\n📊 Backtest Results:")
        if 'backtest_results' in results and 'error' not in results['backtest_results']:
            metrics = results['backtest_results']
            print(f"Total Trades: {metrics.get('total_trades', 0)}")
            print(f"Win Rate: {metrics.get('win_rate', 0)}%")
            print(f"Total P&L: ${metrics.get('total_pnl', 0):.2f}")
            print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%")
            print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"CAGR: {metrics.get('cagr', 0):.2f}%")
            print(f"Final Capital: ${metrics.get('final_capital', 0):.2f}")

            # Record backtest metrics in performance monitor
            if performance_monitor:
                performance_monitor.record_trade(
                    pnl=metrics.get('total_pnl', 0),
                    win_rate=metrics.get('win_rate', 0),
                    total_trades=metrics.get('total_trades', 0),
                    max_drawdown=metrics.get('max_drawdown', 0)
                )

            log_system_status(f"Backtest completed: {metrics.get('total_trades', 0)} trades, {metrics.get('win_rate', 0)}% win rate")
        else:
            error_msg = results.get('backtest_results', {}).get('error', 'Unknown error')
            print(f"Backtest failed: {error_msg}")
            log_error(f"Backtest error: {error_msg}")

    except Exception as e:
        print(f"Backtest failed: {e}")
        log_error(f"Backtest error: {e}")

def run_single_analysis(performance_monitor=None):
    """Run one-time analysis mode."""
    print("Running single analysis...")

    try:
        # Fetch data
        data = fetch_historical_data()

        if data.empty:
            print("No data loaded.")
            return

        print(f"Data loaded: {len(data)} rows.")

        # Calculate indicators
        data = calculate_indicators(data)

        # Generate signals
        data = generate_signals(data)

        # Calculate risk management
        data = calculate_risk_management(data)

        # Record analysis metrics if performance monitor is available
        if performance_monitor and len(data) > 0:
            # Count signals
            buy_signals = data['Buy_Signal'].sum()
            sell_signals = data['Sell_Signal'].sum()
            performance_monitor.record_signal(signal_type="ANALYSIS_SUMMARY", price=0, rsi=0, position_size_pct=buy_signals + sell_signals)

        # Save output
        save_signals_to_json(data)

        # Show latest signals
        latest = get_latest_signals(data)
        print("Analysis completed. Latest signals:")
        print(latest)

        log_system_status("Single analysis completed")

    except Exception as e:
        print(f"Analysis failed: {e}")
        log_error(f"Single analysis error: {e}")

def initialize_rate_limiting():
    """Initialize adaptive rate limiting for data fetching."""
    try:
        from config.settings import ADAPTIVE_RATE_LIMITING_ENABLED
        if not ADAPTIVE_RATE_LIMITING_ENABLED:
            print("ℹ️  Adaptive rate limiting disabled in configuration.")
            return False

        print("Initializing adaptive rate limiting...")
        # The rate limiting is initialized automatically in data_fetcher.py
        # when ADAPTIVE_RATE_LIMITING_ENABLED is True
        print("✅ Adaptive rate limiting initialized.")
        return True

    except Exception as e:
        print(f"⚠️  Rate limiting initialization failed: {e}")
        return False

def run_discovery_mode(args):
    """Run discovery optimization mode using genetic algorithms."""
    print("🧬 Starting Discovery Mode - Genetic Algorithm Optimization")
    print(f"Optimizing indicators for {args.symbol} on {args.timeframe} timeframe")
    print(f"Population: {args.population}, Generations: {args.generations}")
    log_system_status(f"Discovery mode started for {args.symbol} {args.timeframe}")

    try:
        results = run_discovery(
            symbol=args.symbol,
            exchange='kraken',  # Default exchange
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date,
            population_size=args.population,
            generations=args.generations
        )

        print("\n🏆 Discovery Results - Top 10 Configurations:")
        print("=" * 80)

        for i, result in enumerate(results, 1):
            print(f"\n#{i} - Fitness: {result.fitness:.2f}")
            print(f"   P&L: {result.pnl:.2f}%, Win Rate: {result.win_rate:.1%}")
            print(f"   Sharpe: {result.sharpe_ratio:.2f}, Trades: {result.total_trades}")
            print(f"   Daily Perf: {result.pnl / max(result.backtest_duration_days, 1):.3f}%")

            # Show enabled indicators
            enabled = []
            config = result.config
            if config['ema']['enabled']:
                enabled.append(f"EMA{config['ema']['periods']}")
            if config['rsi']['enabled']:
                enabled.append(f"RSI({config['rsi']['period']})")
            if config['bb']['enabled']:
                enabled.append(f"BB({config['bb']['period']})")
            if config['atr']['enabled']:
                enabled.append(f"ATR({config['atr']['period']})")
            if config['adx']['enabled']:
                enabled.append("ADX")

            print(f"   Indicators: {', '.join(enabled) if enabled else 'None'}")

        print(f"\n📁 Results saved to output/discovery_results.json")
        log_system_status(f"Discovery completed: {len(results)} configurations optimized")

    except Exception as e:
        print(f"❌ Discovery failed: {e}")
        log_error(f"Discovery error: {e}")

def initialize_monitoring_stack():
    """Initialize optional monitoring stack (Prometheus, Grafana, Redis)."""
    try:
        from config.settings import MONITORING_STACK_ENABLED
        if not MONITORING_STACK_ENABLED:
            print("ℹ️  Monitoring stack disabled in configuration.")
            return False

        print("Initializing monitoring stack...")
        import subprocess

        # Check if docker-compose is available
        try:
            subprocess.run(['docker-compose', '--version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  Docker Compose not available. Monitoring stack requires Docker.")
            return False

        # Start monitoring services
        result = subprocess.run([
            'docker-compose', 'up', '-d',
            'prometheus', 'grafana', 'redis'
        ], cwd=os.path.dirname(__file__), capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Monitoring stack started successfully.")
            print("   Prometheus: http://localhost:9090")
            print("   Grafana: http://localhost:3000 (admin/admin)")
            print("   Redis: localhost:6379")
            return True
        else:
            print(f"⚠️  Failed to start monitoring stack: {result.stderr}")
            return False

    except Exception as e:
        print(f"⚠️  Monitoring stack initialization failed: {e}")
        return False

def initialize_rate_limiting():
    """Initialize adaptive rate limiting for data fetching."""
    try:
        from config.settings import ADAPTIVE_RATE_LIMITING_ENABLED
        if not ADAPTIVE_RATE_LIMITING_ENABLED:
            print("ℹ️  Adaptive rate limiting disabled in configuration.")
            return False

        print("Initializing adaptive rate limiting...")
        # The rate limiting is initialized automatically in data_fetcher.py
        # when ADAPTIVE_RATE_LIMITING_ENABLED is True
        print("✅ Adaptive rate limiting initialized.")
        return True

    except Exception as e:
        print(f"⚠️  Rate limiting initialization failed: {e}")
        return False

def print_profile_info():
    """Print information about available performance profiles."""
    print("📊 Available Performance Profiles:")
    print("   light    - Minimal resources, basic indicators only (no AI/ML)")
    print("   heavy    - Full functionality, all features enabled (with AI/ML)")
    print("   default  - Use .env file (current behavior)")
    print()

def validate_profile_config(profile_name):
    """
    Validate that the profile configuration is complete and consistent.
    Returns True if valid, False otherwise.
    """
    try:
        from config.settings import (
            ML_ENABLED, ADAPTIVE_OPTIMIZATION_ENABLED_LIVE,
            MONITORING_STACK_ENABLED, PERFORMANCE_MONITORING_ENABLED
        )

        issues = []

        # Light profile should have AI/ML disabled
        if profile_name == 'light':
            if ML_ENABLED:
                issues.append("Light profile should have ML_ENABLED=false")
            if ADAPTIVE_OPTIMIZATION_ENABLED_LIVE:
                issues.append("Light profile should have ADAPTIVE_OPTIMIZATION_ENABLED=false")
            if MONITORING_STACK_ENABLED:
                issues.append("Light profile should have MONITORING_STACK_ENABLED=false")
            if PERFORMANCE_MONITORING_ENABLED:
                issues.append("Light profile should have PERFORMANCE_MONITORING_ENABLED=false")

        # Heavy profile should have advanced features enabled
        elif profile_name == 'heavy':
            # Heavy profile can have everything enabled, but we don't enforce it
            pass

        if issues:
            print(f"⚠️  Profile validation issues for '{profile_name}':")
            for issue in issues:
                print(f"   - {issue}")
            return False

        print(f"✅ Profile '{profile_name}' validation passed")
        return True

    except Exception as e:
        print(f"⚠️  Profile validation failed: {e}")
        return False

def main():
    # Validate configuration at startup
    print("Validating configuration...")
    if not validate_configuration_at_startup():
        print("❌ Configuration validation failed. Please fix the errors above and restart.")
        sys.exit(1)
    print("✅ Configuration validation passed.\n")

    # Show profile information
    print_profile_info()

    # Initialize secrets manager for secure API key handling
    if SECRETS_AVAILABLE:
        print("Initializing secrets manager...")
        try:
            initialize_secrets_manager()
            print("✅ Secrets manager initialized.")
        except Exception as e:
            print(f"⚠️  Secrets manager initialization failed: {e}")
            print("   Continuing with environment variables...")
    else:
        print("ℹ️  Secrets manager not available (optional feature).")

    # Initialize performance monitoring
    performance_monitor = None
    if PERFORMANCE_AVAILABLE:
        print("Initializing performance monitoring...")
        try:
            performance_monitor = PerformanceMonitor()
            performance_monitor.start_monitoring()
            print("✅ Performance monitoring initialized.")
        except Exception as e:
            print(f"⚠️  Performance monitoring initialization failed: {e}")
            print("   Continuing without monitoring...")
    else:
        print("ℹ️  Performance monitoring not available (optional feature).")

    # Initialize monitoring stack (Prometheus, Grafana, Redis)
    monitoring_stack_started = initialize_monitoring_stack()

    # Initialize adaptive rate limiting
    rate_limiting_enabled = initialize_rate_limiting()

    # Log system startup
    audit_logger.log_system_event(
        event_type="SYSTEM_STARTUP",
        message="TradPal Indicator System started",
        details={
            "version": "2.0.0",
            "mode": "initialization",
            "config_validated": True,
            "secrets_manager": SECRETS_AVAILABLE,
            "performance_monitoring": PERFORMANCE_AVAILABLE,
            "monitoring_stack": monitoring_stack_started,
            "rate_limiting": rate_limiting_enabled
        }
    )

    parser = argparse.ArgumentParser(description='TradPal Trading Indicator System')
    parser.add_argument('--mode', choices=['live', 'backtest', 'analysis', 'discovery'],
                       default='live', help='Operation mode (default: live)')
    parser.add_argument('--profile', choices=['light', 'heavy'],
                       default='default', help='Performance profile (default: default .env)')
    parser.add_argument('--symbol', default=SYMBOL, help=f'Trading symbol (default: {SYMBOL})')
    parser.add_argument('--timeframe', default='1m', help='Timeframe (default: 1m)')
    parser.add_argument('--start-date', help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='Backtest end date (YYYY-MM-DD)')
    parser.add_argument('--clear-cache', action='store_true', help='Clear all caches before running')
    parser.add_argument('--population', type=int, default=50, help='GA population size (default: 50)')
    parser.add_argument('--generations', type=int, default=20, help='GA generations (default: 20)')

    args = parser.parse_args()

    # Validate profile configuration if a specific profile was selected
    if hasattr(args, 'profile') and args.profile in ['light', 'heavy']:
        validate_profile_config(args.profile)

    # Handle cache clearing
    if args.clear_cache:
        print("Clearing all caches...")
        clear_all_caches()
        cache_stats = get_cache_stats()
        print(f"Caches cleared. Stats: {cache_stats}")

    if args.mode == 'live':
        run_live_monitoring(performance_monitor)
    elif args.mode == 'backtest':
        run_backtest_mode(args, performance_monitor)
    elif args.mode == 'analysis':
        run_single_analysis(performance_monitor)
    elif args.mode == 'discovery':
        run_discovery_mode(args)
    else:
        print(f"Unknown mode: {args.mode}")

    # Log system shutdown
    audit_logger.log_system_event(
        event_type="SYSTEM_SHUTDOWN",
        message="TradPal Indicator System shutdown",
        details={
            "mode": args.mode,
            "shutdown_reason": "normal_exit"
        }
    )
    audit_logger.log_performance_metrics()

if __name__ == "__main__":
    main()
