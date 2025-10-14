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

def run_paper_trading_mode(args, performance_monitor=None):
    """Run paper trading mode with simulated orders."""
    print("📈 Starting TradPal - Paper Trading Mode")
    print("💰 Virtual portfolio trading - NO REAL MONEY AT RISK")
    print("Press Ctrl+C to stop paper trading\n")

    log_system_status("Paper trading mode started")

    # Initialize virtual portfolio
    virtual_capital = 10000.0  # $10,000 starting capital
    current_position = None  # None, 'long', or 'short'
    entry_price = 0.0
    position_size = 0.0
    total_trades = 0
    winning_trades = 0
    total_pnl = 0.0

    print(f"💼 Initial Virtual Capital: ${virtual_capital:.2f}")
    print(f"📊 Trading {args.symbol} on {args.timeframe} timeframe\n")

    # Initialize adaptive optimizer only if enabled and config file exists
    from config.settings import ADAPTIVE_OPTIMIZATION_ENABLED_LIVE, ADAPTIVE_CONFIG_FILE_LIVE
    import os
    if ADAPTIVE_OPTIMIZATION_ENABLED_LIVE and ADAPTIVE_CONFIG_FILE_LIVE and os.path.exists(ADAPTIVE_CONFIG_FILE_LIVE):
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

            # Handle BUY signals
            if latest['Buy_Signal'] == 1 and (current_time - last_signal_time) > signal_cooldown:
                if current_position is None:  # Only enter if no current position
                    entry_price = latest['close']
                    position_size_pct = latest['Position_Size_Percent']
                    position_size = (virtual_capital * position_size_pct / 100) / entry_price
                    stop_loss = latest['Stop_Loss_Buy']
                    take_profit = latest['Take_Profit_Buy']
                    leverage = latest['Leverage']

                    current_position = 'long'

                    print(f"🟢 PAPER BUY SIGNAL at {time.strftime('%H:%M:%S')}")
                    print(f"   Entry Price: ${entry_price:.5f}")
                    print(f"   Position Size: {position_size:.6f} {args.symbol.split('/')[0]} (${position_size * entry_price:.2f})")
                    print(f"   Stop Loss: ${stop_loss:.5f}")
                    print(f"   Take Profit: ${take_profit:.5f}")
                    print(f"   Leverage: {leverage}x")
                    print(f"   Virtual Capital: ${virtual_capital:.2f}")
                    print()

                    # Log signal for audit trail
                    log_signal(
                        signal_type="PAPER_BUY",
                        price=entry_price,
                        rsi=latest['RSI'],
                        ema9=latest['EMA9'],
                        ema21=latest['EMA21'],
                        position_size_pct=position_size_pct,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        leverage=leverage
                    )

                    last_signal_time = current_time
                    has_new_signal = True

            # Handle SELL signals
            elif latest['Sell_Signal'] == 1 and (current_time - last_signal_time) > signal_cooldown:
                if current_position == 'long':  # Only exit if we have a long position
                    exit_price = latest['close']
                    pnl = (exit_price - entry_price) * position_size * leverage
                    total_pnl += pnl
                    virtual_capital += pnl
                    total_trades += 1

                    if pnl > 0:
                        winning_trades += 1

                    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

                    print(f"🔴 PAPER SELL SIGNAL at {time.strftime('%H:%M:%S')}")
                    print(f"   Exit Price: ${exit_price:.5f}")
                    print(f"   P&L: ${pnl:.2f} ({'+' if pnl > 0 else ''}{pnl/entry_price*100:.2f}%)")
                    print(f"   Virtual Capital: ${virtual_capital:.2f}")
                    print(f"   Total Trades: {total_trades}, Win Rate: {win_rate:.1f}%")
                    print(f"   Total P&L: ${total_pnl:.2f}")
                    print()

                    # Log signal for audit trail
                    log_signal(
                        signal_type="PAPER_SELL",
                        price=exit_price,
                        rsi=latest['RSI'],
                        ema9=latest['EMA9'],
                        ema21=latest['EMA21'],
                        position_size_pct=0,  # Exit position
                        stop_loss=0,
                        take_profit=0,
                        leverage=leverage
                    )

                    # Reset position
                    current_position = None
                    entry_price = 0.0
                    position_size = 0.0

                    last_signal_time = current_time
                    has_new_signal = True

                elif current_position is None:  # Could implement short selling here
                    print(f"🔴 SELL SIGNAL at {time.strftime('%H:%M:%S')} (no position to sell)")
                    print(f"   Price: {latest['close']:.5f}")
                    print()

            # Check for stop loss / take profit hits (simplified)
            elif current_position == 'long':
                current_price = latest['close']

                # Check stop loss
                if current_price <= stop_loss:
                    exit_price = stop_loss
                    pnl = (exit_price - entry_price) * position_size * leverage
                    total_pnl += pnl
                    virtual_capital += pnl
                    total_trades += 1

                    print(f"🛑 PAPER STOP LOSS HIT at {time.strftime('%H:%M:%S')}")
                    print(f"   Exit Price: ${exit_price:.5f}")
                    print(f"   P&L: ${pnl:.2f} ({pnl/entry_price*100:.2f}%)")
                    print(f"   Virtual Capital: ${virtual_capital:.2f}")
                    print()

                    current_position = None
                    entry_price = 0.0
                    position_size = 0.0

                # Check take profit
                elif current_price >= take_profit:
                    exit_price = take_profit
                    pnl = (exit_price - entry_price) * position_size * leverage
                    total_pnl += pnl
                    virtual_capital += pnl
                    total_trades += 1
                    winning_trades += 1

                    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

                    print(f"🎯 PAPER TAKE PROFIT HIT at {time.strftime('%H:%M:%S')}")
                    print(f"   Exit Price: ${exit_price:.5f}")
                    print(f"   P&L: ${pnl:.2f} ({pnl/entry_price*100:.2f}%)")
                    print(f"   Virtual Capital: ${virtual_capital:.2f}")
                    print(f"   Total Trades: {total_trades}, Win Rate: {win_rate:.1f}%")
                    print()

                    current_position = None
                    entry_price = 0.0
                    position_size = 0.0

            # Save signals to JSON only when there are actual signals
            if has_new_signal:
                save_signals_to_json(data)

            # Wait before next check (30 seconds for 1-minute charts)
            time.sleep(30)

        except KeyboardInterrupt:
            print("\nStopping Paper Trading...")
            print(f"📊 Final Results:")
            print(f"   Virtual Capital: ${virtual_capital:.2f}")
            print(f"   Total Trades: {total_trades}")
            if total_trades > 0:
                win_rate = winning_trades / total_trades * 100
                print(f"   Win Rate: {win_rate:.1f}%")
                print(f"   Total P&L: ${total_pnl:.2f}")
                print(f"   Return: {total_pnl/10000*100:.2f}%")

            log_system_status("Paper trading mode stopped by user")
            break
        except Exception as e:
            print(f"Error occurred: {e}")
            log_error(f"Paper trading error: {e}")
            print("Retrying in 60 seconds...")
            time.sleep(60)

def run_live_monitoring(performance_monitor=None):
    """Run continuous live monitoring mode."""
    print("Starting TradPal - Continuous Monitoring Mode...")
    print("Press Ctrl+C to stop monitoring\n")

    log_system_status("Live monitoring mode started")

    # Initialize adaptive optimizer only if enabled and config file exists
    from config.settings import ADAPTIVE_OPTIMIZATION_ENABLED_LIVE, ADAPTIVE_CONFIG_FILE_LIVE
    import os
    if ADAPTIVE_OPTIMIZATION_ENABLED_LIVE and ADAPTIVE_CONFIG_FILE_LIVE and os.path.exists(ADAPTIVE_CONFIG_FILE_LIVE):
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
            print("\nStopping TradPal...")
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
                    symbol=args.symbol,
                    trade_type='backtest',
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

def calculate_buy_hold_performance(symbol, exchange, timeframe, start_date=None, end_date=None):
    """
    Calculate Buy & Hold performance for the given symbol and timeframe.

    Args:
        symbol: Trading symbol (e.g., 'BTC/USDT')
        exchange: Exchange name
        timeframe: Timeframe string
        start_date: Start date for calculation
        end_date: End date for calculation

    Returns:
        Float: Buy & Hold return percentage
    """
    try:
        from src.data_fetcher import fetch_historical_data
        from datetime import datetime, timedelta

        # Set default dates if not provided
        if not start_date:
            end_date_dt = datetime.now()
            start_date_dt = end_date_dt - timedelta(days=365)  # Default 1 year
            start_date = start_date_dt.strftime('%Y-%m-%d')
            end_date = end_date_dt.strftime('%Y-%m-%d')

        # Fetch historical data
        data = fetch_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            limit=5000
        )

        if data.empty or len(data) < 2:
            print(f"⚠️  Insufficient data for Buy & Hold calculation: {len(data)} points")
            return 0.0

        # Calculate Buy & Hold return: (final_price - initial_price) / initial_price * 100
        initial_price = data['close'].iloc[0]
        final_price = data['close'].iloc[-1]

        buy_hold_return = ((final_price - initial_price) / initial_price) * 100

        return round(buy_hold_return, 2)

    except Exception as e:
        print(f"⚠️  Error calculating Buy & Hold performance: {e}")
        return 0.0

# Optional: Initialize monitoring stack (Prometheus, Grafana, Redis)
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

def run_discovery_mode(args):
    """Run discovery optimization mode using genetic algorithms."""
    print("🧬 Starting Discovery Mode - Genetic Algorithm Optimization")
    print(f"Optimizing indicators for {args.symbol} on {args.timeframe} timeframe")
    print(f"Population: {args.population}, Generations: {args.generations}")

    try:
        from src.discovery import run_discovery

        results = run_discovery(
            symbol=args.symbol,
            exchange='kraken',  # Default exchange
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date,
            population_size=args.population,
            generations=args.generations
        )

        # Calculate Buy & Hold performance for comparison
        buy_hold_performance = calculate_buy_hold_performance(
            symbol=args.symbol,
            exchange='kraken',
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date
        )

        print("\n🏆 Discovery Results - Top 10 Configurations:")
        print("=" * 80)
        print(f"📊 Buy & Hold Performance (Benchmark): {buy_hold_performance:.2f}%")
        print("=" * 80)

        for i, result in enumerate(results, 1):
            print(f"\n#{i} - Fitness: {result.fitness:.2f}")
            print(f"   P&L: {result.pnl:.2f}%, Win Rate: {result.win_rate:.1%}")
            print(f"   Sharpe: {result.sharpe_ratio:.2f}, Trades: {result.total_trades}")
            print(f"   Daily Perf: {result.pnl / max(result.backtest_duration_days, 1):.3f}%")

            # Show performance vs benchmark
            vs_benchmark = result.pnl - buy_hold_performance
            print(f"   vs Benchmark: {vs_benchmark:+.2f}% ({'better' if vs_benchmark > 0 else 'worse'})")

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

    except Exception as e:
        print(f"❌ Discovery failed: {e}")
        import traceback
        traceback.print_exc()

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
        message="TradPal System started",
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
    parser.add_argument('--mode', choices=['live', 'backtest', 'analysis', 'discovery', 'paper', 'multi-model'],
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
    parser.add_argument('--models', nargs='+', 
                       choices=['traditional_ml', 'lstm', 'transformer', 'ensemble'],
                       help='ML models to test in multi-model backtest (default: all available)')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Maximum parallel workers for multi-model backtest (default: 4)')
    parser.add_argument('--train-missing', action='store_true',
                       help='Automatically train missing ML models before backtesting')
    parser.add_argument('--retrain-all', action='store_true',
                       help='Retrain all ML models before backtesting (forces complete retraining)')

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
    elif args.mode == 'paper':
        run_paper_trading_mode(args, performance_monitor)
    elif args.mode == 'multi-model':
        run_multi_model_backtest_mode(args, performance_monitor)
    else:
        print(f"Unknown mode: {args.mode}")

    # Log system shutdown
    audit_logger.log_system_event(
        event_type="SYSTEM_SHUTDOWN",
        message="TradPal System shutdown",
        details={
            "mode": args.mode,
            "shutdown_reason": "normal_exit"
        }
    )
    audit_logger.log_performance_metrics()

def run_multi_model_backtest_mode(args, performance_monitor=None):
    """Run multi-model backtesting mode to compare ML models."""
    from config.settings import LOOKBACK_DAYS
    from datetime import datetime, timedelta
    from src.backtester import run_multi_model_backtest
    
    # Set default dates if not provided
    start_date = args.start_date
    end_date = args.end_date
    if not start_date:
        end_date_dt = datetime.now()
        start_date_dt = end_date_dt - timedelta(days=LOOKBACK_DAYS)
        start_date = start_date_dt.strftime('%Y-%m-%d')
        end_date = end_date_dt.strftime('%Y-%m-%d')
    
    print(f"🚀 Running multi-model backtest for {args.symbol} on {args.timeframe} timeframe")
    print(f"📊 Testing models: {args.models or 'all available'}")
    print(f"⏱️  Period: {start_date} to {end_date}")
    print(f"🔄 Max workers: {args.max_workers}")
    if hasattr(args, 'train_missing') and args.train_missing:
        print("🤖 Auto-training missing models: ENABLED")
    if hasattr(args, 'retrain_all') and args.retrain_all:
        print("🔄 Force retraining all models: ENABLED")
    
    log_system_status(f"Multi-model backtest started for {args.symbol} {args.timeframe}")
    
    # Auto-train models if requested
    if hasattr(args, 'retrain_all') and args.retrain_all:
        print("\n🔄 Force retraining all ML models...")
        _auto_train_models(args.models, force_retrain=True, symbol=args.symbol, timeframe=args.timeframe)
    elif hasattr(args, 'train_missing') and args.train_missing:
        print("\n🤖 Auto-training missing ML models...")
        _auto_train_models(args.models, force_retrain=False, symbol=args.symbol, timeframe=args.timeframe)
    
    try:
        # Run multi-model backtest
        results = run_multi_model_backtest(
            symbol=args.symbol,
            exchange='kraken',  # Default exchange
            timeframe=args.timeframe,
            start_date=start_date,
            end_date=end_date,
            models_to_test=args.models,
            max_workers=args.max_workers
        )
        
        if 'error' in results:
            print(f"❌ Multi-model backtest failed: {results['error']}")
            log_error(f"Multi-model backtest error: {results['error']}")
            return
        
        # Display results
        print("\n🏆 Multi-Model Backtest Results:")
        print("=" * 80)
        print(f"Best Model: {results.get('best_model', 'N/A')}")
        print(f"Models Tested: {', '.join(results.get('successful_models', []))}")
        
        if results.get('failed_models'):
            print(f"Failed Models: {', '.join(results['failed_models'])}")
        
        print("\n📊 Performance Comparison:")
        print("-" * 80)
        
        # Display comparison table
        if 'comparison' in results:
            comparison_data = results['comparison']
            if comparison_data:
                # Print header
                headers = list(comparison_data[0].keys())
                print(f"{headers[0]:<15} {' | '.join(f'{h:<12}' for h in headers[1:])}")
                print("-" * (15 + 13 * (len(headers) - 1)))
                
                # Print data rows
                for row in comparison_data:
                    values = [f"{row[h]:<12.2f}" if isinstance(row[h], (int, float)) else f"{row[h]:<12}" for h in headers[1:]]
                    print(f"{row[headers[0]]:<15} {' | '.join(values)}")
        
        print("\n🏅 Rankings by Metric:")
        rankings = results.get('rankings', {})
        for metric, ranking in rankings.items():
            print(f"  {metric}: {' > '.join(ranking)}")
        
        print(f"\n🎯 Best Model Metrics:")
        best_metrics = results.get('best_metrics', {})
        if best_metrics and 'error' not in best_metrics:
            print(f"  Total Trades: {best_metrics.get('total_trades', 0)}")
            print(f"  Win Rate: {best_metrics.get('win_rate', 0)}%")
            print(f"  Total P&L: ${best_metrics.get('total_pnl', 0):.2f}")
            print(f"  Sharpe Ratio: {best_metrics.get('sharpe_ratio', 0):.2f}")
            print(f"  Max Drawdown: {best_metrics.get('max_drawdown', 0):.2f}%")
            print(f"  CAGR: {best_metrics.get('cagr', 0):.2f}%")
            print(f"  Final Capital: ${best_metrics.get('final_capital', 0):.2f}")
            
            # Record best model metrics in performance monitor
            if performance_monitor:
                performance_monitor.record_trade(
                    symbol=args.symbol,
                    trade_type='backtest',
                    pnl=best_metrics.get('total_pnl', 0)
                )
        
        log_system_status(f"Multi-model backtest completed: Best model is {results.get('best_model', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Multi-model backtest failed: {e}")
        log_error(f"Multi-model backtest error: {e}")


def _auto_train_models(models_to_test=None, force_retrain=False, symbol='BTC/USDT', timeframe='1m'):
    """
    Automatically train missing ML models or retrain all models.
    
    Args:
        models_to_test: List of models to check/train
        force_retrain: If True, retrain all models regardless of current status
        symbol: Trading symbol for training data
        timeframe: Timeframe for training data
    """
    from scripts.train_ml_model import train_ml_model as train_traditional_ml
    from src.ml_predictor import (
        get_ml_predictor, is_ml_available,
        get_lstm_predictor, is_lstm_available,
        get_transformer_predictor, is_transformer_available
    )
    
    # Available models to potentially train
    available_models = ['traditional_ml', 'lstm', 'transformer']
    
    if models_to_test is None:
        models_to_train = available_models
    else:
        models_to_train = [m for m in models_to_test if m in available_models]
    
    print(f"🔍 Checking models: {models_to_train}")
    
    trained_count = 0
    failed_count = 0
    
    for model_type in models_to_train:
        try:
            needs_training = force_retrain
            
            if not needs_training:
                # Check if model is already trained
                if model_type == 'traditional_ml' and is_ml_available():
                    predictor = get_ml_predictor()
                    needs_training = not (predictor and predictor.is_trained)
                elif model_type == 'lstm' and is_lstm_available():
                    predictor = get_lstm_predictor()
                    needs_training = not (predictor and predictor.is_trained)
                elif model_type == 'transformer' and is_transformer_available():
                    predictor = get_transformer_predictor()
                    needs_training = not (predictor and predictor.is_trained)
            
            if needs_training:
                print(f"🤖 Training {model_type} model...")
                
                if model_type == 'traditional_ml':
                    success = train_traditional_ml(symbol=symbol, timeframe=timeframe, force_retrain=force_retrain)
                elif model_type == 'lstm':
                    # For LSTM, we need to implement training logic here
                    success = _train_lstm_model(symbol, timeframe, force_retrain)
                elif model_type == 'transformer':
                    # For Transformer, we need to implement training logic here
                    success = _train_transformer_model(symbol, timeframe, force_retrain)
                else:
                    print(f"⚠️  Unknown model type: {model_type}")
                    continue
                
                if success:
                    print(f"✅ {model_type} training completed successfully")
                    trained_count += 1
                else:
                    print(f"❌ {model_type} training failed")
                    failed_count += 1
            else:
                print(f"✅ {model_type} already trained")
                
        except Exception as e:
            print(f"❌ Error training {model_type}: {e}")
            failed_count += 1
    
    print(f"\n📊 Training Summary: {trained_count} trained, {failed_count} failed")
    
    if failed_count > 0:
        print("⚠️  Some models failed to train. Backtest will only use successfully trained models.")
    
    return trained_count, failed_count


def _train_lstm_model(symbol: str, timeframe: str, force_retrain: bool = False):
    """
    Train LSTM model for signal prediction.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe for training
        force_retrain: Force retraining even if model exists
        
    Returns:
        bool: True if training successful, False otherwise
    """
    from src.ml_predictor import get_lstm_predictor, is_lstm_available
    from src.data_fetcher import fetch_historical_data
    from src.indicators import calculate_indicators
    from config.settings import LOOKBACK_DAYS
    from datetime import datetime, timedelta
    
    if not is_lstm_available():
        print("❌ TensorFlow not available for LSTM training")
        return False
    
    try:
        lstm_predictor = get_lstm_predictor()
        if lstm_predictor is None:
            print("❌ Failed to initialize LSTM predictor")
            return False
        
        # Check if already trained
        if lstm_predictor.is_trained and not force_retrain:
            print("ℹ️  LSTM model already trained")
            return True
        
        # Fetch historical data
        print("📊 Fetching data for LSTM training...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=LOOKBACK_DAYS)
        
        df = fetch_historical_data(
            symbol=symbol,
            exchange_name='kraken',
            timeframe=timeframe,
            start_date=start_date,
            limit=5000
        )
        
        if df.empty or len(df) < 200:
            print(f"❌ Insufficient data for LSTM training: {len(df)} samples")
            return False
        
        # Calculate indicators
        df = calculate_indicators(df)
        df_clean = df.dropna()
        
        if len(df_clean) < 100:
            print(f"❌ Insufficient clean data for LSTM training: {len(df_clean)} samples")
            return False
        
        # Train the model
        print("🎯 Training LSTM model...")
        result = lstm_predictor.train_model(df_clean)
        
        if result and result.get('success', False):
            print("✅ LSTM training completed successfully")
            return True
        else:
            print(f"❌ LSTM training failed: {result.get('error', 'Unknown error') if result else 'No result'}")
            return False
            
    except Exception as e:
        print(f"❌ LSTM training error: {e}")
        return False


def _train_transformer_model(symbol: str, timeframe: str, force_retrain: bool = False):
    """
    Train Transformer model for signal prediction.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe for training
        force_retrain: Force retraining even if model exists
        
    Returns:
        bool: True if training successful, False otherwise
    """
    from src.ml_predictor import get_transformer_predictor, is_transformer_available
    from src.data_fetcher import fetch_historical_data
    from src.indicators import calculate_indicators
    from config.settings import LOOKBACK_DAYS
    from datetime import datetime, timedelta
    
    if not is_transformer_available():
        print("❌ PyTorch not available for Transformer training")
        return False
    
    try:
        transformer_predictor = get_transformer_predictor()
        if transformer_predictor is None:
            print("❌ Failed to initialize Transformer predictor")
            return False
        
        # Check if already trained
        if transformer_predictor.is_trained and not force_retrain:
            print("ℹ️  Transformer model already trained")
            return True
        
        # Fetch historical data
        print("📊 Fetching data for Transformer training...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=LOOKBACK_DAYS)
        
        df = fetch_historical_data(
            symbol=symbol,
            exchange_name='kraken',
            timeframe=timeframe,
            start_date=start_date,
            limit=5000
        )
        
        if df.empty or len(df) < 200:
            print(f"❌ Insufficient data for Transformer training: {len(df)} samples")
            return False
        
        # Calculate indicators
        df = calculate_indicators(df)
        df_clean = df.dropna()
        
        if len(df_clean) < 100:
            print(f"❌ Insufficient clean data for Transformer training: {len(df_clean)} samples")
            return False
        
        # Train the model
        print("🎯 Training Transformer model...")
        result = transformer_predictor.train_model(df_clean)
        
        if result and result.get('success', False):
            print("✅ Transformer training completed successfully")
            return True
        else:
            print(f"❌ Transformer training failed: {result.get('error', 'Unknown error') if result else 'No result'}")
            return False
            
    except Exception as e:
        print(f"❌ Transformer training error: {e}")
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
        message="TradPal System started",
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
    parser.add_argument('--mode', choices=['live', 'backtest', 'analysis', 'discovery', 'paper', 'multi-model'],
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
    parser.add_argument('--models', nargs='+', 
                       choices=['traditional_ml', 'lstm', 'transformer', 'ensemble'],
                       help='ML models to test in multi-model backtest (default: all available)')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Maximum parallel workers for multi-model backtest (default: 4)')
    parser.add_argument('--train-missing', action='store_true',
                       help='Automatically train missing ML models before backtesting')
    parser.add_argument('--retrain-all', action='store_true',
                       help='Retrain all ML models before backtesting (forces complete retraining)')

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
    elif args.mode == 'paper':
        run_paper_trading_mode(args, performance_monitor)
    elif args.mode == 'multi-model':
        run_multi_model_backtest_mode(args, performance_monitor)
    else:
        print(f"Unknown mode: {args.mode}")

    # Log system shutdown
    audit_logger.log_system_event(
        event_type="SYSTEM_SHUTDOWN",
        message="TradPal System shutdown",
        details={
            "mode": args.mode,
            "shutdown_reason": "normal_exit"
        }
    )
    audit_logger.log_performance_metrics()
