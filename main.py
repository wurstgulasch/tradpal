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

from src.data_fetcher import fetch_data, fetch_historical_data
from src.indicators import calculate_indicators
from src.signal_generator import generate_signals, calculate_risk_management
from src.output import save_signals_to_json, get_latest_signals
from src.logging_config import logger, log_signal, log_error, log_system_status
from src.backtester import run_backtest

def run_live_monitoring():
    """Run continuous live monitoring mode."""
    print("Starting TradPal Indicator - Continuous Monitoring Mode...")
    print("Press Ctrl+C to stop monitoring\n")

    log_system_status("Live monitoring mode started")

    last_signal_time = 0
    signal_cooldown = 60  # Minimum seconds between signals

    while True:
        try:
            # Fetch latest data
            data = fetch_data()

            if data.empty:
                print("No data available, retrying in 30 seconds...")
                time.sleep(30)
                continue

            # Calculate indicators
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

def run_backtest_mode(args):
    """Run backtesting mode."""
    print(f"Running backtest for {args.symbol} on {args.timeframe} timeframe")
    log_system_status(f"Backtest mode started for {args.symbol} {args.timeframe}")

    try:
        results = run_backtest(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date
        )

        print("\n📊 Backtest Results:")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']}%")
        print(f"Total P&L: ${results['total_pnl']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"CAGR: {results['cagr']:.2f}%")
        print(f"Final Capital: ${results['final_capital']:.2f}")

        log_system_status(f"Backtest completed: {results['total_trades']} trades, {results['win_rate']}% win rate")

    except Exception as e:
        print(f"Backtest failed: {e}")
        log_error(f"Backtest error: {e}")

def run_single_analysis():
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

def main():
    parser = argparse.ArgumentParser(description='TradPal Trading Indicator System')
    parser.add_argument('--mode', choices=['live', 'backtest', 'analysis'],
                       default='live', help='Operation mode (default: live)')
    parser.add_argument('--symbol', default='EUR/USD', help='Trading symbol (default: EUR/USD)')
    parser.add_argument('--timeframe', default='1m', help='Timeframe (default: 1m)')
    parser.add_argument('--start-date', help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='Backtest end date (YYYY-MM-DD)')

    args = parser.parse_args()

    if args.mode == 'live':
        run_live_monitoring()
    elif args.mode == 'backtest':
        run_backtest_mode(args)
    elif args.mode == 'analysis':
        run_single_analysis()
    else:
        print(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()
