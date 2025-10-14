#!/usr/bin/env python3
"""
Walk-Forward Analysis for Trading Strategy Validation

This script implements walk-forward analysis to validate trading strategies
found through genetic algorithm optimization. It uses expanding windows
of historical data to train and test strategies, providing out-of-sample
validation to detect overfitting.
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_fetcher import fetch_historical_data
from src.backtester import Backtester
from main import calculate_buy_hold_performance

def run_walk_forward_analysis(configurations: List[Dict[str, Any]],
                             symbol: str = 'BTC/USDT',
                             timeframe: str = '1h',
                             total_period_days: int = 365,
                             train_window_days: int = 90,
                             test_window_days: int = 30,
                             step_days: int = 30) -> Dict[str, Any]:
    """
    Run walk-forward analysis on multiple trading configurations.

    Args:
        configurations: List of configuration dictionaries to test
        symbol: Trading symbol
        timeframe: Timeframe for analysis
        total_period_days: Total historical period in days
        train_window_days: Training window size in days
        test_window_days: Testing window size in days
        step_days: Step size for moving windows in days

    Returns:
        Dictionary with walk-forward analysis results
    """
    print("🚀 Starting Walk-Forward Analysis")
    print(f"📊 Testing {len(configurations)} configurations on {symbol} {timeframe}")
    print(f"📅 Total period: {total_period_days} days")
    print(f"🎯 Train/Test windows: {train_window_days}/{test_window_days} days")
    print(f"👣 Step size: {step_days} days")
    print()

    # Calculate date ranges
    end_date = datetime.now()
    start_date = end_date - timedelta(days=total_period_days)

    # Fetch all historical data once
    print("📥 Fetching historical data...")
    all_data = fetch_historical_data(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        limit=10000  # Large limit to get all data
    )

    if all_data.empty:
        return {"error": "No historical data available"}

    print(f"✅ Fetched {len(all_data)} data points")
    print()

    # Calculate Buy & Hold for the entire period
    bh_performance = calculate_buy_hold_performance(symbol, 'kraken', timeframe)
    print(f"📊 Buy & Hold Performance (full period): {bh_performance:.2f}%")
    print()

    results = {}

    for i, config in enumerate(configurations):
        config_name = f"Config_{i+1}"
        print(f"🔄 Testing {config_name}: {config.get('ema', {}).get('periods', 'N/A')} EMA, RSI({config.get('rsi', {}).get('period', 'N/A')})")

        # Run walk-forward for this configuration
        wf_results = _run_single_walk_forward(
            config=config,
            all_data=all_data,
            train_window_days=train_window_days,
            test_window_days=test_window_days,
            step_days=step_days,
            symbol=symbol,
            timeframe=timeframe
        )

        results[config_name] = {
            'config': config,
            'walk_forward_results': wf_results
        }

        # Print summary
        if 'error' not in wf_results:
            avg_test_return = wf_results.get('average_test_return', 0)
            avg_train_return = wf_results.get('average_train_return', 0)
            test_windows = len(wf_results.get('test_results', []))
            profitable_windows = sum(1 for r in wf_results.get('test_results', []) if r.get('total_pnl', 0) > 0)

            print(f"   ✅ Completed: {test_windows} test windows, {profitable_windows} profitable")
            print(f"   📈 Avg Test Return: {avg_test_return:.2f}%, Train: {avg_train_return:.2f}%")
        else:
            print(f"   ❌ Failed: {wf_results['error']}")

        print()

    # Compare results
    comparison = _compare_walk_forward_results(results, bh_performance)

    return {
        'walk_forward_analysis': results,
        'comparison': comparison,
        'benchmark_buy_hold': bh_performance,
        'analysis_parameters': {
            'symbol': symbol,
            'timeframe': timeframe,
            'total_period_days': total_period_days,
            'train_window_days': train_window_days,
            'test_window_days': test_window_days,
            'step_days': step_days,
            'configurations_tested': len(configurations)
        }
    }

def _run_single_walk_forward(config: Dict[str, Any],
                           all_data: pd.DataFrame,
                           train_window_days: int,
                           test_window_days: int,
                           step_days: int,
                           symbol: str,
                           timeframe: str) -> Dict[str, Any]:
    """
    Run walk-forward analysis for a single configuration.
    """
    try:
        test_results = []
        train_results = []

        # Calculate window boundaries
        data_start = all_data.index[0]
        data_end = all_data.index[-1]

        current_train_end = data_start + timedelta(days=train_window_days)

        while current_train_end + timedelta(days=test_window_days) <= data_end:
            # Define train and test periods
            train_start = current_train_end - timedelta(days=train_window_days)
            test_end = current_train_end + timedelta(days=test_window_days)

            # Extract data for this window
            train_data = all_data[(all_data.index >= train_start) & (all_data.index <= current_train_end)]
            test_data = all_data[(all_data.index > current_train_end) & (all_data.index <= test_end)]

            # Ensure we have enough data for indicators (minimum 200 rows)
            min_data_points = 200
            if len(train_data) < min_data_points or len(test_data) < min_data_points // 4:  # Test needs less data
                continue  # Skip this window, not enough data

            # Test on training data (in-sample)
            train_metrics = _evaluate_config_on_data(config, train_data.copy(), symbol, timeframe)
            train_results.append({
                'train_start': train_start,
                'train_end': current_train_end,
                'metrics': train_metrics
            })

            # Test on test data (out-of-sample)
            test_metrics = _evaluate_config_on_data(config, test_data.copy(), symbol, timeframe)
            test_results.append({
                'test_start': current_train_end,
                'test_end': test_end,
                'metrics': test_metrics
            })

            # Move to next window
            current_train_end += timedelta(days=step_days)

        # Calculate summary statistics
        if test_results:
            test_returns = [r['metrics'].get('total_pnl', 0) / 10000 * 100 for r in test_results]  # Convert to %
            train_returns = [r['metrics'].get('total_pnl', 0) / 10000 * 100 for r in train_results]

            return {
                'test_results': test_results,
                'train_results': train_results,
                'average_test_return': np.mean(test_returns) if test_returns else 0,
                'average_train_return': np.mean(train_returns) if train_returns else 0,
                'test_volatility': np.std(test_returns) if len(test_returns) > 1 else 0,
                'train_volatility': np.std(train_returns) if len(train_returns) > 1 else 0,
                'profitable_test_windows': sum(1 for r in test_returns if r > 0),
                'total_test_windows': len(test_results),
                'test_win_rate': sum(1 for r in test_returns if r > 0) / len(test_results) * 100 if test_results else 0
            }
        else:
            return {"error": "No test windows could be created"}

    except Exception as e:
        return {"error": f"Walk-forward analysis failed: {str(e)}"}

def _evaluate_config_on_data(config: Dict[str, Any],
                           data: pd.DataFrame,
                           symbol: str,
                           timeframe: str) -> Dict[str, Any]:
    """
    Evaluate a configuration on a specific data window.
    """
    try:
        # Ensure data has enough rows for indicators
        if len(data) < 200:  # Minimum required for most indicators
            return {"error": f"Insufficient data: {len(data)} rows, need at least 200"}

        # Create backtester with config - disable ML to avoid feature mismatch issues
        config_no_ml = config.copy()
        if 'ml' not in config_no_ml:
            config_no_ml['ml'] = {}
        config_no_ml['ml']['enabled'] = False  # Disable ML enhancement

        backtester = Backtester(
            symbol=symbol,
            timeframe=timeframe,
            start_date=data.index[0],
            end_date=data.index[-1],
            config=config_no_ml
        )

        # Run backtest
        results = backtester.run_backtest(strategy='traditional')

        if results.get('success'):
            return results.get('metrics', {})
        else:
            error_msg = results.get('error', 'Backtest failed')
            print(f"   ⚠️  Backtest failed: {error_msg}")
            return {"error": error_msg}

    except KeyError as e:
        return {"error": f"Missing column in data: {str(e)}"}
    except Exception as e:
        print(f"   ⚠️  Evaluation error: {str(e)}")
        return {"error": str(e)}

def _compare_walk_forward_results(results: Dict[str, Any], benchmark: float) -> Dict[str, Any]:
    """
    Compare walk-forward results across configurations.
    """
    comparison = {}

    for config_name, config_results in results.items():
        wf_results = config_results.get('walk_forward_results', {})

        if 'error' in wf_results:
            comparison[config_name] = {
                'status': 'failed',
                'error': wf_results['error']
            }
            continue

        avg_test_return = wf_results.get('average_test_return', 0)
        test_volatility = wf_results.get('test_volatility', 0)
        profitable_windows = wf_results.get('profitable_test_windows', 0)
        total_windows = wf_results.get('total_test_windows', 0)

        # Calculate risk-adjusted metrics
        sharpe_ratio = avg_test_return / test_volatility if test_volatility > 0 else 0

        # Compare to benchmark
        vs_benchmark = avg_test_return - benchmark

        comparison[config_name] = {
            'status': 'completed',
            'average_test_return': avg_test_return,
            'test_volatility': test_volatility,
            'sharpe_ratio': sharpe_ratio,
            'profitable_windows': profitable_windows,
            'total_windows': total_windows,
            'win_rate': profitable_windows / total_windows * 100 if total_windows > 0 else 0,
            'vs_benchmark': vs_benchmark,
            'beats_benchmark': vs_benchmark > 0
        }

    # Rank configurations
    valid_results = {k: v for k, v in comparison.items() if v['status'] == 'completed'}
    if valid_results:
        # Sort by average test return
        ranked = sorted(valid_results.items(),
                       key=lambda x: x[1]['average_test_return'],
                       reverse=True)

        for rank, (config_name, _) in enumerate(ranked, 1):
            comparison[config_name]['rank'] = rank

    return comparison

def main():
    """Main function to run walk-forward analysis on discovery results."""
    print("🔬 Walk-Forward Analysis for Discovery Mode Results")
    print("=" * 60)

    # Load discovery results
    try:
        with open('output/discovery_results.json', 'r') as f:
            discovery_data = json.load(f)
    except FileNotFoundError:
        print("❌ Discovery results file not found. Run discovery mode first.")
        return

    # Get top configurations (limit to top 5 for analysis)
    top_configs = discovery_data.get('top_configurations', [])[:5]

    if not top_configs:
        print("❌ No configurations found in discovery results.")
        return

    print(f"📊 Analyzing top {len(top_configs)} configurations from discovery")
    print()

    # Extract configurations
    configurations = [config['configuration'] for config in top_configs]

    # Run walk-forward analysis
    results = run_walk_forward_analysis(
        configurations=configurations,
        symbol='BTC/USDT',
        timeframe='1h',
        total_period_days=365,  # 1 year
        train_window_days=90,   # 3 months training
        test_window_days=30,    # 1 month testing
        step_days=30           # Monthly steps
    )

    if 'error' in results:
        print(f"❌ Walk-forward analysis failed: {results['error']}")
        return

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'output/walk_forward_analysis_{timestamp}.json'

    with open(filename, 'w') as f:
        json.dump(results, f, indent=4, default=str)

    print(f"💾 Results saved to {filename}")
    print()

    # Print summary
    print("📊 Walk-Forward Analysis Summary")
    print("-" * 40)

    comparison = results.get('comparison', {})
    benchmark = results.get('benchmark_buy_hold', 0)

    print(".2f")
    print()

    # Show results for each configuration
    valid_configs = {k: v for k, v in comparison.items() if v['status'] == 'completed'}

    if valid_configs:
        print("🏆 Configuration Results:")
        for config_name, metrics in valid_configs.items():
            rank = metrics.get('rank', 'N/A')
            avg_return = metrics.get('average_test_return', 0)
            win_rate = metrics.get('win_rate', 0)
            vs_benchmark = metrics.get('vs_benchmark', 0)

            status = "🏅" if rank == 1 else "✅" if vs_benchmark > 0 else "❌"
            print("2d"
                  ".2f"
                  ".1f"
                  "+.2f")

        print()
        print("🎯 Recommendations:")
        print("- Configurations with positive out-of-sample returns are validated")
        print("- Higher win rates indicate more consistent performance")
        print("- Lower volatility suggests more stable returns")
        print("- Compare vs benchmark to assess market-beating potential")

    else:
        print("❌ No valid walk-forward results to display")

if __name__ == '__main__':
    main()