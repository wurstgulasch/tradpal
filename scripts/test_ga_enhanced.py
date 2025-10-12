#!/usr/bin/env python3
"""
Test script for GA optimization with new technical indicators.
Tests the enhanced discovery system with MACD, OBV, Stochastic, and Chaikin Money Flow.
"""

import sys
import os
import logging
from datetime import datetime, timedelta

# Add src to path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from discovery import DiscoveryOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_ga_optimization():
    """Test GA optimization with new indicators."""
    print("Testing GA optimization with enhanced technical indicators...")

    # Use recent data for faster testing
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)  # 3 months of data

    try:
        optimizer = DiscoveryOptimizer(
            symbol='BTC/USDT',
            exchange='binance',
            timeframe='1d',
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            population_size=20,  # Smaller for testing
            generations=5      # Fewer generations for testing
        )

        print("Running GA optimization...")
        results = optimizer.optimize()

        print(f"\nOptimization completed! Found {len(results)} top configurations.")

        if results:
            best_result = results[0]
            print("\nBest Configuration:")
            print(f"  Fitness: {best_result.fitness:.2f}")
            print(f"  Total P&L: {best_result.pnl:.2f}%")
            print(f"  Win Rate: {best_result.win_rate:.2%}")
            print(f"  Sharpe Ratio: {best_result.sharpe_ratio:.2f}")
            print(f"  Max Drawdown: {best_result.max_drawdown:.2%}")
            print(f"  Total Trades: {best_result.total_trades}")

            print("\nConfiguration Details:")
            config = best_result.config
            print(f"  EMA: {config.get('ema', {})}")
            print(f"  RSI: {config.get('rsi', {})}")
            print(f"  Bollinger Bands: {config.get('bb', {})}")
            print(f"  MACD: {config.get('macd', {})}")
            print(f"  Stochastic: {config.get('stochastic', {})}")
            print(f"  OBV: {config.get('obv', {})}")
            print(f"  Chaikin Money Flow: {config.get('cmf', {})}")

        # Save results
        optimizer.save_results(results, 'output/test_discovery_results.json')
        print("\nResults saved to output/test_discovery_results.json")

        return True

    except Exception as e:
        print(f"Error during GA optimization: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ga_optimization()
    sys.exit(0 if success else 1)