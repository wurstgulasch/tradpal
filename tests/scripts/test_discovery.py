#!/usr/bin/env python3
"""
Test script for the Discovery optimization module.

This script runs a quick test of the genetic algorithm optimization
to ensure the discovery module works correctly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from discovery import run_discovery
from logging_config import logger

def test_discovery():
    """Run a quick discovery test with small parameters."""
    print("🧬 Testing Discovery Module")
    print("Running GA optimization with small parameters for testing...")

    try:
        # Run discovery with small parameters for testing
        results = run_discovery(
            symbol='EUR/USD',
            exchange='kraken',
            timeframe='1h',
            start_date='2024-01-01',
            end_date='2024-03-01',  # Short period for testing
            population_size=10,  # Small population
            generations=3       # Few generations
        )

        print(f"\n✅ Discovery test completed successfully!")
        print(f"Found {len(results)} optimized configurations")

        if results:
            best = results[0]
            print(f"\n🏆 Best Configuration:")
            print(f"   Fitness: {best.fitness:.2f}")
            print(f"   P&L: {best.pnl:.2f}%")
            print(f"   Win Rate: {best.win_rate:.1%}")
            print(f"   Total Trades: {best.total_trades}")

        print("\n📁 Results saved to output/discovery_results.json")

    except Exception as e:
        print(f"❌ Discovery test failed: {e}")
        logger.error(f"Discovery test error: {e}")
        return False

    return True

if __name__ == "__main__":
    success = test_discovery()
    sys.exit(0 if success else 1)