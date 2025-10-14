#!/usr/bin/env python3
"""
Extended Discovery Test Script

Tests the Discovery optimization system with extended data periods (1-2 weeks)
to validate the enhanced caching system and ensure robust performance.
"""

import sys
import os
import argparse
import logging
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from discovery import run_discovery, clear_discovery_cache, get_discovery_cache_info
    from config.settings import SYMBOL, TIMEFRAME
    import logging_config
    logger = logging_config.logger
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

def test_extended_discovery(symbol: str = SYMBOL,
                          timeframe: str = TIMEFRAME,
                          weeks: int = 2,
                          population_size: int = 50,
                          generations: int = 10):
    """
    Test Discovery with extended data periods.

    Args:
        symbol: Trading symbol
        timeframe: Timeframe for analysis
        weeks: Number of weeks of data to use
        population_size: GA population size
        generations: Number of GA generations
    """
    logger.info("=" * 60)
    logger.info("EXTENDED DISCOVERY TEST")
    logger.info("=" * 60)
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Timeframe: {timeframe}")
    logger.info(f"Data period: {weeks} weeks")
    logger.info(f"GA Population: {population_size}")
    logger.info(f"GA Generations: {generations}")
    logger.info("=" * 60)

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(weeks=weeks)

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    logger.info(f"Date range: {start_str} to {end_str}")

    # Clear cache before starting
    logger.info("Clearing discovery cache...")
    clear_discovery_cache()

    # Show initial cache status
    cache_info = get_discovery_cache_info()
    logger.info(f"Initial cache status: {cache_info}")

    try:
        # Run discovery optimization
        logger.info("Starting extended discovery optimization...")
        start_time = datetime.now()

        results = run_discovery(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_str,
            end_date=end_str,
            population_size=population_size,
            generations=generations,
            use_walk_forward=True
        )

        end_time = datetime.now()
        duration = end_time - start_time

        logger.info("=" * 60)
        logger.info("DISCOVERY COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Duration: {duration}")
        logger.info(f"Total configurations tested: {len(results) if results else 0}")

        if results:
            best_result = results[0]
            logger.info("TOP CONFIGURATION:")
            logger.info(f"  Fitness: {best_result.fitness:.2f}")
            logger.info(f"  PnL: {best_result.pnl:.2f}%")
            logger.info(f"  Win Rate: {best_result.win_rate:.1f}%")
            logger.info(f"  Sharpe Ratio: {best_result.sharpe_ratio:.2f}")
            logger.info(f"  Max Drawdown: {best_result.max_drawdown:.2f}%")
            logger.info(f"  Total Trades: {best_result.total_trades}")
            logger.info(f"  Combination: {best_result.config.get('combination_name', 'Unknown')}")

            # Show all top 5 results
            logger.info("\nTOP 5 CONFIGURATIONS:")
            for i, result in enumerate(results[:5], 1):
                logger.info(f"{i}. Fitness: {result.fitness:.2f}, "
                          f"PnL: {result.pnl:.2f}%, "
                          f"Trades: {result.total_trades}, "
                          f"Combination: {result.config.get('combination_name', 'Unknown')}")

        # Show final cache status
        final_cache_info = get_discovery_cache_info()
        logger.info(f"\nFinal cache status: {final_cache_info}")

        logger.info("\n✅ Extended Discovery test completed successfully!")

        return results

    except Exception as e:
        logger.error(f"❌ Extended Discovery test failed: {e}")
        raise

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Extended Discovery Test')
    parser.add_argument('--symbol', default=SYMBOL, help=f'Trading symbol (default: {SYMBOL})')
    parser.add_argument('--timeframe', default=TIMEFRAME, help=f'Timeframe (default: {TIMEFRAME})')
    parser.add_argument('--weeks', type=int, default=2, help='Number of weeks of data (default: 2)')
    parser.add_argument('--population', type=int, default=50, help='GA population size (default: 50)')
    parser.add_argument('--generations', type=int, default=10, help='GA generations (default: 10)')
    parser.add_argument('--clear-cache', action='store_true', help='Clear cache before starting')

    args = parser.parse_args()

    if args.clear_cache:
        logger.info("Clearing discovery cache as requested...")
        clear_discovery_cache()

    try:
        results = test_extended_discovery(
            symbol=args.symbol,
            timeframe=args.timeframe,
            weeks=args.weeks,
            population_size=args.population,
            generations=args.generations
        )

        # Exit with success
        sys.exit(0)

    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()