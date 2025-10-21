#!/usr/bin/env python3
"""
Test script for extended bot configuration optimization.
Tests the new genetic algorithm optimization that optimizes both indicators and bot configurations.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.discovery_service.service import DiscoveryService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_bot_optimization():
    """Test the extended bot configuration optimization."""
    try:
        # Initialize service
        service = DiscoveryService()

        # Test parameters
        symbol = "BTC/USDT"
        timeframe = "1d"
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        logger.info(f"Starting bot optimization for {symbol} {timeframe}")
        logger.info(f"Date range: {start_date} to {end_date}")

        # Run optimization
        result = await service.run_bot_optimization_async(
            optimization_id=f"test_bot_opt_{int(datetime.now().timestamp())}",
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            population_size=50,  # Smaller for testing
            generations=10,      # Fewer generations for testing
            use_walk_forward=True
        )

        if result['success']:
            logger.info("Bot optimization completed successfully!")
            logger.info(f"Best fitness: {result['best_fitness']:.4f}")
            logger.info(f"Total evaluations: {result['total_evaluations']}")
            logger.info(".2f")

            # Show best configuration
            best_config = result['best_config']
            logger.info("Best configuration:")
            logger.info(f"  Bot config: {best_config.get('bot_config_name', 'Unknown')}")
            logger.info(f"  Risk multiplier: {best_config.get('bot_config', {}).get('risk_per_trade', 'N/A')}")
            logger.info(f"  Confidence threshold: {best_config.get('bot_config', {}).get('confidence_threshold', 'N/A')}")

            # Show top 3 configurations
            top_configs = result.get('top_configurations', [])
            logger.info("Top 3 configurations:")
            for i, config in enumerate(top_configs[:3], 1):
                logger.info(f"  {i}. Fitness: {config.get('fitness', 0):.4f}, "
                          f"Bot: {config.get('config', {}).get('bot_config_name', 'Unknown')}")

        else:
            logger.error(f"Bot optimization failed: {result.get('error', 'Unknown error')}")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise

async def test_bot_configurations():
    """Test different bot configuration presets."""
    try:
        service = DiscoveryService()

        logger.info("Available bot configurations:")
        for name, config in service.BOT_CONFIGURATIONS.items():
            logger.info(f"  {name}:")
            logger.info(f"    Risk per trade: {config.get('risk_per_trade', 'N/A')}")
            logger.info(f"    Max open trades: {config.get('max_open_trades', 'N/A')}")
            logger.info(f"    Stop loss: {config.get('stop_loss_pct', 'N/A')}")
            logger.info(f"    Take profit: {config.get('take_profit_pct', 'N/A')}")
            logger.info(f"    Confidence threshold: {config.get('confidence_threshold', 'N/A')}")
            logger.info("")

    except Exception as e:
        logger.error(f"Configuration test failed: {e}")
        raise

async def main():
    """Main test function."""
    logger.info("Starting extended bot optimization tests...")

    try:
        # Test configurations
        await test_bot_configurations()

        # Test optimization
        await test_bot_optimization()

        logger.info("All tests completed successfully!")

    except Exception as e:
        logger.error(f"Tests failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())