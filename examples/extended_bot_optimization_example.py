#!/usr/bin/env python3
"""
Example: Extended Bot Configuration Optimization

This example demonstrates how to use the extended Discovery Service
to optimize complete bot configurations including both technical indicators
and trading parameters using genetic algorithms.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.monitoring_service.discovery_service.service import DiscoveryService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def example_bot_optimization():
    """
    Example of running extended bot configuration optimization.

    This optimizes both indicator parameters AND bot trading configurations
    to find the best overall trading strategy.
    """
    try:
        # Initialize the discovery service
        service = DiscoveryService()

        # Define optimization parameters
        symbol = "BTC/USDT"
        timeframe = "1d"
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")  # 2 years

        logger.info("🚀 Starting Extended Bot Configuration Optimization")
        logger.info(f"📊 Symbol: {symbol}, Timeframe: {timeframe}")
        logger.info(f"📅 Period: {start_date} to {end_date}")
        logger.info("")

        # Run the optimization
        result = await service.run_bot_optimization_async(
            optimization_id=f"example_bot_opt_{int(datetime.now().timestamp())}",
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            population_size=100,    # Population size for GA
            generations=50,         # Number of generations
            use_walk_forward=True   # Use walk-forward analysis
        )

        if result['success']:
            logger.info("✅ Optimization completed successfully!")
            logger.info("")
            logger.info("📈 RESULTS SUMMARY:")
            logger.info(f"   Best Fitness Score: {result['best_fitness']:.4f}")
            logger.info(f"   Total Evaluations: {result['total_evaluations']}")
            logger.info(".2f")
            logger.info("")

            # Display best configuration
            best_config = result['best_config']
            bot_config = best_config.get('bot_config', {})

            logger.info("🏆 BEST CONFIGURATION FOUND:")
            logger.info(f"   Bot Strategy: {best_config.get('bot_config_name', 'Unknown')}")
            logger.info(f"   Risk per Trade: {bot_config.get('risk_per_trade', 'N/A')}")
            logger.info(f"   Max Open Trades: {bot_config.get('max_open_trades', 'N/A')}")
            logger.info(f"   Stop Loss: {bot_config.get('stop_loss_pct', 'N/A')}")
            logger.info(f"   Take Profit: {bot_config.get('take_profit_pct', 'N/A')}")
            logger.info(f"   Confidence Threshold: {bot_config.get('confidence_threshold', 'N/A')}")
            logger.info(f"   Regime Adaptation: {bot_config.get('regime_adaptation', 'N/A')}")
            logger.info("")

            # Show indicator configuration
            logger.info("📊 TECHNICAL INDICATORS:")
            indicators = best_config.get('indicators', {})
            for indicator, params in indicators.items():
                logger.info(f"   {indicator}: {params}")
            logger.info("")

            # Show top 5 configurations
            top_configs = result.get('top_configurations', [])
            logger.info("🥇 TOP 5 CONFIGURATIONS:")
            for i, config in enumerate(top_configs[:5], 1):
                fitness = config.get('fitness', 0)
                bot_name = config.get('config', {}).get('bot_config_name', 'Unknown')
                pnl = config.get('pnl', 0)
                win_rate = config.get('win_rate', 0)
                logger.info(f"   {i}. {bot_name} - Fitness: {fitness:.4f}, "
                          f"P&L: {pnl:.2f}, Win Rate: {win_rate:.1%}")

            logger.info("")
            logger.info("💡 NEXT STEPS:")
            logger.info("   1. Backtest the best configuration with paper trading")
            logger.info("   2. Fine-tune parameters based on recent market conditions")
            logger.info("   3. Consider walk-forward analysis for robustness")
            logger.info("   4. Test with different market regimes")

        else:
            logger.error(f"❌ Optimization failed: {result.get('error', 'Unknown error')}")

    except Exception as e:
        logger.error(f"❌ Example failed: {e}")
        raise

async def example_compare_bot_configs():
    """Example showing how to compare different bot configuration presets."""
    try:
        service = DiscoveryService()

        logger.info("🤖 AVAILABLE BOT CONFIGURATIONS:")
        logger.info("")

        for name, config in service.BOT_CONFIGURATIONS.items():
            logger.info(f"📋 {name.upper()} BOT:")
            logger.info(f"   Risk Profile: {config.get('description', 'N/A')}")
            logger.info(f"   Risk per Trade: {config.get('risk_per_trade', 'N/A')}")
            logger.info(f"   Max Open Trades: {config.get('max_open_trades', 'N/A')}")
            logger.info(f"   Stop Loss: {config.get('stop_loss_pct', 'N/A')}")
            logger.info(f"   Take Profit: {config.get('take_profit_pct', 'N/A')}")
            logger.info(f"   Confidence Threshold: {config.get('confidence_threshold', 'N/A')}")
            logger.info(f"   Max Daily Loss: {config.get('max_daily_loss', 'N/A')}")
            logger.info("")

    except Exception as e:
        logger.error(f"Configuration comparison failed: {e}")
        raise

async def main():
    """Main example function."""
    logger.info("🎯 TradPal Extended Bot Optimization Example")
    logger.info("=" * 50)
    logger.info("")

    try:
        # Show available configurations
        await example_compare_bot_configs()

        # Run optimization example
        await example_bot_optimization()

        logger.info("")
        logger.info("🎉 Example completed successfully!")
        logger.info("Check the logs and results for detailed optimization insights.")

    except Exception as e:
        logger.error(f"Example failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())