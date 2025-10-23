#!/usr/bin/env python3
"""
Direct Trading Bot Test - Test if we can outperform the market
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.trading_service.backtesting_service.service import BacktestingService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_trading_bot_performance():
    """Test if the trading bot can outperform buy & hold"""

    logger.info("🚀 Starting Trading Bot Performance Test...")

    try:
        # Initialize backtesting service
        backtesting_service = BacktestingService()

        # Test parameters
        symbol = "BTC/USDT"
        timeframe = "1d"
        start_date = "2024-01-01"
        end_date = "2024-06-01"
        initial_capital = 10000.0

        logger.info(f"📊 Testing {symbol} from {start_date} to {end_date}")
        logger.info(f"💰 Initial capital: ${initial_capital}")

        # Generate sample market data
        logger.info("📥 Generating sample market data...")
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        np.random.seed(42)  # For reproducible results

        # Create realistic BTC price data with trend and volatility
        base_price = 40000
        trend = 0.001  # Slight upward trend
        volatility = 0.03  # 3% daily volatility

        prices = []
        current_price = base_price
        for i in range(len(dates)):
            # Add trend and random walk
            change = trend + np.random.normal(0, volatility)
            current_price *= (1 + change)
            prices.append(current_price)

        df = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.02))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.02))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(100000, 1000000, len(prices))
        }, index=dates)

        logger.info(f"✅ Generated {len(df)} sample records")

        # Calculate buy & hold performance
        start_price = df['close'].iloc[0]
        end_price = df['close'].iloc[-1]
        buy_hold_return = ((end_price - start_price) / start_price) * 100
        buy_hold_final_value = initial_capital * (1 + buy_hold_return / 100)

        logger.info(f"📈 Buy & Hold: ${initial_capital:.2f} → ${buy_hold_final_value:.2f} ({buy_hold_return:.2f}%)")

        # Run backtest with AI-enhanced strategy
        logger.info("🤖 Running AI-enhanced backtest...")

        backtest_result = await backtesting_service.run_backtest_async(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            strategy='ml_enhanced',
            initial_capital=initial_capital
        )

        if backtest_result and backtest_result.get('success'):
            results = backtest_result.get('metrics', {})

            # Extract key metrics
            final_value = results.get('final_capital', initial_capital)
            total_return = results.get('return_pct', 0)
            sharpe_ratio = results.get('sharpe_ratio', 0)
            max_drawdown = results.get('max_drawdown', 0)
            win_rate = results.get('win_rate', 0)
            total_trades = results.get('total_trades', 0)

            logger.info("🎯 AI Trading Bot Results:")
            logger.info(f"💰 Final Value: ${final_value:.2f}")
            logger.info(f"📈 Total Return: {total_return:.2f}%")
            logger.info(f"⚡ Sharpe Ratio: {sharpe_ratio:.2f}")
            logger.info(f"📉 Max Drawdown: {max_drawdown:.2f}%")
            logger.info(f"🎯 Win Rate: {win_rate:.1f}%")
            logger.info(f"📊 Total Trades: {total_trades}")

            # Compare with buy & hold
            outperformance = total_return - buy_hold_return
            logger.info("\n🔥 PERFORMANCE COMPARISON:")
            logger.info(f"Buy & Hold Return: {buy_hold_return:.2f}%")
            logger.info(f"AI Bot Return: {total_return:.2f}%")
            logger.info(f"Outperformance: {outperformance:.2f}% {'🎉' if outperformance > 0 else '😞'}")

            if outperformance > 0:
                logger.info("🎉 SUCCESS: AI Trading Bot outperformed Buy & Hold!")
            else:
                logger.info("😞 The AI Bot did not outperform Buy & Hold this time")

            # Additional analysis
            if total_trades > 0:
                logger.info("\n📊 Trading Statistics:")
                logger.info(f"Average Win: ${results.get('avg_win', 0):.2f}")
                logger.info(f"Average Loss: ${results.get('avg_loss', 0):.2f}")
                logger.info(f"Profit Factor: {results.get('profit_factor', 0):.2f}")

        else:
            logger.error("❌ Backtest failed or returned no results")
            logger.error(f"Backtest result: {backtest_result}")

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_trading_bot_performance())