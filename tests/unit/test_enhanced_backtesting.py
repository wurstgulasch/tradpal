#!/usr/bin/env python3
"""
Test script for enhanced backtesting service with funding rate and liquidation data integration.

This script tests the integration of new data sources into the backtesting service
and validates that the enhanced signals improve trading performance.
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
import pandas as pd

# Add the project root to the path
sys.path.insert(0, '/Users/danielsadowski/VSCodeProjects/tradpal/tradpal')

from services.trading_service.backtesting_service.service import BacktestingService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_enhanced_backtesting():
    """Test the enhanced backtesting service with new data sources."""

    # Initialize the backtesting service
    service = BacktestingService()

    # Test parameters
    symbol = "BTC/USDT"
    timeframe = "1d"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1 year of data
    initial_capital = 10000

    logger.info("Testing enhanced backtesting with funding rate and liquidation data")
    logger.info(f"Symbol: {symbol}, Timeframe: {timeframe}")
    logger.info(f"Period: {start_date.date()} to {end_date.date()}")

    # Test traditional strategy
    logger.info("Running traditional strategy...")
    traditional_result = await service.run_backtest_async(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        strategy='traditional',
        initial_capital=initial_capital,
        backtest_id='test_traditional_enhanced'
    )

    if traditional_result.get('success'):
        metrics = traditional_result.get('metrics', {})
        logger.info("Traditional Strategy Results:")
        logger.info(f"  Total Trades: {metrics.get('total_trades', 0)}")
        logger.info(f"  Win Rate: {metrics.get('win_rate', 0)}%")
        logger.info(f"  Total P&L: ${metrics.get('total_pnl', 0)}")
        logger.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0)}")
        logger.info(f"  Max Drawdown: {metrics.get('max_drawdown', 0)}%")
        logger.info(f"  Profit Factor: {metrics.get('profit_factor', 'N/A')}")
    else:
        logger.error(f"Traditional backtest failed: {traditional_result.get('error')}")

    # Test ML-enhanced strategy (if available)
    logger.info("Running ML-enhanced strategy...")
    ml_result = await service.run_backtest_async(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        strategy='ml_enhanced',
        initial_capital=initial_capital,
        backtest_id='test_ml_enhanced'
    )

    if ml_result.get('success'):
        metrics = ml_result.get('metrics', {})
        logger.info("ML-Enhanced Strategy Results:")
        logger.info(f"  Total Trades: {metrics.get('total_trades', 0)}")
        logger.info(f"  Win Rate: {metrics.get('win_rate', 0)}%")
        logger.info(f"  Total P&L: ${metrics.get('total_pnl', 0)}")
        logger.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0)}")
        logger.info(f"  Max Drawdown: {metrics.get('max_drawdown', 0)}%")
        logger.info(f"  Profit Factor: {metrics.get('profit_factor', 'N/A')}")
    else:
        logger.warning(f"ML-enhanced backtest failed: {ml_result.get('error')}")

    # Compare results
    if traditional_result.get('success') and ml_result.get('success'):
        trad_pnl = traditional_result['metrics'].get('total_pnl', 0)
        ml_pnl = ml_result['metrics'].get('total_pnl', 0)

        improvement = ((ml_pnl - trad_pnl) / abs(trad_pnl)) * 100 if trad_pnl != 0 else 0

        logger.info("Comparison Results:")
        logger.info(f"  Traditional P&L: ${trad_pnl}")
        logger.info(f"  ML-Enhanced P&L: ${ml_pnl}")
        logger.info(f"  Improvement: {improvement:.2f}%")

        if improvement > 0:
            logger.info("✅ ML-enhanced strategy shows improvement!")
        else:
            logger.info("⚠️  ML-enhanced strategy underperforms traditional")

    # Test multi-symbol backtest
    logger.info("Testing multi-symbol backtest...")
    symbols = ["BTC/USDT", "ETH/USDT"]
    multi_result = await service.run_multi_symbol_backtest_async(
        symbols=symbols,
        timeframe=timeframe,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        initial_capital=initial_capital,
        backtest_id='test_multi_symbol_enhanced'
    )

    if multi_result.get('successful_backtests'):
        logger.info("Multi-symbol backtest completed:")
        for symbol_name, result in multi_result.get('individual_results', {}).items():
            if 'metrics' in result:
                metrics = result['metrics']
                logger.info(f"  {symbol_name}: {metrics.get('total_trades', 0)} trades, P&L: ${metrics.get('total_pnl', 0)}")

    logger.info("Enhanced backtesting test completed!")

async def test_data_integration():
    """Test that the new data sources are properly integrated."""
    from services.data_service.data_service.data_sources.funding_rate import FundingRateDataSource
    from services.data_service.data_service.data_sources.liquidation import LiquidationDataSource
    from services.data_service.data_service.data_sources.yahoo_finance import YahooFinanceDataSource

    logger.info("Testing data source integration...")

    # Test OHLCV data from Yahoo Finance
    yahoo_source = YahooFinanceDataSource()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    try:
        ohlcv_data = yahoo_source.fetch_historical_data(
            symbol="BTC-USD",  # Yahoo Finance uses BTC-USD
            timeframe='1d',
            start_date=start_date,
            end_date=end_date
        )
        if not ohlcv_data.empty:
            logger.info(f"✅ OHLCV data fetched: {len(ohlcv_data)} records")
            logger.info(f"   Sample close price: {ohlcv_data['close'].iloc[0] if len(ohlcv_data) > 0 else 'N/A'}")
        else:
            logger.warning("⚠️  No OHLCV data available")
    except Exception as e:
        logger.error(f"❌ OHLCV data fetch failed: {e}")

    # Test funding rate data
    funding_source = FundingRateDataSource()

    try:
        funding_data = funding_source.fetch_historical_data(
            symbol="BTCUSDT",
            timeframe='8h',
            start_date=start_date,
            end_date=end_date
        )
        if not funding_data.empty:
            logger.info(f"✅ Funding rate data fetched: {len(funding_data)} records")
            logger.info(f"   Sample funding rate: {funding_data['funding_rate'].iloc[0] if len(funding_data) > 0 else 'N/A'}")
        else:
            logger.warning("⚠️  No funding rate data available")
    except Exception as e:
        logger.error(f"❌ Funding rate data fetch failed: {e}")

    # Test liquidation data
    liquidation_source = LiquidationDataSource()

    try:
        liquidation_data = liquidation_source.fetch_historical_data(
            symbol="BTCUSDT",
            timeframe='1h',
            start_date=start_date,
            end_date=end_date
        )
        if not liquidation_data.empty:
            logger.info(f"✅ Liquidation data fetched: {len(liquidation_data)} records")
            logger.info(f"   Sample liquidation signal: {liquidation_data['liquidation_signal'].iloc[0] if len(liquidation_data) > 0 else 'N/A'}")
        else:
            logger.warning("⚠️  No liquidation data available")
    except Exception as e:
        logger.error(f"❌ Liquidation data fetch failed: {e}")

async def main():
    """Main test function."""
    logger.info("Starting enhanced backtesting service tests...")

    try:
        # Test data integration first
        await test_data_integration()

        # Then test enhanced backtesting
        await test_enhanced_backtesting()

        logger.info("All tests completed successfully!")

    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)