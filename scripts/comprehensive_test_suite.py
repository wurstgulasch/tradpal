#!/usr/bin/env python3
"""
Comprehensive Trading Bot Testing Suite
Tests multiple timeframes, assets, strategies, and market conditions
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
from typing import Dict, List, Any
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.trading_service.backtesting_service.service import BacktestingService

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingBotTestSuite:
    """Comprehensive testing suite for trading bot performance"""

    def __init__(self):
        self.backtesting_service = BacktestingService()
        self.test_results = {}

    async def run_comprehensive_tests(self):
        """Run all comprehensive tests"""

        logger.info("🚀 Starting Comprehensive Trading Bot Test Suite...")

        # Test 1: Multi-Timeframe Analysis
        await self.test_multiple_timeframes()

        # Test 2: Multi-Asset Performance
        await self.test_multiple_assets()

        # Test 3: Strategy Comparison
        await self.test_strategy_comparison()

        # Test 4: Market Condition Robustness
        await self.test_market_conditions()

        # Test 5: Risk Analysis
        await self.test_risk_analysis()

        # Generate comprehensive report
        self.generate_comprehensive_report()

    async def test_multiple_timeframes(self):
        """Test performance across different timeframes"""

        logger.info("📊 Testing Multiple Timeframes...")

        symbol = "BTC/USDT"
        timeframes = ["1h", "4h", "1d", "1w"]
        base_date = "2024-01-01"

        timeframe_results = {}

        for tf in timeframes:
            logger.info(f"Testing {symbol} on {tf} timeframe...")

            # Adjust end date based on timeframe
            if tf == "1h":
                end_date = "2024-02-01"  # Shorter period for hourly data
            elif tf == "4h":
                end_date = "2024-03-01"
            elif tf == "1d":
                end_date = "2024-06-01"
            else:  # 1w
                end_date = "2024-12-01"

            try:
                result = await self.backtesting_service.run_backtest_async(
                    symbol=symbol,
                    timeframe=tf,
                    start_date=base_date,
                    end_date=end_date,
                    strategy='ml_enhanced',
                    initial_capital=10000.0
                )

                if result and result.get('success'):
                    metrics = result.get('metrics', {})
                    timeframe_results[tf] = {
                        'return_pct': metrics.get('return_pct', 0),
                        'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                        'win_rate': metrics.get('win_rate', 0),
                        'max_drawdown': metrics.get('max_drawdown', 0),
                        'total_trades': metrics.get('total_trades', 0)
                    }
                    logger.info(f"✅ {tf}: {metrics.get('return_pct', 0):.2f}% return, Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
                else:
                    logger.warning(f"❌ {tf} test failed")
                    timeframe_results[tf] = {'error': 'Test failed'}

            except Exception as e:
                logger.error(f"❌ {tf} test error: {e}")
                timeframe_results[tf] = {'error': str(e)}

        self.test_results['timeframes'] = timeframe_results

    async def test_multiple_assets(self):
        """Test performance across different assets"""

        logger.info("📊 Testing Multiple Assets...")

        assets = [
            ("BTC/USDT", "Bitcoin"),
            ("ETH/USDT", "Ethereum"),
            ("ADA/USDT", "Cardano"),
            ("SOL/USDT", "Solana"),
            ("DOT/USDT", "Polkadot")
        ]

        asset_results = {}

        for symbol, name in assets:
            logger.info(f"Testing {name} ({symbol})...")

            try:
                result = await self.backtesting_service.run_backtest_async(
                    symbol=symbol,
                    timeframe="1d",
                    start_date="2024-01-01",
                    end_date="2024-06-01",
                    strategy='ml_enhanced',
                    initial_capital=10000.0
                )

                if result and result.get('success'):
                    metrics = result.get('metrics', {})
                    asset_results[symbol] = {
                        'name': name,
                        'return_pct': metrics.get('return_pct', 0),
                        'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                        'win_rate': metrics.get('win_rate', 0),
                        'max_drawdown': metrics.get('max_drawdown', 0),
                        'total_trades': metrics.get('total_trades', 0)
                    }
                    logger.info(f"✅ {name}: {metrics.get('return_pct', 0):.2f}% return")
                else:
                    logger.warning(f"❌ {name} test failed")
                    asset_results[symbol] = {'name': name, 'error': 'Test failed'}

            except Exception as e:
                logger.error(f"❌ {name} test error: {e}")
                asset_results[symbol] = {'name': name, 'error': str(e)}

        self.test_results['assets'] = asset_results

    async def test_strategy_comparison(self):
        """Compare different trading strategies"""

        logger.info("📊 Comparing Trading Strategies...")

        symbol = "BTC/USDT"
        strategies = ['traditional', 'ml_enhanced']
        strategy_results = {}

        for strategy in strategies:
            logger.info(f"Testing {strategy} strategy...")

            try:
                result = await self.backtesting_service.run_backtest_async(
                    symbol=symbol,
                    timeframe="1d",
                    start_date="2024-01-01",
                    end_date="2024-06-01",
                    strategy=strategy,
                    initial_capital=10000.0
                )

                if result and result.get('success'):
                    metrics = result.get('metrics', {})
                    strategy_results[strategy] = {
                        'return_pct': metrics.get('return_pct', 0),
                        'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                        'win_rate': metrics.get('win_rate', 0),
                        'max_drawdown': metrics.get('max_drawdown', 0),
                        'total_trades': metrics.get('total_trades', 0),
                        'profit_factor': metrics.get('profit_factor', 0)
                    }
                    logger.info(f"✅ {strategy}: {metrics.get('return_pct', 0):.2f}% return, Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
                else:
                    logger.warning(f"❌ {strategy} test failed")
                    strategy_results[strategy] = {'error': 'Test failed'}

            except Exception as e:
                logger.error(f"❌ {strategy} test error: {e}")
                strategy_results[strategy] = {'error': str(e)}

        self.test_results['strategies'] = strategy_results

    async def test_market_conditions(self):
        """Test performance under different market conditions"""

        logger.info("📊 Testing Market Condition Robustness...")

        # Define different market periods
        market_periods = [
            ("2024-01-01", "2024-03-01", "Q1 2024 - Bull Market"),
            ("2024-03-01", "2024-05-01", "Q2 2024 - Volatile"),
            ("2024-05-01", "2024-07-01", "Q3 2024 - Recovery"),
            ("2024-07-01", "2024-09-01", "Q4 2024 - Consolidation")
        ]

        market_results = {}

        for start_date, end_date, description in market_periods:
            logger.info(f"Testing {description}...")

            try:
                result = await self.backtesting_service.run_backtest_async(
                    symbol="BTC/USDT",
                    timeframe="1d",
                    start_date=start_date,
                    end_date=end_date,
                    strategy='ml_enhanced',
                    initial_capital=10000.0
                )

                if result and result.get('success'):
                    metrics = result.get('metrics', {})
                    market_results[description] = {
                        'period': f"{start_date} to {end_date}",
                        'return_pct': metrics.get('return_pct', 0),
                        'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                        'win_rate': metrics.get('win_rate', 0),
                        'max_drawdown': metrics.get('max_drawdown', 0),
                        'total_trades': metrics.get('total_trades', 0)
                    }
                    logger.info(f"✅ {description}: {metrics.get('return_pct', 0):.2f}% return")
                else:
                    logger.warning(f"❌ {description} test failed")
                    market_results[description] = {'period': f"{start_date} to {end_date}", 'error': 'Test failed'}

            except Exception as e:
                logger.error(f"❌ {description} test error: {e}")
                market_results[description] = {'period': f"{start_date} to {end_date}", 'error': str(e)}

        self.test_results['market_conditions'] = market_results

    async def test_risk_analysis(self):
        """Analyze risk metrics and drawdown patterns"""

        logger.info("📊 Performing Risk Analysis...")

        symbol = "BTC/USDT"

        try:
            result = await self.backtesting_service.run_backtest_async(
                symbol=symbol,
                timeframe="1d",
                start_date="2024-01-01",
                end_date="2024-06-01",
                strategy='ml_enhanced',
                initial_capital=10000.0
            )

            if result and result.get('success'):
                metrics = result.get('metrics', {})
                trades = result.get('trades', [])

                # Calculate additional risk metrics
                risk_analysis = {
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'max_drawdown': metrics.get('max_drawdown', 0),
                    'win_rate': metrics.get('win_rate', 0),
                    'profit_factor': metrics.get('profit_factor', 0),
                    'total_trades': metrics.get('total_trades', 0),
                    'avg_win': metrics.get('avg_win', 0),
                    'avg_loss': metrics.get('avg_loss', 0),
                    'largest_win': max([t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0], default=0),
                    'largest_loss': min([t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0], default=0),
                    'consecutive_wins': self._calculate_consecutive_stats(trades, 'win'),
                    'consecutive_losses': self._calculate_consecutive_stats(trades, 'loss'),
                    'monthly_returns': self._calculate_monthly_returns(trades)
                }

                self.test_results['risk_analysis'] = risk_analysis
                logger.info("✅ Risk analysis completed")
                logger.info(f"📊 Max Drawdown: {risk_analysis['max_drawdown']:.2f}%")
                logger.info(f"📊 Sharpe Ratio: {risk_analysis['sharpe_ratio']:.2f}")
                logger.info(f"📊 Profit Factor: {risk_analysis['profit_factor']:.2f}")

            else:
                logger.warning("❌ Risk analysis failed")
                self.test_results['risk_analysis'] = {'error': 'Test failed'}

        except Exception as e:
            logger.error(f"❌ Risk analysis error: {e}")
            self.test_results['risk_analysis'] = {'error': str(e)}

    def _calculate_consecutive_stats(self, trades: List[Dict], stat_type: str) -> Dict[str, Any]:
        """Calculate consecutive win/loss statistics"""

        if not trades:
            return {'max': 0, 'avg': 0}

        consecutive = []
        current = 0

        for trade in trades:
            pnl = trade.get('pnl', 0)
            if stat_type == 'win' and pnl > 0:
                current += 1
            elif stat_type == 'loss' and pnl < 0:
                current += 1
            else:
                if current > 0:
                    consecutive.append(current)
                current = 0

        if current > 0:
            consecutive.append(current)

        return {
            'max': max(consecutive) if consecutive else 0,
            'avg': sum(consecutive) / len(consecutive) if consecutive else 0,
            'total_sequences': len(consecutive)
        }

    def _calculate_monthly_returns(self, trades: List[Dict]) -> List[Dict[str, Any]]:
        """Calculate monthly return statistics"""

        if not trades:
            return []

        # Group trades by month
        monthly_trades = {}
        for trade in trades:
            exit_time = trade.get('exit_time')
            if exit_time:
                if isinstance(exit_time, str):
                    month = exit_time[:7]  # YYYY-MM
                else:
                    month = exit_time.strftime('%Y-%m')

                if month not in monthly_trades:
                    monthly_trades[month] = []
                monthly_trades[month].append(trade)

        monthly_returns = []
        for month, month_trades in monthly_trades.items():
            total_pnl = sum(t.get('pnl', 0) for t in month_trades)
            monthly_returns.append({
                'month': month,
                'total_pnl': total_pnl,
                'num_trades': len(month_trades),
                'avg_pnl_per_trade': total_pnl / len(month_trades) if month_trades else 0
            })

        return monthly_returns

    def generate_comprehensive_report(self):
        """Generate comprehensive test report"""

        logger.info("📊 Generating Comprehensive Test Report...")

        report = {
            'test_suite': 'Trading Bot Comprehensive Test Suite',
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'detailed_results': self.test_results
        }

        # Calculate summary statistics
        summary = {
            'total_tests_run': 0,
            'successful_tests': 0,
            'failed_tests': 0,
            'average_return': 0,
            'best_performing_asset': None,
            'best_performing_timeframe': None,
            'best_strategy': None,
            'overall_assessment': 'pending'
        }

        # Analyze timeframe results
        if 'timeframes' in self.test_results:
            tf_returns = [r.get('return_pct', 0) for r in self.test_results['timeframes'].values() if 'return_pct' in r]
            if tf_returns:
                summary['best_performing_timeframe'] = max(self.test_results['timeframes'].items(), key=lambda x: x[1].get('return_pct', 0) if 'return_pct' in x[1] else -999)[0]

        # Analyze asset results
        if 'assets' in self.test_results:
            asset_returns = [(k, v.get('return_pct', 0)) for k, v in self.test_results['assets'].items() if 'return_pct' in v]
            if asset_returns:
                summary['best_performing_asset'] = max(asset_returns, key=lambda x: x[1])[0]

        # Analyze strategy results
        if 'strategies' in self.test_results:
            strategy_returns = [(k, v.get('return_pct', 0)) for k, v in self.test_results['strategies'].items() if 'return_pct' in v]
            if strategy_returns:
                summary['best_strategy'] = max(strategy_returns, key=lambda x: x[1])[0]

        # Count successful tests
        for test_category, results in self.test_results.items():
            for test_name, result in results.items():
                summary['total_tests_run'] += 1
                # Check if result is a dict and contains 'error' key
                if isinstance(result, dict) and 'error' not in result:
                    summary['successful_tests'] += 1
                else:
                    summary['failed_tests'] += 1

        # Overall assessment
        success_rate = summary['successful_tests'] / summary['total_tests_run'] if summary['total_tests_run'] > 0 else 0

        if success_rate >= 0.9:
            summary['overall_assessment'] = 'excellent'
        elif success_rate >= 0.75:
            summary['overall_assessment'] = 'good'
        elif success_rate >= 0.6:
            summary['overall_assessment'] = 'acceptable'
        else:
            summary['overall_assessment'] = 'needs_improvement'

        report['summary'] = summary

        # Save report
        report_file = f"output/comprehensive_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs('output', exist_ok=True)

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"✅ Comprehensive test report saved to {report_file}")

        # Print summary to console
        self._print_summary_report(summary)

    def _print_summary_report(self, summary: Dict[str, Any]):
        """Print summary report to console"""

        logger.info("\n" + "="*60)
        logger.info("🎯 COMPREHENSIVE TEST SUITE SUMMARY")
        logger.info("="*60)

        logger.info(f"📊 Total Tests Run: {summary['total_tests_run']}")
        logger.info(f"✅ Successful Tests: {summary['successful_tests']}")
        logger.info(f"❌ Failed Tests: {summary['failed_tests']}")

        if summary['best_performing_asset']:
            logger.info(f"🏆 Best Performing Asset: {summary['best_performing_asset']}")

        if summary['best_performing_timeframe']:
            logger.info(f"⏰ Best Performing Timeframe: {summary['best_performing_timeframe']}")

        if summary['best_strategy']:
            logger.info(f"🎯 Best Strategy: {summary['best_strategy']}")

        assessment = summary['overall_assessment'].upper()
        if assessment == 'EXCELLENT':
            logger.info(f"🎉 Overall Assessment: {assessment} - Ready for Live Trading!")
        elif assessment == 'GOOD':
            logger.info(f"👍 Overall Assessment: {assessment} - Good performance, minor improvements possible")
        elif assessment == 'ACCEPTABLE':
            logger.info(f"⚠️  Overall Assessment: {assessment} - Acceptable but needs monitoring")
        else:
            logger.info(f"🔧 Overall Assessment: {assessment} - Further testing and improvements needed")

        logger.info("="*60)

async def main():
    """Main test execution"""

    # Create and run test suite
    test_suite = TradingBotTestSuite()
    await test_suite.run_comprehensive_tests()

    logger.info("🎯 Comprehensive testing completed!")

if __name__ == "__main__":
    asyncio.run(main())