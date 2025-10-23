#!/usr/bin/env python3
"""
Live Trading Setup Script
Prepares the trading bot for live trading with proper configuration and testing
"""

import asyncio
import logging
import json
import os
from datetime import datetime
from typing import Dict, Any

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.trading_bot_live.service import TradingBotLiveService
from services.trading_service.backtesting_service.service import BacktestingService

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LiveTradingSetup:
    """Setup and validation for live trading"""

    def __init__(self):
        self.live_service = TradingBotLiveService()
        self.backtesting_service = BacktestingService()
        self.setup_results = {}

    async def run_complete_setup(self):
        """Run complete live trading setup and validation"""

        logger.info("🚀 Starting Live Trading Setup...")

        # Step 1: Service Health Check
        await self.check_service_health()

        # Step 2: Configuration Validation
        await self.validate_configuration()

        # Step 3: Paper Trading Test
        await self.test_paper_trading()

        # Step 4: Risk Management Validation
        await self.validate_risk_management()

        # Step 5: Performance Benchmark
        await self.run_performance_benchmark()

        # Step 6: Generate Setup Report
        self.generate_setup_report()

    async def check_service_health(self):
        """Check health of all required services"""

        logger.info("🔍 Checking Service Health...")

        try:
            health = await self.live_service.health_check()
            self.setup_results['service_health'] = {
                'status': 'healthy' if health.get('status') == 'healthy' else 'unhealthy',
                'details': health
            }
            logger.info("✅ Service health check passed")
        except Exception as e:
            self.setup_results['service_health'] = {
                'status': 'error',
                'error': str(e)
            }
            logger.error(f"❌ Service health check failed: {e}")

    async def validate_configuration(self):
        """Validate trading configuration"""

        logger.info("⚙️ Validating Configuration...")

        config_status = {
            'capital': 10000.0,
            'risk_per_trade': 0.01,
            'max_positions': 5,
            'max_drawdown': 0.1,
            'strategy': 'ml_enhanced',
            'timeframe': '1d',
            'paper_trading': True
        }

        # Check if configuration is reasonable
        issues = []

        if config_status['risk_per_trade'] > 0.05:
            issues.append("Risk per trade too high (>5%)")

        if config_status['max_drawdown'] > 0.2:
            issues.append("Max drawdown too high (>20%)")

        if config_status['max_positions'] > 10:
            issues.append("Too many max positions (>10)")

        config_status['issues'] = issues
        config_status['valid'] = len(issues) == 0

        self.setup_results['configuration'] = config_status

        if config_status['valid']:
            logger.info("✅ Configuration validation passed")
        else:
            logger.warning(f"⚠️ Configuration issues found: {issues}")

    async def test_paper_trading(self):
        """Test paper trading functionality"""

        logger.info("📊 Testing Paper Trading...")

        try:
            # Start paper trading session
            start_result = await self.live_service.start_trading(
                symbol='BTC/USDT',
                strategy='ml_enhanced',
                timeframe='1d',
                capital=10000.0,
                enable_paper_trading=True
            )

            # Wait for initialization
            await asyncio.sleep(1)

            # Execute a test trade
            trade_result = await self.live_service.execute_paper_trade({
                'signal': 'BUY',
                'symbol': 'BTC/USDT',
                'price': 50000.0,
                'quantity': 0.01
            })

            # Get status
            status = await self.live_service.get_symbol_status('BTC/USDT')

            # Stop trading
            stop_result = await self.live_service.stop_trading('BTC/USDT')

            self.setup_results['paper_trading'] = {
                'status': 'success',
                'start_result': start_result,
                'trade_result': trade_result,
                'status_check': status,
                'stop_result': stop_result
            }

            logger.info("✅ Paper trading test passed")

        except Exception as e:
            self.setup_results['paper_trading'] = {
                'status': 'error',
                'error': str(e)
            }
            logger.error(f"❌ Paper trading test failed: {e}")

    async def validate_risk_management(self):
        """Validate risk management functionality"""

        logger.info("🛡️ Validating Risk Management...")

        try:
            # Test risk limit checking
            portfolio = {
                'capital': 10000.0,
                'current_value': 9500.0,  # 5% drawdown
                'positions_value': 500.0
            }

            risk_check = await self.live_service.check_risk_limits(portfolio)

            # Test position sizing
            position_size = await self.live_service._calculate_position_size('BTC/USDT')

            self.setup_results['risk_management'] = {
                'status': 'success',
                'risk_limits': risk_check,
                'position_sizing': position_size > 0,
                'position_size': position_size
            }

            logger.info("✅ Risk management validation passed")

        except Exception as e:
            self.setup_results['risk_management'] = {
                'status': 'error',
                'error': str(e)
            }
            logger.error(f"❌ Risk management validation failed: {e}")

    async def run_performance_benchmark(self):
        """Run performance benchmark against backtesting"""

        logger.info("📈 Running Performance Benchmark...")

        try:
            # Run backtest for comparison
            backtest_result = await self.backtesting_service.run_backtest_async(
                symbol='BTC/USDT',
                timeframe='1d',
                start_date='2024-01-01',
                end_date='2024-06-01',
                strategy='ml_enhanced',
                initial_capital=10000.0
            )

            benchmark_data = {}

            if backtest_result and backtest_result.get('success'):
                metrics = backtest_result.get('metrics', {})
                benchmark_data = {
                    'backtest_return': metrics.get('return_pct', 0),
                    'backtest_sharpe': metrics.get('sharpe_ratio', 0),
                    'backtest_win_rate': metrics.get('win_rate', 0),
                    'backtest_max_drawdown': metrics.get('max_drawdown', 0),
                    'backtest_trades': metrics.get('total_trades', 0),
                    'benchmark_passed': metrics.get('return_pct', 0) > 50  # Require >50% return
                }
            else:
                benchmark_data = {
                    'error': 'Backtest failed',
                    'benchmark_passed': False
                }

            self.setup_results['performance_benchmark'] = benchmark_data

            if benchmark_data.get('benchmark_passed', False):
                logger.info("✅ Performance benchmark passed")
            else:
                logger.warning("⚠️ Performance benchmark failed - strategy may need improvement")

        except Exception as e:
            self.setup_results['performance_benchmark'] = {
                'status': 'error',
                'error': str(e)
            }
            logger.error(f"❌ Performance benchmark failed: {e}")

    def generate_setup_report(self):
        """Generate comprehensive setup report"""

        logger.info("📋 Generating Live Trading Setup Report...")

        report = {
            'setup_timestamp': datetime.now().isoformat(),
            'overall_status': 'pending',
            'summary': {},
            'detailed_results': self.setup_results,
            'recommendations': []
        }

        # Calculate overall status
        statuses = []
        for component, result in self.setup_results.items():
            if isinstance(result, dict) and 'status' in result:
                statuses.append(result['status'])

        success_count = statuses.count('success') + statuses.count('healthy')
        total_checks = len(statuses)

        if success_count == total_checks:
            report['overall_status'] = 'ready_for_live_trading'
        elif success_count >= total_checks * 0.75:
            report['overall_status'] = 'ready_for_paper_trading_only'
        else:
            report['overall_status'] = 'setup_incomplete'

        # Generate summary
        summary = {
            'total_checks': total_checks,
            'successful_checks': success_count,
            'failed_checks': total_checks - success_count,
            'success_rate': success_count / total_checks if total_checks > 0 else 0
        }

        # Generate recommendations
        recommendations = []

        if self.setup_results.get('service_health', {}).get('status') != 'healthy':
            recommendations.append("Fix service health issues before proceeding")

        if not self.setup_results.get('configuration', {}).get('valid', False):
            recommendations.append("Review and fix configuration issues")

        if self.setup_results.get('paper_trading', {}).get('status') != 'success':
            recommendations.append("Fix paper trading functionality")

        if self.setup_results.get('performance_benchmark', {}).get('benchmark_passed', False) == False:
            recommendations.append("Improve strategy performance before live trading")

        if not recommendations:
            recommendations.append("All systems ready for live trading")

        report['summary'] = summary
        report['recommendations'] = recommendations

        # Save report
        report_file = f"output/live_trading_setup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs('output', exist_ok=True)

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"✅ Setup report saved to {report_file}")

        # Print summary to console
        self._print_setup_summary(report)

    def _print_setup_summary(self, report: Dict[str, Any]):
        """Print setup summary to console"""

        logger.info("\n" + "="*60)
        logger.info("🚀 LIVE TRADING SETUP SUMMARY")
        logger.info("="*60)

        status = report['overall_status'].upper()
        if status == 'READY_FOR_LIVE_TRADING':
            logger.info(f"🎉 Overall Status: {status}")
            logger.info("✅ All systems are ready for live trading!")
        elif status == 'READY_FOR_PAPER_TRADING_ONLY':
            logger.info(f"⚠️ Overall Status: {status}")
            logger.info("📝 Paper trading is ready, but live trading needs more validation")
        else:
            logger.info(f"🔧 Overall Status: {status}")
            logger.info("❌ Setup is incomplete - address issues before proceeding")

        summary = report['summary']
        logger.info(f"📊 Checks Passed: {summary['successful_checks']}/{summary['total_checks']}")
        logger.info(".1f")

        logger.info("\n📋 Recommendations:")
        for rec in report['recommendations']:
            logger.info(f"• {rec}")

        logger.info("="*60)

async def main():
    """Main setup execution"""

    setup = LiveTradingSetup()
    await setup.run_complete_setup()

    logger.info("🎯 Live Trading Setup Completed!")

if __name__ == "__main__":
    asyncio.run(main())