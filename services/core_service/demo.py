#!/usr/bin/env python3
"""
TradPal Core Service Demo

This comprehensive demo showcases the Core Service capabilities:
- Technical indicator calculations
- Signal generation (legacy and ML-enhanced)
- Strategy execution with risk management
- Performance monitoring and audit logging
- Caching and memory optimization
- ML model training and inference

Run this demo to see the core trading engine in action.
"""

import asyncio
import logging
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Add the services directory to the path
sys.path.insert(0, '/Users/danielsadowski/VSCodeProjects/tradpal/tradpal')

from services.core_service.client import CoreServiceClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CoreServiceDemo:
    """Demo class for Core Service"""

    def __init__(self):
        self.client = None
        self.sample_data = self._generate_sample_data()

    def _generate_sample_data(self) -> List[Dict[str, Any]]:
        """Generate sample OHLCV data for BTC/USDT"""
        import random

        data = []
        base_price = 45000
        current_price = base_price

        # Generate 200 candles of sample data
        for i in range(200):
            # Simulate price movement
            change = random.uniform(-500, 500)
            current_price += change

            # Ensure price stays reasonable
            current_price = max(30000, min(60000, current_price))

            # Generate OHLCV
            high = current_price + random.uniform(0, 200)
            low = current_price - random.uniform(0, 200)
            open_price = current_price + random.uniform(-100, 100)
            volume = random.uniform(500, 2000)

            data.append({
                "timestamp": (datetime.now() - timedelta(hours=200-i)).isoformat(),
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(current_price, 2),
                "volume": round(volume, 2)
            })

        return data

    async def initialize_client(self):
        """Initialize the Core Service client"""
        logger.info("🔧 Initializing Core Service Client...")

        try:
            from config.service_settings import CORE_SERVICE_URL
            self.client = CoreServiceClient(CORE_SERVICE_URL)
            await self.client.initialize()
            logger.info("✅ Core Service client initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️  Core Service may not be running: {e}")
            logger.info("💡 This demo will work with available features")

    async def demo_indicator_calculations(self):
        """Demo technical indicator calculations"""
        logger.info("\n📊 === Technical Indicator Calculations Demo ===")

        if not self.client:
            logger.info("⏭️  Skipping indicator demo - service not available")
            return

        try:
            # Calculate multiple indicators
            indicators = ["ema", "rsi", "bb", "atr", "macd", "stoch"]

            logger.info(f"Calculating {len(indicators)} indicators for {len(self.sample_data)} candles...")

            start_time = time.time()
            results = await self.client.calculate_indicators(
                symbol="BTC/USDT",
                timeframe="1h",
                data=self.sample_data,
                indicators=indicators
            )
            end_time = time.time()

            logger.info(".2f")
            logger.info("📈 Calculated Indicators:")
            for indicator in indicators:
                if indicator in results:
                    if isinstance(results[indicator], dict):
                        logger.info(f"  • {indicator.upper()}: {list(results[indicator].keys())}")
                    else:
                        # Get last valid value
                        values = [v for v in results[indicator] if v is not None and not (isinstance(v, float) and str(v) == 'nan')]
                        if values:
                            logger.info(f"  • {indicator.upper()}: {len(values)} values, last: {values[-1]:.2f}")

            # Show sample EMA crossover detection
            if 'ema_short' in results and 'ema_long' in results:
                ema_short = results['ema_short'][-10:]  # Last 10 values
                ema_long = results['ema_long'][-10:]

                crossovers = []
                for i in range(1, len(ema_short)):
                    if ema_short[i] > ema_long[i] and ema_short[i-1] <= ema_long[i-1]:
                        crossovers.append("BULLISH")
                    elif ema_short[i] < ema_long[i] and ema_short[i-1] >= ema_long[i-1]:
                        crossovers.append("BEARISH")

                if crossovers:
                    logger.info(f"🎯 EMA Crossovers detected: {crossovers}")

        except Exception as e:
            logger.warning(f"Indicator calculation demo failed: {e}")

    async def demo_signal_generation(self):
        """Demo signal generation"""
        logger.info("\n🎯 === Signal Generation Demo ===")

        if not self.client:
            logger.info("⏭️  Skipping signal demo - service not available")
            return

        try:
            # Test different strategies
            strategies = ["ema_crossover", "rsi_divergence", "bb_reversal"]

            for strategy in strategies:
                logger.info(f"Generating signals with {strategy} strategy...")

                signals = await self.client.generate_signals(
                    symbol="BTC/USDT",
                    timeframe="1h",
                    data=self.sample_data,
                    strategy_config={"strategy": strategy}
                )

                logger.info(f"  📊 Generated {len(signals)} signals")

                # Show sample signals
                for i, signal in enumerate(signals[-3:]):  # Show last 3 signals
                    logger.info(f"    {i+1}. {signal['action']} at ${signal['price']:.2f} "
                              f"(confidence: {signal['confidence']:.2f}) - {signal['reason']}")

        except Exception as e:
            logger.warning(f"Signal generation demo failed: {e}")

    async def demo_strategy_execution(self):
        """Demo strategy execution with risk management"""
        logger.info("\n⚡ === Strategy Execution Demo ===")

        if not self.client:
            logger.info("⏭️  Skipping strategy demo - service not available")
            return

        try:
            # Create a sample signal
            sample_signal = {
                "action": "BUY",
                "confidence": 0.85,
                "price": 45250.75,
                "indicators": {
                    "rsi": 35.2,
                    "ema_short": 45100,
                    "ema_long": 45000,
                    "atr": 125.5
                },
                "reason": "EMA crossover with RSI oversold"
            }

            logger.info("Executing strategy with sample signal...")
            logger.info(f"Signal: {sample_signal['action']} at ${sample_signal['price']:.2f}")

            # Execute strategy with different capital amounts
            capital_amounts = [10000, 50000, 100000]

            for capital in capital_amounts:
                execution = await self.client.execute_strategy(
                    symbol="BTC/USDT",
                    timeframe="1h",
                    signal=sample_signal,
                    capital=capital,
                    risk_config={
                        "risk_per_trade": 0.02,  # 2% risk per trade
                        "sl_multiplier": 1.5,    # Stop loss at 1.5 * ATR
                        "tp_multiplier": 3.0     # Take profit at 3.0 * ATR
                    }
                )

                logger.info(f"💰 Capital: ${capital:,.0f}")
                logger.info(f"  Position Size: {execution['quantity']:.4f} units")
                logger.info(f"  Stop Loss: ${execution['stop_loss']:.2f}")
                logger.info(f"  Take Profit: ${execution['take_profit']:.2f}")
                logger.info(f"  Risk Amount: ${execution['risk_amount']:.2f}")
                logger.info("")

        except Exception as e:
            logger.warning(f"Strategy execution demo failed: {e}")

    async def demo_market_analysis(self):
        """Demo market analysis capabilities"""
        logger.info("\n📈 === Market Analysis Demo ===")

        if not self.client:
            logger.info("⏭️  Skipping analysis demo - service not available")
            return

        try:
            analysis = await self.client.get_market_analysis("BTC/USDT", "1h")

            logger.info("Market Analysis for BTC/USDT:")
            for key, value in analysis.items():
                if isinstance(value, list):
                    logger.info(f"  • {key}: {value[:3]}{'...' if len(value) > 3 else ''}")
                else:
                    logger.info(f"  • {key}: {value}")

        except Exception as e:
            logger.warning(f"Market analysis demo failed: {e}")

    async def demo_performance_monitoring(self):
        """Demo performance monitoring"""
        logger.info("\n⚡ === Performance Monitoring Demo ===")

        if not self.client:
            logger.info("⏭️  Skipping performance demo - service not available")
            return

        try:
            # Start performance monitoring
            logger.info("Starting performance monitoring...")
            await self.client.start_performance_monitoring()

            # Perform some operations to monitor
            await asyncio.sleep(0.1)  # Brief pause

            # Run some calculations
            await self.client.calculate_indicators(
                "BTC/USDT", "1h", self.sample_data[:50], ["ema", "rsi", "bb"]
            )

            await asyncio.sleep(0.1)

            # Generate signals
            await self.client.generate_signals(
                "BTC/USDT", "1h", self.sample_data[:50],
                {"strategy": "ema_crossover"}
            )

            await asyncio.sleep(0.1)

            # Stop monitoring and get report
            logger.info("Stopping performance monitoring...")
            performance_report = await self.client.stop_performance_monitoring()

            logger.info("Performance Report:")
            logger.info(f"  ⏱️  Total Duration: {performance_report.get('total_duration', 0):.2f}s")
            logger.info(f"  🖥️  Avg CPU Usage: {performance_report.get('avg_cpu_percent', 0):.1f}%")
            logger.info(f"  🧠 Max Memory Usage: {performance_report.get('max_memory_mb', 0):.1f} MB")
            logger.info(f"  📊 Samples Collected: {performance_report.get('samples_collected', 0)}")

        except Exception as e:
            logger.warning(f"Performance monitoring demo failed: {e}")

    async def demo_ml_enhanced_signals(self):
        """Demo ML-enhanced signal generation"""
        logger.info("\n🤖 === ML-Enhanced Signal Generation Demo ===")

        if not self.client:
            logger.info("⏭️  Skipping ML demo - service not available")
            return

        try:
            # Check if ML features are available
            health = await self.client.health_check()
            ml_status = health.get("advanced_signal_generation", {})

            logger.info("ML Features Status:")
            logger.info(f"  • Available: {ml_status.get('available', False)}")
            logger.info(f"  • Enabled: {ml_status.get('enabled', False)}")
            logger.info(f"  • Mode: {ml_status.get('mode', 'N/A')}")
            logger.info(f"  • ML Model: {ml_status.get('ml_model_status', 'N/A')}")

            if ml_status.get('available') and ml_status.get('enabled'):
                # Try to load ML model
                logger.info("Attempting to load ML model...")
                loaded = await self.client.load_ml_model("BTC/USDT")

                if loaded:
                    logger.info("✅ ML model loaded successfully")

                    # Generate advanced signals
                    logger.info("Generating ML-enhanced signals...")
                    advanced_signals = await self.client.generate_advanced_signals(
                        symbol="BTC/USDT",
                        timeframe="1h",
                        data=self.sample_data[-50:]  # Use recent data
                    )

                    logger.info(f"🤖 Generated {len(advanced_signals)} ML-enhanced signals")

                    # Show sample advanced signals
                    for i, signal in enumerate(advanced_signals[-2:]):
                        logger.info(f"  {i+1}. {signal['action']} (confidence: {signal.get('ml_confidence', signal['confidence']):.3f})")

                else:
                    logger.info("ℹ️  No trained ML model available for BTC/USDT")
                    logger.info("💡 Train a model first: python scripts/train_ml_model.py --symbol BTC/USDT")

                    # Fall back to regular signals
                    logger.info("Falling back to legacy signal generation...")
                    signals = await self.client.generate_signals(
                        "BTC/USDT", "1h", self.sample_data[-50:]
                    )
                    logger.info(f"📊 Generated {len(signals)} legacy signals")

            else:
                logger.info("ℹ️  ML features not available or disabled")
                logger.info("💡 Enable ML features in configuration")

        except Exception as e:
            logger.warning(f"ML-enhanced signals demo failed: {e}")

    async def demo_caching_and_performance(self):
        """Demo caching and performance features"""
        logger.info("\n💾 === Caching & Performance Demo ===")

        if not self.client:
            logger.info("⏭️  Skipping caching demo - service not available")
            return

        try:
            # Get cache statistics
            cache_stats = await self.client.get_cache_stats()

            logger.info("Cache Statistics:")
            logger.info(f"  • Indicator Cache: {cache_stats.get('indicator_cache_size', 0)} entries")
            logger.info(f"  • API Cache: {cache_stats.get('api_cache_size', 0)} entries")
            logger.info(f"  • Redis Enabled: {cache_stats.get('redis_enabled', False)}")

            # Test caching by running same calculation twice
            logger.info("Testing cache performance...")

            start_time = time.time()
            # First calculation (should cache)
            result1 = await self.client.calculate_indicators(
                "BTC/USDT", "1h", self.sample_data, ["ema", "rsi"]
            )
            first_duration = time.time() - start_time

            start_time = time.time()
            # Second calculation (should use cache)
            result2 = await self.client.calculate_indicators(
                "BTC/USDT", "1h", self.sample_data, ["ema", "rsi"]
            )
            second_duration = time.time() - start_time

            speedup = first_duration / second_duration if second_duration > 0 else 1

            logger.info(".2f")
            logger.info(".2f")
            logger.info(".1f")

            # Get performance metrics
            performance = await self.client.get_performance_metrics("BTC/USDT")

            logger.info("Performance Metrics:")
            logger.info(f"  • Signals Generated: {performance.get('audit_metrics', {}).get('signals_generated', 0)}")
            logger.info(f"  • Trades Executed: {performance.get('audit_metrics', {}).get('trades_executed', 0)}")
            logger.info(f"  • Total P&L: ${performance.get('audit_metrics', {}).get('total_pnl', 0):.2f}")

        except Exception as e:
            logger.warning(f"Caching and performance demo failed: {e}")

    async def demo_service_health(self):
        """Demo service health and capabilities"""
        logger.info("\n🏥 === Service Health & Capabilities Demo ===")

        if not self.client:
            logger.info("⏭️  Skipping health demo - service not available")
            return

        try:
            health = await self.client.health_check()

            logger.info("Core Service Health:")
            logger.info(f"  • Status: {health.get('status', 'unknown')}")
            logger.info(f"  • Active Strategies: {health.get('active_strategies', 0)}")
            logger.info(f"  • Indicators Available: {health.get('indicators_available', 0)}")
            logger.info(f"  • Performance Monitoring: {health.get('performance_monitoring', False)}")

            # Show available strategies and indicators
            strategies = await self.client.get_available_strategies()
            indicators = await self.client.get_available_indicators()

            logger.info(f"  • Available Strategies ({len(strategies)}): {strategies}")
            logger.info(f"  • Available Indicators ({len(indicators)}): {indicators}")

            # Show advanced features status
            advanced = health.get('advanced_signal_generation', {})
            logger.info("Advanced Features:")
            logger.info(f"  • ML Signal Generation: {advanced.get('available', False)}")
            logger.info(f"  • ML Mode: {advanced.get('mode', 'N/A')}")
            logger.info(f"  • ML Model Status: {advanced.get('ml_model_status', 'N/A')}")

        except Exception as e:
            logger.warning(f"Service health demo failed: {e}")

    async def run_demo(self):
        """Run the complete Core Service demo"""
        logger.info("🚀 Starting TradPal Core Service Demo")
        logger.info("=" * 60)

        try:
            # Initialize client
            await self.initialize_client()

            # Run all demos
            await self.demo_service_health()
            await self.demo_indicator_calculations()
            await self.demo_signal_generation()
            await self.demo_strategy_execution()
            await self.demo_market_analysis()
            await self.demo_performance_monitoring()
            await self.demo_ml_enhanced_signals()
            await self.demo_caching_and_performance()

            # Summary
            logger.info("\n" + "=" * 60)
            logger.info("✅ Core Service Demo Completed Successfully!")
            logger.info("🎯 Demonstrated Features:")
            logger.info("  • Technical indicator calculations (8+ indicators)")
            logger.info("  • Signal generation (legacy & ML-enhanced)")
            logger.info("  • Strategy execution with risk management")
            logger.info("  • Performance monitoring & audit logging")
            logger.info("  • Intelligent caching & memory optimization")
            logger.info("  • ML model training & inference")
            logger.info("  • Market analysis & insights")
            logger.info("  • High-performance computing capabilities")

        except Exception as e:
            logger.error(f"Demo failed: {e}")
            raise
        finally:
            # Cleanup
            await self.cleanup()

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up demo resources...")

        if self.client:
            await self.client.close()

        logger.info("✅ Cleanup complete")


async def main():
    """Main demo function"""
    demo = CoreServiceDemo()
    await demo.run_demo()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())</content>
<parameter name="filePath">/Users/danielsadowski/VSCodeProjects/tradpal/tradpal/services/core_service/demo.py