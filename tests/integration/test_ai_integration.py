#!/usr/bin/env python3
"""
Integration Test for AI Services in TradPal Orchestrator
Tests the integration of Alternative Data, Market Regime Detection, and Reinforcement Learning services.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import TradPalOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_ai_services_integration():
    """Test integration of AI services in the orchestrator."""
    logger.info("🚀 Starting AI Services Integration Test...")

    # Initialize orchestrator
    orchestrator = TradPalOrchestrator()

    try:
        # Initialize services
        logger.info("Initializing services...")
        init_success = await orchestrator.initialize_services()

        if not init_success:
            logger.error("❌ Service initialization failed")
            return False

        # Check which AI services are available
        available_services = list(orchestrator.services.keys())
        logger.info(f"Available services: {available_services}")

        ai_services = ['alternative_data', 'market_regime', 'reinforcement_learning']
        available_ai_services = [s for s in ai_services if s in available_services]

        if not available_ai_services:
            logger.warning("⚠️  No AI services available - testing basic functionality")
            return await test_basic_functionality(orchestrator)

        logger.info(f"✅ AI Services available: {available_ai_services}")

        # Test AI service integration
        test_results = {}

        # Test Alternative Data Service
        if 'alternative_data' in orchestrator.services:
            logger.info("Testing Alternative Data Service...")
            try:
                # Test sentiment data
                sentiment = await orchestrator.services['alternative_data'].get_sentiment_data("BTC/USDT")
                test_results['alternative_data_sentiment'] = bool(sentiment)

                # Test composite score
                composite = await orchestrator.services['alternative_data'].get_composite_score("BTC/USDT")
                test_results['alternative_data_composite'] = bool(composite)

                logger.info("✅ Alternative Data Service working")
            except Exception as e:
                logger.error(f"❌ Alternative Data Service failed: {e}")
                test_results['alternative_data_error'] = str(e)

        # Test Market Regime Detection Service
        if 'market_regime' in orchestrator.services:
            logger.info("Testing Market Regime Detection Service...")
            try:
                # Test regime detection
                regime = await orchestrator.services['market_regime'].get_market_regime("BTC/USDT")
                test_results['market_regime_detection'] = bool(regime)

                # Test regime features
                features = await orchestrator.services['market_regime'].get_regime_features("BTC/USDT")
                test_results['market_regime_features'] = bool(features)

                logger.info("✅ Market Regime Detection Service working")
            except Exception as e:
                logger.error(f"❌ Market Regime Detection Service failed: {e}")
                test_results['market_regime_error'] = str(e)

        # Test Reinforcement Learning Service
        if 'reinforcement_learning' in orchestrator.services:
            logger.info("Testing Reinforcement Learning Service...")
            try:
                # Test model info
                model_info = await orchestrator.services['reinforcement_learning'].get_model_info()
                test_results['rl_model_info'] = bool(model_info)

                # Test health check
                health = await orchestrator.services['reinforcement_learning'].get_health_status()
                test_results['rl_health'] = bool(health)

                logger.info("✅ Reinforcement Learning Service working")
            except Exception as e:
                logger.error(f"❌ Reinforcement Learning Service failed: {e}")
                test_results['rl_error'] = str(e)

        # Test AI-enhanced signal processing
        logger.info("Testing AI-enhanced signal processing...")
        try:
            # Create mock data for testing
            mock_signals = {
                'signals': [
                    {'action': 'BUY', 'strength': 0.7, 'timestamp': datetime.now().isoformat()}
                ]
            }

            mock_indicators = {
                'rsi': 65.0,
                'macd': 0.5,
                'bb_position': 0.8
            }

            mock_market_data = [
                {
                    'close': 50000.0,
                    'high': 51000.0,
                    'low': 49000.0,
                    'volume': 100.0,
                    'timestamp': datetime.now().isoformat()
                }
            ]

            # Test signal enhancement
            enhanced_signals = await orchestrator._enhance_signals_with_ai(
                mock_signals, mock_indicators, {}, {}, mock_market_data
            )

            test_results['signal_enhancement'] = bool(enhanced_signals)
            test_results['ai_enhanced'] = enhanced_signals.get('ai_enhanced', False)

            logger.info("✅ AI Signal Enhancement working")

        except Exception as e:
            logger.error(f"❌ AI Signal Enhancement failed: {e}")
            test_results['signal_enhancement_error'] = str(e)

        # Summary
        successful_tests = sum(1 for v in test_results.values() if isinstance(v, bool) and v)
        total_tests = len([v for v in test_results.values() if isinstance(v, bool)])

        logger.info("📊 Integration Test Results:")
        logger.info(f"   Total AI Services Available: {len(available_ai_services)}")
        logger.info(f"   Tests Passed: {successful_tests}/{total_tests}")

        for test_name, result in test_results.items():
            if isinstance(result, bool):
                status = "✅" if result else "❌"
                logger.info(f"   {status} {test_name}")
            elif isinstance(result, str):
                logger.info(f"   ❌ {test_name}: {result}")

        return successful_tests > 0

    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        return False

    finally:
        # Cleanup
        for service_name, service in orchestrator.services.items():
            try:
                if hasattr(service, 'close'):
                    await service.close()
            except Exception as e:
                logger.warning(f"Failed to close {service_name}: {e}")


async def test_basic_functionality(orchestrator):
    """Test basic orchestrator functionality when AI services are not available."""
    logger.info("Testing basic orchestrator functionality...")

    try:
        # Test service initialization
        init_success = await orchestrator.initialize_services()
        if not init_success:
            logger.error("❌ Basic service initialization failed")
            return False

        # Check core services
        core_services = ['core', 'data', 'backtesting']
        available_core = [s for s in core_services if s in orchestrator.services]

        logger.info(f"✅ Core services available: {available_core}")
        return len(available_core) > 0

    except Exception as e:
        logger.error(f"❌ Basic functionality test failed: {e}")
        return False


async def main():
    """Main test function."""
    logger.info("🤖 TradPal AI Services Integration Test")
    logger.info("=" * 50)

    success = await test_ai_services_integration()

    logger.info("=" * 50)
    if success:
        logger.info("🎉 Integration test PASSED")
        return 0
    else:
        logger.info("💥 Integration test FAILED")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)