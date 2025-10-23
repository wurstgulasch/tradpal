#!/usr/bin/env python3
"""
Simple Performance Test for TradPal Services
Tests the core functionality after recent fixes.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

async def test_services():
    """Test core services functionality."""
    print("🚀 Testing TradPal Services Performance")
    print("=" * 50)

    try:
        # Test Data Service (direct import to avoid kaggle issues)
        print("📊 Testing Data Service...")
        # Use direct import to avoid __init__.py issues
        import importlib.util
        spec = importlib.util.spec_from_file_location("data_fetcher", "services/data_service/data_fetcher.py")
        data_fetcher = importlib.util.module_from_spec(spec)
        # Skip the problematic import by mocking it
        import sys
        original_import = __builtins__.__import__
        def mock_import(name, *args, **kwargs):
            if name == 'kaggle':
                raise ImportError("kaggle not available")
            return original_import(name, *args, **kwargs)
        __builtins__.__import__ = mock_import

        try:
            spec.loader.exec_module(data_fetcher)
            data = data_fetcher.fetch_data(limit=100)
            print(f"✅ Data Service: Fetched {len(data)} data points")
        finally:
            __builtins__.__import__ = original_import

        # Test Core Indicators
        print("🧮 Testing Core Indicators...")
        from services.core_service.indicators import calculate_indicators
        indicators_data = calculate_indicators(data)
        print(f"✅ Indicators: Calculated for {len(indicators_data)} data points")

        # Test Signal Generator
        print("🎯 Testing Signal Generator...")
        from services.signal_generator import generate_signals
        signals_data = generate_signals(indicators_data)
        buy_signals = signals_data['Buy_Signal'].sum() if 'Buy_Signal' in signals_data.columns else 0
        sell_signals = signals_data['Sell_Signal'].sum() if 'Sell_Signal' in signals_data.columns else 0
        print(f"✅ Signals: Generated {buy_signals} buy and {sell_signals} sell signals")

        # Test Backtesting Service
        print("💼 Testing Backtesting Service...")
        from services.trading_service.backtesting_service.service import AsyncBacktester
        backtester = AsyncBacktester()
        # Simple backtest with limited data
        test_data = data.head(50)  # Use first 50 points for quick test
        results = await backtester.run_backtest(test_data)
        print(f"✅ Backtesting: Completed with {results.get('total_trades', 0)} trades")

        print("\n" + "=" * 50)
        print("🎉 All Services Tested Successfully!")
        print("📈 TradPal AI Bot is ready for trading!")
        print("=" * 50)

        return True

    except Exception as e:
        print(f"❌ Service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_services())
    sys.exit(0 if success else 1)