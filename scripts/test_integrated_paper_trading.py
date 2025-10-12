#!/usr/bin/env python3
"""
Test script for integrated paper trading with signal generation.

This script tests the complete integration of:
- Signal generation (traditional + ML + sentiment)
- Paper trading execution
- Portfolio tracking and performance metrics
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.append(os.path.join(project_root, 'src'))

def test_integrated_paper_trading():
    """Test integrated paper trading with signal generation."""
    print("🧪 Testing Integrated Paper Trading with Signal Generation")
    print("=" * 60)

    try:
        # Force enable paper trading for testing
        import config.settings as settings
        original_paper_trading = settings.PAPER_TRADING_ENABLED
        settings.PAPER_TRADING_ENABLED = True

        # Import required modules
        from src.data_fetcher import fetch_historical_data
        from src.indicators import calculate_indicators
        from src.signal_generator import generate_signals, calculate_risk_management
        from src.paper_trading import get_paper_portfolio_summary, reset_paper_portfolio
        from config.settings import SYMBOL, TIMEFRAME, PAPER_TRADING_ENABLED

        print(f"📊 Testing with {SYMBOL} on {TIMEFRAME} timeframe")
        print(f"💰 Paper trading enabled: {PAPER_TRADING_ENABLED}")

        # Reset paper portfolio for clean test
        reset_paper_portfolio()
        print("🔄 Paper portfolio reset for testing")

        # Fetch historical data
        print("\n📥 Fetching historical data...")
        end_date = datetime.now()
        start_date_dt = end_date - timedelta(days=30)  # Last 30 days for testing

        df = fetch_historical_data(
            symbol=SYMBOL,
            exchange_name='kraken',
            timeframe=TIMEFRAME,
            start_date=start_date_dt,
            limit=500
        )

        if df.empty:
            print("❌ No data fetched for testing")
            return False

        print(f"✅ Fetched {len(df)} data points")

        # Calculate indicators
        print("\n📈 Calculating indicators...")
        df = calculate_indicators(df)
        print("✅ Indicators calculated")

        # Generate signals (this will now include paper trading execution)
        print("\n🎯 Generating signals with paper trading...")
        df = generate_signals(df)
        print("✅ Signals generated")

        # Calculate risk management
        print("\n⚖️  Calculating risk management...")
        df = calculate_risk_management(df)
        print("✅ Risk management calculated")

        # Analyze results
        print("\n📊 Analysis Results:")
        print("-" * 40)

        # Count signals
        buy_signals = df['Buy_Signal'].sum()
        sell_signals = df['Sell_Signal'].sum()
        total_signals = buy_signals + sell_signals

        print(f"Total signals generated: {total_signals}")
        print(f"Buy signals: {buy_signals}")
        print(f"Sell signals: {sell_signals}")

        # Count paper trades executed
        if 'Paper_Trade_Executed' in df.columns:
            executed_trades = df['Paper_Trade_Executed'].sum()
            buy_trades = (df['Paper_Trade_Type'] == 'BUY').sum()
            sell_trades = (df['Paper_Trade_Type'] == 'SELL').sum()

            print(f"\n📈 Paper Trading Results:")
            print(f"Trades executed: {executed_trades}")
            print(f"Buy trades: {buy_trades}")
            print(f"Sell trades: {sell_trades}")
        else:
            print("⚠️  Paper trading columns not found in DataFrame")

        # Get portfolio summary
        portfolio_summary = get_paper_portfolio_summary()
        if portfolio_summary:
            print("\n💼 Portfolio Summary:")
            print(f"Initial capital: ${portfolio_summary.get('initial_capital', 0):.2f}")
            print(f"Current capital: ${portfolio_summary.get('current_capital', 0):.2f}")
            print(f"Total P&L: ${portfolio_summary.get('total_pnl', 0):.2f}")
            print(f"Total trades: {portfolio_summary.get('total_trades', 0)}")
            print(f"Win rate: {portfolio_summary.get('win_rate', 0):.1f}%")

            if portfolio_summary.get('total_trades', 0) > 0:
                print(f"Return: {portfolio_summary.get('total_pnl', 0) / portfolio_summary.get('initial_capital', 1) * 100:.2f}%")

        # Show recent signals with paper trading info
        print("\n🎯 Recent Signals:")
        recent_signals = df.tail(10)
        for idx, row in recent_signals.iterrows():
            timestamp = idx if isinstance(idx, pd.Timestamp) else pd.Timestamp.now()
            signal_info = []

            if row.get('Buy_Signal', 0) == 1:
                signal_info.append("BUY")
            if row.get('Sell_Signal', 0) == 1:
                signal_info.append("SELL")

            if signal_info:
                trade_executed = row.get('Paper_Trade_Executed', False)
                trade_type = row.get('Paper_Trade_Type', None)
                portfolio_value = row.get('Paper_Portfolio_Value', 0)

                print(f"  {timestamp.strftime('%Y-%m-%d %H:%M')} - {', '.join(signal_info)} "
                      f"@ ${row['close']:.4f} - Paper Trade: {trade_executed} "
                      f"({trade_type or 'None'}) - Portfolio: ${portfolio_value:.2f}")

        # Save test results
        output_file = f"output/test_integrated_paper_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        result_data = {
            'test_timestamp': datetime.now().isoformat(),
            'symbol': SYMBOL,
            'timeframe': TIMEFRAME,
            'data_points': len(df),
            'total_signals': total_signals,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'portfolio_summary': portfolio_summary,
            'test_success': True
        }

        # Save to JSON
        import json
        with open(output_file, 'w') as f:
            json.dump(result_data, f, indent=2, default=str)

        print(f"\n💾 Test results saved to: {output_file}")

        print("\n✅ Integrated paper trading test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Restore original paper trading setting
        try:
            settings.PAPER_TRADING_ENABLED = original_paper_trading
        except:
            pass


def test_paper_trading_edge_cases():
    """Test edge cases for paper trading integration."""
    print("\n🧪 Testing Paper Trading Edge Cases")
    print("=" * 40)

    try:
        from src.paper_trading import reset_paper_portfolio, get_paper_portfolio_summary
        from config.settings import PAPER_TRADING_ENABLED
        import pandas as pd

        if not PAPER_TRADING_ENABLED:
            print("⚠️  Paper trading disabled, skipping edge case tests")
            return True

        # Test 1: Empty DataFrame
        print("Test 1: Empty DataFrame handling")
        from src.signal_generator import execute_paper_trades
        empty_df = pd.DataFrame()
        result_df = execute_paper_trades(empty_df)
        print(f"✅ Empty DataFrame handled: {len(result_df)} rows")

        # Test 2: DataFrame without signals
        print("\nTest 2: DataFrame without signals")
        no_signal_df = pd.DataFrame({
            'close': [100, 101, 102],
            'timestamp': pd.date_range('2023-01-01', periods=3)
        })
        result_df = execute_paper_trades(no_signal_df)
        executed_trades = result_df.get('Paper_Trade_Executed', pd.Series([False]*len(result_df))).sum()
        print(f"✅ No signals handled: {executed_trades} trades executed")

        # Test 3: Invalid signal data
        print("\nTest 3: Invalid signal data handling")
        invalid_df = pd.DataFrame({
            'close': [100],
            'Buy_Signal': [None],  # Invalid signal
            'Sell_Signal': [None],  # Invalid signal
            'timestamp': [pd.Timestamp.now()]
        })
        result_df = execute_paper_trades(invalid_df)
        executed_trades = result_df.get('Paper_Trade_Executed', pd.Series([False]*len(result_df))).sum()
        print(f"✅ Invalid signals handled: {executed_trades} trades executed")

        print("\n✅ Edge case tests completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Edge case test failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Starting Integrated Paper Trading Tests")
    print("=" * 50)

    # Run main integration test
    success1 = test_integrated_paper_trading()

    # Run edge case tests
    success2 = test_paper_trading_edge_cases()

    # Summary
    print("\n" + "=" * 50)
    if success1 and success2:
        print("🎉 All tests passed! Paper trading integration is working correctly.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Check the output above for details.")
        sys.exit(1)