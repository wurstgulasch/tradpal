#!/usr/bin/env python3
"""
Test script for audit logging functionality.
Tests signal decision logging, risk assessment logging, and system events.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add services to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'services'))

from services.audit_logger import (
    audit_logger, log_signal_buy, log_signal_sell, log_trade_buy, log_trade_sell,
    log_system_event, log_error, log_performance_metrics,
    get_recent_logs, analyze_audit_trail,
    SignalDecision, TradeExecution, RiskAssessment
)
from config.settings import SYMBOL, TIMEFRAME


def create_test_dataframe():
    """Create a test DataFrame with sample trading data."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
    np.random.seed(42)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': 50000 + np.random.normal(0, 100, 100),
        'high': 50100 + np.random.normal(0, 100, 100),
        'low': 49900 + np.random.normal(0, 100, 100),
        'close': 50000 + np.random.normal(0, 100, 100),
        'volume': np.random.randint(100, 1000, 100)
    })

    # Add some calculated indicators
    df['EMA9'] = df['close'].ewm(span=9).mean()
    df['EMA21'] = df['close'].ewm(span=21).mean()
    df['RSI'] = 50 + np.random.normal(0, 10, 100)  # Mock RSI
    df['BB_upper'] = df['close'] + 200
    df['BB_lower'] = df['close'] - 200
    df['ATR'] = 100 + np.random.normal(0, 20, 100)

    # Add signals
    df['Buy_Signal'] = 0
    df['Sell_Signal'] = 0

    # Create some test signals
    df.loc[20, 'Buy_Signal'] = 1  # Buy signal at index 20
    df.loc[60, 'Sell_Signal'] = 1  # Sell signal at index 60

    # Add risk management columns
    df['Position_Size_Absolute'] = 1000.0
    df['Position_Size_Percent'] = 1.0
    df['Stop_Loss_Buy'] = df['close'] - 50
    df['Take_Profit_Buy'] = df['close'] + 100
    df['Leverage'] = 2

    return df


def test_signal_logging():
    """Test signal decision logging."""
    print("Testing signal decision logging...")

    df = create_test_dataframe()

    # Test buy signal logging
    buy_row = df.iloc[20]  # Row with buy signal
    indicators = {
        'EMA9': float(buy_row['EMA9']),
        'EMA21': float(buy_row['EMA21']),
        'RSI': float(buy_row['RSI']),
        'BB_upper': float(buy_row['BB_upper']),
        'BB_lower': float(buy_row['BB_lower']),
        'ATR': float(buy_row['ATR'])
    }

    risk_metrics = {
        'Position_Size_Absolute': float(buy_row['Position_Size_Absolute']),
        'Position_Size_Percent': float(buy_row['Position_Size_Percent']),
        'Stop_Loss_Buy': float(buy_row['Stop_Loss_Buy']),
        'Take_Profit_Buy': float(buy_row['Take_Profit_Buy']),
        'Leverage': float(buy_row['Leverage'])
    }

    log_signal_buy(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        confidence=0.85,
        indicators=indicators,
        risk_metrics=risk_metrics,
        reasoning="EMA9 > EMA21, RSI oversold, price above BB lower"
    )

    # Test sell signal logging
    sell_row = df.iloc[60]  # Row with sell signal
    indicators_sell = {
        'EMA9': float(sell_row['EMA9']),
        'EMA21': float(sell_row['EMA21']),
        'RSI': float(sell_row['RSI']),
        'BB_upper': float(sell_row['BB_upper']),
        'BB_lower': float(sell_row['BB_lower']),
        'ATR': float(sell_row['ATR'])
    }

    risk_metrics_sell = {
        'Position_Size_Absolute': float(sell_row['Position_Size_Absolute']),
        'Position_Size_Percent': float(sell_row['Position_Size_Percent']),
        'Stop_Loss_Buy': float(sell_row['Stop_Loss_Buy']),
        'Take_Profit_Buy': float(sell_row['Take_Profit_Buy']),
        'Leverage': float(sell_row['Leverage'])
    }

    log_signal_sell(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        confidence=0.80,
        indicators=indicators_sell,
        risk_metrics=risk_metrics_sell,
        reasoning="EMA9 < EMA21, RSI overbought, price below BB upper"
    )

    print("✅ Signal logging tests completed")


def test_trade_logging():
    """Test trade execution logging."""
    print("Testing trade execution logging...")

    # Test buy trade
    log_trade_buy(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        quantity=0.1,
        price=50000.0,
        risk_amount=500.0,
        stop_loss=49500.0,
        take_profit=51000.0,
        reason="EMA crossover buy signal"
    )

    # Test sell trade
    log_trade_sell(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        quantity=0.1,
        price=50500.0,
        pnl=500.0,
        reason="Take profit hit"
    )

    print("✅ Trade logging tests completed")


def test_risk_assessment_logging():
    """Test risk assessment logging."""
    print("Testing risk assessment logging...")

    assessment = RiskAssessment(
        timestamp=datetime.now().isoformat(),
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        capital=10000.0,
        risk_per_trade=0.01,
        position_size=1000.0,
        atr_value=150.0,
        leverage=2.0,
        max_drawdown_limit=0.1,
        assessment_result="APPROVED"
    )

    audit_logger.log_risk_assessment(assessment)
    print("✅ Risk assessment logging test completed")


def test_system_events():
    """Test system event logging."""
    print("Testing system event logging...")

    # Test system startup
    log_system_event(
        event_type="TEST_STARTUP",
        message="Audit logging test started",
        details={"test_mode": True, "components": ["signals", "trades", "risk"]}
    )

    # Test error logging
    try:
        raise ValueError("Test error for audit logging")
    except Exception as e:
        log_error(
            error_type="TEST_ERROR",
            message="Test error occurred",
            traceback=str(e),
            context={"test_function": "test_system_events"}
        )

    # Test performance metrics
    log_performance_metrics()

    print("✅ System event logging tests completed")


def test_log_analysis():
    """Test log analysis functionality."""
    print("Testing log analysis...")

    # Get recent logs
    recent_logs = get_recent_logs(lines=10)
    print(f"Retrieved {len(recent_logs)} recent log entries")

    # Analyze audit trail
    analysis = analyze_audit_trail(days=1)
    print(f"Audit trail analysis: {analysis}")

    print("✅ Log analysis tests completed")


def main():
    """Run all audit logging tests."""
    print("🧪 Starting Audit Logging Tests")
    print("=" * 50)

    try:
        test_signal_logging()
        test_trade_logging()
        test_risk_assessment_logging()
        test_system_events()
        test_log_analysis()

        print("=" * 50)
        print("✅ All audit logging tests completed successfully!")
        print("Check the audit log file for the logged entries.")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        log_error(
            error_type="TEST_FAILURE",
            message=f"Audit logging test failed: {str(e)}",
            traceback=str(e)
        )
        sys.exit(1)


if __name__ == "__main__":
    main()