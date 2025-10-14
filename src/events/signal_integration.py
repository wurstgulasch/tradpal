"""
Event Integration for Signal Generator

This module integrates the event-driven architecture with the existing
signal generation system, enabling event publishing and processing.
"""

import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime

from . import (
    EventType, MarketDataEvent, SignalEvent, RiskEvent, TradeEvent,
    event_bus, event_store, publish_market_data_event, publish_signal_event,
    publish_risk_event, publish_trade_event
)
from ..audit_logger import audit_logger


async def integrate_events_with_signals(df: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Integrate event publishing with signal generation.

    This function wraps the existing signal generation and publishes
    appropriate events for monitoring and audit trails.
    """
    if df.empty:
        return df

    try:
        # Publish market data event
        await publish_market_data_event(symbol, timeframe, df)

        # Check for signals and publish signal events
        if 'Buy_Signal' in df.columns and 'Sell_Signal' in df.columns:
            # Get latest signals
            latest = df.iloc[-1]
            buy_signal = latest.get('Buy_Signal', 0) == 1
            sell_signal = latest.get('Sell_Signal', 0) == 1

            if buy_signal or sell_signal:
                # Extract indicators for event
                indicators = {}
                for col in df.columns:
                    if col not in ['open', 'high', 'low', 'close', 'volume', 'Buy_Signal', 'Sell_Signal']:
                        if pd.notna(latest.get(col)):
                            indicators[col] = float(latest[col])

                # Determine signal type and reasoning
                if buy_signal:
                    signal_type = "BUY"
                    reasoning = _generate_signal_reasoning(df, "BUY", indicators)
                elif sell_signal:
                    signal_type = "SELL"
                    reasoning = _generate_signal_reasoning(df, "SELL", indicators)
                else:
                    return df

                # Get confidence (use ML confidence if available, otherwise default)
                confidence = latest.get('ML_Confidence', 0.8)
                source = latest.get('Signal_Source', 'traditional')

                # Publish signal event
                await publish_signal_event(
                    symbol, timeframe, signal_type, confidence,
                    indicators, reasoning, source
                )

        # Check for risk assessments and publish risk events
        if 'Position_Size_Absolute' in df.columns:
            latest = df.iloc[-1]

            # Determine assessment result based on position sizing
            position_size = latest.get('Position_Size_Absolute', 0)
            risk_amount = latest.get('Position_Size_Percent', 0)

            if risk_amount > 5:  # High risk
                assessment_result = "MODIFIED"
            elif risk_amount > 10:  # Too risky
                assessment_result = "REJECTED"
            else:
                assessment_result = "APPROVED"

            atr_value = latest.get('ATR', 0)

            # Publish risk event
            await publish_risk_event(
                symbol, timeframe, assessment_result,
                position_size, risk_amount, atr_value
            )

    except Exception as e:
        audit_logger.log_error(
            error_type="EVENT_INTEGRATION_ERROR",
            message=f"Failed to integrate events with signals: {str(e)}",
            context={'symbol': symbol, 'timeframe': timeframe}
        )

    return df


def _generate_signal_reasoning(df: pd.DataFrame, signal_type: str, indicators: Dict[str, Any]) -> str:
    """
    Generate human-readable reasoning for signal generation.
    """
    reasoning_parts = []

    if signal_type == "BUY":
        # EMA crossover
        if 'EMA9' in indicators and 'EMA21' in indicators:
            if indicators['EMA9'] > indicators['EMA21']:
                reasoning_parts.append(f"EMA9({indicators['EMA9']:.4f}) > EMA21({indicators['EMA21']:.4f})")

        # RSI oversold
        if 'RSI' in indicators and indicators['RSI'] < 30:
            reasoning_parts.append(f"RSI({indicators['RSI']:.2f}) oversold")

        # BB position
        if 'close' in df.columns and 'BB_lower' in indicators:
            latest_close = df.iloc[-1]['close']
            if latest_close > indicators['BB_lower']:
                reasoning_parts.append(f"Price({latest_close:.4f}) above BB_lower({indicators['BB_lower']:.4f})")

    elif signal_type == "SELL":
        # EMA crossover
        if 'EMA9' in indicators and 'EMA21' in indicators:
            if indicators['EMA9'] < indicators['EMA21']:
                reasoning_parts.append(f"EMA9({indicators['EMA9']:.4f}) < EMA21({indicators['EMA21']:.4f})")

        # RSI overbought
        if 'RSI' in indicators and indicators['RSI'] > 70:
            reasoning_parts.append(f"RSI({indicators['RSI']:.2f}) overbought")

        # BB position
        if 'close' in df.columns and 'BB_upper' in indicators:
            latest_close = df.iloc[-1]['close']
            if latest_close < indicators['BB_upper']:
                reasoning_parts.append(f"Price({latest_close:.4f}) below BB_upper({indicators['BB_upper']:.4f})")

    return " | ".join(reasoning_parts) if reasoning_parts else "Signal generated"


async def publish_trade_execution_event(symbol: str, timeframe: str, side: str,
                                       size: float, price: float, order_id: str,
                                       reason: str):
    """
    Publish trade execution event (can be called from trading logic).
    """
    try:
        await publish_trade_event(symbol, timeframe, side, size, price, order_id, reason)

        audit_logger.log_system_event(
            event_type="TRADE_EXECUTED",
            message=f"Trade executed: {side} {size} {symbol} at {price}",
            details={
                'symbol': symbol,
                'timeframe': timeframe,
                'side': side,
                'size': size,
                'price': price,
                'order_id': order_id,
                'reason': reason
            }
        )

    except Exception as e:
        audit_logger.log_error(
            error_type="TRADE_EVENT_PUBLISH_ERROR",
            message=f"Failed to publish trade event: {str(e)}",
            context={
                'symbol': symbol,
                'side': side,
                'size': size,
                'price': price
            }
        )


async def replay_events_for_analysis(symbol: str, date: str, event_types: Optional[list] = None) -> list:
    """
    Replay stored events for analysis and backtesting.

    This allows reconstructing the complete trading history from events.
    """
    try:
        events = []

        if event_types:
            for event_type in event_types:
                type_events = await event_store.get_events(symbol, date, event_type)
                events.extend(type_events)
        else:
            events = await event_store.get_events(symbol, date)

        # Sort events by timestamp
        events.sort(key=lambda e: e.timestamp)

        return events

    except Exception as e:
        audit_logger.log_error(
            error_type="EVENT_REPLAY_ERROR",
            message=f"Failed to replay events: {str(e)}",
            context={'symbol': symbol, 'date': date}
        )
        return []


def create_event_summary_report(symbol: str, events: list) -> Dict[str, Any]:
    """
    Create a summary report from events for analysis.
    """
    summary = {
        'symbol': symbol,
        'total_events': len(events),
        'event_types': {},
        'signals_generated': 0,
        'trades_executed': 0,
        'risk_assessments': 0,
        'time_range': {}
    }

    if not events:
        return summary

    # Count event types
    for event in events:
        event_type = event.event_type.value
        summary['event_types'][event_type] = summary['event_types'].get(event_type, 0) + 1

        if event.event_type == EventType.SIGNAL_GENERATED:
            summary['signals_generated'] += 1
        elif event.event_type == EventType.TRADE_EXECUTED:
            summary['trades_executed'] += 1
        elif event.event_type == EventType.RISK_ASSESSMENT_COMPLETED:
            summary['risk_assessments'] += 1

    # Time range
    timestamps = [e.timestamp for e in events]
    summary['time_range'] = {
        'start': min(timestamps).isoformat(),
        'end': max(timestamps).isoformat(),
        'duration_hours': (max(timestamps) - min(timestamps)).total_seconds() / 3600
    }

    return summary