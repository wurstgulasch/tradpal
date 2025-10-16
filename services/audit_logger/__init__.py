"""
Audit Logger Service - Comprehensive audit logging for trading activities.

Provides structured logging for signals, trades, risk assessments, and system events.
"""

import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict
import pandas as pd

from config.settings import LOG_FILE, SYMBOL, TIMEFRAME

logger = logging.getLogger(__name__)


@dataclass
class SignalDecision:
    """Signal decision audit record."""
    timestamp: str
    symbol: str
    timeframe: str
    action: str
    confidence: float
    indicators: Dict[str, float]
    reasoning: str


@dataclass
class TradeExecution:
    """Trade execution audit record."""
    timestamp: str
    symbol: str
    timeframe: str
    action: str
    quantity: float
    price: float
    order_type: str = "market"


@dataclass
class RiskAssessment:
    """Risk assessment audit record."""
    timestamp: str
    symbol: str
    timeframe: str
    capital: float
    risk_per_trade: float
    position_size: float
    atr_value: float
    leverage: float
    max_drawdown_limit: float
    assessment_result: str


class StructuredJSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add extra fields if present
        if hasattr(record, 'extra_data'):
            log_entry.update(record.extra_data)

        return json.dumps(log_entry, default=str)


class TradingLogger:
    """Specialized logger for trading activities."""

    def __init__(self, name: str = "trading_audit"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Create logs directory if it doesn't exist
        log_dir = Path(LOG_FILE).parent
        log_dir.mkdir(exist_ok=True)

        # File handler with JSON formatting
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setFormatter(StructuredJSONFormatter())
        self.logger.addHandler(file_handler)

        # Also add to root logger to avoid duplicate messages
        self.logger.propagate = False

    def log_signal(self, signal_data: Dict[str, Any]):
        """Log signal decision."""
        self.logger.info("Signal decision", extra={
            'extra_data': {
                'event_type': 'signal_decision',
                'data': signal_data
            }
        })

    def log_trade(self, trade_data: Dict[str, Any]):
        """Log trade execution."""
        self.logger.info("Trade execution", extra={
            'extra_data': {
                'event_type': 'trade_execution',
                'data': trade_data
            }
        })

    def log_risk_assessment(self, assessment: RiskAssessment):
        """Log risk assessment."""
        self.logger.info("Risk assessment", extra={
            'extra_data': {
                'event_type': 'risk_assessment',
                'data': asdict(assessment)
            }
        })

    def log_error(self, error_data: Dict[str, Any]):
        """Log error event."""
        self.logger.error("Error occurred", extra={
            'extra_data': {
                'event_type': 'error',
                'data': error_data
            }
        })

    def log_system_event(self, event_data: Dict[str, Any]):
        """Log system event."""
        self.logger.info("System event", extra={
            'extra_data': {
                'event_type': 'system_event',
                'data': event_data
            }
        })

    def get_recent_logs(self, lines: int = 100) -> List[Dict[str, Any]]:
        """Get recent log entries."""
        try:
            log_file = Path(LOG_FILE)
            if not log_file.exists():
                return []

            with open(log_file, 'r') as f:
                lines_content = f.readlines()[-lines:]

            logs = []
            for line in lines_content:
                try:
                    logs.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

            return logs
        except Exception as e:
            logger.error(f"Failed to read logs: {e}")
            return []

    def analyze_audit_trail(self, days: int = 7) -> Dict[str, Any]:
        """Analyze audit trail for the specified number of days."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_logs = self.get_recent_logs(1000)

            # Filter by date
            filtered_logs = []
            for log in recent_logs:
                try:
                    log_date = datetime.fromisoformat(log['timestamp'])
                    if log_date >= cutoff_date:
                        filtered_logs.append(log)
                except:
                    continue

            # Analyze different event types
            analysis = {
                'total_events': len(filtered_logs),
                'signal_decisions': 0,
                'trade_executions': 0,
                'errors': 0,
                'system_events': 0,
                'period_days': days
            }

            for log in filtered_logs:
                event_type = log.get('extra_data', {}).get('event_type', '')
                if event_type == 'signal_decision':
                    analysis['signal_decisions'] += 1
                elif event_type == 'trade_execution':
                    analysis['trade_executions'] += 1
                elif event_type == 'error':
                    analysis['errors'] += 1
                elif event_type == 'system_event':
                    analysis['system_events'] += 1

            return analysis
        except Exception as e:
            logger.error(f"Failed to analyze audit trail: {e}")
            return {'error': str(e)}


# Global instances
trading_logger = TradingLogger()
audit_logger = trading_logger  # Alias for backward compatibility


def log_signal_buy(symbol: str = SYMBOL,
                  timeframe: str = TIMEFRAME,
                  confidence: float = 0.0,
                  indicators: Dict[str, float] = None,
                  risk_metrics: Dict[str, float] = None,
                  reasoning: str = ""):
    """Log buy signal decision."""
    signal_data = {
        'timestamp': datetime.now().isoformat(),
        'symbol': symbol,
        'timeframe': timeframe,
        'action': 'BUY',
        'confidence': confidence,
        'indicators': indicators or {},
        'risk_metrics': risk_metrics or {},
        'reasoning': reasoning
    }
    audit_logger.log_signal(signal_data)


def log_signal_sell(symbol: str = SYMBOL,
                   timeframe: str = TIMEFRAME,
                   confidence: float = 0.0,
                   indicators: Dict[str, float] = None,
                   risk_metrics: Dict[str, float] = None,
                   reasoning: str = ""):
    """Log sell signal decision."""
    signal_data = {
        'timestamp': datetime.now().isoformat(),
        'symbol': symbol,
        'timeframe': timeframe,
        'action': 'SELL',
        'confidence': confidence,
        'indicators': indicators or {},
        'risk_metrics': risk_metrics or {},
        'reasoning': reasoning
    }
    audit_logger.log_signal(signal_data)


def log_trade_buy(symbol: str = SYMBOL,
                 timeframe: str = TIMEFRAME,
                 quantity: float = 0.0,
                 price: float = 0.0,
                 risk_amount: float = 0.0,
                 stop_loss: float = 0.0,
                 take_profit: float = 0.0,
                 reason: str = ""):
    """Log buy trade execution."""
    trade_data = {
        'timestamp': datetime.now().isoformat(),
        'symbol': symbol,
        'timeframe': timeframe,
        'action': 'BUY',
        'quantity': quantity,
        'price': price,
        'risk_amount': risk_amount,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'reason': reason
    }
    audit_logger.log_trade(trade_data)


def log_trade_sell(symbol: str = SYMBOL,
                  timeframe: str = TIMEFRAME,
                  quantity: float = 0.0,
                  price: float = 0.0,
                  pnl: float = 0.0,
                  reason: str = ""):
    """Log sell trade execution."""
    trade_data = {
        'timestamp': datetime.now().isoformat(),
        'symbol': symbol,
        'timeframe': timeframe,
        'action': 'SELL',
        'quantity': quantity,
        'price': price,
        'pnl': pnl,
        'reason': reason
    }
    audit_logger.log_trade(trade_data)


def log_system_event(event_type: str, message: str, details: Dict[str, Any] = None):
    """Log system event."""
    event_data = {
        'timestamp': datetime.now().isoformat(),
        'event_type': event_type,
        'message': message,
        'details': details or {}
    }
    audit_logger.log_system_event(event_data)


def log_error(error_type: str, message: str, traceback: str = "", context: Dict[str, Any] = None):
    """Log error event."""
    error_data = {
        'timestamp': datetime.now().isoformat(),
        'error_type': error_type,
        'message': message,
        'traceback': traceback,
        'context': context or {}
    }
    audit_logger.log_error(error_data)


def log_api_call(method: str, endpoint: str, status: str, duration: float = 0.0, details: Dict[str, Any] = None):
    """Log API call."""
    api_data = {
        'timestamp': datetime.now().isoformat(),
        'method': method,
        'endpoint': endpoint,
        'status': status,
        'duration': duration,
        'details': details or {}
    }
    audit_logger.log_system_event({
        'event_type': 'api_call',
        'message': f"API call: {method} {endpoint} - {status}",
        'details': api_data
    })


def log_performance_metrics(metrics: Dict[str, Any] = None):
    """Log performance metrics."""
    metrics_data = {
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics or {},
        'cpu_percent': 0.0,  # Would be populated by actual monitoring
        'memory_mb': 0.0,
        'active_threads': 0
    }
    audit_logger.log_system_event({
        'event_type': 'performance_metrics',
        'message': "Performance metrics logged",
        'details': metrics_data
    })


def get_recent_logs(lines: int = 100) -> List[Dict[str, Any]]:
    """Get recent log entries."""
    return audit_logger.get_recent_logs(lines)


def analyze_audit_trail(days: int = 7) -> Dict[str, Any]:
    """Analyze audit trail for the specified number of days."""
    return audit_logger.analyze_audit_trail(days)