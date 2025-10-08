"""
Audit Logger for Trading System

Provides structured logging for trading decisions, signal generation,
risk management calculations, and trade executions with JSON format
and log rotation for compliance and debugging purposes.
"""

import json
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import pandas as pd

from config.settings import LOG_FILE


@dataclass
class SignalDecision:
    """Represents a trading signal decision for audit logging."""
    timestamp: str
    symbol: str
    timeframe: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    confidence_score: float
    reasoning: Dict[str, Any]
    technical_indicators: Dict[str, float]
    risk_metrics: Dict[str, float]
    market_conditions: Dict[str, Any]


@dataclass
class TradeExecution:
    """Represents a trade execution for audit logging."""
    timestamp: str
    symbol: str
    timeframe: str
    side: str  # 'BUY', 'SELL'
    quantity: float
    price: float
    order_type: str
    pnl_realized: Optional[float] = None
    pnl_unrealized: Optional[float] = None
    risk_amount: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    execution_reason: str = ""


@dataclass
class RiskAssessment:
    """Represents a risk assessment for audit logging."""
    timestamp: str
    symbol: str
    timeframe: str
    capital: float
    risk_per_trade: float
    position_size: float
    atr_value: float
    leverage: float
    max_drawdown_limit: float
    assessment_result: str  # 'APPROVED', 'REJECTED', 'MODIFIED'


class AuditLogger:
    """
    Enhanced audit logger for trading system compliance and debugging.

    Features:
    - Structured JSON logging
    - Log rotation (size and time-based)
    - Multiple log levels for different event types
    - Performance metrics tracking
    - Compliance-ready audit trails
    """

    def __init__(self, log_file: str = LOG_FILE, max_bytes: int = 10*1024*1024, backup_count: int = 5):
        """
        Initialize the audit logger.

        Args:
            log_file: Path to the log file
            max_bytes: Maximum log file size before rotation (10MB default)
            backup_count: Number of backup files to keep
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Create logger
        self.logger = logging.getLogger('tradpal_audit')
        self.logger.setLevel(logging.INFO)

        # Remove any existing handlers
        self.logger.handlers.clear()

        # Create rotating file handler
        handler = logging.handlers.RotatingFileHandler(
            self.log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )

        # Create JSON formatter
        formatter = AuditJSONFormatter()
        handler.setFormatter(formatter)

        self.logger.addHandler(handler)

        # Performance tracking
        self.performance_metrics = {
            'signals_generated': 0,
            'trades_executed': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'start_time': datetime.now().isoformat()
        }

    def log_signal_decision(self, decision: SignalDecision):
        """Log a trading signal decision."""
        self.logger.info("SIGNAL_DECISION", extra={
            'event_type': 'SIGNAL_DECISION',
            'data': asdict(decision)
        })
        self.performance_metrics['signals_generated'] += 1

    def log_trade_execution(self, execution: TradeExecution):
        """Log a trade execution."""
        self.logger.info("TRADE_EXECUTION", extra={
            'event_type': 'TRADE_EXECUTION',
            'data': asdict(execution)
        })
        self.performance_metrics['trades_executed'] += 1
        if execution.pnl_realized is not None:
            self.performance_metrics['total_pnl'] += execution.pnl_realized

    def log_risk_assessment(self, assessment: RiskAssessment):
        """Log a risk assessment."""
        self.logger.info("RISK_ASSESSMENT", extra={
            'event_type': 'RISK_ASSESSMENT',
            'data': asdict(assessment)
        })

    def log_system_event(self, event_type: str, message: str, details: Optional[Dict[str, Any]] = None):
        """Log system-level events (startup, shutdown, errors, etc.)."""
        self.logger.info("SYSTEM_EVENT", extra={
            'event_type': 'SYSTEM_EVENT',
            'data': {
                'event_type': event_type,
                'message': message,
                'details': details or {},
                'timestamp': datetime.now().isoformat()
            }
        })

    def log_performance_metrics(self):
        """Log current performance metrics."""
        self.logger.info("PERFORMANCE_METRICS", extra={
            'event_type': 'PERFORMANCE_METRICS',
            'data': {
                **self.performance_metrics,
                'timestamp': datetime.now().isoformat(),
                'uptime_seconds': (datetime.now() - datetime.fromisoformat(self.performance_metrics['start_time'])).total_seconds()
            }
        })

    def log_error(self, error_type: str, message: str, traceback: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        """Log errors with full context."""
        self.logger.error("ERROR", extra={
            'event_type': 'ERROR',
            'data': {
                'error_type': error_type,
                'message': message,
                'traceback': traceback,
                'context': context or {},
                'timestamp': datetime.now().isoformat()
            }
        })

    def get_recent_logs(self, lines: int = 100) -> list:
        """Get recent log entries for analysis."""
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Split by newlines and get last N lines
                all_lines = content.strip().split('\n')
                return all_lines[-lines:] if len(all_lines) >= lines else all_lines
        except FileNotFoundError:
            return []

    def analyze_audit_trail(self, days: int = 7) -> Dict[str, Any]:
        """Analyze audit trail for the specified number of days."""
        # This would parse the JSON logs and provide analytics
        # For now, return basic metrics
        return {
            'period_days': days,
            'total_signals': self.performance_metrics['signals_generated'],
            'total_trades': self.performance_metrics['trades_executed'],
            'total_pnl': self.performance_metrics['total_pnl'],
            'analysis_timestamp': datetime.now().isoformat()
        }


class AuditJSONFormatter(logging.Formatter):
    """Custom JSON formatter for audit logs."""

    def format(self, record):
        # Create base log entry
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'event_type': getattr(record, 'event_type', 'UNKNOWN'),
        }

        # Add data if present
        if hasattr(record, 'data'):
            log_entry['data'] = record.data

        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False, default=str)


# Global audit logger instance
audit_logger = AuditLogger()


def log_signal_buy(symbol: str, timeframe: str, confidence: float, indicators: Dict[str, float],
                   risk_metrics: Dict[str, float], reasoning: str = ""):
    """Convenience function to log a buy signal."""
    decision = SignalDecision(
        timestamp=datetime.now().isoformat(),
        symbol=symbol,
        timeframe=timeframe,
        signal_type='BUY',
        confidence_score=confidence,
        reasoning={'description': reasoning},
        technical_indicators=indicators,
        risk_metrics=risk_metrics,
        market_conditions={}
    )
    audit_logger.log_signal_decision(decision)


def log_signal_sell(symbol: str, timeframe: str, confidence: float, indicators: Dict[str, float],
                    risk_metrics: Dict[str, float], reasoning: str = ""):
    """Convenience function to log a sell signal."""
    decision = SignalDecision(
        timestamp=datetime.now().isoformat(),
        symbol=symbol,
        timeframe=timeframe,
        signal_type='SELL',
        confidence_score=confidence,
        reasoning={'description': reasoning},
        technical_indicators=indicators,
        risk_metrics=risk_metrics,
        market_conditions={}
    )
    audit_logger.log_signal_decision(decision)


def log_trade_buy(symbol: str, timeframe: str, quantity: float, price: float,
                  risk_amount: float, stop_loss: Optional[float] = None,
                  take_profit: Optional[float] = None, reason: str = ""):
    """Convenience function to log a buy trade execution."""
    execution = TradeExecution(
        timestamp=datetime.now().isoformat(),
        symbol=symbol,
        timeframe=timeframe,
        side='BUY',
        quantity=quantity,
        price=price,
        order_type='MARKET',
        risk_amount=risk_amount,
        stop_loss=stop_loss,
        take_profit=take_profit,
        execution_reason=reason
    )
    audit_logger.log_trade_execution(execution)


def log_trade_sell(symbol: str, timeframe: str, quantity: float, price: float,
                   pnl: Optional[float] = None, reason: str = ""):
    """Convenience function to log a sell trade execution."""
    execution = TradeExecution(
        timestamp=datetime.now().isoformat(),
        symbol=symbol,
        timeframe=timeframe,
        side='SELL',
        quantity=quantity,
        price=price,
        order_type='MARKET',
        pnl_realized=pnl,
        execution_reason=reason
    )
    audit_logger.log_trade_execution(execution)