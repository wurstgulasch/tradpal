#!/usr/bin/env python3
"""
Core Service - Core trading logic and signal generation.

This service provides:
- Signal generation using technical indicators
- Indicator calculations
- Strategy execution with risk management
- Market analysis and insights
"""

import asyncio
import logging
import logging.handlers
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import functools
import hashlib
import json
import os
import pickle
import psutil
import threading
import time
from pathlib import Path

# Try to import Redis
try:
    import redis
    from redis.connection import ConnectionPool
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Try to import Prometheus
try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from config.settings import (
    SYMBOL, TIMEFRAME, EMA_SHORT, EMA_LONG, RSI_PERIOD,
    RSI_OVERSOLD, RSI_OVERBOUGHT, BB_PERIOD, BB_STD_DEV,
    ATR_PERIOD, RISK_PER_TRADE, INITIAL_CAPITAL,
    LOG_FILE, PERFORMANCE_ENABLED, PARALLEL_PROCESSING_ENABLED,
    VECTORIZATION_ENABLED, MEMORY_OPTIMIZATION_ENABLED,
    PERFORMANCE_MONITORING_ENABLED, MAX_WORKERS, CHUNK_SIZE,
    PERFORMANCE_LOG_LEVEL, REDIS_ENABLED, REDIS_HOST, REDIS_PORT,
    REDIS_DB, REDIS_PASSWORD, REDIS_TTL_INDICATORS, REDIS_TTL_API,
    REDIS_MAX_CONNECTIONS
)

logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """Trading signal data structure."""
    timestamp: datetime
    symbol: str
    timeframe: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    indicators: Dict[str, Any]
    price: float
    reason: str


@dataclass
class StrategyExecution:
    """Strategy execution result."""
    action: str
    symbol: str
    quantity: float
    price: float
    stop_loss: float
    take_profit: float
    risk_amount: float
    timestamp: datetime


class EventSystem:
    """Simple event system for service communication."""

    def __init__(self):
        self.subscribers: Dict[str, List] = {}

    def subscribe(self, event_type: str, callback):
        """Subscribe to an event type."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)

    async def publish(self, event_type: str, data: Dict[str, Any]):
        """Publish an event."""
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                try:
                    await callback(data)
                except Exception as e:
                    logger.error(f"Event callback failed: {e}")


# Integrated Performance Monitoring (from performance.py)
class PerformanceMonitor:
    """Überwacht System-Performance während der Ausführung."""

    def __init__(self):
        self.start_time = None
        self.cpu_percentages = []
        self.memory_usages = []
        self.monitoring = False
        self.monitor_thread = None

    def start_monitoring(self):
        """Starte Performance-Monitoring."""
        if not PERFORMANCE_MONITORING_ENABLED:
            return

        self.start_time = time.time()
        self.cpu_percentages = []
        self.memory_usages = []
        self.monitoring = True

        # Starte Monitoring-Thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Performance-Monitoring gestartet")

    def stop_monitoring(self) -> Dict[str, Any]:
        """Stoppe Performance-Monitoring und erstelle Bericht."""
        if not self.monitoring:
            return {}

        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)

        end_time = time.time()
        total_duration = end_time - (self.start_time or end_time)

        # Berechne Statistiken
        report = {
            'total_duration': total_duration,
            'avg_cpu_percent': np.mean(self.cpu_percentages) if self.cpu_percentages else 0,
            'max_cpu_percent': np.max(self.cpu_percentages) if self.cpu_percentages else 0,
            'avg_memory_mb': np.mean(self.memory_usages) if self.memory_usages else 0,
            'max_memory_mb': np.max(self.memory_usages) if self.memory_usages else 0,
            'samples_collected': len(self.cpu_percentages)
        }

        logger.info(f"Performance-Monitoring beendet: {report}")
        return report

    def _monitor_loop(self):
        """Monitoring-Schleife für System-Metriken."""
        while self.monitoring:
            try:
                # CPU-Auslastung messen
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.cpu_percentages.append(cpu_percent)

                # Speichernutzung messen
                memory = psutil.virtual_memory()
                memory_mb = memory.used / 1024 / 1024
                self.memory_usages.append(memory_mb)

                time.sleep(0.5)  # Alle 0.5 Sekunden messen

            except Exception as e:
                logger.warning(f"Fehler beim Performance-Monitoring: {e}")
                break


# Integrated Audit Logging (from audit_logger.py)
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


class AuditLogger:
    """
    Enhanced audit logger for trading system compliance and debugging.
    """

    def __init__(self, log_file: str = LOG_FILE, max_bytes: int = 10*1024*1024, backup_count: int = 5):
        """
        Initialize the audit logger.
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Create logger
        self.audit_logger = logging.getLogger('tradpal_audit')
        self.audit_logger.setLevel(logging.INFO)

        # Remove any existing handlers
        self.audit_logger.handlers.clear()

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

        self.audit_logger.addHandler(handler)

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
        self.audit_logger.info("SIGNAL_DECISION", extra={
            'event_type': 'SIGNAL_DECISION',
            'data': asdict(decision)
        })
        self.performance_metrics['signals_generated'] += 1

    def log_trade_execution(self, execution: TradeExecution):
        """Log a trade execution."""
        self.audit_logger.info("TRADE_EXECUTION", extra={
            'event_type': 'TRADE_EXECUTION',
            'data': asdict(execution)
        })
        self.performance_metrics['trades_executed'] += 1
        if execution.pnl_realized is not None:
            self.performance_metrics['total_pnl'] += execution.pnl_realized

    def log_system_event(self, event_type: str, message: str, details: Optional[Dict[str, Any]] = None):
        """Log system-level events."""
        self.audit_logger.info("SYSTEM_EVENT", extra={
            'event_type': 'SYSTEM_EVENT',
            'data': {
                'event_type': event_type,
                'message': message,
                'details': details or {},
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


# Integrated Caching (from cache.py)
class Cache:
    """Simple file-based cache with TTL support."""

    def __init__(self, cache_dir: str = "cache", ttl_seconds: int = 3600):
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_seconds
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_path(self, key: str) -> str:
        """Generate cache file path for a key."""
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{safe_key}.pkl")

    def _is_expired(self, cache_path: str) -> bool:
        """Check if cache file is expired."""
        if not os.path.exists(cache_path):
            return True

        file_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        return datetime.now() - file_time > timedelta(seconds=self.ttl_seconds)

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if it exists and is not expired."""
        cache_path = self._get_cache_path(key)

        if self._is_expired(cache_path):
            return None

        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except (FileNotFoundError, pickle.PickleError):
            return None

    def set(self, key: str, value: Any) -> None:
        """Store value in cache."""
        cache_path = self._get_cache_path(key)

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
        except (OSError, pickle.PickleError):
            pass

    def clear(self) -> None:
        """Clear all cache files."""
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl'):
                try:
                    os.remove(os.path.join(self.cache_dir, filename))
                except OSError:
                    pass


class HybridCache:
    """
    Hybrid cache that uses Redis for distributed caching with file-based fallback.
    """

    def __init__(self, cache_dir: str = "cache", ttl_seconds: int = 3600):
        self.ttl_seconds = ttl_seconds

        # Initialize both cache backends
        self.redis_cache = None
        if REDIS_ENABLED and REDIS_AVAILABLE:
            try:
                pool = ConnectionPool(
                    host=REDIS_HOST,
                    port=REDIS_PORT,
                    db=REDIS_DB,
                    password=REDIS_PASSWORD,
                    max_connections=REDIS_MAX_CONNECTIONS,
                    decode_responses=False
                )
                redis_client = redis.Redis(connection_pool=pool)
                redis_client.ping()  # Test connection
                self.redis_cache = redis_client
                logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")

        self.file_cache = Cache(cache_dir=cache_dir, ttl_seconds=ttl_seconds)
        self.use_redis = self.redis_cache is not None

        if self.use_redis:
            logger.info("Using Redis cache for distributed caching")
        else:
            logger.info("Using file-based cache (Redis not available or disabled)")

    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage."""
        if isinstance(value, pd.DataFrame):
            return b"PDF:" + pickle.dumps(value)
        else:
            try:
                json_str = json.dumps(value, default=str)
                return b"JSON:" + json_str.encode("utf-8")
            except (TypeError, ValueError):
                return b"PICKLE:" + pickle.dumps(value)

    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        try:
            if data.startswith(b"PDF:"):
                return pickle.loads(data[4:])
            elif data.startswith(b"JSON:"):
                json_str = data[5:].decode("utf-8")
                return json.loads(json_str)
            elif data.startswith(b"PICKLE:"):
                return pickle.loads(data[7:])
            else:
                return pickle.loads(data)
        except Exception as e:
            logger.error(f"Failed to deserialize value: {e}")
            return None

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if self.use_redis:
            data = self.redis_cache.get(key)
            if data:
                return self._deserialize_value(data)

        return self.file_cache.get(key)

    def set(self, key: str, value: Any) -> None:
        """Store value in cache."""
        if self.use_redis:
            try:
                serialized = self._serialize_value(value)
                self.redis_cache.setex(key, self.ttl_seconds, serialized)
            except Exception as e:
                logger.error(f"Error setting value in Redis: {e}")

        self.file_cache.set(key, value)

    def clear(self) -> None:
        """Clear all cache entries."""
        if self.use_redis:
            try:
                keys = self.redis_cache.keys("*")
                if keys:
                    self.redis_cache.delete(*keys)
            except Exception as e:
                logger.error(f"Error clearing Redis cache: {e}")

        self.file_cache.clear()

class CoreService:
    """Core trading logic service with integrated performance monitoring, audit logging, and caching."""

    def __init__(self, event_system: Optional[EventSystem] = None):
        self.event_system = event_system or EventSystem()
        self.active_strategies: Dict[str, Dict[str, Any]] = {}
        self.performance_metrics: Dict[str, Dict[str, Any]] = {}

        # Available indicators
        self.available_indicators = [
            'ema', 'rsi', 'bb', 'atr', 'adx', 'macd', 'obv', 'stochastic'
        ]

        # Available strategies
        self.available_strategies_list = [
            'ema_crossover', 'rsi_divergence', 'bb_reversal', 'trend_following'
        ]

        # Initialize integrated components
        self.performance_monitor = PerformanceMonitor()
        self.audit_logger = AuditLogger()
        self.indicator_cache = HybridCache(cache_dir="cache/indicators", ttl_seconds=REDIS_TTL_INDICATORS)
        self.api_cache = HybridCache(cache_dir="cache/api", ttl_seconds=REDIS_TTL_API)

    async def health_check(self) -> Dict[str, Any]:
        """Health check for the service."""
        cache_stats = self.get_cache_stats()

        return {
            "service": "core",
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "active_strategies": len(self.active_strategies),
            "indicators_available": len(self.available_indicators),
            "performance_monitoring": self.performance_monitor.monitoring,
            "cache_stats": cache_stats,
            "audit_logs_count": self.get_recent_audit_logs(10)
        }

    async def calculate_indicators(
        self,
        symbol: str,
        timeframe: str,
        data: pd.DataFrame,
        indicators: List[str]
    ) -> Dict[str, Any]:
        """
        Calculate technical indicators for market data with caching support.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            data: OHLCV DataFrame
            indicators: List of indicators to calculate

        Returns:
            Dictionary of calculated indicators
        """
        try:
            # Create cache key
            cache_key = f"{symbol}_{timeframe}_{'_'.join(sorted(indicators))}_{hash(str(data.values.tobytes())[:100]):x}"
            cached_result = self.get_cached_indicator(cache_key)

            if cached_result is not None:
                logger.info(f"Using cached indicators for {symbol} {timeframe}")
                return cached_result

            results = {}

            for indicator in indicators:
                if indicator.lower() not in self.available_indicators:
                    raise ValueError(f"Unsupported indicator: {indicator}. Available indicators: {self.available_indicators}")

                if indicator.lower() == 'ema':
                    results['ema_short'] = self._calculate_ema(data, EMA_SHORT)
                    results['ema_long'] = self._calculate_ema(data, EMA_LONG)
                    results['ema_crossover'] = results['ema_short'] > results['ema_long']

                elif indicator.lower() == 'rsi':
                    results['rsi'] = self._calculate_rsi(data, RSI_PERIOD)
                    results['rsi_oversold'] = results['rsi'] < RSI_OVERSOLD
                    results['rsi_overbought'] = results['rsi'] > RSI_OVERBOUGHT

                elif indicator.lower() == 'bb':
                    bb_data = self._calculate_bollinger_bands(data, BB_PERIOD, BB_STD_DEV)
                    results.update(bb_data)

                elif indicator.lower() == 'atr':
                    results['atr'] = self._calculate_atr(data['high'], data['low'], data['close'], ATR_PERIOD)

                elif indicator.lower() == 'adx':
                    results['adx'] = self._calculate_adx(data)

                elif indicator.lower() == 'macd':
                    macd_data = self._calculate_macd(data)
                    results.update(macd_data)

                elif indicator.lower() == 'obv':
                    results['obv'] = self._calculate_obv(data)

                elif indicator.lower() == 'stochastic':
                    stoch_data = self._calculate_stochastic(data)
                    results.update(stoch_data)

            # Cache the results
            self.cache_indicator_result(cache_key, results)

            return results

        except Exception as e:
            logger.error(f"Indicator calculation failed: {e}")
            raise

    async def generate_signals(
        self,
        symbol: str,
        timeframe: str,
        data: pd.DataFrame,
        strategy_config: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate trading signals based on market data and strategy.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            data: OHLCV DataFrame
            strategy_config: Strategy configuration

        Returns:
            List of trading signals
        """
        try:
            signals = []

            # Calculate base indicators
            indicators = await self.calculate_indicators(
                symbol, timeframe, data,
                ['ema', 'rsi', 'bb', 'atr']
            )

            # Generate signals based on strategy
            strategy = strategy_config.get('strategy', 'ema_crossover') if strategy_config else 'ema_crossover'

            if strategy == 'ema_crossover':
                signals.extend(self._generate_ema_crossover_signals(symbol, timeframe, data, indicators))
            elif strategy == 'rsi_divergence':
                signals.extend(self._generate_rsi_signals(symbol, timeframe, data, indicators))
            elif strategy == 'bb_reversal':
                signals.extend(self._generate_bb_signals(symbol, timeframe, data, indicators))

            # Convert to dict format
            signal_dicts = []
            for signal in signals:
                signal_dict = asdict(signal)
                signal_dict['timestamp'] = signal.timestamp.isoformat()
                signal_dicts.append(signal_dict)

                # Log signal decision
                self.log_signal_decision(
                    symbol=symbol,
                    timeframe=timeframe,
                    signal_type=signal.action,
                    confidence=signal.confidence,
                    indicators=signal.indicators,
                    risk_metrics={'atr': signal.indicators.get('atr', 1.0)},
                    reasoning=signal.reason
                )

            # Publish event
            await self.event_system.publish("core.signal_generated", {
                "symbol": symbol,
                "signals_count": len(signal_dicts)
            })

            return signal_dicts

        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            raise

    async def execute_strategy(
        self,
        symbol: str,
        timeframe: str,
        signal: Dict[str, Any],
        capital: float,
        risk_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute trading strategy based on signal and risk parameters.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            signal: Trading signal
            capital: Available capital
            risk_config: Risk management configuration

        Returns:
            Strategy execution details
        """
        try:
            # Calculate position size based on risk
            risk_per_trade = risk_config.get('risk_per_trade', RISK_PER_TRADE)
            risk_amount = capital * risk_per_trade

            # Get ATR for stop loss calculation
            atr_value = signal.get('indicators', {}).get('atr', 1.0)
            sl_multiplier = risk_config.get('sl_multiplier', 1.5)
            tp_multiplier = risk_config.get('tp_multiplier', 3.0)

            current_price = signal['price']
            action = signal['action']

            if action == 'BUY':
                stop_loss = current_price - (atr_value * sl_multiplier)
                take_profit = current_price + (atr_value * tp_multiplier)
            elif action == 'SELL':
                stop_loss = current_price + (atr_value * sl_multiplier)
                take_profit = current_price - (atr_value * tp_multiplier)
            else:
                return {"action": "HOLD", "reason": "No clear signal"}

            # Calculate position size
            risk_per_unit = abs(current_price - stop_loss)
            if risk_per_unit > 0:
                quantity = risk_amount / risk_per_unit
            else:
                quantity = 0

            execution = StrategyExecution(
                action=action,
                symbol=symbol,
                quantity=quantity,
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_amount=risk_amount,
                timestamp=datetime.now()
            )

            # Publish event
            await self.event_system.publish("core.strategy_executed", {
                "symbol": symbol,
                "action": action,
                "quantity": quantity
            })

            # Log trade execution
            self.log_trade_execution(
                symbol=symbol,
                timeframe=timeframe,
                side=action,
                quantity=quantity,
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=f"Strategy execution: {signal.get('reason', 'N/A')}"
            )

            return asdict(execution)

        except Exception as e:
            logger.error(f"Strategy execution failed: {e}")
            raise

    async def get_market_analysis(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Get comprehensive market analysis."""
        # This would integrate with data service in a real implementation
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "trend": "bullish",  # Placeholder
            "volatility": "medium",  # Placeholder
            "momentum": "strong",  # Placeholder
            "support_levels": [45000, 44000],  # Placeholder
            "resistance_levels": [48000, 49000]  # Placeholder
        }

    def get_available_strategies(self) -> List[str]:
        """Get list of available strategies."""
        return self.available_strategies_list

    def get_available_indicators(self) -> List[str]:
        """Get list of available indicators."""
        return self.available_indicators

    async def get_performance_metrics(self, symbol: str) -> Dict[str, Any]:
        """Get performance metrics for a symbol."""
        # Get current performance report
        performance_report = self.performance_monitor.stop_monitoring()
        self.performance_monitor.start_monitoring()  # Restart monitoring

        # Get audit logger metrics
        audit_metrics = {
            'signals_generated': self.audit_logger.performance_metrics['signals_generated'],
            'trades_executed': self.audit_logger.performance_metrics['trades_executed'],
            'total_pnl': self.audit_logger.performance_metrics['total_pnl']
        }

        return {
            "symbol": symbol,
            "performance_report": performance_report,
            "audit_metrics": audit_metrics,
            "cache_stats": self.get_cache_stats()
        }

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        indicator_files = len([f for f in os.listdir("cache/indicators") if f.endswith('.pkl')]) if os.path.exists("cache/indicators") else 0
        api_files = len([f for f in os.listdir("cache/api") if f.endswith('.pkl')]) if os.path.exists("cache/api") else 0

        return {
            'indicator_cache_size': indicator_files,
            'api_cache_size': api_files,
            'redis_enabled': self.indicator_cache.use_redis
        }

    def start_performance_monitoring(self) -> bool:
        """Start performance monitoring."""
        try:
            self.performance_monitor.start_monitoring()
            return True
        except Exception as e:
            logger.error(f"Failed to start performance monitoring: {e}")
            return False

    def stop_performance_monitoring(self) -> Dict[str, Any]:
        """Stop performance monitoring and get report."""
        return self.performance_monitor.stop_monitoring()

    def log_signal_decision(self, symbol: str, timeframe: str, signal_type: str,
                           confidence: float, indicators: Dict[str, float],
                           risk_metrics: Dict[str, float], reasoning: str = ""):
        """Log a trading signal decision."""
        decision = SignalDecision(
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            timeframe=timeframe,
            signal_type=signal_type,
            confidence_score=confidence,
            reasoning={'description': reasoning},
            technical_indicators=indicators,
            risk_metrics=risk_metrics,
            market_conditions={}
        )
        self.audit_logger.log_signal_decision(decision)

    def log_trade_execution(self, symbol: str, timeframe: str, side: str,
                           quantity: float, price: float, pnl: Optional[float] = None,
                           stop_loss: Optional[float] = None, take_profit: Optional[float] = None,
                           reason: str = ""):
        """Log a trade execution."""
        execution = TradeExecution(
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            timeframe=timeframe,
            side=side,
            quantity=quantity,
            price=price,
            order_type='MARKET',
            pnl_realized=pnl,
            stop_loss=stop_loss,
            take_profit=take_profit,
            execution_reason=reason
        )
        self.audit_logger.log_trade_execution(execution)

    def log_system_event(self, event_type: str, message: str, details: Optional[Dict[str, Any]] = None):
        """Log a system event."""
        self.audit_logger.log_system_event(event_type, message, details)

    def get_recent_audit_logs(self, lines: int = 100) -> list:
        """Get recent audit log entries."""
        return self.audit_logger.get_recent_logs(lines)

    def cache_indicator_result(self, key: str, result: Any) -> None:
        """Cache indicator calculation result."""
        self.indicator_cache.set(key, result)

    def get_cached_indicator(self, key: str) -> Optional[Any]:
        """Get cached indicator result."""
        return self.indicator_cache.get(key)

    def cache_api_result(self, key: str, result: Any) -> None:
        """Cache API call result."""
        self.api_cache.set(key, result)

    def get_cached_api_result(self, key: str) -> Optional[Any]:
        """Get cached API result."""
        return self.api_cache.get(key)

    def clear_all_caches(self) -> None:
        """Clear all caches."""
        self.indicator_cache.clear()
        self.api_cache.clear()

    # Technical Indicator Calculation Methods (from indicators.py)
    def _calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average.
        Uses TA-Lib if available, otherwise pandas implementation.
        """
        if len(series) < period:
            return pd.Series([np.nan] * len(series), index=series.index)

        # Try TA-Lib first
        try:
            import talib
            values = series.values.astype(float)
            ema_values = talib.EMA(values, timeperiod=period)
            return pd.Series(ema_values, index=series.index)
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"TA-Lib EMA failed, using pandas: {e}")

        # Pandas implementation (fallback)
        return series.ewm(span=period, adjust=False).mean()

    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.
        Uses TA-Lib if available, otherwise pandas implementation.
        """
        if len(series) < period + 1:
            return pd.Series([np.nan] * len(series), index=series.index)

        # Try TA-Lib first
        try:
            import talib
            values = series.values.astype(float)
            rsi_values = talib.RSI(values, timeperiod=period)
            return pd.Series(rsi_values, index=series.index)
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"TA-Lib RSI failed, using pandas: {e}")

        # Pandas implementation (fallback)
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()

        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.where(loss != 0, np.nan)

        return rsi

    def _calculate_bollinger_bands(self, series: pd.Series, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands.
        Returns dict with upper, middle, lower bands.
        """
        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)

        return {
            'bb_upper': upper,
            'bb_middle': sma,
            'bb_lower': lower
        }

    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range.
        Uses TA-Lib if available, otherwise pandas implementation.
        """
        # Try TA-Lib first
        try:
            import talib
            high_values = high.values.astype(float)
            low_values = low.values.astype(float)
            close_values = close.values.astype(float)
            atr_values = talib.ATR(high_values, low_values, close_values, timeperiod=period)
            return pd.Series(atr_values, index=high.index)
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"TA-Lib ATR failed, using pandas: {e}")

        # Pandas implementation (fallback)
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def _calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Average Directional Index.
        Uses TA-Lib if available, otherwise pandas implementation.
        """
        # Try TA-Lib first
        try:
            import talib
            high_values = high.values.astype(float)
            low_values = low.values.astype(float)
            close_values = close.values.astype(float)
            adx_values = talib.ADX(high_values, low_values, close_values, timeperiod=period)
            return pd.Series(adx_values, index=high.index)
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"TA-Lib ADX failed, using pandas: {e}")

        # Pandas implementation (fallback)
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        dm_plus = pd.Series(np.where((high - high.shift(1)) > (low.shift(1) - low),
                                     np.maximum(high - high.shift(1), 0), 0), index=high.index)
        dm_minus = pd.Series(np.where((low.shift(1) - low) > (high - high.shift(1)),
                                      np.maximum(low.shift(1) - low, 0), 0), index=low.index)

        di_plus = 100 * (dm_plus.rolling(window=period).mean() / atr)
        di_minus = 100 * (dm_minus.rolling(window=period).mean() / atr)

        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window=period).mean()

        return adx

    def _calculate_macd(self, series: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        Returns dict with macd_line, signal_line, histogram.
        """
        # Try TA-Lib first
        try:
            import talib
            values = series.values.astype(float)
            macd_line, signal_line, histogram = talib.MACD(values, fastperiod=fast_period,
                                                          slowperiod=slow_period, signalperiod=signal_period)
            return {
                'macd_line': pd.Series(macd_line, index=series.index),
                'macd_signal': pd.Series(signal_line, index=series.index),
                'macd_histogram': pd.Series(histogram, index=series.index)
            }
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"TA-Lib MACD failed, using pandas: {e}")

        # Pandas implementation (fallback)
        fast_ema = self._calculate_ema(series, fast_period)
        slow_ema = self._calculate_ema(series, slow_period)
        macd_line = fast_ema - slow_ema
        signal_line = self._calculate_ema(macd_line, signal_period)
        histogram = macd_line - signal_line

        return {
            'macd_line': macd_line,
            'macd_signal': signal_line,
            'macd_histogram': histogram
        }

    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate On-Balance Volume.
        Uses TA-Lib if available, otherwise pandas implementation.
        """
        # Try TA-Lib first
        try:
            import talib
            close_values = close.values.astype(float)
            volume_values = volume.values.astype(float)
            obv_values = talib.OBV(close_values, volume_values)
            return pd.Series(obv_values, index=close.index)
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"TA-Lib OBV failed, using pandas: {e}")

        # Pandas implementation (fallback)
        obv_series = pd.Series(0, index=close.index)
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv_series.iloc[i] = obv_series.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv_series.iloc[i] = obv_series.iloc[i-1] - volume.iloc[i]
            else:
                obv_series.iloc[i] = obv_series.iloc[i-1]

        return obv_series

    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """
        Calculate Stochastic Oscillator.
        Returns dict with stoch_k, stoch_d.
        """
        # Try TA-Lib first
        try:
            import talib
            high_values = high.values.astype(float)
            low_values = low.values.astype(float)
            close_values = close.values.astype(float)
            k_values, d_values = talib.STOCH(high_values, low_values, close_values,
                                           fastk_period=k_period, slowk_period=d_period, slowd_period=d_period)
            return {
                'stoch_k': pd.Series(k_values, index=high.index),
                'stoch_d': pd.Series(d_values, index=high.index)
            }
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"TA-Lib STOCH failed, using pandas: {e}")

        # Pandas implementation (fallback)
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()

        return {
            'stoch_k': k_percent,
            'stoch_d': d_percent
        }

    def optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage.

        Args:
            df: DataFrame to optimize

        Returns:
            Optimized DataFrame
        """
        if not MEMORY_OPTIMIZATION_ENABLED:
            return df

        try:
            optimized_df = df.copy()

            # Optimize numeric columns
            for col in optimized_df.select_dtypes(include=[np.number]).columns:
                col_min = optimized_df[col].min()
                col_max = optimized_df[col].max()

                # Integer optimization
                if optimized_df[col].dtype == 'int64':
                    if col_min >= 0:
                        if col_max < 2**8:
                            optimized_df[col] = optimized_df[col].astype('uint8')
                        elif col_max < 2**16:
                            optimized_df[col] = optimized_df[col].astype('uint16')
                        elif col_max < 2**32:
                            optimized_df[col] = optimized_df[col].astype('uint32')
                    else:
                        if col_min > -2**7 and col_max < 2**7:
                            optimized_df[col] = optimized_df[col].astype('int8')
                        elif col_min > -2**15 and col_max < 2**15:
                            optimized_df[col] = optimized_df[col].astype('int16')
                        elif col_min > -2**31 and col_max < 2**31:
                            optimized_df[col] = optimized_df[col].astype('int32')

                # Float optimization
                elif optimized_df[col].dtype == 'float64':
                    if (optimized_df[col] % 1 == 0).all():
                        if col_min >= 0 and col_max < 2**32:
                            optimized_df[col] = optimized_df[col].astype('uint32')
                        else:
                            optimized_df[col] = optimized_df[col].astype('int32')
                    else:
                        optimized_df[col] = optimized_df[col].astype('float32')

            # Optimize categorical columns
            for col in optimized_df.select_dtypes(include=['object']).columns:
                if optimized_df[col].nunique() / len(optimized_df) < 0.5:
                    optimized_df[col] = optimized_df[col].astype('category')

            logger.info(f"DataFrame memory optimized: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB -> {optimized_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            return optimized_df

        except Exception as e:
            logger.warning(f"Memory optimization failed: {e}")
            return df

    # Signal Generation Methods (from signal_generator.py)
    def _generate_ema_crossover_signals(self, symbol: str, timeframe: str, data: pd.DataFrame, indicators: Dict[str, Any]) -> List[TradingSignal]:
        """Generate signals based on EMA crossover strategy."""
        signals = []

        try:
            ema_short = indicators.get('ema_short', pd.Series([np.nan] * len(data)))
            ema_long = indicators.get('ema_long', pd.Series([np.nan] * len(data)))
            close_prices = data['close']

            # EMA crossover logic
            crossover_up = (ema_short > ema_long) & (ema_short.shift(1) <= ema_long.shift(1))
            crossover_down = (ema_short < ema_long) & (ema_short.shift(1) >= ema_long.shift(1))

            # Generate buy signals
            buy_mask = crossover_up & ema_short.notna() & ema_long.notna()
            for idx in buy_mask[buy_mask].index:
                signals.append(TradingSignal(
                    timestamp=idx if isinstance(idx, pd.Timestamp) else pd.Timestamp.now(),
                    symbol=symbol,
                    timeframe=timeframe,
                    action='BUY',
                    confidence=0.7,
                    indicators={
                        'ema_short': ema_short.loc[idx],
                        'ema_long': ema_long.loc[idx],
                        'crossover': 'bullish'
                    },
                    price=close_prices.loc[idx],
                    reason='EMA crossover bullish'
                ))

            # Generate sell signals
            sell_mask = crossover_down & ema_short.notna() & ema_long.notna()
            for idx in sell_mask[sell_mask].index:
                signals.append(TradingSignal(
                    timestamp=idx if isinstance(idx, pd.Timestamp) else pd.Timestamp.now(),
                    symbol=symbol,
                    timeframe=timeframe,
                    action='SELL',
                    confidence=0.7,
                    indicators={
                        'ema_short': ema_short.loc[idx],
                        'ema_long': ema_long.loc[idx],
                        'crossover': 'bearish'
                    },
                    price=close_prices.loc[idx],
                    reason='EMA crossover bearish'
                ))

        except Exception as e:
            logger.error(f"EMA crossover signal generation failed: {e}")

        return signals

    def _generate_rsi_signals(self, symbol: str, timeframe: str, data: pd.DataFrame, indicators: Dict[str, Any]) -> List[TradingSignal]:
        """Generate signals based on RSI strategy."""
        signals = []

        try:
            rsi = indicators.get('rsi', pd.Series([np.nan] * len(data)))
            close_prices = data['close']

            # RSI thresholds
            rsi_oversold = 30
            rsi_overbought = 70

            # Generate buy signals (RSI oversold)
            buy_mask = (rsi < rsi_oversold) & rsi.notna()
            for idx in buy_mask[buy_mask].index:
                signals.append(TradingSignal(
                    timestamp=idx if isinstance(idx, pd.Timestamp) else pd.Timestamp.now(),
                    symbol=symbol,
                    timeframe=timeframe,
                    action='BUY',
                    confidence=0.6,
                    indicators={'rsi': rsi.loc[idx]},
                    price=close_prices.loc[idx],
                    reason=f'RSI oversold ({rsi.loc[idx]:.2f})'
                ))

            # Generate sell signals (RSI overbought)
            sell_mask = (rsi > rsi_overbought) & rsi.notna()
            for idx in sell_mask[sell_mask].index:
                signals.append(TradingSignal(
                    timestamp=idx if isinstance(idx, pd.Timestamp) else pd.Timestamp.now(),
                    symbol=symbol,
                    timeframe=timeframe,
                    action='SELL',
                    confidence=0.6,
                    indicators={'rsi': rsi.loc[idx]},
                    price=close_prices.loc[idx],
                    reason=f'RSI overbought ({rsi.loc[idx]:.2f})'
                ))

        except Exception as e:
            logger.error(f"RSI signal generation failed: {e}")

        return signals

    def _generate_bb_signals(self, symbol: str, timeframe: str, data: pd.DataFrame, indicators: Dict[str, Any]) -> List[TradingSignal]:
        """Generate signals based on Bollinger Bands strategy."""
        signals = []

        try:
            bb_upper = indicators.get('bb_upper', pd.Series([np.nan] * len(data)))
            bb_middle = indicators.get('bb_middle', pd.Series([np.nan] * len(data)))
            bb_lower = indicators.get('bb_lower', pd.Series([np.nan] * len(data)))
            close_prices = data['close']

            # Bollinger Band signals
            # Buy when price touches lower band
            buy_mask = (close_prices <= bb_lower) & bb_lower.notna()
            for idx in buy_mask[buy_mask].index:
                signals.append(TradingSignal(
                    timestamp=idx if isinstance(idx, pd.Timestamp) else pd.Timestamp.now(),
                    symbol=symbol,
                    timeframe=timeframe,
                    action='BUY',
                    confidence=0.65,
                    indicators={
                        'bb_lower': bb_lower.loc[idx],
                        'bb_middle': bb_middle.loc[idx],
                        'bb_upper': bb_upper.loc[idx]
                    },
                    price=close_prices.loc[idx],
                    reason='Price touched lower Bollinger Band'
                ))

            # Sell when price touches upper band
            sell_mask = (close_prices >= bb_upper) & bb_upper.notna()
            for idx in sell_mask[sell_mask].index:
                signals.append(TradingSignal(
                    timestamp=idx if isinstance(idx, pd.Timestamp) else pd.Timestamp.now(),
                    symbol=symbol,
                    timeframe=timeframe,
                    action='SELL',
                    confidence=0.65,
                    indicators={
                        'bb_lower': bb_lower.loc[idx],
                        'bb_middle': bb_middle.loc[idx],
                        'bb_upper': bb_upper.loc[idx]
                    },
                    price=close_prices.loc[idx],
                    reason='Price touched upper Bollinger Band'
                ))

        except Exception as e:
            logger.error(f"Bollinger Bands signal generation failed: {e}")

        return signals