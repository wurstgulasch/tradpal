"""
Error handling and recovery utilities for the trading indicator system.
Provides comprehensive error boundaries, recovery mechanisms, and graceful degradation.
"""

import functools
import time
import logging
from typing import Callable, Any, Optional, Dict, List
from dataclasses import dataclass
from enum import Enum
import traceback


logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better classification."""
    NETWORK = "network"
    DATA = "data"
    CALCULATION = "calculation"
    CONFIGURATION = "configuration"
    VALIDATION = "validation"
    SYSTEM = "system"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for errors."""
    operation: str
    parameters: Dict[str, Any]
    timestamp: float
    retry_count: int = 0
    max_retries: int = 3
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    category: ErrorCategory = ErrorCategory.UNKNOWN


class TradingError(Exception):
    """Base exception for trading system errors."""

    def __init__(self, message: str, context: Optional[ErrorContext] = None,
                 original_error: Optional[Exception] = None):
        super().__init__(message)
        self.message = message
        self.context = context
        self.original_error = original_error

    def __str__(self):
        context_str = f" [{self.context.operation}]" if self.context else ""
        return f"{self.message}{context_str}"


class NetworkError(TradingError):
    """Network-related errors."""
    pass


class DataError(TradingError):
    """Data processing errors."""
    pass


class CalculationError(TradingError):
    """Mathematical calculation errors."""
    pass


class ConfigurationError(TradingError):
    """Configuration-related errors."""
    pass


class ValidationError(TradingError):
    """Input validation errors."""
    pass


class APIError(NetworkError):
    """API-related errors."""
    pass


class RateLimitError(NetworkError):
    """Rate limiting errors."""
    pass


class ErrorHandler:
    """Centralized error handling and recovery."""

    def __init__(self):
        self.error_history: List[Dict[str, Any]] = []
        self.recovery_strategies: Dict[ErrorCategory, Callable] = {
            ErrorCategory.NETWORK: self._recover_network_error,
            ErrorCategory.DATA: self._recover_data_error,
            ErrorCategory.CALCULATION: self._recover_calculation_error,
            ErrorCategory.CONFIGURATION: self._recover_configuration_error,
            ErrorCategory.VALIDATION: self._recover_validation_error,
        }

    def handle_error(self, error: Exception, context: Optional[ErrorContext] = None) -> Any:
        """
        Handle an error with appropriate logging, classification, and recovery.

        Args:
            error: The exception that occurred
            context: Additional context about the error

        Returns:
            Recovery result or re-raises the error if unrecoverable
        """
        # Classify the error
        error_category = self._classify_error(error)
        error_severity = self._assess_severity(error, error_category)

        # Create or update context
        if context is None:
            context = ErrorContext(
                operation="unknown",
                parameters={},
                timestamp=time.time(),
                severity=error_severity,
                category=error_category
            )
        else:
            context.severity = error_severity
            context.category = error_category

        # Log the error
        self._log_error(error, context)

        # Store in history
        self._store_error(error, context)

        # Attempt recovery
        recovery_result = self._attempt_recovery(error, context)

        if recovery_result is not None:
            logger.info(f"Successfully recovered from error: {error.__class__.__name__}")
            return recovery_result

        # If recovery failed, re-raise with enhanced context
        if isinstance(error, TradingError):
            raise error
        else:
            # Wrap non-trading errors
            trading_error = self._wrap_error(error, context)
            raise trading_error

    def _classify_error(self, error: Exception) -> ErrorCategory:
        """Classify an error into a category."""
        error_type = type(error).__name__
        error_msg = str(error).lower()

        # Network errors - check error type first for specific exceptions
        if error_type in ['RateLimitExceeded', 'RequestTimeout', 'NetworkError', 'DDoSProtection']:
            return ErrorCategory.NETWORK

        # Network errors - check message keywords
        if any(keyword in error_msg for keyword in ['connection', 'timeout', 'network', 'http', 'api', 'rate limit']):
            return ErrorCategory.NETWORK

        # Data errors
        if any(keyword in error_msg for keyword in ['data', 'dataframe', 'nan', 'missing', 'invalid']):
            return ErrorCategory.DATA

        # Calculation errors
        if any(keyword in error_msg for keyword in ['calculation', 'math', 'division', 'overflow']):
            return ErrorCategory.CALCULATION

        # Configuration errors
        if any(keyword in error_msg for keyword in ['config', 'setting', 'parameter']):
            return ErrorCategory.CONFIGURATION

        # Validation errors
        if any(keyword in error_msg for keyword in ['validation', 'invalid input', 'type error']):
            return ErrorCategory.VALIDATION

        # System errors
        if any(keyword in error_msg for keyword in ['memory', 'disk', 'system', 'io']):
            return ErrorCategory.SYSTEM

        return ErrorCategory.UNKNOWN

    def _assess_severity(self, error: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Assess the severity of an error."""
        # Critical errors
        if category in [ErrorCategory.SYSTEM]:
            return ErrorSeverity.CRITICAL

        # High severity
        if category in [ErrorCategory.NETWORK, ErrorCategory.DATA]:
            return ErrorSeverity.HIGH

        # Medium severity (default)
        return ErrorSeverity.MEDIUM

    def _log_error(self, error: Exception, context: ErrorContext):
        """Log an error with appropriate level."""
        log_data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'operation': context.operation,
            'severity': context.severity.value,
            'category': context.category.value,
            'retry_count': context.retry_count,
            'timestamp': context.timestamp
        }

        if context.severity == ErrorSeverity.CRITICAL:
            logger.critical("Critical error occurred", extra=log_data)
        elif context.severity == ErrorSeverity.HIGH:
            logger.error("High severity error occurred", extra=log_data)
        elif context.severity == ErrorSeverity.MEDIUM:
            logger.warning("Medium severity error occurred", extra=log_data)
        else:
            logger.info("Low severity error occurred", extra=log_data)

        # Log full traceback for debugging
        logger.debug(f"Error traceback: {traceback.format_exc()}")

    def _store_error(self, error: Exception, context: ErrorContext):
        """Store error in history for analysis."""
        error_record = {
            'timestamp': context.timestamp,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'operation': context.operation,
            'severity': context.severity.value,
            'category': context.category.value,
            'retry_count': context.retry_count,
            'parameters': context.parameters
        }

        self.error_history.append(error_record)

        # Keep only last 1000 errors
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]

    def _attempt_recovery(self, error: Exception, context: ErrorContext) -> Any:
        """Attempt to recover from an error."""
        if context.retry_count >= context.max_retries:
            logger.warning(f"Max retries ({context.max_retries}) exceeded for {context.operation}")
            return None

        recovery_strategy = self.recovery_strategies.get(context.category)
        if recovery_strategy:
            try:
                context.retry_count += 1
                logger.info(f"Attempting recovery (attempt {context.retry_count}) for {context.category.value} error")
                return recovery_strategy(error, context)
            except Exception as recovery_error:
                logger.warning(f"Recovery attempt failed: {recovery_error}")
                return None

        return None

    def _recover_network_error(self, error: Exception, context: ErrorContext) -> Any:
        """Recover from network errors with exponential backoff."""
        if context.retry_count == 0:
            # First retry immediately
            time.sleep(0.1)
        else:
            # Exponential backoff
            delay = min(2 ** context.retry_count, 30)  # Max 30 seconds
            logger.info(f"Network error recovery: waiting {delay} seconds")
            time.sleep(delay)

        # For network errors, we typically want to retry the operation
        # The calling code should handle the actual retry
        return "retry"

    def _recover_data_error(self, error: Exception, context: ErrorContext) -> Any:
        """Recover from data errors."""
        # For data errors, try to provide fallback data or skip the operation
        if "fallback_data" in context.parameters:
            logger.info("Using fallback data for recovery")
            return context.parameters["fallback_data"]

        return None

    def _recover_calculation_error(self, error: Exception, context: ErrorContext) -> Any:
        """Recover from calculation errors."""
        # For calculation errors, return NaN or default values
        logger.info("Returning NaN for calculation error recovery")
        return float('nan')

    def _recover_configuration_error(self, error: Exception, context: ErrorContext) -> Any:
        """Recover from configuration errors."""
        # For config errors, try to use default values
        if "default_value" in context.parameters:
            logger.info("Using default value for configuration error recovery")
            return context.parameters["default_value"]

        return None

    def _recover_validation_error(self, error: Exception, context: ErrorContext) -> Any:
        """Recover from validation errors."""
        # For validation errors, try to sanitize or provide default values
        if "default_value" in context.parameters:
            logger.info("Using default value for validation error recovery")
            return context.parameters["default_value"]

        return None

    def _wrap_error(self, error: Exception, context: ErrorContext) -> TradingError:
        """Wrap a non-trading error in an appropriate TradingError."""
        category = context.category

        if category == ErrorCategory.NETWORK:
            return NetworkError(f"Network error: {error}", context, error)
        elif category == ErrorCategory.DATA:
            return DataError(f"Data error: {error}", context, error)
        elif category == ErrorCategory.CALCULATION:
            return CalculationError(f"Calculation error: {error}", context, error)
        elif category == ErrorCategory.CONFIGURATION:
            return ConfigurationError(f"Configuration error: {error}", context, error)
        elif category == ErrorCategory.VALIDATION:
            return ValidationError(f"Validation error: {error}", context, error)
        else:
            return TradingError(f"System error: {error}", context, error)

    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        if not self.error_history:
            return {"total_errors": 0}

        df = pd.DataFrame(self.error_history)

        stats = {
            "total_errors": len(df),
            "errors_by_category": df['category'].value_counts().to_dict(),
            "errors_by_severity": df['severity'].value_counts().to_dict(),
            "recent_errors": df.tail(10).to_dict('records')
        }

        return stats


# Global error handler instance
error_handler = ErrorHandler()


def error_boundary(operation: str = None, max_retries: int = 3,
                  fallback: Any = None, log_errors: bool = True):
    """
    Decorator for adding error boundaries to functions.

    Args:
        operation: Name of the operation for logging
        max_retries: Maximum number of retries
        fallback: Fallback value to return on error
        log_errors: Whether to log errors
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation or f"{func.__module__}.{func.__name__}"

            context = ErrorContext(
                operation=op_name,
                parameters={"args": args, "kwargs": kwargs},
                timestamp=time.time(),
                max_retries=max_retries
            )

            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    try:
                        result = error_handler.handle_error(e, context)
                        if result is not None:
                            return result
                    except Exception as handler_error:
                        logger.error(f"Error handler failed: {handler_error}")

                # If we reach here, recovery failed
                if fallback is not None:
                    logger.warning(f"Using fallback value for {op_name}")
                    return fallback

                # Re-raise the original error
                raise

        return wrapper
    return decorator


def graceful_degradation(supported_features: List[str] = None):
    """
    Decorator for graceful degradation when features fail.

    Args:
        supported_features: List of features that can be disabled
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Feature {func.__name__} failed, disabling: {e}")

                # Could implement feature toggling here
                # For now, just log and continue

                # Return a no-op result or None
                return None

        return wrapper
    return decorator


def handle_api_errors(func: Callable) -> Callable:
    """
    Decorator for handling API errors with automatic retry and rate limiting.

    Args:
        func: The function to decorate

    Returns:
        Decorated function with error handling
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        context = ErrorContext(
            operation=f"{func.__module__}.{func.__name__}",
            parameters={"args": args, "kwargs": kwargs},
            timestamp=time.time(),
            max_retries=3
        )

        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Check if it's a rate limit error
            if "rate limit" in str(e).lower() or "429" in str(e):
                raise RateLimitError(f"Rate limit exceeded: {e}", context, e)
            else:
                raise APIError(f"API error: {e}", context, e)

    return wrapper


# Import pandas here to avoid circular imports
import pandas as pd