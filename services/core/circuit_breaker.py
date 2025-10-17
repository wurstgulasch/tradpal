#!/usr/bin/env python3
"""
Circuit Breaker Pattern Implementation

Provides resilience for service-to-service communication by:
- Detecting failures and opening circuit to prevent cascading failures
- Allowing limited requests through when in half-open state for recovery testing
- Automatically closing circuit when service recovers
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Any, Callable, Optional, Dict, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior"""
    failure_threshold: int = 5  # Failures before opening
    recovery_timeout: float = 60.0  # Seconds to wait before trying recovery
    expected_exception: tuple = (Exception,)  # Exceptions that count as failures
    success_threshold: int = 3  # Successes needed to close circuit in half-open
    timeout: float = 30.0  # Request timeout
    name: str = "default"  # Circuit breaker name for logging


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker monitoring"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    state_changes: List[Dict[str, Any]] = field(default_factory=list)


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open"""
    pass


class AsyncCircuitBreaker:
    """
    Asynchronous Circuit Breaker implementation

    Prevents cascading failures by temporarily stopping calls to failing services.
    """

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        self._lock = asyncio.Lock()

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection

        Args:
            func: Async function to call
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenException: When circuit is open
        """
        async with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if not self._should_attempt_reset():
                    self.metrics.total_requests += 1
                    raise CircuitBreakerOpenException(
                        f"Circuit breaker '{self.config.name}' is OPEN"
                    )
                else:
                    self._transition_to_half_open()

            self.metrics.total_requests += 1

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout
            )

            await self._on_success()
            return result

        except self.config.expected_exception as e:
            await self._on_failure()
            raise e
        except asyncio.TimeoutError as e:
            await self._on_failure()
            raise e

    async def _on_success(self):
        """Handle successful call"""
        async with self._lock:
            self.metrics.successful_requests += 1
            self.metrics.consecutive_successes += 1
            self.metrics.consecutive_failures = 0
            self.metrics.last_success_time = time.time()

            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.metrics.consecutive_successes >= self.config.success_threshold:
                    self._transition_to_closed()

    async def _on_failure(self):
        """Handle failed call"""
        async with self._lock:
            self.metrics.failed_requests += 1
            self.metrics.consecutive_failures += 1
            self.metrics.consecutive_successes = 0
            self.metrics.last_failure_time = time.time()

            if self.state == CircuitBreakerState.CLOSED:
                if self.metrics.consecutive_failures >= self.config.failure_threshold:
                    self._transition_to_open()
            elif self.state == CircuitBreakerState.HALF_OPEN:
                self._transition_to_open()

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit"""
        if self.metrics.last_failure_time is None:
            return True

        elapsed = time.time() - self.metrics.last_failure_time
        return elapsed >= self.config.recovery_timeout

    def _transition_to_open(self):
        """Transition to OPEN state"""
        old_state = self.state
        self.state = CircuitBreakerState.OPEN
        self._log_state_change(old_state, self.state, "Circuit opened due to failures")

    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state"""
        old_state = self.state
        self.state = CircuitBreakerState.HALF_OPEN
        self.metrics.consecutive_successes = 0
        self._log_state_change(old_state, self.state, "Circuit half-opened for recovery test")

    def _transition_to_closed(self):
        """Transition to CLOSED state"""
        old_state = self.state
        self.state = CircuitBreakerState.CLOSED
        self.metrics.consecutive_failures = 0
        self._log_state_change(old_state, self.state, "Circuit closed - service recovered")

    def _log_state_change(self, old_state: CircuitBreakerState, new_state: CircuitBreakerState, reason: str):
        """Log state change with metrics"""
        change = {
            "timestamp": time.time(),
            "from_state": old_state.value,
            "to_state": new_state.value,
            "reason": reason,
            "consecutive_failures": self.metrics.consecutive_failures,
            "total_requests": self.metrics.total_requests
        }
        self.metrics.state_changes.append(change)

        logger.info(
            f"Circuit Breaker '{self.config.name}': {old_state.value} -> {new_state.value} "
            f"({reason}) - Failures: {self.metrics.consecutive_failures}"
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get current circuit breaker metrics"""
        return {
            "name": self.config.name,
            "state": self.state.value,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
                "timeout": self.config.timeout
            },
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "failed_requests": self.metrics.failed_requests,
                "consecutive_failures": self.metrics.consecutive_failures,
                "consecutive_successes": self.metrics.consecutive_successes,
                "last_failure_time": self.metrics.last_failure_time,
                "last_success_time": self.metrics.last_success_time,
                "success_rate": (
                    self.metrics.successful_requests / self.metrics.total_requests
                    if self.metrics.total_requests > 0 else 0
                )
            },
            "recent_state_changes": self.metrics.state_changes[-5:]  # Last 5 changes
        }


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers
    """

    def __init__(self):
        self.breakers: Dict[str, AsyncCircuitBreaker] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(self, name: str, config: CircuitBreakerConfig) -> AsyncCircuitBreaker:
        """Get existing circuit breaker or create new one"""
        async with self._lock:
            if name not in self.breakers:
                self.breakers[name] = AsyncCircuitBreaker(config)
            return self.breakers[name]

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all circuit breakers"""
        return {name: breaker.get_metrics() for name, breaker in self.breakers.items()}


# Global registry instance
circuit_breaker_registry = CircuitBreakerRegistry()


async def with_circuit_breaker(
    name: str,
    config: CircuitBreakerConfig,
    func: Callable,
    *args,
    **kwargs
) -> Any:
    """
    Convenience function to call a function with circuit breaker protection

    Args:
        name: Circuit breaker name
        config: Circuit breaker configuration
        func: Function to call
        *args: Function arguments
        **kwargs: Function keyword arguments

    Returns:
        Function result
    """
    breaker = await circuit_breaker_registry.get_or_create(name, config)
    return await breaker.call(func, *args, **kwargs)


# Default circuit breaker configurations
SERVICE_CIRCUIT_CONFIG = CircuitBreakerConfig(
    name="service_call",
    failure_threshold=5,  # More tolerant for service calls
    recovery_timeout=60.0,  # Longer recovery time
    success_threshold=3,  # Need 3 successes
    timeout=30.0,  # 30 second timeout
    expected_exception=(Exception,)
)