#!/usr/bin/env python3
"""
TradPal Circuit Breaker Service

Implements Circuit Breaker pattern for resilient service communication.
Provides fault tolerance and automatic recovery for service-to-service calls.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Callable, Awaitable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import aiohttp
from aiohttp import ClientTimeout
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global Prometheus metrics registry to avoid duplicates
circuit_breaker_metrics_registry = CollectorRegistry()

# Global metrics instances to avoid duplicates across all circuit breakers
_requests_counter = Counter(
    'circuit_breaker_requests_total',
    'Total requests through circuit breaker',
    ['service', 'state'],
    registry=circuit_breaker_metrics_registry
)

_failures_counter = Counter(
    'circuit_breaker_failures_total',
    'Total failures through circuit breaker',
    ['service'],
    registry=circuit_breaker_metrics_registry
)

_state_gauge = Gauge(
    'circuit_breaker_state',
    'Current state of circuit breaker (0=closed, 1=open, 2=half_open)',
    ['service'],
    registry=circuit_breaker_metrics_registry
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5  # Number of failures before opening
    recovery_timeout: float = 60.0  # Seconds to wait before trying again
    success_threshold: int = 3  # Number of successes needed in half-open state
    timeout: float = 30.0  # Request timeout in seconds
    expected_exception: tuple = (Exception,)  # Exceptions that count as failures
    name: str = "default"  # Circuit breaker name for monitoring


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker"""
    requests_total: int = 0
    failures_total: int = 0
    successes_total: int = 0
    state_changes_total: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None


class CircuitBreaker:
    """
    Circuit Breaker implementation for resilient HTTP calls.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service is failing, requests are blocked
    - HALF_OPEN: Testing if service has recovered
    """

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        self.half_open_successes = 0
        self._lock = asyncio.Lock()

        # Use global Prometheus metrics to avoid duplicates
        self.requests_counter = _requests_counter
        self.failures_counter = _failures_counter
        self.state_gauge = _state_gauge

        logger.info(f"Circuit breaker '{config.name}' initialized in {self.state.value} state")

    async def call(self, func: Callable[[], Awaitable[Any]], *args, **kwargs) -> Any:
        """
        Execute a function call through the circuit breaker.

        Args:
            func: Async function to call
            *args: Positional arguments for the function
            *kwargs: Keyword arguments for the function

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenException: If circuit is open
            Exception: Original exception from the function call
        """
        async with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if not self._should_attempt_reset():
                    self.requests_counter.labels(
                        service=self.config.name,
                        state=self.state.value
                    ).inc()
                    raise CircuitBreakerOpenException(
                        f"Circuit breaker '{self.config.name}' is OPEN"
                    )

                # Transition to half-open for testing
                await self._transition_to_half_open()

            self.metrics.requests_total += 1

        try:
            # Execute the function call
            result = await func(*args, **kwargs)

            # Success - handle based on current state
            await self._on_success()
            return result

        except self.config.expected_exception as e:
            # Failure - handle based on current state
            await self._on_failure()
            raise e

    async def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery"""
        if self.metrics.last_failure_time is None:
            return True

        elapsed = time.time() - self.metrics.last_failure_time
        return elapsed >= self.config.recovery_timeout

    async def _transition_to_half_open(self):
        """Transition from OPEN to HALF_OPEN state"""
        self.state = CircuitBreakerState.HALF_OPEN
        self.half_open_successes = 0
        self.metrics.state_changes_total += 1
        self.state_gauge.labels(service=self.config.name).set(2)

        logger.info(f"Circuit breaker '{self.config.name}' transitioned to HALF_OPEN")

    async def _transition_to_closed(self):
        """Transition to CLOSED state"""
        self.state = CircuitBreakerState.CLOSED
        self.metrics.state_changes_total += 1
        self.state_gauge.labels(service=self.config.name).set(0)

        logger.info(f"Circuit breaker '{self.config.name}' transitioned to CLOSED")

    async def _transition_to_open(self):
        """Transition to OPEN state"""
        self.state = CircuitBreakerState.OPEN
        self.metrics.last_failure_time = time.time()
        self.metrics.state_changes_total += 1
        self.state_gauge.labels(service=self.config.name).set(1)

        logger.warning(f"Circuit breaker '{self.config.name}' transitioned to OPEN")

    async def _on_success(self):
        """Handle successful call"""
        self.metrics.successes_total += 1
        self.metrics.last_success_time = time.time()

        if self.state == CircuitBreakerState.HALF_OPEN:
            self.half_open_successes += 1
            if self.half_open_successes >= self.config.success_threshold:
                await self._transition_to_closed()
        elif self.state == CircuitBreakerState.CLOSED:
            # Reset any failure tracking in closed state
            pass

    async def _on_failure(self):
        """Handle failed call"""
        self.metrics.failures_total += 1
        self.metrics.last_failure_time = time.time()

        self.failures_counter.labels(service=self.config.name).inc()

        if self.state == CircuitBreakerState.HALF_OPEN:
            # Any failure in half-open immediately opens the circuit
            await self._transition_to_open()
        elif self.state == CircuitBreakerState.CLOSED:
            # Check if we've exceeded the failure threshold
            if self.metrics.failures_total >= self.config.failure_threshold:
                await self._transition_to_open()

    def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state"""
        return self.state

    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics"""
        return {
            "state": self.state.value,
            "requests_total": self.metrics.requests_total,
            "failures_total": self.metrics.failures_total,
            "successes_total": self.metrics.successes_total,
            "state_changes_total": self.metrics.state_changes_total,
            "last_failure_time": self.metrics.last_failure_time,
            "last_success_time": self.metrics.last_success_time,
            "half_open_successes": self.half_open_successes,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
                "timeout": self.config.timeout
            }
        }


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open"""
    pass


class HttpCircuitBreaker(CircuitBreaker):
    """
    Circuit breaker specifically for HTTP calls with aiohttp.

    Provides automatic retry logic and timeout handling.
    """

    def __init__(self, config: CircuitBreakerConfig, session: Optional[aiohttp.ClientSession] = None):
        super().__init__(config)
        self.session = session

    async def get(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """HTTP GET request through circuit breaker"""
        return await self.call(self._http_get, url, **kwargs)

    async def post(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """HTTP POST request through circuit breaker"""
        return await self.call(self._http_post, url, **kwargs)

    async def put(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """HTTP PUT request through circuit breaker"""
        return await self.call(self._http_put, url, **kwargs)

    async def delete(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """HTTP DELETE request through circuit breaker"""
        return await self.call(self._http_delete, url, **kwargs)

    async def _http_get(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Execute HTTP GET request"""
        session = self.session or aiohttp.ClientSession(timeout=ClientTimeout(total=self.config.timeout))
        try:
            async with session.get(url, **kwargs) as response:
                # Clone response to return it (since context manager will close it)
                return await self._clone_response(response)
        finally:
            if not self.session:
                await session.close()

    async def _http_post(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Execute HTTP POST request"""
        session = self.session or aiohttp.ClientSession(timeout=ClientTimeout(total=self.config.timeout))
        try:
            async with session.post(url, **kwargs) as response:
                return await self._clone_response(response)
        finally:
            if not self.session:
                await session.close()

    async def _http_put(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Execute HTTP PUT request"""
        session = self.session or aiohttp.ClientSession(timeout=ClientTimeout(total=self.config.timeout))
        try:
            async with session.put(url, **kwargs) as response:
                return await self._clone_response(response)
        finally:
            if not self.session:
                await session.close()

    async def _http_delete(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Execute HTTP DELETE request"""
        session = self.session or aiohttp.ClientSession(timeout=ClientTimeout(total=self.config.timeout))
        try:
            async with session.delete(url, **kwargs) as response:
                return await self._clone_response(response)
        finally:
            if not self.session:
                await session.close()

    async def _clone_response(self, response: aiohttp.ClientResponse) -> aiohttp.ClientResponse:
        """Clone aiohttp response for returning"""
        # Read response content
        content = await response.read()

        # Create a new response-like object
        class ClonedResponse:
            def __init__(self, original_response, content):
                self.status = original_response.status
                self.headers = original_response.headers
                self.content = content
                self.url = original_response.url
                self.method = original_response.method

            async def json(self):
                import json
                return json.loads(self.content.decode('utf-8'))

            async def text(self):
                return self.content.decode('utf-8')

            def raise_for_status(self):
                if self.status >= 400:
                    raise aiohttp.ClientResponseError(
                        self.url, self.status, message=f"HTTP {self.status}"
                    )

        return ClonedResponse(response, content)


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.

    Provides centralized monitoring and management of all circuit breakers.
    """

    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(self, name: str, config: CircuitBreakerConfig) -> CircuitBreaker:
        """Get existing circuit breaker or create new one"""
        async with self._lock:
            if name not in self.breakers:
                self.breakers[name] = CircuitBreaker(config)
            return self.breakers[name]

    def get_all_breakers(self) -> Dict[str, CircuitBreaker]:
        """Get all registered circuit breakers"""
        return self.breakers.copy()

    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all circuit breakers"""
        return {
            name: breaker.get_metrics()
            for name, breaker in self.breakers.items()
        }

    async def reset_all(self):
        """Reset all circuit breakers to closed state"""
        async with self._lock:
            for breaker in self.breakers.values():
                if breaker.state != CircuitBreakerState.CLOSED:
                    await breaker._transition_to_closed()


# Global registry instance
circuit_breaker_registry = CircuitBreakerRegistry()


def get_circuit_breaker_metrics_registry() -> CollectorRegistry:
    """
    Get the circuit breaker metrics registry for testing or monitoring.

    Returns:
        Prometheus CollectorRegistry containing circuit breaker metrics
    """
    return circuit_breaker_metrics_registry


async def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """
    Get or create a circuit breaker instance.

    Args:
        name: Circuit breaker name
        config: Configuration (uses defaults if not provided)

    Returns:
        Circuit breaker instance
    """
    if config is None:
        config = CircuitBreakerConfig(name=name)

    return await circuit_breaker_registry.get_or_create(name, config)


async def get_http_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None,
                                  session: Optional[aiohttp.ClientSession] = None) -> HttpCircuitBreaker:
    """
    Get or create an HTTP circuit breaker instance.

    Args:
        name: Circuit breaker name
        config: Configuration (uses defaults if not provided)
        session: aiohttp session (creates new if not provided)

    Returns:
        HTTP circuit breaker instance
    """
    if config is None:
        config = CircuitBreakerConfig(name=name)

    async with circuit_breaker_registry._lock:
        if name not in circuit_breaker_registry.breakers:
            circuit_breaker_registry.breakers[name] = HttpCircuitBreaker(config, session)
        return circuit_breaker_registry.breakers[name]


# Convenience functions for common service calls
async def call_service_with_circuit_breaker(
    service_name: str,
    func: Callable[[], Awaitable[Any]],
    config: Optional[CircuitBreakerConfig] = None,
    *args,
    **kwargs
) -> Any:
    """
    Call a service function with circuit breaker protection.

    Args:
        service_name: Name of the service for circuit breaker identification
        func: Async function to call
        config: Circuit breaker configuration
        *args: Arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Function result
    """
    breaker = await get_circuit_breaker(service_name, config)
    return await breaker.call(func, *args, **kwargs)


async def http_get_with_circuit_breaker(
    service_name: str,
    url: str,
    config: Optional[CircuitBreakerConfig] = None,
    session: Optional[aiohttp.ClientSession] = None,
    **kwargs
) -> aiohttp.ClientResponse:
    """
    HTTP GET with circuit breaker protection.

    Args:
        service_name: Name of the service
        url: URL to request
        config: Circuit breaker configuration
        session: aiohttp session
        **kwargs: Additional request parameters

    Returns:
        HTTP response
    """
    breaker = await get_http_circuit_breaker(service_name, config, session)
    return await breaker.get(url, **kwargs)


async def http_post_with_circuit_breaker(
    service_name: str,
    url: str,
    config: Optional[CircuitBreakerConfig] = None,
    session: Optional[aiohttp.ClientSession] = None,
    **kwargs
) -> aiohttp.ClientResponse:
    """
    HTTP POST with circuit breaker protection.

    Args:
        service_name: Name of the service
        url: URL to request
        config: Circuit breaker configuration
        session: aiohttp session
        **kwargs: Additional request parameters

    Returns:
        HTTP response
    """
    breaker = await get_http_circuit_breaker(service_name, config, session)
    return await breaker.post(url, **kwargs)


# Export service-specific circuit breaker configurations
SERVICE_CIRCUIT_BREAKER_CONFIGS = {
    'data_service': CircuitBreakerConfig(
        failure_threshold=3,  # More sensitive for data services
        recovery_timeout=30.0,
        success_threshold=2,
        timeout=15.0
    ),
    'core_service': CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout=60.0,
        success_threshold=3,
        timeout=30.0
    ),
    'trading_service': CircuitBreakerConfig(
        failure_threshold=3,  # Critical service, faster recovery
        recovery_timeout=45.0,
        success_threshold=2,
        timeout=20.0
    ),
    'ml_service': CircuitBreakerConfig(
        failure_threshold=7,  # ML services can be more tolerant
        recovery_timeout=120.0,
        success_threshold=5,
        timeout=60.0
    ),
    'notification_service': CircuitBreakerConfig(
        failure_threshold=10,  # Notifications can fail more before opening
        recovery_timeout=300.0,
        success_threshold=3,
        timeout=10.0
    )
}
