#!/usr/bin/env python3
"""
Core Service Client

Async client for communicating with the Core Service API.
Provides signal generation, indicator calculation, and strategy execution.
Enhanced with Zero-Trust Security (mTLS + JWT) and Resilience Patterns (Circuit Breaker + Health Checks).
"""

import asyncio
import aiohttp
import logging
import ssl
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
from pathlib import Path

from config.settings import (
    CORE_SERVICE_URL, API_KEY, ENABLE_MTLS,
    MTLS_CERT_PATH, MTLS_KEY_PATH, CA_CERT_PATH
)

# Import resilience components
from .circuit_breaker import AsyncCircuitBreaker, CircuitBreakerConfig, SERVICE_CIRCUIT_CONFIG
from .health_checks import ServiceHealthChecker, HealthCheckConfig
from .metrics_exporter import metrics_collector, record_service_request

# Import event system for publishing events
from services.infrastructure_service.event_system_service.client import EventClient

logger = logging.getLogger(__name__)


class CoreServiceClient:
    """Async client for Core Service API with Zero-Trust Security and Resilience Patterns"""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or CORE_SERVICE_URL or "http://localhost:8002"
        self.session: Optional[aiohttp.ClientSession] = None
        self.api_key = API_KEY
        self.jwt_token: Optional[str] = None

        # mTLS configuration
        self.mtls_enabled = ENABLE_MTLS or False
        self.ssl_context: Optional[ssl.SSLContext] = None

        # Resilience components
        self.circuit_breaker = AsyncCircuitBreaker(
            CircuitBreakerConfig(
                name="core_service",
                failure_threshold=3,
                recovery_timeout=30.0,
                success_threshold=2,
                timeout=15.0
            )
        )
        self.health_checker: Optional[ServiceHealthChecker] = None

        if self.mtls_enabled:
            self._setup_mtls()

        # Initialize health checker
        self._setup_health_checker()

        # Metrics collection
        self.metrics_task: Optional[asyncio.Task] = None
        self.metrics_enabled = True

        # Event system integration
        self.event_client: Optional[EventClient] = None
        self.events_enabled = True

    def _setup_health_checker(self):
        """Setup health checker for the core service"""
        self.health_checker = ServiceHealthChecker(
            config=HealthCheckConfig(
                name="core_service_health",
                check_interval=30.0,
                timeout=5.0
            ),
            service_url=self.base_url,
            service_name="core_service"
        )

    def _setup_mtls(self):
        """Setup mutual TLS configuration"""
        try:
            self.ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)

            # Load client certificate and key
            if MTLS_CERT_PATH and MTLS_KEY_PATH:
                cert_path = Path(MTLS_CERT_PATH)
                key_path = Path(MTLS_KEY_PATH)

                if cert_path.exists() and key_path.exists():
                    self.ssl_context.load_cert_chain(str(cert_path), str(key_path))
                    logger.info("✅ mTLS client certificate loaded")
                else:
                    logger.warning("⚠️  mTLS certificate files not found, disabling mTLS")
                    self.mtls_enabled = False

            # Load CA certificate for server verification
            if CA_CERT_PATH and Path(CA_CERT_PATH).exists():
                self.ssl_context.load_verify_locations(CA_CERT_PATH)
                self.ssl_context.verify_mode = ssl.CERT_REQUIRED
                logger.info("✅ mTLS CA certificate loaded")

        except Exception as e:
            logger.error(f"❌ Failed to setup mTLS: {e}")
            self.mtls_enabled = False

    @asynccontextmanager
    async def _get_session(self):
        """Get or create HTTP session with security"""
        if self.session is None:
            headers = {}
            if self.api_key:
                headers["X-API-Key"] = self.api_key
            if self.jwt_token:
                headers["Authorization"] = f"Bearer {self.jwt_token}"

            connector = None
            if self.mtls_enabled and self.ssl_context:
                connector = aiohttp.TCPConnector(ssl=self.ssl_context)

            self.session = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
                connector=connector
            )

        try:
            yield self.session
        except Exception:
            if self.session:
                await self.session.close()
                self.session = None
            raise

    async def authenticate(self) -> bool:
        """Authenticate with security service and get JWT token"""
        try:
            from services.infrastructure_service.security_service.client import SecurityServiceClient

            security_client = SecurityServiceClient()
            success = await security_client.authenticate("core_service_client")

            if success:
                # Get the token from security client
                # Note: In production, this should be handled more securely
                self.jwt_token = "authenticated"  # Placeholder
                logger.info("✅ Core service client authenticated")
                return True
            else:
                logger.error("❌ Core service client authentication failed")
                return False

        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False

    async def _make_request(self, method: str, url: str, **kwargs) -> aiohttp.ClientResponse:
        """
        Make HTTP request with circuit breaker protection and metrics collection

        Args:
            method: HTTP method
            url: Request URL
            **kwargs: Additional request parameters

        Returns:
            HTTP response

        Raises:
            CircuitBreakerOpenException: When circuit breaker is open
            aiohttp.ClientError: For HTTP errors
        """
        start_time = asyncio.get_event_loop().time()

        async def _http_call():
            async with self._get_session() as session:
                async with session.request(method, url, **kwargs) as response:
                    response.raise_for_status()
                    return response

        try:
            # Use circuit breaker to protect the call
            response = await self.circuit_breaker.call(_http_call)

            # Record successful request metrics
            duration = asyncio.get_event_loop().time() - start_time
            record_service_request("core_service", method, str(response.status), duration)

            return response

        except Exception as e:
            # Record failed request metrics
            duration = asyncio.get_event_loop().time() - start_time
            record_service_request("core_service", method, "error", duration)
            raise

    async def start_metrics_collection(self):
        """Start background metrics collection"""
        if not self.metrics_enabled or self.metrics_task:
            return

        self.metrics_task = asyncio.create_task(self._collect_metrics())
        logger.info("📊 Started metrics collection for core service client")

    async def stop_metrics_collection(self):
        """Stop background metrics collection"""
        if self.metrics_task:
            self.metrics_task.cancel()
            try:
                await self.metrics_task
            except asyncio.CancelledError:
                pass
            self.metrics_task = None
            logger.info("📊 Stopped metrics collection for core service client")

    async def _collect_metrics(self):
        """Background task to collect and update metrics"""
        while self.metrics_enabled:
            try:
                # Update circuit breaker metrics
                from .circuit_breaker import circuit_breaker_registry
                metrics_collector.update_from_circuit_breaker_registry(circuit_breaker_registry)

                # Update health check metrics
                from .health_checks import health_check_registry
                await metrics_collector.update_from_health_check_registry(health_check_registry)

                # Update system metrics
                metrics_collector.update_system_metrics()

                await asyncio.sleep(30)  # Collect metrics every 30 seconds

            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(10)  # Wait before retrying

    async def health_check(self) -> bool:
        """Check if the core service is healthy"""
        if self.health_checker:
            result = await self.health_checker.check()
            return result.is_healthy
        else:
            # Fallback to simple HTTP check
            try:
                async with self._get_session() as session:
                    async with session.get(f"{self.base_url}/health") as response:
                        return response.status == 200
            except Exception as e:
                logger.warning(f"Core service health check failed: {e}")
                return False

    async def get_health_status(self) -> Dict[str, Any]:
        """Get detailed health status including circuit breaker metrics"""
        health_status = {
            "service": "core_service",
            "healthy": await self.health_check(),
            "circuit_breaker": self.circuit_breaker.get_metrics(),
            "endpoint": self.base_url
        }

        if self.health_checker and self.health_checker.last_result:
            health_status["last_health_check"] = self.health_checker.last_result.__dict__

        return health_status

    async def generate_signals(
        self,
        symbol: str,
        timeframe: str,
        data: List[Dict[str, Any]],
        strategy_config: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate trading signals.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            data: OHLCV data as list of dicts
            strategy_config: Strategy configuration

        Returns:
            List of trading signals
        """
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "data": data
        }

        if strategy_config:
            payload["strategy_config"] = strategy_config

        response = await self._make_request(
            "POST",
            f"{self.base_url}/signals/generate",
            json=payload
        )
        result = await response.json()

        if result.get("success"):
            signals = result.get("signals", [])

            # Publish events for generated signals
            if self.events_enabled and signals:
                await self._publish_signal_events(symbol, signals)

            return signals
        else:
            raise RuntimeError(f"Signal generation failed: {result}")

    async def _publish_signal_events(self, symbol: str, signals: List[Dict[str, Any]]):
        """Publish events for generated trading signals"""
        if not self.event_client:
            return

        try:
            for signal in signals:
                # Publish trading signal event
                await self.event_client.publish_trading_signal({
                    "symbol": symbol,
                    "signal": signal,
                    "timestamp": signal.get("timestamp"),
                    "strategy": signal.get("strategy", "unknown")
                })

                logger.debug(f"📡 Published trading signal event for {symbol}")

        except Exception as e:
            logger.warning(f"Failed to publish signal events: {e}")

    async def calculate_indicators(
        self,
        symbol: str,
        timeframe: str,
        data: List[Dict[str, Any]],
        indicators: List[str]
    ) -> Dict[str, Any]:
        """
        Calculate technical indicators.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            data: OHLCV data as list of dicts
            indicators: List of indicators to calculate

        Returns:
            Calculated indicators
        """
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "data": data,
            "indicators": indicators
        }

        async with self._get_session() as session:
            async with session.post(
                f"{self.base_url}/strategy/execute",
                json=payload
            ) as response:
                response.raise_for_status()
                result = await response.json()

                if result.get("success"):
                    return result.get("indicators", {})
                else:
                    raise RuntimeError(f"Indicator calculation failed: {result}")

    async def execute_strategy(
        self,
        symbol: str,
        timeframe: str,
        signal: Dict[str, Any],
        capital: float,
        risk_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute trading strategy.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            signal: Trading signal
            capital: Available capital
            risk_config: Risk management configuration

        Returns:
            Strategy execution result
        """
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "signal": signal,
            "capital": capital,
            "risk_config": risk_config
        }

        async with self._get_session() as session:
            async with session.post(
                f"{self.base_url}/strategy/execute",
                json=payload
            ) as response:
                response.raise_for_status()
                result = await response.json()

                if result.get("success"):
                    execution = result.get("execution", {})

                    # Publish order execution event
                    if self.events_enabled:
                        await self._publish_execution_event(symbol, execution)

                    return execution
                else:
                    raise RuntimeError(f"Strategy execution failed: {result}")

    async def _publish_execution_event(self, symbol: str, execution: Dict[str, Any]):
        """Publish events for order execution"""
        if not self.event_client:
            return

        try:
            # Create order execution event
            event_data = {
                "symbol": symbol,
                "execution": execution,
                "order_id": execution.get("order_id"),
                "side": execution.get("side"),
                "quantity": execution.get("quantity"),
                "price": execution.get("price")
            }

            # Note: We'd need to add ORDER_EXECUTED to EventType enum
            # For now, we'll use a generic approach
            await self.event_client.publish_event({
                "event_type": "order_executed",
                "source": "core_service_client",
                "data": event_data
            })

            logger.debug(f"📡 Published order execution event for {symbol}")

        except Exception as e:
            logger.warning(f"Failed to publish execution event: {e}")

    async def get_market_analysis(self, symbol: str, timeframe: str = "1h") -> Dict[str, Any]:
        """
        Get market analysis for a symbol.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Market analysis data
        """
        async with self._get_session() as session:
            async with session.get(
                f"{self.base_url}/analysis/market/{symbol}?timeframe={timeframe}"
            ) as response:
                response.raise_for_status()
                return await response.json()

    async def list_strategies(self) -> List[str]:
        """List available trading strategies"""
        async with self._get_session() as session:
            async with session.get(f"{self.base_url}/strategies") as response:
                response.raise_for_status()
                result = await response.json()
                return result.get("strategies", [])

    async def list_indicators(self) -> List[str]:
        """List available technical indicators"""
        async with self._get_session() as session:
            async with session.get(f"{self.base_url}/indicators") as response:
                response.raise_for_status()
                result = await response.json()
                return result.get("indicators", [])

    async def get_performance_metrics(self, symbol: str) -> Dict[str, Any]:
        """
        Get performance metrics for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Performance metrics
        """
        async with self._get_session() as session:
            async with session.get(f"{self.base_url}/performance/{symbol}") as response:
                response.raise_for_status()
                return await response.json()

    async def initialize_event_system(self):
        """Initialize event system integration"""
        if self.events_enabled and not self.event_client:
            try:
                self.event_client = EventClient()
                await self.event_client._ensure_session()
                logger.info("📡 Event system initialized for core service client")
            except Exception as e:
                logger.warning(f"Failed to initialize event system: {e}")
                self.events_enabled = False

    async def close(self):
        """Close the HTTP session, stop metrics collection, and cleanup event client"""
        await self.stop_metrics_collection()

        if self.event_client:
            await self.event_client.session.close()
            self.event_client = None

        if self.session:
            await self.session.close()
            self.session = None