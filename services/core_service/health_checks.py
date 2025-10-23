#!/usr/bin/env python3
"""
Health Check System for Services

Provides comprehensive health monitoring including:
- Service availability checks
- System resource monitoring
- Performance metrics collection
- Automated recovery mechanisms
"""

import asyncio
import logging
import time
import psutil
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0

    @property
    def is_healthy(self) -> bool:
        return self.status == HealthStatus.HEALTHY

    @property
    def is_degraded(self) -> bool:
        return self.status == HealthStatus.DEGRADED

    @property
    def is_unhealthy(self) -> bool:
        return self.status == HealthStatus.UNHEALTHY


@dataclass
class HealthCheckConfig:
    """Configuration for health checks"""
    name: str
    check_interval: float = 30.0  # Check every 30 seconds
    timeout: float = 10.0  # 10 second timeout
    failure_threshold: int = 3  # Mark unhealthy after 3 failures
    recovery_threshold: int = 2  # Mark healthy after 2 successes
    enabled: bool = True


class HealthChecker:
    """
    Base class for health check implementations
    """

    def __init__(self, config: HealthCheckConfig):
        self.config = config
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.last_check_time = 0.0
        self.last_result: Optional[HealthCheckResult] = None

    async def check(self) -> HealthCheckResult:
        """Perform health check"""
        start_time = time.time()

        try:
            result = await self._perform_check()
            duration = (time.time() - start_time) * 1000

            result.duration_ms = duration
            self.last_result = result
            self.last_check_time = start_time

            # Update consecutive counters
            if result.is_healthy:
                self.consecutive_successes += 1
                self.consecutive_failures = 0
            else:
                self.consecutive_failures += 1
                self.consecutive_successes = 0

            return result

        except Exception as e:
            duration = (time.time() - start_time) * 1000
            error_result = HealthCheckResult(
                name=self.config.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                details={"error": str(e), "error_type": type(e).__name__},
                duration_ms=duration
            )

            self.consecutive_failures += 1
            self.consecutive_successes = 0
            self.last_result = error_result
            self.last_check_time = start_time

            return error_result

    async def _perform_check(self) -> HealthCheckResult:
        """Override this method to implement specific health checks"""
        raise NotImplementedError("Subclasses must implement _perform_check")

    def get_status(self) -> HealthStatus:
        """Get current health status based on thresholds"""
        if self.consecutive_failures >= self.config.failure_threshold:
            return HealthStatus.UNHEALTHY
        elif self.consecutive_failures > 0:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    def should_check(self) -> bool:
        """Check if it's time to perform another health check"""
        if not self.config.enabled:
            return False

        elapsed = time.time() - self.last_check_time
        return elapsed >= self.config.check_interval


class SystemHealthChecker(HealthChecker):
    """Health checker for system resources"""

    def __init__(self, config: HealthCheckConfig, cpu_threshold: float = 90.0, memory_threshold: float = 90.0):
        super().__init__(config)
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold

    async def _perform_check(self) -> HealthCheckResult:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        details = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "memory_used_gb": memory.used / (1024**3),
            "memory_total_gb": memory.total / (1024**3)
        }

        if cpu_percent > self.cpu_threshold or memory_percent > self.memory_threshold:
            return HealthCheckResult(
                name=self.config.name,
                status=HealthStatus.UNHEALTHY,
                message=f"System resources critical: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%",
                details=details
            )
        elif cpu_percent > self.cpu_threshold * 0.8 or memory_percent > self.memory_threshold * 0.8:
            return HealthCheckResult(
                name=self.config.name,
                status=HealthStatus.DEGRADED,
                message=f"System resources high: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%",
                details=details
            )
        else:
            return HealthCheckResult(
                name=self.config.name,
                status=HealthStatus.HEALTHY,
                message=f"System resources normal: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%",
                details=details
            )


class ServiceHealthChecker(HealthChecker):
    """Health checker for other services via their health endpoints"""

    def __init__(self, config: HealthCheckConfig, service_url: str, service_name: str):
        super().__init__(config)
        self.service_url = service_url
        self.service_name = service_name
        self.health_url = f"{service_url}/health"

    async def _perform_check(self) -> HealthCheckResult:
        try:
            # Simple TCP connection check as fallback
            import socket
            import urllib.parse

            parsed = urllib.parse.urlparse(self.service_url)
            host = parsed.hostname
            port = parsed.port or (443 if parsed.scheme == 'https' else 80)

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.config.timeout)
            result = sock.connect_ex((host, port))
            sock.close()

            if result == 0:
                return HealthCheckResult(
                    name=self.config.name,
                    status=HealthStatus.HEALTHY,
                    message=f"Service {self.service_name} is reachable",
                    details={"service": self.service_name, "host": host, "port": port}
                )
            else:
                return HealthCheckResult(
                    name=self.config.name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Service {self.service_name} is not reachable",
                    details={"service": self.service_name, "host": host, "port": port, "error_code": result}
                )
        except Exception as e:
            return HealthCheckResult(
                name=self.config.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check service {self.service_name}: {str(e)}",
                details={"service": self.service_name, "error": str(e)}
            )


class HealthCheckRegistry:
    """
    Registry for managing multiple health checkers
    """

    def __init__(self):
        self.checkers: Dict[str, HealthChecker] = {}
        self._lock = asyncio.Lock()

    async def register(self, checker: HealthChecker):
        """Register a health checker"""
        async with self._lock:
            self.checkers[checker.config.name] = checker
            logger.info(f"Registered health checker: {checker.config.name}")

    async def unregister(self, name: str):
        """Unregister a health checker"""
        async with self._lock:
            if name in self.checkers:
                del self.checkers[name]
                logger.info(f"Unregistered health checker: {name}")

    async def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks"""
        results = {}

        for name, checker in self.checkers.items():
            if checker.should_check():
                result = await checker.check()
                results[name] = result
            else:
                # Return last result if check not due yet
                if checker.last_result:
                    results[name] = checker.last_result

        return results

    async def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        results = await self.run_all_checks()

        healthy_count = sum(1 for r in results.values() if r.is_healthy)
        degraded_count = sum(1 for r in results.values() if r.is_degraded)
        unhealthy_count = sum(1 for r in results.values() if r.is_unhealthy)

        total_checks = len(results)

        # Determine overall status
        if unhealthy_count > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        return {
            "status": overall_status.value,
            "timestamp": time.time(),
            "checks": {
                "total": total_checks,
                "healthy": healthy_count,
                "degraded": degraded_count,
                "unhealthy": unhealthy_count
            },
            "details": {name: result.__dict__ for name, result in results.items()}
        }

    def get_checker_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all checkers"""
        metrics = {}
        for name, checker in self.checkers.items():
            metrics[name] = {
                "config": checker.config.__dict__,
                "consecutive_failures": checker.consecutive_failures,
                "consecutive_successes": checker.consecutive_successes,
                "last_check_time": checker.last_check_time,
                "current_status": checker.get_status().value,
                "last_result": checker.last_result.__dict__ if checker.last_result else None
            }
        return metrics


# Global registry instance
health_check_registry = HealthCheckRegistry()


async def create_service_health_checks():
    """
    Create and register health checks for all services
    """
    try:
        from config.settings import (
            CORE_SERVICE_URL, DATA_SERVICE_URL, BACKTESTING_SERVICE_URL,
            DISCOVERY_SERVICE_URL, NOTIFICATION_SERVICE_URL, RISK_SERVICE_URL,
            WEB_UI_SERVICE_URL, ML_TRAINER_SERVICE_URL, TRADING_BOT_LIVE_SERVICE_URL
        )

        # Service health checks
        service_checks = [
            ("core_service", CORE_SERVICE_URL),
            ("data_service", DATA_SERVICE_URL),
            ("backtesting_service", BACKTESTING_SERVICE_URL),
            ("discovery_service", DISCOVERY_SERVICE_URL),
            ("notification_service", NOTIFICATION_SERVICE_URL),
            ("risk_service", RISK_SERVICE_URL),
            ("web_ui_service", WEB_UI_SERVICE_URL),
            ("ml_trainer_service", ML_TRAINER_SERVICE_URL),
            ("trading_bot_live_service", TRADING_BOT_LIVE_SERVICE_URL),
        ]

        for service_name, service_url in service_checks:
            if service_url:
                checker = ServiceHealthChecker(
                    config=HealthCheckConfig(
                        name=f"{service_name}_health",
                        check_interval=30.0,
                        timeout=5.0
                    ),
                    service_url=service_url,
                    service_name=service_name
                )
                await health_check_registry.register(checker)

        # System health check
        system_checker = SystemHealthChecker(
            config=HealthCheckConfig(
                name="system_health",
                check_interval=60.0  # Check system resources every minute
            )
        )
        await health_check_registry.register(system_checker)

        logger.info("Service health checks initialized")

    except ImportError:
        logger.warning("Could not import service URLs from config.settings - health checks not initialized")


# Background health check runner
class HealthCheckRunner:
    """Background runner for periodic health checks"""

    def __init__(self, registry: HealthCheckRegistry):
        self.registry = registry
        self.running = False
        self.task: Optional[asyncio.Task] = None

    async def start(self):
        """Start background health checking"""
        if self.running:
            return

        self.running = True
        self.task = asyncio.create_task(self._run_checks())
        logger.info("Health check runner started")

    async def stop(self):
        """Stop background health checking"""
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("Health check runner stopped")

    async def _run_checks(self):
        """Run health checks in background"""
        while self.running:
            try:
                await self.registry.run_all_checks()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Health check runner error: {e}")
                await asyncio.sleep(10)  # Wait before retrying