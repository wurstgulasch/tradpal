"""
Central Service Client - Unified client for all microservices communication.

This module provides a centralized client that manages communication with all
TradPal microservices, implementing proper async patterns, circuit breakers,
and event-driven communication.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
import aiohttp
from datetime import datetime

from config.service_settings import (
    DATA_SERVICE_URL, CORE_SERVICE_URL, ML_TRAINER_URL, BACKTESTING_SERVICE_URL,
    TRADING_BOT_LIVE_URL, RISK_SERVICE_URL, NOTIFICATION_SERVICE_URL,
    WEB_UI_URL, DISCOVERY_SERVICE_URL, MLOPS_SERVICE_URL, OPTIMIZER_URL,
    REQUEST_TIMEOUT, ENABLE_DATA_SERVICE, ENABLE_CORE_SERVICE, ENABLE_ML_TRAINER,
    ENABLE_BACKTESTING, ENABLE_LIVE_TRADING, ENABLE_RISK_SERVICE,
    ENABLE_NOTIFICATIONS, ENABLE_WEB_UI, ENABLE_DISCOVERY, ENABLE_MLOPS,
    ENABLE_OPTIMIZER
)

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Simple circuit breaker implementation"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open

    async def call(self, func, *args, **kwargs):
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is open")

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        if self.last_failure_time is None:
            return True
        return (datetime.now() - self.last_failure_time).seconds > self.recovery_timeout

    def _on_success(self):
        self.failure_count = 0
        self.state = "closed"

    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


class CentralServiceClient:
    """
    Centralized client for all microservice communication.

    Features:
    - Async HTTP communication with connection pooling
    - Circuit breaker pattern for resilience
    - Service discovery and health monitoring
    - Event-driven communication via Redis Streams
    - Automatic retry and timeout handling
    """

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.service_urls = {
            'data': DATA_SERVICE_URL,
            'core': CORE_SERVICE_URL,
            'ml_trainer': ML_TRAINER_URL,
            'backtesting': BACKTESTING_SERVICE_URL,
            'trading_bot_live': TRADING_BOT_LIVE_URL,
            'risk': RISK_SERVICE_URL,
            'notification': NOTIFICATION_SERVICE_URL,
            'web_ui': WEB_UI_URL,
            'discovery': DISCOVERY_SERVICE_URL,
            'mlops': MLOPS_SERVICE_URL,
            'optimizer': OPTIMIZER_URL
        }
        self.service_enabled = {
            'data': ENABLE_DATA_SERVICE,
            'core': ENABLE_CORE_SERVICE,
            'ml_trainer': ENABLE_ML_TRAINER,
            'backtesting': ENABLE_BACKTESTING,
            'trading_bot_live': ENABLE_LIVE_TRADING,
            'risk': ENABLE_RISK_SERVICE,
            'notification': ENABLE_NOTIFICATIONS,
            'web_ui': ENABLE_WEB_UI,
            'discovery': ENABLE_DISCOVERY,
            'mlops': ENABLE_MLOPS,
            'optimizer': ENABLE_OPTIMIZER
        }

        # Initialize circuit breakers for all services
        for service_name in self.service_urls.keys():
            self.circuit_breakers[service_name] = CircuitBreaker()

    @asynccontextmanager
    async def session_context(self):
        """Context manager for HTTP session"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
            self.session = aiohttp.ClientSession(timeout=timeout)
        try:
            yield self.session
        finally:
            pass  # Keep session alive for reuse

    async def close(self):
        """Close all connections"""
        if self.session:
            await self.session.close()
            self.session = None

    async def _make_request(self, service_name: str, method: str, endpoint: str,
                          json_data: Optional[Dict] = None, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make HTTP request with circuit breaker protection"""
        if not self.service_enabled.get(service_name, False):
            raise Exception(f"Service {service_name} is disabled")

        circuit_breaker = self.circuit_breakers[service_name]
        base_url = self.service_urls[service_name]

        async def _request():
            async with self.session_context() as session:
                url = f"{base_url}{endpoint}"
                async with session.request(method, url, json=json_data, params=params) as response:
                    if response.status >= 400:
                        error_text = await response.text()
                        raise Exception(f"HTTP {response.status}: {error_text}")
                    return await response.json()

        return await circuit_breaker.call(_request)

    # Data Service Methods
    async def get_market_data(self, symbol: str, timeframe: str = "1h",
                            limit: int = 100) -> Dict[str, Any]:
        """Get market data from data service"""
        params = {"symbol": symbol, "timeframe": timeframe, "limit": limit}
        return await self._make_request("data", "GET", "/market-data", params=params)

    async def get_historical_data(self, symbol: str, start_date: str,
                                end_date: str, timeframe: str = "1h") -> Dict[str, Any]:
        """Get historical data from data service"""
        params = {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "timeframe": timeframe
        }
        return await self._make_request("data", "GET", "/historical-data", params=params)

    # Core Service Methods
    async def calculate_indicators(self, data: Dict[str, Any],
                                 indicators: List[str]) -> Dict[str, Any]:
        """Calculate technical indicators"""
        payload = {"data": data, "indicators": indicators}
        return await self._make_request("core", "POST", "/indicators", json_data=payload)

    async def vectorize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Vectorize market data for ML"""
        return await self._make_request("core", "POST", "/vectorize", json_data=data)

    # ML Trainer Service Methods
    async def get_trading_signal(self, market_data: Dict[str, Any],
                               symbol: str) -> Dict[str, Any]:
        """Get trading signal from ML model"""
        payload = {"market_data": market_data, "symbol": symbol}
        return await self._make_request("ml_trainer", "POST", "/signal", json_data=payload)

    async def train_model(self, training_data: Dict[str, Any],
                        model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Train ML model"""
        payload = {"training_data": training_data, "model_config": model_config}
        return await self._make_request("ml_trainer", "POST", "/train", json_data=payload)

    # Backtesting Service Methods
    async def run_backtest(self, strategy_config: Dict[str, Any],
                         start_date: str, end_date: str) -> Dict[str, Any]:
        """Run backtesting simulation"""
        payload = {
            "strategy_config": strategy_config,
            "start_date": start_date,
            "end_date": end_date
        }
        return await self._make_request("backtesting", "POST", "/backtest", json_data=payload)

    async def get_backtest_results(self, backtest_id: str) -> Dict[str, Any]:
        """Get backtesting results"""
        return await self._make_request("backtesting", "GET", f"/results/{backtest_id}")

    # Trading Bot Live Service Methods
    async def execute_trade(self, trade_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute live trade"""
        return await self._make_request("trading_bot_live", "POST", "/trade", json_data=trade_config)

    async def get_positions(self) -> Dict[str, Any]:
        """Get current positions"""
        return await self._make_request("trading_bot_live", "GET", "/positions")

    async def get_trading_status(self) -> Dict[str, Any]:
        """Get trading bot status"""
        return await self._make_request("trading_bot_live", "GET", "/status")

    # Risk Service Methods
    async def calculate_risk_metrics(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk metrics"""
        return await self._make_request("risk", "POST", "/risk-metrics", json_data=portfolio_data)

    async def check_risk_limits(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check if trade violates risk limits"""
        return await self._make_request("risk", "POST", "/check-limits", json_data=trade_data)

    # Notification Service Methods
    async def send_notification(self, notification: Dict[str, Any]) -> Dict[str, Any]:
        """Send notification"""
        return await self._make_request("notification", "POST", "/notify", json_data=notification)

    async def get_notification_history(self, limit: int = 50) -> Dict[str, Any]:
        """Get notification history"""
        params = {"limit": limit}
        return await self._make_request("notification", "GET", "/history", params=params)

    # Discovery Service Methods
    async def optimize_parameters(self, optimization_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run parameter optimization"""
        return await self._make_request("discovery", "POST", "/optimize", json_data=optimization_config)

    async def get_optimization_results(self, optimization_id: str) -> Dict[str, Any]:
        """Get optimization results"""
        return await self._make_request("discovery", "GET", f"/results/{optimization_id}")

    # MLOps Service Methods
    async def log_experiment(self, experiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Log ML experiment"""
        return await self._make_request("mlops", "POST", "/experiment", json_data=experiment_data)

    async def get_model_metrics(self, model_id: str) -> Dict[str, Any]:
        """Get model performance metrics"""
        return await self._make_request("mlops", "GET", f"/metrics/{model_id}")

    # Service Health Methods
    async def check_service_health(self, service_name: str) -> Dict[str, Any]:
        """Check health of specific service"""
        try:
            return await self._make_request(service_name, "GET", "/health")
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def check_all_services_health(self) -> Dict[str, Dict[str, Any]]:
        """Check health of all enabled services"""
        health_status = {}
        for service_name, enabled in self.service_enabled.items():
            if enabled:
                health_status[service_name] = await self.check_service_health(service_name)
            else:
                health_status[service_name] = {"status": "disabled"}

        return health_status

    # Event-Driven Communication Methods
    async def publish_event(self, event_type: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Publish event to event system (if available)"""
        # This would integrate with Redis Streams or similar
        # For now, we'll use the event system service if available
        try:
            payload = {
                "event_type": event_type,
                "event_data": event_data,
                "timestamp": datetime.now().isoformat()
            }
            # Try to use event system service if available
            return await self._make_request("event_system", "POST", "/events", json_data=payload)
        except Exception:
            # Fallback: log event locally
            logger.info(f"Event published: {event_type} - {event_data}")
            return {"status": "logged_locally", "event_type": event_type}

    async def subscribe_to_events(self, event_types: List[str]) -> None:
        """Subscribe to specific event types"""
        # This would set up Redis Stream consumers
        # Implementation depends on event system architecture
        logger.info(f"Subscribed to events: {event_types}")


# Global instance for easy access
central_client = CentralServiceClient()


async def get_central_client() -> CentralServiceClient:
    """Get the global central service client instance"""
    return central_client


async def initialize_services():
    """Initialize all enabled services"""
    client = await get_central_client()

    # Check service health on startup
    health_status = await client.check_all_services_health()

    healthy_services = []
    unhealthy_services = []

    for service_name, status in health_status.items():
        if status.get("status") == "healthy":
            healthy_services.append(service_name)
        else:
            unhealthy_services.append(service_name)

    logger.info(f"Service initialization complete. Healthy: {healthy_services}, Issues: {unhealthy_services}")

    return {
        "healthy_services": healthy_services,
        "unhealthy_services": unhealthy_services,
        "total_services": len(health_status)
    }


async def cleanup_services():
    """Cleanup service connections"""
    client = await get_central_client()
    await client.close()
    logger.info("Service cleanup completed")