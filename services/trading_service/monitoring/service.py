"""
TradPal Trading Service - Monitoring
Simplified implementation for unified service consolidation
"""

import logging
from typing import Dict, Any, Optional, List
import asyncio
from datetime import datetime
import psutil

logger = logging.getLogger(__name__)


class MonitoringService:
    """Simplified monitoring service for trading operations"""

    def __init__(self, event_system=None):
        self.event_system = event_system
        self.metrics: Dict[str, Any] = {}
        self.alerts: List[Dict[str, Any]] = []
        self.is_initialized = False

    async def initialize(self):
        """Initialize the monitoring service"""
        logger.info("Initializing Monitoring Service...")
        self.is_initialized = True
        logger.info("Monitoring Service initialized")

    async def shutdown(self):
        """Shutdown the monitoring service"""
        logger.info("Monitoring Service shut down")
        self.is_initialized = False

    async def record_metric(self, metric_name: str, value: float, tags: Dict[str, Any] = None, timestamp: datetime = None) -> None:
        """Record a metric"""
        if not self.is_initialized:
            return

        if timestamp is None:
            timestamp = datetime.now()

        metric = {
            "name": metric_name,
            "value": value,
            "timestamp": timestamp.isoformat()
        }

        if metric_name not in self.metrics:
            self.metrics[metric_name] = []

        self.metrics[metric_name].append(metric)

    def create_alert(self, alert_type: str, message: str, severity: str) -> str:
        """Create an alert"""
        if not self.is_initialized:
            return ""

        alert_id = f"alert_{len(self.alerts)}"
        alert = {
            "alert_id": alert_id,
            "type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": datetime.now().isoformat(),
            "acknowledged": False
        }

        self.alerts.append(alert)
        return alert_id

    def get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics"""
        if not self.is_initialized:
            return {"status": "not_initialized"}

        try:
            # Try to get system metrics, fallback if psutil not available
            cpu_usage = psutil.cpu_percent(interval=1) if psutil else 0.0
            memory = psutil.virtual_memory() if psutil else None
            memory_usage = memory.percent if memory else 0.0
        except:
            cpu_usage = 0.0
            memory_usage = 0.0

        return {
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "status": "healthy",
            "timestamp": datetime.now().isoformat()
        }

        # Keep only last 1000 metrics per name
        if len(self.metrics[name]) > 1000:
            self.metrics[name] = self.metrics[name][-1000:]

    async def get_metric(self, name: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get metric values"""
        if name not in self.metrics:
            return []

        return self.metrics[name][-limit:]

    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "timestamp": datetime.now().isoformat()
            }
        except ImportError:
            # Fallback if psutil not available
            return {
                "cpu_percent": 0.0,
                "memory_percent": 0.0,
                "disk_usage": 0.0,
                "timestamp": datetime.now().isoformat()
            }

    async def create_alert(self, alert_type: str, message: str, severity: str = "info",
                          context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create an alert"""
        if not self.is_initialized:
            raise RuntimeError("Monitoring service not initialized")

        alert = {
            "alert_id": f"{alert_type}_{int(datetime.now().timestamp())}",
            "type": alert_type,
            "message": message,
            "severity": severity,
            "context": context or {},
            "timestamp": datetime.now().isoformat(),
            "acknowledged": False
        }

        self.alerts.append(alert)

        # Keep only last 1000 alerts
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]

        logger.warning(f"Alert created: {alert_type} - {message}")

        return alert

    async def get_alerts(self, severity: str = None, acknowledged: bool = None,
                        limit: int = 50) -> List[Dict[str, Any]]:
        """Get alerts with optional filtering"""
        alerts = self.alerts

        if severity:
            alerts = [a for a in alerts if a["severity"] == severity]

        if acknowledged is not None:
            alerts = [a for a in alerts if a["acknowledged"] == acknowledged]

        return alerts[-limit:]

    async def acknowledge_alert(self, alert_id: str) -> Dict[str, Any]:
        """Acknowledge an alert"""
        for alert in self.alerts:
            if alert["alert_id"] == alert_id:
                alert["acknowledged"] = True
                alert["acknowledged_at"] = datetime.now().isoformat()
                return {"alert_id": alert_id, "acknowledged": True}

        return {"error": f"Alert {alert_id} not found"}

    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data for monitoring"""
        if not self.is_initialized:
            return {"error": "Monitoring service not initialized"}

        # Get recent metrics
        recent_metrics = {}
        for name, values in self.metrics.items():
            if values:
                recent_metrics[name] = values[-10:]  # Last 10 values

        # Get active alerts
        active_alerts = [a for a in self.alerts if not a["acknowledged"]]

        # Get system metrics
        system_metrics = await self.get_system_metrics()

        return {
            "metrics": recent_metrics,
            "active_alerts": active_alerts,
            "system_metrics": system_metrics,
            "total_alerts": len(self.alerts),
            "timestamp": datetime.now().isoformat()
        }

    async def check_health(self) -> Dict[str, Any]:
        """Check service health"""
        return {
            "service": "trading_monitoring",
            "status": "healthy" if self.is_initialized else "unhealthy",
            "metrics_count": sum(len(values) for values in self.metrics.values()),
            "alerts_count": len(self.alerts),
            "timestamp": datetime.now().isoformat()
        }


# Simplified model classes for API compatibility
class MetricRecord:
    """Metric record model"""
    def __init__(self, name: str, value: float, tags: Dict[str, Any] = None):
        self.name = name
        self.value = value
        self.tags = tags or {}

class AlertCreateRequest:
    """Alert creation request model"""
    def __init__(self, alert_type: str, message: str, severity: str = "info", context: Dict[str, Any] = None):
        self.alert_type = alert_type
        self.message = message
        self.severity = severity
        self.context = context or {}

class AlertCreateResponse:
    """Alert creation response model"""
    def __init__(self, success: bool, alert: Dict[str, Any] = None, error: str = None):
        self.success = success
        self.alert = alert or {}
        self.error = error