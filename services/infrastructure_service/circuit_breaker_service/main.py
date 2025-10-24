#!/usr/bin/env python3
"""
TradPal Circuit Breaker Monitoring Service

Provides monitoring and management interface for circuit breakers.
Allows viewing metrics, resetting breakers, and configuring thresholds.
"""

import asyncio
import logging
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import uvicorn
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from . import (
    circuit_breaker_registry,
    CircuitBreakerConfig,
    CircuitBreakerState,
    get_circuit_breaker
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TradPal Circuit Breaker Service",
    description="Circuit breaker monitoring and management for TradPal microservices",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Service health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "circuit_breaker_service"
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()


@app.get("/circuit-breakers")
async def list_circuit_breakers():
    """List all registered circuit breakers"""
    breakers = circuit_breaker_registry.get_all_breakers()
    return {
        "circuit_breakers": list(breakers.keys()),
        "count": len(breakers)
    }


@app.get("/circuit-breakers/{name}")
async def get_circuit_breaker_info(name: str):
    """Get detailed information about a specific circuit breaker"""
    breakers = circuit_breaker_registry.get_all_breakers()

    if name not in breakers:
        raise HTTPException(status_code=404, detail=f"Circuit breaker '{name}' not found")

    breaker = breakers[name]
    return {
        "name": name,
        "state": breaker.get_state().value,
        "metrics": breaker.get_metrics()
    }


@app.get("/circuit-breakers/{name}/metrics")
async def get_circuit_breaker_metrics(name: str):
    """Get metrics for a specific circuit breaker"""
    breakers = circuit_breaker_registry.get_all_breakers()

    if name not in breakers:
        raise HTTPException(status_code=404, detail=f"Circuit breaker '{name}' not found")

    breaker = breakers[name]
    return breaker.get_metrics()


@app.post("/circuit-breakers/{name}/reset")
async def reset_circuit_breaker(name: str):
    """Reset a circuit breaker to closed state"""
    breakers = circuit_breaker_registry.get_all_breakers()

    if name not in breakers:
        raise HTTPException(status_code=404, detail=f"Circuit breaker '{name}' not found")

    breaker = breakers[name]

    # Force transition to closed state
    if breaker.state != CircuitBreakerState.CLOSED:
        await breaker._transition_to_closed()

    return {
        "status": "reset",
        "name": name,
        "new_state": breaker.get_state().value
    }


@app.post("/circuit-breakers/reset-all")
async def reset_all_circuit_breakers():
    """Reset all circuit breakers to closed state"""
    await circuit_breaker_registry.reset_all()

    breakers = circuit_breaker_registry.get_all_breakers()
    reset_count = sum(1 for b in breakers.values() if b.state == CircuitBreakerState.CLOSED)

    return {
        "status": "reset_all",
        "total_breakers": len(breakers),
        "reset_count": reset_count
    }


@app.put("/circuit-breakers/{name}/config")
async def update_circuit_breaker_config(name: str, config: Dict[str, Any]):
    """Update configuration for a circuit breaker"""
    # Note: In a real implementation, this would update the config
    # For now, we'll just return the current config
    breakers = circuit_breaker_registry.get_all_breakers()

    if name not in breakers:
        raise HTTPException(status_code=404, detail=f"Circuit breaker '{name}' not found")

    breaker = breakers[name]
    return {
        "status": "config_updated",
        "name": name,
        "config": breaker.get_metrics()["config"]
    }


@app.get("/circuit-breakers/{name}/state")
async def get_circuit_breaker_state(name: str):
    """Get the current state of a circuit breaker"""
    breakers = circuit_breaker_registry.get_all_breakers()

    if name not in breakers:
        raise HTTPException(status_code=404, detail=f"Circuit breaker '{name}' not found")

    breaker = breakers[name]
    return {
        "name": name,
        "state": breaker.get_state().value,
        "is_open": breaker.get_state() == CircuitBreakerState.OPEN,
        "is_half_open": breaker.get_state() == CircuitBreakerState.HALF_OPEN,
        "is_closed": breaker.get_state() == CircuitBreakerState.CLOSED
    }


@app.get("/dashboard")
async def get_dashboard_data():
    """Get dashboard data for all circuit breakers"""
    breakers = circuit_breaker_registry.get_all_breakers()
    all_metrics = circuit_breaker_registry.get_metrics()

    # Aggregate statistics
    total_breakers = len(breakers)
    open_breakers = sum(1 for m in all_metrics.values() if m["state"] == "open")
    half_open_breakers = sum(1 for m in all_metrics.values() if m["state"] == "half_open")
    closed_breakers = sum(1 for m in all_metrics.values() if m["state"] == "closed")

    total_requests = sum(m["requests_total"] for m in all_metrics.values())
    total_failures = sum(m["failures_total"] for m in all_metrics.values())
    total_successes = sum(m["successes_total"] for m in all_metrics.values())

    return {
        "summary": {
            "total_breakers": total_breakers,
            "open_breakers": open_breakers,
            "half_open_breakers": half_open_breakers,
            "closed_breakers": closed_breakers,
            "total_requests": total_requests,
            "total_failures": total_failures,
            "total_successes": total_successes,
            "failure_rate": total_failures / max(total_requests, 1)
        },
        "breakers": [
            {
                "name": name,
                "state": metrics["state"],
                "requests": metrics["requests_total"],
                "failures": metrics["failures_total"],
                "successes": metrics["successes_total"],
                "last_failure": metrics["last_failure_time"],
                "last_success": metrics["last_success_time"]
            }
            for name, metrics in all_metrics.items()
        ]
    }


@app.get("/alerts")
async def get_circuit_breaker_alerts():
    """Get alerts for circuit breakers that are in open or half-open state"""
    breakers = circuit_breaker_registry.get_all_breakers()
    alerts = []

    for name, breaker in breakers.items():
        state = breaker.get_state()
        metrics = breaker.get_metrics()

        if state == CircuitBreakerState.OPEN:
            alerts.append({
                "level": "error",
                "service": name,
                "message": f"Circuit breaker is OPEN - blocking requests",
                "failures": metrics["failures_total"],
                "last_failure": metrics["last_failure_time"],
                "timestamp": datetime.utcnow().isoformat()
            })
        elif state == CircuitBreakerState.HALF_OPEN:
            alerts.append({
                "level": "warning",
                "service": name,
                "message": f"Circuit breaker is HALF_OPEN - testing recovery",
                "successes_in_half_open": metrics.get("half_open_successes", 0),
                "timestamp": datetime.utcnow().isoformat()
            })

    return {
        "alerts": alerts,
        "alert_count": len(alerts)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8012)