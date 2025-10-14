#!/usr/bin/env python3
"""
Notification Service API

FastAPI REST API for the Notification Service providing:
- Notification sending endpoints
- Queue management
- Statistics and monitoring
- Integration management
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from enum import Enum

from .service import (
    NotificationService, NotificationConfig, NotificationMessage,
    NotificationType, NotificationPriority, NotificationChannel,
    EventSystem
)


# Pydantic models for API
class NotificationTypeEnum(str, Enum):
    SIGNAL = "signal"
    ALERT = "alert"
    STATUS = "status"
    ERROR = "error"
    INFO = "info"


class NotificationPriorityEnum(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationChannelEnum(str, Enum):
    TELEGRAM = "telegram"
    DISCORD = "discord"
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"


class SendNotificationRequest(BaseModel):
    """Request model for sending notifications"""
    message: str = Field(..., description="Notification message content")
    title: Optional[str] = Field("", description="Notification title")
    type: NotificationTypeEnum = Field(NotificationTypeEnum.INFO, description="Notification type")
    priority: NotificationPriorityEnum = Field(NotificationPriorityEnum.NORMAL, description="Notification priority")
    channels: List[NotificationChannelEnum] = Field([], description="Target channels (empty for default)")
    data: Dict[str, Any] = Field(default_factory=dict, description="Additional data")


class SendSignalRequest(BaseModel):
    """Request model for sending signal notifications"""
    symbol: str = Field(..., description="Trading symbol")
    signal_type: str = Field(..., description="Signal type (BUY/SELL)")
    price: float = Field(..., description="Current price")
    timeframe: str = Field("1m", description="Timeframe")
    indicators: Dict[str, Any] = Field(default_factory=dict, description="Technical indicators")
    risk_management: Dict[str, Any] = Field(default_factory=dict, description="Risk management data")
    channels: List[NotificationChannelEnum] = Field([], description="Target channels")


class SendAlertRequest(BaseModel):
    """Request model for sending alerts"""
    message: str = Field(..., description="Alert message")
    level: str = Field("info", description="Alert level (info/warning/error/critical)")
    data: Dict[str, Any] = Field(default_factory=dict, description="Additional data")
    channels: List[NotificationChannelEnum] = Field([], description="Target channels")


class BatchNotificationRequest(BaseModel):
    """Request model for batch notifications"""
    notifications: List[SendNotificationRequest] = Field(..., description="List of notifications to send")


class NotificationResponse(BaseModel):
    """Response model for notification operations"""
    message_id: str = Field(..., description="Unique notification ID")
    status: str = Field(..., description="Operation status")
    queued: bool = Field(..., description="Whether notification was queued")
    timestamp: datetime = Field(..., description="Operation timestamp")


class QueueStatusResponse(BaseModel):
    """Response model for queue status"""
    queue_size: int = Field(..., description="Current queue size")
    max_queue_size: int = Field(..., description="Maximum queue size")
    active_workers: int = Field(..., description="Number of active workers")
    total_workers: int = Field(..., description="Total number of workers")


class StatisticsResponse(BaseModel):
    """Response model for service statistics"""
    messages_sent: int = Field(..., description="Total messages sent")
    messages_failed: int = Field(..., description="Total messages failed")
    messages_queued: int = Field(..., description="Total messages queued")
    messages_retried: int = Field(..., description="Total messages retried")
    channel_stats: Dict[str, Dict[str, int]] = Field(..., description="Per-channel statistics")
    integrations: Dict[str, Any] = Field(..., description="Integration status")


class HealthResponse(BaseModel):
    """Response model for health check"""
    service: str = Field(..., description="Service name")
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Health check timestamp")
    queue_status: QueueStatusResponse
    integrations_active: int = Field(..., description="Number of active integrations")
    statistics: Dict[str, Any] = Field(..., description="Service statistics")


# Global service instance
notification_service: Optional[NotificationService] = None
event_system = EventSystem()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global notification_service

    # Startup
    logger = logging.getLogger(__name__)
    logger.info("Starting Notification Service API...")

    notification_service = NotificationService(event_system=event_system)
    await notification_service.start()

    yield

    # Shutdown
    logger.info("Shutting down Notification Service API...")
    await notification_service.stop()


# Create FastAPI app
app = FastAPI(
    title="TradPal Notification Service",
    description="REST API for managing notifications across multiple channels",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Get service health status"""
    if not notification_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    health = await notification_service.health_check()
    return HealthResponse(**health)


@app.post("/notifications/send", response_model=NotificationResponse)
async def send_notification(request: SendNotificationRequest):
    """Send a notification"""
    if not notification_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Convert enums
        notification_type = NotificationType(request.type.value)
        priority = NotificationPriority(request.priority.value)
        channels = [NotificationChannel(c.value) for c in request.channels]

        message_id = await notification_service.send_notification(
            message=request.message,
            title=request.title,
            notification_type=notification_type,
            priority=priority,
            channels=channels,
            data=request.data
        )

        return NotificationResponse(
            message_id=message_id,
            status="queued",
            queued=True,
            timestamp=datetime.now()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send notification: {str(e)}")


@app.post("/notifications/signal", response_model=NotificationResponse)
async def send_signal_notification(request: SendSignalRequest):
    """Send a trading signal notification"""
    if not notification_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Convert to signal data format
        signal_data = {
            'symbol': request.symbol,
            'signal_type': request.signal_type,
            'price': request.price,
            'timeframe': request.timeframe,
            'indicators': request.indicators,
            'risk_management': request.risk_management
        }

        message_id = await notification_service.send_signal_notification(signal_data)

        return NotificationResponse(
            message_id=message_id,
            status="queued",
            queued=True,
            timestamp=datetime.now()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send signal notification: {str(e)}")


@app.post("/notifications/alert", response_model=NotificationResponse)
async def send_alert_notification(request: SendAlertRequest):
    """Send an alert notification"""
    if not notification_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        message_id = await notification_service.send_alert_notification(
            alert_message=request.message,
            alert_level=request.level,
            data=request.data
        )

        return NotificationResponse(
            message_id=message_id,
            status="queued",
            queued=True,
            timestamp=datetime.now()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send alert notification: {str(e)}")


@app.post("/notifications/batch", response_model=List[NotificationResponse])
async def send_batch_notifications(request: BatchNotificationRequest, background_tasks: BackgroundTasks):
    """Send multiple notifications in batch"""
    if not notification_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        responses = []

        for notification_req in request.notifications:
            # Convert enums
            notification_type = NotificationType(notification_req.type.value)
            priority = NotificationPriority(notification_req.priority.value)
            channels = [NotificationChannel(c.value) for c in notification_req.channels]

            message_id = await notification_service.send_notification(
                message=notification_req.message,
                title=notification_req.title,
                notification_type=notification_type,
                priority=priority,
                channels=channels,
                data=notification_req.data
            )

            responses.append(NotificationResponse(
                message_id=message_id,
                status="queued",
                queued=True,
                timestamp=datetime.now()
            ))

        return responses

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send batch notifications: {str(e)}")


@app.get("/queue/status", response_model=QueueStatusResponse)
async def get_queue_status():
    """Get current queue status"""
    if not notification_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        status = await notification_service.get_queue_status()
        return QueueStatusResponse(**status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get queue status: {str(e)}")


@app.get("/statistics", response_model=StatisticsResponse)
async def get_statistics():
    """Get service statistics"""
    if not notification_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        stats = await notification_service.get_statistics()
        return StatisticsResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


@app.post("/integrations/test")
async def test_integrations():
    """Test all notification integrations"""
    if not notification_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        from integrations import integration_manager
        results = integration_manager.test_all_connections()

        return {
            'status': 'completed',
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to test integrations: {str(e)}")


@app.get("/integrations/status")
async def get_integrations_status():
    """Get status of all notification integrations"""
    if not notification_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        from integrations import integration_manager
        status = integration_manager.get_status_overview()

        return {
            'status': 'success',
            'integrations': status,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get integrations status: {str(e)}")


@app.post("/events/{event_type}")
async def publish_event(event_type: str, data: Dict[str, Any]):
    """Publish a custom event to the notification service"""
    if not notification_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        await event_system.publish(event_type, data)
        return {
            'status': 'published',
            'event_type': event_type,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to publish event: {str(e)}")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            'error': 'HTTPException',
            'detail': exc.detail,
            'timestamp': datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logging.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            'error': 'InternalServerError',
            'detail': 'An unexpected error occurred',
            'timestamp': datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)