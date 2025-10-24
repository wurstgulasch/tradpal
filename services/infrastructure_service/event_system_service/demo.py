#!/usr/bin/env python3
"""
TradPal Event System Service Demo

This demo showcases the Event System Service capabilities:
- Event publishing and subscription
- REST API usage
- Client library integration
- Event persistence and replay
- Monitoring and metrics

Run this demo to see the event system in action.
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from typing import Dict, Any

# Add the services directory to the path
sys.path.insert(0, '/Users/danielsadowski/VSCodeProjects/tradpal/tradpal')

from services.infrastructure_service.event_system_service import (
    EventSystem, Event, EventType, EventSubscriber,
    publish_market_data, publish_trading_signal, publish_portfolio_update
)
from services.infrastructure_service.event_system_service.client import (
    EventClient, EventSubscriberClient
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EventDemo:
    """Demo class for Event System Service"""

    def __init__(self):
        self.event_system = None
        self.subscriber = None
        self.events_received = []

    async def initialize(self):
        """Initialize the event system"""
        logger.info("Initializing Event System...")

        # Initialize event system
        self.event_system = EventSystem()
        await self.event_system.initialize()

        # Create subscriber
        self.subscriber = EventSubscriber(self.event_system.redis)

        # Register event handlers
        self.subscriber.register_handler(EventType.MARKET_DATA_UPDATE, self.handle_market_data)
        self.subscriber.register_handler(EventType.TRADING_SIGNAL, self.handle_trading_signal)
        self.subscriber.register_handler(EventType.PORTFOLIO_UPDATE, self.handle_portfolio_update)

        await self.subscriber.initialize()

        logger.info("Event System initialized successfully")

    async def handle_market_data(self, event: Event):
        """Handle market data events"""
        self.events_received.append(event)
        logger.info(f"📊 Market Data Event: {event.data}")

    async def handle_trading_signal(self, event: Event):
        """Handle trading signal events"""
        self.events_received.append(event)
        logger.info(f"🎯 Trading Signal Event: {event.data}")

    async def handle_portfolio_update(self, event: Event):
        """Handle portfolio update events"""
        self.events_received.append(event)
        logger.info(f"💼 Portfolio Update Event: {event.data}")

    async def demo_basic_publishing(self):
        """Demo basic event publishing"""
        logger.info("\n=== Basic Event Publishing Demo ===")

        # Publish market data
        await publish_market_data(
            "BTC/USDT",
            {"price": 45000.50, "volume": 1250.75, "change_24h": 2.5}
        )

        # Publish trading signal
        await publish_trading_signal({
            "symbol": "BTC/USDT",
            "action": "BUY",
            "confidence": 0.85,
            "quantity": 0.1,
            "reason": "Strong bullish momentum"
        })

        # Publish portfolio update
        await publish_portfolio_update({
            "total_value": 125000.00,
            "pnl_24h": 1250.50,
            "pnl_percentage": 1.0,
            "positions": ["BTC/USDT", "ETH/USDT"]
        })

        # Wait for events to be processed
        await asyncio.sleep(2)

        logger.info(f"Published 3 events, received {len(self.events_received)} events")

    async def demo_custom_events(self):
        """Demo custom event publishing"""
        logger.info("\n=== Custom Event Publishing Demo ===")

        # Create custom events
        risk_alert = Event(
            event_type=EventType.RISK_ALERT,
            source="risk_service",
            data={
                "alert_type": "high_volatility",
                "symbol": "BTC/USDT",
                "current_volatility": 0.85,
                "threshold": 0.8,
                "action_required": "reduce_position"
            }
        )

        ml_update = Event(
            event_type=EventType.ML_MODEL_UPDATE,
            source="ml_service",
            data={
                "model_name": "ensemble_model_v2",
                "accuracy": 0.78,
                "training_time": 3600,
                "features_used": 45
            }
        )

        # Publish custom events
        await self.event_system.publish_event(risk_alert)
        await self.event_system.publish_event(ml_update)

        # Wait for processing
        await asyncio.sleep(2)

        logger.info("Published 2 custom events")

    async def demo_event_replay(self):
        """Demo event replay functionality"""
        logger.info("\n=== Event Replay Demo ===")

        # Get events from the last hour
        start_time = datetime.utcnow() - timedelta(hours=1)
        events = await self.event_system.store.get_events(
            start_time=start_time,
            limit=10
        )

        logger.info(f"Found {len(events)} events in the last hour")

        # Replay events
        replayed_count = 0
        async def replay_handler(event: Event):
            nonlocal replayed_count
            logger.info(f"🔄 Replaying: {event.event_type.value} from {event.source}")
            replayed_count += 1

        await self.event_system.replay_events(
            start_time=start_time,
            handler=replay_handler
        )

        logger.info(f"Replayed {replayed_count} events")

    async def demo_rest_api_client(self):
        """Demo REST API client usage"""
        logger.info("\n=== REST API Client Demo ===")

        try:
            async with EventClient("http://localhost:8011") as client:
                # Check health
                health = await client.health_check()
                logger.info(f"Service Health: {health}")

                # Publish via REST API
                message_id = await client.publish_market_data(
                    "ETH/USDT",
                    {"price": 2800.00, "volume": 500.25}
                )
                logger.info(f"Published via REST API: {message_id}")

                # Get recent events
                events = await client.get_events(limit=5)
                logger.info(f"Retrieved {len(events)} recent events via REST API")

                # Get stats
                stats = await client.get_event_stats()
                logger.info(f"Event Stream Stats: {stats}")

        except Exception as e:
            logger.warning(f"REST API demo failed (service may not be running): {e}")

    async def demo_subscriber_client(self):
        """Demo subscriber client"""
        logger.info("\n=== Subscriber Client Demo ===")

        try:
            client = EventClient("http://localhost:8011")
            subscriber = EventSubscriberClient(client, poll_interval=2.0)

            # Register handlers
            async def demo_handler(event: Event):
                logger.info(f"📨 Subscriber received: {event.event_type.value}")

            subscriber.register_handler(EventType.MARKET_DATA_UPDATE, demo_handler)
            subscriber.register_handler(EventType.TRADING_SIGNAL, demo_handler)

            # Start subscribing for a short time
            logger.info("Starting subscriber for 5 seconds...")
            subscribe_task = asyncio.create_task(subscriber.start_subscribing())

            # Publish some events while subscribing
            await asyncio.sleep(1)
            await publish_market_data("ADA/USDT", {"price": 0.45, "volume": 10000})

            await asyncio.sleep(1)
            await publish_trading_signal({"symbol": "ADA/USDT", "action": "HOLD"})

            # Stop subscribing
            await asyncio.sleep(3)
            await subscriber.stop_subscribing()
            subscribe_task.cancel()

            logger.info("Subscriber demo completed")

        except Exception as e:
            logger.warning(f"Subscriber client demo failed: {e}")

    async def demo_performance_test(self):
        """Demo performance capabilities"""
        logger.info("\n=== Performance Test Demo ===")

        import time

        # Publish multiple events quickly
        start_time = time.time()
        event_count = 100

        for i in range(event_count):
            await publish_market_data(
                f"BTC/USDT_{i}",
                {"price": 45000 + i, "volume": 1000 + i}
            )

        end_time = time.time()
        duration = end_time - start_time

        logger.info(f"Published {event_count} events in {duration:.2f} seconds")
        logger.info(".2f")

    async def run_demo(self):
        """Run the complete demo"""
        logger.info("🚀 Starting TradPal Event System Service Demo")
        logger.info("=" * 50)

        try:
            # Initialize
            await self.initialize()

            # Run demos
            await self.demo_basic_publishing()
            await self.demo_custom_events()
            await self.demo_event_replay()
            await self.demo_rest_api_client()
            await self.demo_subscriber_client()
            await self.demo_performance_test()

            # Summary
            logger.info("\n" + "=" * 50)
            logger.info("✅ Event System Demo Completed Successfully!")
            logger.info(f"📊 Total Events Processed: {len(self.events_received)}")
            logger.info("🎯 Key Features Demonstrated:")
            logger.info("  • Event publishing and subscription")
            logger.info("  • Custom event types and data")
            logger.info("  • Event persistence and replay")
            logger.info("  • REST API integration")
            logger.info("  • Client library usage")
            logger.info("  • Performance capabilities")

        except Exception as e:
            logger.error(f"Demo failed: {e}")
            raise
        finally:
            # Cleanup
            if self.event_system:
                await self.event_system.close()


async def main():
    """Main demo function"""
    demo = EventDemo()
    await demo.run_demo()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())