#!/usr/bin/env python3
"""
Event System Demo

Demonstrates the TradPal Event-Driven Architecture with Redis Streams.
Shows how to publish and consume events across microservices.
"""

import asyncio
import logging
from datetime import datetime

from services.event_system import (
    EventSystem, Event, EventType,
    publish_market_data, publish_trading_signal, publish_portfolio_update
)
from services.event_system.client import EventClient, EventSubscriberClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_event_publishing():
    """Demonstrate event publishing"""
    print("🚀 Starting Event Publishing Demo")

    # Initialize event system
    event_system = EventSystem()
    await event_system.initialize()

    try:
        # Publish market data event
        print("📊 Publishing market data event...")
        message_id = await publish_market_data(
            symbol="BTC/USDT",
            data={
                "price": 45000.0,
                "volume": 1234.56,
                "timestamp": datetime.utcnow().isoformat()
            },
            source="demo_script"
        )
        print(f"✅ Market data published with ID: {message_id}")

        # Publish trading signal event
        print("📈 Publishing trading signal event...")
        message_id = await publish_trading_signal(
            signal={
                "symbol": "BTC/USDT",
                "action": "BUY",
                "confidence": 0.85,
                "reason": "RSI oversold + MACD crossover"
            },
            source="demo_script"
        )
        print(f"✅ Trading signal published with ID: {message_id}")

        # Publish portfolio update event
        print("💼 Publishing portfolio update event...")
        message_id = await publish_portfolio_update(
            updates={
                "total_value": 100000.0,
                "pnl": 2500.50,
                "positions": ["BTC/USDT", "ETH/USDT"]
            },
            source="demo_script"
        )
        print(f"✅ Portfolio update published with ID: {message_id}")

        # Wait a bit for events to be processed
        await asyncio.sleep(2)

        # Retrieve recent events
        print("📋 Retrieving recent events...")
        events = await event_system.store.get_events(limit=10)
        print(f"📊 Found {len(events)} recent events:")

        for event in events[-3:]:  # Show last 3 events
            print(f"  • {event.event_type.value} from {event.source} at {event.timestamp}")

    finally:
        await event_system.close()


async def demo_event_client():
    """Demonstrate event client usage"""
    print("\n🌐 Starting Event Client Demo")

    async with EventClient() as client:
        try:
            # Check service health
            health = await client.health_check()
            print(f"🏥 Event service health: {health}")

            # Publish events via client
            print("📤 Publishing events via client...")

            message_id = await client.publish_market_data(
                "ETH/USDT",
                {"price": 2800.0, "change_24h": 2.5}
            )
            print(f"✅ Market data published: {message_id}")

            message_id = await client.publish_trading_signal({
                "symbol": "ETH/USDT",
                "action": "HOLD",
                "confidence": 0.92
            })
            print(f"✅ Trading signal published: {message_id}")

            # Get recent events
            events = await client.get_events(limit=5)
            print(f"📊 Retrieved {len(events)} events")

            # Get stream stats
            stats = await client.get_event_stats()
            print(f"📈 Stream stats: {stats['stream_length']} events total")

        except Exception as e:
            print(f"❌ Client demo error: {e}")


async def demo_event_subscription():
    """Demonstrate event subscription"""
    print("\n👂 Starting Event Subscription Demo")

    # Create subscriber
    async with EventClient() as event_client:
        subscriber = EventSubscriberClient(event_client, poll_interval=2.0)

        # Register event handlers
        async def handle_market_data(event: Event):
            print(f"📊 Received market data: {event.data}")

        async def handle_trading_signal(event: Event):
            print(f"📈 Received trading signal: {event.data}")

        subscriber.register_handler(EventType.MARKET_DATA_UPDATE, handle_market_data)
        subscriber.register_handler(EventType.TRADING_SIGNAL, handle_trading_signal)

        # Start subscription in background
        subscription_task = asyncio.create_task(subscriber.start_subscribing())

        # Publish some test events
        print("📤 Publishing test events...")
        await event_client.publish_market_data("ADA/USDT", {"price": 0.45})
        await event_client.publish_trading_signal({"symbol": "ADA/USDT", "action": "BUY"})

        # Let subscription run for a bit
        await asyncio.sleep(5)

        # Stop subscription
        await subscriber.stop_subscribing()
        subscription_task.cancel()

        try:
            await subscription_task
        except asyncio.CancelledError:
            pass

        print("✅ Subscription demo completed")


async def demo_event_replay():
    """Demonstrate event replay functionality"""
    print("\n⏪ Starting Event Replay Demo")

    event_system = EventSystem()
    await event_system.initialize()

    try:
        # Replay recent events
        print("🎬 Replaying recent events...")

        replayed_count = 0
        async def replay_handler(event: Event):
            nonlocal replayed_count
            print(f"  • Replaying: {event.event_type.value} from {event.source}")
            replayed_count += 1

        await event_system.replay_events(handler=replay_handler, limit=10)
        print(f"✅ Replayed {replayed_count} events")

    finally:
        await event_system.close()


async def main():
    """Run all demos"""
    print("🎯 TradPal Event System Demo")
    print("=" * 50)

    try:
        await demo_event_publishing()
        await demo_event_client()
        await demo_event_subscription()
        await demo_event_replay()

        print("\n🎉 All demos completed successfully!")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())