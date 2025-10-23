#!/usr/bin/env python3
"""
Event-Driven Trading Demo

Demonstrates how the event-driven architecture enables real-time communication
between TradPal microservices for trading operations.
"""

import asyncio
import logging
from datetime import datetime

from services.event_system import EventSystem, Event, EventType
from services.event_system.client import EventClient, EventSubscriberClient
from services.core_service.client import CoreServiceClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def trading_workflow_demo():
    """
    Demonstrate a complete trading workflow using event-driven communication:

    1. Market data arrives → triggers signal generation
    2. Trading signal generated → triggers strategy execution
    3. Order executed → triggers portfolio update
    4. Risk check performed → triggers alert if needed
    """
    print("🚀 Starting Event-Driven Trading Workflow Demo")
    print("=" * 60)

    # Initialize components
    event_client = EventClient()
    core_client = CoreServiceClient()

    # Initialize event system in core client
    await core_client.initialize_event_system()

    # Create event subscriber
    subscriber = EventSubscriberClient(event_client, poll_interval=1.0)

    # Event handlers for the trading workflow
    workflow_events = []

    async def handle_market_data(event: Event):
        """Handle incoming market data"""
        data = event.data
        symbol = data["symbol"]
        workflow_events.append(f"📊 Market data received for {symbol}")

        # Simulate signal generation based on market data
        try:
            # Generate signals using core service
            signals = await core_client.generate_signals(
                symbol=symbol,
                timeframe="1h",
                data=[data],  # Simplified - in reality we'd have historical data
                strategy_config={"type": "momentum"}
            )

            if signals:
                workflow_events.append(f"📈 Generated {len(signals)} signals for {symbol}")

        except Exception as e:
            logger.error(f"Signal generation failed: {e}")

    async def handle_trading_signal(event: Event):
        """Handle trading signals"""
        signal_data = event.data
        symbol = signal_data["symbol"]
        workflow_events.append(f"🎯 Trading signal received for {symbol}")

        # Simulate strategy execution
        try:
            execution = await core_client.execute_strategy(
                symbol=symbol,
                timeframe="1h",
                signal=signal_data["signal"],
                capital=10000.0,
                risk_config={"max_risk": 0.02}
            )

            workflow_events.append(f"💰 Strategy executed for {symbol}")

        except Exception as e:
            logger.error(f"Strategy execution failed: {e}")

    async def handle_portfolio_update(event: Event):
        """Handle portfolio updates"""
        update_data = event.data
        workflow_events.append("📊 Portfolio updated")

        # Simulate risk assessment
        pnl = update_data.get("pnl", 0)
        if pnl < -1000:  # Risk threshold
            workflow_events.append("⚠️  Risk alert triggered!")

    # Register event handlers
    subscriber.register_handler(EventType.MARKET_DATA_UPDATE, handle_market_data)
    subscriber.register_handler(EventType.TRADING_SIGNAL, handle_trading_signal)
    subscriber.register_handler(EventType.PORTFOLIO_UPDATE, handle_portfolio_update)

    # Start event subscription
    subscription_task = asyncio.create_task(subscriber.start_subscribing())

    try:
        # Simulate market data flow
        print("📡 Simulating market data flow...")

        # Publish market data events
        test_symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT"]

        for symbol in test_symbols:
            await event_client.publish_market_data(
                symbol,
                {
                    "price": 50000 if symbol == "BTC/USDT" else 3000 if symbol == "ETH/USDT" else 0.5,
                    "volume": 1000.0,
                    "change_24h": 2.5,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            workflow_events.append(f"📤 Published market data for {symbol}")

            # Wait a bit between symbols
            await asyncio.sleep(2)

        # Let events propagate through the system
        await asyncio.sleep(10)

        # Display workflow results
        print("\n📋 Event-Driven Workflow Results:")
        print("-" * 40)
        for i, event in enumerate(workflow_events, 1):
            print(f"{i:2d}. {event}")

        # Get event statistics
        stats = await event_client.get_event_stats()
        print(f"\n📈 Event Stream Statistics:")
        print(f"   • Total events: {stats.get('stream_length', 0)}")
        print(f"   • Consumer groups: {len(stats.get('groups', []))}")

    finally:
        # Cleanup
        await subscriber.stop_subscribing()
        subscription_task.cancel()

        try:
            await subscription_task
        except asyncio.CancelledError:
            pass

        await core_client.close()
        await event_client.session.close()

        print("\n✅ Event-Driven Trading Workflow Demo completed!")


async def real_time_monitoring_demo():
    """
    Demonstrate real-time monitoring of the event stream
    """
    print("\n📊 Starting Real-Time Event Monitoring Demo")
    print("=" * 50)

    event_client = EventClient()

    try:
        # Monitor events for 30 seconds
        print("👀 Monitoring event stream for 30 seconds...")

        start_time = asyncio.get_event_loop().time()
        event_count = 0

        while asyncio.get_event_loop().time() - start_time < 30:
            try:
                events = await event_client.get_events(limit=10)

                for event in events:
                    if event["timestamp"] > datetime.utcnow().isoformat():  # Recent events
                        event_count += 1
                        event_type = event["event_type"]
                        source = event["source"]
                        print(f"📡 [{event_type}] from {source}")

                await asyncio.sleep(5)  # Check every 5 seconds

            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(5)

        print(f"\n📈 Monitoring complete. Observed {event_count} events in 30 seconds.")

    finally:
        await event_client.session.close()


async def event_replay_demo():
    """
    Demonstrate event replay functionality
    """
    print("\n⏪ Starting Event Replay Demo")
    print("=" * 30)

    event_client = EventClient()

    try:
        # Get recent events
        events = await event_client.get_events(limit=20)
        print(f"📚 Found {len(events)} recent events to replay")

        # Replay trading signals
        signal_events = [e for e in events if e.get("event_type") == "trading_signal"]
        print(f"🎯 Replaying {len(signal_events)} trading signals...")

        for event in signal_events[-5:]:  # Replay last 5 signals
            print(f"   • Replaying signal for {event['data'].get('symbol', 'unknown')}")

        print("✅ Event replay completed")

    finally:
        await event_client.session.close()


async def main():
    """Run all event-driven demos"""
    print("🎯 TradPal Event-Driven Architecture Demo")
    print("This demonstrates how microservices communicate via events")
    print("=" * 60)

    try:
        await trading_workflow_demo()
        await real_time_monitoring_demo()
        await event_replay_demo()

        print("\n🎉 All event-driven demos completed successfully!")
        print("\nKey Benefits Demonstrated:")
        print("• 🔄 Loose coupling between services")
        print("• 📡 Real-time communication")
        print("• 🔍 Event persistence and replay")
        print("• 📊 Centralized monitoring")
        print("• 🛡️ Fault tolerance and resilience")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())