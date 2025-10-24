#!/usr/bin/env python3
"""
TradPal Infrastructure Service Demo

This comprehensive demo showcases all infrastructure service components:
- API Gateway routing and authentication
- Event System publishing and subscription
- Security Service mTLS and JWT
- Circuit Breaker resilience patterns
- Falco Security monitoring (simulated)

Run this demo to see the complete infrastructure stack in action.
"""

import asyncio
import logging
import sys
from datetime import datetime
from typing import Dict, Any

# Add the services directory to the path
sys.path.insert(0, '/Users/danielsadowski/VSCodeProjects/tradpal/tradpal')

# Import service clients
from services.infrastructure_service.api_gateway_service.client import APIGatewayClient
from services.infrastructure_service.event_system_service.client import EventClient, EventSubscriberClient
from services.infrastructure_service.security_service.client import SecurityClient
from services.infrastructure_service.circuit_breaker_service.client import CircuitBreakerClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InfrastructureDemo:
    """Demo class for Infrastructure Service"""

    def __init__(self):
        self.api_gateway = None
        self.event_client = None
        self.security_client = None
        self.circuit_breaker_client = None
        self.event_subscriber = None

    async def initialize_clients(self):
        """Initialize all service clients"""
        logger.info("🔧 Initializing Infrastructure Service Clients...")

        try:
            # API Gateway Client
            self.api_gateway = APIGatewayClient("http://localhost:8000")
            await self.api_gateway.initialize()

            # Event System Client
            self.event_client = EventClient("http://localhost:8011")
            await self.event_client._ensure_session()

            # Security Service Client
            self.security_client = SecurityClient("http://localhost:8009")
            await self.security_client.initialize()

            # Circuit Breaker Client
            self.circuit_breaker_client = CircuitBreakerClient("http://localhost:8012")
            await self.circuit_breaker_client.initialize()

            logger.info("✅ All clients initialized successfully")

        except Exception as e:
            logger.warning(f"⚠️  Some services may not be running: {e}")
            logger.info("💡 This demo will work with available services")

    async def demo_security_service(self):
        """Demo Security Service features"""
        logger.info("\n🔐 === Security Service Demo ===")

        if not self.security_client:
            logger.info("⏭️  Security service not available, skipping...")
            return

        try:
            # Issue mTLS credentials
            logger.info("Issuing mTLS credentials for demo_service...")
            creds = await self.security_client.issue_credentials("demo_service")
            logger.info(f"✅ Credentials issued - expires: {creds.expires_at}")

            # Generate JWT token
            logger.info("Generating JWT token...")
            token = await self.security_client.generate_token("demo_service", ["read", "write"])
            logger.info(f"✅ JWT token generated - expires: {token.expires_at}")

            # Validate token
            logger.info("Validating JWT token...")
            validation = await self.security_client.validate_token(token.token)
            logger.info(f"✅ Token validation: {'Valid' if validation else 'Invalid'}")

            # Store and retrieve secret
            logger.info("Storing secret...")
            await self.security_client.store_secret("demo/api_key", {"key": "secret-key-123"})
            logger.info("✅ Secret stored")

            logger.info("Retrieving secret...")
            secret = await self.security_client.retrieve_secret("demo/api_key")
            logger.info(f"✅ Secret retrieved: {secret}")

        except Exception as e:
            logger.warning(f"Security service demo failed: {e}")

    async def demo_event_system(self):
        """Demo Event System features"""
        logger.info("\n📡 === Event System Demo ===")

        if not self.event_client:
            logger.info("⏭️  Event service not available, skipping...")
            return

        try:
            # Setup event subscriber
            self.event_subscriber = EventSubscriberClient(self.event_client, poll_interval=2.0)

            # Register event handlers
            async def handle_market_data(event):
                logger.info(f"📊 Received market data: {event['data']}")

            async def handle_trading_signal(event):
                logger.info(f"🎯 Received trading signal: {event['data']}")

            from services.infrastructure_service.event_system_service import EventType
            self.event_subscriber.register_handler(EventType.MARKET_DATA_UPDATE, handle_market_data)
            self.event_subscriber.register_handler(EventType.TRADING_SIGNAL, handle_trading_signal)

            # Start subscriber
            logger.info("Starting event subscriber...")
            subscriber_task = asyncio.create_task(self.event_subscriber.start_subscribing())

            # Publish market data
            logger.info("Publishing market data event...")
            await self.event_client.publish_market_data(
                "BTC/USDT",
                {"price": 45000.75, "volume": 1250.50, "change_24h": 2.3}
            )

            # Publish trading signal
            logger.info("Publishing trading signal event...")
            await self.event_client.publish_trading_signal({
                "symbol": "BTC/USDT",
                "action": "BUY",
                "confidence": 0.87,
                "quantity": 0.05,
                "reason": "Strong momentum detected"
            })

            # Wait for events to be processed
            await asyncio.sleep(3)

            # Get event stats
            logger.info("Getting event stream statistics...")
            stats = await self.event_client.get_event_stats()
            logger.info(f"📈 Event stats: {stats}")

            # Stop subscriber
            await self.event_subscriber.stop_subscribing()
            subscriber_task.cancel()

        except Exception as e:
            logger.warning(f"Event system demo failed: {e}")

    async def demo_api_gateway(self):
        """Demo API Gateway features"""
        logger.info("\n🌐 === API Gateway Demo ===")

        if not self.api_gateway:
            logger.info("⏭️  API Gateway not available, skipping...")
            return

        try:
            # Check gateway health
            logger.info("Checking API Gateway health...")
            health = await self.api_gateway.health_check()
            logger.info(f"🏥 Gateway health: {health}")

            # List services
            logger.info("Listing registered services...")
            services = await self.api_gateway.list_services()
            logger.info(f"📋 Registered services: {len(services.get('services', []))}")

            # Try authentication (demo credentials)
            logger.info("Attempting authentication...")
            try:
                auth_result = await self.api_gateway.authenticate("admin", "admin")
                logger.info(f"🔑 Authentication successful: {auth_result.get('user', 'unknown')}")
            except Exception as e:
                logger.info(f"🔑 Authentication failed (expected): {e}")

        except Exception as e:
            logger.warning(f"API Gateway demo failed: {e}")

    async def demo_circuit_breaker(self):
        """Demo Circuit Breaker features"""
        logger.info("\n🛡️ === Circuit Breaker Demo ===")

        if not self.circuit_breaker_client:
            logger.info("⏭️  Circuit Breaker service not available, skipping...")
            return

        try:
            # List circuit breakers
            logger.info("Listing circuit breakers...")
            breakers = await self.circuit_breaker_client.list_breakers()
            logger.info(f"🔌 Active breakers: {breakers}")

            if breakers.get('circuit_breakers'):
                breaker_name = breakers['circuit_breakers'][0]

                # Get breaker info
                logger.info(f"Getting info for breaker: {breaker_name}")
                info = await self.circuit_breaker_client.get_breaker_info(breaker_name)
                logger.info(f"ℹ️  Breaker info: {info}")

                # Get breaker metrics
                logger.info(f"Getting metrics for breaker: {breaker_name}")
                metrics = await self.circuit_breaker_client.get_breaker_metrics(breaker_name)
                logger.info(f"📊 Breaker metrics: {metrics}")

                # Get dashboard data
                logger.info("Getting dashboard data...")
                dashboard = await self.circuit_breaker_client.get_dashboard_data()
                logger.info(f"📈 Dashboard summary: {dashboard.get('summary', {})}")

                # Check for alerts
                logger.info("Checking for alerts...")
                alerts = await self.circuit_breaker_client.get_alerts()
                logger.info(f"🚨 Active alerts: {alerts.get('alert_count', 0)}")

        except Exception as e:
            logger.warning(f"Circuit Breaker demo failed: {e}")

    async def demo_falco_security(self):
        """Demo Falco Security monitoring (simulated)"""
        logger.info("\n👁️ === Falco Security Demo ===")

        logger.info("Falco Security Service provides runtime monitoring...")
        logger.info("🔍 Container activity monitoring")
        logger.info("📁 Filesystem protection")
        logger.info("🌐 Network security monitoring")
        logger.info("⚡ Real-time threat detection")
        logger.info("🚨 Trading-specific security rules")

        # Simulate some security events
        security_events = [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "level": "warning",
                "message": "Suspicious file access detected",
                "container": "trading_service",
                "file": "/app/config/api_keys.json"
            },
            {
                "timestamp": datetime.utcnow().isoformat(),
                "level": "info",
                "message": "Network connection to exchange API",
                "container": "data_service",
                "destination": "api.binance.com:443"
            }
        ]

        for event in security_events:
            logger.info(f"🔔 Security Event: {event['message']} (Level: {event['level']})")

        logger.info("💡 In production, these events would be sent to notification services")

    async def demo_service_integration(self):
        """Demo integration between services"""
        logger.info("\n🔗 === Service Integration Demo ===")

        logger.info("Demonstrating how infrastructure services work together...")

        # 1. Security: Get authentication token
        if self.security_client:
            try:
                token = await self.security_client.generate_token("integration_demo", ["read"])
                logger.info("🔐 Step 1: Obtained JWT token from Security Service")
            except Exception as e:
                logger.warning(f"Security integration failed: {e}")
                token = None

        # 2. Events: Publish service startup event
        if self.event_client:
            try:
                await self.event_client.publish_event({
                    "event_type": "system.service_startup",
                    "source": "infrastructure_demo",
                    "data": {"service": "integration_demo", "status": "running"}
                })
                logger.info("📡 Step 2: Published startup event via Event System")
            except Exception as e:
                logger.warning(f"Event integration failed: {e}")

        # 3. API Gateway: Route authenticated request
        if self.api_gateway and token:
            try:
                # This would normally route through the gateway
                logger.info("🌐 Step 3: API Gateway would route authenticated requests")
                logger.info(f"🔑 Using token: {token[:20]}...")
            except Exception as e:
                logger.warning(f"Gateway integration failed: {e}")

        # 4. Circuit Breaker: Monitor service health
        if self.circuit_breaker_client:
            try:
                dashboard = await self.circuit_breaker_client.get_dashboard_data()
                healthy_services = dashboard.get('summary', {}).get('closed_breakers', 0)
                logger.info(f"🛡️ Step 4: Circuit Breaker monitoring {healthy_services} healthy services")
            except Exception as e:
                logger.warning(f"Circuit breaker integration failed: {e}")

        logger.info("✅ Service integration demonstration complete")

    async def run_demo(self):
        """Run the complete infrastructure demo"""
        logger.info("🚀 Starting TradPal Infrastructure Service Demo")
        logger.info("=" * 60)

        try:
            # Initialize clients
            await self.initialize_clients()

            # Run individual service demos
            await self.demo_security_service()
            await self.demo_event_system()
            await self.demo_api_gateway()
            await self.demo_circuit_breaker()
            await self.demo_falco_security()

            # Run integration demo
            await self.demo_service_integration()

            # Summary
            logger.info("\n" + "=" * 60)
            logger.info("✅ Infrastructure Service Demo Completed Successfully!")
            logger.info("🎯 Demonstrated Features:")
            logger.info("  • Zero-Trust Security (mTLS + JWT)")
            logger.info("  • Event-Driven Communication (Redis Streams)")
            logger.info("  • API Gateway (Load Balancing + Auth)")
            logger.info("  • Circuit Breaker Resilience")
            logger.info("  • Runtime Security Monitoring")
            logger.info("  • Service Integration Patterns")

        except Exception as e:
            logger.error(f"Demo failed: {e}")
            raise
        finally:
            # Cleanup
            await self.cleanup()

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up demo resources...")

        if self.api_gateway:
            await self.api_gateway.close()
        if self.event_client:
            pass  # EventClient uses aiohttp session
        if self.security_client:
            await self.security_client.close()
        if self.circuit_breaker_client:
            await self.circuit_breaker_client.close()

        logger.info("✅ Cleanup complete")


async def main():
    """Main demo function"""
    demo = InfrastructureDemo()
    await demo.run_demo()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())</content>
<parameter name="filePath">/Users/danielsadowski/VSCodeProjects/tradpal/tradpal/services/infrastructure_service/demo.py