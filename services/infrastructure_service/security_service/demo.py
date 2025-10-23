#!/usr/bin/env python3
"""
Security Service Demo
Demonstrates Zero-Trust Security with mTLS, JWT tokens, and secrets management.
"""

import asyncio
import logging
import os
from pathlib import Path

from service import SecurityService, SecurityConfig


class SecurityDemo:
    """Demo class for security service functionality."""

    def __init__(self):
        self.service: SecurityService = None
        self.logger = logging.getLogger(__name__)

    async def setup_service(self):
        """Setup the security service."""
        print("🔐 Setting up Security Service...")

        config = SecurityConfig()
        self.service = SecurityService(config)

        await self.service.start()
        print("✅ Security Service started successfully")

    async def demo_mtls_credentials(self):
        """Demonstrate mTLS credential issuance."""
        print("\n=== mTLS Credentials Demo ===")

        # Issue credentials for trading service
        credentials = await self.service.issue_service_credentials("trading_bot_service")

        print("✅ Issued mTLS credentials for trading_bot_service")
        print(f"   Certificate length: {len(credentials.certificate)} characters")
        print(f"   Private key length: {len(credentials.private_key)} characters")
        print(f"   Issued: {credentials.issued_at}")
        print(f"   Expires: {credentials.expires_at}")

    async def demo_jwt_tokens(self):
        """Demonstrate JWT token generation and validation."""
        print("\n=== JWT Token Demo ===")

        # Generate token for trading service
        token = await self.service.generate_jwt_token(
            service_name="trading_bot_service",
            permissions=["read", "write", "trade"]
        )

        print("✅ Generated JWT token for trading_bot_service")
        print(f"   Token: {token.token[:50]}...")
        print(f"   Permissions: {token.permissions}")
        print(f"   Expires: {token.expires_at}")

        # Validate the token
        validated = await self.service.validate_jwt_token(token.token)

        if validated:
            print("✅ Token validation successful")
            print(f"   Service: {validated.service_name}")
            print(f"   Permissions: {validated.permissions}")
        else:
            print("❌ Token validation failed")

    async def demo_secrets_management(self):
        """Demonstrate secrets management."""
        print("\n=== Secrets Management Demo ===")

        # Store API keys
        api_secrets = {
            "binance_api_key": "your-binance-api-key",
            "binance_secret_key": "your-binance-secret-key"
        }

        success = await self.service.store_secret("api_keys/binance", api_secrets)

        if success:
            print("✅ Stored API secrets successfully")
        else:
            print("❌ Failed to store API secrets")

        # Retrieve secrets
        retrieved_api = await self.service.retrieve_secret("api_keys/binance")

        if retrieved_api:
            print("✅ Retrieved API secrets successfully")
        else:
            print("❌ Failed to retrieve API secrets")

    async def demo_service_monitoring(self):
        """Demonstrate service monitoring and statistics."""
        print("\n=== Service Monitoring Demo ===")

        # Get health status
        health = await self.service.health_check()
        print("🏥 Service Health:")
        print(f"   Status: {health['status']}")
        print(f"   mTLS: {'✅' if health['components']['mtls'] else '❌'}")
        print(f"   JWT: {'✅' if health['components']['jwt'] else '❌'}")
        print(f"   Active Credentials: {health['active_credentials']}")
        print(f"   Active Tokens: {health['active_tokens']}")

    async def run_demo(self):
        """Run the complete security demo."""
        print("🛡️  Starting Security Service Demo")
        print("=" * 50)

        try:
            # Setup service
            await self.setup_service()

            # Run demonstrations
            await self.demo_mtls_credentials()
            await self.demo_jwt_tokens()
            await self.demo_secrets_management()
            await self.demo_service_monitoring()

            print("\n✅ Security Service Demo completed successfully!")

        except Exception as e:
            print(f"❌ Demo failed: {e}")
            raise
        finally:
            if self.service:
                await self.service.stop()
                print("🛑 Security Service stopped")


async def main():
    """Main demo function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run demo
    demo = SecurityDemo()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())
