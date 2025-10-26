#!/usr/bin/env python3
"""
TradPal Monitoring Service Comprehensive Demo

This demo showcases the complete monitoring service ecosystem:
- MLOps Service: Experiment tracking and model deployment
- Discovery Service: Genetic algorithm optimization
- Model Monitoring Service: ML model performance monitoring
- Alert Forwarder Service: Security alert processing
- Notification Service: Multi-channel notifications

Run this demo to see the full monitoring stack in action.
"""

import asyncio
import logging
import sys
import time
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Add the services directory to the path
sys.path.insert(0, '/Users/danielsadowski/VSCodeProjects/tradpal/tradpal')

# Import service clients
try:
    from services.monitoring_service.mlops_service.client import MLOpsClient
    from services.monitoring_service.discovery_service.client import DiscoveryClient
    from services.monitoring_service.model_monitoring_service.client import ModelMonitoringClient
    from services.monitoring_service.notification_service.client import NotificationServiceClient
except ImportError as e:
    print(f"⚠️  Some service clients not available: {e}")
    print("💡 This demo will work with available services")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MonitoringServiceDemo:
    """Comprehensive demo for the monitoring service ecosystem"""

    def __init__(self):
        self.mlops_client = None
        self.discovery_client = None
        self.monitoring_client = None
        self.notification_client = None
        self.sample_data = self._generate_sample_data()

    def _generate_sample_data(self) -> Dict[str, Any]:
        """Generate sample data for demonstrations"""
        return {
            "experiment_data": {
                "experiment_name": "btc_trading_model_demo",
                "parameters": {
                    "learning_rate": 0.001,
                    "epochs": 100,
                    "batch_size": 32,
                    "hidden_layers": [64, 32]
                },
                "metrics": {
                    "accuracy": 0.85,
                    "precision": 0.82,
                    "recall": 0.78,
                    "f1_score": 0.80,
                    "directional_accuracy": 0.67
                }
            },
            "model_data": {
                "model_id": "btc_trading_model_demo",
                "baseline_features": {
                    "rsi": [25.0, 35.0, 45.0, 55.0, 65.0, 75.0],
                    "macd": [-0.5, -0.2, 0.0, 0.2, 0.5],
                    "bb_position": [-0.3, -0.1, 0.0, 0.1, 0.3]
                },
                "baseline_metrics": {
                    "mse": 0.0234,
                    "directional_accuracy": 0.67,
                    "sharpe_ratio": 1.45
                }
            },
            "optimization_data": {
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "start_date": "2024-01-01",
                "end_date": "2024-06-30",
                "population_size": 20,  # Smaller for demo
                "generations": 5       # Smaller for demo
            }
        }

    async def initialize_clients(self):
        """Initialize all monitoring service clients"""
        logger.info("🔧 Initializing Monitoring Service Clients...")

        try:
            # Initialize MLOps client
            try:
                from config.service_settings import MLOPS_SERVICE_URL
                self.mlops_client = MLOpsClient(MLOPS_SERVICE_URL)
                await self.mlops_client.initialize()
                logger.info("✅ MLOps Service client initialized")
            except Exception as e:
                logger.warning(f"⚠️  MLOps Service not available: {e}")

            # Initialize Discovery client
            try:
                from config.service_settings import DISCOVERY_SERVICE_URL
                self.discovery_client = DiscoveryClient(DISCOVERY_SERVICE_URL)
                await self.discovery_client.initialize()
                logger.info("✅ Discovery Service client initialized")
            except Exception as e:
                logger.warning(f"⚠️  Discovery Service not available: {e}")

            # Initialize Model Monitoring client
            try:
                from config.service_settings import MODEL_MONITORING_SERVICE_URL
                self.monitoring_client = ModelMonitoringClient(MODEL_MONITORING_SERVICE_URL)
                await self.monitoring_client.initialize()
                logger.info("✅ Model Monitoring Service client initialized")
            except Exception as e:
                logger.warning(f"⚠️  Model Monitoring Service not available: {e}")

            # Initialize Notification client
            try:
                from config.service_settings import NOTIFICATION_SERVICE_URL
                self.notification_client = NotificationServiceClient(NOTIFICATION_SERVICE_URL)
                await self.notification_client.initialize()
                logger.info("✅ Notification Service client initialized")
            except Exception as e:
                logger.warning(f"⚠️  Notification Service not available: {e}")

        except Exception as e:
            logger.warning(f"⚠️  Client initialization issues: {e}")
            logger.info("💡 Demo will continue with available services")

    async def demo_mlops_workflow(self):
        """Demo complete MLOps workflow"""
        logger.info("\n🤖 === MLOps Service Demo ===")

        if not self.mlops_client:
            logger.info("⏭️  Skipping MLOps demo - service not available")
            return

        try:
            # Log experiment
            logger.info("Logging ML experiment...")
            experiment_result = await self.mlops_client.log_experiment(
                self.sample_data["experiment_data"]
            )
            logger.info(f"✅ Experiment logged: {experiment_result}")

            # Get experiment history
            logger.info("Retrieving experiment history...")
            history = await self.mlops_client.get_experiment_history()
            logger.info(f"📊 Found {len(history)} experiments")

            # Deploy model
            logger.info("Deploying model...")
            deployment_result = await self.mlops_client.deploy_model({
                "model_name": "btc_trading_model_demo",
                "model_version": "v1.0",
                "model_path": "/tmp/demo_model.pkl"  # Mock path
            })
            logger.info(f"🚀 Model deployed: {deployment_result}")

            # Create drift detector
            logger.info("Creating drift detector...")
            drift_detector = await self.mlops_client.create_drift_detector({
                "model_name": "btc_trading_model_demo",
                "baseline_data": self.sample_data["model_data"]["baseline_features"]
            })
            logger.info(f"🔍 Drift detector created: {drift_detector}")

        except Exception as e:
            logger.warning(f"MLOps workflow demo failed: {e}")

    async def demo_discovery_optimization(self):
        """Demo parameter optimization"""
        logger.info("\n🔍 === Discovery Service Demo ===")

        if not self.discovery_client:
            logger.info("⏭️  Skipping Discovery demo - service not available")
            return

        try:
            # Start optimization
            logger.info("Starting parameter optimization...")
            optimization_id = await self.discovery_client.start_optimization(
                self.sample_data["optimization_data"]
            )
            logger.info(f"🎯 Optimization started: {optimization_id}")

            # Monitor progress
            logger.info("Monitoring optimization progress...")
            for i in range(3):  # Check a few times
                await asyncio.sleep(2)  # Brief pause

                status = await self.discovery_client.get_optimization_status(optimization_id)
                logger.info(f"📈 Generation {status.get('current_generation', 0)}/{status.get('total_generations', 0)} - "
                          f"Best fitness: {status.get('best_fitness', 0):.4f}")

                if status.get('status') == 'completed':
                    break

            # Get final results
            final_status = await self.discovery_client.get_optimization_status(optimization_id)
            logger.info(f"🏆 Optimization completed!")
            logger.info(f"  • Best fitness: {final_status.get('best_fitness', 0):.4f}")
            logger.info(f"  • Best parameters: {final_status.get('best_parameters', {})}")

        except Exception as e:
            logger.warning(f"Discovery optimization demo failed: {e}")

    async def demo_model_monitoring(self):
        """Demo model monitoring and drift detection"""
        logger.info("\n📊 === Model Monitoring Service Demo ===")

        if not self.monitoring_client:
            logger.info("⏭️  Skipping Model Monitoring demo - service not available")
            return

        try:
            # Register model
            logger.info("Registering model for monitoring...")
            registration_result = await self.monitoring_client.register_model(
                self.sample_data["model_data"]
            )
            logger.info(f"✅ Model registered: {registration_result}")

            # Simulate predictions with drift
            logger.info("Simulating model predictions...")
            predictions = []

            for i in range(10):
                # Simulate some drift in features
                drift_factor = i * 0.1  # Increasing drift

                prediction_data = {
                    "model_id": "btc_trading_model_demo",
                    "prediction": 0.5 + random.uniform(-0.2, 0.2),
                    "actual": 0.5 + random.uniform(-0.2, 0.2),
                    "features": {
                        "rsi": 50.0 + drift_factor * 10,  # Drifting RSI
                        "macd": 0.0 + drift_factor * 0.2,  # Drifting MACD
                        "bb_position": 0.0 + drift_factor * 0.1
                    },
                    "metadata": {
                        "confidence": 0.8 + random.uniform(-0.1, 0.1),
                        "timestamp": datetime.now().isoformat()
                    }
                }

                # Monitor prediction
                monitoring_result = await self.monitoring_client.monitor_prediction(prediction_data)
                predictions.append(monitoring_result)

                logger.info(f"📈 Prediction {i+1}: Drift score = {monitoring_result.get('drift_score', 0):.3f}")

            # Check model status
            status = await self.monitoring_client.get_model_status("btc_trading_model_demo")
            logger.info("📊 Model Status:")
            logger.info(f"  • Current drift score: {status.get('drift', {}).get('current_score', 0):.3f}")
            logger.info(f"  • Drift detected: {status.get('drift', {}).get('drift_detected', False)}")
            logger.info(f"  • Performance degradation: {status.get('performance', {}).get('degraded', False)}")

        except Exception as e:
            logger.warning(f"Model monitoring demo failed: {e}")

    async def demo_notification_system(self):
        """Demo notification system"""
        logger.info("\n📱 === Notification Service Demo ===")

        if not self.notification_client:
            logger.info("⏭️  Skipping Notification demo - service not available")
            return

        try:
            # Send different types of notifications
            notifications = [
                {
                    "type": "info",
                    "title": "Monitoring Demo Started",
                    "message": "TradPal monitoring service demo has begun",
                    "priority": "normal"
                },
                {
                    "type": "signal",
                    "title": "Trading Signal",
                    "message": "BTC/USDT Buy signal generated",
                    "priority": "high",
                    "data": {
                        "symbol": "BTC/USDT",
                        "action": "BUY",
                        "price": 45000.0,
                        "confidence": 0.85
                    }
                },
                {
                    "type": "alert",
                    "title": "Model Drift Detected",
                    "message": "Drift detected in BTC trading model",
                    "priority": "critical",
                    "data": {
                        "model_id": "btc_trading_model_demo",
                        "drift_score": 0.15,
                        "threshold": 0.1
                    }
                }
            ]

            for notification in notifications:
                logger.info(f"Sending {notification['type']} notification...")
                result = await self.notification_client.send_notification(notification)
                logger.info(f"✅ Notification sent: {result}")

                await asyncio.sleep(0.5)  # Brief pause between notifications

            # Check queue status
            queue_status = await self.notification_client.get_queue_status()
            logger.info("📋 Queue Status:")
            logger.info(f"  • Queue size: {queue_status.get('queue_size', 0)}")
            logger.info(f"  • Processing: {queue_status.get('processing', 0)}")

            # Get statistics
            stats = await self.notification_client.get_statistics()
            logger.info("📊 Notification Statistics:")
            logger.info(f"  • Messages sent: {stats.get('messages_sent', 0)}")
            logger.info(f"  • Success rate: {stats.get('success_rate', 0):.1%}")

        except Exception as e:
            logger.warning(f"Notification demo failed: {e}")

    async def demo_service_integration(self):
        """Demo integration between services"""
        logger.info("\n🔗 === Service Integration Demo ===")

        # This demo shows how services work together
        logger.info("Demonstrating service integration workflow...")

        # 1. MLOps: Log experiment and deploy model
        if self.mlops_client:
            logger.info("1️⃣ MLOps: Logging experiment and deploying model...")
            try:
                await self.mlops_client.log_experiment(self.sample_data["experiment_data"])
                await self.mlops_client.deploy_model({
                    "model_name": "integrated_demo_model",
                    "model_version": "v1.0"
                })
                logger.info("✅ Model deployed via MLOps")
            except Exception as e:
                logger.warning(f"MLOps integration failed: {e}")

        # 2. Model Monitoring: Register deployed model
        if self.monitoring_client:
            logger.info("2️⃣ Model Monitoring: Registering deployed model...")
            try:
                await self.monitoring_client.register_model({
                    "model_id": "integrated_demo_model",
                    "baseline_features": self.sample_data["model_data"]["baseline_features"],
                    "baseline_metrics": self.sample_data["model_data"]["baseline_metrics"]
                })
                logger.info("✅ Model registered for monitoring")
            except Exception as e:
                logger.warning(f"Model monitoring integration failed: {e}")

        # 3. Discovery: Optimize parameters for the model
        if self.discovery_client:
            logger.info("3️⃣ Discovery: Optimizing model parameters...")
            try:
                optimization_id = await self.discovery_client.start_optimization(
                    self.sample_data["optimization_data"]
                )
                logger.info(f"✅ Parameter optimization started: {optimization_id}")
            except Exception as e:
                logger.warning(f"Discovery integration failed: {e}")

        # 4. Notification: Send integration status
        if self.notification_client:
            logger.info("4️⃣ Notification: Sending integration status...")
            try:
                await self.notification_client.send_notification({
                    "type": "info",
                    "title": "Service Integration Complete",
                    "message": "All monitoring services successfully integrated",
                    "priority": "normal"
                })
                logger.info("✅ Integration status notification sent")
            except Exception as e:
                logger.warning(f"Notification integration failed: {e}")

        logger.info("🔗 Service integration demonstration complete!")

    async def demo_alert_forwarder_simulation(self):
        """Simulate alert forwarder functionality"""
        logger.info("\n🚨 === Alert Forwarder Simulation Demo ===")

        logger.info("Simulating Falco security alert processing...")
        logger.info("💡 Alert Forwarder runs as background service")
        logger.info("💡 It monitors Falco logs and forwards alerts to Notification Service")

        # Simulate different types of security alerts
        sample_alerts = [
            {
                "priority": "CRITICAL",
                "rule": "Unexpected process spawned",
                "output": "Process spawned with suspicious parent",
                "source": "syscall",
                "tags": ["process", "spawn", "suspicious"]
            },
            {
                "priority": "WARNING",
                "rule": "File accessed outside of allowed directories",
                "output": "Access to sensitive file detected",
                "source": "syscall",
                "tags": ["file", "access", "sensitive"]
            },
            {
                "priority": "ERROR",
                "rule": "Network connection to suspicious IP",
                "output": "Outbound connection to known malicious IP",
                "source": "syscall",
                "tags": ["network", "connection", "malicious"]
            }
        ]

        for alert in sample_alerts:
            logger.info(f"🚨 Processing {alert['priority']} alert: {alert['rule']}")

            # In real implementation, this would be forwarded to notification service
            if self.notification_client:
                try:
                    await self.notification_client.send_alert_notification(
                        alert_message=f"Security Alert: {alert['rule']}",
                        alert_level=alert['priority'].lower(),
                        data=alert
                    )
                    logger.info(f"✅ Alert forwarded to notification service")
                except Exception as e:
                    logger.warning(f"Alert forwarding failed: {e}")
            else:
                logger.info("ℹ️  Alert would be forwarded to notification service")

            await asyncio.sleep(0.5)

    async def demo_performance_monitoring(self):
        """Demo performance monitoring across services"""
        logger.info("\n⚡ === Performance Monitoring Demo ===")

        # Import service URLs for health checks
        from config.service_settings import (
            MLOPS_SERVICE_URL,
            DISCOVERY_SERVICE_URL,
            MODEL_MONITORING_SERVICE_URL,
            NOTIFICATION_SERVICE_URL
        )

        # Test service response times and health
        services_to_check = [
            ("MLOps", self.mlops_client, f"{MLOPS_SERVICE_URL}/health"),
            ("Discovery", self.discovery_client, f"{DISCOVERY_SERVICE_URL}/health"),
            ("Model Monitoring", self.monitoring_client, f"{MODEL_MONITORING_SERVICE_URL}/health"),
            ("Notification", self.notification_client, f"{NOTIFICATION_SERVICE_URL}/health")
        ]

        logger.info("Checking service health and response times...")

        for service_name, client, health_url in services_to_check:
            if client:
                try:
                    start_time = time.time()
                    health = await client.health_check()
                    response_time = time.time() - start_time

                    status = health.get('status', 'unknown')
                    status_emoji = "✅" if status == 'healthy' else "⚠️"

                    logger.info(".2f")
                except Exception as e:
                    logger.warning(f"⚠️  {service_name} health check failed: {e}")
            else:
                logger.info(f"⏭️  {service_name} not available")

    async def run_comprehensive_demo(self):
        """Run the complete monitoring service demo"""
        logger.info("🚀 Starting TradPal Monitoring Service Comprehensive Demo")
        logger.info("=" * 80)

        try:
            # Initialize all clients
            await self.initialize_clients()

            # Run all demo components
            await self.demo_mlops_workflow()
            await self.demo_discovery_optimization()
            await self.demo_model_monitoring()
            await self.demo_notification_system()
            await self.demo_service_integration()
            await self.demo_alert_forwarder_simulation()
            await self.demo_performance_monitoring()

            # Final summary
            logger.info("\n" + "=" * 80)
            logger.info("✅ Monitoring Service Comprehensive Demo Completed!")
            logger.info("🎯 Demonstrated Features:")
            logger.info("  • 🤖 MLOps: Experiment tracking, model deployment, drift detection")
            logger.info("  • 🔍 Discovery: Genetic algorithm parameter optimization")
            logger.info("  • 📊 Model Monitoring: Real-time performance and drift monitoring")
            logger.info("  • 📱 Notifications: Multi-channel alert delivery")
            logger.info("  • 🚨 Alert Forwarder: Security alert processing and forwarding")
            logger.info("  • 🔗 Integration: Cross-service communication and workflows")
            logger.info("  • ⚡ Performance: Health monitoring and response time tracking")

            # Show service availability summary
            available_services = sum([
                1 if self.mlops_client else 0,
                1 if self.discovery_client else 0,
                1 if self.monitoring_client else 0,
                1 if self.notification_client else 0
            ])
            logger.info(f"📊 Services Available: {available_services}/4")
            logger.info("💡 Start individual services to see full functionality")

        except Exception as e:
            logger.error(f"Demo failed: {e}")
            raise
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up demo resources...")

        # Close all clients
        clients_to_close = [
            ("MLOps", self.mlops_client),
            ("Discovery", self.discovery_client),
            ("Model Monitoring", self.monitoring_client),
            ("Notification", self.notification_client)
        ]

        for service_name, client in clients_to_close:
            if client:
                try:
                    await client.close()
                    logger.info(f"✅ {service_name} client closed")
                except Exception as e:
                    logger.warning(f"Error closing {service_name} client: {e}")

        logger.info("✅ Cleanup complete")


async def main():
    """Main demo function"""
    demo = MonitoringServiceDemo()
    await demo.run_comprehensive_demo()


if __name__ == "__main__":
    # Run the comprehensive demo
    asyncio.run(main())</content>
<parameter name="filePath">/Users/danielsadowski/VSCodeProjects/tradpal/tradpal/services/monitoring_service/demo.py