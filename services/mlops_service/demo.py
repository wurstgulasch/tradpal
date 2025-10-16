#!/usr/bin/env python3
"""
MLOps Service Demo
Demonstrates MLflow experiment tracking, BentoML model deployment, and drift detection.
"""

import asyncio
import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import sys
import os
from typing import Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from service import MLOpsService, MLOpsConfig, MLflowConfig


class MLOpsDemo:
    """Demo class for MLOps service functionality."""

    def __init__(self):
        self.service: Optional[MLOpsService] = None
        self.logger = logging.getLogger(__name__)

    async def setup_service(self):
        """Setup the MLOps service."""
        print("🚀 Setting up MLOps Service...")

        # Configure for local demo (no server required)
        import tempfile
        import mlflow
        temp_dir = tempfile.mkdtemp()
        tracking_uri = f"file://{temp_dir}"
        mlflow.set_tracking_uri(tracking_uri)

        config = MLOpsConfig(
            mlflow=MLflowConfig(
                tracking_uri=tracking_uri,
                experiment_name="tradpal_trading_models"
            )
        )
        self.service = MLOpsService(config)

        await self.service.start()
        print("✅ MLOps Service started successfully")

    async def demo_experiment_tracking(self):
        """Demonstrate MLflow experiment tracking."""
        print("\n=== MLflow Experiment Tracking Demo ===")

        # Generate sample data
        X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a simple model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        # Log experiment
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall
        }

        parameters = {
            "n_estimators": 100,
            "random_state": 42,
            "model_type": "RandomForestClassifier"
        }

        result = await self.service.log_experiment(
            model_name="demo_trading_model",
            metrics=metrics,
            parameters=parameters,
            model=model,
            framework="sklearn"
        )

        print(f"✅ Experiment logged: {result.model_name}")
        print(f"   Run ID: {result.run_id}")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall: {recall:.3f}")
    async def demo_model_deployment(self):
        """Demonstrate BentoML model deployment."""
        print("\n=== BentoML Model Deployment Demo ===")

        # Create a simple model for deployment
        X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)

        # Deploy model
        deployment_path = await self.service.deploy_model(
            model_name="demo_deployed_model",
            model=model,
            version="1.0.0"
        )

        print(f"✅ Model deployed to BentoML: {deployment_path}")

    async def demo_drift_detection(self):
        """Demonstrate drift detection setup."""
        print("\n=== Drift Detection Demo ===")

        # Generate reference data
        reference_data = np.random.randn(1000, 10)

        # Create drift detector
        detector_id = await self.service.create_drift_detector(
            model_name="demo_drift_model",
            reference_data=reference_data
        )

        print(f"✅ Drift detector created: {detector_id}")
        print("   Reference data size: 1000 samples")
        print("   Features: 10")

    async def demo_service_monitoring(self):
        """Demonstrate service monitoring and statistics."""
        print("\n=== Service Monitoring Demo ===")

        # Get health status
        health = await self.service.health_check()
        print("🏥 Service Health:")
        print(f"   Status: {health['status']}")
        print(f"   MLflow: {'✅' if health['components']['mlflow'] else '❌'}")
        print(f"   BentoML: {'✅' if health['components']['bentoml'] else '❌'}")
        print(f"   Drift Detection: {'✅' if health['components']['drift_detection'] else '❌'}")
        print(f"   Models Deployed: {health['models_deployed']}")

        # Get experiment history
        experiments = await self.service.get_experiment_history(limit=5)
        print(f"\n📊 Recent Experiments: {len(experiments)}")

        for exp in experiments:
            print(f"   {exp.model_name} - Run {exp.run_id[:8]} - Accuracy: {exp.metrics.get('accuracy', 'N/A'):.3f}")

    async def run_demo(self):
        """Run the complete MLOps demo."""
        print("🎯 Starting MLOps Service Demo")
        print("=" * 50)

        try:
            # Setup service
            await self.setup_service()

            # Run demonstrations
            await self.demo_experiment_tracking()
            await self.demo_model_deployment()
            await self.demo_drift_detection()
            await self.demo_service_monitoring()

            print("\n✅ MLOps Service Demo completed successfully!")

        except Exception as e:
            print(f"❌ Demo failed: {e}")
            raise
        finally:
            if self.service:
                await self.service.stop()
                print("🛑 MLOps Service stopped")


async def main():
    """Main demo function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run demo
    demo = MLOpsDemo()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())