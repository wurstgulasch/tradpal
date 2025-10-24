#!/usr/bin/env python3
"""
Test Script for Model Monitoring Service
Demonstrates basic functionality and integration
"""

import asyncio
import sys
import os

# Add service path for imports
sys.path.insert(0, os.path.dirname(__file__))

from monitoring import DriftDetector, PerformanceTracker, AlertManager


async def test_monitoring_modules():
    """Test the monitoring modules directly."""

    print("🧪 Testing Model Monitoring Modules")
    print("=" * 50)

    try:
        # Test 1: Drift Detector
        print("\n1. Testing Drift Detector...")
        detector = DriftDetector()

        baseline_features = {
            "rsi": [30.5, 45.2, 67.8, 52.1, 71.3, 48.9, 62.4],
            "macd": [0.12, -0.05, 0.33, -0.08, 0.21, -0.15, 0.08],
            "volume_ratio": [0.8, 1.2, 0.9, 1.5, 0.7, 1.1, 0.95]
        }

        detector.register_model("test_trading_model", baseline_features, threshold=0.15)
        print("✅ Model registered for drift detection")

        # Test drift calculation
        drift_score = detector.calculate_drift({
            "rsi": 65.4,
            "macd": 0.12,
            "volume_ratio": 1.1
        }, "test_trading_model")
        print(f"✅ Drift score calculated: {drift_score:.4f}")

        # Test 2: Performance Tracker
        print("\n2. Testing Performance Tracker...")
        tracker = PerformanceTracker()

        baseline_metrics = {
            "mse": 0.0234,
            "directional_accuracy": 0.67,
            "mae": 0.0456
        }

        tracker.register_model("test_trading_model", baseline_metrics)
        print("✅ Model registered for performance tracking")

        # Track some predictions
        tracker.track_prediction("test_trading_model", 0.0234, 0.0189)
        tracker.track_prediction("test_trading_model", -0.0123, -0.0087)
        tracker.track_prediction("test_trading_model", 0.0345, 0.0412)
        print("✅ Predictions tracked")

        metrics = tracker.get_current_metrics("test_trading_model")
        mse_value = metrics.get('mse', 'N/A')
        mse_str = f"{mse_value:.4f}" if isinstance(mse_value, (int, float)) else str(mse_value)
        print(f"✅ Current metrics: MSE={mse_str}")

        # Test 3: Alert Manager
        print("\n3. Testing Alert Manager...")
        alert_manager = AlertManager()

        # Generate test alerts
        alert_id1 = alert_manager.generate_alert(
            "test_trading_model",
            "drift",
            "Drift detected: score 0.087 exceeds threshold 0.1",
            severity="warning"
        )
        print(f"✅ Alert generated: {alert_id1}")

        alert_id2 = alert_manager.generate_alert(
            "test_trading_model",
            "performance",
            "Performance degradation detected",
            severity="error"
        )
        print(f"✅ Alert generated: {alert_id2}")

        # Check active alerts
        active_alerts = alert_manager.get_active_alerts("test_trading_model")
        print(f"✅ Active alerts: {len(active_alerts)}")

        # Resolve an alert
        alert_manager.resolve_alert(alert_id1, "Auto-resolved")
        print("✅ Alert resolved")

        # Test 4: Integration Test
        print("\n4. Testing Module Integration...")

        # Simulate monitoring workflow
        print("   - Registering model...")
        detector.register_model("integration_test", baseline_features)
        tracker.register_model("integration_test", baseline_metrics)

        print("   - Processing predictions...")
        predictions = [
            (0.0234, 0.0189, {"rsi": 65.4, "macd": 0.12, "volume_ratio": 1.1}),
            (-0.0123, -0.0087, {"rsi": 45.2, "macd": -0.05, "volume_ratio": 0.9}),
            (0.0345, 0.0412, {"rsi": 72.1, "macd": 0.28, "volume_ratio": 1.3}),
        ]

        for pred, actual, features in predictions:
            tracker.track_prediction("integration_test", pred, actual)
            drift_score = detector.calculate_drift(features, "integration_test")

            # Check for alerts
            if drift_score > 0.1:
                alert_manager.generate_alert(
                    "integration_test",
                    "drift",
                    f"High drift detected: {drift_score:.4f}",
                    severity="warning"
                )

        degradation_alerts = tracker.detect_performance_degradation("integration_test")
        if degradation_alerts:
            alert_manager.generate_alert(
                "integration_test",
                "performance",
                f"Performance issues: {', '.join(degradation_alerts)}",
                severity="error"
            )

        print("   - Checking final status...")
        final_drift = detector.get_drift_score("integration_test")
        final_metrics = tracker.get_current_metrics("integration_test")
        final_alerts = alert_manager.get_active_alerts("integration_test")

        print(f"   - Final drift score: {final_drift:.4f}")
        mse_value = final_metrics.get('mse', 'N/A')
        mse_str = f"{mse_value:.4f}" if isinstance(mse_value, (int, float)) else str(mse_value)
        print(f"   - Final MSE: {mse_str}")
        print(f"   - Final alerts: {len(final_alerts)}")

        print("\n🎉 All module tests completed successfully!")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run tests
    success = asyncio.run(test_monitoring_modules())

    if not success:
        sys.exit(1)