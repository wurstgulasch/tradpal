import asyncio
from typing import Any, Dict

import pytest
import pandas as pd
import numpy as np
import httpx
from unittest.mock import MagicMock, patch
import sys
import os

# Add services to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'services'))

try:
    from market_regime_service.main import app
    SERVICE_AVAILABLE = True
except ImportError as e:
    SERVICE_AVAILABLE = False
    print(f"Market Regime Service not available: {e}")


class SyncASGIClient:
    """Synchronous wrapper around httpx.AsyncClient for ASGI apps."""

    def __init__(self, app):
        self._client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        )

    def _run(self, coro):
        return asyncio.run(coro)

    def request(self, method, url, **kwargs):
        return self._run(self._client.request(method, url, **kwargs))

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)

    def options(self, url, **kwargs):
        return self.request("OPTIONS", url, **kwargs)

    def close(self):
        self._run(self._client.aclose())


@pytest.fixture
def client():
    """Provide synchronous testing client"""
    client = SyncASGIClient(app)
    try:
        yield client
    finally:
        client.close()


def sample_market_payload() -> Dict[str, Any]:
    """Build minimal market payload expected by the API."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=50, freq='H')
    return {
        "symbol": "BTC/USDT",
        "data": {
            "timestamp": dates.astype(str).tolist(),
            "open": np.random.uniform(40000, 50000, len(dates)).tolist(),
            "high": np.random.uniform(41000, 51000, len(dates)).tolist(),
            "low": np.random.uniform(39000, 49000, len(dates)).tolist(),
            "close": np.random.uniform(40000, 50000, len(dates)).tolist(),
            "volume": np.random.uniform(1_000_000, 5_000_000, len(dates)).tolist(),
        },
        "include_signals": True,
        "background_analysis": False,
    }


@pytest.mark.skipif(not SERVICE_AVAILABLE, reason="Market Regime Service not available")
class TestMarketRegimeServiceIntegration:
    """Integration tests aligned with current Market Regime Service API."""

    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "market_regime_detection"

    def test_analyze_market_regime(self, client):
        payload = sample_market_payload()
        mock_result = {
            "analysis": {
                "timestamp": "2023-01-01T00:00:00Z",
                "symbol": payload["symbol"],
                "current_regime": "bull",
                "regime_confidence": 0.82,
                "volatility_regime": "low",
                "trend_strength": 0.75,
                "feature_importance": {"momentum": 0.5, "volatility": 0.3}
            },
            "signal": {
                "timestamp": "2023-01-01T00:00:00Z",
                "symbol": payload["symbol"],
                "regime": "bull",
                "signal": "buy",
                "confidence": 0.82,
                "risk_level": "medium",
                "position_size_multiplier": 1.2,
                "stop_loss_multiplier": 0.9,
                "take_profit_multiplier": 1.4,
                "reasoning": "Strong momentum"
            }
        }

        async def _mock_perform_analysis(*_args, **_kwargs):
            return mock_result

        with patch("market_regime_service.main.regime_analyzer", MagicMock()):
            with patch("market_regime_service.main.perform_analysis", _mock_perform_analysis):
                response = client.post("/analyze", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["analysis"]["current_regime"] == "bull"
        assert "signal" in data

    def test_analyze_missing_columns(self, client):
        payload = sample_market_payload()
        del payload["data"]["close"]

        with patch("market_regime_service.main.regime_analyzer", MagicMock()):
            response = client.post("/analyze", json=payload)

        assert response.status_code == 400
        assert "Missing required columns" in response.json()["detail"]

    def test_batch_analyze(self, client):
        payload = sample_market_payload()
        mock_result = {"analysis": {"current_regime": "bear"}}

        async def _mock_perform_analysis(*_args, **_kwargs):
            return mock_result

        with patch("market_regime_service.main.regime_analyzer", MagicMock()):
            with patch("market_regime_service.main.perform_analysis", _mock_perform_analysis):
                response = client.post("/batch_analyze", json=[payload])

        assert response.status_code == 200
        data = response.json()
        assert data["total_requested"] == 1
        assert data["completed"] == 1
        assert data["batch_results"][0]["analysis"]["current_regime"] == "bear"

    def test_get_regime_statistics(self, client):
        mock_stats = {
            "total_observations": 1000,
            "regime_frequencies": {"bull": 600, "bear": 300, "sideways": 100},
            "most_common_regime": "bull",
            "avg_duration_days": {"bull": 20.5, "bear": 10.2, "sideways": 5.3},
            "observation_period_days": 365
        }

        with patch("market_regime_service.main.regime_analyzer") as mock_analyzer:
            mock_analyzer.get_regime_statistics.return_value = mock_stats
            response = client.get("/statistics/BTC-USDT")

        assert response.status_code == 200
        data = response.json()
        assert data["most_common_regime"] == "bull"
        assert data["observation_period_days"] == 365

    def test_get_regime_statistics_not_found(self, client):
        with patch("market_regime_service.main.regime_analyzer") as mock_analyzer:
            mock_analyzer.get_regime_statistics.return_value = {"error": "Symbol not found"}
            response = client.get("/statistics/UNKNOWN")

        assert response.status_code == 404

    def test_predict_regime(self, client):
        mock_prediction = {
            "current_regime": "sideways",
            "days_in_current_regime": 15,
            "expected_duration_days": 30,
            "transition_probability": 0.42,
            "predicted_next_regimes": {"bull": 0.3, "bear": 0.28, "sideways": 0.42},
            "horizon_days": 30
        }

        with patch("market_regime_service.main.regime_analyzer") as mock_analyzer:
            mock_analyzer.get_regime_prediction.return_value = mock_prediction
            response = client.get("/predict/BTC-USDT?horizon_days=30")

        assert response.status_code == 200
        data = response.json()
        assert data["current_regime"] == "sideways"
        assert data["transition_probability"] == 0.42

    def test_predict_regime_not_found(self, client):
        with patch("market_regime_service.main.regime_analyzer") as mock_analyzer:
            mock_analyzer.get_regime_prediction.return_value = {"error": "Symbol not found"}
            response = client.get("/predict/UNKNOWN")

        assert response.status_code == 404

    def test_list_available_regimes(self, client):
        response = client.get("/regimes")
        assert response.status_code == 200
        data = response.json()
        assert "regimes" in data
        assert any(value.startswith("bull") for value in data["regimes"])

    def test_service_uninitialized_returns_503(self, client):
        payload = sample_market_payload()

        with patch("market_regime_service.main.regime_analyzer", None):
            response = client.post("/analyze", json=payload)

        assert response.status_code == 503
        assert "Service not initialized" in response.json()["detail"]

    def test_openapi_docs_available(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "/analyze" in data["paths"]
