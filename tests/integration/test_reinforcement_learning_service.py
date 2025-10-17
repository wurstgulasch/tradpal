"""Integration tests for the Reinforcement Learning Service API."""

import asyncio
from typing import Any, Dict

import numpy as np
import httpx
import pytest
from unittest.mock import MagicMock, AsyncMock
import sys
import os

# Ensure project root is on path so `services` package can be imported
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from services.reinforcement_learning_service import main as rl_main
    from services.reinforcement_learning_service.rl_agent import TradingAction
    from services.reinforcement_learning_service.main import app
    SERVICE_AVAILABLE = True
except ImportError as e:
    SERVICE_AVAILABLE = False
    rl_main = None
    TradingAction = None
    app = None
    print(f"Reinforcement Learning Service not available: {e}")


class SyncASGIClient:
    """Synchronous wrapper for httpx.AsyncClient against ASGI apps."""

    def __init__(self, asgi_app: Any):
        self._client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=asgi_app),
            base_url="http://testserver"
        )

    def _run(self, coro):
        return asyncio.run(coro)

    def request(self, method: str, url: str, **kwargs):
        return self._run(self._client.request(method, url, **kwargs))

    def get(self, url: str, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs):
        return self.request("POST", url, **kwargs)

    def delete(self, url: str, **kwargs):
        return self.request("DELETE", url, **kwargs)

    def close(self):
        self._run(self._client.aclose())


@pytest.fixture
def client():
    """Yield a synchronous testing client for the RL service."""
    if not SERVICE_AVAILABLE or app is None:
        pytest.skip("Reinforcement Learning Service not available")

    client = SyncASGIClient(app)
    try:
        yield client
    finally:
        client.close()


def sample_training_payload() -> Dict[str, Any]:
    """Build minimal training payload."""
    market_entries = [
        {"timestamp": "2023-01-01T00:00:00Z", "close": 30000.0},
        {"timestamp": "2023-01-01T01:00:00Z", "close": 30100.0},
    ]
    return {
        "algorithm": "q_learning",
        "episodes": 10,
        "symbols": ["BTC/USDT"],
        "market_data": {"BTC/USDT": market_entries},
        "config": {"max_steps_per_episode": 10}
    }


@pytest.fixture(autouse=True)
def reset_training_tasks():
    """Ensure training task registry is empty for each test."""
    if not SERVICE_AVAILABLE:
        yield
        return

    rl_main.training_tasks.clear()
    yield
    rl_main.training_tasks.clear()


@pytest.mark.skipif(not SERVICE_AVAILABLE, reason="Reinforcement Learning Service package not importable")
class TestReinforcementLearningService:
    """Integration coverage for RL service endpoints."""

    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "reinforcement_learning"
        assert "agent_loaded" in data

    def test_get_trading_action(self, client, monkeypatch):
        class DummyAgent:
            def __init__(self):
                self.actions = [
                    TradingAction.BUY,
                    TradingAction.SELL,
                    TradingAction.HOLD,
                    TradingAction.REDUCE,
                    TradingAction.INCREASE,
                ]
                self.state_bins = {"position": 5, "price": 7, "trend": 5}
                self.q_table = np.array([[1.0, 0.5, 0.2, 0.1, 0.05]], dtype=np.float64)

            def get_action(self, _state, training: bool = False):
                return TradingAction.BUY

            def _state_to_index(self, _state):
                return 0

        monkeypatch.setattr(rl_main, "rl_agent", DummyAgent())
        monkeypatch.setattr(rl_main, "EVENT_SYSTEM_AVAILABLE", False)

        payload = {
            "symbol": "BTC/USDT",
            "position_size": 0.1,
            "current_price": 30000,
            "portfolio_value": 100000,
            "market_regime": "bull_market",
            "volatility_regime": "normal",
            "trend_strength": 0.7,
            "technical_indicators": {"rsi": 55.0, "macd": 0.01, "bb_position": 0.2}
        }

        response = client.post("/action", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["action"] == TradingAction.BUY.value
        assert "confidence" in data

    def test_get_trading_action_unavailable(self, client, monkeypatch):
        monkeypatch.setattr(rl_main, "rl_agent", None)

        payload = {
            "symbol": "BTC/USDT",
            "position_size": 0.0,
            "current_price": 30000,
            "portfolio_value": 100000,
            "market_regime": "sideways",
            "volatility_regime": "normal",
            "trend_strength": 0.0,
            "technical_indicators": {}
        }

        response = client.post("/action", json=payload)
        assert response.status_code == 503

    def test_start_training(self, client, monkeypatch):
        class DummyTrainer:
            def __init__(self):
                self.config = MagicMock()
                self.agent = MagicMock()

            async def train_async(self, market_data, progress_callback=None):
                self.market_data = market_data
                return {"status": "training_started"}

        dummy_trainer = DummyTrainer()
        monkeypatch.setattr(rl_main, "trainer", dummy_trainer)
        monkeypatch.setattr(rl_main, "EVENT_SYSTEM_AVAILABLE", False)

        payload = sample_training_payload()
        response = client.post("/train", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "training_started"
        assert data["task_id"] is not None

    def test_get_training_status_completed(self, client, monkeypatch):
        class CompletedTrainer:
            def __init__(self):
                self.config = MagicMock()
                self.config.episodes = 10
                self.agent = MagicMock()

            def get_training_status(self):
                return {"is_training": False, "current_episode": 10, "total_episodes": 10, "progress": 1.0, "metrics": {}}

            async def get_training_result(self):
                return {
                    "status": "completed",
                    "episodes_completed": 10,
                    "final_metrics": {"win_rate": 0.6},
                    "training_summary": {"avg_reward": 1.2}
                }

        trainer = CompletedTrainer()
        rl_main.training_tasks["task123"] = trainer
        monkeypatch.setattr(rl_main, "trainer", trainer)
        monkeypatch.setattr(rl_main, "EVENT_SYSTEM_AVAILABLE", False)
        monkeypatch.setattr(rl_main, "publish_rl_model_update", AsyncMock())

        response = client.get("/training/status/task123")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["progress"] == 1.0

    def test_stop_training(self, client, monkeypatch):
        class StoppableTrainer:
            def __init__(self):
                self.stopped = False

            def stop_training_async(self):
                self.stopped = True

        trainer = StoppableTrainer()
        rl_main.training_tasks["task456"] = trainer

        response = client.delete("/training/task456")
        assert response.status_code == 200
        assert response.json()["status"] == "stop_requested"
        assert trainer.stopped is True

    def test_get_model_info_success(self, client, monkeypatch):
        class DummyAgent:
            def __init__(self):
                self.config = {"algorithm": "q_learning"}
                self.state_bins = {"position": 5}
                self.actions = [TradingAction.BUY]
                self.q_table = np.array([[1.0]])

            def get_training_metrics(self):
                return {"episodes": 5}

        monkeypatch.setattr(rl_main, "rl_agent", DummyAgent())

        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert data["algorithm"] == "q_learning"

    def test_get_model_info_unavailable(self, client, monkeypatch):
        monkeypatch.setattr(rl_main, "rl_agent", None)

        response = client.get("/model/info")
        assert response.status_code == 503

    def test_list_algorithms(self, client):
        response = client.get("/algorithms")
        assert response.status_code == 200
        data = response.json()
        assert "q_learning" in data["algorithms"]

    def test_list_actions(self, client):
        response = client.get("/actions")
        assert response.status_code == 200
        data = response.json()
        assert "buy" in data["actions"]