"""
Central test configuration and fixtures for TradPal

This conftest.py provides shared fixtures, test utilities, and configuration
for all test types (unit, integration, service, e2e).
"""

import asyncio
import os
import sys
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from unittest.mock import Mock, AsyncMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import settings (avoid circular imports by importing specific modules)
try:
    from config.core_settings import *
    from config.ml_settings import *
    from config.service_settings import *
except ImportError:
    # Fallback for test environment
    pass


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)  # For reproducible tests

    # Generate 1000 rows of sample data
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='1H')
    prices = np.random.normal(50000, 5000, 1000).cumsum() + 50000

    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        high = price + np.random.uniform(0, 1000)
        low = price - np.random.uniform(0, 1000)
        open_price = prices[i-1] if i > 0 else price
        close = price
        volume = np.random.uniform(100, 10000)

        data.append({
            'timestamp': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })

    return pd.DataFrame(data)


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    mock_redis = Mock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.set = AsyncMock(return_value=True)
    mock_redis.publish = AsyncMock(return_value=True)
    mock_redis.subscribe = AsyncMock(return_value=Mock())
    return mock_redis


@pytest.fixture
def mock_http_session():
    """Mock aiohttp ClientSession for testing."""
    session = AsyncMock()
    session.get = AsyncMock()
    session.post = AsyncMock()
    session.close = AsyncMock()
    return session


@pytest.fixture
def test_config():
    """Test configuration with safe defaults."""
    return {
        'trading': {
            'base_currency': 'USD',
            'quote_currency': 'BTC',
            'max_position_size': 1.0,
            'risk_per_trade': 0.02
        },
        'data': {
            'cache_enabled': False,  # Disable caching in tests
            'timeout': 5
        },
        'ml': {
            'model_cache_dir': '/tmp/test_models',
            'gpu_enabled': False  # Disable GPU in tests
        }
    }


@pytest.fixture
async def mock_service_client():
    """Mock service client for integration testing."""
    client = AsyncMock()
    client.authenticate = AsyncMock(return_value=True)
    client.health_check = AsyncMock(return_value={'status': 'healthy'})
    client.close = AsyncMock()
    return client


@pytest.fixture
def performance_metrics():
    """Sample performance metrics for testing."""
    return {
        'sharpe_ratio': 1.5,
        'max_drawdown': 0.15,
        'total_return': 0.25,
        'win_rate': 0.55,
        'profit_factor': 1.3,
        'calmar_ratio': 1.2
    }


@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Clean up temporary files after each test."""
    # This runs before each test
    temp_files = []

    yield temp_files

    # This runs after each test
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)


# Test markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line("markers", "asyncio: marks tests as async tests")
    config.addinivalue_line("markers", "service: marks tests as service tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")


# Test utilities
class TestUtils:
    """Utility functions for tests."""

    @staticmethod
    def create_mock_response(status: int = 200, json_data: Optional[Dict[str, Any]] = None):
        """Create a mock HTTP response."""
        response = Mock()
        response.status = status
        response.json = AsyncMock(return_value=json_data or {})
        response.text = AsyncMock(return_value="mock response")
        return response

    @staticmethod
    def generate_trading_signals(count: int = 100) -> pd.DataFrame:
        """Generate sample trading signals for testing."""
        np.random.seed(42)
        signals = []

        for i in range(count):
            signal = {
                'timestamp': datetime.now() - timedelta(hours=i),
                'signal': np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3]),
                'confidence': np.random.uniform(0.1, 1.0),
                'price': np.random.normal(50000, 1000)
            }
            signals.append(signal)

        return pd.DataFrame(signals)

    @staticmethod
    def mock_market_data(symbol: str = 'BTC/USDT', periods: int = 100) -> pd.DataFrame:
        """Generate mock market data."""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='1H')

        # Generate realistic price movements
        base_price = 50000
        price_changes = np.random.normal(0, 0.02, periods)  # 2% volatility
        prices = base_price * (1 + price_changes).cumprod()

        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            data.append({
                'timestamp': date,
                'open': prices[i-1] if i > 0 else price,
                'high': price * (1 + abs(np.random.normal(0, 0.01))),
                'low': price * (1 - abs(np.random.normal(0, 0.01))),
                'close': price,
                'volume': np.random.uniform(1000, 10000)
            })

        return pd.DataFrame(data)


# Make TestUtils available as a fixture
@pytest.fixture
def test_utils():
    """Provide test utilities."""
    return TestUtils()