"""
Unit tests for OnChainMetricsDataSource.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from services.data_service.data_sources.onchain import OnChainMetricsDataSource


class TestOnChainMetricsDataSource:
    """Test cases for OnChainMetricsDataSource."""

    @pytest.fixture
    def onchain_source(self):
        """Create OnChainMetricsDataSource instance for testing."""
        return OnChainMetricsDataSource()

    @pytest.fixture
    def onchain_source_with_api(self):
        """Create OnChainMetricsDataSource with API key."""
        config = {'api_key': 'test_api_key'}
        return OnChainMetricsDataSource(config)

    def test_initialization_no_api(self, onchain_source):
        """Test OnChainMetricsDataSource initialization without API key."""
        assert onchain_source.name == "On-Chain Metrics"
        assert onchain_source.config['glassnode_api'] == 'https://api.glassnode.com/v1/metrics'
        assert onchain_source.use_simulated_data is True

    def test_initialization_with_api(self, onchain_source_with_api):
        """Test OnChainMetricsDataSource initialization with API key."""
        assert onchain_source_with_api.use_simulated_data is False

    def test_fetch_recent_data_simulated(self, onchain_source):
        """Test fetch of recent on-chain data with simulated data."""
        result = onchain_source.fetch_recent_data('BTC/USDT', '1h', limit=1)

        assert not result.empty
        assert len(result) == 1
        assert 'onchain_signal' in result.columns
        assert 'onchain_strength' in result.columns
        assert 'network_health_proxy' in result.columns
        assert 'active_addresses' in result.columns
        assert 'transaction_volume' in result.columns

    @patch('services.data_service.data_sources.onchain.requests.get')
    def test_fetch_glassnode_recent_data_success(self, mock_get, onchain_source_with_api):
        """Test successful Glassnode API call for recent data."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            [1638360000, 1000000],  # active_addresses
            [1638360000, 500000],   # transaction_volume
            [1638360000, 400000000], # hash_rate
            [1638360000, 10000]     # exchange_net_flow
        ]
        mock_get.return_value = mock_response

        result = onchain_source_with_api._fetch_glassnode_recent_data()

        assert result is not None
        assert 'active_addresses' in result
        assert 'transaction_volume' in result
        assert 'hash_rate' in result
        assert 'exchange_net_flow' in result

    @patch('services.data_service.data_sources.onchain.requests.get')
    def test_fetch_glassnode_recent_data_failure(self, mock_get, onchain_source_with_api):
        """Test Glassnode API call failure."""
        mock_get.side_effect = Exception("API Error")

        result = onchain_source_with_api._fetch_glassnode_recent_data()

        # Should return dict with default values, not None
        assert isinstance(result, dict)
        assert 'active_addresses' in result
        assert 'transaction_volume' in result
        assert 'hash_rate' in result
        assert 'exchange_net_flow' in result
        # All values should be 0 due to API failure
        assert result['active_addresses'] == 0
        assert result['transaction_volume'] == 0
        assert result['hash_rate'] == 0
        assert result['exchange_net_flow'] == 0

    def test_generate_current_onchain_data(self, onchain_source):
        """Test generation of current synthetic on-chain data."""
        result = onchain_source._generate_current_onchain_data()

        assert 'timestamp' in result
        assert 'active_addresses' in result
        assert 'transaction_volume' in result
        assert 'hash_rate' in result
        assert 'exchange_net_flow' in result

        assert result['active_addresses'] > 0
        assert result['transaction_volume'] > 0
        assert result['hash_rate'] > 0

    def test_generate_synthetic_onchain_data(self, onchain_source):
        """Test generation of synthetic historical on-chain data."""
        start_date = datetime.now() - timedelta(days=3)
        end_date = datetime.now()

        result = onchain_source._generate_synthetic_onchain_data(start_date, end_date, limit=24)

        assert not result.empty
        assert len(result) <= 24
        assert 'active_addresses' in result.columns
        assert 'transaction_volume' in result.columns
        assert 'hash_rate' in result.columns
        assert 'exchange_net_flow' in result.columns
        assert result.index.name == 'timestamp'

    def test_add_simulated_metrics(self, onchain_source):
        """Test adding simulated metrics to existing data."""
        base_data = pd.DataFrame({
            'active_addresses': [800000, 850000, 900000]
        })
        base_data.index = pd.date_range(start='2023-01-01', periods=3, freq='H')

        result = onchain_source._add_simulated_metrics(base_data)

        assert 'transaction_volume' in result.columns
        assert 'hash_rate' in result.columns
        assert 'exchange_net_flow' in result.columns
        assert len(result) == 3

    def test_calculate_onchain_signals(self, onchain_source):
        """Test on-chain signal calculation."""
        # Create test data with enough rows for rolling calculations
        dates = pd.date_range(start='2023-01-01', periods=30, freq='H')
        data = pd.DataFrame({
            'active_addresses': [800000 + i*1000 for i in range(30)],
            'transaction_volume': [500000 + i*5000 for i in range(30)],
            'hash_rate': [400000000 + i*1000000 for i in range(30)],
            'exchange_net_flow': [10000 + i*500 for i in range(30)]
        }, index=dates)

        result = onchain_source._calculate_onchain_signals(data)

        assert 'onchain_signal' in result.columns
        assert 'onchain_strength' in result.columns
        assert 'network_health_proxy' in result.columns

        # Check that signals are calculated (non-zero for some entries)
        assert not result['onchain_signal'].isna().all()

    def test_calculate_onchain_signals_insufficient_data(self, onchain_source):
        """Test on-chain signal calculation with insufficient data."""
        data = pd.DataFrame({
            'active_addresses': [800000, 850000],
            'transaction_volume': [500000, 550000],
            'hash_rate': [400000000, 410000000],
            'exchange_net_flow': [10000, 10500]
        })

        result = onchain_source._calculate_onchain_signals(data)

        # With insufficient data, signals should remain default values
        assert 'onchain_signal' in result.columns
        assert 'onchain_strength' in result.columns
        assert 'network_health_proxy' in result.columns

    def test_validate_data_valid(self, onchain_source):
        """Test data validation with valid data."""
        data = pd.DataFrame({
            'onchain_signal': [1, -1, 0],
            'onchain_strength': [0.8, 0.6, 0.3],
            'network_health_proxy': [50, -25, 0],
            'active_addresses': [800000, 850000, 900000]
        })

        result = onchain_source.validate_data(data)
        assert result is True

    def test_validate_data_missing_columns(self, onchain_source):
        """Test data validation with missing columns."""
        data = pd.DataFrame({
            'some_other_column': [1, 2, 3]
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            onchain_source.validate_data(data)

    def test_get_onchain_stats_simulated(self, onchain_source):
        """Test on-chain statistics retrieval with simulated data."""
        stats = onchain_source.get_onchain_stats('BTC/USDT')

        assert 'current_onchain_signal' in stats
        assert 'current_onchain_strength' in stats
        assert 'active_addresses' in stats
        assert 'transaction_volume' in stats
        assert 'hash_rate' in stats
        assert 'exchange_net_flow' in stats
        assert 'data_source' in stats
        assert stats['data_source'] == 'simulated'

    def test_fetch_historical_data_simulated(self, onchain_source):
        """Test fetch of historical on-chain data with simulated data."""
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()

        result = onchain_source.fetch_historical_data(
            'BTC/USDT', '1h', start_date, end_date, limit=24
        )

        assert not result.empty
        assert len(result) <= 24
        assert 'onchain_signal' in result.columns
        assert 'onchain_strength' in result.columns
        assert result.index.name == 'timestamp'

    @patch('services.data_service.data_sources.onchain.requests.get')
    def test_fetch_glassnode_historical_data_success(self, mock_get, onchain_source_with_api):
        """Test successful Glassnode API call for historical data."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            [1638360000, 800000],
            [1638363600, 850000],
            [1638367200, 900000]
        ]
        mock_get.return_value = mock_response

        start_date = datetime.now() - timedelta(days=1)
        end_date = datetime.now()

        result = onchain_source_with_api._fetch_glassnode_historical_data(start_date, end_date, limit=10)

        assert not result.empty
        assert 'active_addresses' in result.columns
        assert result.index.name == 'timestamp'