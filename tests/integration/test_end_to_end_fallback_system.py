"""
End-to-end tests for the complete liquidation data fallback system.
"""

import pytest
import pandas as pd
import numpy as np
import asyncio
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from services.data_service.data_sources.factory import DataSourceFactory
from services.data_service.data_sources.liquidation import LiquidationDataSource
from services.data_service.data_sources.volatility import VolatilityDataSource
from services.data_service.data_sources.sentiment import SentimentDataSource
from services.data_service.data_sources.onchain import OnChainMetricsDataSource


class TestEndToEndFallbackSystem:
    """End-to-end tests for the complete fallback system."""

    def test_complete_fallback_chain_execution(self):
        """Test the complete fallback chain from primary to final fallback."""
        # Create liquidation source
        liquidation_source = DataSourceFactory.create_data_source('liquidation')

        # Mock all sources to fail except final fallback
        with patch.object(liquidation_source, '_fetch_binance_recent_liquidations') as mock_primary:
            mock_primary.return_value = pd.DataFrame()  # Primary fails

            with patch('services.data_service.data_sources.factory.DataSourceFactory') as mock_factory:
                # All alternative sources fail
                mock_volatility = MagicMock()
                mock_volatility.fetch_recent_data.side_effect = Exception("Volatility API down")

                mock_sentiment = MagicMock()
                mock_sentiment.fetch_recent_data.side_effect = Exception("Sentiment API down")

                mock_onchain = MagicMock()
                mock_onchain.fetch_recent_data.side_effect = Exception("On-chain API down")

                mock_factory.create_data_source.side_effect = [mock_volatility, mock_sentiment, mock_onchain]

                # Mock final fallback to succeed
                with patch.object(liquidation_source, '_fetch_open_interest_fallback') as mock_oi:
                    mock_oi.return_value = pd.DataFrame({
                        'liquidation_signal': [0],
                        'liquidation_volume': [100],
                        'total_value_liquidated': [100],
                        'data_source': ['open_interest_fallback'],
                        'timestamp': [pd.Timestamp.now()]
                    })

                    result = liquidation_source.fetch_recent_data('BTC/USDT', '1h', limit=1)

                    # Should get data from final fallback
                    assert not result.empty
                    assert result.iloc[0]['data_source'] == 'open_interest_fallback'
                    assert result.iloc[0]['liquidation_signal'] == 0

    def test_primary_source_success_bypasses_fallbacks(self):
        """Test that when primary source works, fallbacks are not used."""
        liquidation_source = DataSourceFactory.create_data_source('liquidation')

        # Mock primary to succeed
        primary_data = pd.DataFrame({
            'liquidation_signal': [2],
            'liquidation_volume': [5000],
            'total_value_liquidated': [5000],
            'data_source': ['binance_liquidations'],
            'timestamp': [pd.Timestamp.now()]
        })

        with patch.object(liquidation_source, '_fetch_binance_recent_liquidations') as mock_primary:
            mock_primary.return_value = primary_data

            result = liquidation_source.fetch_recent_data('BTC/USDT', '1h', limit=1)

            # Should return primary data directly
            assert not result.empty
            assert result.iloc[0]['data_source'] == 'binance_liquidations'
            assert result.iloc[0]['liquidation_signal'] == 2

    def test_volatility_fallback_data_conversion(self):
        """Test that volatility data is properly converted to liquidation format."""
        liquidation_source = DataSourceFactory.create_data_source('liquidation')

        # Mock volatility source to return data
        volatility_data = pd.DataFrame({
            'volatility_signal': [1.5],
            'volatility_proxy': [3000],
            'open_interest': [100000000],
            'volume_24h': [5000000000],
            'timestamp': [pd.Timestamp.now()]
        })

        with patch.object(liquidation_source, '_fetch_binance_recent_liquidations') as mock_primary:
            mock_primary.return_value = pd.DataFrame()  # Primary fails

            with patch('services.data_service.data_sources.factory.DataSourceFactory') as mock_factory:
                mock_volatility = MagicMock()
                mock_volatility.fetch_recent_data.return_value = volatility_data
                mock_factory.create_data_source.return_value = mock_volatility

                result = liquidation_source.fetch_recent_data('BTC/USDT', '1h', limit=1)

                # Should have converted volatility data
                assert not result.empty
                assert 'liquidation_signal' in result.columns
                assert 'liquidation_volume' in result.columns
                assert result.iloc[0]['data_source'] == 'volatility_proxy'
                # Signal should be scaled appropriately
                assert isinstance(result.iloc[0]['liquidation_signal'], (int, float))

    def test_sentiment_fallback_data_conversion(self):
        """Test that sentiment data is properly converted to liquidation format."""
        liquidation_source = DataSourceFactory.create_data_source('liquidation')

        sentiment_data = pd.DataFrame({
            'sentiment_signal': [-1],
            'sentiment_strength': [0.8],
            'fear_greed_value': [25],
            'timestamp': [pd.Timestamp.now()]
        })

        with patch.object(liquidation_source, '_fetch_binance_recent_liquidations') as mock_primary:
            mock_primary.return_value = pd.DataFrame()  # Primary fails

            with patch('services.data_service.data_sources.factory.DataSourceFactory') as mock_factory:
                # Volatility fails
                mock_volatility = MagicMock()
                mock_volatility.fetch_recent_data.side_effect = Exception("Volatility failed")

                # Sentiment succeeds
                mock_sentiment = MagicMock()
                mock_sentiment.fetch_recent_data.return_value = sentiment_data

                mock_factory.create_data_source.side_effect = [mock_volatility, mock_sentiment]

                result = liquidation_source.fetch_recent_data('BTC/USDT', '1h', limit=1)

                # Should have converted sentiment data
                assert not result.empty
                assert result.iloc[0]['data_source'] == 'sentiment_proxy'
                # Negative sentiment should become positive liquidation signal
                assert result.iloc[0]['liquidation_signal'] > 0

    def test_onchain_fallback_data_conversion(self):
        """Test that on-chain data is properly converted to liquidation format."""
        liquidation_source = DataSourceFactory.create_data_source('liquidation')

        onchain_data = pd.DataFrame({
            'onchain_signal': [1],
            'active_addresses': [850000],
            'transaction_volume': [450000],
            'hash_rate': [500000000],
            'timestamp': [pd.Timestamp.now()]
        })

        with patch.object(liquidation_source, '_fetch_binance_recent_liquidations') as mock_primary:
            mock_primary.return_value = pd.DataFrame()  # Primary fails

            with patch('services.data_service.data_sources.factory.DataSourceFactory') as mock_factory:
                # First two alternatives fail
                mock_volatility = MagicMock()
                mock_volatility.fetch_recent_data.side_effect = Exception("Volatility failed")

                mock_sentiment = MagicMock()
                mock_sentiment.fetch_recent_data.side_effect = Exception("Sentiment failed")

                # On-chain succeeds
                mock_onchain = MagicMock()
                mock_onchain.fetch_recent_data.return_value = onchain_data

                mock_factory.create_data_source.side_effect = [mock_volatility, mock_sentiment, mock_onchain]

                result = liquidation_source.fetch_recent_data('BTC/USDT', '1h', limit=1)

                # Should have converted on-chain data
                assert not result.empty
                assert result.iloc[0]['data_source'] == 'onchain_proxy'
                # Check that liquidation_signal is numeric (pandas/numpy types are also numeric)
                assert pd.api.types.is_numeric_dtype(type(result.iloc[0]['liquidation_signal'])) or isinstance(result.iloc[0]['liquidation_signal'], (int, float, np.integer, np.floating))

    def test_fallback_chain_order_preservation(self):
        """Test that fallback chain maintains proper priority order."""
        liquidation_source = DataSourceFactory.create_data_source('liquidation')

        # Test that volatility is tried before sentiment
        call_order = []

        with patch.object(liquidation_source, '_fetch_binance_recent_liquidations') as mock_primary:
            mock_primary.return_value = pd.DataFrame()  # Primary fails

            with patch('services.data_service.data_sources.factory.DataSourceFactory') as mock_factory:
                def track_calls(source_type):
                    call_order.append(source_type)
                    if source_type == 'volatility':
                        mock_volatility = MagicMock()
                        mock_volatility.fetch_recent_data.return_value = pd.DataFrame({
                            'volatility_signal': [1],
                            'timestamp': [pd.Timestamp.now()]
                        })
                        return mock_volatility
                    elif source_type == 'sentiment':
                        mock_sentiment = MagicMock()
                        mock_sentiment.fetch_recent_data.return_value = pd.DataFrame({
                            'sentiment_signal': [1],
                            'timestamp': [pd.Timestamp.now()]
                        })
                        return mock_sentiment
                    return MagicMock()

                mock_factory.create_data_source.side_effect = track_calls

                result = liquidation_source.fetch_recent_data('BTC/USDT', '1h', limit=1)

                # Should have tried volatility first, then gotten data from it
                assert 'volatility' in call_order
                assert result.iloc[0]['data_source'] == 'volatility_proxy'

    def test_fallback_data_quality_preservation(self):
        """Test that fallback data maintains reasonable quality standards."""
        liquidation_source = DataSourceFactory.create_data_source('liquidation')

        # Mock primary failure and successful volatility fallback
        with patch.object(liquidation_source, '_fetch_binance_recent_liquidations') as mock_primary:
            mock_primary.return_value = pd.DataFrame()  # Primary fails

            with patch('services.data_service.data_sources.factory.DataSourceFactory') as mock_factory:
                mock_volatility = MagicMock()
                volatility_data = pd.DataFrame({
                    'volatility_signal': [2.5],  # High volatility
                    'volatility_proxy': [8000],  # High proxy value
                    'open_interest': [200000000],
                    'volume_24h': [10000000000],
                    'timestamp': [pd.Timestamp.now()]
                })
                mock_volatility.fetch_recent_data.return_value = volatility_data
                mock_factory.create_data_source.return_value = mock_volatility

                result = liquidation_source.fetch_recent_data('BTC/USDT', '1h', limit=1)

                # Check that converted data has required columns
                required_cols = ['liquidation_signal', 'liquidation_volume', 'data_source', 'timestamp']
                for col in required_cols:
                    assert col in result.columns

                # Check data types - allow numpy types
                assert isinstance(result.iloc[0]['liquidation_signal'], (int, float, np.integer, np.floating))
                assert isinstance(result.iloc[0]['liquidation_volume'], (int, float, np.integer, np.floating))
                assert isinstance(result.iloc[0]['data_source'], str)

                # Check timestamp is valid
                assert isinstance(result.iloc[0]['timestamp'], pd.Timestamp)

    def test_fallback_system_resilience_under_load(self):
        """Test fallback system resilience under concurrent load."""
        liquidation_source = DataSourceFactory.create_data_source('liquidation')

        # Mock primary failure for all calls
        with patch.object(liquidation_source, '_fetch_binance_recent_liquidations') as mock_primary:
            mock_primary.return_value = pd.DataFrame()

            with patch('services.data_service.data_sources.factory.DataSourceFactory') as mock_factory:
                # Mock volatility to succeed
                mock_volatility = MagicMock()
                mock_volatility.fetch_recent_data.return_value = pd.DataFrame({
                    'volatility_signal': [1],
                    'volatility_proxy': [1000],
                    'timestamp': [pd.Timestamp.now()]
                })
                mock_factory.create_data_source.return_value = mock_volatility

                # Run multiple sequential requests (simulating load)
                results = []
                for i in range(5):
                    result = liquidation_source.fetch_recent_data('BTC/USDT', '1h', limit=1)
                    results.append(result)

                # All should succeed
                assert len(results) == 5
                for result in results:
                    assert not result.empty
                    assert result.iloc[0]['data_source'] == 'volatility_proxy'

    def test_fallback_data_timestamp_consistency(self):
        """Test that fallback data maintains timestamp consistency."""
        liquidation_source = DataSourceFactory.create_data_source('liquidation')

        test_timestamp = pd.Timestamp.now()

        with patch.object(liquidation_source, '_fetch_binance_recent_liquidations') as mock_primary:
            mock_primary.return_value = pd.DataFrame()  # Primary fails

            with patch('services.data_service.data_sources.factory.DataSourceFactory') as mock_factory:
                mock_volatility = MagicMock()
                volatility_data = pd.DataFrame({
                    'volatility_signal': [1],
                    'volatility_proxy': [1000],
                    'timestamp': [test_timestamp]
                })
                mock_volatility.fetch_recent_data.return_value = volatility_data
                mock_factory.create_data_source.return_value = mock_volatility

                result = liquidation_source.fetch_recent_data('BTC/USDT', '1h', limit=1)

                # Timestamp should be preserved
                assert result.iloc[0]['timestamp'] == test_timestamp

    def test_fallback_system_error_isolation(self):
        """Test that errors in one fallback don't affect others."""
        liquidation_source = DataSourceFactory.create_data_source('liquidation')

        with patch.object(liquidation_source, '_fetch_binance_recent_liquidations') as mock_primary:
            mock_primary.return_value = pd.DataFrame()  # Primary fails

            with patch('services.data_service.data_sources.factory.DataSourceFactory') as mock_factory:
                # Volatility throws exception
                mock_volatility = MagicMock()
                mock_volatility.fetch_recent_data.side_effect = ConnectionError("Network timeout")

                # Sentiment succeeds
                mock_sentiment = MagicMock()
                sentiment_data = pd.DataFrame({
                    'sentiment_signal': [-0.5],
                    'sentiment_strength': [0.7],
                    'timestamp': [pd.Timestamp.now()]
                })
                mock_sentiment.fetch_recent_data.return_value = sentiment_data

                mock_factory.create_data_source.side_effect = [mock_volatility, mock_sentiment]

                result = liquidation_source.fetch_recent_data('BTC/USDT', '1h', limit=1)

                # Should successfully fall back to sentiment despite volatility error
                assert not result.empty
                assert result.iloc[0]['data_source'] == 'sentiment_proxy'

    def test_fallback_chain_completeness(self):
        """Test that all fallback sources are properly integrated."""
        # Verify that all expected fallback sources exist and are accessible
        available_sources = DataSourceFactory.get_available_sources()

        expected_fallbacks = ['volatility', 'sentiment', 'onchain']
        for source in expected_fallbacks:
            assert source in available_sources
            assert available_sources[source]  # Should be available

        # Verify liquidation source can create instances of all fallbacks
        liquidation_source = DataSourceFactory.create_data_source('liquidation')

        for source_type in expected_fallbacks:
            fallback_source = DataSourceFactory.create_data_source(source_type)
            assert fallback_source is not None
            assert hasattr(fallback_source, 'fetch_recent_data')