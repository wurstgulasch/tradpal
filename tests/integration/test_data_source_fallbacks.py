"""
Integration tests for data source fallback chain and alternative data sources.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from services.data_service.data_sources.factory import DataSourceFactory
from services.data_service.data_sources.liquidation import LiquidationDataSource
from services.data_service.data_sources.volatility import VolatilityDataSource
from services.data_service.data_sources.sentiment import SentimentDataSource
from services.data_service.data_sources.onchain import OnChainMetricsDataSource


class TestDataSourceFallbackChain:
    """Integration tests for data source fallback chain."""

    def test_factory_creates_all_data_sources(self):
        """Test that factory can create all data source types."""
        available_sources = DataSourceFactory.get_available_sources()

        # Check that all expected sources are available
        expected_sources = [
            'yahoo_finance', 'ccxt', 'kaggle', 'funding_rate',
            'liquidation', 'volatility', 'sentiment', 'onchain'
        ]

        for source in expected_sources:
            assert source in available_sources
            if source in ['liquidation', 'volatility', 'sentiment', 'onchain']:
                # These should be available in our setup
                assert available_sources[source] is True

    def test_liquidation_fallback_to_volatility(self):
        """Test liquidation data source falls back to volatility when primary fails."""
        # Create liquidation source
        liquidation_source = DataSourceFactory.create_data_source('liquidation')

        # Mock the primary liquidation fetch to fail
        with patch.object(liquidation_source, '_fetch_binance_recent_liquidations') as mock_primary:
            mock_primary.return_value = pd.DataFrame()  # Empty = failed

            # Mock volatility data source to succeed
            with patch('services.data_service.data_sources.factory.DataSourceFactory') as mock_factory:
                mock_volatility = MagicMock()
                mock_volatility.fetch_recent_data.return_value = pd.DataFrame({
                    'volatility_signal': [1],
                    'volatility_proxy': [1000],
                    'timestamp': pd.Timestamp.now()
                })
                mock_factory.create_data_source.return_value = mock_volatility

                result = liquidation_source.fetch_recent_data('BTC/USDT', '1h', limit=1)

                # Should have converted volatility data to liquidation format
                assert not result.empty
                assert 'liquidation_signal' in result.columns
                assert 'data_source' in result.columns
                assert result.iloc[0]['data_source'] == 'volatility_proxy'

    def test_liquidation_fallback_to_sentiment(self):
        """Test liquidation data source falls back to sentiment when volatility fails."""
        liquidation_source = DataSourceFactory.create_data_source('liquidation')

        # Mock all previous fallbacks to fail
        with patch.object(liquidation_source, '_fetch_binance_recent_liquidations') as mock_primary:
            mock_primary.return_value = pd.DataFrame()

            # Mock the factory to return different sources for different calls
            with patch('services.data_service.data_sources.factory.DataSourceFactory') as mock_factory:
                # Volatility fails
                mock_volatility = MagicMock()
                mock_volatility.fetch_recent_data.side_effect = Exception("Volatility failed")

                # Sentiment succeeds
                mock_sentiment = MagicMock()
                mock_sentiment.fetch_recent_data.return_value = pd.DataFrame({
                    'sentiment_signal': [-2],
                    'sentiment_strength': [0.8],
                    'timestamp': [pd.Timestamp.now()]
                })

                # Factory returns volatility first, then sentiment
                mock_factory.create_data_source.side_effect = [mock_volatility, mock_sentiment]

                # Should fall back to sentiment
                result = liquidation_source.fetch_recent_data('BTC/USDT', '1h', limit=1)

                # Should have sentiment data converted to liquidation format
                assert not result.empty
                assert 'liquidation_signal' in result.columns
                assert 'data_source' in result.columns
                assert result.iloc[0]['data_source'] == 'sentiment_proxy'

    def test_liquidation_fallback_to_onchain(self):
        """Test liquidation data source falls back to on-chain when sentiment fails."""
        liquidation_source = DataSourceFactory.create_data_source('liquidation')

        # Mock all previous fallbacks to fail
        with patch.object(liquidation_source, '_fetch_binance_recent_liquidations') as mock_primary:
            mock_primary.return_value = pd.DataFrame()

            with patch('services.data_service.data_sources.factory.DataSourceFactory') as mock_factory:
                # Volatility fails
                mock_volatility = MagicMock()
                mock_volatility.fetch_recent_data.side_effect = Exception("Volatility failed")

                # Sentiment fails
                mock_sentiment = MagicMock()
                mock_sentiment.fetch_recent_data.side_effect = Exception("Sentiment failed")

                # On-chain succeeds
                mock_onchain = MagicMock()
                mock_onchain.fetch_recent_data.return_value = pd.DataFrame({
                    'onchain_signal': [1],
                    'onchain_strength': [0.6],
                    'timestamp': [pd.Timestamp.now()]
                })

                # Factory returns volatility, sentiment, then onchain
                mock_factory.create_data_source.side_effect = [mock_volatility, mock_sentiment, mock_onchain]

                # Should fall back to on-chain
                result = liquidation_source.fetch_recent_data('BTC/USDT', '1h', limit=1)

                # Should have on-chain data converted to liquidation format
                assert not result.empty
                assert 'liquidation_signal' in result.columns
                assert 'data_source' in result.columns
                assert result.iloc[0]['data_source'] == 'onchain_proxy'

    def test_liquidation_fallback_to_open_interest(self):
        """Test liquidation data source falls back to open interest as final fallback."""
        liquidation_source = DataSourceFactory.create_data_source('liquidation')

        # Mock all previous fallbacks to fail
        with patch.object(liquidation_source, '_fetch_binance_recent_liquidations') as mock_primary:
            mock_primary.return_value = pd.DataFrame()

            with patch('services.data_service.data_sources.factory.DataSourceFactory') as mock_factory:
                # All alternative sources fail
                mock_volatility = MagicMock()
                mock_volatility.fetch_recent_data.side_effect = Exception("All alternatives failed")

                mock_sentiment = MagicMock()
                mock_sentiment.fetch_recent_data.side_effect = Exception("All alternatives failed")

                mock_onchain = MagicMock()
                mock_onchain.fetch_recent_data.side_effect = Exception("All alternatives failed")

                mock_factory.create_data_source.side_effect = [
                    mock_volatility, mock_sentiment, mock_onchain
                ]

                # Mock open interest fallback to succeed
                with patch.object(liquidation_source, '_fetch_open_interest_fallback') as mock_oi:
                    mock_oi.return_value = pd.DataFrame({
                        'liquidation_signal': [0],
                        'liquidation_volume': [100],
                        'total_value_liquidated': [100],
                        'data_source': ['open_interest_fallback'],
                        'timestamp': [pd.Timestamp.now()]
                    })

                    result = liquidation_source.fetch_recent_data('BTC/USDT', '1h', limit=1)

                    assert not result.empty
                    assert result.iloc[0]['data_source'] == 'open_interest_fallback'

    def test_volatility_data_source_comprehensive(self):
        """Test volatility data source provides comprehensive market data."""
        volatility_source = DataSourceFactory.create_data_source('volatility')

        # Mock API calls to avoid real network requests
        with patch.object(volatility_source, '_fetch_binance_current_volatility') as mock_fetch:
            # Create mock data that simulates the combined DataFrame
            mock_df = pd.DataFrame({
                'open_interest': [100000000],
                'volume_24h': [5000000000],
                'funding_rate': [0.0001],
                'order_book_imbalance': [0.3],
                'recent_trades_ratio': [1.2],
                'price_momentum': [0.5],
                'volatility_signal': [2],
                'volatility_proxy': [5000],
                'timestamp': [pd.Timestamp.now()]
            })
            mock_df = mock_df.set_index('timestamp')
            mock_fetch.return_value = mock_df

            result = volatility_source.fetch_recent_data('BTC/USDT', '1h', limit=1)

            assert not result.empty
            assert 'volatility_signal' in result.columns
            assert 'volatility_proxy' in result.columns
            assert 'open_interest' in result.columns
            assert 'volume_24h' in result.columns
            assert 'funding_rate' in result.columns

    def test_sentiment_data_source_fear_greed_integration(self):
        """Test sentiment data source integrates Fear & Greed Index properly."""
        sentiment_source = DataSourceFactory.create_data_source('sentiment')

        # Mock Fear & Greed API response
        with patch.object(sentiment_source, '_fetch_fear_greed_index') as mock_fg:
            mock_fg.return_value = {
                'timestamp': 1638360000,
                'value': 25,  # Extreme Fear
                'value_classification': 'Extreme Fear'
            }

            result = sentiment_source.fetch_recent_data('BTC/USDT', '1h', limit=1)

            assert not result.empty
            assert result.iloc[0]['fear_greed_value'] == 25
            assert result.iloc[0]['fear_greed_classification'] == 'Extreme Fear'
            assert result.iloc[0]['sentiment_signal'] == -3  # Extreme fear signal

    def test_onchain_data_source_simulated_mode(self):
        """Test on-chain data source works in simulated mode."""
        onchain_source = DataSourceFactory.create_data_source('onchain')

        result = onchain_source.fetch_recent_data('BTC/USDT', '1h', limit=1)

        assert not result.empty
        assert 'active_addresses' in result.columns
        assert 'transaction_volume' in result.columns
        assert 'hash_rate' in result.columns
        assert 'onchain_signal' in result.columns

        stats = onchain_source.get_onchain_stats('BTC/USDT')
        assert stats['data_source'] == 'simulated'

    def test_data_source_conversion_formats(self):
        """Test that alternative data sources are properly converted to liquidation format."""
        liquidation_source = DataSourceFactory.create_data_source('liquidation')

        # Test volatility to liquidation conversion
        volatility_df = pd.DataFrame({
            'volatility_signal': [2],
            'volatility_proxy': [5000],
            'timestamp': [pd.Timestamp.now()]
        })

        converted = liquidation_source._convert_alternative_to_liquidation_format(
            volatility_df, 'volatility_indicators'
        )

        assert 'liquidation_signal' in converted.columns
        assert 'liquidation_volume' in converted.columns
        assert 'data_source' in converted.columns
        assert converted.iloc[0]['data_source'] == 'volatility_proxy'

        # Test sentiment to liquidation conversion
        sentiment_df = pd.DataFrame({
            'sentiment_signal': [-2],
            'sentiment_strength': [0.8],
            'timestamp': [pd.Timestamp.now()]
        })

        converted = liquidation_source._convert_alternative_to_liquidation_format(
            sentiment_df, 'sentiment_analysis'
        )

        assert converted.iloc[0]['data_source'] == 'sentiment_proxy'
        assert converted.iloc[0]['liquidation_signal'] == 2  # Inverted

    def test_end_to_end_fallback_workflow(self):
        """Test complete end-to-end fallback workflow."""
        # This test simulates the real-world scenario where primary liquidation
        # data is unavailable and the system falls back through alternatives

        liquidation_source = DataSourceFactory.create_data_source('liquidation')

        # Mock primary failure (simulating API auth issues)
        with patch.object(liquidation_source, '_fetch_binance_recent_liquidations') as mock_primary:
            mock_primary.return_value = pd.DataFrame()  # Primary fails

            # The system should automatically try alternatives and succeed
            result = liquidation_source.fetch_recent_data('BTC/USDT', '1h', limit=1)

            # Should get data from some fallback source
            assert not result.empty
            assert 'liquidation_signal' in result.columns
            assert 'data_source' in result.columns

            # Data source should indicate which fallback was used
            assert result.iloc[0]['data_source'] in [
                'volatility_proxy', 'sentiment_proxy', 'onchain_proxy', 'open_interest_fallback'
            ]

    def test_fallback_performance(self):
        """Test that fallback system performs adequately."""
        import time

        liquidation_source = DataSourceFactory.create_data_source('liquidation')

        start_time = time.time()

        # Mock primary failure to trigger fallback
        with patch.object(liquidation_source, '_fetch_binance_recent_liquidations') as mock_primary:
            mock_primary.return_value = pd.DataFrame()

            result = liquidation_source.fetch_recent_data('BTC/USDT', '1h', limit=1)

        end_time = time.time()
        duration = end_time - start_time

        # Should complete within reasonable time (allowing for fallback chain)
        assert duration < 5.0  # 5 seconds max for fallback chain
        assert not result.empty