"""
Integration tests for Alternative Data Service with event publishing.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock


class TestAlternativeDataServiceIntegration:
    """Integration tests for Alternative Data Service."""

    @pytest.mark.asyncio
    async def test_collect_data_background_with_event_publishing(self):
        """Test background data collection with event publishing."""
        symbol = "BTC/USDT"

        # Mock data responses
        mock_sentiment = [{"sentiment": "bullish", "score": 0.8}]
        mock_onchain = [{"metric": "hash_rate", "value": 1000000}]
        mock_economic = [{"indicator": "GDP", "value": 20000}]
        mock_fear_greed = {"value": 65, "value_classification": "Greed"}

        # Mock the entire module to avoid import issues
        with patch('services.alternative_data_service.main.sentiment_analyzer') as mock_sentiment_analyzer, \
             patch('services.alternative_data_service.main.onchain_collector') as mock_onchain_collector, \
             patch('services.alternative_data_service.main.economic_collector') as mock_economic_collector, \
             patch('services.alternative_data_service.main.data_processor') as mock_data_processor, \
             patch('services.alternative_data_service.main.publish_sentiment_data') as mock_publish_sentiment, \
             patch('services.alternative_data_service.main.publish_onchain_data') as mock_publish_onchain, \
             patch('services.alternative_data_service.main.publish_economic_data') as mock_publish_economic, \
             patch('services.alternative_data_service.main.publish_alternative_data') as mock_publish_alternative, \
             patch('services.alternative_data_service.main.publish_feature_vector') as mock_publish_features, \
             patch('services.alternative_data_service.main.logger') as mock_logger:

            # Setup mock returns with AsyncMock for async methods
            mock_sentiment_analyzer.analyze_symbol_sentiment = AsyncMock(return_value=mock_sentiment)
            mock_onchain_collector.get_metrics = AsyncMock(return_value=mock_onchain)
            mock_economic_collector.get_indicators = AsyncMock(return_value=mock_economic)
            mock_sentiment_analyzer.get_fear_greed_index = AsyncMock(return_value=mock_fear_greed)

            # Create a proper mock features object
            mock_features = MagicMock()
            mock_features.__dict__ = {"feature1": 0.5, "feature2": 0.3}
            mock_data_processor.process_to_features = AsyncMock(return_value=mock_features)

            # Import and call the function
            from services.alternative_data_service.main import collect_data_background
            await collect_data_background(symbol)

            # Verify all analyzers were called
            mock_sentiment_analyzer.analyze_symbol_sentiment.assert_called_once_with(symbol)
            mock_onchain_collector.get_metrics.assert_called_once_with(symbol)
            mock_economic_collector.get_indicators.assert_called_once()
            mock_sentiment_analyzer.get_fear_greed_index.assert_called_once()

            # Verify event publishing was called
            mock_publish_sentiment.assert_called_once()
            mock_publish_onchain.assert_called_once()
            mock_publish_economic.assert_called_once()
            mock_publish_alternative.assert_called_once()
            mock_publish_features.assert_called_once()

            # Verify data processor was called
            mock_data_processor.process_to_features.assert_called_once()

    @pytest.mark.asyncio
    async def test_collect_data_background_handles_exceptions(self):
        """Test background collection handles exceptions gracefully."""
        symbol = "BTC/USDT"

        # Mock one analyzer to raise exception
        with patch('services.alternative_data_service.main.sentiment_analyzer') as mock_sentiment_analyzer, \
             patch('services.alternative_data_service.main.onchain_collector') as mock_onchain_collector, \
             patch('services.alternative_data_service.main.economic_collector') as mock_economic_collector, \
             patch('services.alternative_data_service.main.publish_sentiment_data') as mock_publish_sentiment, \
             patch('services.alternative_data_service.main.publish_onchain_data') as mock_publish_onchain, \
             patch('services.alternative_data_service.main.publish_economic_data') as mock_publish_economic, \
             patch('services.alternative_data_service.main.publish_alternative_data') as mock_publish_alternative, \
             patch('services.alternative_data_service.main.publish_feature_vector') as mock_publish_features, \
             patch('services.alternative_data_service.main.logger') as mock_logger:

            # Setup mock returns with AsyncMock for async methods
            mock_sentiment_analyzer.analyze_symbol_sentiment = AsyncMock(side_effect=Exception("API Error"))
            mock_onchain_collector.get_metrics = AsyncMock(return_value=[{"metric": "hash_rate"}])
            mock_economic_collector.get_indicators = AsyncMock(return_value=[{"indicator": "GDP"}])
            mock_sentiment_analyzer.get_fear_greed_index = AsyncMock(return_value={"value": 50})

            # Import and call the function
            from services.alternative_data_service.main import collect_data_background
            await collect_data_background(symbol)

            # Verify analyzers were called
            mock_sentiment_analyzer.analyze_symbol_sentiment.assert_called_once_with(symbol)
            mock_onchain_collector.get_metrics.assert_called_once_with(symbol)
            mock_economic_collector.get_indicators.assert_called_once()
            mock_sentiment_analyzer.get_fear_greed_index.assert_called_once()

            # Verify only successful data was published
            mock_publish_sentiment.assert_not_called()  # Failed
            mock_publish_onchain.assert_called_once()
            mock_publish_economic.assert_called_once()
            mock_publish_alternative.assert_not_called()  # Not all successful
            mock_publish_features.assert_not_called()

    @pytest.mark.asyncio
    async def test_collect_data_background_without_event_system(self):
        """Test background collection works without event system."""
        symbol = "BTC/USDT"

        # Mock EVENT_SYSTEM_AVAILABLE to False
        with patch('services.alternative_data_service.main.EVENT_SYSTEM_AVAILABLE', False), \
             patch('services.alternative_data_service.main.sentiment_analyzer') as mock_sentiment_analyzer, \
             patch('services.alternative_data_service.main.onchain_collector') as mock_onchain_collector, \
             patch('services.alternative_data_service.main.economic_collector') as mock_economic_collector, \
             patch('services.alternative_data_service.main.publish_sentiment_data') as mock_publish_sentiment, \
             patch('services.alternative_data_service.main.logger') as mock_logger:

            # Setup mock returns with AsyncMock for async methods
            mock_sentiment_analyzer.analyze_symbol_sentiment = AsyncMock(return_value=[{"sentiment": "neutral"}])
            mock_onchain_collector.get_metrics = AsyncMock(return_value=[{"metric": "volume"}])
            mock_economic_collector.get_indicators = AsyncMock(return_value=[{"indicator": "inflation"}])
            mock_sentiment_analyzer.get_fear_greed_index = AsyncMock(return_value={"value": 40})

            # Import and call the function
            from services.alternative_data_service.main import collect_data_background
            await collect_data_background(symbol)

            # Verify analyzers were called
            mock_sentiment_analyzer.analyze_symbol_sentiment.assert_called_once_with(symbol)
            mock_onchain_collector.get_metrics.assert_called_once_with(symbol)
            mock_economic_collector.get_indicators.assert_called_once()
            mock_sentiment_analyzer.get_fear_greed_index.assert_called_once()

            # Verify no events were published
            mock_publish_sentiment.assert_not_called()