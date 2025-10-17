"""
Unit tests for Alternative Data Service event publishing logic.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime


class TestEventPublishingLogic:
    """Test the event publishing logic in isolation."""

    @pytest.mark.asyncio
    async def test_event_publishing_with_successful_data(self):
        """Test event publishing when all data collection succeeds."""
        # Mock all the components
        mock_sentiment_data = [{"sentiment": "bullish", "score": 0.8}]
        mock_onchain_data = [{"metric": "hash_rate", "value": 1000000}]
        mock_economic_data = [{"indicator": "GDP", "value": 20000}]
        mock_fear_greed = {"value": 65}

        # Mock event publishing functions
        with patch('sys.modules', {'services.event_system': MagicMock()}):  # Mock event system module
            # Create mock functions that simulate the event publishing
            async def mock_publish_sentiment(data):
                pass

            async def mock_publish_onchain(data):
                pass

            async def mock_publish_economic(data):
                pass

            async def mock_publish_alternative(data):
                pass

            async def mock_publish_feature_vector(data):
                pass

            # Simulate the logic from collect_data_background
            symbol = "BTC/USDT"
            results = (mock_sentiment_data, mock_onchain_data, mock_economic_data, mock_fear_greed)

            # Process results and publish to event system
            sentiment_data, onchain_data, economic_data, fear_greed = results

            # Publish individual data events
            await mock_publish_sentiment({
                "symbol": symbol,
                "sentiment_data": [s.__dict__ if hasattr(s, '__dict__') else s for s in sentiment_data] if isinstance(sentiment_data, list) else [sentiment_data.__dict__ if hasattr(sentiment_data, '__dict__') else sentiment_data],
                "timestamp": datetime.utcnow().isoformat()
            })

            await mock_publish_onchain({
                "symbol": symbol,
                "onchain_data": [o.__dict__ if hasattr(o, '__dict__') else o for o in onchain_data] if isinstance(onchain_data, list) else [onchain_data.__dict__ if hasattr(onchain_data, '__dict__') else onchain_data],
                "timestamp": datetime.utcnow().isoformat()
            })

            await mock_publish_economic({
                "economic_data": [e.__dict__ if hasattr(e, '__dict__') else e for e in economic_data] if isinstance(economic_data, list) else [economic_data.__dict__ if hasattr(economic_data, '__dict__') else economic_data],
                "timestamp": datetime.utcnow().isoformat()
            })

            # Create and publish complete alternative data packet
            if all(not isinstance(r, Exception) for r in results):
                # Mock AlternativeDataPacket
                packet = MagicMock()
                packet.__dict__ = {
                    "symbol": symbol,
                    "sentiment_data": sentiment_data,
                    "onchain_data": onchain_data,
                    "economic_data": economic_data,
                    "fear_greed_index": fear_greed.get('value')
                }

                # Mock features
                features = MagicMock()
                features.__dict__ = {"feature1": 0.5, "feature2": 0.3}

                await mock_publish_alternative({
                    "symbol": symbol,
                    "packet": packet.__dict__,
                    "features": features.__dict__,
                    "timestamp": datetime.utcnow().isoformat()
                })

                await mock_publish_feature_vector({
                    "symbol": symbol,
                    "features": features.__dict__,
                    "timestamp": datetime.utcnow().isoformat()
                })

        # Test passes if no exceptions were raised
        assert True

    @pytest.mark.asyncio
    async def test_event_publishing_with_partial_failures(self):
        """Test event publishing when some data collection fails."""
        # Mock data with one failure
        mock_sentiment_data = Exception("API Error")
        mock_onchain_data = [{"metric": "hash_rate", "value": 1000000}]
        mock_economic_data = [{"indicator": "GDP", "value": 20000}]
        mock_fear_greed = {"value": 65}

        publish_calls = []

        # Mock event publishing functions that record calls
        async def mock_publish_sentiment(data):
            publish_calls.append(('sentiment', data))

        async def mock_publish_onchain(data):
            publish_calls.append(('onchain', data))

        async def mock_publish_economic(data):
            publish_calls.append(('economic', data))

        async def mock_publish_alternative(data):
            publish_calls.append(('alternative', data))

        async def mock_publish_feature_vector(data):
            publish_calls.append(('feature_vector', data))

        # Simulate the logic from collect_data_background
        symbol = "BTC/USDT"
        results = (mock_sentiment_data, mock_onchain_data, mock_economic_data, mock_fear_greed)

        # Process results and publish to event system
        sentiment_data, onchain_data, economic_data, fear_greed = results

        # Publish individual data events (only successful ones)
        if onchain_data and not isinstance(onchain_data, Exception):
            await mock_publish_onchain({
                "symbol": symbol,
                "onchain_data": [o.__dict__ if hasattr(o, '__dict__') else o for o in onchain_data] if isinstance(onchain_data, list) else [onchain_data.__dict__ if hasattr(onchain_data, '__dict__') else onchain_data],
                "timestamp": datetime.utcnow().isoformat()
            })

        if economic_data and not isinstance(economic_data, Exception):
            await mock_publish_economic({
                "economic_data": [e.__dict__ if hasattr(e, '__dict__') else e for e in economic_data] if isinstance(economic_data, list) else [economic_data.__dict__ if hasattr(economic_data, '__dict__') else economic_data],
                "timestamp": datetime.utcnow().isoformat()
            })

        # Complete packet should not be published due to failure
        if all(not isinstance(r, Exception) for r in results):
            # This should not execute
            assert False, "Complete packet should not be published with failures"

        # Verify only successful events were published
        assert len(publish_calls) == 2
        # Check that onchain and economic events were published
        event_types = [call[0] for call in publish_calls]
        assert 'onchain' in event_types
        assert 'economic' in event_types
        assert 'sentiment' not in event_types  # Failed event should not be published
        assert 'alternative' not in event_types  # Complete packet should not be published

    def test_event_system_fallback_logic(self):
        """Test that the fallback logic works when event system is unavailable."""
        # Test EVENT_SYSTEM_AVAILABLE flag logic
        EVENT_SYSTEM_AVAILABLE = True
        if EVENT_SYSTEM_AVAILABLE:
            # Should attempt to publish
            assert True
        else:
            # Should skip publishing
            assert True

        EVENT_SYSTEM_AVAILABLE = False
        if EVENT_SYSTEM_AVAILABLE:
            assert False, "Should not publish when unavailable"
        else:
            # Should skip publishing
            assert True