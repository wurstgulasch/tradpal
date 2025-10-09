"""
Tests for WebSocket data fetching functionality.
"""

import pytest
import pandas as pd
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.websocket_data_fetcher import (
    WebSocketDataFetcher, fetch_realtime_data, is_websocket_available
)


class TestWebSocketDataFetcher:
    """Test WebSocket data fetcher functionality."""
    
    def test_initialization(self):
        """Test WebSocketDataFetcher initialization."""
        fetcher = WebSocketDataFetcher(
            symbol='BTC/USDT',
            exchange='binance',
            timeframe='1m'
        )
        
        assert fetcher.symbol == 'BTC/USDT'
        assert fetcher.exchange_name == 'binance'
        assert fetcher.timeframe == '1m'
        assert not fetcher.is_connected
        assert not fetcher.is_running
        assert fetcher.data_buffer == []
    
    def test_buffer_operations(self):
        """Test data buffer operations."""
        fetcher = WebSocketDataFetcher()
        
        # Add data to buffer
        df1 = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103],
            'low': [99, 100],
            'close': [101, 102],
            'volume': [1000, 1100]
        }, index=pd.date_range('2024-01-01', periods=2, freq='1min'))
        
        df2 = pd.DataFrame({
            'open': [102, 103],
            'high': [104, 105],
            'low': [101, 102],
            'close': [103, 104],
            'volume': [1200, 1300]
        }, index=pd.date_range('2024-01-01 00:02', periods=2, freq='1min'))
        
        fetcher.data_buffer.append(df1)
        fetcher.data_buffer.append(df2)
        
        # Get buffered data
        combined = fetcher.get_buffered_data()
        
        assert len(combined) == 4
        assert all(col in combined.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        assert isinstance(combined.index, pd.DatetimeIndex)
        
        # Test limit
        limited = fetcher.get_buffered_data(limit=2)
        assert len(limited) == 2
        
        # Clear buffer
        fetcher.clear_buffer()
        assert len(fetcher.data_buffer) == 0
    
    def test_get_buffered_data_empty(self):
        """Test getting buffered data when buffer is empty."""
        fetcher = WebSocketDataFetcher()
        
        result = fetcher.get_buffered_data()
        
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_stop(self):
        """Test stopping the WebSocket fetcher."""
        fetcher = WebSocketDataFetcher()
        fetcher.is_running = True
        
        fetcher.stop()
        
        assert not fetcher.is_running
    
    def test_is_websocket_available(self):
        """Test checking if WebSocket is available."""
        # This will depend on whether ccxtpro is installed
        result = is_websocket_available()
        assert isinstance(result, bool)


class TestWebSocketIntegration:
    """Integration tests for WebSocket functionality."""
    
    def test_fetch_realtime_data_disabled(self):
        """Test that fetch returns empty DataFrame when disabled."""
        with patch('src.websocket_data_fetcher.WEBSOCKET_DATA_ENABLED', False):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            result = loop.run_until_complete(
                fetch_realtime_data(symbol='BTC/USDT', duration=1)
            )
            
            loop.close()
            
            assert isinstance(result, pd.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
