"""
WebSocket Data Fetcher for real-time market data streaming.
Provides WebSocket-based data fetching using ccxt's WebSocket support or websocket-client.
"""

import asyncio
import json
import time
import pandas as pd
from datetime import datetime
from typing import Optional, Callable, Dict, Any
import logging
from config.settings import (
    SYMBOL, EXCHANGE, TIMEFRAME, WEBSOCKET_DATA_ENABLED,
    WEBSOCKET_RECONNECT_ATTEMPTS, WEBSOCKET_RECONNECT_DELAY,
    WEBSOCKET_PING_TIMEOUT
)

# Setup logging
logger = logging.getLogger(__name__)

# Try to import ccxt pro for WebSocket support
try:
    import ccxtpro
    CCXTPRO_AVAILABLE = True
except ImportError:
    CCXTPRO_AVAILABLE = False
    logger.warning("ccxtpro not available. WebSocket streaming will use fallback method.")
    logger.info("Install with: pip install ccxtpro")

# Fallback to websocket-client
try:
    import websocket
    WEBSOCKET_CLIENT_AVAILABLE = True
except ImportError:
    WEBSOCKET_CLIENT_AVAILABLE = False
    logger.warning("websocket-client not available. WebSocket features will be limited.")
    logger.info("Install with: pip install websocket-client")


class WebSocketDataFetcher:
    """
    WebSocket-based data fetcher for real-time market data streaming.
    
    Features:
    - Real-time OHLCV data streaming
    - Automatic reconnection on connection loss
    - Support for multiple symbols
    - Data buffering and aggregation
    """
    
    def __init__(self, symbol: str = SYMBOL, exchange: str = EXCHANGE, 
                 timeframe: str = TIMEFRAME, callback: Optional[Callable] = None):
        """
        Initialize WebSocket data fetcher.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            exchange: Exchange name (e.g., 'binance')
            timeframe: Timeframe for candlestick data (e.g., '1m')
            callback: Optional callback function for received data
        """
        self.symbol = symbol
        self.exchange_name = exchange
        self.timeframe = timeframe
        self.callback = callback
        self.is_connected = False
        self.is_running = False
        self.reconnect_attempts = 0
        self.data_buffer = []
        
        # Initialize exchange
        if CCXTPRO_AVAILABLE:
            self.exchange = self._initialize_ccxtpro_exchange()
        else:
            self.exchange = None
            logger.warning("ccxtpro not available. WebSocket features disabled.")
    
    def _initialize_ccxtpro_exchange(self):
        """Initialize ccxtpro exchange for WebSocket support."""
        try:
            exchange_class = getattr(ccxtpro, self.exchange_name)
            return exchange_class({
                'enableRateLimit': True,
                'options': {
                    'watchOHLCV': {
                        'timeframe': self.timeframe
                    }
                }
            })
        except (AttributeError, Exception) as e:
            logger.error(f"Failed to initialize ccxtpro exchange: {e}")
            return None
    
    async def watch_ohlcv(self):
        """
        Watch OHLCV data via WebSocket.
        
        This method continuously watches for new candlestick data
        and calls the callback function when new data arrives.
        """
        if not self.exchange or not CCXTPRO_AVAILABLE:
            logger.error("WebSocket exchange not initialized")
            return
        
        self.is_running = True
        
        while self.is_running:
            try:
                # Watch OHLCV data
                ohlcv = await self.exchange.watch_ohlcv(self.symbol, self.timeframe)
                
                if ohlcv:
                    # Convert to DataFrame
                    df = pd.DataFrame(
                        ohlcv, 
                        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    )
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    # Store in buffer
                    self.data_buffer.append(df)
                    
                    # Call callback if provided
                    if self.callback:
                        await self._execute_callback(df)
                    
                    self.is_connected = True
                    self.reconnect_attempts = 0
                    
            except Exception as e:
                logger.error(f"Error watching OHLCV: {e}")
                self.is_connected = False
                
                # Handle reconnection
                if self.reconnect_attempts < WEBSOCKET_RECONNECT_ATTEMPTS:
                    self.reconnect_attempts += 1
                    logger.info(f"Reconnecting... Attempt {self.reconnect_attempts}/{WEBSOCKET_RECONNECT_ATTEMPTS}")
                    await asyncio.sleep(WEBSOCKET_RECONNECT_DELAY)
                else:
                    logger.error("Max reconnection attempts reached. Stopping.")
                    self.is_running = False
                    break
    
    async def _execute_callback(self, data: pd.DataFrame):
        """Execute callback function with received data."""
        try:
            if asyncio.iscoroutinefunction(self.callback):
                await self.callback(data)
            else:
                self.callback(data)
        except Exception as e:
            logger.error(f"Error executing callback: {e}")
    
    def get_buffered_data(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Get buffered data as a single DataFrame.
        
        Args:
            limit: Maximum number of recent records to return
            
        Returns:
            DataFrame with buffered OHLCV data
        """
        if not self.data_buffer:
            return pd.DataFrame()
        
        # Concatenate all buffered data
        combined_df = pd.concat(self.data_buffer)
        combined_df = combined_df.sort_index()
        
        # Remove duplicates (keep last occurrence)
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        
        if limit:
            combined_df = combined_df.tail(limit)
        
        return combined_df
    
    def clear_buffer(self):
        """Clear the data buffer."""
        self.data_buffer = []
    
    def stop(self):
        """Stop the WebSocket connection."""
        self.is_running = False
        logger.info("WebSocket data fetcher stopped")
    
    async def close(self):
        """Close the WebSocket connection and cleanup."""
        self.stop()
        if self.exchange and CCXTPRO_AVAILABLE:
            await self.exchange.close()
        logger.info("WebSocket connection closed")


async def fetch_realtime_data(symbol: str = SYMBOL, exchange: str = EXCHANGE,
                              timeframe: str = TIMEFRAME, 
                              duration: int = 60,
                              callback: Optional[Callable] = None) -> pd.DataFrame:
    """
    Fetch real-time data via WebSocket for a specified duration.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe for candlestick data
        duration: Duration in seconds to fetch data
        callback: Optional callback for each data update
        
    Returns:
        DataFrame with collected OHLCV data
    """
    if not WEBSOCKET_DATA_ENABLED:
        logger.warning("WebSocket data fetching is disabled in settings")
        return pd.DataFrame()
    
    if not CCXTPRO_AVAILABLE:
        logger.error("ccxtpro not available. Cannot fetch WebSocket data.")
        return pd.DataFrame()
    
    fetcher = WebSocketDataFetcher(symbol, exchange, timeframe, callback)
    
    try:
        # Start watching in background
        watch_task = asyncio.create_task(fetcher.watch_ohlcv())
        
        # Wait for specified duration
        await asyncio.sleep(duration)
        
        # Stop watching
        fetcher.stop()
        
        # Wait for task to complete
        try:
            await asyncio.wait_for(watch_task, timeout=5)
        except asyncio.TimeoutError:
            watch_task.cancel()
        
        # Return collected data
        return fetcher.get_buffered_data()
    
    finally:
        await fetcher.close()


def is_websocket_available() -> bool:
    """Check if WebSocket data fetching is available."""
    return CCXTPRO_AVAILABLE and WEBSOCKET_DATA_ENABLED
