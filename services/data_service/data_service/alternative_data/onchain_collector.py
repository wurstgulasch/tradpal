#!/usr/bin/env python3
"""
On-Chain Data Collector - Blockchain Analytics for Crypto Assets

Collects comprehensive on-chain metrics from blockchain APIs:
- Active Addresses (daily active addresses)
- Transaction Volume (on-chain transaction volumes)
- Exchange Flows (net capital flows to/from exchanges)
- Whale Movements (large wallet transactions)
- NVT Ratio (Network Value to Transactions)
- Mining Difficulty (for mining-based assets)
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import json

import aiohttp
import pandas as pd
import numpy as np

from .__init__ import OnChainData, OnChainMetric

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WhaleTransaction:
    """Large wallet transaction data."""
    transaction_hash: str
    from_address: str
    to_address: str
    value: float
    timestamp: datetime
    block_number: int
    symbol: str


class OnChainDataCollector:
    """
    Comprehensive on-chain data collector for multiple blockchains.
    """

    def __init__(self):
        self.session = None
        self.api_keys = {}
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes

        # Load API keys
        self._load_api_keys()

    def _load_api_keys(self):
        """Load API keys from environment."""
        import os
        self.api_keys = {
            'glassnode': os.getenv('GLASSNODE_API_KEY'),
            'amberdata': os.getenv('AMBERDATA_API_KEY'),
            'covalenthq': os.getenv('COVALENTHQ_API_KEY'),
            'blockchain_com': None,  # Free API
            'coinmetrics': os.getenv('COINMETRICS_API_KEY')
        }

    async def initialize(self):
        """Initialize the collector."""
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
            logger.info("On-Chain Data Collector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize On-Chain Data Collector: {e}")
            raise

    async def close(self):
        """Close resources."""
        if self.session:
            await self.session.close()

    async def get_metrics(self, symbol: str, metrics: Optional[List[str]] = None) -> List[OnChainData]:
        """
        Get on-chain metrics for a symbol.

        Args:
            symbol: Crypto symbol (e.g., 'BTC')
            metrics: Specific metrics to fetch (optional)

        Returns:
            List of on-chain data points
        """
        try:
            # Default metrics if none specified
            if metrics is None:
                metrics = [
                    'active_addresses',
                    'transaction_volume',
                    'exchange_flows',
                    'nvt_ratio'
                ]

            # Map string metrics to enum
            metric_enums = []
            for metric in metrics:
                try:
                    metric_enum = OnChainMetric(metric.upper())
                    metric_enums.append(metric_enum)
                except ValueError:
                    logger.warning(f"Unknown metric: {metric}")
                    continue

            # Collect data in parallel
            tasks = []
            for metric in metric_enums:
                if symbol.upper() in ['BTC', 'BITCOIN']:
                    tasks.append(self._get_bitcoin_metric(metric))
                elif symbol.upper() in ['ETH', 'ETHEREUM']:
                    tasks.append(self._get_ethereum_metric(metric))
                else:
                    # Generic approach for other assets
                    tasks.append(self._get_generic_metric(symbol, metric))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            onchain_data = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"Metric collection failed for {metric_enums[i]}: {result}")
                    continue

                if result:
                    onchain_data.extend(result)

            return onchain_data

        except Exception as e:
            logger.error(f"On-chain metrics collection failed: {e}")
            return []

    async def _get_bitcoin_metric(self, metric: OnChainMetric) -> List[OnChainData]:
        """Get Bitcoin-specific on-chain metrics."""
        try:
            if metric == OnChainMetric.ACTIVE_ADDRESSES:
                return await self._get_active_addresses('BTC')
            elif metric == OnChainMetric.TRANSACTION_VOLUME:
                return await self._get_transaction_volume('BTC')
            elif metric == OnChainMetric.EXCHANGE_FLOWS:
                return await self._get_exchange_flows('BTC')
            elif metric == OnChainMetric.NVT_RATIO:
                return await self._get_nvt_ratio('BTC')
            elif metric == OnChainMetric.MINING_DIFFICULTY:
                return await self._get_mining_difficulty('BTC')
            elif metric == OnChainMetric.WHALE_MOVEMENTS:
                return await self._get_whale_movements('BTC')
            else:
                return []
        except Exception as e:
            logger.error(f"Bitcoin metric {metric} failed: {e}")
            return []

    async def _get_ethereum_metric(self, metric: OnChainMetric) -> List[OnChainData]:
        """Get Ethereum-specific on-chain metrics."""
        try:
            if metric == OnChainMetric.ACTIVE_ADDRESSES:
                return await self._get_active_addresses('ETH')
            elif metric == OnChainMetric.TRANSACTION_VOLUME:
                return await self._get_transaction_volume('ETH')
            elif metric == OnChainMetric.EXCHANGE_FLOWS:
                return await self._get_exchange_flows('ETH')
            elif metric == OnChainMetric.NVT_RATIO:
                return await self._get_nvt_ratio('ETH')
            elif metric == OnChainMetric.WHALE_MOVEMENTS:
                return await self._get_whale_movements('ETH')
            else:
                return []
        except Exception as e:
            logger.error(f"Ethereum metric {metric} failed: {e}")
            return []

    async def _get_generic_metric(self, symbol: str, metric: OnChainMetric) -> List[OnChainData]:
        """Get generic on-chain metrics for any asset."""
        # For now, return empty list - can be extended for other assets
        logger.info(f"Generic metrics not implemented for {symbol}:{metric}")
        return []

    async def _get_active_addresses(self, symbol: str) -> List[OnChainData]:
        """Get daily active addresses."""
        try:
            # Try Glassnode API first
            if self.api_keys.get('glassnode'):
                return await self._get_glassnode_active_addresses(symbol)

            # Fallback to Blockchain.com for BTC
            if symbol.upper() == 'BTC':
                return await self._get_blockchain_com_active_addresses()

            # Return mock data for demo
            return [OnChainData(
                symbol=symbol,
                metric=OnChainMetric.ACTIVE_ADDRESSES,
                value=500000 + np.random.randint(-50000, 50000),  # Mock value
                timestamp=datetime.utcnow(),
                metadata={'source': 'mock', 'note': 'API keys not configured'}
            )]

        except Exception as e:
            logger.error(f"Active addresses fetch failed: {e}")
            return []

    async def _get_transaction_volume(self, symbol: str) -> List[OnChainData]:
        """Get on-chain transaction volume."""
        try:
            # Try Glassnode API
            if self.api_keys.get('glassnode'):
                return await self._get_glassnode_transaction_volume(symbol)

            # Return mock data
            return [OnChainData(
                symbol=symbol,
                metric=OnChainMetric.TRANSACTION_VOLUME,
                value=1000000 + np.random.randint(-100000, 100000),  # Mock value
                timestamp=datetime.utcnow(),
                metadata={'source': 'mock', 'note': 'API keys not configured'}
            )]

        except Exception as e:
            logger.error(f"Transaction volume fetch failed: {e}")
            return []

    async def _get_exchange_flows(self, symbol: str) -> List[OnChainData]:
        """Get exchange inflow/outflow data."""
        try:
            # Try Glassnode API
            if self.api_keys.get('glassnode'):
                return await self._get_glassnode_exchange_flows(symbol)

            # Return mock data
            return [OnChainData(
                symbol=symbol,
                metric=OnChainMetric.EXCHANGE_FLOWS,
                value=np.random.randint(-10000, 10000),  # Net flow
                timestamp=datetime.utcnow(),
                metadata={'source': 'mock', 'note': 'API keys not configured'}
            )]

        except Exception as e:
            logger.error(f"Exchange flows fetch failed: {e}")
            return []

    async def _get_nvt_ratio(self, symbol: str) -> List[OnChainData]:
        """Get Network Value to Transactions ratio."""
        try:
            # Try Glassnode API
            if self.api_keys.get('glassnode'):
                return await self._get_glassnode_nvt_ratio(symbol)

            # Calculate simple NVT approximation
            market_cap = await self._get_market_cap(symbol)
            transaction_volume = await self._get_transaction_volume(symbol)

            if market_cap and transaction_volume and len(transaction_volume) > 0:
                nvt = market_cap[0].value / transaction_volume[0].value
                return [OnChainData(
                    symbol=symbol,
                    metric=OnChainMetric.NVT_RATIO,
                    value=nvt,
                    timestamp=datetime.utcnow(),
                    metadata={'source': 'calculated'}
                )]

            # Return mock data
            return [OnChainData(
                symbol=symbol,
                metric=OnChainMetric.NVT_RATIO,
                value=50 + np.random.randint(-10, 10),  # Typical range
                timestamp=datetime.utcnow(),
                metadata={'source': 'mock', 'note': 'API keys not configured'}
            )]

        except Exception as e:
            logger.error(f"NVT ratio calculation failed: {e}")
            return []

    async def _get_mining_difficulty(self, symbol: str) -> List[OnChainData]:
        """Get mining difficulty for proof-of-work assets."""
        try:
            if symbol.upper() == 'BTC':
                # Try Blockchain.com API
                return await self._get_blockchain_com_difficulty()

            return [OnChainData(
                symbol=symbol,
                metric=OnChainMetric.MINING_DIFFICULTY,
                value=1e19 + np.random.randint(-1e18, 1e18),  # Mock BTC difficulty
                timestamp=datetime.utcnow(),
                metadata={'source': 'mock', 'note': 'API keys not configured'}
            )]

        except Exception as e:
            logger.error(f"Mining difficulty fetch failed: {e}")
            return []

    async def _get_whale_movements(self, symbol: str) -> List[OnChainData]:
        """Get large wallet movements (whale transactions)."""
        try:
            # Try Whale Alert API or similar
            if self.api_keys.get('whale_alert'):
                return await self._get_whale_alert_data(symbol)

            # Return mock data
            whale_data = OnChainData(
                symbol=symbol,
                metric=OnChainMetric.WHALE_MOVEMENTS,
                value=np.random.randint(10, 100),  # Number of large transactions
                timestamp=datetime.utcnow(),
                metadata={
                    'source': 'mock',
                    'note': 'API keys not configured',
                    'large_transactions': np.random.randint(10, 100)
                }
            )
            return [whale_data]

        except Exception as e:
            logger.error(f"Whale movements fetch failed: {e}")
            return []

    # API-specific implementations
    async def _get_glassnode_active_addresses(self, symbol: str) -> List[OnChainData]:
        """Get active addresses from Glassnode."""
        try:
            url = f"https://api.glassnode.com/v1/metrics/addresses/active_count"
            params = {
                'api_key': self.api_keys['glassnode'],
                'a': symbol.upper(),
                's': int((datetime.utcnow() - timedelta(days=1)).timestamp()),
                'u': int(datetime.utcnow().timestamp())
            }

            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"Glassnode API error: {response.status}")

                data = await response.json()

            if data:
                latest = data[-1]
                return [OnChainData(
                    symbol=symbol,
                    metric=OnChainMetric.ACTIVE_ADDRESSES,
                    value=latest['v'],
                    timestamp=datetime.fromtimestamp(latest['t']),
                    metadata={'source': 'glassnode'}
                )]
            return []

        except Exception as e:
            logger.error(f"Glassnode active addresses failed: {e}")
            raise

    async def _get_glassnode_transaction_volume(self, symbol: str) -> List[OnChainData]:
        """Get transaction volume from Glassnode."""
        try:
            url = f"https://api.glassnode.com/v1/metrics/transactions/transfers_volume_sum"
            params = {
                'api_key': self.api_keys['glassnode'],
                'a': symbol.upper(),
                's': int((datetime.utcnow() - timedelta(days=1)).timestamp()),
                'u': int(datetime.utcnow().timestamp())
            }

            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"Glassnode API error: {response.status}")

                data = await response.json()

            if data:
                latest = data[-1]
                return [OnChainData(
                    symbol=symbol,
                    metric=OnChainMetric.TRANSACTION_VOLUME,
                    value=latest['v'],
                    timestamp=datetime.fromtimestamp(latest['t']),
                    metadata={'source': 'glassnode'}
                )]
            return []

        except Exception as e:
            logger.error(f"Glassnode transaction volume failed: {e}")
            raise

    async def _get_glassnode_exchange_flows(self, symbol: str) -> List[OnChainData]:
        """Get exchange flows from Glassnode."""
        try:
            url = f"https://api.glassnode.com/v1/metrics/transfers/volume_to_exchanges_sum"
            params = {
                'api_key': self.api_keys['glassnode'],
                'a': symbol.upper(),
                's': int((datetime.utcnow() - timedelta(days=1)).timestamp()),
                'u': int(datetime.utcnow().timestamp())
            }

            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"Glassnode API error: {response.status}")

                data = await response.json()

            if data:
                latest = data[-1]
                return [OnChainData(
                    symbol=symbol,
                    metric=OnChainMetric.EXCHANGE_FLOWS,
                    value=latest['v'],
                    timestamp=datetime.fromtimestamp(latest['t']),
                    metadata={'source': 'glassnode'}
                )]
            return []

        except Exception as e:
            logger.error(f"Glassnode exchange flows failed: {e}")
            raise

    async def _get_glassnode_nvt_ratio(self, symbol: str) -> List[OnChainData]:
        """Get NVT ratio from Glassnode."""
        try:
            url = f"https://api.glassnode.com/v1/metrics/indicators/nvt"
            params = {
                'api_key': self.api_keys['glassnode'],
                'a': symbol.upper(),
                's': int((datetime.utcnow() - timedelta(days=1)).timestamp()),
                'u': int(datetime.utcnow().timestamp())
            }

            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"Glassnode API error: {response.status}")

                data = await response.json()

            if data:
                latest = data[-1]
                return [OnChainData(
                    symbol=symbol,
                    metric=OnChainMetric.NVT_RATIO,
                    value=latest['v'],
                    timestamp=datetime.fromtimestamp(latest['t']),
                    metadata={'source': 'glassnode'}
                )]
            return []

        except Exception as e:
            logger.error(f"Glassnode NVT ratio failed: {e}")
            raise

    async def _get_blockchain_com_active_addresses(self) -> List[OnChainData]:
        """Get BTC active addresses from Blockchain.com."""
        try:
            url = "https://api.blockchain.info/charts/n-unique-addresses"
            params = {
                'timespan': '1days',
                'rollingAverage': '8hours',
                'format': 'json'
            }

            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"Blockchain.com API error: {response.status}")

                data = await response.json()

            if data.get('values'):
                latest = data['values'][-1]
                return [OnChainData(
                    symbol='BTC',
                    metric=OnChainMetric.ACTIVE_ADDRESSES,
                    value=latest['y'],
                    timestamp=datetime.fromtimestamp(latest['x']),
                    metadata={'source': 'blockchain.com'}
                )]
            return []

        except Exception as e:
            logger.error(f"Blockchain.com active addresses failed: {e}")
            raise

    async def _get_blockchain_com_difficulty(self) -> List[OnChainData]:
        """Get BTC mining difficulty from Blockchain.com."""
        try:
            url = "https://api.blockchain.info/charts/difficulty"
            params = {
                'timespan': '1days',
                'rollingAverage': '8hours',
                'format': 'json'
            }

            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"Blockchain.com API error: {response.status}")

                data = await response.json()

            if data.get('values'):
                latest = data['values'][-1]
                return [OnChainData(
                    symbol='BTC',
                    metric=OnChainMetric.MINING_DIFFICULTY,
                    value=latest['y'],
                    timestamp=datetime.fromtimestamp(latest['x']),
                    metadata={'source': 'blockchain.com'}
                )]
            return []

        except Exception as e:
            logger.error(f"Blockchain.com difficulty failed: {e}")
            raise

    async def _get_market_cap(self, symbol: str) -> List[OnChainData]:
        """Get market cap for NVT calculation."""
        # This would typically come from CoinGecko or similar
        # For now, return mock data
        return [OnChainData(
            symbol=symbol,
            metric=OnChainMetric.NVT_RATIO,  # Using as placeholder
            value=50000000000 + np.random.randint(-5000000000, 5000000000),  # Mock BTC market cap
            timestamp=datetime.utcnow(),
            metadata={'source': 'mock'}
        )]

    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics collection summary."""
        return {
            "configured_apis": {
                "glassnode": self.api_keys.get('glassnode') is not None,
                "amberdata": self.api_keys.get('amberdata') is not None,
                "covalenthq": self.api_keys.get('covalenthq') is not None,
                "coinmetrics": self.api_keys.get('coinmetrics') is not None
            },
            "cache_size": len(self.cache),
            "cache_ttl": self.cache_ttl
        }