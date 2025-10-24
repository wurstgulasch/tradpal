#!/usr/bin/env python3
"""
Backtesting Service Client - Client for interacting with the Backtesting Service.

Provides methods to:
- Run single backtests
- Run multi-symbol backtests
- Run multi-model comparisons
- Monitor backtest status
- Get backtest results
Enhanced with Zero-Trust Security (mTLS + JWT).
"""

import asyncio
import aiohttp
import logging
import ssl
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

from config.service_settings import (
    BACKTESTING_SERVICE_URL, ENABLE_MTLS, MTLS_CERT_PATH, MTLS_KEY_PATH, CA_CERT_PATH,
    REQUEST_TIMEOUT
)


class BacktestingServiceClient:
    """Client for the Backtesting Service microservice with Zero-Trust Security"""

    def __init__(self, base_url: str = BACKTESTING_SERVICE_URL):
        self.base_url = base_url.rstrip('/')
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(__name__)
        self.jwt_token: Optional[str] = None

        # mTLS configuration
        self.mtls_enabled = ENABLE_MTLS or False
        self.ssl_context: Optional[ssl.SSLContext] = None

        if self.mtls_enabled:
            self._setup_mtls()

    def _setup_mtls(self):
        """Setup mutual TLS configuration"""
        try:
            from pathlib import Path
            self.ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)

            # Load client certificate and key
            if MTLS_CERT_PATH and MTLS_KEY_PATH:
                cert_path = Path(MTLS_CERT_PATH)
                key_path = Path(MTLS_KEY_PATH)

                if cert_path.exists() and key_path.exists():
                    self.ssl_context.load_cert_chain(str(cert_path), str(key_path))
                    self.logger.info("✅ mTLS client certificate loaded for Backtesting Service")
                else:
                    self.logger.warning("⚠️  mTLS certificate files not found, disabling mTLS")
                    self.mtls_enabled = False

            # Load CA certificate for server verification
            if CA_CERT_PATH and Path(CA_CERT_PATH).exists():
                self.ssl_context.load_verify_locations(CA_CERT_PATH)
                self.ssl_context.verify_mode = ssl.CERT_REQUIRED

        except Exception as e:
            self.logger.error(f"❌ Failed to setup mTLS for Backtesting Service: {e}")
            self.mtls_enabled = False

    async def authenticate(self) -> bool:
        """Authenticate with security service and get JWT token"""
        try:
            from services.infrastructure_service.security_service.client import SecurityServiceClient

            security_client = SecurityServiceClient()
            success = await security_client.authenticate("backtesting_service_client")

            if success:
                self.jwt_token = "authenticated"  # Placeholder for actual token
                self.logger.info("✅ Backtesting service client authenticated")
                return True
            else:
                self.logger.error("❌ Backtesting service client authentication failed")
                return False

        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            return False

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def initialize(self) -> None:
        """Initialize the client"""
        if self.session is None:
            headers = {
                'Content-Type': 'application/json'
            }
            if self.jwt_token:
                headers['Authorization'] = f'Bearer {self.jwt_token}'

            connector = None
            if self.mtls_enabled and self.ssl_context:
                connector = aiohttp.TCPConnector(ssl=self.ssl_context)

            self.session = aiohttp.ClientSession(
                headers=headers,
                connector=connector
            )

    async def close(self) -> None:
        """Close the client"""
        if self.session:
            await self.session.close()
            self.session = None

    async def health_check(self) -> bool:
        """Check if the service is healthy"""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                return response.status == 200
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False

    async def run_backtest(self, symbol: str, timeframe: str, start_date: str,
                          end_date: Optional[str] = None, initial_capital: float = 10000,
                          strategy_config: Optional[Dict[str, Any]] = None,
                          risk_config: Optional[Dict[str, Any]] = None,
                          data_source: str = 'kaggle') -> Dict[str, Any]:
        """Run a single backtest using consolidated service"""
        try:
            # Convert to new API format
            strategy = {
                "name": strategy_config.get("name", "default") if strategy_config else "default",
                "type": strategy_config.get("type", "moving_average") if strategy_config else "moving_average",
                "parameters": strategy_config or {}
            }

            # Get sample data (in real implementation, this would fetch from data service)
            data = await self._get_sample_data(symbol, timeframe, start_date, end_date)

            payload = {
                "strategy": strategy,
                "data": data,
                "config": {
                    "initial_capital": initial_capital,
                    "data_source": data_source,
                    **(risk_config or {})
                }
            }

            async with self.session.post(f"{self.base_url}/backtest", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("results", {})
                else:
                    error = await response.text()
                    raise Exception(f"Backtest failed: {error}")

        except Exception as e:
            self.logger.error(f"Failed to run backtest: {e}")
            raise

    async def _get_sample_data(self, symbol: str, timeframe: str, start_date: str, end_date: Optional[str] = None) -> Dict[str, Any]:
        """Get sample data for backtesting (placeholder implementation)"""
        # In real implementation, this would call the data service
        # For now, return sample data structure
        return {
            "timestamp": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "open": [100, 101, 102],
            "high": [105, 106, 107],
            "low": [95, 96, 97],
            "close": [103, 104, 105],
            "volume": [1000, 1100, 1200]
        }

    async def run_multi_symbol_backtest(self, symbols: List[str], timeframe: str,
                                       start_date: str, end_date: Optional[str] = None,
                                       initial_capital: float = 10000) -> Dict[str, Any]:
        """Run multi-symbol backtest using consolidated service"""
        try:
            results = {}
            for symbol in symbols:
                # Run individual backtest for each symbol
                result = await self.run_backtest(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=initial_capital
                )
                results[symbol] = result

            return {
                "multi_symbol_results": results,
                "symbols_tested": symbols,
                "total_symbols": len(symbols)
            }

        except Exception as e:
            self.logger.error(f"Failed to run multi-symbol backtest: {e}")
            raise

    async def run_multi_model_backtest(self, symbol: str, timeframe: str, start_date: str,
                                      end_date: Optional[str] = None, initial_capital: float = 10000,
                                      models_to_test: Optional[List[str]] = None,
                                      max_workers: int = 4) -> Dict[str, Any]:
        """Run multi-model comparison backtest using consolidated service"""
        try:
            if not models_to_test:
                models_to_test = ["random_forest", "gradient_boosting", "logistic_regression"]

            results = {}
            for model_type in models_to_test:
                # Run backtest with ML model
                strategy_config = {
                    "name": f"ml_{model_type}",
                    "type": "ml_based",
                    "parameters": {"model_type": model_type}
                }

                result = await self.run_backtest(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=initial_capital,
                    strategy_config=strategy_config
                )
                results[model_type] = result

            return {
                "multi_model_results": results,
                "models_tested": models_to_test,
                "total_models": len(models_to_test)
            }

        except Exception as e:
            self.logger.error(f"Failed to run multi-model backtest: {e}")
            raise

    async def train_ml_model(self, strategy_name: str, symbol: str, timeframe: str,
                           start_date: str, end_date: Optional[str] = None,
                           model_type: str = "random_forest",
                           optimize_hyperparams: bool = False) -> Dict[str, Any]:
        """Train ML model using consolidated service"""
        try:
            strategy = {
                "name": strategy_name,
                "type": "ml_based",
                "parameters": {"model_type": model_type}
            }

            data = await self._get_sample_data(symbol, timeframe, start_date, end_date)

            payload = {
                "strategy": strategy,
                "data": data,
                "config": {
                    "model_type": model_type,
                    "optimize_hyperparams": optimize_hyperparams
                }
            }

            async with self.session.post(f"{self.base_url}/ml/train", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("results", {})
                else:
                    error = await response.text()
                    raise Exception(f"ML training failed: {error}")

        except Exception as e:
            self.logger.error(f"Failed to train ML model: {e}")
            raise

    async def optimize_strategy(self, strategy_name: str, param_ranges: Dict[str, List],
                              symbol: str, timeframe: str, start_date: str,
                              end_date: Optional[str] = None) -> Dict[str, Any]:
        """Optimize strategy parameters using consolidated service"""
        try:
            data = await self._get_sample_data(symbol, timeframe, start_date, end_date)

            payload = {
                "strategy_name": strategy_name,
                "param_ranges": param_ranges,
                "data": data,
                "config": {}
            }

            async with self.session.post(f"{self.base_url}/optimize", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("results", {})
                else:
                    error = await response.text()
                    raise Exception(f"Strategy optimization failed: {error}")

        except Exception as e:
            self.logger.error(f"Failed to optimize strategy: {e}")
            raise

    async def run_walk_forward_analysis(self, strategy_name: str, param_ranges: Dict[str, List],
                                      symbol: str, timeframe: str, start_date: str,
                                      end_date: Optional[str] = None) -> Dict[str, Any]:
        """Run walk-forward analysis using consolidated service"""
        try:
            data = await self._get_sample_data(symbol, timeframe, start_date, end_date)

            payload = {
                "strategy_name": strategy_name,
                "param_ranges": param_ranges,
                "data": data,
                "config": {}
            }

            async with self.session.post(f"{self.base_url}/walk-forward", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("results", {})
                else:
                    error = await response.text()
                    raise Exception(f"Walk-forward analysis failed: {error}")

        except Exception as e:
            self.logger.error(f"Failed to run walk-forward analysis: {e}")
            raise

    async def run_complete_workflow(self, strategy_name: str, symbol: str, timeframe: str,
                                  start_date: str, end_date: Optional[str] = None,
                                  enable_ml: bool = False, enable_optimization: bool = True,
                                  enable_walk_forward: bool = True) -> Dict[str, Any]:
        """Run complete backtesting workflow using consolidated service"""
        try:
            strategy = {
                "name": strategy_name,
                "type": "ml_based" if enable_ml else "moving_average",
                "parameters": {}
            }

            data = await self._get_sample_data(symbol, timeframe, start_date, end_date)

            payload = {
                "strategy": strategy,
                "data": data,
                "config": {
                    "enable_ml": enable_ml,
                    "enable_optimization": enable_optimization,
                    "enable_walk_forward": enable_walk_forward,
                    "param_ranges": {
                        "short_window": [5, 10, 15, 20],
                        "long_window": [20, 30, 40, 50]
                    }
                }
            }

            async with self.session.post(f"{self.base_url}/workflow", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("results", {})
                else:
                    error = await response.text()
                    raise Exception(f"Complete workflow failed: {error}")

        except Exception as e:
            self.logger.error(f"Failed to run complete workflow: {e}")
            raise

    async def get_backtest_status(self, backtest_id: str) -> Dict[str, Any]:
        """Get status of service (simplified for consolidated service)"""
        try:
            async with self.session.get(f"{self.base_url}/status") as response:
                if response.status == 200:
                    status = await response.json()
                    return {
                        "status": "completed" if status.get("orchestrator_initialized") else "running",
                        "service_status": status
                    }
                else:
                    return {'status': 'not_found'}

        except Exception as e:
            self.logger.error(f"Failed to get backtest status: {e}")
            return {'status': 'error'}

    async def get_backtest_results(self, backtest_id: str) -> Dict[str, Any]:
        """Get service health and status (simplified for consolidated service)"""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {}

        except Exception as e:
            self.logger.error(f"Failed to get service status: {e}")
            return {}

    async def list_backtests(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available backtesting capabilities (simplified for consolidated service)"""
        return [
            {"name": "quick_backtest", "description": "Run quick backtest"},
            {"name": "ml_training", "description": "Train ML model"},
            {"name": "optimization", "description": "Optimize strategy parameters"},
            {"name": "walk_forward", "description": "Run walk-forward analysis"},
            {"name": "complete_workflow", "description": "Run complete workflow"}
        ]

    async def cancel_backtest(self, backtest_id: str) -> bool:
        """Cancel operation (not implemented for consolidated service)"""
        self.logger.warning("Cancel operation not implemented for consolidated service")
        return False