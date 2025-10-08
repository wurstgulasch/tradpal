"""
Test discovery module for genetic algorithm optimization of trading indicators.
Tests GA optimization, configuration conversion, and adaptive config management.
"""

import json
import os
import tempfile
from unittest.mock import patch, MagicMock
import pytest
import pandas as pd
import numpy as np

from src.discovery import (
    DiscoveryOptimizer, IndividualResult, load_adaptive_config,
    save_adaptive_config, apply_adaptive_config, run_discovery,
    DEAP_AVAILABLE
)


class TestDiscoveryBasics:
    """Test basic discovery functionality."""

    def test_deap_availability(self):
        """Test DEAP library availability detection."""
        # This should be a boolean
        assert isinstance(DEAP_AVAILABLE, bool)

    def test_individual_result_creation(self):
        """Test IndividualResult dataclass creation."""
        config = {"ema": {"enabled": True, "periods": [9, 21]}}
        result = IndividualResult(
            config=config,
            fitness=1.5,
            pnl=15.5,
            win_rate=0.65,
            sharpe_ratio=1.2,
            max_drawdown=5.5,
            total_trades=25,
            evaluation_time=2.3,
            backtest_duration_days=100
        )

        assert result.config == config
        assert result.fitness == 1.5
        assert result.pnl == 15.5
        assert result.win_rate == 0.65
        assert result.sharpe_ratio == 1.2
        assert result.max_drawdown == 5.5
        assert result.total_trades == 25
        assert result.evaluation_time == 2.3
        assert result.backtest_duration_days == 100


class TestDiscoveryOptimizer:
    """Test DiscoveryOptimizer class."""

    def test_optimizer_initialization_deap_available(self):
        """Test optimizer initialization when DEAP is available."""
        if not DEAP_AVAILABLE:
            pytest.skip("DEAP library not available")

        optimizer = DiscoveryOptimizer(
            symbol="BTC/USDT",
            exchange="binance",
            timeframe="1h",
            population_size=10,
            generations=5
        )

        assert optimizer.symbol == "BTC/USDT"
        assert optimizer.exchange == "binance"
        assert optimizer.timeframe == "1h"
        assert optimizer.population_size == 10
        assert optimizer.generations == 5
        assert hasattr(optimizer, 'toolbox')
        assert optimizer.results == []

    def test_optimizer_initialization_deap_unavailable(self):
        """Test optimizer initialization when DEAP is not available."""
        if DEAP_AVAILABLE:
            pytest.skip("DEAP library is available - cannot test unavailable case")

        with pytest.raises(ImportError, match="Discovery module requires 'deap' package"):
            DiscoveryOptimizer()

    def test_individual_to_config_conversion(self):
        """Test conversion from GA individual to configuration dict."""
        if not DEAP_AVAILABLE:
            pytest.skip("DEAP library not available")

        optimizer = DiscoveryOptimizer(population_size=10, generations=1)

        # Create a mock individual
        individual = [9, 21, 14, 30, 70, 20, 2.0, 14, True, True, True, False]

        config = optimizer._individual_to_config(individual)

        assert config['ema']['enabled'] is True
        assert config['ema']['periods'] == [9, 21]
        assert config['rsi']['enabled'] is True
        assert config['rsi']['period'] == 14
        assert config['rsi']['oversold'] == 30
        assert config['rsi']['overbought'] == 70
        assert config['bb']['enabled'] is True
        assert config['bb']['period'] == 20
        assert config['bb']['std_dev'] == 2.0
        assert config['atr']['enabled'] is True
        assert config['atr']['period'] == 14
        assert config['adx']['enabled'] is False

    def test_config_to_tuple_conversion(self):
        """Test conversion from config to tuple for duplicate detection."""
        if not DEAP_AVAILABLE:
            pytest.skip("DEAP library not available")

        optimizer = DiscoveryOptimizer(population_size=10, generations=1)

        config = {
            'ema': {'enabled': True, 'periods': [9, 21]},
            'rsi': {'enabled': True, 'period': 14, 'oversold': 30, 'overbought': 70},
            'bb': {'enabled': True, 'period': 20, 'std_dev': 2.0},
            'atr': {'enabled': True, 'period': 14},
            'adx': {'enabled': False}
        }

        config_tuple = optimizer._config_to_tuple(config)

        expected = ((9, 21), True, 14, 30, 70, True, 20, 2.0, True, 14, False)
        assert config_tuple == expected

    @patch('src.discovery.fetch_historical_data')
    def test_load_historical_data(self, mock_fetch):
        """Test loading historical data."""
        if not DEAP_AVAILABLE:
            pytest.skip("DEAP library not available")

        # Mock historical data
        mock_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
            'open': np.random.randn(100),
            'high': np.random.randn(100),
            'low': np.random.randn(100),
            'close': np.random.randn(100),
            'volume': np.random.randint(100, 1000, 100)
        })
        mock_fetch.return_value = mock_data

        optimizer = DiscoveryOptimizer(population_size=10, generations=1)

        data = optimizer._load_historical_data()

        assert len(data) == 100
        assert optimizer.historical_data is not None
        mock_fetch.assert_called_once()

        # Second call should use cached data
        data2 = optimizer._load_historical_data()
        assert data is data2
        mock_fetch.assert_called_once()  # Should not be called again

    def test_simulate_trades(self):
        """Test trade simulation from signals."""
        if not DEAP_AVAILABLE:
            pytest.skip("DEAP library not available")

        optimizer = DiscoveryOptimizer(population_size=10, generations=1)

        # Create mock data with signals
        data = pd.DataFrame({
            'close': [100, 105, 102, 108, 106],
            'Buy_Signal': [1, 0, 0, 0, 0],
            'Sell_Signal': [0, 0, 0, 1, 0]
        })

        trades_df = optimizer._simulate_trades(data)

        assert len(trades_df) == 1
        assert trades_df.iloc[0]['entry_price'] == 100
        assert trades_df.iloc[0]['exit_price'] == 108
        assert trades_df.iloc[0]['pnl'] == (108 - 100) / 100 * 100  # 8% profit

    def test_simulate_trades_no_signals(self):
        """Test trade simulation with no signals."""
        if not DEAP_AVAILABLE:
            pytest.skip("DEAP library not available")

        optimizer = DiscoveryOptimizer(population_size=10, generations=1)

        # Create mock data with no signals
        data = pd.DataFrame({
            'close': [100, 105, 102, 108, 106],
            'Buy_Signal': [0, 0, 0, 0, 0],
            'Sell_Signal': [0, 0, 0, 0, 0]
        })

        trades_df = optimizer._simulate_trades(data)

        assert len(trades_df) == 0

    def test_calculate_fitness_metrics(self):
        """Test fitness metrics calculation."""
        if not DEAP_AVAILABLE:
            pytest.skip("DEAP library not available")

        optimizer = DiscoveryOptimizer(population_size=10, generations=1)

        # Create mock trades DataFrame
        trades_df = pd.DataFrame({
            'pnl': [5.0, -2.0, 8.0, 3.0, -1.0]  # Mix of profits and losses
        })

        fitness, metrics = optimizer._calculate_fitness_metrics(trades_df)

        assert fitness > 0  # Should be positive
        assert metrics['total_pnl'] == 13.0  # 5 + (-2) + 8 + 3 + (-1)
        assert metrics['win_rate'] == 0.6  # 3 wins out of 5 trades
        assert metrics['total_trades'] == 5
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics

    def test_calculate_fitness_metrics_empty_trades(self):
        """Test fitness metrics calculation with no trades."""
        if not DEAP_AVAILABLE:
            pytest.skip("DEAP library not available")

        optimizer = DiscoveryOptimizer(population_size=10, generations=1)

        trades_df = pd.DataFrame()

        fitness, metrics = optimizer._calculate_fitness_metrics(trades_df)

        assert fitness == 0.0
        assert metrics['total_pnl'] == 0.0
        assert metrics['win_rate'] == 0.0
        assert metrics['total_trades'] == 0

    def test_remove_duplicate_configs(self):
        """Test removal of duplicate configurations."""
        if not DEAP_AVAILABLE:
            pytest.skip("DEAP library not available")

        optimizer = DiscoveryOptimizer(population_size=10, generations=1)

        # Create mock results with duplicates
        config1 = {'ema': {'periods': [9, 21]}, 'rsi': {'enabled': True}}
        config2 = {'ema': {'periods': [12, 26]}, 'rsi': {'enabled': False}}

        results = [
            IndividualResult(config=config1, fitness=1.5, pnl=0, win_rate=0, sharpe_ratio=0, max_drawdown=0, total_trades=0, evaluation_time=0, backtest_duration_days=0),
            IndividualResult(config=config1, fitness=1.2, pnl=0, win_rate=0, sharpe_ratio=0, max_drawdown=0, total_trades=0, evaluation_time=0, backtest_duration_days=0),  # Duplicate with lower fitness
            IndividualResult(config=config2, fitness=1.8, pnl=0, win_rate=0, sharpe_ratio=0, max_drawdown=0, total_trades=0, evaluation_time=0, backtest_duration_days=0)
        ]

        unique_results = optimizer._remove_duplicate_configs(results, max_results=10)

        assert len(unique_results) == 2
        # Should keep the higher fitness duplicate
        assert unique_results[0].fitness == 1.8  # config2 with highest fitness
        assert unique_results[1].fitness == 1.5  # config1 with lower fitness

    @patch('src.discovery.json.dump')
    def test_save_results(self, mock_json_dump):
        """Test saving optimization results."""
        if not DEAP_AVAILABLE:
            pytest.skip("DEAP library not available")

        optimizer = DiscoveryOptimizer(population_size=10, generations=1)

        # Create mock results
        results = [
            IndividualResult(
                config={'ema': {'periods': [9, 21]}},
                fitness=1.5,
                pnl=15.5,
                win_rate=0.65,
                sharpe_ratio=1.2,
                max_drawdown=5.5,
                total_trades=25,
                evaluation_time=2.3,
                backtest_duration_days=100
            )
        ]

        with patch('builtins.open', create=True):
            optimizer.save_results(results, 'test_output.json')

        # Verify json.dump was called
        assert mock_json_dump.called

    def test_evaluate_individual_error_handling(self):
        """Test error handling in individual evaluation."""
        if not DEAP_AVAILABLE:
            pytest.skip("DEAP library not available")

        optimizer = DiscoveryOptimizer(population_size=10, generations=1)

        # Create an individual that should cause issues
        individual = [0, 0, 0, 0, 0, 0, 0, 0, False, False, False, False]  # All disabled

        fitness = optimizer._evaluate_individual(individual)

        # Should return 0.0 fitness on error
        assert fitness == (0.0,)


class TestAdaptiveConfig:
    """Test adaptive configuration management."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_adaptive_config(self):
        """Test saving adaptive configuration."""
        config_file = os.path.join(self.temp_dir, 'adaptive_config.json')
        config = {'ema': {'periods': [9, 21]}, 'rsi': {'enabled': True}}
        fitness = 1.5

        save_adaptive_config(config, fitness, config_file)

        assert os.path.exists(config_file)

        # Verify content
        with open(config_file, 'r') as f:
            data = json.load(f)

        assert data['fitness_score'] == 1.5
        assert data['best_configuration'] == config
        assert 'timestamp' in data

    def test_load_adaptive_config_success(self):
        """Test loading adaptive configuration successfully."""
        config_file = os.path.join(self.temp_dir, 'adaptive_config.json')
        config = {'ema': {'periods': [9, 21]}, 'rsi': {'enabled': True}}
        fitness = 1.5

        # Save first
        save_adaptive_config(config, fitness, config_file)

        # Load
        loaded_config = load_adaptive_config(config_file)

        assert loaded_config == config

    def test_load_adaptive_config_file_not_found(self):
        """Test loading adaptive config when file doesn't exist."""
        config_file = os.path.join(self.temp_dir, 'nonexistent.json')

        loaded_config = load_adaptive_config(config_file)

        assert loaded_config is None

    def test_load_adaptive_config_invalid_structure(self):
        """Test loading adaptive config with invalid structure."""
        config_file = os.path.join(self.temp_dir, 'invalid_config.json')

        # Create invalid config
        with open(config_file, 'w') as f:
            json.dump({'invalid': 'structure'}, f)

        loaded_config = load_adaptive_config(config_file)

        assert loaded_config is None

    @patch('config.settings.DEFAULT_INDICATOR_CONFIG')
    def test_apply_adaptive_config(self, mock_default_config):
        """Test applying adaptive configuration."""
        # Mock the copy method and the resulting dict
        mock_copied_config = {
            'ema': {'enabled': False},
            'rsi': {'enabled': False}
        }
        mock_default_config.copy.return_value = mock_copied_config

        config = {
            'ema': {'enabled': True, 'periods': [9, 21]},
            'rsi': {'enabled': True, 'period': 14}
        }

        updated_config = apply_adaptive_config(config)

        assert updated_config['ema']['enabled'] is True
        assert updated_config['ema']['periods'] == [9, 21]
        assert updated_config['rsi']['enabled'] is True


class TestDiscoveryIntegration:
    """Test discovery module integration."""

    def test_run_discovery_deap_unavailable(self):
        """Test run_discovery when DEAP is not available."""
        if DEAP_AVAILABLE:
            pytest.skip("DEAP library is available - cannot test unavailable case")

        with pytest.raises(ImportError, match="Discovery module requires 'deap' package"):
            run_discovery()

    @patch('src.discovery.DiscoveryOptimizer')
    def test_run_discovery_success(self, mock_optimizer_class):
        """Test successful run_discovery execution."""
        if not DEAP_AVAILABLE:
            pytest.skip("DEAP library not available")

        # Mock the optimizer
        mock_optimizer = MagicMock()
        mock_results = [IndividualResult(
            config={'ema': {'periods': [9, 21]}},
            fitness=1.5,
            pnl=0, win_rate=0, sharpe_ratio=0, max_drawdown=0,
            total_trades=0, evaluation_time=0, backtest_duration_days=0
        )]
        mock_optimizer.optimize.return_value = mock_results
        mock_optimizer_class.return_value = mock_optimizer

        results = run_discovery(
            symbol="BTC/USDT",
            population_size=10,
            generations=2
        )

        assert results == mock_results
        mock_optimizer_class.assert_called_once()
        mock_optimizer.optimize.assert_called_once()
        mock_optimizer.save_results.assert_called_once()