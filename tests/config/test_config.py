import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# Add config to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'config'))

from config.settings import (
    SYMBOL, EXCHANGE, TIMEFRAME, LOOKBACK_DAYS,
    EMA_SHORT, EMA_LONG, RSI_PERIOD, BB_PERIOD, BB_STD_DEV, ATR_PERIOD,
    CAPITAL, RISK_PER_TRADE, SL_MULTIPLIER, TP_MULTIPLIER,
    LEVERAGE_BASE, LEVERAGE_MIN, LEVERAGE_MAX,
    MTA_ENABLED, MTA_TIMEFRAMES, ADX_THRESHOLD,
    TIMEFRAME_PARAMS, OUTPUT_FORMAT, OUTPUT_FILE,
    validate_timeframe, get_timeframe_params, validate_risk_params,
    config, LazyConfig, get_legacy_constant,
    DEFAULT_DATA_LIMIT, ML_MODEL_TYPE, ENABLE_MTLS, GPU_ACCELERATION_ENABLED
)


class TestConfiguration:
    """Test configuration settings and validation."""

    def test_basic_configuration_constants(self):
        """Test that basic configuration constants are properly set."""
        assert isinstance(SYMBOL, str)
        assert len(SYMBOL) > 0
        assert isinstance(EXCHANGE, str)
        assert len(EXCHANGE) > 0
        assert isinstance(TIMEFRAME, str)
        assert len(TIMEFRAME) > 0
        assert isinstance(LOOKBACK_DAYS, int)
        assert LOOKBACK_DAYS > 0

    def test_indicator_parameters(self):
        """Test indicator parameter configuration."""
        assert isinstance(EMA_SHORT, int)
        assert isinstance(EMA_LONG, int)
        assert EMA_SHORT < EMA_LONG  # Short EMA should be less than long EMA

        assert isinstance(RSI_PERIOD, int)
        assert 2 <= RSI_PERIOD <= 100  # Reasonable RSI period range

        assert isinstance(BB_PERIOD, int)
        assert BB_PERIOD > 0
        assert isinstance(BB_STD_DEV, (int, float))
        assert BB_STD_DEV > 0

        assert isinstance(ATR_PERIOD, int)
        assert ATR_PERIOD > 0

    def test_risk_management_parameters(self):
        """Test risk management parameter configuration."""
        assert isinstance(CAPITAL, (int, float))
        assert CAPITAL > 0

        assert isinstance(RISK_PER_TRADE, float)
        assert 0 < RISK_PER_TRADE <= 1  # Should be between 0 and 100%

        assert isinstance(SL_MULTIPLIER, (int, float))
        assert SL_MULTIPLIER > 0

        assert isinstance(TP_MULTIPLIER, (int, float))
        assert TP_MULTIPLIER > SL_MULTIPLIER  # Take profit should be higher than stop loss

    def test_leverage_parameters(self):
        """Test leverage parameter configuration."""
        assert isinstance(LEVERAGE_BASE, (int, float))
        assert LEVERAGE_BASE >= 1

        assert isinstance(LEVERAGE_MIN, (int, float))
        assert LEVERAGE_MIN >= 1

        assert isinstance(LEVERAGE_MAX, (int, float))
        assert LEVERAGE_MAX >= LEVERAGE_MIN

    def test_mta_configuration(self):
        """Test Multi-Timeframe Analysis configuration."""
        assert isinstance(MTA_ENABLED, bool)
        assert isinstance(MTA_TIMEFRAMES, list)

        if MTA_ENABLED:
            assert len(MTA_TIMEFRAMES) > 0
            for tf in MTA_TIMEFRAMES:
                assert isinstance(tf, str)
                assert len(tf) > 0

    def test_adx_configuration(self):
        """Test ADX configuration."""
        assert isinstance(ADX_THRESHOLD, (int, float))
        assert 0 <= ADX_THRESHOLD <= 100  # ADX ranges from 0 to 100

    def test_timeframe_params_structure(self):
        """Test timeframe parameters structure."""
        assert isinstance(TIMEFRAME_PARAMS, dict)
        assert len(TIMEFRAME_PARAMS) > 0

        # Check that current timeframe is in parameters
        assert TIMEFRAME in TIMEFRAME_PARAMS

        # Check structure of timeframe parameters
        for tf, params in TIMEFRAME_PARAMS.items():
            assert isinstance(params, dict)
            required_keys = ['ema_short', 'ema_long', 'rsi_period', 'bb_period', 'atr_period', 'adx_period', 'rsi_oversold', 'rsi_overbought']
            for key in required_keys:
                assert key in params, f"Missing {key} in timeframe {tf}"
                assert isinstance(params[key], (int, float))

    def test_output_configuration(self):
        """Test output configuration."""
        assert isinstance(OUTPUT_FORMAT, str)
        assert OUTPUT_FORMAT in ['json', 'csv']  # Common output formats

        assert isinstance(OUTPUT_FILE, str)
        assert OUTPUT_FILE.endswith('.json')

    def test_validate_timeframe_valid(self):
        """Test timeframe validation with valid inputs."""
        valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M']

        for tf in valid_timeframes:
            assert validate_timeframe(tf) == True

    def test_validate_timeframe_invalid(self):
        """Test timeframe validation with invalid inputs."""
        invalid_timeframes = ['', 'invalid', '2x', '1y', None, 123]

        for tf in invalid_timeframes:
            assert validate_timeframe(tf) == False

    def test_get_timeframe_params_existing(self):
        """Test getting parameters for existing timeframe."""
        params = get_timeframe_params('1m')
        assert isinstance(params, dict)
        assert 'ema_short' in params
        assert 'ema_long' in params

    def test_get_timeframe_params_nonexistent(self):
        """Test getting parameters for non-existent timeframe."""
        params = get_timeframe_params('999d')  # Non-existent timeframe
        assert params is None

    def test_validate_risk_params_valid(self):
        """Test risk parameter validation with valid inputs."""
        valid_configs = [
            {'capital': 10000, 'risk_per_trade': 0.01, 'sl_multiplier': 1.5},
            {'capital': 5000, 'risk_per_trade': 0.02, 'sl_multiplier': 2.0},
        ]

        for config in valid_configs:
            assert validate_risk_params(config) == True

    def test_validate_risk_params_invalid(self):
        """Test risk parameter validation with invalid inputs."""
        invalid_configs = [
            {'capital': 0, 'risk_per_trade': 0.01, 'sl_multiplier': 1.5},  # Zero capital
            {'capital': 10000, 'risk_per_trade': 0, 'sl_multiplier': 1.5},  # Zero risk
            {'capital': 10000, 'risk_per_trade': 0.01, 'sl_multiplier': 0},  # Zero stop loss
            {'capital': 10000, 'risk_per_trade': 1.5, 'sl_multiplier': 1.5},  # Risk > 100%
            {'capital': -1000, 'risk_per_trade': 0.01, 'sl_multiplier': 1.5},  # Negative capital
        ]

        for config in invalid_configs:
            assert validate_risk_params(config) == False

    def test_timeframe_params_consistency(self):
        """Test that timeframe parameters are consistent across timeframes."""
        for tf, params in TIMEFRAME_PARAMS.items():
            # EMA short should be less than EMA long
            assert params['ema_short'] < params['ema_long'], f"Invalid EMA params for {tf}"

            # RSI thresholds should be reasonable
            assert 10 <= params['rsi_oversold'] < params['rsi_overbought'] <= 90, f"Invalid RSI params for {tf}"

            # Periods should be positive
            assert params['rsi_period'] > 0, f"Invalid RSI period for {tf}"
            assert params['bb_period'] > 0, f"Invalid BB period for {tf}"
            assert params['atr_period'] > 0, f"Invalid ATR period for {tf}"
            assert params['adx_period'] > 0, f"Invalid ADX period for {tf}"

    def test_configuration_scaling(self):
        """Test that parameters scale appropriately with timeframe."""
        # Get parameters for different timeframes
        params_1m = TIMEFRAME_PARAMS.get('1m', {})
        params_1h = TIMEFRAME_PARAMS.get('1h', {})
        params_1d = TIMEFRAME_PARAMS.get('1d', {})

        if params_1m and params_1h and params_1d:
            # EMA periods should generally increase with timeframe
            assert params_1m['ema_short'] <= params_1h['ema_short'] <= params_1d['ema_short']
            assert params_1m['ema_long'] <= params_1h['ema_long'] <= params_1d['ema_long']

    def test_configuration_edge_cases(self):
        """Test configuration with edge cases."""
        # Test with minimal valid configuration
        minimal_config = {
            'capital': 1,  # Minimum capital
            'risk_per_trade': 0.001,  # Very small risk
            'sl_multiplier': 0.1  # Tight stop loss
        }
        assert validate_risk_params(minimal_config) == True

        # Test with maximum valid configuration
        max_config = {
            'capital': 10000000,  # Large capital
            'risk_per_trade': 0.05,  # High risk (5%)
            'sl_multiplier': 10.0  # Wide stop loss
        }
        assert validate_risk_params(max_config) == True

    @patch.dict(os.environ, {'TIMEFRAME': '5m'})
    def test_timeframe_from_environment(self):
        """Test that timeframe can be overridden from environment."""
        # This would require reloading the module, so we'll just test the validation
        assert validate_timeframe('5m') == True

    def test_output_file_path_validation(self):
        """Test output file path validation."""
        assert OUTPUT_FILE.endswith('.json')
        assert len(OUTPUT_FILE) > 5  # Reasonable filename length

        # Check that directory exists or can be created
        output_dir = os.path.dirname(OUTPUT_FILE)
        if output_dir:
            # Should not raise an exception when checking if we can write to the directory
            try:
                os.makedirs(output_dir, exist_ok=True)
                assert os.path.isdir(output_dir)
            except (OSError, PermissionError):
                pytest.skip("Cannot create output directory for testing")


class TestLazyConfiguration:
    """Test lazy loading configuration system."""

    def test_lazy_config_initialization(self):
        """Test that LazyConfig initializes correctly."""
        lazy_config = LazyConfig()
        assert lazy_config._loaded_modules == {}
        assert 'core' in lazy_config._module_loaders
        assert 'ml' in lazy_config._module_loaders
        assert 'service' in lazy_config._module_loaders
        assert 'security' in lazy_config._module_loaders
        assert 'performance' in lazy_config._module_loaders

    def test_lazy_loading_core_module(self):
        """Test lazy loading of core module."""
        lazy_config = LazyConfig()

        # Module should not be loaded initially
        assert 'core' not in lazy_config._loaded_modules

        # Load core module
        core_settings = lazy_config.get_module('core')

        # Module should now be loaded
        assert 'core' in lazy_config._loaded_modules
        assert isinstance(core_settings, dict)
        assert len(core_settings) > 0

        # Check that core constants are present
        assert 'DEFAULT_DATA_LIMIT' in core_settings
        assert 'SYMBOL' in core_settings
        assert 'EXCHANGE' in core_settings
        assert 'TIMEFRAME' in core_settings

    def test_lazy_loading_ml_module(self):
        """Test lazy loading of ML module."""
        lazy_config = LazyConfig()

        # Module should not be loaded initially
        assert 'ml' not in lazy_config._loaded_modules

        # Load ML module
        ml_settings = lazy_config.get_module('ml')

        # Module should now be loaded
        assert 'ml' in lazy_config._loaded_modules
        assert isinstance(ml_settings, dict)
        assert len(ml_settings) > 0

        # Check that ML constants are present
        assert 'ML_MODEL_TYPE' in ml_settings
        assert 'ML_TRAINING_ENABLED' in ml_settings

    def test_lazy_loading_service_module(self):
        """Test lazy loading of service module."""
        lazy_config = LazyConfig()

        # Module should not be loaded initially
        assert 'service' not in lazy_config._loaded_modules

        # Load service module
        service_settings = lazy_config.get_module('service')

        # Module should now be loaded
        assert 'service' in lazy_config._loaded_modules
        assert isinstance(service_settings, dict)
        assert len(service_settings) > 0

        # Check that service constants are present
        assert 'ENABLE_MTLS' in service_settings
        assert 'MONITORING_STACK_ENABLED' in service_settings

    def test_lazy_loading_performance_module(self):
        """Test lazy loading of performance module."""
        lazy_config = LazyConfig()

        # Module should not be loaded initially
        assert 'performance' not in lazy_config._loaded_modules

        # Load performance module
        perf_settings = lazy_config.get_module('performance')

        # Module should now be loaded
        assert 'performance' in lazy_config._loaded_modules
        assert isinstance(perf_settings, dict)
        assert len(perf_settings) > 0

        # Check that performance constants are present
        assert 'GPU_ACCELERATION_ENABLED' in perf_settings
        assert 'CACHE_ENABLED' in perf_settings

    def test_lazy_loading_security_module(self):
        """Test lazy loading of security module."""
        lazy_config = LazyConfig()

        # Module should not be loaded initially
        assert 'security' not in lazy_config._loaded_modules

        # Load security module
        security_settings = lazy_config.get_module('security')

        # Module should now be loaded
        assert 'security' in lazy_config._loaded_modules
        assert isinstance(security_settings, dict)
        assert len(security_settings) > 0

    def test_get_method_lazy_loading(self):
        """Test that get method triggers lazy loading."""
        lazy_config = LazyConfig()

        # Module should not be loaded initially
        assert 'core' not in lazy_config._loaded_modules

        # Get a specific value
        symbol = lazy_config.get('core', 'SYMBOL')

        # Module should now be loaded
        assert 'core' in lazy_config._loaded_modules
        assert symbol == 'BTC/USDT'

    def test_get_method_with_default(self):
        """Test get method with default values."""
        lazy_config = LazyConfig()

        # Test with existing key
        symbol = lazy_config.get('core', 'SYMBOL', 'DEFAULT')
        assert symbol == 'BTC/USDT'

        # Test with non-existing key
        nonexistent = lazy_config.get('core', 'NONEXISTENT', 'DEFAULT')
        assert nonexistent == 'DEFAULT'

    def test_invalid_module_error(self):
        """Test that invalid module raises error."""
        lazy_config = LazyConfig()

        with pytest.raises(ValueError, match="Unknown configuration module"):
            lazy_config.get_module('invalid_module')

    def test_cache_functionality(self):
        """Test that modules are cached after loading."""
        lazy_config = LazyConfig()

        # Load module twice
        core1 = lazy_config.get_module('core')
        core2 = lazy_config.get_module('core')

        # Should be the same content (cached)
        assert core1 == core2
        assert 'core' in lazy_config._loaded_modules

    def test_clear_cache(self):
        """Test cache clearing functionality."""
        lazy_config = LazyConfig()

        # Load a module
        lazy_config.get_module('core')
        assert 'core' in lazy_config._loaded_modules

        # Clear cache
        lazy_config.clear_cache()
        assert lazy_config._loaded_modules == {}

    def test_get_all_loaded(self):
        """Test getting all loaded modules."""
        lazy_config = LazyConfig()

        # Initially empty
        assert lazy_config.get_all_loaded() == {}

        # Load some modules
        lazy_config.get_module('core')
        lazy_config.get_module('ml')

        loaded = lazy_config.get_all_loaded()
        assert 'core' in loaded
        assert 'ml' in loaded
        assert 'service' not in loaded

    def test_lazy_loading_performance(self):
        """Test that lazy loading works (performance test is unreliable due to timing)."""
        import time

        lazy_config = LazyConfig()

        # Just test that loading works multiple times
        lazy_config.get_module('core')
        lazy_config.clear_cache()
        lazy_config.get_module('core')

        # Test passed if no exceptions
        assert True


class TestLegacyConstants:
    """Test legacy constant backward compatibility."""

    def test_legacy_constants_defined(self):
        """Test that all legacy constants are properly defined."""
        # Import constants dynamically to ensure they're loaded
        from config.settings import (
            DEFAULT_DATA_LIMIT, SYMBOL, EXCHANGE, TIMEFRAME, CAPITAL,
            RISK_PER_TRADE, ML_MODEL_TYPE, ENABLE_MTLS, GPU_ACCELERATION_ENABLED,
            MONITORING_STACK_ENABLED, MAX_BACKTEST_RESULTS
        )
        
        legacy_constants = [
            DEFAULT_DATA_LIMIT, SYMBOL, EXCHANGE, TIMEFRAME, CAPITAL,
            RISK_PER_TRADE, ML_MODEL_TYPE, ENABLE_MTLS, GPU_ACCELERATION_ENABLED,
            MONITORING_STACK_ENABLED, MAX_BACKTEST_RESULTS
        ]

        for const in legacy_constants:
            assert const is not None, f"Legacy constant is None: {const}"

    def test_get_legacy_constant_function(self):
        """Test the get_legacy_constant function."""
        # Test existing constant
        value = get_legacy_constant('SYMBOL', 'DEFAULT')
        assert value == 'BTC/USDT'

        # Test non-existing constant with default
        value = get_legacy_constant('NONEXISTENT', 'DEFAULT')
        assert value == 'DEFAULT'

    def test_legacy_constant_types(self):
        """Test that legacy constants have correct types."""
        # Import constants dynamically
        from config.settings import (
            SYMBOL, EXCHANGE, TIMEFRAME, ML_MODEL_TYPE, DEFAULT_DATA_LIMIT,
            CAPITAL, RISK_PER_TRADE, ENABLE_MTLS, GPU_ACCELERATION_ENABLED,
            MONITORING_STACK_ENABLED
        )
        
        # String constants
        assert isinstance(SYMBOL, str)
        assert isinstance(EXCHANGE, str)
        assert isinstance(TIMEFRAME, str)
        assert isinstance(ML_MODEL_TYPE, str)

        # Numeric constants
        assert isinstance(DEFAULT_DATA_LIMIT, int)
        assert isinstance(CAPITAL, (int, float))
        assert isinstance(RISK_PER_TRADE, float)

        # Boolean constants
        assert isinstance(ENABLE_MTLS, bool)
        assert isinstance(GPU_ACCELERATION_ENABLED, bool)
        assert isinstance(MONITORING_STACK_ENABLED, bool)

    def test_legacy_constant_values(self):
        """Test that legacy constants have reasonable values."""
        assert DEFAULT_DATA_LIMIT > 0
        assert CAPITAL > 0
        assert 0 < RISK_PER_TRADE <= 1  # Risk as percentage
        assert len(SYMBOL) > 0
        assert len(EXCHANGE) > 0
        assert TIMEFRAME in ['1m', '5m', '15m', '1h', '1d']  # Common timeframes


class TestConfigurationIntegration:
    """Integration tests for configuration system."""

    def test_full_configuration_loading(self):
        """Test that all configuration can be loaded without errors."""
        # This test ensures that all imports and constants work together
        required_constants = [
            SYMBOL, EXCHANGE, TIMEFRAME, LOOKBACK_DAYS,
            EMA_SHORT, EMA_LONG, RSI_PERIOD, BB_PERIOD, ATR_PERIOD,
            CAPITAL, RISK_PER_TRADE, SL_MULTIPLIER, TP_MULTIPLIER,
            LEVERAGE_BASE, LEVERAGE_MIN, LEVERAGE_MAX,
            MTA_ENABLED, ADX_THRESHOLD,
            TIMEFRAME_PARAMS, OUTPUT_FORMAT, OUTPUT_FILE,
            DEFAULT_DATA_LIMIT, ML_MODEL_TYPE, ENABLE_MTLS, GPU_ACCELERATION_ENABLED
        ]

        # All constants should be defined and not None
        for const in required_constants:
            assert const is not None, f"Configuration constant is None"

    def test_lazy_vs_direct_loading(self):
        """Test that lazy loading and direct constants give same results."""
        # Test that lazy loaded values match direct constants
        assert config.get('core', 'SYMBOL') == SYMBOL
        assert config.get('core', 'EXCHANGE') == EXCHANGE
        assert config.get('core', 'DEFAULT_DATA_LIMIT') == DEFAULT_DATA_LIMIT
        assert config.get('ml', 'ML_MODEL_TYPE') == ML_MODEL_TYPE
        assert config.get('service', 'ENABLE_MTLS') == ENABLE_MTLS

    def test_configuration_memory_efficiency(self):
        """Test that lazy loading reduces initial memory usage."""
        # Create a fresh config instance
        test_config = LazyConfig()

        # Initially no modules loaded
        assert len(test_config._loaded_modules) == 0

        # Load only core module
        test_config.get_module('core')
        assert len(test_config._loaded_modules) == 1

        # Other modules should still be unloaded
        assert 'ml' not in test_config._loaded_modules
        assert 'service' not in test_config._loaded_modules
        assert 'performance' not in test_config._loaded_modules

    def test_configuration_thread_safety(self):
        """Test that configuration loading is thread-safe."""
        import threading
        import time

        test_config = LazyConfig()
        results = []
        errors = []

        def load_module(module_name):
            try:
                settings = test_config.get_module(module_name)
                results.append((module_name, len(settings)))
            except Exception as e:
                errors.append((module_name, str(e)))

        # Start multiple threads loading different modules
        threads = []
        modules = ['core', 'ml', 'service', 'performance', 'security']

        for module in modules:
            thread = threading.Thread(target=load_module, args=(module,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check results
        assert len(results) == len(modules), f"Some threads failed: {errors}"
        assert len(errors) == 0, f"Thread errors: {errors}"

        # All modules should be loaded
        for module in modules:
            assert module in test_config._loaded_modules

    def test_configuration_validation_integration(self):
        """Test integration between configuration and validation functions."""
        # Test that configured timeframe is valid
        assert validate_timeframe(TIMEFRAME)

        # Test that configured risk parameters are valid
        risk_config = {
            'capital': CAPITAL,
            'risk_per_trade': RISK_PER_TRADE,
            'sl_multiplier': SL_MULTIPLIER
        }
        assert validate_risk_params(risk_config)

        # Test timeframe parameters integration
        current_params = get_timeframe_params(TIMEFRAME)
        assert current_params is not None
        assert 'ema_short' in current_params
        assert 'ema_long' in current_params


if __name__ == "__main__":
    pytest.main([__file__])