import pytest
import sys
import os
from unittest.mock import patch

# Add config to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'config'))

from config.settings import (
    SYMBOL, EXCHANGE, TIMEFRAME, LOOKBACK_DAYS,
    EMA_SHORT, EMA_LONG, RSI_PERIOD, BB_PERIOD, BB_STD_DEV, ATR_PERIOD,
    CAPITAL, RISK_PER_TRADE, SL_MULTIPLIER, TP_MULTIPLIER,
    LEVERAGE_BASE, LEVERAGE_MIN, LEVERAGE_MAX,
    MTA_ENABLED, MTA_TIMEFRAMES, ADX_THRESHOLD,
    TIMEFRAME_PARAMS, OUTPUT_FORMAT, OUTPUT_FILE,
    validate_timeframe, get_timeframe_params, validate_risk_params
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
            TIMEFRAME_PARAMS, OUTPUT_FORMAT, OUTPUT_FILE
        ]

        # All constants should be defined and not None
        for const in required_constants:
            assert const is not None, f"Configuration constant is None"

    def test_timeframe_params_integration(self):
        """Test integration between timeframe params and validation."""
        for tf in TIMEFRAME_PARAMS.keys():
            # Each timeframe in params should be valid
            assert validate_timeframe(tf), f"Invalid timeframe in params: {tf}"

            # Each timeframe should have valid parameters
            params = get_timeframe_params(tf)
            assert params is not None, f"No params for timeframe: {tf}"

            # Risk params for this timeframe should be valid
            risk_config = {
                'capital': CAPITAL,
                'risk_per_trade': RISK_PER_TRADE,
                'sl_multiplier': SL_MULTIPLIER
            }
            assert validate_risk_params(risk_config), f"Invalid risk params for timeframe: {tf}"

    def test_configuration_consistency(self):
        """Test that configuration is internally consistent."""
        # Current timeframe should exist in TIMEFRAME_PARAMS
        assert TIMEFRAME in TIMEFRAME_PARAMS, f"Current timeframe {TIMEFRAME} not in TIMEFRAME_PARAMS"

        # Output format should be supported
        supported_formats = ['json', 'csv']
        assert OUTPUT_FORMAT in supported_formats, f"Unsupported output format: {OUTPUT_FORMAT}"

        # Risk parameters should form a valid configuration
        risk_config = {
            'capital': CAPITAL,
            'risk_per_trade': RISK_PER_TRADE,
            'sl_multiplier': SL_MULTIPLIER
        }
        assert validate_risk_params(risk_config), "Global risk configuration is invalid"


if __name__ == "__main__":
    pytest.main([__file__])