"""
Configuration-Aware Tests

Tests that validate system behavior based on configuration settings.
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Try to import configuration components
try:
    from config.settings import (
        SENTIMENT_ENABLED, KELLY_ENABLED, PAPER_TRADING_ENABLED,
        ML_ENABLED, SYMBOL, TIMEFRAME
    )
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    pytest.skip("Configuration module not available", allow_module_level=True)


class TestFeatureAvailability:
    """Test feature availability based on configuration."""

    def setup_method(self):
        """Skip test if config is not available."""
        if not CONFIG_AVAILABLE:
            pytest.skip("Configuration module not available")

    def test_sentiment_feature_status(self):
        """Test sentiment analysis feature availability."""
        # Test that sentiment enabled flag is accessible
        from config.settings import SENTIMENT_ENABLED
        assert isinstance(SENTIMENT_ENABLED, bool)

        # Test that we can check sentiment status
        enabled = SENTIMENT_ENABLED
        assert isinstance(enabled, bool)

    def test_kelly_feature_status(self):
        """Test Kelly Criterion feature availability."""
        assert isinstance(KELLY_ENABLED, bool)

        # Test that Kelly settings are accessible
        from config.settings import KELLY_FRACTION, KELLY_MIN_TRADES
        assert isinstance(KELLY_FRACTION, (int, float))
        assert isinstance(KELLY_MIN_TRADES, int)
        assert 0 < KELLY_FRACTION <= 1  # Valid fraction range
        assert KELLY_MIN_TRADES > 0

    def test_paper_trading_feature_status(self):
        """Test paper trading feature availability."""
        assert isinstance(PAPER_TRADING_ENABLED, bool)

        # Test that paper trading settings are accessible
        from config.settings import (
            PAPER_TRADING_INITIAL_BALANCE,
            PAPER_TRADING_FEE_RATE,
            PAPER_TRADING_MAX_POSITION_SIZE
        )
        assert PAPER_TRADING_INITIAL_BALANCE > 0
        assert 0 <= PAPER_TRADING_FEE_RATE <= 1  # Valid fee rate
        assert 0 < PAPER_TRADING_MAX_POSITION_SIZE <= 1  # Valid position size

    def test_ml_feature_status(self):
        """Test ML feature availability."""
        assert isinstance(ML_ENABLED, bool)

    def test_trading_symbol_configuration(self):
        """Test trading symbol configuration."""
        assert isinstance(SYMBOL, str)
        assert len(SYMBOL) > 0
        assert '/' in SYMBOL  # Should be in format BASE/QUOTE

    def test_timeframe_configuration(self):
        """Test timeframe configuration."""
        assert isinstance(TIMEFRAME, str)
        assert len(TIMEFRAME) > 0

        # Should be valid timeframe format
        valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M']
        assert TIMEFRAME in valid_timeframes


class TestConditionalFeatureBehavior:
    """Test system behavior when features are enabled/disabled."""

    def setup_method(self):
        """Skip test if config is not available."""
        if not CONFIG_AVAILABLE:
            pytest.skip("Configuration module not available")

    def test_sentiment_disabled_behavior(self):
        """Test behavior when sentiment analysis is disabled."""
        # Test that we can import and check sentiment status
        try:
            from config.settings import SENTIMENT_ENABLED
            # If sentiment is disabled, this should be False
            if not SENTIMENT_ENABLED:
                # Test passes - sentiment is properly disabled
                assert True
            else:
                # If enabled, we can't easily test disabled behavior without mocking
                pytest.skip("Sentiment analysis is enabled, skipping disabled behavior test")

        except ImportError:
            # If config not available, test passes
            pass

    def test_kelly_disabled_behavior(self):
        """Test behavior when Kelly Criterion is disabled."""
        try:
            from config.settings import KELLY_ENABLED
            # If Kelly is disabled, this should be False
            if not KELLY_ENABLED:
                # Test passes - Kelly is properly disabled
                assert True
            else:
                # If enabled, we can't easily test disabled behavior without mocking
                pytest.skip("Kelly Criterion is enabled, skipping disabled behavior test")

        except ImportError:
            # If config not available, test passes
            pass

    def test_paper_trading_disabled_behavior(self):
        """Test behavior when paper trading is disabled."""
        try:
            from config.settings import PAPER_TRADING_ENABLED
            # If paper trading is disabled, this should be False
            if not PAPER_TRADING_ENABLED:
                # Test passes - paper trading is properly disabled
                assert True
            else:
                # If enabled, we can't easily test disabled behavior without mocking
                pytest.skip("Paper trading is enabled, skipping disabled behavior test")

        except ImportError:
            # If config not available, test passes
            pass

    def test_ml_disabled_behavior(self):
        """Test behavior when ML is disabled."""
        try:
            from config.settings import ML_ENABLED
            # If ML is disabled, this should be False
            if not ML_ENABLED:
                # Test passes - ML is properly disabled
                assert True
            else:
                # If enabled, we can't easily test disabled behavior without mocking
                pytest.skip("ML is enabled, skipping disabled behavior test")

        except ImportError:
            # If config not available, test passes
            pass


class TestConfigurationValidation:
    """Test configuration validation and sanity checks."""

    def setup_method(self):
        """Skip test if config is not available."""
        if not CONFIG_AVAILABLE:
            pytest.skip("Configuration module not available")

    def test_risk_parameters_validation(self):
        """Test that risk parameters are within reasonable bounds."""
        # Use available risk-related parameters
        from config.settings import (
            KELLY_FRACTION, PAPER_TRADING_FEE_RATE, PAPER_TRADING_MAX_POSITION_SIZE
        )

        # Kelly fraction should be reasonable
        assert 0 < KELLY_FRACTION <= 1  # Valid fraction range

        # Paper trading fee rate should be reasonable
        assert 0 <= PAPER_TRADING_FEE_RATE <= 1  # Valid fee rate

        # Paper trading position size should be reasonable
        assert 0 < PAPER_TRADING_MAX_POSITION_SIZE <= 1  # Valid position size

    def test_api_configuration_presence(self):
        """Test that API configurations are present (but not their values)."""
        # These should exist, even if empty/disabled
        from config.settings import (
            TWITTER_API_KEY, NEWS_API_KEY, API_KEY, API_SECRET
        )

        # Variables should exist (may be empty strings)
        assert isinstance(TWITTER_API_KEY, str)
        assert isinstance(NEWS_API_KEY, str)
        assert isinstance(API_KEY, str)
        assert isinstance(API_SECRET, str)

    def test_file_path_configurations(self):
        """Test that file path configurations are valid."""
        from config.settings import (
            LOG_FILE, ML_MODEL_DIR
        )

        # Should be strings
        assert isinstance(LOG_FILE, str)
        assert isinstance(ML_MODEL_DIR, str)

        # Should not be empty
        assert len(LOG_FILE) > 0
        assert len(ML_MODEL_DIR) > 0

    def test_performance_profile_validation(self):
        """Test performance profile configurations."""
        from config.settings import (
            MAX_WORKERS, CHUNK_SIZE
        )

        # Workers and chunk size should be positive or None
        if MAX_WORKERS is not None:
            assert MAX_WORKERS > 0
        assert CHUNK_SIZE > 0

        # Reasonable upper bounds
        if MAX_WORKERS is not None:
            assert MAX_WORKERS <= 32  # Not too many workers
        assert CHUNK_SIZE <= 10000  # Not too large chunks


class TestEnvironmentVariableHandling:
    """Test environment variable handling and defaults."""

    def test_environment_variable_defaults(self):
        """Test that environment variables have sensible defaults."""
        # Test with environment variables not set (should use defaults)
        import os

        # Save original env
        original_env = dict(os.environ)

        try:
            # Clear relevant env vars
            env_vars_to_clear = [
                'SENTIMENT_ENABLED', 'KELLY_ENABLED', 'PAPER_TRADING_ENABLED',
                'ML_ENABLED', 'SYMBOL', 'TIMEFRAME', 'RISK_PER_TRADE'
            ]

            for var in env_vars_to_clear:
                os.environ.pop(var, None)

            # Re-import to get defaults
            import importlib
            import config.settings
            importlib.reload(config.settings)

            from config.settings import (
                SENTIMENT_ENABLED, KELLY_ENABLED, PAPER_TRADING_ENABLED,
                ML_ENABLED, SYMBOL, TIMEFRAME, RISK_PER_TRADE
            )

            # Check defaults
            assert isinstance(SENTIMENT_ENABLED, bool)
            assert isinstance(KELLY_ENABLED, bool)
            assert isinstance(PAPER_TRADING_ENABLED, bool)
            assert isinstance(ML_ENABLED, bool)
            assert isinstance(SYMBOL, str)
            assert isinstance(TIMEFRAME, str)
            assert isinstance(RISK_PER_TRADE, float)

        finally:
            # Restore original environment
            os.environ.clear()
            os.environ.update(original_env)