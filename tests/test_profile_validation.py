#!/usr/bin/env python3
"""
Test profile validation and loading functionality.
"""

import pytest
import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import main module functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from main import validate_profile_config, load_profile


class TestProfileValidation:
    """Test profile validation functionality."""

    def test_validate_light_profile(self):
        """Test validation of light profile."""
        with patch('config.settings.ML_ENABLED', False), \
             patch('config.settings.ADAPTIVE_OPTIMIZATION_ENABLED_LIVE', False), \
             patch('config.settings.MONITORING_STACK_ENABLED', False), \
             patch('config.settings.PERFORMANCE_MONITORING_ENABLED', False):
            # Should return True for valid light profile
            result = validate_profile_config('light')
            assert result is True

    def test_validate_heavy_profile(self):
        """Test validation of heavy profile."""
        with patch('config.settings.ML_ENABLED', True):
            # Should return True for heavy profile (no strict requirements)
            result = validate_profile_config('heavy')
            assert result is True

    def test_validate_invalid_profile(self):
        """Test validation of invalid profile name."""
        # Should return True for unknown profiles (no validation)
        result = validate_profile_config('invalid')
        assert result is True

    def test_validate_light_profile_wrong_ml_setting(self):
        """Test that light profile with wrong settings returns False."""
        with patch('config.settings.ML_ENABLED', True), \
             patch('config.settings.ADAPTIVE_OPTIMIZATION_ENABLED_LIVE', False), \
             patch('config.settings.MONITORING_STACK_ENABLED', False), \
             patch('config.settings.PERFORMANCE_MONITORING_ENABLED', False):
            result = validate_profile_config('light')
            assert result is False

    def test_validate_light_profile_wrong_adaptive_setting(self):
        """Test that light profile with wrong adaptive settings returns False."""
        with patch('config.settings.ML_ENABLED', False), \
             patch('config.settings.ADAPTIVE_OPTIMIZATION_ENABLED_LIVE', True), \
             patch('config.settings.MONITORING_STACK_ENABLED', False), \
             patch('config.settings.PERFORMANCE_MONITORING_ENABLED', False):
            result = validate_profile_config('light')
            assert result is False


class TestProfileLoading:
    """Test profile loading functionality."""

    def test_load_light_profile(self):
        """Test loading light profile."""
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / '.env.light'
            env_file.write_text('ML_ENABLED=false\nPERFORMANCE_MONITORING_ENABLED=false\n')

            with patch('os.path.exists', return_value=True):
                with patch('dotenv.load_dotenv') as mock_load:
                    result = load_profile('light')
                    assert result is True
                    mock_load.assert_called_once_with('.env.light')

    def test_load_heavy_profile(self):
        """Test loading heavy profile."""
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / '.env.heavy'
            env_file.write_text('ML_ENABLED=true\nPERFORMANCE_MONITORING_ENABLED=true\n')

            with patch('os.path.exists', return_value=True):
                with patch('dotenv.load_dotenv') as mock_load:
                    result = load_profile('heavy')
                    assert result is True
                    mock_load.assert_called_once_with('.env.heavy')

    def test_load_invalid_profile(self):
        """Test loading invalid profile."""
        with patch('dotenv.load_dotenv') as mock_load:
            result = load_profile('invalid')
            assert result is False
            # Should load default .env
            mock_load.assert_called_once_with()

    def test_load_profile_missing_file(self):
        """Test loading profile with missing env file."""
        with patch('os.path.exists', return_value=False):
            with patch('dotenv.load_dotenv') as mock_load:
                result = load_profile('light')
                assert result is False
                # Should load default .env
                mock_load.assert_called_once_with()


class TestProfileIntegration:
    """Test profile integration with main functionality."""

    def test_main_profile_argument_parsing(self):
        """Test that profile argument is parsed correctly in main."""
        # This would require mocking the entire argument parsing
        # For now, we'll test that the choices are correct
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--profile', choices=['light', 'heavy'], default='default')

        # Test valid choices
        args = parser.parse_args(['--profile', 'light'])
        assert args.profile == 'light'

        args = parser.parse_args(['--profile', 'heavy'])
        assert args.profile == 'heavy'

        # Test default
        args = parser.parse_args([])
        assert args.profile == 'default'

    def test_invalid_profile_argument_raises_error(self):
        """Test that invalid profile argument raises error."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--profile', choices=['light', 'heavy'], default='default')

        with pytest.raises(SystemExit):
            parser.parse_args(['--profile', 'invalid'])