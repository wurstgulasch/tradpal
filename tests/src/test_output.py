import pytest
import pandas as pd
import numpy as np
import json
import tempfile
import os
from unittest.mock import patch, mock_open
from datetime import datetime
import sys

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.output import save_signals_to_json, load_signals_from_json, format_signal_data


class TestOutputModule:
    """Test output functionality."""

    def test_format_signal_data_basic(self):
        """Test basic signal data formatting."""
        # Create test data
        data = pd.DataFrame({
            'timestamp': pd.to_datetime(['2023-01-01 10:00:00', '2023-01-01 10:01:00']),
            'open': [1.0, 1.01],
            'high': [1.05, 1.06],
            'low': [0.95, 0.96],
            'close': [1.02, 1.03],
            'volume': [1000, 1100],
            'EMA9': [1.01, 1.02],
            'EMA21': [1.00, 1.01],
            'RSI': [60.0, 65.0],
            'BB_upper': [1.08, 1.09],
            'BB_middle': [1.02, 1.03],
            'BB_lower': [0.96, 0.97],
            'ATR': [0.02, 0.021],
            'Buy_Signal': [0, 1],
            'Sell_Signal': [0, 0]
        })

        result = format_signal_data(data)

        assert isinstance(result, list)
        assert len(result) == 2

        # Check first signal
        signal = result[0]
        assert 'timestamp' in signal
        assert 'open' in signal
        assert 'close' in signal
        assert 'Buy_Signal' in signal
        assert 'Sell_Signal' in signal
        assert signal['Buy_Signal'] == 0
        assert signal['Sell_Signal'] == 0

        # Check second signal (with buy signal)
        signal = result[1]
        assert signal['Buy_Signal'] == 1
        assert signal['Sell_Signal'] == 0

    def test_format_signal_data_with_risk_management(self):
        """Test signal formatting with risk management data."""
        data = pd.DataFrame({
            'timestamp': pd.to_datetime(['2023-01-01 10:00:00']),
            'open': [1.0],
            'high': [1.05],
            'low': [0.95],
            'close': [1.02],
            'volume': [1000],
            'EMA9': [1.01],
            'EMA21': [1.00],
            'RSI': [60.0],
            'BB_upper': [1.08],
            'BB_middle': [1.02],
            'BB_lower': [0.96],
            'ATR': [0.02],
            'Buy_Signal': [1],
            'Sell_Signal': [0],
            'Position_Size_Absolute': [1000.0],
            'Position_Size_Percent': [1.0],
            'Stop_Loss_Buy': [1.00],
            'Take_Profit_Buy': [1.04],
            'Leverage': [5.0]
        })

        result = format_signal_data(data)

        assert len(result) == 1
        signal = result[0]

        # Check risk management fields are included
        assert 'risk_management' in signal
        risk = signal['risk_management']
        assert risk['position_size'] == 1000.0
        assert risk['stop_loss'] == 1.00
        assert risk['take_profit'] == 1.04
        assert risk['leverage'] == 5.0

    def test_format_signal_data_empty_dataframe(self):
        """Test formatting with empty DataFrame."""
        data = pd.DataFrame()

        result = format_signal_data(data)
        assert result == []

    def test_format_signal_data_missing_columns(self):
        """Test formatting with missing required columns."""
        data = pd.DataFrame({
            'timestamp': pd.to_datetime(['2023-01-01']),
            'close': [1.0]
            # Missing other required columns
        })

        # Should still work but with limited data
        result = format_signal_data(data)
        assert len(result) == 1
        assert result[0]['close'] == 1.0

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_save_signals_to_json_success(self, mock_json_dump, mock_file):
        """Test successful JSON saving."""
        test_data = [
            {
                'timestamp': '2023-01-01T10:00:00',
                'close': 1.0,
                'Buy_Signal': 1,
                'Sell_Signal': 0
            }
        ]

        save_signals_to_json(test_data, 'test_signals.json')

        # Verify file was opened for writing
        mock_file.assert_called_once_with('test_signals.json', 'w', encoding='utf-8')

        # Verify json.dump was called
        mock_json_dump.assert_called_once()
        args, kwargs = mock_json_dump.call_args
        assert args[0] == test_data
        assert kwargs['indent'] == 4

    @patch('builtins.open', side_effect=PermissionError("Permission denied"))
    def test_save_signals_to_json_permission_error(self, mock_file):
        """Test handling of permission errors when saving."""
        test_data = [{'test': 'data'}]

        with pytest.raises(PermissionError):
            save_signals_to_json(test_data, '/root/test.json')

    @patch('builtins.open', side_effect=OSError("Disk full"))
    def test_save_signals_to_json_disk_error(self, mock_file):
        """Test handling of disk space errors."""
        test_data = [{'test': 'data'}]

        with pytest.raises(OSError):
            save_signals_to_json(test_data, 'test.json')

    @patch('builtins.open', new_callable=mock_open, read_data='[{"test": "data"}]')
    @patch('json.load')
    def test_load_signals_from_json_success(self, mock_json_load, mock_file):
        """Test successful JSON loading."""
        mock_json_load.return_value = [{"test": "data"}]

        result = load_signals_from_json('test_signals.json')

        assert result == [{"test": "data"}]
        mock_file.assert_called_once_with('test_signals.json', 'r', encoding='utf-8')
        mock_json_load.assert_called_once()

    @patch('builtins.open', side_effect=FileNotFoundError("File not found"))
    def test_load_signals_from_json_file_not_found(self, mock_file):
        """Test handling of missing files."""
        with pytest.raises(FileNotFoundError):
            load_signals_from_json('nonexistent.json')

    @patch('builtins.open', new_callable=mock_open, read_data='invalid json')
    @patch('json.load', side_effect=json.JSONDecodeError("Invalid JSON", "invalid json", 0))
    def test_load_signals_from_json_invalid_json(self, mock_json_load, mock_file):
        """Test handling of invalid JSON files."""
        with pytest.raises(json.JSONDecodeError):
            load_signals_from_json('invalid.json')

    def test_format_signal_data_datetime_serialization(self):
        """Test that datetime objects are properly serialized."""
        timestamp = datetime(2023, 1, 1, 10, 0, 0)
        data = pd.DataFrame({
            'timestamp': [timestamp],
            'open': [1.0],
            'high': [1.05],
            'low': [0.95],
            'close': [1.02],
            'volume': [1000],
            'EMA9': [1.01],
            'EMA21': [1.00],
            'RSI': [60.0],
            'BB_upper': [1.08],
            'BB_middle': [1.02],
            'BB_lower': [0.96],
            'ATR': [0.02],
            'Buy_Signal': [1],
            'Sell_Signal': [0]
        })

        result = format_signal_data(data)

        assert len(result) == 1
        # Should be serialized to ISO format
        assert '2023-01-01T10:00:00' in result[0]['timestamp']

    def test_format_signal_data_nan_values(self):
        """Test handling of NaN values in data."""
        data = pd.DataFrame({
            'timestamp': pd.to_datetime(['2023-01-01']),
            'open': [1.0],
            'high': [1.05],
            'low': [0.95],
            'close': [1.02],
            'volume': [1000],
            'EMA9': [np.nan],  # NaN value
            'EMA21': [1.00],
            'RSI': [60.0],
            'BB_upper': [1.08],
            'BB_middle': [1.02],
            'BB_lower': [0.96],
            'ATR': [0.02],
            'Buy_Signal': [1],
            'Sell_Signal': [0]
        })

        result = format_signal_data(data)

        assert len(result) == 1
        # NaN should be converted to None/null in JSON
        assert result[0]['EMA9'] is None or pd.isna(result[0]['EMA9'])

    def test_format_signal_data_large_dataset(self):
        """Test formatting with large dataset."""
        # Create larger dataset
        n_rows = 1000
        timestamps = pd.date_range('2023-01-01', periods=n_rows, freq='1min')

        data = pd.DataFrame({
            'timestamp': timestamps,
            'open': np.random.randn(n_rows) + 100,
            'high': np.random.randn(n_rows) + 101,
            'low': np.random.randn(n_rows) + 99,
            'close': np.random.randn(n_rows) + 100,
            'volume': np.random.randint(1000, 10000, n_rows),
            'EMA9': np.random.randn(n_rows) + 100,
            'EMA21': np.random.randn(n_rows) + 100,
            'RSI': np.random.uniform(0, 100, n_rows),
            'BB_upper': np.random.randn(n_rows) + 102,
            'BB_middle': np.random.randn(n_rows) + 100,
            'BB_lower': np.random.randn(n_rows) + 98,
            'ATR': np.random.uniform(0.01, 0.05, n_rows),
            'Buy_Signal': np.random.choice([0, 1], n_rows),
            'Sell_Signal': np.random.choice([0, 1], n_rows)
        })

        result = format_signal_data(data)

        assert len(result) == n_rows
        assert all('timestamp' in signal for signal in result)
        assert all('Buy_Signal' in signal for signal in result)
        assert all('Sell_Signal' in signal for signal in result)


class TestOutputIntegration:
    """Integration tests for output functionality."""

    def test_save_and_load_roundtrip(self):
        """Test saving and loading creates identical data."""
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as f:
            temp_file = f.name

        try:
            # Create test data
            original_data = [
                {
                    'timestamp': '2023-01-01T10:00:00',
                    'close': 1.0,
                    'Buy_Signal': 1,
                    'Sell_Signal': 0,
                    'risk_management': {
                        'position_size': 1000,
                        'stop_loss': 0.98,
                        'take_profit': 1.05
                    }
                }
            ]

            # Save and load
            save_signals_to_json(original_data, temp_file)
            loaded_data = load_signals_from_json(temp_file)

            # Compare
            assert loaded_data == original_data

        finally:
            # Cleanup
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_output_file_permissions(self):
        """Test file permission handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, 'test.json')

            test_data = [{'test': 'data'}]
            save_signals_to_json(test_data, test_file)

            # Verify file was created
            assert os.path.exists(test_file)

            # Verify content
            loaded = load_signals_from_json(test_file)
            assert loaded == test_data


if __name__ == "__main__":
    pytest.main([__file__])