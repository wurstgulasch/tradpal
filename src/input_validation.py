"""
Input validation and sanitization utilities for the trading indicator system.
Provides comprehensive validation for all user inputs and configuration parameters.
"""

import re
import pandas as pd
from typing import Dict, List, Any, Union, Optional, Tuple
from datetime import datetime, date
import numbers


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class InputValidator:
    """Comprehensive input validation and sanitization."""

    # Valid trading symbols (forex, crypto, stocks)
    VALID_SYMBOL_PATTERN = re.compile(r'^[A-Z]{3,10}(/[A-Z]{3,10})?$|^[A-Z]{1,5}\d*$')

    # Valid timeframes
    VALID_TIMEFRAMES = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']

    # Valid exchanges
    VALID_EXCHANGES = ['kraken', 'binance', 'coinbase', 'bitfinex', 'bitstamp', 'gemini', 'okx', 'huobi']

    @staticmethod
    def validate_symbol(symbol: str) -> str:
        """
        Validate and sanitize trading symbol.

        Args:
            symbol: Trading symbol (e.g., 'EUR/USD', 'BTC/USDT', 'AAPL')

        Returns:
            Sanitized symbol

        Raises:
            ValidationError: If symbol is invalid
        """
        if not isinstance(symbol, str):
            raise ValidationError("Symbol must be a string")

        symbol = symbol.strip().upper()

        if not symbol:
            raise ValidationError("Symbol cannot be empty")

        if len(symbol) > 20:
            raise ValidationError("Symbol too long (max 20 characters)")

        # Check against pattern
        if not InputValidator.VALID_SYMBOL_PATTERN.match(symbol):
            raise ValidationError(f"Invalid symbol format: {symbol}")

        return symbol

    @staticmethod
    def validate_timeframe(timeframe: str) -> str:
        """
        Validate timeframe.

        Args:
            timeframe: Timeframe string (e.g., '1m', '1h', '1d')

        Returns:
            Validated timeframe

        Raises:
            ValidationError: If timeframe is invalid
        """
        if not isinstance(timeframe, str):
            raise ValidationError("Timeframe must be a string")

        timeframe = timeframe.strip().lower()

        if timeframe not in InputValidator.VALID_TIMEFRAMES:
            raise ValidationError(f"Invalid timeframe: {timeframe}. Valid options: {InputValidator.VALID_TIMEFRAMES}")

        return timeframe

    @staticmethod
    def validate_exchange(exchange: str) -> str:
        """
        Validate exchange name.

        Args:
            exchange: Exchange name

        Returns:
            Validated exchange name

        Raises:
            ValidationError: If exchange is invalid
        """
        if not isinstance(exchange, str):
            raise ValidationError("Exchange must be a string")

        exchange = exchange.strip().lower()

        if exchange not in InputValidator.VALID_EXCHANGES:
            raise ValidationError(f"Invalid exchange: {exchange}. Valid options: {InputValidator.VALID_EXCHANGES}")

        return exchange

    @staticmethod
    def validate_positive_number(value: Union[int, float], field_name: str = "value",
                               min_value: Optional[float] = None, max_value: Optional[float] = None) -> float:
        """
        Validate positive numeric value.

        Args:
            value: Numeric value to validate
            field_name: Name of the field for error messages
            min_value: Minimum allowed value (optional)
            max_value: Maximum allowed value (optional)

        Returns:
            Validated float value

        Raises:
            ValidationError: If validation fails
        """
        try:
            num_value = float(value)
        except (ValueError, TypeError):
            raise ValidationError(f"{field_name} must be a number")

        if not isinstance(value, numbers.Number) or not pd.notna(value):
            raise ValidationError(f"{field_name} must be a valid number")

        if min_value is not None and num_value < min_value:
            raise ValidationError(f"{field_name} must be >= {min_value}")

        if max_value is not None and num_value > max_value:
            raise ValidationError(f"{field_name} must be <= {max_value}")

        return num_value

    @staticmethod
    def validate_date(date_str: Union[str, date, datetime], field_name: str = "date") -> datetime:
        """
        Validate and parse date.

        Args:
            date_str: Date string or date object
            field_name: Name of the field for error messages

        Returns:
            datetime object

        Raises:
            ValidationError: If date is invalid
        """
        if isinstance(date_str, datetime):
            return date_str
        elif isinstance(date_str, date):
            return datetime.combine(date_str, datetime.min.time())

        if not isinstance(date_str, str):
            raise ValidationError(f"{field_name} must be a string or date object")

        try:
            # Try common date formats
            for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S']:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            raise ValidationError(f"Invalid date format: {date_str}")
        except Exception:
            raise ValidationError(f"Invalid date: {date_str}")

    @staticmethod
    def validate_date_range(start_date: Union[str, date, datetime],
                          end_date: Union[str, date, datetime]) -> Tuple[datetime, datetime]:
        """
        Validate date range.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            Tuple of (start_datetime, end_datetime)

        Raises:
            ValidationError: If date range is invalid
        """
        start_dt = InputValidator.validate_date(start_date, "start_date")
        end_dt = InputValidator.validate_date(end_date, "end_date")

        if start_dt >= end_dt:
            raise ValidationError("start_date must be before end_date")

        # Check reasonable date range (not too far in future/past)
        now = datetime.now()
        if start_dt < now - pd.Timedelta(days=365*10):  # 10 years ago
            raise ValidationError("start_date cannot be more than 10 years ago")

        if end_dt > now + pd.Timedelta(days=365):  # 1 year in future
            raise ValidationError("end_date cannot be more than 1 year in the future")

        return start_dt, end_dt

    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None) -> pd.DataFrame:
        """
        Validate DataFrame structure.

        Args:
            df: DataFrame to validate
            required_columns: List of required column names

        Returns:
            Validated DataFrame

        Raises:
            ValidationError: If DataFrame is invalid
        """
        if not isinstance(df, pd.DataFrame):
            raise ValidationError("Input must be a pandas DataFrame")

        if df.empty:
            raise ValidationError("DataFrame cannot be empty")

        if required_columns:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValidationError(f"Missing required columns: {missing_cols}")

        # Check for completely NaN columns
        nan_cols = df.columns[df.isna().all()].tolist()
        if nan_cols:
            raise ValidationError(f"Columns are completely NaN: {nan_cols}")

        return df

    @staticmethod
    def sanitize_string(value: str, max_length: int = 100) -> str:
        """
        Sanitize string input.

        Args:
            value: String to sanitize
            max_length: Maximum allowed length

        Returns:
            Sanitized string
        """
        if not isinstance(value, str):
            value = str(value)

        # Remove leading/trailing whitespace
        value = value.strip()

        # Truncate if too long
        if len(value) > max_length:
            value = value[:max_length]

        # Remove potentially dangerous characters
        value = re.sub(r'[<>]', '', value)

        return value

    @staticmethod
    def validate_config_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration parameters.

        Args:
            params: Configuration dictionary

        Returns:
            Validated parameters

        Raises:
            ValidationError: If parameters are invalid
        """
        validated = {}

        # Validate common parameters
        if 'symbol' in params:
            validated['symbol'] = InputValidator.validate_symbol(params['symbol'])

        if 'timeframe' in params:
            validated['timeframe'] = InputValidator.validate_timeframe(params['timeframe'])

        if 'exchange' in params:
            validated['exchange'] = InputValidator.validate_exchange(params['exchange'])

        if 'capital' in params:
            validated['capital'] = InputValidator.validate_positive_number(
                params['capital'], 'capital', min_value=1, max_value=10000000
            )

        if 'risk_per_trade' in params:
            validated['risk_per_trade'] = InputValidator.validate_positive_number(
                params['risk_per_trade'], 'risk_per_trade', min_value=0.001, max_value=1.0
            )

        # Add other parameters without validation
        for key, value in params.items():
            if key not in validated:
                if isinstance(value, str):
                    validated[key] = InputValidator.sanitize_string(value)
                else:
                    validated[key] = value

        return validated


def validate_api_inputs(**kwargs) -> Dict[str, Any]:
    """
    Convenience function to validate common API inputs.

    Args:
        **kwargs: Input parameters

    Returns:
        Validated parameters

    Raises:
        ValidationError: If validation fails
    """
    validator = InputValidator()
    validated = {}

    for key, value in kwargs.items():
        if key == 'symbol':
            validated[key] = validator.validate_symbol(value)
        elif key == 'timeframe':
            validated[key] = validator.validate_timeframe(value)
        elif key == 'exchange':
            validated[key] = validator.validate_exchange(value)
        elif key in ['limit', 'capital']:
            validated[key] = validator.validate_positive_number(value, key, min_value=1)
        elif key == 'start_date':
            if 'end_date' in kwargs:
                start_dt, end_dt = validator.validate_date_range(value, kwargs['end_date'])
                validated['start_date'] = start_dt
                validated['end_date'] = end_dt
                break  # Skip end_date processing
            else:
                validated[key] = validator.validate_date(value, key)
        elif key == 'end_date':
            if 'start_date' not in validated:
                validated[key] = validator.validate_date(value, key)
        else:
            # For unknown parameters, just sanitize if string
            if isinstance(value, str):
                validated[key] = validator.sanitize_string(value)
            else:
                validated[key] = value

    return validated