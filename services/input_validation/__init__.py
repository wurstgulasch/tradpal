"""
Input Validation Service - Comprehensive input validation for trading system.

Provides validation for symbols, timeframes, exchanges, dates, and configuration parameters.
"""

import re
import pandas as pd
from datetime import datetime, date
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


@dataclass
class ValidationResult:
    """Result of validation operations."""
    is_valid: bool
    value: Any
    error_message: Optional[str] = None


class InputValidator:
    """Comprehensive input validator for trading system parameters."""

    # Valid exchanges
    VALID_EXCHANGES = [
        'kraken', 'binance', 'coinbase', 'bitfinex', 'bitstamp',
        'gemini', 'okx', 'huobi', 'bybit', 'kucoin'
    ]

    # Valid timeframes
    VALID_TIMEFRAMES = [
        '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h',
        '1d', '3d', '1w', '1M'
    ]

    # Maximum symbol length
    MAX_SYMBOL_LENGTH = 20

    # Date limits
    MAX_YEARS_PAST = 10
    MAX_YEARS_FUTURE = 1

    @classmethod
    def validate_symbol(cls, symbol: Union[str, Any]) -> str:
        """Validate and normalize trading symbol."""
        if not isinstance(symbol, str):
            raise ValidationError("Symbol must be a string")

        symbol = symbol.strip().upper()

        if not symbol:
            raise ValidationError("Symbol cannot be empty")

        if len(symbol) > cls.MAX_SYMBOL_LENGTH:
            raise ValidationError(f"Symbol too long (max {cls.MAX_SYMBOL_LENGTH} characters)")

        # Basic format validation for common symbol types
        # Forex: EUR/USD, Crypto: BTC/USDT, Stocks: AAPL (must contain at least one letter)
        if not re.match(r'^[A-Z]+[A-Z0-9]*(/[A-Z]+[A-Z0-9]*)?$', symbol):
            raise ValidationError("Invalid symbol format")

        return symbol

    @classmethod
    def validate_exchange(cls, exchange: Union[str, Any]) -> str:
        """Validate exchange name."""
        if not isinstance(exchange, str):
            raise ValidationError("Exchange must be a string")

        exchange = exchange.strip().lower()

        if exchange not in cls.VALID_EXCHANGES:
            raise ValidationError(f"Invalid exchange: {exchange}. Valid exchanges: {', '.join(cls.VALID_EXCHANGES)}")

        return exchange

    @classmethod
    def validate_timeframe(cls, timeframe: Union[str, Any]) -> str:
        """Validate timeframe."""
        if not isinstance(timeframe, str):
            raise ValidationError("Timeframe must be a string")

        timeframe = timeframe.strip().lower()

        if timeframe not in cls.VALID_TIMEFRAMES:
            raise ValidationError(f"Invalid timeframe: {timeframe}. Valid timeframes: {', '.join(cls.VALID_TIMEFRAMES)}")

        return timeframe

    @classmethod
    def validate_positive_number(cls, value: Union[int, float, str, Any],
                               min_value: Optional[float] = None,
                               max_value: Optional[float] = None) -> float:
        """Validate positive number with optional range."""
        try:
            if isinstance(value, str):
                # Handle string numbers
                if value.strip() == '':
                    raise ValueError("Empty string")
                num_value = float(value)
            elif isinstance(value, (int, float)):
                num_value = float(value)
            else:
                raise ValidationError("value must be a number or numeric string")

            # Check for NaN
            if pd.isna(num_value):
                raise ValidationError("value must be a valid number")

            if min_value is not None and num_value < min_value:
                raise ValidationError(f"value must be >= {min_value}")

            if max_value is not None and num_value > max_value:
                raise ValidationError(f"value must be <= {max_value}")

            return num_value

        except (ValueError, TypeError):
            raise ValidationError("value must be a number")

    @classmethod
    def validate_date(cls, date_input: Union[str, datetime, date, Any]) -> datetime:
        """Validate and parse date input."""
        if isinstance(date_input, datetime):
            return date_input
        elif isinstance(date_input, date):
            return datetime.combine(date_input, datetime.min.time())

        if not isinstance(date_input, str):
            raise ValidationError("Date must be a string or date object")

        # Try different date formats
        date_formats = [
            '%Y-%m-%d',
            '%Y/%m/%d',
            '%m-%d-%Y',
            '%m/%d/%Y',
            '%Y-%m-%d %H:%M:%S',
            '%Y/%m/%d %H:%M:%S'
        ]

        for fmt in date_formats:
            try:
                return datetime.strptime(date_input.strip(), fmt)
            except ValueError:
                continue

        raise ValidationError(f"Invalid date: {date_input}")

    @classmethod
    def validate_date_range(cls, start_date: Union[str, datetime, date, Any],
                           end_date: Union[str, datetime, date, Any]) -> tuple[datetime, datetime]:
        """Validate date range."""
        start_dt = cls.validate_date(start_date)
        end_dt = cls.validate_date(end_date)

        if start_dt >= end_dt:
            raise ValidationError("start_date must be before end_date")

        # Check date limits
        now = datetime.now()
        max_past = now.replace(year=now.year - cls.MAX_YEARS_PAST)
        max_future = now.replace(year=now.year + cls.MAX_YEARS_FUTURE)

        if start_dt < max_past:
            raise ValidationError(f"start_date cannot be more than {cls.MAX_YEARS_PAST} years ago")

        if end_dt > max_future:
            raise ValidationError(f"end_date cannot be more than {cls.MAX_YEARS_FUTURE} year in the future")

        return start_dt, end_dt

    @classmethod
    def validate_dataframe(cls, df: Any, required_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Validate pandas DataFrame."""
        if not isinstance(df, pd.DataFrame):
            raise ValidationError("Input must be a pandas DataFrame")

        if df.empty:
            raise ValidationError("DataFrame cannot be empty")

        if required_columns:
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                raise ValidationError(f"Missing required columns: {', '.join(missing_columns)}")

            # Check for completely NaN columns
            for col in required_columns:
                if df[col].isna().all():
                    raise ValidationError(f"Column '{col}' is completely NaN")
        else:
            # If no required columns specified, check all columns for completely NaN
            for col in df.columns:
                if df[col].isna().all():
                    raise ValidationError(f"Column '{col}' is completely NaN")

        return df

    @classmethod
    def sanitize_string(cls, input_str: Any, max_length: Optional[int] = None) -> str:
        """Sanitize string input."""
        if input_str is None:
            return "None"

        string_val = str(input_str).strip()

        # Remove potentially dangerous characters (basic XSS prevention)
        string_val = re.sub(r'[<>]', '', string_val)

        if max_length and len(string_val) > max_length:
            string_val = string_val[:max_length]

        return string_val

    @classmethod
    def validate_config_params(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration parameters."""
        validated = {}

        for key, value in params.items():
            if key == 'symbol' and value is not None:
                validated[key] = cls.validate_symbol(value)
            elif key == 'exchange' and value is not None:
                validated[key] = cls.validate_exchange(value)
            elif key == 'timeframe' and value is not None:
                validated[key] = cls.validate_timeframe(value)
            elif key in ['capital', 'risk_per_trade', 'limit'] and value is not None:
                validated[key] = cls.validate_positive_number(value, min_value=0)
            elif key in ['start_date', 'end_date'] and value is not None:
                validated[key] = cls.validate_date(value)
            elif isinstance(value, str):
                validated[key] = cls.sanitize_string(value)
            else:
                validated[key] = value

        return validated


def validate_api_inputs(**kwargs) -> Dict[str, Any]:
    """Convenience function for API input validation."""
    # Validate known parameters
    validated = {}

    for key, value in kwargs.items():
        try:
            if key == 'symbol':
                validated[key] = InputValidator.validate_symbol(value)
            elif key == 'exchange':
                validated[key] = InputValidator.validate_exchange(value)
            elif key == 'timeframe':
                validated[key] = InputValidator.validate_timeframe(value)
            elif key in ['start_date', 'end_date']:
                validated[key] = InputValidator.validate_date(value)
            elif key in ['capital', 'risk_per_trade', 'limit']:
                validated[key] = InputValidator.validate_positive_number(value, min_value=0)
            else:
                # For unknown parameters, just sanitize if string
                if isinstance(value, str):
                    validated[key] = InputValidator.sanitize_string(value)
                else:
                    validated[key] = value
        except ValidationError:
            raise  # Re-raise validation errors

    return validated