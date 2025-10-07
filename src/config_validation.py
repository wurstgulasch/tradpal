#!/usr/bin/env python3
"""
Configuration Validation Module

Validates configuration settings at startup to ensure system integrity.
Checks for required parameters, validates ranges, and provides warnings for
potentially problematic configurations.
"""

import os
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    SYMBOL, EXCHANGE, TIMEFRAME, CAPITAL, RISK_PER_TRADE,
    MAX_LEVERAGE, EMA_SHORT, EMA_LONG, RSI_PERIOD, RSI_OVERSOLD, RSI_OVERBOUGHT,
    BB_PERIOD, BB_STD_DEV, ATR_PERIOD, SL_MULTIPLIER,
    TP_MULTIPLIER, ADX_THRESHOLD,
    OUTPUT_FILE
)
from src.logging_config import trading_logger


class ValidationSeverity(Enum):
    """Severity levels for configuration validation."""
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class ValidationResult:
    """Result of a configuration validation check."""
    severity: ValidationSeverity
    message: str
    field: str
    value: Any
    suggestion: Optional[str] = None


class ConfigurationValidator:
    """
    Validates configuration settings at system startup.

    Performs comprehensive checks on all configuration parameters
    to ensure system stability and provide early warnings.
    """

    # Valid exchanges supported by ccxt
    VALID_EXCHANGES = [
        'kraken', 'binance', 'coinbase', 'bitfinex', 'bitstamp',
        'gemini', 'okx', 'huobi', 'bybit', 'kucoin'
    ]

    # Valid timeframes
    VALID_TIMEFRAMES = [
        '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h',
        '6h', '8h', '12h', '1d', '3d', '1w', '1M'
    ]

    def __init__(self):
        """Initialize the configuration validator."""
        self.results: List[ValidationResult] = []

    def validate_all(self) -> List[ValidationResult]:
        """
        Run all configuration validations.

        Returns:
            List of validation results
        """
        self.results = []

        # Core trading parameters
        self._validate_symbol()
        self._validate_exchange()
        self._validate_timeframe()
        self._validate_capital()
        self._validate_risk_parameters()

        # Technical indicators
        self._validate_ema_parameters()
        self._validate_rsi_parameters()
        self._validate_bb_parameters()
        self._validate_atr_parameters()
        self._validate_adx_parameters()

        # Advanced features
        self._validate_fibonacci_levels()
        self._validate_volatility_threshold()

        # Output and backtesting
        self._validate_output_path()
        self._validate_backtest_dates()

        # Cross-parameter validations
        self._validate_parameter_consistency()

        return self.results

    def _validate_symbol(self):
        """Validate trading symbol."""
        if not isinstance(SYMBOL, str) or not SYMBOL.strip():
            self.results.append(ValidationResult(
                ValidationSeverity.ERROR,
                "Symbol must be a non-empty string",
                "SYMBOL",
                SYMBOL,
                "Set SYMBOL to a valid trading pair like 'EUR/USD' or 'BTC/USDT'"
            ))
            return

        symbol = SYMBOL.strip().upper()
        if '/' not in symbol:
            self.results.append(ValidationResult(
                ValidationSeverity.WARNING,
                "Symbol should be in format 'BASE/QUOTE'",
                "SYMBOL",
                SYMBOL,
                "Use format like 'EUR/USD' or 'BTC/USDT'"
            ))

        # Check for common valid patterns
        parts = symbol.split('/')
        if len(parts) == 2:
            base, quote = parts
            if len(base) < 2 or len(quote) < 2:
                self.results.append(ValidationResult(
                    ValidationSeverity.WARNING,
                    "Base and quote currencies should be at least 2 characters",
                    "SYMBOL",
                    SYMBOL
                ))

    def _validate_exchange(self):
        """Validate exchange configuration."""
        if not isinstance(EXCHANGE, str) or not EXCHANGE.strip():
            self.results.append(ValidationResult(
                ValidationSeverity.ERROR,
                "Exchange must be a non-empty string",
                "EXCHANGE",
                EXCHANGE,
                f"Set EXCHANGE to one of: {', '.join(self.VALID_EXCHANGES)}"
            ))
            return

        exchange = EXCHANGE.strip().lower()
        if exchange not in self.VALID_EXCHANGES:
            self.results.append(ValidationResult(
                ValidationSeverity.ERROR,
                f"Unsupported exchange: {exchange}",
                "EXCHANGE",
                EXCHANGE,
                f"Valid exchanges: {', '.join(self.VALID_EXCHANGES)}"
            ))

    def _validate_timeframe(self):
        """Validate timeframe configuration."""
        if not isinstance(TIMEFRAME, str) or not TIMEFRAME.strip():
            self.results.append(ValidationResult(
                ValidationSeverity.ERROR,
                "Timeframe must be a non-empty string",
                "TIMEFRAME",
                TIMEFRAME,
                f"Set TIMEFRAME to one of: {', '.join(self.VALID_TIMEFRAMES)}"
            ))
            return

        timeframe = TIMEFRAME.strip().lower()
        if timeframe not in self.VALID_TIMEFRAMES:
            self.results.append(ValidationResult(
                ValidationSeverity.WARNING,
                f"Non-standard timeframe: {timeframe}",
                "TIMEFRAME",
                TIMEFRAME,
                f"Recommended timeframes: {', '.join(self.VALID_TIMEFRAMES[:5])}"
            ))

    def _validate_capital(self):
        """Validate capital amount."""
        if not isinstance(CAPITAL, (int, float)):
            self.results.append(ValidationResult(
                ValidationSeverity.ERROR,
                "Capital must be a number",
                "CAPITAL",
                CAPITAL,
                "Set CAPITAL to a positive number representing your trading capital"
            ))
            return

        if CAPITAL <= 0:
            self.results.append(ValidationResult(
                ValidationSeverity.ERROR,
                "Capital must be positive",
                "CAPITAL",
                CAPITAL,
                "Set CAPITAL to a positive number"
            ))
        elif CAPITAL < 100:
            self.results.append(ValidationResult(
                ValidationSeverity.WARNING,
                "Very low capital amount may limit trading opportunities",
                "CAPITAL",
                CAPITAL,
                "Consider increasing capital for better risk management"
            ))

    def _validate_risk_parameters(self):
        """Validate risk management parameters."""
        # Risk per trade
        if not isinstance(RISK_PER_TRADE, (int, float)):
            self.results.append(ValidationResult(
                ValidationSeverity.ERROR,
                "Risk per trade must be a number",
                "RISK_PER_TRADE",
                RISK_PER_TRADE
            ))
        elif not 0.001 <= RISK_PER_TRADE <= 0.1:
            self.results.append(ValidationResult(
                ValidationSeverity.WARNING,
                f"Risk per trade {RISK_PER_TRADE:.3f} is outside recommended range 0.1%-10%",
                "RISK_PER_TRADE",
                RISK_PER_TRADE,
                "Recommended range: 0.001 to 0.1 (0.1% to 10%)"
            ))

        # Max leverage
        if not isinstance(MAX_LEVERAGE, (int, float)):
            self.results.append(ValidationResult(
                ValidationSeverity.ERROR,
                "Max leverage must be a number",
                "MAX_LEVERAGE",
                MAX_LEVERAGE
            ))
        elif MAX_LEVERAGE < 1:
            self.results.append(ValidationResult(
                ValidationSeverity.ERROR,
                "Max leverage must be at least 1",
                "MAX_LEVERAGE",
                MAX_LEVERAGE
            ))
        elif MAX_LEVERAGE > 100:
            self.results.append(ValidationResult(
                ValidationSeverity.WARNING,
                "Very high leverage increases risk significantly",
                "MAX_LEVERAGE",
                MAX_LEVERAGE,
                "Consider reducing leverage for safer trading"
            ))

    def _validate_ema_parameters(self):
        """Validate EMA parameters."""
        if not isinstance(EMA_SHORT, int) or EMA_SHORT <= 0:
            self.results.append(ValidationResult(
                ValidationSeverity.ERROR,
                "EMA short period must be a positive integer",
                "EMA_SHORT",
                EMA_SHORT
            ))

        if not isinstance(EMA_LONG, int) or EMA_LONG <= 0:
            self.results.append(ValidationResult(
                ValidationSeverity.ERROR,
                "EMA long period must be a positive integer",
                "EMA_LONG",
                EMA_LONG
            ))

        if EMA_SHORT >= EMA_LONG:
            self.results.append(ValidationResult(
                ValidationSeverity.ERROR,
                "EMA short period must be less than long period",
                "EMA_SHORT/EMA_LONG",
                f"{EMA_SHORT}/{EMA_LONG}",
                "Set EMA_SHORT < EMA_LONG for proper crossover signals"
            ))

    def _validate_rsi_parameters(self):
        """Validate RSI parameters."""
        if not isinstance(RSI_PERIOD, int) or RSI_PERIOD <= 0:
            self.results.append(ValidationResult(
                ValidationSeverity.ERROR,
                "RSI period must be a positive integer",
                "RSI_PERIOD",
                RSI_PERIOD
            ))

        if not isinstance(RSI_OVERSOLD, (int, float)) or not 0 < RSI_OVERSOLD < 50:
            self.results.append(ValidationResult(
                ValidationSeverity.WARNING,
                "RSI oversold should be between 1 and 49",
                "RSI_OVERSOLD",
                RSI_OVERSOLD,
                "Typical range: 20-40"
            ))

        if not isinstance(RSI_OVERBOUGHT, (int, float)) or not 50 < RSI_OVERBOUGHT < 100:
            self.results.append(ValidationResult(
                ValidationSeverity.WARNING,
                "RSI overbought should be between 51 and 99",
                "RSI_OVERBOUGHT",
                RSI_OVERBOUGHT,
                "Typical range: 60-80"
            ))

        if RSI_OVERSOLD >= RSI_OVERBOUGHT:
            self.results.append(ValidationResult(
                ValidationSeverity.ERROR,
                "RSI oversold must be less than overbought",
                "RSI_OVERSOLD/RSI_OVERBOUGHT",
                f"{RSI_OVERSOLD}/{RSI_OVERBOUGHT}"
            ))

    def _validate_bb_parameters(self):
        """Validate Bollinger Bands parameters."""
        if not isinstance(BB_PERIOD, int) or BB_PERIOD <= 0:
            self.results.append(ValidationResult(
                ValidationSeverity.ERROR,
                "BB period must be a positive integer",
                "BB_PERIOD",
                BB_PERIOD
            ))

        if not isinstance(BB_STD_DEV, (int, float)) or BB_STD_DEV <= 0:
            self.results.append(ValidationResult(
                ValidationSeverity.ERROR,
                "BB standard deviation must be positive",
                "BB_STD_DEV",
                BB_STD_DEV
            ))
        elif BB_STD_DEV != 2.0:
            self.results.append(ValidationResult(
                ValidationSeverity.INFO,
                f"BB standard deviation {BB_STD_DEV} differs from default 2.0",
                "BB_STD_DEV",
                BB_STD_DEV
            ))

    def _validate_atr_parameters(self):
        """Validate ATR parameters."""
        if not isinstance(ATR_PERIOD, int) or ATR_PERIOD <= 0:
            self.results.append(ValidationResult(
                ValidationSeverity.ERROR,
                "ATR period must be a positive integer",
                "ATR_PERIOD",
                ATR_PERIOD
            ))

        if not isinstance(SL_MULTIPLIER, (int, float)) or SL_MULTIPLIER <= 0:
            self.results.append(ValidationResult(
                ValidationSeverity.ERROR,
                "ATR SL multiplier must be positive",
                "SL_MULTIPLIER",
                SL_MULTIPLIER
            ))

        if not isinstance(TP_MULTIPLIER, (int, float)) or TP_MULTIPLIER <= 0:
            self.results.append(ValidationResult(
                ValidationSeverity.ERROR,
                "ATR TP multiplier must be positive",
                "TP_MULTIPLIER",
                TP_MULTIPLIER
            ))

        if SL_MULTIPLIER >= TP_MULTIPLIER:
            self.results.append(ValidationResult(
                ValidationSeverity.WARNING,
                "Stop loss multiplier should be less than take profit multiplier",
                "SL_MULTIPLIER/TP_MULTIPLIER",
                f"{SL_MULTIPLIER}/{TP_MULTIPLIER}",
                "Consider SL_MULTIPLIER < TP_MULTIPLIER for profitable trades"
            ))

    def _validate_adx_parameters(self):
        """Validate ADX parameters."""
        # ADX_PERIOD is not defined in settings, only ADX_THRESHOLD
        if not isinstance(ADX_THRESHOLD, (int, float)) or not 0 < ADX_THRESHOLD < 100:
            self.results.append(ValidationResult(
                ValidationSeverity.WARNING,
                "ADX threshold should be between 1 and 99",
                "ADX_THRESHOLD",
                ADX_THRESHOLD,
                "Typical range: 20-30 for trend strength"
            ))

    def _validate_fibonacci_levels(self):
        """Validate Fibonacci levels."""
        # Skip validation as FIB_LEVELS is not defined in settings
        pass

    def _validate_volatility_threshold(self):
        """Validate volatility threshold."""
        # Skip validation as VOLATILITY_THRESHOLD is not defined in settings
        pass

    def _validate_output_path(self):
        """Validate output file path."""
        if not isinstance(OUTPUT_FILE, str):
            self.results.append(ValidationResult(
                ValidationSeverity.ERROR,
                "Output file path must be a string",
                "OUTPUT_FILE",
                OUTPUT_FILE
            ))
            return

        # Check if directory exists and is writable
        output_dir = os.path.dirname(OUTPUT_FILE)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except Exception as e:
                self.results.append(ValidationResult(
                    ValidationSeverity.ERROR,
                    f"Cannot create output directory: {e}",
                    "OUTPUT_FILE",
                    OUTPUT_FILE,
                    "Ensure the output directory is writable"
                ))

        # Check if file is writable
        try:
            with open(OUTPUT_FILE, 'w') as f:
                f.write('')  # Test write
            os.remove(OUTPUT_FILE)  # Clean up
        except Exception as e:
            self.results.append(ValidationResult(
                ValidationSeverity.ERROR,
                f"Output file path is not writable: {e}",
                "OUTPUT_FILE",
                OUTPUT_FILE,
                "Ensure the output file path is writable"
            ))

    def _validate_backtest_dates(self):
        """Validate backtest date parameters."""
        # Skip validation as backtest dates are not defined in settings
        pass

    def _validate_parameter_consistency(self):
        """Validate consistency across parameters."""
        # Check for very short timeframes with high leverage
        if TIMEFRAME in ['1m', '3m', '5m'] and MAX_LEVERAGE > 10:
            self.results.append(ValidationResult(
                ValidationSeverity.WARNING,
                f"High leverage ({MAX_LEVERAGE}x) with short timeframe ({TIMEFRAME}) increases risk",
                "MAX_LEVERAGE/TIMEFRAME",
                f"{MAX_LEVERAGE}x / {TIMEFRAME}",
                "Consider reducing leverage for short timeframes"
            ))

        # Check risk per trade vs capital
        if isinstance(CAPITAL, (int, float)) and isinstance(RISK_PER_TRADE, (int, float)):
            risk_amount = CAPITAL * RISK_PER_TRADE
            if risk_amount < 1:
                self.results.append(ValidationResult(
                    ValidationSeverity.WARNING,
                    f"Risk amount (${risk_amount:.2f}) is very low for effective trading",
                    "CAPITAL/RISK_PER_TRADE",
                    f"${CAPITAL} * {RISK_PER_TRADE}",
                    "Consider increasing RISK_PER_TRADE for meaningful position sizes"
                ))


def validate_configuration_at_startup() -> bool:
    """
    Validate configuration at system startup.

    Returns:
        True if configuration is valid (no errors), False otherwise
    """
    validator = ConfigurationValidator()
    results = validator.validate_all()

    # Log results
    errors = 0
    warnings = 0

    for result in results:
        if result.severity == ValidationSeverity.ERROR:
            trading_logger.log_error(f"CONFIG ERROR [{result.field}]: {result.message}")
            if result.suggestion:
                trading_logger.log_error(f"  Suggestion: {result.suggestion}")
            errors += 1
        elif result.severity == ValidationSeverity.WARNING:
            trading_logger.log_system_status(f"CONFIG WARNING [{result.field}]: {result.message}")
            if result.suggestion:
                trading_logger.log_system_status(f"  Suggestion: {result.suggestion}")
            warnings += 1
        else:  # INFO
            trading_logger.log_system_status(f"CONFIG INFO [{result.field}]: {result.message}")

    # Summary
    if errors > 0:
        trading_logger.log_error(f"Configuration validation failed: {errors} errors, {warnings} warnings")
        return False
    elif warnings > 0:
        trading_logger.log_system_status(f"Configuration validation passed with {warnings} warnings")
        return True
    else:
        trading_logger.log_system_status("Configuration validation passed")
        return True


if __name__ == "__main__":
    # Run validation when script is executed directly
    success = validate_configuration_at_startup()
    sys.exit(0 if success else 1)