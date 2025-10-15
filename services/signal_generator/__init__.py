"""
Signal Generator Service - Trading signal generation and enhancement.

Provides traditional and ML-enhanced signal generation capabilities.
"""

import logging
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

from config.settings import (
    EMA_SHORT, EMA_LONG, RSI_PERIOD, RSI_OVERSOLD, RSI_OVERBOUGHT,
    BB_PERIOD, BB_STD_DEV, ATR_PERIOD
)
from services.core.indicators import ema, rsi, bb, atr
from services.ml_predictor import get_ml_predictor

logger = logging.getLogger(__name__)


def generate_traditional_signals(data: pd.DataFrame) -> pd.DataFrame:
    """Generate traditional technical analysis signals."""
    df = data.copy()

    # Calculate indicators
    df['EMA_short'] = ema(df['close'], EMA_SHORT)
    df['EMA_long'] = ema(df['close'], EMA_LONG)
    df['RSI'] = rsi(df['close'], RSI_PERIOD)
    bb_result = bb(df['close'], BB_PERIOD, BB_STD_DEV)
    df['BB_upper'] = bb_result['upper']
    df['BB_lower'] = bb_result['lower']
    df['ATR'] = atr(df['high'], df['low'], df['close'], ATR_PERIOD)

    # Generate signals
    df['Buy_Signal'] = 0
    df['Sell_Signal'] = 0

    # EMA crossover signals
    ema_short = df['EMA_short']
    ema_long = df['EMA_long']

    # Buy signal: EMA_short crosses above EMA_long and RSI oversold
    buy_condition = (
        (ema_short > ema_long) &
        (df['RSI'] < RSI_OVERSOLD) &
        (df['close'] > df['BB_lower'])
    )
    df.loc[buy_condition, 'Buy_Signal'] = 1

    # Sell signal: EMA_short crosses below EMA_long and RSI overbought
    sell_condition = (
        (ema_short < ema_long) &
        (df['RSI'] > RSI_OVERBOUGHT) &
        (df['close'] < df['BB_upper'])
    )
    df.loc[sell_condition, 'Sell_Signal'] = 1

    return df


def apply_ml_signal_enhancement(data: pd.DataFrame) -> pd.DataFrame:
    """Apply ML-based signal enhancement to existing signals."""
    ml_predictor = get_ml_predictor()

    if ml_predictor is None:
        # Return original data if ML not available
        return data

    return ml_predictor.enhance_signals(data)


def apply_funding_rate_signal_enhancement(data: pd.DataFrame) -> pd.DataFrame:
    """Apply funding rate-based signal enhancement."""
    # Placeholder for funding rate enhancement
    # This would integrate with funding rate data
    return data


def combine_signals(data: pd.DataFrame,
                   use_ml: bool = True,
                   use_funding_rate: bool = False) -> pd.DataFrame:
    """Combine multiple signal sources for enhanced trading decisions."""
    df = generate_traditional_signals(data)

    if use_ml:
        df = apply_ml_signal_enhancement(df)

    if use_funding_rate:
        df = apply_funding_rate_signal_enhancement(df)

    return df