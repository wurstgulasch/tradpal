import pandas as pd
import numpy as np
from typing import Tuple, Dict

# Check if TA-Lib is available
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    talib = None

def ema(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average.
    Uses TA-Lib if available, otherwise pandas implementation.
    Returns NaN for all values if period > data length.
    """
    if len(series) < period:
        return pd.Series([np.nan] * len(series), index=series.index)

    if TALIB_AVAILABLE:
        try:
            values = series.values.astype(float)
            ema_values = talib.EMA(values, timeperiod=period)
            return pd.Series(ema_values, index=series.index)
        except Exception as e:
            pass

    # Pandas implementation (fallback)
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index.
    Uses TA-Lib if available, otherwise pandas implementation.
    Returns NaN for initial values until enough data is available.
    """
    if len(series) < period + 1:
        return pd.Series([np.nan] * len(series), index=series.index)

    if TALIB_AVAILABLE:
        try:
            values = series.values.astype(float)
            rsi_values = talib.RSI(values, timeperiod=period)
            return pd.Series(rsi_values, index=series.index)
        except Exception as e:
            pass

    # Pandas implementation (fallback)
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()

    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(loss != 0, np.nan)

    return rsi

def bb(series: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range for volatility measurement.
    Uses TA-Lib if available, otherwise pandas implementation.
    """
    if TALIB_AVAILABLE:
        try:
            high_values = high.values.astype(float)
            low_values = low.values.astype(float)
            close_values = close.values.astype(float)
            atr_values = talib.ATR(high_values, low_values, close_values, timeperiod=period)
            return pd.Series(atr_values, index=high.index)
        except Exception as e:
            pass

    # Pandas implementation (fallback)
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def macd(series: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    Returns: macd_line, signal_line, histogram
    Uses TA-Lib if available, otherwise pandas implementation.
    """
    if TALIB_AVAILABLE:
        try:
            values = series.values.astype(float)
            macd_line, signal_line, histogram = talib.MACD(values, fastperiod=fast_period,
                                                          slowperiod=slow_period, signalperiod=signal_period)
            return pd.Series(macd_line, index=series.index), pd.Series(signal_line, index=series.index), pd.Series(histogram, index=series.index)
        except Exception as e:
            pass

    # Pandas implementation (fallback)
    fast_ema = ema(series, fast_period)
    slow_ema = ema(series, slow_period)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram

def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic Oscillator.
    Returns: %K, %D (smoothed %K)
    Uses TA-Lib if available, otherwise pandas implementation.
    """
    if TALIB_AVAILABLE:
        try:
            high_values = high.values.astype(float)
            low_values = low.values.astype(float)
            close_values = close.values.astype(float)
            k_values, d_values = talib.STOCH(high_values, low_values, close_values,
                                           fastk_period=k_period, slowk_period=d_period, slowd_period=d_period)
            return pd.Series(k_values, index=high.index), pd.Series(d_values, index=high.index)
        except Exception as e:
            pass

    # Pandas implementation (fallback)
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period).mean()

    return k_percent, d_percent

def calculate_indicators(df: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
    """
    Calculate basic technical indicators for trading signals.

    Args:
        df: OHLCV DataFrame
        config: Optional configuration for indicators

    Returns:
        DataFrame with added indicator columns
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    # Basic EMA indicators (use expected column names from tests)
    try:
        df['EMA9'] = ema(df['close'], 9)
        df['EMA21'] = ema(df['close'], 21)
    except Exception as e:
        df['EMA9'] = np.nan
        df['EMA21'] = np.nan

    # RSI
    try:
        df['RSI'] = rsi(df['close'], 14)
    except Exception as e:
        df['RSI'] = np.nan

    # Bollinger Bands
    try:
        bb_upper, bb_middle, bb_lower = bb(df['close'], 20, 2)
        df['BB_upper'] = bb_upper
        df['BB_middle'] = bb_middle
        df['BB_lower'] = bb_lower
    except Exception as e:
        df['BB_upper'] = np.nan
        df['BB_middle'] = np.nan
        df['BB_lower'] = np.nan

    # ATR for risk management
    try:
        df['ATR'] = atr(df['high'], df['low'], df['close'], 14)
    except Exception as e:
        df['ATR'] = np.nan

    return df
