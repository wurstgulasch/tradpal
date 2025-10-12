import pandas as pd
import numpy as np
from typing import Tuple
from config.settings import EMA_SHORT, EMA_LONG, RSI_PERIOD, BB_PERIOD, BB_STD_DEV, ATR_PERIOD
# Import cache functionality
try:
    from .cache import cache_indicators
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    def cache_indicators(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Check if TA-Lib is available
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    talib = None

@cache_indicators()
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
            # TA-Lib EMA expects numpy array
            values = series.values.astype(float)
            ema_values = talib.EMA(values, timeperiod=period)
            return pd.Series(ema_values, index=series.index)
        except Exception as e:
            # Fallback to pandas if TA-Lib fails
            pass

    # Pandas implementation (fallback)
    return series.ewm(span=period, adjust=False).mean()

@cache_indicators()
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index.
    Uses TA-Lib if available, otherwise pandas implementation.
    Returns NaN for initial values until enough data is available.
    For constant data, RSI is 50 (neutral).
    """
    if len(series) < period + 1:
        return pd.Series([np.nan] * len(series), index=series.index)

    if TALIB_AVAILABLE:
        try:
            # TA-Lib RSI expects numpy array
            values = series.values.astype(float)
            rsi_values = talib.RSI(values, timeperiod=period)
            return pd.Series(rsi_values, index=series.index)
        except Exception as e:
            # Fallback to pandas if TA-Lib fails
            pass

    # Pandas implementation (fallback)
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()

    # Handle division by zero for constant data
    rs = gain / loss.replace(0, np.nan)  # Replace 0 with NaN to avoid 0/0
    rsi = 100 - (100 / (1 + rs))

    # For constant data (no gains/losses), RSI is undefined, so set to NaN
    rsi = rsi.where(loss != 0, np.nan)

    return rsi

@cache_indicators()
def bb(series: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

@cache_indicators()
def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range for volatility measurement.
    Uses TA-Lib if available, otherwise pandas implementation.
    """
    if TALIB_AVAILABLE:
        try:
            # TA-Lib ATR expects numpy arrays
            high_values = high.values.astype(float)
            low_values = low.values.astype(float)
            close_values = close.values.astype(float)
            atr_values = talib.ATR(high_values, low_values, close_values, timeperiod=period)
            return pd.Series(atr_values, index=high.index)
        except Exception as e:
            # Fallback to pandas if TA-Lib fails
            pass

    # Pandas implementation (fallback)
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def adx(high, low, close, period=14):
    """
    Calculate Average Directional Index (ADX) for trend strength.
    Uses TA-Lib if available, otherwise pandas implementation.
    Returns: adx, di_plus, di_minus
    """
    if TALIB_AVAILABLE:
        try:
            # TA-Lib ADX expects numpy arrays
            high_values = high.values.astype(float)
            low_values = low.values.astype(float)
            close_values = close.values.astype(float)
            adx_values = talib.ADX(high_values, low_values, close_values, timeperiod=period)
            di_plus_values = talib.PLUS_DI(high_values, low_values, close_values, timeperiod=period)
            di_minus_values = talib.MINUS_DI(high_values, low_values, close_values, timeperiod=period)
            return pd.Series(adx_values, index=high.index), pd.Series(di_plus_values, index=high.index), pd.Series(di_minus_values, index=high.index)
        except Exception as e:
            # Fallback to pandas if TA-Lib fails
            pass

    # Pandas implementation (fallback)
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    # Calculate Directional Movement
    dm_plus = pd.Series(np.where((high - high.shift(1)) > (low.shift(1) - low),
                                 np.maximum(high - high.shift(1), 0), 0), index=high.index)
    dm_minus = pd.Series(np.where((low.shift(1) - low) > (high - high.shift(1)),
                                  np.maximum(low.shift(1) - low, 0), 0), index=low.index)

    # Calculate Directional Indicators
    di_plus = 100 * (dm_plus.rolling(window=period).mean() / atr)
    di_minus = 100 * (dm_minus.rolling(window=period).mean() / atr)

    # Calculate DX and ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.rolling(window=period).mean()

    return adx, di_plus, di_minus

def fibonacci_extensions(high, low, close, trend='bullish'):
    """
    Calculate Fibonacci extension levels for take-profit targets.
    Common extensions: 161.8%, 261.8%, 423.6%
    """
    if trend == 'bullish':
        # For bullish trend: extensions above recent swing high
        swing_high = high.rolling(window=20).max()
        range_size = swing_high - low.rolling(window=20).min()

        fib_161 = swing_high + range_size * 0.618
        fib_262 = swing_high + range_size * 1.618
        fib_424 = swing_high + range_size * 2.618
    else:
        # For bearish trend: extensions below recent swing low
        swing_low = low.rolling(window=20).min()
        range_size = high.rolling(window=20).max() - swing_low

        fib_161 = swing_low - range_size * 0.618
        fib_262 = swing_low - range_size * 1.618
        fib_424 = swing_low - range_size * 2.618

    return fib_161, fib_262, fib_424

@cache_indicators()
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
            # Fallback to pandas if TA-Lib fails
            pass

    # Pandas implementation (fallback)
    fast_ema = ema(series, fast_period)
    slow_ema = ema(series, slow_period)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram

@cache_indicators()
def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV).
    Uses TA-Lib if available, otherwise pandas implementation.
    """
    if TALIB_AVAILABLE:
        try:
            close_values = close.values.astype(float)
            volume_values = volume.values.astype(float)
            obv_values = talib.OBV(close_values, volume_values)
            return pd.Series(obv_values, index=close.index)
        except Exception as e:
            # Fallback to pandas if TA-Lib fails
            pass

    # Pandas implementation (fallback)
    obv_series = pd.Series(0, index=close.index)
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv_series.iloc[i] = obv_series.iloc[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv_series.iloc[i] = obv_series.iloc[i-1] - volume.iloc[i]
        else:
            obv_series.iloc[i] = obv_series.iloc[i-1]

    return obv_series

@cache_indicators()
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
            # Fallback to pandas if TA-Lib fails
            pass

    # Pandas implementation (fallback)
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period).mean()

    return k_percent, d_percent

@cache_indicators()
def chaikin_money_flow(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 21) -> pd.Series:
    """
    Calculate Chaikin Money Flow (CMF).
    Uses TA-Lib if available, otherwise pandas implementation.
    """
    if TALIB_AVAILABLE:
        try:
            high_values = high.values.astype(float)
            low_values = low.values.astype(float)
            close_values = close.values.astype(float)
            volume_values = volume.values.astype(float)
            cmf_values = talib.AD(high_values, low_values, close_values, volume_values)
            # CMF is accumulation/distribution normalized by volume
            cmf = pd.Series(cmf_values, index=high.index)
            return cmf.rolling(window=period).mean() / volume.rolling(window=period).mean()
        except Exception as e:
            # Fallback to pandas if TA-Lib fails
            pass

    # Pandas implementation (fallback)
    # Money Flow Multiplier
    mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mfm = mfm.fillna(0)  # Handle division by zero

    # Money Flow Volume
    mfv = mfm * volume

    # Chaikin Money Flow
    cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()

    return cmf

def calculate_indicators(df, config=None):
    """
    Calculate all technical indicators for the dataset based on configuration.
    If no config provided, uses DEFAULT_INDICATOR_CONFIG.
    """
    from config.settings import DEFAULT_INDICATOR_CONFIG
    if config is None:
        config = DEFAULT_INDICATOR_CONFIG

    # EMA
    if config.get('ema', {}).get('enabled', False):
        periods = config['ema'].get('periods', [9, 21])
        for period in periods:
            df[f'EMA{period}'] = ema(df['close'], period)

    # RSI
    if config.get('rsi', {}).get('enabled', False):
        period = config['rsi'].get('period', 14)
        df['RSI'] = rsi(df['close'], period)

    # Bollinger Bands
    if config.get('bb', {}).get('enabled', False):
        period = config['bb'].get('period', 20)
        std_dev = config['bb'].get('std_dev', 2)
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = bb(df['close'], period, std_dev)

    # ATR
    if config.get('atr', {}).get('enabled', False):
        period = config['atr'].get('period', 14)
        df['ATR'] = atr(df['high'], df['low'], df['close'], period)

    # ADX (optional)
    if config.get('adx', {}).get('enabled', False):
        period = config['adx'].get('period', 14)
        df['ADX'], df['DI_plus'], df['DI_minus'] = adx(df['high'], df['low'], df['close'], period)

    # Fibonacci (optional)
    if config.get('fibonacci', {}).get('enabled', False):
        # Determine trend based on EMA crossover if available
        trend = 'bullish'
        if 'EMA9' in df.columns and 'EMA21' in df.columns:
            trend = 'bullish' if df['EMA9'].iloc[-1] > df['EMA21'].iloc[-1] else 'bearish'
        df['Fib_161'], df['Fib_262'], df['Fib_424'] = fibonacci_extensions(df['high'], df['low'], df['close'], trend)

    # MACD (optional)
    if config.get('macd', {}).get('enabled', False):
        fast_period = config['macd'].get('fast_period', 12)
        slow_period = config['macd'].get('slow_period', 26)
        signal_period = config['macd'].get('signal_period', 9)
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = macd(df['close'], fast_period, slow_period, signal_period)

    # OBV (optional)
    if config.get('obv', {}).get('enabled', False):
        df['OBV'] = obv(df['close'], df['volume'])

    # Stochastic Oscillator (optional)
    if config.get('stochastic', {}).get('enabled', False):
        k_period = config['stochastic'].get('k_period', 14)
        d_period = config['stochastic'].get('d_period', 3)
        df['Stoch_K'], df['Stoch_D'] = stochastic(df['high'], df['low'], df['close'], k_period, d_period)

    # Chaikin Money Flow (optional)
    if config.get('cmf', {}).get('enabled', False):
        period = config['cmf'].get('period', 21)
        df['CMF'] = chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'], period)

    return df

def plot_indicators(df):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    ax1.plot(df.index, df['Close'], label='Close Price')
    ax1.plot(df.index, df['EMA9'], label='EMA9')
    ax1.plot(df.index, df['EMA21'], label='EMA21')
    ax1.fill_between(df.index, df['BB_upper'], df['BB_lower'], alpha=0.1, color='blue', label='BB')
    ax1.scatter(df[df['Buy_Signal'] == 1].index, df[df['Buy_Signal'] == 1]['Close'], marker='^', color='green', label='Buy')
    ax1.scatter(df[df['Sell_Signal'] == 1].index, df[df['Sell_Signal'] == 1]['Close'], marker='v', color='red', label='Sell')
    ax1.legend()
    ax1.set_title('Price and Indicators')
    ax2.plot(df.index, df['RSI'], label='RSI')
    ax2.axhline(30, color='green', linestyle='--')
    ax2.axhline(70, color='red', linestyle='--')
    ax2.legend()
    ax2.set_title('RSI')
    ax3.plot(df.index, df['ATR'], label='ATR')
    ax3.legend()
    ax3.set_title('ATR')
    plt.tight_layout()
    plt.show()
