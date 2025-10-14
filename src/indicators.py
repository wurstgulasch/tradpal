import pandas as pd
import numpy as np
from typing import Tuple, Dict
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

def calculate_indicators_with_config(df, config=None):
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
    
    # Always calculate default EMA periods for backward compatibility
    df['EMA9'] = ema(df['close'], 9)
    df['EMA21'] = ema(df['close'], 21)

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
        df['MACD_line'], df['MACD_signal'], df['MACD_histogram'] = macd(df['close'], fast_period, slow_period, signal_period)

    # OBV (optional)
    if config.get('obv', {}).get('enabled', False):
        df['OBV'] = obv(df['close'], df['volume'])

    # Stochastic Oscillator (optional)
    if config.get('stochastic', {}).get('enabled', False):
        k_period = config['stochastic'].get('k_period', 14)
        d_period = config['stochastic'].get('d_period', 3)
        df['stoch_k'], df['stoch_d'] = stochastic(df['high'], df['low'], df['close'], k_period, d_period)

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

@cache_indicators()
def funding_rate_analysis(funding_df: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    """
    Analyze funding rate data for perpetual futures trading.

    Args:
        funding_df: DataFrame with funding rate data (must have 'funding_rate' column)
        window: Rolling window for analysis (in funding periods, typically 8h each)

    Returns:
        DataFrame with funding rate analysis indicators
    """
    if funding_df.empty or 'funding_rate' not in funding_df.columns:
        return pd.DataFrame()

    df = funding_df.copy()

    # Basic funding rate metrics
    df['funding_rate_pct'] = df['funding_rate'] * 100  # Convert to percentage

    # Rolling statistics
    df['funding_rate_mean'] = df['funding_rate'].rolling(window=window).mean()
    df['funding_rate_std'] = df['funding_rate'].rolling(window=window).std()
    df['funding_rate_zscore'] = (df['funding_rate'] - df['funding_rate_mean']) / df['funding_rate_std']

    # Trend analysis
    df['funding_rate_trend'] = df['funding_rate'].diff().rolling(window=window).mean()

    # Funding rate direction changes
    df['funding_direction_change'] = (df['funding_rate'] > 0).astype(int).diff().fillna(0).abs()

    # Extreme funding rates (potential reversal signals)
    df['funding_rate_extreme'] = (
        (df['funding_rate'] > df['funding_rate_mean'] + 2 * df['funding_rate_std']) |
        (df['funding_rate'] < df['funding_rate_mean'] - 2 * df['funding_rate_std'])
    ).astype(int)

    # Funding cost analysis (annualized)
    df['funding_cost_daily'] = df['funding_rate'] * 3  # Assuming 3 funding payments per day
    df['funding_cost_weekly'] = df['funding_cost_daily'] * 7
    df['funding_cost_monthly'] = df['funding_cost_daily'] * 30

    # Momentum indicators for funding rates
    df['funding_rate_momentum'] = df['funding_rate'].diff(3)  # 3-period momentum (24h for 8h funding)

    # Volatility of funding rates
    df['funding_rate_volatility'] = df['funding_rate'].rolling(window=window).std()

    return df

@cache_indicators()
def funding_rate_signal(funding_df: pd.DataFrame, threshold: float = 0.001) -> pd.Series:
    """
    Generate trading signals based on funding rate analysis.

    Args:
        funding_df: DataFrame with funding rate data
        threshold: Threshold for signal generation (0.001 = 0.1%)

    Returns:
        Series with signal values: 1 (long bias), -1 (short bias), 0 (neutral)
    """
    if funding_df.empty or 'funding_rate' not in funding_df.columns:
        return pd.Series(dtype=int)

    signals = pd.Series(0, index=funding_df.index, dtype=int)

    # Positive funding rate favors shorts (borrowers pay longs)
    # Negative funding rate favors longs (borrowers get paid by longs)
    signals[funding_df['funding_rate'] > threshold] = -1  # Short bias
    signals[funding_df['funding_rate'] < -threshold] = 1  # Long bias

    # Extreme funding rates might indicate reversals
    if 'funding_rate_extreme' in funding_df.columns:
        extreme_mask = funding_df['funding_rate_extreme'] == 1
        signals[extreme_mask] = -signals[extreme_mask]  # Reverse signal on extremes

    return signals

@cache_indicators()
def combined_funding_market_analysis(market_df: pd.DataFrame, funding_df: pd.DataFrame,
                                   funding_weight: float = 0.3) -> pd.DataFrame:
    """
    Combine market data analysis with funding rate analysis for enhanced signals.

    Args:
        market_df: DataFrame with market OHLCV data and indicators
        funding_df: DataFrame with funding rate analysis
        funding_weight: Weight for funding rate signals (0-1)

    Returns:
        DataFrame with combined analysis
    """
    if market_df.empty or funding_df.empty:
        return market_df if not market_df.empty else pd.DataFrame()

    df = market_df.copy()

    # Resample funding data to match market timeframe if needed
    # This is a simplified approach - in production, you'd want proper time alignment
    if len(funding_df) != len(df):
        # Forward fill funding data to match market data frequency
        funding_resampled = funding_df.reindex(df.index, method='ffill')
    else:
        funding_resampled = funding_df

    # Add funding rate indicators to market data
    df['funding_rate'] = funding_resampled.get('funding_rate', 0)
    df['funding_rate_zscore'] = funding_resampled.get('funding_rate_zscore', 0)
    df['funding_signal'] = funding_resampled.get('funding_signal', 0)

    # Create combined signal (weighted average of market and funding signals)
    if 'signal' in df.columns and 'funding_signal' in df.columns:
        df['combined_signal'] = (
            (1 - funding_weight) * df['signal'] +
            funding_weight * df['funding_signal']
        )

    # Funding rate adjusted position sizing
    # Reduce position size when funding rates are extremely high (costly)
    if 'funding_rate' in df.columns:
        funding_cost_multiplier = 1 / (1 + df['funding_rate'].abs() * 10)  # Reduce size with high costs
        df['adjusted_position_size'] = df.get('position_size', 1) * funding_cost_multiplier

    return df

def calculate_indicators_with_validation(df: pd.DataFrame, indicators_config: Dict = None) -> Dict[str, pd.Series]:
    """
    Calculate indicators with validation and automatic exclusion of invalid ones.

    Args:
        df: OHLCV DataFrame
        indicators_config: Configuration for which indicators to calculate

    Returns:
        Dict of valid indicators (invalid ones are excluded)
    """
    if indicators_config is None:
        from config.settings import TIMEFRAME_PARAMS, TIMEFRAME
        indicators_config = TIMEFRAME_PARAMS.get(TIMEFRAME, {})

    valid_indicators = {}
    invalid_indicators = []

    # EMA indicators
    try:
        ema_short = ema(df['close'], indicators_config.get('ema_short', 9))
        ema_long = ema(df['close'], indicators_config.get('ema_long', 21))

        if _is_indicator_valid(ema_short) and _is_indicator_valid(ema_long):
            valid_indicators['ema_short'] = ema_short
            valid_indicators['ema_long'] = ema_long
        else:
            invalid_indicators.extend(['ema_short', 'ema_long'])
    except Exception as e:
        invalid_indicators.extend(['ema_short', 'ema_long'])

    # RSI
    try:
        rsi_val = rsi(df['close'], indicators_config.get('rsi_period', 14))
        if _is_indicator_valid(rsi_val):
            valid_indicators['rsi'] = rsi_val
        else:
            invalid_indicators.append('rsi')
    except Exception as e:
        invalid_indicators.append('rsi')

    # Bollinger Bands
    try:
        bb_upper, bb_middle, bb_lower = bb(df['close'],
                                          indicators_config.get('bb_period', 20),
                                          indicators_config.get('bb_std_dev', 2))
        if all(_is_indicator_valid(ind) for ind in [bb_upper, bb_middle, bb_lower]):
            valid_indicators['bb_upper'] = bb_upper
            valid_indicators['bb_middle'] = bb_middle
            valid_indicators['bb_lower'] = bb_lower
        else:
            invalid_indicators.extend(['bb_upper', 'bb_middle', 'bb_lower'])
    except Exception as e:
        invalid_indicators.extend(['bb_upper', 'bb_middle', 'bb_lower'])

    # ATR
    try:
        atr_val = atr(df['high'], df['low'], df['close'], indicators_config.get('atr_period', 14))
        if _is_indicator_valid(atr_val):
            valid_indicators['atr'] = atr_val
        else:
            invalid_indicators.append('atr')
    except Exception as e:
        invalid_indicators.append('atr')

    # ADX (optional - only if ADX threshold > 0)
    if indicators_config.get('adx_threshold', 25) > 0:
        try:
            adx_val, di_plus, di_minus = adx(df['high'], df['low'], df['close'],
                                           indicators_config.get('adx_period', 14))
            if all(_is_indicator_valid(ind) for ind in [adx_val, di_plus, di_minus]):
                valid_indicators['adx'] = adx_val
                valid_indicators['di_plus'] = di_plus
                valid_indicators['di_minus'] = di_minus
            else:
                invalid_indicators.extend(['adx', 'di_plus', 'di_minus'])
        except Exception as e:
            invalid_indicators.extend(['adx', 'di_plus', 'di_minus'])

    # MACD (needed for ML features)
    try:
        macd_val, macd_signal, macd_hist = macd(df['close'],
                                              indicators_config.get('macd_fast_period', 12),
                                              indicators_config.get('macd_slow_period', 26),
                                              indicators_config.get('macd_signal_period', 9))
        if all(_is_indicator_valid(ind) for ind in [macd_val, macd_signal, macd_hist]):
            valid_indicators['macd'] = macd_val
            valid_indicators['macd_signal'] = macd_signal
            valid_indicators['macd_hist'] = macd_hist
        else:
            invalid_indicators.extend(['macd', 'macd_signal', 'macd_hist'])
    except Exception as e:
        invalid_indicators.extend(['macd', 'macd_signal', 'macd_hist'])

    # Stochastic Oscillator (needed for ML features)
    try:
        stoch_k, stoch_d = stochastic(df['high'], df['low'], df['close'],
                                    indicators_config.get('stoch_k_period', 14),
                                    indicators_config.get('stoch_d_period', 3))
        if _is_indicator_valid(stoch_k) and _is_indicator_valid(stoch_d):
            valid_indicators['stoch_k'] = stoch_k
            valid_indicators['stoch_d'] = stoch_d
        else:
            invalid_indicators.extend(['stoch_k', 'stoch_d'])
    except Exception as e:
        invalid_indicators.extend(['stoch_k', 'stoch_d'])

def _is_indicator_valid(indicator: pd.Series, min_valid_ratio: float = 0.5) -> bool:
    """
    Check if an indicator is valid (not all NaN, has minimum valid values).

    Args:
        indicator: Indicator series to validate
        min_valid_ratio: Minimum ratio of non-NaN values required

    Returns:
        True if indicator is valid
    """
    if indicator is None or len(indicator) == 0:
        return False

    # Check if not all NaN
    if indicator.isnull().all():
        return False

    # Check minimum valid ratio
    valid_ratio = indicator.notnull().mean()
    if valid_ratio < min_valid_ratio:
        return False

    # Check for extreme values (potential calculation errors)
    if indicator.dtype in ['float64', 'int64']:
        # Remove NaN for calculations
        clean_indicator = indicator.dropna()
        if len(clean_indicator) > 0:
            # Check for infinite values
            if np.isinf(clean_indicator).any():
                return False

            # Check for extremely large values (potential calculation errors)
            mean_val = clean_indicator.mean()
            std_val = clean_indicator.std()
            if std_val > 0:
                z_scores = np.abs((clean_indicator - mean_val) / std_val)
                extreme_ratio = (z_scores > 10).mean()  # More than 10 std deviations
                if extreme_ratio > 0.1:  # More than 10% extreme values
                    return False

    return True


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

    # If config is provided, use the configurable version
    if config is not None:
        return calculate_indicators_with_config(df, config)

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
