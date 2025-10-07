import pandas as pd
import numpy as np
from config.settings import EMA_SHORT, EMA_LONG, RSI_PERIOD, BB_PERIOD, BB_STD_DEV, ATR_PERIOD

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def bb(series, period=20, std_dev=2):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

def atr(high, low, close, period=14):
    """
    Calculate Average True Range for volatility measurement.
    """
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def adx(high, low, close, period=14):
    """
    Calculate Average Directional Index (ADX) for trend strength.
    ADX > 25 indicates strong trend, ADX < 20 indicates weak trend.
    Returns: adx, di_plus, di_minus
    """
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

def calculate_indicators(df):
    """
    Calculate all technical indicators for the dataset.
    """
    from config.settings import EMA_SHORT, EMA_LONG, RSI_PERIOD, BB_PERIOD, BB_STD_DEV, ATR_PERIOD, ADX_ENABLED, FIBONACCI_ENABLED

    df['EMA9'] = ema(df['close'], EMA_SHORT)
    df['EMA21'] = ema(df['close'], EMA_LONG)
    df['RSI'] = rsi(df['close'], RSI_PERIOD)
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = bb(df['close'], BB_PERIOD, BB_STD_DEV)
    df['ATR'] = atr(df['high'], df['low'], df['close'], ATR_PERIOD)

    # Optional indicators
    if ADX_ENABLED:
        df['ADX'], df['DI_plus'], df['DI_minus'] = adx(df['high'], df['low'], df['close'], ATR_PERIOD)

    if FIBONACCI_ENABLED:
        # Determine trend based on EMA crossover
        trend = 'bullish' if df['EMA9'].iloc[-1] > df['EMA21'].iloc[-1] else 'bearish'
        df['Fib_161'], df['Fib_262'], df['Fib_424'] = fibonacci_extensions(df['high'], df['low'], df['close'], trend)

    return df

def generate_signals(df):
    df.dropna(inplace=True)  # Remove NaN
    df.reset_index(drop=True, inplace=True)  # Reset Index
    df['EMA_crossover'] = np.where(df['EMA9'].values > df['EMA21'].values, 1, -1)
    buy_condition = (df['EMA_crossover'].values == 1) & (df['RSI'].values < 30) & (df['Close'].values > df['BB_lower'].values)
    df['Buy_Signal'] = buy_condition.astype(int)
    sell_condition = (df['EMA_crossover'].values == -1) & (df['RSI'].values > 70) & (df['Close'].values < df['BB_upper'].values)
    df['Sell_Signal'] = sell_condition.astype(int)
    return df

def calculate_risk_management(df, capital=10000, risk_per_trade=0.01):
    df['Position_Size'] = (capital * risk_per_trade) / df['ATR']
    df['Stop_Loss_Buy'] = df['Close'] - df['ATR']
    df['Take_Profit_Buy'] = df['Close'] + 2 * df['ATR']
    atr_avg = df['ATR'].expanding().mean()
    df['Leverage'] = np.where(df['ATR'] < atr_avg, 10, 5)
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
