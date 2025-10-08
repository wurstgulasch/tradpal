import numpy as np
import pandas as pd
import asyncio
from typing import Dict, Any
from config.settings import CAPITAL, RISK_PER_TRADE, MTA_ENABLED, MTA_HIGHER_TIMEFRAME
from src.data_fetcher import fetch_data
from src.indicators import calculate_indicators

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates Buy/Sell signals based on EMA crossover, RSI and BB.
    Includes optional Multi-Timeframe Analysis (MTA) for signal confirmation.
    """
    from config.settings import STRICT_SIGNALS_ENABLED, get_timeframe_params, TIMEFRAME
    # Don't drop NaN values here - let indicators handle their own NaN values
    # df.dropna(inplace=True)  # Remove this line
    df.reset_index(drop=True, inplace=True)  # Reset Index

    # Get timeframe-specific parameters
    params = get_timeframe_params(TIMEFRAME)

    # Basic signal generation
    df['EMA_crossover'] = np.where(df['EMA9'].values > df['EMA21'].values, 1, -1)
    
    if STRICT_SIGNALS_ENABLED:
        buy_condition = (df['EMA_crossover'].values == 1) & (df['RSI'].values < params['rsi_oversold']) & (df['close'].values > df['BB_lower'].values)
        df['Buy_Signal'] = buy_condition.astype(int)
        sell_condition = (df['EMA_crossover'].values == -1) & (df['RSI'].values > params['rsi_overbought']) & (df['close'].values < df['BB_upper'].values)
        df['Sell_Signal'] = sell_condition.astype(int)
    else:
        # Simplified signals: only EMA crossover
        df['Buy_Signal'] = (df['EMA_crossover'].values == 1).astype(int)
        df['Sell_Signal'] = (df['EMA_crossover'].values == -1).astype(int)

    # Multi-Timeframe Analysis (MTA) for signal confirmation
    if MTA_ENABLED:
        df = apply_multi_timeframe_analysis(df)

    return df

def apply_multi_timeframe_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Multi-Timeframe Analysis to confirm signals with higher timeframe trends.
    Only keeps signals that are confirmed by the higher timeframe trend.
    """
    try:
        # Fetch higher timeframe data
        higher_tf_data = fetch_data(limit=50)  # Fetch recent data for higher timeframe

        # For demo purposes, we'll simulate higher timeframe analysis
        # In a real implementation, you'd need to resample the data to higher timeframe

        # Get trend from higher timeframe (simplified)
        # This would compare EMA crossover on higher timeframe
        higher_trend = 1 if higher_tf_data['close'].iloc[-1] > higher_tf_data['close'].iloc[-10] else -1

        # Filter signals based on higher timeframe confirmation
        df['Buy_Signal_Confirmed'] = np.where(
            (df['Buy_Signal'] == 1) & (higher_trend == 1), 1, 0
        )
        df['Sell_Signal_Confirmed'] = np.where(
            (df['Sell_Signal'] == 1) & (higher_trend == -1), 1, 0
        )

        # Replace original signals with confirmed signals
        df['Buy_Signal'] = df['Buy_Signal_Confirmed']
        df['Sell_Signal'] = df['Sell_Signal_Confirmed']

        # Clean up temporary columns
        df.drop(['Buy_Signal_Confirmed', 'Sell_Signal_Confirmed'], axis=1, inplace=True)

    except Exception as e:
        print(f"MTA analysis failed: {e}")
        # Continue without MTA if it fails

    return df

def calculate_risk_management(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates position size, stop-loss, take-profit based on ATR with multipliers.
    Position size is calculated as both absolute amount and percentage of total capital.
    Requires Buy_Signal and Sell_Signal columns.
    """
    from config.settings import get_timeframe_params, TIMEFRAME

    # Check for required signal columns
    if 'Buy_Signal' not in df.columns or 'Sell_Signal' not in df.columns:
        raise KeyError("DataFrame must contain 'Buy_Signal' and 'Sell_Signal' columns")

    # Get timeframe-specific parameters
    params = get_timeframe_params(TIMEFRAME)

    # Calculate absolute position size with ATR multiplier
    atr_multiplier = params.get('atr_sl_multiplier', 1.0)
    df['Position_Size_Absolute'] = (CAPITAL * RISK_PER_TRADE) / (df['ATR'] * atr_multiplier)

    # Calculate the actual percentage of capital that would be at risk
    # Risk amount = Position_Size_Absolute * ATR * SL_Multiplier
    risk_amount = df['Position_Size_Absolute'] * df['ATR'] * params['atr_sl_multiplier']
    df['Position_Size_Percent'] = (risk_amount / CAPITAL) * 100

    # Calculate stop-loss and take-profit with multipliers
    df['Stop_Loss_Buy'] = df['close'] - (df['ATR'] * params['atr_sl_multiplier'])
    df['Take_Profit_Buy'] = df['close'] + (df['ATR'] * params['atr_tp_multiplier'])

    # Dynamic leverage based on ATR vs expanding mean
    atr_avg = df['ATR'].expanding().mean()
    df['Leverage'] = np.where(df['ATR'] < atr_avg, params['leverage_max'], params['leverage_max'] // 2)

    return df


# Async versions for improved performance
async def generate_signals_async(df: pd.DataFrame) -> pd.DataFrame:
    """
    Async version of generate_signals for improved performance.
    """
    # Signal generation is CPU-bound, so run in thread pool
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, generate_signals, df)


async def calculate_risk_management_async(df: pd.DataFrame) -> pd.DataFrame:
    """
    Async version of calculate_risk_management for improved performance.
    """
    # Risk calculation is CPU-bound, so run in thread pool
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, calculate_risk_management, df)


async def process_signals_pipeline_async(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete async signal processing pipeline.
    """
    # Calculate indicators (cached, so fast)
    df = calculate_indicators(df)

    # Generate signals
    df = await generate_signals_async(df)

    # Calculate risk management
    df = await calculate_risk_management_async(df)

    return df


async def process_multiple_timeframes_async(symbol: str, timeframes: list) -> Dict[str, pd.DataFrame]:
    """
    Process signals for multiple timeframes concurrently.
    """
    from src.data_fetcher import fetch_historical_data_async

    async def process_timeframe(timeframe):
        try:
            # Fetch data for this timeframe
            df = await fetch_historical_data_async(symbol, 'kraken', timeframe, 1000)
            if df.empty:
                return timeframe, None

            # Process signals
            df_processed = await process_signals_pipeline_async(df)
            return timeframe, df_processed
        except Exception as e:
            print(f"Error processing {timeframe}: {e}")
            return timeframe, None

    # Process all timeframes concurrently
    tasks = [process_timeframe(tf) for tf in timeframes]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    data = {}
    for result in results:
        if isinstance(result, Exception):
            print(f"Task failed with exception: {result}")
            continue
        timeframe, df = result
        if df is not None:
            data[timeframe] = df

    return data
