import numpy as np
import pandas as pd
import asyncio
from typing import Dict, Any
from config.settings import CAPITAL, RISK_PER_TRADE, MTA_ENABLED, MTA_HIGHER_TIMEFRAME, MTA_DATA_LIMIT, VOLATILITY_WINDOW, TREND_LOOKBACK
from src.data_fetcher import fetch_data
from src.indicators import calculate_indicators
from src.audit_logger import audit_logger, log_signal_buy, log_signal_sell, SignalDecision, RiskAssessment

# Conditional imports for ML features
try:
    from src.ml_predictor import get_ml_predictor, is_ml_available, get_lstm_predictor, is_lstm_available, get_transformer_predictor, is_transformer_available
    ML_IMPORTS_AVAILABLE = True
except ImportError:
    print("⚠️  ML predictor imports not available (optional dependencies not installed)")
    ML_IMPORTS_AVAILABLE = False
    def get_ml_predictor(*args, **kwargs):
        return None
    def is_ml_available():
        return False
    def get_lstm_predictor(*args, **kwargs):
        return None
    def is_lstm_available():
        return False
    def get_transformer_predictor(*args, **kwargs):
        return None
    def is_transformer_available():
        return False

# Conditional imports for sentiment analysis
try:
    from src.sentiment_analyzer import get_sentiment_score, SENTIMENT_ENABLED
    SENTIMENT_IMPORTS_AVAILABLE = True
except ImportError:
    print("⚠️  Sentiment analyzer imports not available (optional dependencies not installed)")
    SENTIMENT_IMPORTS_AVAILABLE = False
    SENTIMENT_ENABLED = False
    def get_sentiment_score():
        return 0.0, 0.0

# Conditional imports for paper trading
try:
    from src.paper_trading import execute_paper_trade, update_paper_portfolio, PAPER_TRADING_ENABLED
    PAPER_TRADING_AVAILABLE = True
except ImportError:
    print("⚠️  Paper trading imports not available")
    PAPER_TRADING_AVAILABLE = False
    PAPER_TRADING_ENABLED = False
    def execute_paper_trade(*args, **kwargs):
        return None
    def update_paper_portfolio(*args, **kwargs):
        pass

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates Buy/Sell signals based on EMA crossover, RSI and BB.
    Includes optional Multi-Timeframe Analysis (MTA) for signal confirmation.
    """
    from config.settings import STRICT_SIGNALS_ENABLED, get_timeframe_params, TIMEFRAME, get_current_indicator_config
    # Don't drop NaN values here - let indicators handle their own NaN values
    # df.dropna(inplace=True)  # Remove this line
    df.reset_index(drop=True, inplace=True)  # Reset Index

    # Get current indicator configuration
    config = get_current_indicator_config()

    # Get timeframe-specific parameters
    params = get_timeframe_params(TIMEFRAME)

    # Get EMA periods from config (dynamic)
    ema_periods = config.get('ema', {}).get('periods', [9, 21])
    if len(ema_periods) >= 2:
        ema_short_col = f'EMA{ema_periods[0]}'
        ema_long_col = f'EMA{ema_periods[1]}'
    else:
        # Fallback to defaults
        ema_short_col = 'EMA9'
        ema_long_col = 'EMA21'

    # Basic signal generation - use dynamic EMA column names
    if ema_short_col in df.columns and ema_long_col in df.columns:
        df['EMA_crossover'] = np.where(df[ema_short_col].values > df[ema_long_col].values, 1, -1)
    else:
        # Fallback if columns don't exist
        df['EMA_crossover'] = np.where(df['EMA9'].values > df['EMA21'].values, 1, -1)
    
    if STRICT_SIGNALS_ENABLED:
        # Enhanced logic for crash scenarios - generate buy signals on extreme oversold conditions
        # even without bullish EMA crossover (for crash buying opportunities)
        ema_buy_condition = (df['EMA_crossover'].values == 1) & (df['RSI'].values < params['rsi_oversold']) & (df['close'].values > df['BB_lower'].values)
        crash_buy_condition = ((df['RSI'].values < 35) & (df['close'].values < df['BB_lower'].values)) | \
                             ((df['RSI'].values < 40) & (df['close'].values < df['BB_lower'].values * 1.02))  # More lenient crash condition
        
        buy_condition = ema_buy_condition | crash_buy_condition
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

    # ML Signal Enhancement
    df = apply_ml_signal_enhancement(df)

    # Sentiment Analysis Enhancement
    df = apply_sentiment_signal_enhancement(df)

    # Execute Paper Trades if enabled
    if PAPER_TRADING_ENABLED and PAPER_TRADING_AVAILABLE:
        df = execute_paper_trades(df)

    # Audit logging for signal decisions
    _log_signal_decisions(df, config, params)

    return df

def apply_multi_timeframe_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Multi-Timeframe Analysis to confirm signals with higher timeframe trends.
    Only keeps signals that are confirmed by the higher timeframe trend.
    """
    try:
        # Fetch higher timeframe data
        higher_tf_data = fetch_data(limit=MTA_DATA_LIMIT)  # Fetch recent data for higher timeframe

        # For demo purposes, we'll simulate higher timeframe analysis
        # In a real implementation, you'd need to resample the data to higher timeframe

        # Get trend from higher timeframe (simplified)
        # This would compare EMA crossover on higher timeframe
        higher_trend = 1 if higher_tf_data['close'].iloc[-1] > higher_tf_data['close'].iloc[-TREND_LOOKBACK] else -1

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

def calculate_kelly_position_size(df: pd.DataFrame, capital: float, risk_per_trade: float) -> pd.Series:
    """
    Calculate optimal position size using Kelly Criterion.

    The Kelly Criterion maximizes long-term growth by calculating the optimal fraction
    of capital to risk based on win probability and win/loss ratio.

    Formula: K = (p * (b+1) - 1) / b
    Where:
    - p = probability of winning trade
    - b = ratio of average win to average loss

    Returns fractional Kelly (50% of full Kelly) for risk management.
    """
    from config.settings import KELLY_ENABLED, KELLY_FRACTION, KELLY_LOOKBACK_TRADES, KELLY_MIN_TRADES

    if not KELLY_ENABLED:
        # Return traditional position sizing if Kelly is disabled
        return pd.Series([risk_per_trade] * len(df), index=df.index)

    # Check if we have minimum required trades for Kelly calculation
    if len(df) < KELLY_MIN_TRADES:
        # Fall back to traditional sizing if insufficient data
        return pd.Series([risk_per_trade] * len(df), index=df.index)

    try:
        # Calculate base Kelly parameters from historical trades
        base_win_prob = 0.5
        base_win_loss_ratio = 2.0

        if 'Trade_Result' in df.columns:
            trade_results = df['Trade_Result'].dropna()
            if len(trade_results) >= KELLY_MIN_TRADES:
                wins = trade_results > 0
                base_win_prob = wins.mean()

                winning_trades = trade_results[trade_results > 0]
                losing_trades = trade_results[trade_results < 0]

                if len(winning_trades) > 0 and len(losing_trades) > 0:
                    avg_win = winning_trades.mean()
                    avg_loss = abs(losing_trades.mean())
                    base_win_loss_ratio = avg_win / avg_loss

        # Vectorized Kelly calculation for each row
        kelly_sizes = pd.Series([risk_per_trade] * len(df), index=df.index)

        # Check if required indicators are available - vectorized approach
        rsi_available = df.get('RSI', pd.Series([np.nan] * len(df))).notna()
        ml_confidence_available = df.get('ML_Confidence', pd.Series([np.nan] * len(df))).notna()
        atr_available = df.get('ATR', pd.Series([np.nan] * len(df))).notna()

        # Always calculate base Kelly from historical data, then modulate with current conditions
        # Start with base win probability
        win_prob = pd.Series([base_win_prob] * len(df), index=df.index)

        # Adjust based on current market conditions (RSI) - vectorized, only if RSI available
        if rsi_available.any():
            rsi = df['RSI'].fillna(50)  # Neutral RSI if missing
            win_prob = np.where(
                rsi < 30,
                np.minimum(0.7, base_win_prob + 0.1),  # Oversold - higher win probability for longs
                np.where(
                    rsi > 70,
                    np.minimum(0.7, base_win_prob + 0.1),  # Overbought - higher win probability for shorts
                    np.where(
                        (rsi >= 45) & (rsi <= 55),
                        np.maximum(0.3, base_win_prob - 0.05),  # Neutral RSI - slightly lower win probability
                        win_prob
                    )
                )
            )

        # Adjust based on ML confidence if available - vectorized
        if ml_confidence_available.any():
            ml_confidence = df['ML_Confidence'].fillna(0)
            win_prob = win_prob * (1 - ml_confidence * 0.3) + ml_confidence * win_prob

        # Adjust win/loss ratio based on ATR (volatility) - vectorized
        win_loss_ratio = pd.Series([base_win_loss_ratio] * len(df), index=df.index)
        if atr_available.any():
            atr = df['ATR'].fillna(0)
            # Higher ATR (volatility) suggests higher risk/reward potential
            atr_adjustment = np.minimum(0.5, atr / 1000)  # Normalize ATR adjustment
            win_loss_ratio = base_win_loss_ratio * (1 + atr_adjustment)

        # Calculate full Kelly fraction - vectorized
        valid_kelly = (win_prob > 0) & (win_prob < 1) & (win_loss_ratio > 0)
        kelly_full = np.where(
            valid_kelly,
            (win_prob * (win_loss_ratio + 1) - 1) / win_loss_ratio,
            risk_per_trade
        )

        # Apply fractional Kelly for risk management
        kelly_fractional = kelly_full * KELLY_FRACTION
        # Ensure reasonable bounds
        kelly_fractional = np.clip(kelly_fractional, 0.01, 0.25)  # 1% to 25%

        kelly_sizes = kelly_fractional

        return pd.Series(kelly_sizes, index=df.index)

    except Exception as e:
        print(f"⚠️  Kelly Criterion calculation failed: {e}")
        # Fallback to traditional position sizing
        return pd.Series([risk_per_trade] * len(df), index=df.index)


def calculate_risk_management(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates position size, stop-loss, take-profit based on ATR with multipliers.
    Position size is calculated as both absolute amount and percentage of total capital.
    Optionally uses Kelly Criterion for optimal position sizing.
    Requires Buy_Signal and Sell_Signal columns.
    """
    from config.settings import get_timeframe_params, TIMEFRAME, get_current_indicator_config, KELLY_ENABLED

    # Check for required signal columns
    if 'Buy_Signal' not in df.columns or 'Sell_Signal' not in df.columns:
        raise KeyError("DataFrame must contain 'Buy_Signal' and 'Sell_Signal' columns")

    # Get current indicator configuration
    config = get_current_indicator_config()

    # Get timeframe-specific parameters
    params = get_timeframe_params(TIMEFRAME)

    # Calculate position size using Kelly Criterion if enabled
    if KELLY_ENABLED:
        print("🎯 Using Kelly Criterion for position sizing")
        df['Kelly_Fraction'] = calculate_kelly_position_size(df, CAPITAL, RISK_PER_TRADE)
        risk_fraction = df['Kelly_Fraction']
    else:
        risk_fraction = RISK_PER_TRADE

    # Calculate absolute position size with ATR multiplier
    atr_multiplier = params.get('atr_sl_multiplier', 1.0)

    # Check if ATR is available, otherwise use a default volatility measure
    if 'ATR' in df.columns and not df['ATR'].isna().all():
        df['Position_Size_Absolute'] = (CAPITAL * risk_fraction) / (df['ATR'] * atr_multiplier)
        atr_for_risk = df['ATR']
    else:
        # Fallback: use a simple volatility measure based on price changes
        price_volatility = df['close'].pct_change().rolling(window=VOLATILITY_WINDOW).std() * df['close']
        df['Position_Size_Absolute'] = (CAPITAL * risk_fraction) / (price_volatility * atr_multiplier)
        atr_for_risk = price_volatility
        print("Warning: ATR not available, using price volatility fallback")

    # Calculate the actual percentage of capital that would be at risk
    # Risk amount = Position_Size_Absolute * ATR * SL_Multiplier
    risk_amount = df['Position_Size_Absolute'] * atr_for_risk * params['atr_sl_multiplier']
    df['Position_Size_Percent'] = (risk_amount / CAPITAL) * 100

    # Calculate stop-loss and take-profit with multipliers
    df['Stop_Loss_Buy'] = df['close'] - (atr_for_risk * params['atr_sl_multiplier'])
    df['Take_Profit_Buy'] = df['close'] + (atr_for_risk * params['atr_tp_multiplier'])

    # Dynamic leverage based on ATR vs expanding mean
    if 'ATR' in df.columns and not df['ATR'].isna().all():
        atr_avg = df['ATR'].expanding().mean()
    else:
        atr_avg = atr_for_risk.expanding().mean()
    df['Leverage'] = np.where(atr_for_risk < atr_avg, params['leverage_max'], params['leverage_max'] // 2)

    # Audit logging for risk assessments
    _log_risk_assessments(df, params)

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


def _log_signal_decisions(df: pd.DataFrame, config: Dict[str, Any], params: Dict[str, Any]):
    """
    Log signal decisions for audit trail.
    Only logs the most recent signals to avoid log spam.
    """
    from config.settings import TIMEFRAME, SYMBOL
    import datetime

    try:
        # Get the last row for current signal state
        if df.empty:
            return

        last_row = df.iloc[-1]

        # Extract technical indicators - use dynamic EMA column names
        indicators = {}
        ema_periods = config.get('ema', {}).get('periods', [9, 21])
        ema_short_col = f'EMA{ema_periods[0]}' if len(ema_periods) >= 1 else 'EMA9'
        ema_long_col = f'EMA{ema_periods[1]}' if len(ema_periods) >= 2 else 'EMA21'
        indicator_cols = [ema_short_col, ema_long_col] + ['RSI', 'BB_upper', 'BB_lower', 'ATR', 'ADX']
        
        for indicator in indicator_cols:
            if indicator in last_row and not pd.isna(last_row[indicator]):
                indicators[indicator] = float(last_row[indicator])

        # Extract risk metrics
        risk_metrics = {}
        for metric in ['Position_Size_Absolute', 'Position_Size_Percent', 'Stop_Loss_Buy',
                      'Take_Profit_Buy', 'Leverage']:
            if metric in last_row and not pd.isna(last_row[metric]):
                risk_metrics[metric] = float(last_row[metric])

        # Log buy signal if present
        if 'Buy_Signal' in last_row and last_row['Buy_Signal'] == 1:
            ema_short_val = indicators.get(ema_short_col, 'N/A')
            ema_long_val = indicators.get(ema_long_col, 'N/A')
            reasoning = f"{ema_short_col}({ema_short_val}) > {ema_long_col}({ema_long_val})"
            if 'RSI' in indicators:
                reasoning += f", RSI({indicators['RSI']:.2f}) < oversold({params['rsi_oversold']})"
            if 'BB_lower' in indicators:
                reasoning += f", close({last_row['close']:.4f}) > BB_lower({indicators['BB_lower']:.4f})"

            log_signal_buy(
                symbol=SYMBOL,
                timeframe=TIMEFRAME,
                confidence=0.8,  # Default confidence, could be calculated based on signal strength
                indicators=indicators,
                risk_metrics=risk_metrics,
                reasoning=reasoning
            )

        # Log sell signal if present
        elif 'Sell_Signal' in last_row and last_row['Sell_Signal'] == 1:
            ema_short_val = indicators.get(ema_short_col, 'N/A')
            ema_long_val = indicators.get(ema_long_col, 'N/A')
            reasoning = f"{ema_short_col}({ema_short_val}) < {ema_long_col}({ema_long_val})"
            if 'RSI' in indicators:
                reasoning += f", RSI({indicators['RSI']:.2f}) > overbought({params['rsi_overbought']})"
            if 'BB_upper' in indicators:
                reasoning += f", close({last_row['close']:.4f}) < BB_upper({indicators['BB_upper']:.4f})"

            log_signal_sell(
                symbol=SYMBOL,
                timeframe=TIMEFRAME,
                confidence=0.8,  # Default confidence, could be calculated based on signal strength
                indicators=indicators,
                risk_metrics=risk_metrics,
                reasoning=reasoning
            )

    except Exception as e:
        audit_logger.log_error(
            error_type="SIGNAL_LOGGING_ERROR",
            message=f"Failed to log signal decisions: {str(e)}",
            context={"config": config, "params": params}
        )


def _log_risk_assessments(df: pd.DataFrame, params: Dict[str, Any]):
    """
    Log risk assessments for audit trail.
    Logs risk calculations for positions with signals.
    """
    from config.settings import TIMEFRAME, SYMBOL, CAPITAL, RISK_PER_TRADE
    import datetime

    try:
        # Only log for rows with signals to avoid spam - vectorized filtering
        signal_mask = (df['Buy_Signal'] == 1) | (df['Sell_Signal'] == 1)
        signal_rows = df[signal_mask]

        if len(signal_rows) == 0:
            return

        # Extract values for vectorized processing
        atr_values = signal_rows.get('ATR', 0).values
        position_size_absolute = signal_rows.get('Position_Size_Absolute', 0).values
        position_size_percent = signal_rows.get('Position_Size_Percent', 0).values
        leverage_values = signal_rows.get('Leverage', 1).values

        # Vectorized risk assessment calculations
        max_risk_limit = RISK_PER_TRADE * 100  # Convert to percentage

        # Calculate ATR fallback values
        atr_fallback = np.where(
            position_size_absolute > 0,
            position_size_absolute / (CAPITAL * RISK_PER_TRADE),
            0
        )
        final_atr_values = np.where(atr_values > 0, atr_values, atr_fallback)

        # Vectorized assessment results
        too_risky = position_size_percent > max_risk_limit * 1.5
        high_risk = (position_size_percent > max_risk_limit * 1.2) & ~too_risky
        approved = ~(too_risky | high_risk)

        assessment_results = np.full(len(signal_rows), '', dtype=object)
        assessment_results[too_risky] = "REJECTED"
        assessment_results[high_risk] = "MODIFIED"
        assessment_results[approved] = "APPROVED"

        # Create and log assessments - batch processing for performance
        max_drawdown_limit = params.get('max_drawdown_limit', 0.1)

        # Batch log risk assessments instead of individual logging for performance
        try:
            # Create summary assessment for batch logging
            total_signals = len(signal_rows)
            rejected_count = (assessment_results == "REJECTED").sum()
            modified_count = (assessment_results == "MODIFIED").sum()
            approved_count = (assessment_results == "APPROVED").sum()

            # Log summary instead of individual assessments for performance
            batch_assessment = RiskAssessment(
                timestamp=datetime.datetime.now().isoformat(),
                symbol=SYMBOL,
                timeframe=TIMEFRAME,
                capital=CAPITAL,
                risk_per_trade=RISK_PER_TRADE,
                position_size=np.mean(position_size_absolute),  # Average position size
                atr_value=np.mean(final_atr_values),  # Average ATR
                leverage=np.mean(leverage_values),  # Average leverage
                max_drawdown_limit=max_drawdown_limit,
                assessment_result=f"BATCH: {approved_count} approved, {modified_count} modified, {rejected_count} rejected"
            )

            audit_logger.log_risk_assessment(batch_assessment)

        except Exception as e:
            audit_logger.log_error(
                error_type="RISK_ASSESSMENT_BATCH_LOGGING_ERROR",
                message=f"Failed to batch log risk assessments: {str(e)}",
                context={
                    "total_signals": len(signal_rows),
                    "approved": (assessment_results == "APPROVED").sum(),
                    "modified": (assessment_results == "MODIFIED").sum(),
                    "rejected": (assessment_results == "REJECTED").sum()
                }
            )

    except Exception as e:
        audit_logger.log_error(
            error_type="RISK_LOGGING_ERROR",
            message=f"Failed to log risk assessments: {str(e)}",
            context={"params": params}
        )


def apply_ml_signal_enhancement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply ML signal enhancement to traditional trading signals.

    Uses trained ML models (traditional ML, LSTM, Transformer) to enhance or override
    traditional signals based on confidence thresholds and historical performance.
    Creates an ensemble prediction from all available models.
    """
    from config.settings import ML_ENABLED, ML_CONFIDENCE_THRESHOLD

    # Always add Signal_Source column, defaulting to TRADITIONAL
    df = df.copy()  # Avoid modifying original
    df['Signal_Source'] = 'TRADITIONAL'

    # Check if ML is enabled and available
    if not ML_ENABLED or not ML_IMPORTS_AVAILABLE:
        return df

    try:
        # Initialize available ML predictors
        predictors = []
        
        # Add traditional ML predictor if available
        if is_ml_available():
            predictor = get_ml_predictor()
            if predictor:
                predictors.append(('traditional_ml', predictor))
        
        # Add LSTM predictor if available
        if is_lstm_available():
            predictor = get_lstm_predictor()
            if predictor:
                predictors.append(('lstm', predictor))
        
        # Add Transformer predictor if available
        if is_transformer_available():
            predictor = get_transformer_predictor()
            if predictor:
                predictors.append(('transformer', predictor))
        
        # If no predictors available, return early
        if not predictors:
            return df

        # Pre-allocate arrays for vectorized operations
        n_rows = len(df)
        ensemble_signals = np.full(n_rows, 'HOLD', dtype=object)
        ensemble_confidences = np.zeros(n_rows)
        ensemble_reasons = np.full(n_rows, '', dtype=object)
        ensemble_votes = np.zeros(n_rows)

        # Process all rows at once for each model (batch processing)
        for model_name, predictor in predictors:
            try:
                # Get predictions for all rows at once (batch prediction)
                predictions = predictor.predict_signal(df, threshold=ML_CONFIDENCE_THRESHOLD)

                # Handle both single prediction and batch prediction results
                if isinstance(predictions, dict):
                    # Single prediction result - apply to all rows
                    signal = predictions['signal']
                    confidence = predictions['confidence']
                    reason = predictions.get('reason', '')

                    # Apply to all rows
                    ensemble_signals = np.where(
                        ensemble_confidences < confidence,
                        signal,
                        ensemble_signals
                    )
                    ensemble_confidences = np.maximum(ensemble_confidences, confidence)
                    ensemble_reasons = np.where(
                        ensemble_confidences == confidence,
                        reason,
                        ensemble_reasons
                    )

                    # Weight by model type (LSTM and Transformer get higher weight)
                    weight = 1.5 if model_name in ['lstm', 'transformer'] else 1.0
                    signal_score = confidence if signal == 'BUY' else (-confidence if signal == 'SELL' else 0)
                    ensemble_votes += signal_score * weight * confidence

                elif isinstance(predictions, list) and len(predictions) == n_rows:
                    # Batch prediction results - one per row
                    for idx, prediction in enumerate(predictions):
                        signal = prediction['signal']
                        confidence = prediction['confidence']
                        reason = prediction.get('reason', '')

                        # Update ensemble if this model has higher confidence
                        if confidence > ensemble_confidences[idx]:
                            ensemble_signals[idx] = signal
                            ensemble_confidences[idx] = confidence
                            ensemble_reasons[idx] = reason

                        # Weight by model type (LSTM and Transformer get higher weight)
                        weight = 1.5 if model_name in ['lstm', 'transformer'] else 1.0
                        signal_score = confidence if signal == 'BUY' else (-confidence if signal == 'SELL' else 0)
                        ensemble_votes[idx] += signal_score * weight * confidence

            except Exception as e:
                print(f"⚠️  {model_name} prediction failed: {e}")
                continue

        # Apply ensemble results to DataFrame - vectorized
        df['ML_Signal'] = ensemble_signals
        df['ML_Confidence'] = ensemble_confidences
        df['ML_Reason'] = ensemble_reasons
        df['Ensemble_Vote'] = ensemble_votes

        # Apply enhancement logic - prioritize strong traditional signals over ML
        # ML signals only used when traditional signals are weak/contradictory
        traditional_buy = df.get('Buy_Signal', 0).values == 1
        traditional_sell = df.get('Sell_Signal', 0).values == 1
        
        # ML signals
        ml_buy = (ensemble_confidences >= ML_CONFIDENCE_THRESHOLD) & (ensemble_signals == 'BUY')
        ml_sell = (ensemble_confidences >= ML_CONFIDENCE_THRESHOLD) & (ensemble_signals == 'SELL')
        
        # Prioritize traditional signals, use ML only for enhancement when no traditional signal exists
        # or when ML strongly contradicts (but give traditional signals priority for crash scenarios)
        final_buy = traditional_buy | (ml_buy & ~traditional_sell)  # Keep traditional buy, add ML buy only if no traditional sell
        final_sell = traditional_sell | (ml_sell & ~traditional_buy)  # Keep traditional sell, add ML sell only if no traditional buy
        
        # Update signals - vectorized
        df['Buy_Signal'] = final_buy.astype(int)
        df['Sell_Signal'] = final_sell.astype(int)

        # Set signal source - vectorized
        df['Signal_Source'] = np.where(
            ml_buy | ml_sell,
            'ML_ENHANCED',
            'TRADITIONAL'
        )

        # Set enhanced signal - vectorized
        df['Enhanced_Signal'] = np.where(
            ml_buy,
            'BUY',
            np.where(ml_sell, 'SELL', 'HOLD')
        )

        # Log ML enhancement completion
        enhanced_signals_count = (df['Signal_Source'] != 'TRADITIONAL').sum()
        audit_logger.log_system_event(
            event_type="ML_SIGNAL_ENHANCEMENT_COMPLETED",
            message=f"Advanced ML signal enhancement completed: {enhanced_signals_count} signals enhanced using {len(predictors)} models",
            details={
                'total_signals': len(df),
                'enhanced_signals': int(enhanced_signals_count),
                'models_used': [name for name, _ in predictors],
                'ml_available': True
            }
        )

    except Exception as e:
        audit_logger.log_error(
            error_type="ML_ENHANCEMENT_ERROR",
            message=f"ML signal enhancement failed: {str(e)}",
            context={'ml_enabled': ML_ENABLED}
        )

    return df


def apply_sentiment_signal_enhancement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply sentiment analysis enhancement to trading signals.

    Integrates sentiment scores from multiple sources (Twitter, News, Reddit)
    to enhance or filter traditional trading signals based on market psychology.
    """
    from config.settings import SENTIMENT_ENABLED, SENTIMENT_WEIGHT, SENTIMENT_THRESHOLD

    # Check if sentiment analysis is enabled and available
    if not SENTIMENT_ENABLED or not SENTIMENT_IMPORTS_AVAILABLE:
        return df

    try:
        print("🧠 Applying sentiment analysis enhancement...")

        # Get current sentiment score
        sentiment_score, sentiment_confidence = get_sentiment_score()

        # Add sentiment columns to DataFrame - vectorized initialization
        df = df.copy()  # Avoid modifying original
        df['Sentiment_Score'] = sentiment_score
        df['Sentiment_Confidence'] = sentiment_confidence
        df['Sentiment_Enhanced'] = False

        # Only enhance signals if sentiment confidence is sufficient
        if sentiment_confidence >= SENTIMENT_THRESHOLD:
            print(f"🧠 Applying sentiment enhancement (score: {sentiment_score:.3f}, confidence: {sentiment_confidence:.3f})")

            # Get original signals - vectorized
            buy_signal = df.get('Buy_Signal', 0) == 1
            sell_signal = df.get('Sell_Signal', 0) == 1
            ml_confidence = df.get('ML_Confidence', 0.5)

            # Vectorized sentiment enhancement logic
            sentiment_positive = sentiment_score > SENTIMENT_THRESHOLD
            sentiment_negative = sentiment_score < -SENTIMENT_THRESHOLD

            # Positive sentiment strengthens buy signals
            buy_enhanced = buy_signal & sentiment_positive
            enhanced_buy_confidence = np.minimum(1.0, ml_confidence + sentiment_score * SENTIMENT_WEIGHT)

            # Negative sentiment strengthens sell signals
            sell_enhanced = sell_signal & sentiment_negative
            enhanced_sell_confidence = np.minimum(1.0, ml_confidence + np.abs(sentiment_score) * SENTIMENT_WEIGHT)

            # Strong sentiment can create new signals or override weak ones
            new_buy_from_sentiment = ~buy_signal & sentiment_positive
            new_sell_from_sentiment = ~sell_signal & sentiment_negative

            # Update signals - vectorized
            df.loc[buy_enhanced, 'Sentiment_Enhanced'] = True
            df.loc[buy_enhanced, 'Enhanced_Signal'] = 'BUY_STRONG'

            df.loc[sell_enhanced, 'Sentiment_Enhanced'] = True
            df.loc[sell_enhanced, 'Enhanced_Signal'] = 'SELL_STRONG'

            df.loc[new_buy_from_sentiment, 'Buy_Signal'] = 1
            df.loc[new_buy_from_sentiment, 'Sentiment_Enhanced'] = True
            df.loc[new_buy_from_sentiment, 'Enhanced_Signal'] = 'BUY_SENTIMENT'

            df.loc[new_sell_from_sentiment, 'Sell_Signal'] = 1
            df.loc[new_sell_from_sentiment, 'Sentiment_Enhanced'] = True
            df.loc[new_sell_from_sentiment, 'Enhanced_Signal'] = 'SELL_SENTIMENT'

            # Log sentiment enhancement completion
            enhanced_count = df['Sentiment_Enhanced'].sum()
            audit_logger.log_system_event(
                event_type="SENTIMENT_SIGNAL_ENHANCEMENT_COMPLETED",
                message=f"Sentiment analysis enhancement completed: {enhanced_count} signals enhanced",
                details={
                    'total_signals': len(df),
                    'enhanced_signals': int(enhanced_count),
                    'sentiment_score': sentiment_score,
                    'sentiment_confidence': sentiment_confidence,
                    'sentiment_enabled': True
                }
            )
        else:
            print(f"🧠 Sentiment confidence too low ({sentiment_confidence:.3f} < {SENTIMENT_THRESHOLD}), skipping enhancement")
    except Exception as e:
        audit_logger.log_error(
            error_type="SENTIMENT_ENHANCEMENT_ERROR",
            message=f"Sentiment signal enhancement failed: {str(e)}",
            context={'sentiment_enabled': SENTIMENT_ENABLED}
        )

    return df


def execute_paper_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute paper trades based on generated signals.

    This function simulates trades when signals are generated, maintaining
    virtual portfolio state and tracking performance metrics.
    """
    from config.settings import SYMBOL, TIMEFRAME

    if not PAPER_TRADING_ENABLED or not PAPER_TRADING_AVAILABLE:
        return df

    try:
        print("📈 Executing paper trades for generated signals...")

        # Add paper trading columns to DataFrame - vectorized initialization
        df = df.copy()  # Avoid modifying original
        df['Paper_Trade_Executed'] = False
        df['Paper_Trade_Type'] = None
        df['Paper_Trade_Size'] = 0.0
        df['Paper_Portfolio_Value'] = 0.0

        # Get signal conditions - vectorized
        buy_signals = df.get('Buy_Signal', 0) == 1
        sell_signals = df.get('Sell_Signal', 0) == 1

        # Prepare data for vectorized processing
        close_prices = df['close'].values
        position_sizes = df.get('Position_Size_Absolute', 0).values
        timestamps = df.index if hasattr(df.index, 'values') else [pd.Timestamp.now()] * len(df)
        rsi_values = df.get('RSI', np.nan).values
        ema9_values = df.get('EMA9', np.nan).values
        ema21_values = df.get('EMA21', np.nan).values
        bb_lower_values = df.get('BB_lower', np.nan).values
        bb_upper_values = df.get('BB_upper', np.nan).values
        ml_confidence_values = df.get('ML_Confidence', 0).values
        sentiment_score_values = df.get('Sentiment_Score', 0).values

        # Pre-allocate result arrays
        n_rows = len(df)
        trade_executed = np.zeros(n_rows, dtype=bool)
        trade_types = np.full(n_rows, None, dtype=object)
        trade_sizes = np.zeros(n_rows)
        portfolio_values = np.zeros(n_rows)

        # Process buy signals - vectorized
        buy_indices = np.where(buy_signals)[0]
        for idx in buy_indices:
            try:
                trade_result = execute_paper_trade(
                    symbol=SYMBOL,
                    side='BUY',
                    price=close_prices[idx],
                    size=position_sizes[idx],
                    timestamp=timestamps[idx],
                    reason='Signal generated',
                    signal_data={
                        'rsi': rsi_values[idx],
                        'ema_short': ema9_values[idx],
                        'ema_long': ema21_values[idx],
                        'bb_lower': bb_lower_values[idx],
                        'ml_confidence': ml_confidence_values[idx],
                        'sentiment_score': sentiment_score_values[idx]
                    }
                )

                if trade_result:
                    trade_executed[idx] = True
                    trade_types[idx] = 'BUY'
                    trade_sizes[idx] = trade_result.get('size', 0)

            except Exception as e:
                audit_logger.log_error(
                    error_type="PAPER_TRADE_EXECUTION_ERROR",
                    message=f"Failed to execute paper buy trade for row {idx}: {str(e)}",
                    context={"row_index": idx}
                )
                continue

        # Process sell signals - vectorized
        sell_indices = np.where(sell_signals)[0]
        for idx in sell_indices:
            try:
                trade_result = execute_paper_trade(
                    symbol=SYMBOL,
                    side='SELL',
                    price=close_prices[idx],
                    size=position_sizes[idx],
                    timestamp=timestamps[idx],
                    reason='Signal generated',
                    signal_data={
                        'rsi': rsi_values[idx],
                        'ema_short': ema9_values[idx],
                        'ema_long': ema21_values[idx],
                        'bb_upper': bb_upper_values[idx],
                        'ml_confidence': ml_confidence_values[idx],
                        'sentiment_score': sentiment_score_values[idx]
                    }
                )

                if trade_result:
                    trade_executed[idx] = True
                    trade_types[idx] = 'SELL'
                    trade_sizes[idx] = trade_result.get('size', 0)

            except Exception as e:
                audit_logger.log_error(
                    error_type="PAPER_TRADE_EXECUTION_ERROR",
                    message=f"Failed to execute paper sell trade for row {idx}: {str(e)}",
                    context={"row_index": idx}
                )
                continue

        # Update portfolio values for all rows - vectorized
        for idx in range(n_rows):
            try:
                portfolio_value = update_paper_portfolio(current_price=close_prices[idx])
                portfolio_values[idx] = portfolio_value
            except Exception as e:
                portfolio_values[idx] = 0.0
                continue

        # Apply results to DataFrame - vectorized
        df['Paper_Trade_Executed'] = trade_executed
        df['Paper_Trade_Type'] = trade_types
        df['Paper_Trade_Size'] = trade_sizes
        df['Paper_Portfolio_Value'] = portfolio_values

        # Log paper trading execution completion
        executed_trades = trade_executed.sum()
        if executed_trades > 0:
            audit_logger.log_system_event(
                event_type="PAPER_TRADES_EXECUTED",
                message=f"Paper trading execution completed: {executed_trades} trades executed",
                details={
                    'total_signals': len(df),
                    'executed_trades': int(executed_trades),
                    'symbol': SYMBOL,
                    'timeframe': TIMEFRAME,
                    'paper_trading_enabled': True
                }
            )

    except Exception as e:
        audit_logger.log_error(
            error_type="PAPER_TRADING_EXECUTION_ERROR",
            message=f"Paper trading execution failed: {str(e)}",
            context={'paper_trading_enabled': PAPER_TRADING_ENABLED}
        )

    return df
