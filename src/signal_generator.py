import numpy as np
import pandas as pd
import asyncio
from typing import Dict, Any
from config.settings import CAPITAL, RISK_PER_TRADE, MTA_ENABLED, MTA_HIGHER_TIMEFRAME, MTA_DATA_LIMIT, VOLATILITY_WINDOW, TREND_LOOKBACK
from src.data_fetcher import fetch_data
from src.indicators import calculate_indicators
from src.audit_logger import audit_logger, log_signal_buy, log_signal_sell, SignalDecision, RiskAssessment
from src.ml_predictor import get_ml_predictor, is_ml_available, get_lstm_predictor, is_lstm_available, get_transformer_predictor, is_transformer_available

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

    # ML Signal Enhancement
    df = apply_ml_signal_enhancement(df)

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

def calculate_risk_management(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates position size, stop-loss, take-profit based on ATR with multipliers.
    Position size is calculated as both absolute amount and percentage of total capital.
    Requires Buy_Signal and Sell_Signal columns.
    """
    from config.settings import get_timeframe_params, TIMEFRAME, get_current_indicator_config

    # Check for required signal columns
    if 'Buy_Signal' not in df.columns or 'Sell_Signal' not in df.columns:
        raise KeyError("DataFrame must contain 'Buy_Signal' and 'Sell_Signal' columns")

    # Get current indicator configuration
    config = get_current_indicator_config()

    # Get timeframe-specific parameters
    params = get_timeframe_params(TIMEFRAME)

    # Calculate absolute position size with ATR multiplier
    atr_multiplier = params.get('atr_sl_multiplier', 1.0)
    
    # Check if ATR is available, otherwise use a default volatility measure
    if 'ATR' in df.columns and not df['ATR'].isna().all():
        df['Position_Size_Absolute'] = (CAPITAL * RISK_PER_TRADE) / (df['ATR'] * atr_multiplier)
        atr_for_risk = df['ATR']
    else:
        # Fallback: use a simple volatility measure based on price changes
        price_volatility = df['close'].pct_change().rolling(window=VOLATILITY_WINDOW).std() * df['close']
        df['Position_Size_Absolute'] = (CAPITAL * RISK_PER_TRADE) / (price_volatility * atr_multiplier)
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
        # Only log for rows with signals to avoid spam
        signal_rows = df[(df['Buy_Signal'] == 1) | (df['Sell_Signal'] == 1)]

        for idx, row in signal_rows.iterrows():
            try:
                # Extract ATR value for risk calculation
                atr_value = row.get('ATR', row.get('Position_Size_Absolute', 0) / (CAPITAL * RISK_PER_TRADE) if row.get('Position_Size_Absolute', 0) > 0 else 0)

                # Determine assessment result based on risk limits
                position_size_percent = row.get('Position_Size_Percent', 0)
                max_risk_limit = RISK_PER_TRADE * 100  # Convert to percentage

                if position_size_percent > max_risk_limit * 1.5:  # Too risky
                    assessment_result = "REJECTED"
                elif position_size_percent > max_risk_limit * 1.2:  # High risk but acceptable
                    assessment_result = "MODIFIED"
                else:
                    assessment_result = "APPROVED"

                assessment = RiskAssessment(
                    timestamp=datetime.datetime.now().isoformat(),
                    symbol=SYMBOL,
                    timeframe=TIMEFRAME,
                    capital=CAPITAL,
                    risk_per_trade=RISK_PER_TRADE,
                    position_size=row.get('Position_Size_Absolute', 0),
                    atr_value=atr_value,
                    leverage=row.get('Leverage', 1),
                    max_drawdown_limit=params.get('max_drawdown_limit', 0.1),
                    assessment_result=assessment_result
                )

                audit_logger.log_risk_assessment(assessment)

            except Exception as e:
                audit_logger.log_error(
                    error_type="RISK_ASSESSMENT_LOGGING_ERROR",
                    message=f"Failed to log risk assessment for row {idx}: {str(e)}",
                    context={"row_data": row.to_dict()}
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
    from src.ml_predictor import get_ml_predictor, is_ml_available, get_lstm_predictor, is_lstm_available, get_transformer_predictor, is_transformer_available

    # Check if ML is enabled and available
    if not ML_ENABLED:
        return df

    try:
        # Initialize model predictors and predictions
        predictors = []
        predictions = []

        # Get traditional ML predictor
        if is_ml_available():
            ml_predictor = get_ml_predictor()
            if ml_predictor and ml_predictor.is_trained:
                predictors.append(('traditional_ml', ml_predictor))

        # Get LSTM predictor
        if is_lstm_available():
            lstm_predictor = get_lstm_predictor()
            if lstm_predictor and lstm_predictor.is_trained:
                predictors.append(('lstm', lstm_predictor))

        # Get Transformer predictor
        if is_transformer_available():
            transformer_predictor = get_transformer_predictor()
            if transformer_predictor and transformer_predictor.is_trained:
                predictors.append(('transformer', transformer_predictor))

        # If no ML models are available, return original dataframe
        if not predictors:
            return df

        print(f"🔬 Using {len(predictors)} ML models for signal enhancement: {[name for name, _ in predictors]}")

        # Add ML signal columns
        df['ML_Signal'] = 'HOLD'
        df['ML_Confidence'] = 0.0
        df['ML_Reason'] = ''
        df['Enhanced_Signal'] = 'HOLD'
        df['Signal_Source'] = 'TRADITIONAL'
        df['Ensemble_Vote'] = 0.0  # Weighted ensemble score

        # Process each row for ML prediction
        for idx, row in df.iterrows():
            try:
                # Create single-row DataFrame for prediction
                row_df = pd.DataFrame([row])

                # Collect predictions from all models
                model_predictions = []
                ensemble_score = 0.0
                total_weight = 0.0

                for model_name, predictor in predictors:
                    try:
                        # Get prediction from this model
                        prediction = predictor.predict_signal(row_df, threshold=ML_CONFIDENCE_THRESHOLD)

                        model_predictions.append({
                            'model': model_name,
                            'signal': prediction['signal'],
                            'confidence': prediction['confidence'],
                            'reason': prediction.get('reason', '')
                        })

                        # Convert signal to numerical score for ensemble
                        signal_score = 0.0
                        if prediction['signal'] == 'BUY':
                            signal_score = prediction['confidence']
                        elif prediction['signal'] == 'SELL':
                            signal_score = -prediction['confidence']

                        # Weight by model type (LSTM and Transformer get higher weight for time-series)
                        weight = 1.0
                        if model_name in ['lstm', 'transformer']:
                            weight = 1.5  # Higher weight for deep learning models

                        ensemble_score += signal_score * weight * prediction['confidence']
                        total_weight += weight * prediction['confidence']

                    except Exception as e:
                        print(f"⚠️  {model_name} prediction failed for row {idx}: {e}")
                        continue

                # Calculate final ensemble prediction
                if total_weight > 0:
                    ensemble_score /= total_weight

                    # Determine ensemble signal
                    if ensemble_score > ML_CONFIDENCE_THRESHOLD:
                        ensemble_signal = 'BUY'
                        ensemble_confidence = min(abs(ensemble_score), 1.0)
                    elif ensemble_score < -ML_CONFIDENCE_THRESHOLD:
                        ensemble_signal = 'SELL'
                        ensemble_confidence = min(abs(ensemble_score), 1.0)
                    else:
                        ensemble_signal = 'HOLD'
                        ensemble_confidence = 0.0
                else:
                    ensemble_signal = 'HOLD'
                    ensemble_confidence = 0.0

                # Determine traditional signal
                traditional_signal = 'HOLD'
                if row.get('Buy_Signal', 0) == 1:
                    traditional_signal = 'BUY'
                elif row.get('Sell_Signal', 0) == 1:
                    traditional_signal = 'SELL'

                # Create ensemble prediction dict for compatibility
                ensemble_prediction = {
                    'signal': ensemble_signal,
                    'confidence': ensemble_confidence,
                    'reason': f'Ensemble prediction from {len(model_predictions)} models'
                }

                # Use traditional ML predictor for enhancement logic (if available)
                if predictors and predictors[0][0] == 'traditional_ml':
                    ml_predictor = predictors[0][1]
                    enhanced = ml_predictor.enhance_signal(
                        traditional_signal,
                        ensemble_prediction,
                        confidence_threshold=ML_CONFIDENCE_THRESHOLD,
                        df=df
                    )
                else:
                    # Simple enhancement logic if no traditional ML
                    enhanced = {
                        'signal': ensemble_signal if ensemble_confidence > ML_CONFIDENCE_THRESHOLD else traditional_signal,
                        'source': 'ENSEMBLE' if ensemble_confidence > ML_CONFIDENCE_THRESHOLD else 'TRADITIONAL'
                    }

                # Update DataFrame
                df.at[idx, 'ML_Signal'] = ensemble_signal
                df.at[idx, 'ML_Confidence'] = ensemble_confidence
                df.at[idx, 'ML_Reason'] = ensemble_prediction['reason']
                df.at[idx, 'Enhanced_Signal'] = enhanced['signal']
                df.at[idx, 'Signal_Source'] = enhanced['source']
                df.at[idx, 'Ensemble_Vote'] = ensemble_score

                # Override original signals if enhanced
                if enhanced['source'] in ['ENSEMBLE', 'ML_ENHANCED']:
                    if enhanced['signal'] == 'BUY':
                        df.at[idx, 'Buy_Signal'] = 1
                        df.at[idx, 'Sell_Signal'] = 0
                    elif enhanced['signal'] == 'SELL':
                        df.at[idx, 'Buy_Signal'] = 0
                        df.at[idx, 'Sell_Signal'] = 1
                    else:
                        df.at[idx, 'Buy_Signal'] = 0
                        df.at[idx, 'Sell_Signal'] = 0

            except Exception as e:
                # Log error but continue processing
                audit_logger.log_error(
                    error_type="ML_SIGNAL_ENHANCEMENT_ERROR",
                    message=f"Failed to enhance signal for row {idx}: {str(e)}",
                    context={"row_data": row.to_dict()}
                )
                continue

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
