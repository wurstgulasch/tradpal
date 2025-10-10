import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from config.settings import CAPITAL, RISK_PER_TRADE, OUTPUT_FILE, SYMBOL, EXCHANGE, TIMEFRAME
from src.data_fetcher import fetch_historical_data
from src.indicators import calculate_indicators
from src.signal_generator import generate_signals, calculate_risk_management

# Import parallel backtesting if available
try:
    from src.parallel_backtester import run_parallel_backtests as _run_parallel_backtests
    PARALLEL_BACKTESTING_AVAILABLE = True
except ImportError:
    PARALLEL_BACKTESTING_AVAILABLE = False

class Backtester:
    """
    Historical backtesting module for trading strategies.
    Calculates performance metrics like win rate, CAGR, drawdown, etc.
    """

    def __init__(self, symbol=SYMBOL, exchange=EXCHANGE, timeframe=TIMEFRAME,
                 start_date=None, end_date=None, initial_capital=10000):
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe

        # Convert date strings to datetime objects
        if isinstance(start_date, str):
            self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        else:
            self.start_date = start_date or (datetime.now() - timedelta(days=30))

        if isinstance(end_date, str):
            self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            self.end_date = end_date or datetime.now()

        # Validate date range
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")

        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trades = []
        self.portfolio_values = []

    def run_backtest(self, df=None, strategy='traditional', symbol=None, timeframe=None,
                     initial_capital=None, commission=0.001):
        """
        Run historical backtest with specified strategy.

        Args:
            df: DataFrame with data (if None, fetches data)
            strategy: Strategy type ('traditional', 'ml_enhanced', 'lstm_enhanced', 'transformer_enhanced')
            symbol: Trading symbol
            timeframe: Timeframe
            initial_capital: Initial capital
            commission: Commission per trade

        Returns:
            Dictionary with backtest results and metrics
        """
        try:
            # Update instance variables if provided
            if symbol:
                self.symbol = symbol
            if timeframe:
                self.timeframe = timeframe
            if initial_capital:
                self.initial_capital = initial_capital
                self.current_capital = initial_capital

            self.commission = commission

            print(f"Running {strategy} backtest for {self.symbol} on {self.timeframe} timeframe")
            print(f"Period: {self.start_date.date()} to {self.end_date.date()}")

            # Fetch or use provided data
            if df is None:
                data = self._fetch_data()
                if isinstance(data, str):
                    return {"success": False, "error": f"Data fetch failed: {data}"}
                if data.empty:
                    return {"success": False, "error": "No data available for backtest period"}
            else:
                data = df.copy()

            # Calculate indicators and signals based on strategy
            if strategy == 'traditional':
                data = self._prepare_traditional_signals(data)
            elif strategy == 'ml_enhanced':
                data = self._prepare_ml_enhanced_signals(data)
            elif strategy == 'lstm_enhanced':
                data = self._prepare_lstm_enhanced_signals(data)
            elif strategy == 'transformer_enhanced':
                data = self._prepare_transformer_enhanced_signals(data)
            else:
                return {"success": False, "error": f"Unknown strategy: {strategy}"}

            # Simulate trades
            trades = self._simulate_trades(data)
            self.trades = trades

            # Calculate performance metrics
            metrics = self._calculate_metrics()

            # Add strategy info
            metrics['strategy'] = strategy
            metrics['total_return_pct'] = metrics.get('return_pct', 0)

            return {
                "success": True,
                "metrics": metrics,
                "trades": trades,
                "trades_count": len(trades)
            }

        except Exception as e:
            error_msg = f"Backtest failed: {str(e)}"
            print(f"❌ {error_msg}")
            return {"success": False, "error": error_msg}

    def _fetch_data(self):
        """Fetch historical data for backtest period."""
        # Calculate limit based on timeframe and date range
        days = (self.end_date - self.start_date).days
        if self.timeframe == '1m':
            limit = min(days * 24 * 60, 50000)  # Increased limit for 1m
        elif self.timeframe == '5m':
            limit = min(days * 24 * 12, 50000)
        elif self.timeframe == '15m':
            limit = min(days * 24 * 4, 50000)
        elif self.timeframe == '1h':
            limit = min(days * 24, 50000)
        elif self.timeframe == '4h':
            limit = min(days * 6, 50000)
        elif self.timeframe == '1d':
            limit = min(days, 50000)
        else:
            limit = 10000  # Default

        # Ensure limit is a Python int, not numpy int
        limit = int(limit)

        return fetch_historical_data(
            symbol=self.symbol,
            exchange_name=self.exchange,
            timeframe=self.timeframe,
            limit=limit,
            start_date=self.start_date
        )

    def _simulate_trades(self, data):
        """
        Simulate trades based on signals and calculate P&L using vectorized operations.
        """
        if data.empty:
            return []

        # Drop rows with NaN ATR values to avoid invalid position sizes
        if 'ATR' in data.columns:
            data = data.dropna(subset=['ATR'])

        # Ensure we have required columns
        required_cols = ['close', 'Buy_Signal', 'Sell_Signal']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data missing required columns: {required_cols}")

        # Create position signals using vectorized operations
        buy_signals = (data['Buy_Signal'] == 1)
        sell_signals = (data['Sell_Signal'] == 1)

        # Calculate entry and exit points
        position_changes = pd.Series(0, index=data.index)

        # Long entries
        position_changes[buy_signals] = 1

        # Short entries (overwrite if both signals occur)
        position_changes[sell_signals] = -1

        # Calculate cumulative position
        positions = position_changes.cumsum()

        # Find trade entries and exits
        trades = []
        current_position = 0
        entry_idx = None
        entry_price = None

        for idx, (timestamp, row) in enumerate(data.iterrows()):
            new_position = positions.iloc[idx]

            # Position entry
            if current_position == 0 and new_position != 0:
                entry_idx = idx
                entry_price = row['close']
                current_position = new_position
                position_size = row.get('Position_Size_Absolute', 1000)

                trade = {
                    'type': 'buy' if current_position == 1 else 'sell',
                    'entry_time': timestamp,
                    'entry_price': entry_price,
                    'position_size': position_size,
                    'stop_loss': row.get('Stop_Loss_Buy' if current_position == 1 else 'Stop_Loss_Sell', entry_price * (0.98 if current_position == 1 else 1.02)),
                    'take_profit': row.get('Take_Profit_Buy' if current_position == 1 else 'Take_Profit_Sell', entry_price * (1.02 if current_position == 1 else 0.98)),
                    'exit_time': None,
                    'exit_price': None,
                    'pnl': 0,
                    'status': 'open'
                }
                trades.append(trade)

            # Position exit or change
            elif current_position != 0 and new_position != current_position:
                if entry_idx is not None:
                    exit_price = row['close']
                    exit_time = timestamp

                    # Calculate P&L
                    if current_position == 1:  # Long position
                        pnl = (exit_price - entry_price) * trades[-1]['position_size'] / entry_price
                    else:  # Short position
                        pnl = (entry_price - exit_price) * trades[-1]['position_size'] / entry_price

                    # Update trade
                    trades[-1].update({
                        'exit_time': exit_time,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'status': 'closed'
                    })

                    # Update capital
                    self.current_capital += pnl

                # Handle position change
                if new_position == 0:
                    current_position = 0
                    entry_idx = None
                else:
                    # Position reversal - start new trade
                    current_position = new_position
                    entry_idx = idx
                    entry_price = row['close']

                    position_size = row.get('Position_Size_Absolute', 1000)
                    trade = {
                        'type': 'buy' if current_position == 1 else 'sell',
                        'entry_time': timestamp,
                        'entry_price': entry_price,
                        'position_size': position_size,
                        'stop_loss': row.get('Stop_Loss_Buy' if current_position == 1 else 'Stop_Loss_Sell', entry_price * (0.98 if current_position == 1 else 1.02)),
                        'take_profit': row.get('Take_Profit_Buy' if current_position == 1 else 'Take_Profit_Sell', entry_price * (1.02 if current_position == 1 else 0.98)),
                        'exit_time': None,
                        'exit_price': None,
                        'pnl': 0,
                        'status': 'open'
                    }
                    trades.append(trade)

        # Close any remaining open positions at the end
        if current_position != 0 and trades and trades[-1]['status'] == 'open':
            final_price = data.iloc[-1]['close']
            final_time = data.index[-1]

            if current_position == 1:
                pnl = (final_price - entry_price) * trades[-1]['position_size'] / entry_price
            else:
                pnl = (entry_price - final_price) * trades[-1]['position_size'] / entry_price

            trades[-1].update({
                'exit_time': final_time,
                'exit_price': final_price,
                'pnl': pnl,
                'status': 'closed'
            })

            self.current_capital += pnl

        return trades

    def _calculate_metrics(self):
        """
        Calculate comprehensive performance metrics using vectorized operations.
        """
        if not self.trades:
            return {"error": "No trades executed during backtest"}

        closed_trades = [t for t in self.trades if t['status'] == 'closed']

        if not closed_trades:
            return {"error": "No closed trades to analyze"}

        # Convert to DataFrame for vectorized operations
        trades_df = pd.DataFrame(closed_trades)

        # Basic metrics using vectorized operations
        total_trades = len(trades_df)
        winning_trades = (trades_df['pnl'] > 0).sum()
        losing_trades = (trades_df['pnl'] <= 0).sum()

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0

        # Risk metrics using vectorized operations
        returns = trades_df['pnl'] / self.initial_capital
        sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

        # Drawdown calculation using vectorized operations
        capital_series = self.initial_capital + trades_df['pnl'].cumsum()
        peak = capital_series.expanding().max()
        drawdown = (peak - capital_series) / peak * 100
        max_drawdown = drawdown.max()

        # CAGR calculation
        days = (self.end_date - self.start_date).days
        if days > 0:
            cagr = ((self.current_capital / self.initial_capital) ** (365 / days) - 1) * 100
        else:
            cagr = 0

        # Profit factor
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

        return {
            'total_trades': total_trades,
            'winning_trades': int(winning_trades),
            'losing_trades': int(losing_trades),
            'win_rate': round(win_rate, 2),
            'total_pnl': round(total_pnl, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else float('inf'),
            'max_drawdown': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'cagr': round(cagr, 2),
            'final_capital': round(self.current_capital, 2),
            'return_pct': round((self.current_capital / self.initial_capital - 1) * 100, 2)
        }

    def _save_results(self, metrics):
        """
        Save backtest results to JSON file.
        """
        results = {
            'backtest_info': {
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'start_date': self.start_date.isoformat(),
                'end_date': self.end_date.isoformat(),
                'initial_capital': self.initial_capital
            },
            'metrics': metrics,
            'trades': self.trades,
            'portfolio_values': self.portfolio_values
        }

        output_file = OUTPUT_FILE.replace('.json', '_backtest.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4, default=str)

        print(f"Backtest results saved to {output_file}")

    def _prepare_traditional_signals(self, data):
        """Prepare data with traditional indicators and signals."""
        # Calculate indicators and signals (skip if already processed)
        required_cols = ['EMA9', 'EMA21', 'RSI', 'BB_upper', 'BB_middle', 'BB_lower', 'ATR', 'Buy_Signal', 'Sell_Signal', 'Position_Size_Absolute']
        if not all(col in data.columns for col in required_cols):
            data = calculate_indicators(data)
            data = generate_signals(data)
            data = calculate_risk_management(data)
        return data

    def _prepare_ml_enhanced_signals(self, data):
        """Prepare data with ML-enhanced signals."""
        from src.ml_predictor import get_ml_predictor, is_ml_available

        # First prepare traditional signals
        data = self._prepare_traditional_signals(data)

        # Add ML enhancement if available
        if is_ml_available():
            predictor = get_ml_predictor(symbol=self.symbol, timeframe=self.timeframe)
            if predictor and predictor.is_trained:
                print("🤖 Applying ML signal enhancement...")
                data = self._apply_ml_enhancement(data, predictor, 'ml_enhanced')
            else:
                print("⚠️  ML predictor not available or not trained, using traditional signals")

        return data

    def _prepare_lstm_enhanced_signals(self, data):
        """Prepare data with LSTM-enhanced signals."""
        from src.ml_predictor import get_lstm_predictor, is_lstm_available

        # First prepare traditional signals
        data = self._prepare_traditional_signals(data)

        # Add LSTM enhancement if available
        if is_lstm_available():
            predictor = get_lstm_predictor(symbol=self.symbol, timeframe=self.timeframe)
            if predictor and predictor.is_trained:
                print("🧠 Applying LSTM signal enhancement...")
                data = self._apply_ml_enhancement(data, predictor, 'lstm_enhanced')
            else:
                print("⚠️  LSTM predictor not available or not trained, using traditional signals")

        return data

    def _prepare_transformer_enhanced_signals(self, data):
        """Prepare data with Transformer-enhanced signals."""
        from src.ml_predictor import get_transformer_predictor, is_transformer_available

        # First prepare traditional signals
        data = self._prepare_traditional_signals(data)

        # Add Transformer enhancement if available
        if is_transformer_available():
            predictor = get_transformer_predictor(symbol=self.symbol, timeframe=self.timeframe)
            if predictor and predictor.is_trained:
                print("🔄 Applying Transformer signal enhancement...")
                data = self._apply_ml_enhancement(data, predictor, 'transformer_enhanced')
            else:
                print("⚠️  Transformer predictor not available or not trained, using traditional signals")

        return data

    def _apply_ml_enhancement(self, data, predictor, strategy_name):
        """Apply ML enhancement to signals."""
        from config.settings import ML_CONFIDENCE_THRESHOLD

        # Add ML signal columns
        data['ML_Signal'] = 'HOLD'
        data['ML_Confidence'] = 0.0
        data['ML_Reason'] = ''
        data['Enhanced_Signal'] = 'HOLD'
        data['Signal_Source'] = 'TRADITIONAL'

        print(f"🔬 Processing {len(data)} rows for {strategy_name}...")

        # Process each row for ML prediction
        for idx, row in data.iterrows():
            try:
                # Create single-row DataFrame for prediction
                row_df = pd.DataFrame([row])

                # Get prediction from the model
                prediction = predictor.predict_signal(row_df, threshold=ML_CONFIDENCE_THRESHOLD)

                # Update DataFrame
                data.at[idx, 'ML_Signal'] = prediction['signal']
                data.at[idx, 'ML_Confidence'] = prediction['confidence']
                data.at[idx, 'ML_Reason'] = prediction.get('reason', '')
                data.at[idx, 'Signal_Source'] = strategy_name.upper()

                # Determine traditional signal
                traditional_signal = 'HOLD'
                if row.get('Buy_Signal', 0) == 1:
                    traditional_signal = 'BUY'
                elif row.get('Sell_Signal', 0) == 1:
                    traditional_signal = 'SELL'

                # Enhancement logic: use ML if confidence is high enough
                if prediction['confidence'] >= ML_CONFIDENCE_THRESHOLD:
                    enhanced_signal = prediction['signal']
                    source = strategy_name.upper()
                else:
                    enhanced_signal = traditional_signal
                    source = 'TRADITIONAL'

                data.at[idx, 'Enhanced_Signal'] = enhanced_signal
                data.at[idx, 'Signal_Source'] = source

                # Override original signals if enhanced
                if source == strategy_name.upper():
                    if enhanced_signal == 'BUY':
                        data.at[idx, 'Buy_Signal'] = 1
                        data.at[idx, 'Sell_Signal'] = 0
                    elif enhanced_signal == 'SELL':
                        data.at[idx, 'Buy_Signal'] = 0
                        data.at[idx, 'Sell_Signal'] = 1
                    else:
                        data.at[idx, 'Buy_Signal'] = 0
                        data.at[idx, 'Sell_Signal'] = 0

            except Exception as e:
                # On prediction error, keep traditional signals
                continue

        # Count enhanced signals
        enhanced_count = (data['Signal_Source'] == strategy_name.upper()).sum()
        print(f"✅ Applied {strategy_name} to {enhanced_count} signals")

        return data

def run_backtest(symbol=SYMBOL, timeframe=TIMEFRAME, start_date=None, end_date=None):
    """
    Convenience function to run a backtest.
    """
    backtester = Backtester(symbol, 'kraken', timeframe, start_date, end_date)
    metrics = backtester.run_backtest()
    return {
        'backtest_results': metrics,
        'trades': pd.DataFrame(backtester.trades)
    }

def calculate_performance_metrics(trades_df, initial_capital=10000):
    """
    Calculate comprehensive performance metrics from trades DataFrame using vectorized operations.
    """
    if trades_df.empty:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'final_capital': initial_capital,
            'return_pct': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }

    # Ensure we have the required columns
    required_cols = ['entry_price', 'exit_price', 'position_size', 'direction']
    if not all(col in trades_df.columns for col in required_cols):
        raise ValueError(f"Trades DataFrame missing required columns: {required_cols}")

    # Calculate P&L for each trade using vectorized operations
    trades_df = trades_df.copy()

    # Vectorized P&L calculation
    long_trades = trades_df['direction'] == 'long'
    short_trades = trades_df['direction'] == 'short'

    pnl_long = (trades_df.loc[long_trades, 'exit_price'] - trades_df.loc[long_trades, 'entry_price']) * \
               trades_df.loc[long_trades, 'position_size'] / trades_df.loc[long_trades, 'entry_price']

    pnl_short = (trades_df.loc[short_trades, 'entry_price'] - trades_df.loc[short_trades, 'exit_price']) * \
                trades_df.loc[short_trades, 'position_size'] / trades_df.loc[short_trades, 'entry_price']

    trades_df['pnl'] = 0.0
    trades_df.loc[long_trades, 'pnl'] = pnl_long
    trades_df.loc[short_trades, 'pnl'] = pnl_short

    # Basic metrics using vectorized operations
    total_trades = len(trades_df)
    winning_trades = (trades_df['pnl'] > 0).sum()
    losing_trades = (trades_df['pnl'] <= 0).sum()
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

    total_pnl = trades_df['pnl'].sum()
    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    final_capital = initial_capital + total_pnl
    return_pct = (final_capital / initial_capital - 1) * 100

    # Risk metrics using vectorized operations
    returns = trades_df['pnl'] / initial_capital
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    # Max drawdown calculation using vectorized operations
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    max_drawdown_percentage = abs(max_drawdown)

    return {
        'total_trades': total_trades,
        'winning_trades': int(winning_trades),
        'losing_trades': int(losing_trades),
        'win_rate': round(win_rate, 2),
        'total_pnl': round(total_pnl, 2),
        'gross_profit': round(gross_profit, 2),
        'gross_loss': round(gross_loss, 2),
        'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else float('inf'),
        'final_capital': round(final_capital, 2),
        'return_pct': round(return_pct, 2),
        'max_drawdown': round(abs(max_drawdown), 2),
        'max_drawdown_percentage': round(max_drawdown_percentage, 2),
        'sharpe_ratio': round(sharpe_ratio, 2)
    }

def simulate_trades(data):
    """
    Simulate trades based on signals in the data.
    Returns a DataFrame of executed trades.
    """
    if data.empty:
        return pd.DataFrame()

    # Check for required columns
    required_cols = ['close']
    if not all(col in data.columns for col in required_cols):
        raise KeyError(f"Data must contain required columns: {required_cols}")

    # Check for signal columns (they can be missing, but if present should be valid)
    signal_cols = ['Buy_Signal', 'Sell_Signal']
    if not any(col in data.columns for col in signal_cols):
        raise KeyError(f"Data must contain at least one signal column: {signal_cols}")

    trades = []
    position = 0  # 0 = no position, 1 = long, -1 = short
    entry_price = 0
    entry_time = None
    position_size = 1000  # Default position size

    for idx, row in data.iterrows():
        current_time = idx
        current_price = row['close']

        # Check for entry signals
        if position == 0:
            if row.get('Buy_Signal', 0) == 1:
                position = 1
                entry_price = current_price
                entry_time = current_time
                position_size = row.get('Position_Size_Absolute', 1000)

                # Store entry trade info
                entry_trade = {
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'position_size': position_size,
                    'direction': 'long'
                }
                trades.append(entry_trade)

            elif row.get('Sell_Signal', 0) == 1:
                position = -1
                entry_price = current_price
                entry_time = current_time
                position_size = row.get('Position_Size_Absolute', 1000)

                # Store entry trade info
                entry_trade = {
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'position_size': position_size,
                    'direction': 'short'
                }
                trades.append(entry_trade)

        # Check for exit conditions (simplified - exit on opposite signal or stop loss/take profit)
        elif position != 0:
            exit_signal = False
            exit_price = current_price
            exit_reason = ''

            if position == 1:  # Long position
                # Check stop loss (if low <= stop loss)
                stop_loss = row.get('Stop_Loss_Buy', row.get('Stop_Loss', current_price * 0.98))
                if row['low'] <= stop_loss:
                    exit_signal = True
                    exit_price = stop_loss
                    exit_reason = 'stop_loss'
                # Check take profit (if high >= take profit)
                take_profit = row.get('Take_Profit_Buy', row.get('Take_Profit', current_price * 1.02))
                if row['high'] >= take_profit:
                    exit_signal = True
                    exit_price = take_profit
                    exit_reason = 'take_profit'
                # Check sell signal
                if row.get('Sell_Signal', 0) == 1:
                    exit_signal = True
                    exit_reason = 'sell_signal'

            else:  # Short position
                # Check stop loss (if high >= stop loss)
                stop_loss = row.get('Stop_Loss_Sell', row.get('Stop_Loss', current_price * 1.02))
                if row['high'] >= stop_loss:
                    exit_signal = True
                    exit_price = stop_loss
                    exit_reason = 'stop_loss'
                # Check take profit (if low <= take profit)
                take_profit = row.get('Take_Profit_Sell', row.get('Take_Profit', current_price * 0.98))
                if row['low'] <= take_profit:
                    exit_signal = True
                    exit_price = take_profit
                    exit_reason = 'take_profit'
                # Check buy signal
                if row.get('Buy_Signal', 0) == 1:
                    exit_signal = True
                    exit_reason = 'buy_signal'

            if exit_signal:
                # Update the last trade with exit info
                if trades:
                    trades[-1].update({
                        'exit_time': current_time,
                        'exit_price': exit_price,
                        'exit_reason': exit_reason
                    })

                # Reset position
                position = 0

    # Convert to DataFrame and filter only completed trades
    if trades:
        trades_df = pd.DataFrame(trades)
        # Only return completed trades (those with exit info)
        if 'exit_time' in trades_df.columns:
            completed_trades = trades_df.dropna(subset=['exit_time']).copy()
            # Calculate P&L for completed trades
            completed_trades['pnl'] = 0.0
            for idx, trade in completed_trades.iterrows():
                if trade['direction'] == 'long':
                    pnl = (trade['exit_price'] - trade['entry_price']) * trade['position_size'] / trade['entry_price']
                else:  # short
                    pnl = (trade['entry_price'] - trade['exit_price']) * trade['position_size'] / trade['entry_price']
                completed_trades.loc[idx, 'pnl'] = pnl
            return completed_trades
        else:
            return trades_df  # Return all trades if no exits
    else:
        return pd.DataFrame()
def run_walk_forward_backtest(parameter_grid: dict, evaluation_metric: str = 'sharpe_ratio',
                             symbol: str = SYMBOL, timeframe: str = TIMEFRAME) -> dict:
    """
    Run walk-forward backtest analysis using the WalkForwardOptimizer.

    Args:
        parameter_grid: Dictionary of parameters to optimize
        evaluation_metric: Metric to use for evaluation ('sharpe_ratio', 'win_rate', 'total_return')
        symbol: Trading symbol
        timeframe: Timeframe for analysis

    Returns:
        Dictionary with optimization results and final backtest
    """
    try:
        from .walk_forward_optimizer import WalkForwardOptimizer

        # Create optimizer instance
        optimizer = WalkForwardOptimizer(symbol=symbol, timeframe=timeframe)

        # Create sample data for demonstration (in real usage, this would be historical data)
        dates = pd.date_range(start='2024-01-01', periods=2000, freq='1h')
        np.random.seed(42)
        df = pd.DataFrame({
            'timestamp': dates,
            'open': 50000 + np.random.normal(0, 1000, 2000),
            'high': 51000 + np.random.normal(0, 1000, 2000),
            'low': 49000 + np.random.normal(0, 1000, 2000),
            'close': 50000 + np.random.normal(0, 1000, 2000),
            'volume': np.random.randint(1000, 10000, 2000)
        })

        # Add trend for more realistic data
        trend = np.linspace(0, 5000, 2000)
        df['close'] = df['close'] + trend

        # Run walk-forward optimization
        results = optimizer.optimize_strategy_parameters(
            df=df,
            parameter_grid=parameter_grid,
            evaluation_metric=evaluation_metric,
            initial_train_size=1000,
            test_size=200,
            step_size=100
        )

        if results.get('success'):
            # Run final backtest with best parameters
            best_params = results.get('best_parameters', {})
            final_backtest = run_backtest(symbol=symbol, timeframe=timeframe)

            return {
                'optimization_results': results,
                'final_backtest': {
                    'metrics': final_backtest.get('metrics', {}),
                    'trades': final_backtest.get('trades', [])
                }
            }
        else:
            return {
                'error': results.get('error', 'Optimization failed')
            }

    except Exception as e:
        return {
            'error': f'Walk-forward backtest failed: {str(e)}'
        }


def run_multi_symbol_backtest(symbols, exchange=EXCHANGE, timeframe=TIMEFRAME,
                              start_date=None, end_date=None, initial_capital=CAPITAL,
                              max_workers=None):
    """
    Run backtests for multiple symbols in parallel.
    
    Args:
        symbols: List of trading symbols (e.g., ['BTC/USDT', 'ETH/USDT'])
        exchange: Exchange name
        timeframe: Timeframe for backtesting
        start_date: Start date for backtest period
        end_date: End date for backtest period
        initial_capital: Initial capital for each backtest
        max_workers: Maximum number of worker processes (None = auto)
        
    Returns:
        Dictionary with aggregated results for all symbols
    """
    if not PARALLEL_BACKTESTING_AVAILABLE:
        return {
            'error': 'Parallel backtesting not available. Install required dependencies.'
        }
    
    try:
        results = _run_parallel_backtests(
            symbols=symbols,
            exchange=exchange,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            max_workers=max_workers
        )
        return results
    except Exception as e:
        return {
            'error': f'Multi-symbol backtest failed: {str(e)}'
        }

def run_multi_model_backtest(symbol=SYMBOL, exchange=EXCHANGE, timeframe=TIMEFRAME,
                            start_date=None, end_date=None, initial_capital=10000,
                            models_to_test=None, max_workers=None):
    """
    Run parallel backtests for multiple ML models and compare their performance.
    
    Args:
        symbol: Trading symbol (e.g., 'BTC/USDT')
        exchange: Exchange name
        timeframe: Timeframe for backtesting
        start_date: Start date for backtest period
        end_date: End date for backtest period
        initial_capital: Initial capital for each backtest
        models_to_test: List of model types to test ('traditional_ml', 'lstm', 'transformer', 'ensemble')
                        If None, tests all available models
        max_workers: Maximum number of parallel workers
        
    Returns:
        Dictionary with comparison results and best performing model
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time
    
    # Available model types
    available_models = ['traditional_ml', 'lstm', 'transformer', 'ensemble']
    
    if models_to_test is None:
        models_to_test = available_models
    else:
        # Validate model types
        models_to_test = [m for m in models_to_test if m in available_models]
        if not models_to_test:
            return {"error": "No valid models specified for testing"}
    
    # Filter to only trained models
    trained_models = []
    for model_type in models_to_test:
        if model_type == 'ensemble':
            # Ensemble is always available (uses multiple models)
            trained_models.append(model_type)
        elif _is_model_trained(model_type):
            trained_models.append(model_type)
        else:
            print(f"⚠️  Skipping {model_type} - model not trained")
    
    if not trained_models:
        return {"error": "No trained models available for testing"}
    
    print(f"🚀 Starting multi-model backtest for {len(trained_models)} models: {trained_models}")
    print(f"📊 Testing on {symbol} {timeframe} from {start_date} to {end_date}")
    
    results = {}
    futures = {}
    
    # Use ThreadPoolExecutor for parallel execution
    max_workers = max_workers or min(len(trained_models), 4)  # Limit to 4 concurrent workers
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit backtest tasks for each model
        for model_type in trained_models:
            future = executor.submit(
                _run_single_model_backtest,
                symbol, exchange, timeframe, start_date, end_date, 
                initial_capital, model_type
            )
            futures[future] = model_type
        
        # Collect results as they complete
        for future in as_completed(futures):
            model_type = futures[future]
            try:
                result = future.result()
                results[model_type] = result
                print(f"✅ Completed backtest for {model_type}")
            except Exception as e:
                print(f"❌ Failed backtest for {model_type}: {e}")
                results[model_type] = {"error": str(e)}
    
    # Compare results and determine best model
    comparison = _compare_model_results(results)
    
    # Save comparison results
    _save_multi_model_results(comparison, symbol, timeframe, start_date, end_date)
    
    return comparison


def _is_model_trained(model_type):
    """
    Check if a specific ML model is trained and available.
    
    Args:
        model_type: Type of model to check ('traditional_ml', 'lstm', 'transformer')
        
    Returns:
        Boolean indicating if model is trained
    """
    try:
        if model_type == 'traditional_ml':
            from src.ml_predictor import get_ml_predictor, is_ml_available
            if is_ml_available():
                predictor = get_ml_predictor()
                return predictor and predictor.is_trained
        elif model_type == 'lstm':
            from src.ml_predictor import get_lstm_predictor, is_lstm_available
            if is_lstm_available():
                predictor = get_lstm_predictor()
                return predictor and predictor.is_trained
        elif model_type == 'transformer':
            from src.ml_predictor import get_transformer_predictor, is_transformer_available
            if is_transformer_available():
                predictor = get_transformer_predictor()
                return predictor and predictor.is_trained
    except Exception as e:
        print(f"⚠️  Error checking {model_type} status: {e}")
    
    return False


def _run_single_model_backtest(symbol, exchange, timeframe, start_date, end_date, 
                              initial_capital, model_type):
    """
    Run backtest for a single ML model type.
    
    Args:
        model_type: Type of model to use ('traditional_ml', 'lstm', 'transformer', 'ensemble')
        
    Returns:
        Dictionary with backtest results for this model
    """
    from config.settings import ML_ENABLED
    
    print(f"🔍 Starting backtest for {model_type} on {symbol} {timeframe}")
    
    try:
        # Create backtester instance
        backtester = Backtester(symbol, exchange, timeframe, start_date, end_date, initial_capital)
        
        # Modify signal generation to use only the specified model
        if model_type != 'ensemble':
            # For single models, we'll need to modify the signal generation temporarily
            # This requires monkey-patching the apply_ml_signal_enhancement function
            import src.signal_generator as sg_module
            
            # Store original function
            original_apply_ml = sg_module.apply_ml_signal_enhancement
            
            # Create model-specific version
            def model_specific_enhancement(df):
                return _apply_single_model_enhancement(df, model_type)
            
            # Monkey patch for this backtest
            sg_module.apply_ml_signal_enhancement = model_specific_enhancement
            
            # Run backtest
            print(f"📊 Running backtest with {model_type} model...")
            metrics = backtester.run_backtest()
            
            # Restore original function
            sg_module.apply_ml_signal_enhancement = original_apply_ml
            
        else:
            # For ensemble, use the normal enhancement
            print(f"📊 Running backtest with ensemble model...")
            metrics = backtester.run_backtest()
        
        # Check if backtest was successful
        if 'error' in metrics:
            print(f"❌ Backtest failed for {model_type}: {metrics['error']}")
            return metrics
        
        # Add model info to results
        metrics['model_type'] = model_type
        metrics['trades_count'] = len(backtester.trades)
        
        print(f"✅ Backtest completed for {model_type}: {metrics.get('total_trades', 0)} trades, "
              f"P&L: ${metrics.get('total_pnl', 0):.2f}, Win Rate: {metrics.get('win_rate', 0)}%")
        
        return metrics
        
    except Exception as e:
        error_msg = f"Backtest failed for {model_type}: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "error": error_msg,
            "model_type": model_type
        }


def _apply_single_model_enhancement(df, model_type):
    """
    Apply enhancement using only a specific ML model type.
    
    Args:
        df: DataFrame with signals
        model_type: Type of model to use ('traditional_ml', 'lstm', 'transformer')
        
    Returns:
        Enhanced DataFrame
    """
    from config.settings import ML_ENABLED, ML_CONFIDENCE_THRESHOLD
    
    # Check if ML is enabled and available
    if not ML_ENABLED:
        print(f"⚠️  ML disabled in config, using traditional signals only for {model_type}")
        return df
    
    try:
        # Initialize specific model predictor
        predictor = None
        predictor_name = ""
        
        if model_type == 'traditional_ml':
            from src.ml_predictor import get_ml_predictor, is_ml_available
            if is_ml_available():
                predictor = get_ml_predictor()
                predictor_name = "traditional_ml"
                
        elif model_type == 'lstm':
            from src.ml_predictor import get_lstm_predictor, is_lstm_available
            if is_lstm_available():
                predictor = get_lstm_predictor()
                predictor_name = "lstm"
                
        elif model_type == 'transformer':
            from src.ml_predictor import get_transformer_predictor, is_transformer_available
            if is_transformer_available():
                predictor = get_transformer_predictor()
                predictor_name = "transformer"
        
        # If model not available or not trained, return original dataframe with warning
        if not predictor or not predictor.is_trained:
            print(f"⚠️  {model_type} model not available or not trained, using traditional signals only")
            return df
        
        print(f"🔬 Using single {model_type} model for signal enhancement")
        
        # Add ML signal columns
        df['ML_Signal'] = 'HOLD'
        df['ML_Confidence'] = 0.0
        df['ML_Reason'] = ''
        df['Enhanced_Signal'] = 'HOLD'
        df['Signal_Source'] = 'TRADITIONAL'
        
        # Process each row for ML prediction
        for idx, row in df.iterrows():
            try:
                # Create single-row DataFrame for prediction
                row_df = pd.DataFrame([row])
                
                # Get prediction from the specific model
                prediction = predictor.predict_signal(row_df, threshold=ML_CONFIDENCE_THRESHOLD)
                
                # Update DataFrame
                df.at[idx, 'ML_Signal'] = prediction['signal']
                df.at[idx, 'ML_Confidence'] = prediction['confidence']
                df.at[idx, 'ML_Reason'] = prediction.get('reason', '')
                df.at[idx, 'Signal_Source'] = predictor_name.upper()
                
                # Determine traditional signal
                traditional_signal = 'HOLD'
                if row.get('Buy_Signal', 0) == 1:
                    traditional_signal = 'BUY'
                elif row.get('Sell_Signal', 0) == 1:
                    traditional_signal = 'SELL'
                
                # Simple enhancement logic
                if prediction['confidence'] > ML_CONFIDENCE_THRESHOLD:
                    enhanced_signal = prediction['signal']
                    source = predictor_name.upper()
                else:
                    enhanced_signal = traditional_signal
                    source = 'TRADITIONAL'
                
                df.at[idx, 'Enhanced_Signal'] = enhanced_signal
                df.at[idx, 'Signal_Source'] = source
                
                # Override original signals if enhanced
                if source == predictor_name.upper():
                    if enhanced_signal == 'BUY':
                        df.at[idx, 'Buy_Signal'] = 1
                        df.at[idx, 'Sell_Signal'] = 0
                    elif enhanced_signal == 'SELL':
                        df.at[idx, 'Buy_Signal'] = 0
                        df.at[idx, 'Sell_Signal'] = 1
                    else:
                        df.at[idx, 'Buy_Signal'] = 0
                        df.at[idx, 'Sell_Signal'] = 0
                        
            except Exception as e:
                print(f"⚠️  Prediction failed for row {idx} in {model_type}: {e}")
                continue
        
        return df
        
    except Exception as e:
        # Return original dataframe if enhancement fails
        print(f"⚠️  ML enhancement failed for {model_type}: {e}, using traditional signals")
        return df


def _compare_model_results(results):
    """
    Compare results from different models and determine the best performer.
    
    Args:
        results: Dictionary with results for each model
        
    Returns:
        Dictionary with comparison and ranking
    """
    # Filter out failed backtests
    successful_results = {k: v for k, v in results.items() if 'error' not in v}
    failed_results = {k: v for k, v in results.items() if 'error' in v}
    
    print(f"📊 Backtest Summary: {len(successful_results)} successful, {len(failed_results)} failed")
    
    if failed_results:
        print("❌ Failed models:")
        for model, error_info in failed_results.items():
            print(f"   - {model}: {error_info.get('error', 'Unknown error')}")
    
    if not successful_results:
        return {"error": "All model backtests failed"}
    
    # Define metrics for comparison (higher is better)
    comparison_metrics = {
        'sharpe_ratio': 'Sharpe Ratio',
        'win_rate': 'Win Rate (%)',
        'total_pnl': 'Total P&L',
        'profit_factor': 'Profit Factor',
        'cagr': 'CAGR (%)'
    }
    
    # Create comparison DataFrame
    comparison_data = []
    
    for model_type, metrics in successful_results.items():
        row = {'Model': model_type}
        for metric_key, metric_name in comparison_metrics.items():
            value = metrics.get(metric_key, 0)
            # Handle infinite profit factor
            if metric_key == 'profit_factor' and value == float('inf'):
                value = 999.0  # Large finite number for comparison
            row[metric_name] = value
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Rank models for each metric
    rankings = {}
    for metric_name in comparison_metrics.values():
        if metric_name in comparison_df.columns:
            # Sort by metric (descending for all metrics)
            sorted_df = comparison_df.sort_values(metric_name, ascending=False)
            rankings[metric_name] = sorted_df['Model'].tolist()
    
    # Calculate composite score (weighted average of rankings)
    model_scores = {}
    for model in successful_results.keys():
        score = 0
        count = 0
        for metric_name, ranking in rankings.items():
            if model in ranking:
                # Lower rank number is better (1st place = 1, 2nd place = 2, etc.)
                rank = ranking.index(model) + 1
                score += rank
                count += 1
        model_scores[model] = score / count if count > 0 else float('inf')
    
    # Find best model (lowest average rank)
    best_model = min(model_scores, key=model_scores.get)
    
    return {
        'comparison': comparison_df.to_dict('records'),
        'rankings': rankings,
        'model_scores': model_scores,
        'best_model': best_model,
        'best_metrics': successful_results[best_model],
        'all_results': results,
        'successful_models': list(successful_results.keys()),
        'failed_models': list(failed_results.keys())
    }


def _save_multi_model_results(comparison, symbol, timeframe, start_date, end_date):
    """
    Save multi-model backtest comparison results to file.
    """
    import json
    from datetime import datetime
    
    if 'error' in comparison:
        return
    
    # Create results dictionary
    results = {
        'multi_model_backtest_info': {
            'symbol': symbol,
            'timeframe': timeframe,
            'start_date': str(start_date) if start_date else None,
            'end_date': str(end_date) if end_date else None,
            'timestamp': datetime.now().isoformat(),
            'best_model': comparison.get('best_model'),
            'models_tested': comparison.get('successful_models', [])
        },
        'comparison': comparison.get('comparison', []),
        'rankings': comparison.get('rankings', {}),
        'model_scores': comparison.get('model_scores', {}),
        'best_model_metrics': comparison.get('best_metrics', {})
    }
    
    # Save to JSON file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"output/multi_model_backtest_{symbol.replace('/', '_')}_{timeframe}_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4, default=str)
    
    print(f"📊 Multi-model backtest results saved to {filename}")
    
    # Also save comparison as CSV for easy viewing
    if comparison.get('comparison'):
        comparison_df = pd.DataFrame(comparison['comparison'])
        csv_filename = f"output/multi_model_comparison_{symbol.replace('/', '_')}_{timeframe}_{timestamp}.csv"
        comparison_df.to_csv(csv_filename, index=False)
        print(f"📊 Comparison table saved to {csv_filename}")
