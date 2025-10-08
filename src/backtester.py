import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from config.settings import CAPITAL, RISK_PER_TRADE, OUTPUT_FILE
from src.data_fetcher import fetch_historical_data
from src.indicators import calculate_indicators
from src.signal_generator import generate_signals, calculate_risk_management

class Backtester:
    """
    Historical backtesting module for trading strategies.
    Calculates performance metrics like win rate, CAGR, drawdown, etc.
    """

    def __init__(self, symbol='EUR/USD', exchange='kraken', timeframe='1m',
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

    def run_backtest(self):
        """
        Run historical backtest and return performance metrics.
        """
        print(f"Running backtest for {self.symbol} on {self.timeframe} timeframe")
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")

        # Fetch historical data
        data = self._fetch_data()

        # Check if data fetch failed (returned error string)
        if isinstance(data, str):
            return {"error": f"Data fetch failed: {data}"}

        if data.empty:
            return {"error": "No data available for backtest period"}

        # Calculate indicators and signals (skip if already processed)
        required_cols = ['EMA9', 'EMA21', 'RSI', 'BB_upper', 'BB_middle', 'BB_lower', 'ATR', 'Buy_Signal', 'Sell_Signal', 'Position_Size_Absolute']
        if not all(col in data.columns for col in required_cols):
            data = calculate_indicators(data)
            data = generate_signals(data)
            data = calculate_risk_management(data)

        # Simulate trades
        results = self._simulate_trades(data)
        self.trades = results

        # Calculate performance metrics
        metrics = self._calculate_metrics()

        # Save backtest results
        self._save_results(metrics)

        return metrics

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

def run_backtest(symbol='EUR/USD', timeframe='1m', start_date=None, end_date=None):
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