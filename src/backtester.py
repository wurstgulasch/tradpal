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

        # Calculate performance metrics
        metrics = self._calculate_metrics()

        # Save backtest results
        self._save_results(metrics)

        return metrics

    def _fetch_data(self):
        """Fetch historical data for backtest period."""
        # This would need to be implemented to fetch data for specific date ranges
        # For now, using existing fetch_historical_data
        return fetch_historical_data()

    def _simulate_trades(self, data):
        """
        Simulate trades based on signals and calculate P&L.
        """
        position = 0  # 0 = no position, 1 = long, -1 = short
        entry_price = 0
        entry_time = None

        for idx, row in data.iterrows():
            current_time = row.name if hasattr(row, 'name') else idx
            current_price = row['close']

            # Check for entry signals
            if position == 0:
                if row['Buy_Signal'] == 1:
                    position = 1
                    entry_price = current_price
                    entry_time = current_time
                    position_size = row.get('Position_Size_Absolute', 1000)
                    stop_loss = row.get('Stop_Loss_Buy', current_price * 0.98)
                    take_profit = row.get('Take_Profit_Buy', current_price * 1.02)

                    self.trades.append({
                        'type': 'buy',
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'position_size': position_size,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'exit_time': None,
                        'exit_price': None,
                        'pnl': 0,
                        'status': 'open'
                    })

                elif row['Sell_Signal'] == 1:
                    position = -1
                    entry_price = current_price
                    entry_time = current_time
                    position_size = row.get('Position_Size_Absolute', 1000)
                    stop_loss = row.get('Stop_Loss_Buy', current_price * 1.02)  # Would need sell SL calc
                    take_profit = row.get('Take_Profit_Buy', current_price * 0.98)  # Would need sell TP calc

                    self.trades.append({
                        'type': 'sell',
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'position_size': position_size,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'exit_time': None,
                        'exit_price': None,
                        'pnl': 0,
                        'status': 'open'
                    })

            # Check for exit conditions
            elif position != 0:
                exit_trade = False
                exit_price = current_price
                exit_reason = ''

                # Check stop loss and take profit
                if position == 1:  # Long position
                    if current_price <= self.trades[-1]['stop_loss']:
                        exit_trade = True
                        exit_reason = 'stop_loss'
                    elif current_price >= self.trades[-1]['take_profit']:
                        exit_trade = True
                        exit_reason = 'take_profit'
                else:  # Short position
                    if current_price >= self.trades[-1]['stop_loss']:
                        exit_trade = True
                        exit_reason = 'stop_loss'
                    elif current_price <= self.trades[-1]['take_profit']:
                        exit_trade = True
                        exit_reason = 'take_profit'

                if exit_trade:
                    # Calculate P&L
                    if position == 1:
                        pnl = (exit_price - entry_price) * self.trades[-1]['position_size'] / entry_price
                    else:
                        pnl = (entry_price - exit_price) * self.trades[-1]['position_size'] / entry_price

                    # Update trade record
                    self.trades[-1].update({
                        'exit_time': current_time,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'exit_reason': exit_reason,
                        'status': 'closed'
                    })

                    # Update capital
                    self.current_capital += pnl

                    # Reset position
                    position = 0

            # Track portfolio value
            self.portfolio_values.append({
                'timestamp': current_time,
                'value': self.current_capital
            })

        return self.trades

    def _calculate_metrics(self):
        """
        Calculate comprehensive performance metrics.
        """
        if not self.trades:
            return {"error": "No trades executed during backtest"}

        closed_trades = [t for t in self.trades if t['status'] == 'closed']

        if not closed_trades:
            return {"error": "No closed trades to analyze"}

        # Basic metrics
        total_trades = len(closed_trades)
        winning_trades = [t for t in closed_trades if t['pnl'] > 0]
        losing_trades = [t for t in closed_trades if t['pnl'] <= 0]

        win_rate = len(winning_trades) / total_trades * 100
        total_pnl = sum(t['pnl'] for t in closed_trades)
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0

        # Risk metrics
        returns = [t['pnl'] / self.initial_capital for t in closed_trades]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if returns else 0

        # Drawdown calculation
        portfolio_values = [self.initial_capital]
        for trade in closed_trades:
            portfolio_values.append(portfolio_values[-1] + trade['pnl'])

        peak = self.initial_capital
        max_drawdown = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)

        # CAGR (simplified for backtest period)
        days = (self.end_date - self.start_date).days
        if days > 0:
            cagr = ((self.current_capital / self.initial_capital) ** (365 / days) - 1) * 100
        else:
            cagr = 0

        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': round(win_rate, 2),
            'total_pnl': round(total_pnl, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'profit_factor': round(abs(sum(t['pnl'] for t in winning_trades) /
                                     sum(t['pnl'] for t in losing_trades)), 2) if losing_trades else float('inf'),
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
    Calculate comprehensive performance metrics from trades DataFrame.
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

    # Calculate P&L for each trade
    trades_df = trades_df.copy()
    trades_df['pnl'] = 0.0

    for idx, trade in trades_df.iterrows():
        if trade['direction'] == 'long':
            pnl = (trade['exit_price'] - trade['entry_price']) * trade['position_size'] / trade['entry_price']
        else:  # short
            pnl = (trade['entry_price'] - trade['exit_price']) * trade['position_size'] / trade['entry_price']
        trades_df.loc[idx, 'pnl'] = pnl

    # Basic metrics
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    losing_trades = len(trades_df[trades_df['pnl'] <= 0])
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

    total_pnl = trades_df['pnl'].sum()
    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    final_capital = initial_capital + total_pnl
    return_pct = (final_capital / initial_capital - 1) * 100

    # Risk metrics
    returns = trades_df['pnl'] / initial_capital
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    # Max drawdown calculation
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    max_drawdown_percentage = abs(max_drawdown)

    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
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