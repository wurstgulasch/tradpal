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

        # Calculate indicators and signals
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
    return backtester.run_backtest()