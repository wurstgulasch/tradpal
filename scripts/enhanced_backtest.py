#!/usr/bin/env python3
"""
Enhanced Backtest Script for TradPal
Creates detailed reports with parameters, results, and exports.
"""

import sys
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_fetcher import fetch_data
from src.indicators import calculate_indicators
from src.signal_generator import generate_signals

# Set plotting style
plt.style.use('default')

class EnhancedBacktester:
    """
    Enhanced backtesting with detailed reporting and export capabilities.
    """

    def __init__(self, symbol='BTC/USDT', exchange='kraken', timeframe='1h',
                 start_date=None, end_date=None, initial_capital=10000):
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        self.start_date = start_date or (datetime.now() - timedelta(days=90))
        self.end_date = end_date or datetime.now()
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trades = []
        self.portfolio_values = []

        # Backtest parameters
        self.parameters = {
            'symbol': symbol,
            'exchange': exchange,
            'timeframe': timeframe,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'initial_capital': initial_capital,
            'indicators': {
                'ema_short': 9,
                'ema_long': 21,
                'rsi_period': 14,
                'rsi_oversold': 35,  # Relaxed for better signals
                'rsi_overbought': 65,
                'bb_period': 20,
                'bb_std_dev': 2.0,
                'atr_period': 14
            },
            'risk_management': {
                'risk_per_trade': 0.02,  # 2% risk per trade
                'max_position_size': 0.1,  # Max 10% of capital
                'stop_loss_atr_multiplier': 1.5,
                'take_profit_atr_multiplier': 3.0
            },
            'trading_fees': {
                'maker_fee': 0.0016,
                'taker_fee': 0.0026
            }
        }

    def run_backtest(self):
        """Run enhanced backtest with detailed reporting."""
        print(f"🚀 Starting Enhanced Backtest")
        print(f"📊 {self.symbol} on {self.timeframe} timeframe")
        print(f"📅 Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"💰 Initial Capital: ${self.initial_capital:,.0f}")

        # Fetch data
        data = self._fetch_data()
        if data.empty:
            return {"error": "No data available"}

        print(f"📈 Loaded {len(data)} data points")

        # Calculate indicators with custom config
        data = self._calculate_indicators(data)
        print(f"🧮 Calculated {len([col for col in data.columns if col not in ['open','high','low','close','volume']])} indicators")

        # Generate signals with relaxed conditions
        data = self._generate_signals(data)
        buy_signals = data['Buy_Signal'].sum()
        sell_signals = data['Sell_Signal'].sum()
        print(f"🎯 Generated {buy_signals} buy and {sell_signals} sell signals")

        # Apply risk management
        data = self._apply_risk_management(data)
        print("⚠️ Applied risk management parameters")

        # Simulate trades
        trades_df = self._simulate_trades(data)
        print(f"💼 Simulated {len(trades_df)} trades")

        # Calculate metrics
        metrics = self._calculate_metrics(trades_df)

        # Create comprehensive report
        report = self._create_report(data, trades_df, metrics)

        # Save results
        self._save_results(report)

        return report

    def _fetch_data(self):
        """Fetch historical data for the specified period."""
        try:
            from src.data_fetcher import fetch_historical_data
            from datetime import datetime
            
            # Calculate days between start and end date
            days = (self.end_date - self.start_date).days
            
            # Estimate limit based on timeframe and days
            if self.timeframe == '1m':
                limit = min(days * 24 * 60, 50000)  # Max 50k for 1m
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
                limit = 10000
            
            print(f"Fetching {limit} data points for {days} days...")
            
            # Try to fetch historical data with proper date range
            data = fetch_historical_data(
                symbol=self.symbol,
                exchange_name=self.exchange,
                timeframe=self.timeframe,
                limit=limit,
                start_date=self.start_date
            )
            
            if data.empty or len(data) < 100:
                print(f"Insufficient historical data ({len(data)} points), falling back to recent data")
                # Fallback to fetch_data for more recent data
                from src.data_fetcher import fetch_data
                data = fetch_data(limit=max(limit, 1000))
            
            return data
        except Exception as e:
            print(f"Error fetching historical data: {e}, using fallback")
            # Fallback to basic fetch_data
            from src.data_fetcher import fetch_data
            return fetch_data(limit=1000)

    def _calculate_indicators(self, data):
        """Calculate indicators with custom configuration."""
        config = {
            'ema': {'enabled': True, 'periods': [self.parameters['indicators']['ema_short'], self.parameters['indicators']['ema_long']]},
            'rsi': {'enabled': True, 'period': self.parameters['indicators']['rsi_period'], 'oversold': self.parameters['indicators']['rsi_oversold'], 'overbought': self.parameters['indicators']['rsi_overbought']},
            'bb': {'enabled': True, 'period': self.parameters['indicators']['bb_period'], 'std_dev': self.parameters['indicators']['bb_std_dev']},
            'atr': {'enabled': True, 'period': self.parameters['indicators']['atr_period']}
        }
        return calculate_indicators(data, config=config)

    def _generate_signals(self, data):
        """Generate signals using the full TradPal signal generation pipeline (including ML features)."""
        # Use the full TradPal signal generation which includes ML enhancement
        from src.signal_generator import generate_signals as tradpal_generate_signals

        # Apply the full signal generation pipeline
        data_with_signals = tradpal_generate_signals(data)

        return data_with_signals

    def _apply_risk_management(self, data):
        """Apply risk management parameters."""
        # Simple risk management for demo
        data['Position_Size_Percent'] = self.parameters['risk_management']['max_position_size']
        data['Stop_Loss_Buy'] = data['close'] * (1 - 0.02)  # 2% stop loss
        data['Take_Profit_Buy'] = data['close'] * (1 + 0.04)  # 4% take profit
        data['Stop_Loss_Sell'] = data['close'] * (1 + 0.02)  # 2% stop loss
        data['Take_Profit_Sell'] = data['close'] * (1 - 0.04)  # 4% take profit
        data['Leverage'] = 1.0  # No leverage for demo

        return data

    def _simulate_trades(self, data):
        """Simulate trades based on signals."""
        trades = []
        position = 0  # 0 = no position, 1 = long, -1 = short

        for idx, row in data.iterrows():
            current_price = row['close']

            # Check for entry signals
            if position == 0:
                if row.get('Buy_Signal', 0) == 1:
                    position = 1
                    entry_price = current_price
                    entry_time = idx
                    position_size_pct = row.get('Position_Size_Percent', 0.1)
                    stop_loss = row.get('Stop_Loss_Buy', current_price * 0.98)
                    take_profit = row.get('Take_Profit_Buy', current_price * 1.02)

                    trades.append({
                        'type': 'buy',
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'position_size_pct': position_size_pct,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'exit_time': None,
                        'exit_price': None,
                        'pnl': 0,
                        'status': 'open'
                    })

                elif row.get('Sell_Signal', 0) == 1:
                    position = -1
                    entry_price = current_price
                    entry_time = idx
                    position_size_pct = row.get('Position_Size_Percent', 0.1)
                    stop_loss = row.get('Stop_Loss_Sell', current_price * 1.02)
                    take_profit = row.get('Take_Profit_Sell', current_price * 0.98)

                    trades.append({
                        'type': 'sell',
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'position_size_pct': position_size_pct,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'exit_time': None,
                        'exit_price': None,
                        'pnl': 0,
                        'status': 'open'
                    })

            # Check for exit conditions
            elif position != 0:
                exit_signal = False
                exit_price = current_price
                exit_reason = ''

                if position == 1:  # Long position
                    if current_price <= trades[-1]['stop_loss']:
                        exit_signal = True
                        exit_price = trades[-1]['stop_loss']
                        exit_reason = 'stop_loss'
                    elif current_price >= trades[-1]['take_profit']:
                        exit_signal = True
                        exit_price = trades[-1]['take_profit']
                        exit_reason = 'take_profit'
                    elif row.get('Sell_Signal', 0) == 1:
                        exit_signal = True
                        exit_reason = 'sell_signal'

                else:  # Short position
                    if current_price >= trades[-1]['stop_loss']:
                        exit_signal = True
                        exit_price = trades[-1]['stop_loss']
                        exit_reason = 'stop_loss'
                    elif current_price <= trades[-1]['take_profit']:
                        exit_signal = True
                        exit_price = trades[-1]['take_profit']
                        exit_reason = 'take_profit'
                    elif row.get('Buy_Signal', 0) == 1:
                        exit_signal = True
                        exit_reason = 'buy_signal'

                if exit_signal:
                    # Calculate P&L
                    if position == 1:
                        pnl = (exit_price - entry_price) * self.current_capital * trades[-1]['position_size_pct'] / entry_price
                    else:
                        pnl = (entry_price - exit_price) * self.current_capital * trades[-1]['position_size_pct'] / entry_price

                    # Update trade
                    trades[-1].update({
                        'exit_time': idx,
                        'exit_price': exit_price,
                        'exit_reason': exit_reason,
                        'pnl': pnl,
                        'status': 'closed'
                    })

                    # Update capital
                    self.current_capital += pnl
                    position = 0

        # Close any remaining positions
        if position != 0 and trades:
            final_price = data.iloc[-1]['close']
            final_time = data.index[-1]

            if position == 1:
                pnl = (final_price - entry_price) * self.current_capital * trades[-1]['position_size_pct'] / entry_price
            else:
                pnl = (entry_price - final_price) * self.current_capital * trades[-1]['position_size_pct'] / entry_price

            trades[-1].update({
                'exit_time': final_time,
                'exit_price': final_price,
                'exit_reason': 'end_of_data',
                'pnl': pnl,
                'status': 'closed'
            })

            self.current_capital += pnl

        return pd.DataFrame(trades)

    def _calculate_metrics(self, trades_df):
        """Calculate comprehensive performance metrics."""
        if trades_df.empty:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'final_capital': self.initial_capital,
                'return_pct': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }

        closed_trades = trades_df[trades_df['status'] == 'closed']
        if closed_trades.empty:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'final_capital': self.initial_capital,
                'return_pct': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }

        # Basic metrics
        total_trades = len(closed_trades)
        winning_trades = (closed_trades['pnl'] > 0).sum()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_pnl = closed_trades['pnl'].sum()

        # Returns
        return_pct = (self.current_capital / self.initial_capital - 1) * 100

        # Risk metrics
        returns = closed_trades['pnl'] / self.initial_capital
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        # Max drawdown (simplified)
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(drawdown.min()) * 100 if not drawdown.empty else 0

        return {
            'total_trades': total_trades,
            'winning_trades': int(winning_trades),
            'losing_trades': int(total_trades - winning_trades),
            'win_rate': round(win_rate, 2),
            'total_pnl': round(total_pnl, 2),
            'avg_win': round(closed_trades[closed_trades['pnl'] > 0]['pnl'].mean(), 2) if winning_trades > 0 else 0,
            'avg_loss': round(closed_trades[closed_trades['pnl'] <= 0]['pnl'].mean(), 2) if (total_trades - winning_trades) > 0 else 0,
            'final_capital': round(self.current_capital, 2),
            'return_pct': round(return_pct, 2),
            'max_drawdown': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'profit_factor': round(closed_trades[closed_trades['pnl'] > 0]['pnl'].sum() / abs(closed_trades[closed_trades['pnl'] <= 0]['pnl'].sum()), 2) if closed_trades[closed_trades['pnl'] <= 0]['pnl'].sum() != 0 else float('inf')
        }

    def _calculate_benchmark_performance(self, data):
        """Calculate benchmark performance (buy & hold strategy)."""
        if data.empty:
            return {
                'benchmark_return_pct': 0,
                'benchmark_final_value': self.initial_capital,
                'outperformance_pct': 0,
                'alpha': 0
            }

        # Get first and last price
        first_price = data['close'].iloc[0]
        last_price = data['close'].iloc[-1]

        # Calculate buy & hold return
        benchmark_return_pct = ((last_price - first_price) / first_price) * 100
        benchmark_final_value = self.initial_capital * (1 + benchmark_return_pct / 100)

        # Calculate outperformance
        strategy_return_pct = (self.current_capital / self.initial_capital - 1) * 100
        outperformance_pct = strategy_return_pct - benchmark_return_pct

        return {
            'benchmark_return_pct': round(benchmark_return_pct, 2),
            'benchmark_final_value': round(benchmark_final_value, 2),
            'outperformance_pct': round(outperformance_pct, 2),
            'alpha': round(outperformance_pct, 2),  # Alpha is essentially outperformance here
            'first_price': round(first_price, 2),
            'last_price': round(last_price, 2)
        }

    def _create_report(self, data, trades_df, metrics):
        """Create comprehensive backtest report."""
        # Calculate benchmark performance
        benchmark = self._calculate_benchmark_performance(data)

        report = {
            'backtest_info': {
                'timestamp': datetime.now().isoformat(),
                'symbol': self.symbol,
                'exchange': self.exchange,
                'timeframe': self.timeframe,
                'start_date': self.start_date.isoformat(),
                'end_date': self.end_date.isoformat(),
                'data_points': len(data),
                'duration_days': (self.end_date - self.start_date).days
            },
            'parameters': self.parameters,
            'metrics': metrics,
            'benchmark': benchmark,
            'trades': trades_df.to_dict('records') if not trades_df.empty else [],
            'data_summary': {
                'price_range': {
                    'min': float(data['close'].min()),
                    'max': float(data['close'].max()),
                    'avg': float(data['close'].mean())
                },
                'volatility': float(data['close'].pct_change().std() * 100),
                'total_volume': float(data['volume'].sum())
            }
        }

        return report

    def _save_results(self, report):
        """Save backtest results to files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"backtest_{self.symbol.replace('/', '_')}_{self.timeframe}_{timestamp}"

        # Create output directory
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)

        # Save JSON report
        json_file = output_dir / f"{base_filename}.json"
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"💾 JSON report saved: {json_file}")

        # Save CSV trades
        if report['trades']:
            trades_df = pd.DataFrame(report['trades'])
            csv_file = output_dir / f"{base_filename}_trades.csv"
            trades_df.to_csv(csv_file, index=False)
            print(f"💾 Trades CSV saved: {csv_file}")

        # Create summary text file
        summary_file = output_dir / f"{base_filename}_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("TRADPAL ENHANCED BACKTEST REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Symbol: {self.symbol}\n")
            f.write(f"Timeframe: {self.timeframe}\n")
            f.write(f"Period: {self.start_date.date()} to {self.end_date.date()}\n")
            f.write(f"Data Points: {report['backtest_info']['data_points']}\n\n")

            f.write("PARAMETERS:\n")
            f.write("-" * 20 + "\n")
            for category, params in report['parameters'].items():
                if isinstance(params, dict):
                    f.write(f"{category.upper()}:\n")
                    for key, value in params.items():
                        f.write(f"  {key}: {value}\n")
                    f.write("\n")

            f.write("PERFORMANCE METRICS:\n")
            f.write("-" * 25 + "\n")
            for key, value in report['metrics'].items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            f.write("\n")

            f.write("BENCHMARK COMPARISON:\n")
            f.write("-" * 25 + "\n")
            benchmark = report['benchmark']
            f.write(f"Strategy Return: {report['metrics']['return_pct']}%\n")
            f.write(f"Buy & Hold Return: {benchmark['benchmark_return_pct']}%\n")
            f.write(f"Outperformance: {benchmark['outperformance_pct']}%\n")
            f.write(f"Alpha: {benchmark['alpha']}%\n")
            f.write(f"Price Change: ${benchmark['first_price']} → ${benchmark['last_price']}\n")

        print(f"💾 Summary report saved: {summary_file}")

        # Create visualization
        self._create_visualization(report, base_filename)

    def _create_visualization(self, report, base_filename):
        """Create performance visualization."""
        if not report['trades']:
            return

        trades_df = pd.DataFrame(report['trades'])

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Backtest Performance Report - {self.symbol} {self.timeframe}', fontsize=16)

        # Performance metrics
        metrics = report['metrics']
        benchmark = report['benchmark']
        metrics_text = f"""
        Total Trades: {metrics['total_trades']}
        Win Rate: {metrics['win_rate']}%
        Total P&L: ${metrics['total_pnl']:,.2f}
        Return: {metrics['return_pct']}%
        Max Drawdown: {metrics['max_drawdown']}%
        Sharpe Ratio: {metrics['sharpe_ratio']}

        BENCHMARK COMPARISON:
        Strategy: {metrics['return_pct']}%
        Buy & Hold: {benchmark['benchmark_return_pct']}%
        Outperformance: {benchmark['outperformance_pct']}%
        Alpha: {benchmark['alpha']}%
        """

        axes[0, 0].text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
                       fontfamily='monospace')
        axes[0, 0].set_title('Performance Summary')
        axes[0, 0].set_xlim(0, 1)
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].axis('off')

        # Trade P&L distribution
        if not trades_df.empty and 'pnl' in trades_df.columns:
            pnl_data = trades_df['pnl'].dropna()
            if not pnl_data.empty:
                axes[0, 1].hist(pnl_data, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
                axes[0, 1].set_title('Trade P&L Distribution')
                axes[0, 1].set_xlabel('P&L ($)')
                axes[0, 1].set_ylabel('Frequency')

        # Cumulative returns (simplified)
        if not trades_df.empty:
            trades_df = trades_df.sort_values('entry_time')
            cumulative_pnl = trades_df['pnl'].cumsum()
            axes[1, 0].plot(cumulative_pnl.values, linewidth=2)
            axes[1, 0].set_title('Cumulative P&L')
            axes[1, 0].set_xlabel('Trade Number')
            axes[1, 0].set_ylabel('Cumulative P&L ($)')
            axes[1, 0].grid(True, alpha=0.3)

        # Win/Loss pie chart
        if metrics['total_trades'] > 0:
            win_loss_data = [metrics['winning_trades'], metrics['losing_trades']]
            labels = ['Wins', 'Losses']
            colors = ['green', 'red']
            axes[1, 1].pie(win_loss_data, labels=labels, colors=colors, autopct='%1.1f%%',
                           startangle=90)
            axes[1, 1].set_title('Win/Loss Ratio')

        plt.tight_layout()

        # Save plot
        output_dir = Path('output')
        plot_file = output_dir / f"{base_filename}_chart.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"💾 Performance chart saved: {plot_file}")
        plt.close()

def run_enhanced_backtest(symbol='BTC/USDT', timeframe='1h', start_date=None, end_date=None):
    """Run enhanced backtest with detailed reporting."""
    backtester = EnhancedBacktester(symbol, 'kraken', timeframe, start_date, end_date)
    results = backtester.run_backtest()

    if 'error' in results:
        print(f"❌ Backtest failed: {results['error']}")
        return None

    # Print summary
    metrics = results['metrics']
    benchmark = results['benchmark']
    print("\n" + "="*60)
    print("🎉 ENHANCED BACKTEST RESULTS")
    print("="*60)
    print(f"Symbol: {symbol} | Timeframe: {timeframe}")
    print(f"Period: {results['backtest_info']['start_date'][:10]} to {results['backtest_info']['end_date'][:10]}")
    print(f"Data Points: {results['backtest_info']['data_points']}")
    print()
    print("📊 PERFORMANCE METRICS:")
    print(f"  Total Trades: {metrics['total_trades']}")
    print(f"  Win Rate: {metrics['win_rate']}%")
    print(f"  Total P&L: ${metrics['total_pnl']:,.2f}")
    print(f"  Return: {metrics['return_pct']}%")
    print(f"  Max Drawdown: {metrics['max_drawdown']}%")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']}")
    print()
    print("🏆 BENCHMARK COMPARISON:")
    print(f"  Strategy Return: {metrics['return_pct']}%")
    print(f"  Buy & Hold Return: {benchmark['benchmark_return_pct']}%")
    print(f"  Outperformance: {benchmark['outperformance_pct']}%")
    print(f"  Alpha: {benchmark['alpha']}%")
    print(f"  Price Change: ${benchmark['first_price']} → ${benchmark['last_price']}")
    print()
    print("💾 Reports saved to output/ directory")
    print("="*60)

    return results

if __name__ == "__main__":
    # Run enhanced backtest
    results = run_enhanced_backtest(
        symbol='BTC/USDT',
        timeframe='1h',
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now()
    )