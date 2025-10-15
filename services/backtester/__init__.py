"""
Backtester Service - Historical trading simulation and performance analysis.

Provides comprehensive backtesting capabilities for trading strategies.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from config.settings import (
    INITIAL_CAPITAL, RISK_PER_TRADE, SYMBOL, TIMEFRAME
)
from services.signal_generator import generate_traditional_signals

logger = logging.getLogger(__name__)


def run_backtest(data: pd.DataFrame,
                strategy: str = 'traditional',
                initial_capital: float = INITIAL_CAPITAL,
                risk_per_trade: float = RISK_PER_TRADE) -> Dict[str, Any]:
    """Run a backtest on historical data."""
    try:
        # Generate signals based on strategy
        if strategy == 'traditional':
            signal_data = generate_traditional_signals(data)
        else:
            signal_data = data.copy()

        # Simulate trades
        trades = simulate_trades(signal_data, initial_capital, risk_per_trade)

        # Calculate performance metrics
        metrics = calculate_performance_metrics(trades, initial_capital)

        return {
            'trades': trades.to_dict('records') if not trades.empty else [],
            'metrics': metrics,
            'strategy': strategy,
            'symbol': SYMBOL,
            'timeframe': TIMEFRAME,
            'period': {
                'start': data.index.min().isoformat() if not data.empty else None,
                'end': data.index.max().isoformat() if not data.empty else None
            }
        }
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        return {
            'error': str(e),
            'trades': [],
            'metrics': {},
            'strategy': strategy
        }


def simulate_trades(data: pd.DataFrame,
                   initial_capital: float,
                   risk_per_trade: float) -> pd.DataFrame:
    """Simulate trades based on signals."""
    trades = []
    capital = initial_capital
    position = 0  # 0 = no position, 1 = long, -1 = short

    for idx, row in data.iterrows():
        try:
            price = row['close']

            # Check for buy signal
            if row.get('Buy_Signal', 0) == 1 and position == 0:
                # Calculate position size based on risk
                risk_amount = capital * risk_per_trade
                atr_value = row.get('ATR', price * 0.02)  # Default ATR if not available
                stop_loss_distance = atr_value * 2  # 2 ATR stop loss
                position_size = risk_amount / stop_loss_distance

                if position_size * price <= capital:
                    position = 1
                    entry_price = price
                    stop_loss = entry_price - stop_loss_distance
                    take_profit = entry_price + (stop_loss_distance * 2)  # 2:1 reward/risk

                    trades.append({
                        'entry_time': idx,
                        'entry_price': entry_price,
                        'position_size': position_size,
                        'direction': 'long',
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'exit_time': None,
                        'exit_price': None,
                        'pnl': 0,
                        'status': 'open'
                    })

            # Check for sell signal
            elif row.get('Sell_Signal', 0) == 1 and position == 0:
                # Calculate position size based on risk
                risk_amount = capital * risk_per_trade
                atr_value = row.get('ATR', price * 0.02)
                stop_loss_distance = atr_value * 2
                position_size = risk_amount / stop_loss_distance

                if position_size * price <= capital:
                    position = -1
                    entry_price = price
                    stop_loss = entry_price + stop_loss_distance
                    take_profit = entry_price - (stop_loss_distance * 2)

                    trades.append({
                        'entry_time': idx,
                        'entry_price': entry_price,
                        'position_size': position_size,
                        'direction': 'short',
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'exit_time': None,
                        'exit_price': None,
                        'pnl': 0,
                        'status': 'open'
                    })

            # Check for exit conditions on open positions
            if position != 0 and trades and trades[-1]['status'] == 'open':
                current_trade = trades[-1]

                if position == 1:  # Long position
                    # Check stop loss
                    if price <= current_trade['stop_loss']:
                        exit_price = current_trade['stop_loss']
                        pnl = (exit_price - current_trade['entry_price']) * current_trade['position_size']
                        capital += pnl
                        position = 0
                    # Check take profit
                    elif price >= current_trade['take_profit']:
                        exit_price = current_trade['take_profit']
                        pnl = (exit_price - current_trade['entry_price']) * current_trade['position_size']
                        capital += pnl
                        position = 0
                    else:
                        continue

                elif position == -1:  # Short position
                    # Check stop loss
                    if price >= current_trade['stop_loss']:
                        exit_price = current_trade['stop_loss']
                        pnl = (current_trade['entry_price'] - exit_price) * current_trade['position_size']
                        capital += pnl
                        position = 0
                    # Check take profit
                    elif price <= current_trade['take_profit']:
                        exit_price = current_trade['take_profit']
                        pnl = (current_trade['entry_price'] - exit_price) * current_trade['position_size']
                        capital += pnl
                        position = 0
                    else:
                        continue

                # Update trade record
                if position == 0:
                    current_trade['exit_time'] = idx
                    current_trade['exit_price'] = exit_price
                    current_trade['pnl'] = pnl
                    current_trade['status'] = 'closed'

        except Exception as e:
            logger.error(f"Error simulating trade at {idx}: {e}")
            continue

    return pd.DataFrame(trades)


def calculate_performance_metrics(trades: pd.DataFrame,
                                initial_capital: float) -> Dict[str, Any]:
    """Calculate comprehensive performance metrics."""
    if trades.empty:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'gross_profit': 0,
            'gross_loss': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'final_capital': initial_capital
        }

    # Basic metrics
    total_trades = len(trades)
    winning_trades = len(trades[trades['pnl'] > 0])
    losing_trades = len(trades[trades['pnl'] < 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    # P&L metrics
    total_pnl = trades['pnl'].sum()
    gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades[trades['pnl'] < 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Final capital
    final_capital = initial_capital + total_pnl

    # Drawdown calculation (simplified)
    cumulative_pnl = trades['pnl'].cumsum()
    running_max = cumulative_pnl.expanding().max()
    drawdowns = running_max - cumulative_pnl
    max_drawdown = drawdowns.max() if not drawdowns.empty else 0

    # Sharpe ratio (simplified, assuming daily returns)
    if len(trades) > 1:
        returns = trades['pnl'] / initial_capital
        sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
    else:
        sharpe_ratio = 0

    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'final_capital': final_capital
    }


def run_multi_model_backtest(data: pd.DataFrame,
                           models: List[str] = None) -> Dict[str, Any]:
    """Run backtest with multiple ML models."""
    if models is None:
        models = ['traditional']

    results = {}

    for model in models:
        try:
            result = run_backtest(data, strategy=model)
            results[model] = result
        except Exception as e:
            logger.error(f"Failed to run backtest for model {model}: {e}")
            results[model] = {'error': str(e)}

    return results