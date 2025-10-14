"""
Shared fitness calculation functions for both Discovery and Backtester modes.

This module provides a unified fitness calculation system that can be used
across different optimization and evaluation contexts.
"""

import numpy as np
from typing import Dict, Any, Tuple
from config.settings import FITNESS_WEIGHTS, FITNESS_BOUNDS, FITNESS_RISK_PENALTIES


def calculate_fitness_from_metrics(backtest_metrics: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate fitness score from backtest metrics using configurable weights.

    This function provides a unified fitness calculation used by both Discovery
    and Backtester modes, ensuring consistent evaluation criteria.

    Args:
        backtest_metrics: Dictionary containing backtest performance metrics

    Returns:
        Tuple of (fitness_score, metrics_dict)
    """
    # Extract metrics with defaults
    total_pnl = backtest_metrics.get('total_pnl', 0)
    win_rate = backtest_metrics.get('win_rate', 0)
    sharpe_ratio = backtest_metrics.get('sharpe_ratio', 0)
    max_drawdown = backtest_metrics.get('max_drawdown', 0)
    total_trades = backtest_metrics.get('total_trades', 0)
    profit_factor = backtest_metrics.get('profit_factor', 1.0)
    calmar_ratio = backtest_metrics.get('calmar_ratio', 0)

    # Get weights from configuration
    sharpe_weight = FITNESS_WEIGHTS['sharpe_ratio']
    calmar_weight = FITNESS_WEIGHTS['calmar_ratio']
    pnl_weight = FITNESS_WEIGHTS['total_pnl']
    profit_factor_weight = FITNESS_WEIGHTS['profit_factor']
    win_rate_weight = FITNESS_WEIGHTS['win_rate']

    # Get bounds for normalization
    sharpe_bounds = FITNESS_BOUNDS['sharpe_ratio']
    calmar_bounds = FITNESS_BOUNDS['calmar_ratio']
    pnl_bounds = FITNESS_BOUNDS['total_pnl']
    profit_factor_bounds = FITNESS_BOUNDS['profit_factor']
    win_rate_bounds = FITNESS_BOUNDS['win_rate']

    # Normalize and score each metric
    # Sharpe Ratio: Cap at reasonable range, penalize negative heavily
    sharpe_score = _normalize_metric(sharpe_ratio, sharpe_bounds['min'], sharpe_bounds['max']) * sharpe_weight

    # Calmar Ratio: Return-to-drawdown ratio, very important for risk management
    calmar_score = _normalize_metric(calmar_ratio, calmar_bounds['min'], calmar_bounds['max']) * calmar_weight

    # Total P&L (Outperformance): Direct percentage contribution - HIGHER WEIGHT FOR OUTPERFORMANCE
    pnl_score = _normalize_metric(total_pnl, pnl_bounds['min'], pnl_bounds['max']) * pnl_weight

    # Profit Factor: >1 means profitable, penalize <1 heavily
    profit_factor_score = _normalize_metric(profit_factor - 1, profit_factor_bounds['min'], profit_factor_bounds['max']) * profit_factor_weight

    # Win Rate: Reward consistency, but lower weight
    win_rate_score = (win_rate / win_rate_bounds['max']) * win_rate_weight

    # Base fitness from risk-adjusted metrics
    fitness = sharpe_score + calmar_score + pnl_score + profit_factor_score + win_rate_score

    # Apply risk management penalties
    fitness = _apply_risk_penalties(fitness, backtest_metrics)

    metrics = {
        'total_pnl': total_pnl,
        'win_rate': win_rate,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'total_trades': total_trades,
        'profit_factor': profit_factor,
        'calmar_ratio': calmar_ratio,
        'fitness_breakdown': {
            'sharpe_score': sharpe_score,
            'calmar_score': calmar_score,
            'pnl_score': pnl_score,
            'profit_factor_score': profit_factor_score,
            'win_rate_score': win_rate_score
        }
    }

    return fitness, metrics


def _normalize_metric(value: float, min_val: float, max_val: float) -> float:
    """
    Normalize a metric value to 0-1 scale using min-max normalization.

    Args:
        value: Raw metric value
        min_val: Minimum expected value
        max_val: Maximum expected value

    Returns:
        Normalized value between 0 and 1
    """
    if max_val == min_val:
        return 0.5  # Neutral score if no range

    # Clamp value to bounds
    clamped_value = max(min_val, min(max_val, value))

    # Normalize to 0-1 scale
    normalized = (clamped_value - min_val) / (max_val - min_val)

    return normalized


def _apply_risk_penalties(fitness: float, metrics: Dict[str, Any]) -> float:
    """
    Apply risk management penalties to the fitness score.

    Args:
        fitness: Current fitness score
        metrics: Backtest metrics dictionary

    Returns:
        Adjusted fitness score with penalties applied
    """
    max_drawdown = metrics.get('max_drawdown', 0)
    total_trades = metrics.get('total_trades', 0)
    total_pnl = metrics.get('total_pnl', 0)

    # Drawdown penalties
    if max_drawdown > 30:
        fitness *= FITNESS_RISK_PENALTIES['max_drawdown_30']
    elif max_drawdown > 20:
        fitness *= FITNESS_RISK_PENALTIES['max_drawdown_20']
    elif max_drawdown > 15:
        fitness *= FITNESS_RISK_PENALTIES['max_drawdown_15']

    # Trade frequency penalties
    if total_trades < 10:
        fitness *= FITNESS_RISK_PENALTIES['insufficient_trades']
    elif total_trades > 500:
        fitness *= FITNESS_RISK_PENALTIES['overtrading']

    # Combined risk penalty for negative P&L with high drawdown
    if total_pnl < -10 and max_drawdown > 15:
        fitness *= FITNESS_RISK_PENALTIES['negative_pnl_high_risk']

    # Small bonus for positive P&L
    if total_pnl > 0:
        fitness *= FITNESS_RISK_PENALTIES['positive_pnl_bonus']

    return fitness


def calculate_fitness_from_trades(trades_df, initial_capital: float = 10000) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate fitness from a DataFrame of trades.

    This is used when we have raw trade data instead of aggregated metrics.

    Args:
        trades_df: DataFrame with trade data
        initial_capital: Initial capital amount

    Returns:
        Tuple of (fitness_score, metrics_dict)
    """
    if trades_df.empty:
        return 0.0, {
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'total_trades': 0,
            'profit_factor': 0.0,
            'calmar_ratio': 0.0
        }

    # Calculate basic metrics
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

    total_pnl = trades_df['pnl'].sum()
    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Calculate Sharpe ratio (simplified)
    if len(trades_df) > 1:
        returns = trades_df['pnl'] / initial_capital
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0.0
    else:
        sharpe_ratio = 0.0

    # Calculate max drawdown (simplified)
    cumulative_pnl = trades_df['pnl'].cumsum()
    running_max = cumulative_pnl.expanding().max()
    drawdown = running_max - cumulative_pnl
    max_drawdown = drawdown.max() if not drawdown.empty else 0.0

    # Calculate Calmar ratio
    calmar_ratio = total_pnl / max_drawdown if max_drawdown > 0 else 0.0

    # Use the shared fitness calculation
    metrics = {
        'total_pnl': total_pnl,
        'win_rate': win_rate,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'total_trades': total_trades,
        'profit_factor': profit_factor,
        'calmar_ratio': calmar_ratio
    }

    fitness, enhanced_metrics = calculate_fitness_from_metrics(metrics)
    return fitness, enhanced_metrics


def get_fitness_weights() -> Dict[str, float]:
    """
    Get current fitness weights configuration.

    Returns:
        Dictionary of fitness weights
    """
    return FITNESS_WEIGHTS.copy()


def update_fitness_weights(new_weights: Dict[str, float]) -> None:
    """
    Update fitness weights configuration.

    Args:
        new_weights: Dictionary with new weight values
    """
    global FITNESS_WEIGHTS
    FITNESS_WEIGHTS.update(new_weights)


def get_fitness_configuration() -> Dict[str, Any]:
    """
    Get complete fitness configuration.

    Returns:
        Dictionary with all fitness settings
    """
    return {
        'weights': FITNESS_WEIGHTS.copy(),
        'bounds': FITNESS_BOUNDS.copy(),
        'risk_penalties': FITNESS_RISK_PENALTIES.copy()
    }