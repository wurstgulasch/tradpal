"""
Parallel Backtesting Module for Multi-Symbol Analysis.
Provides multiprocessing support for running backtests on multiple symbols concurrently.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import os

from config.settings import (
    SYMBOL, EXCHANGE, TIMEFRAME, CAPITAL, PARALLEL_BACKTESTING_ENABLED,
    MAX_BACKTEST_WORKERS, BACKTEST_BATCH_SIZE
)

# Setup logging
logger = logging.getLogger(__name__)


def _run_single_backtest(args: Tuple[str, str, str, datetime, datetime, float]) -> Dict[str, Any]:
    """
    Run a single backtest (worker function for multiprocessing).
    
    This function is designed to be serializable for use with multiprocessing.
    
    Args:
        args: Tuple of (symbol, exchange, timeframe, start_date, end_date, initial_capital)
        
    Returns:
        Dictionary with backtest results
    """
    symbol, exchange, timeframe, start_date, end_date, initial_capital = args
    
    try:
        # Import here to avoid issues with multiprocessing
        from src.backtester import Backtester
        
        # Run backtest
        backtester = Backtester(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital
        )
        
        metrics = backtester.run_backtest()
        
        return {
            'symbol': symbol,
            'success': True,
            'metrics': metrics,
            'trades': backtester.trades,
            'error': None
        }
        
    except Exception as e:
        logger.error(f"Backtest failed for {symbol}: {e}")
        return {
            'symbol': symbol,
            'success': False,
            'metrics': None,
            'trades': [],
            'error': str(e)
        }


class ParallelBacktester:
    """
    Parallel backtester for running multiple symbol backtests concurrently.
    
    Features:
    - Concurrent backtest execution using multiprocessing
    - Automatic worker pool management
    - Progress tracking and reporting
    - Error handling and recovery
    """
    
    def __init__(self, symbols: List[str], exchange: str = EXCHANGE, 
                 timeframe: str = TIMEFRAME, start_date: Optional[datetime] = None,
                 end_date: Optional[datetime] = None, initial_capital: float = CAPITAL,
                 max_workers: Optional[int] = None):
        """
        Initialize parallel backtester.
        
        Args:
            symbols: List of trading symbols to backtest
            exchange: Exchange name
            timeframe: Timeframe for backtesting
            start_date: Start date for backtest period
            end_date: End date for backtest period
            initial_capital: Initial capital for each backtest
            max_workers: Maximum number of worker processes (None = auto)
        """
        self.symbols = symbols
        self.exchange = exchange
        self.timeframe = timeframe
        self.start_date = start_date or (datetime.now() - timedelta(days=30))
        self.end_date = end_date or datetime.now()
        self.initial_capital = initial_capital
        
        # Determine number of workers
        if max_workers is None or max_workers == 0:
            # Auto-detect based on CPU cores
            self.max_workers = max(1, os.cpu_count() - 1)  # Leave one core free
        else:
            self.max_workers = max_workers
        
        logger.info(f"Initialized ParallelBacktester with {len(symbols)} symbols and {self.max_workers} workers")
    
    def run_parallel_backtests(self) -> Dict[str, Any]:
        """
        Run backtests for all symbols in parallel.
        
        Returns:
            Dictionary with aggregated results for all symbols
        """
        if not PARALLEL_BACKTESTING_ENABLED:
            logger.warning("Parallel backtesting is disabled. Running sequentially.")
            return self._run_sequential_backtests()
        
        logger.info(f"Starting parallel backtests for {len(self.symbols)} symbols")
        start_time = datetime.now()
        
        # Prepare arguments for each backtest
        backtest_args = [
            (symbol, self.exchange, self.timeframe, self.start_date, self.end_date, self.initial_capital)
            for symbol in self.symbols
        ]
        
        results = []
        completed = 0
        
        # Run backtests in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(_run_single_backtest, args): args[0]
                for args in backtest_args
            }
            
            # Process completed tasks
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                completed += 1
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result['success']:
                        logger.info(f"Completed backtest for {symbol} ({completed}/{len(self.symbols)})")
                    else:
                        logger.error(f"Failed backtest for {symbol}: {result['error']}")
                        
                except Exception as e:
                    logger.error(f"Exception processing backtest for {symbol}: {e}")
                    results.append({
                        'symbol': symbol,
                        'success': False,
                        'metrics': None,
                        'trades': [],
                        'error': str(e)
                    })
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Completed all backtests in {elapsed_time:.2f} seconds")
        
        # Aggregate results
        return self._aggregate_results(results, elapsed_time)
    
    def _run_sequential_backtests(self) -> Dict[str, Any]:
        """Run backtests sequentially (fallback when parallel is disabled)."""
        logger.info(f"Starting sequential backtests for {len(self.symbols)} symbols")
        start_time = datetime.now()
        
        results = []
        for i, symbol in enumerate(self.symbols, 1):
            logger.info(f"Running backtest for {symbol} ({i}/{len(self.symbols)})")
            args = (symbol, self.exchange, self.timeframe, self.start_date, self.end_date, self.initial_capital)
            result = _run_single_backtest(args)
            results.append(result)
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Completed all backtests in {elapsed_time:.2f} seconds")
        
        return self._aggregate_results(results, elapsed_time)
    
    def _aggregate_results(self, results: List[Dict[str, Any]], elapsed_time: float) -> Dict[str, Any]:
        """
        Aggregate results from all backtests.
        
        Args:
            results: List of individual backtest results
            elapsed_time: Total execution time
            
        Returns:
            Dictionary with aggregated metrics and individual results
        """
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]
        
        if not successful_results:
            return {
                'summary': {
                    'total_symbols': len(self.symbols),
                    'successful': 0,
                    'failed': len(failed_results),
                    'execution_time_seconds': elapsed_time
                },
                'results': results,
                'aggregated_metrics': None,
                'error': 'All backtests failed'
            }
        
        # Calculate aggregated metrics
        aggregated_metrics = self._calculate_aggregated_metrics(successful_results)
        
        return {
            'summary': {
                'total_symbols': len(self.symbols),
                'successful': len(successful_results),
                'failed': len(failed_results),
                'execution_time_seconds': elapsed_time,
                'average_time_per_backtest': elapsed_time / len(self.symbols)
            },
            'results': results,
            'aggregated_metrics': aggregated_metrics,
            'best_symbol': self._get_best_symbol(successful_results),
            'worst_symbol': self._get_worst_symbol(successful_results)
        }
    
    def _calculate_aggregated_metrics(self, successful_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate aggregated metrics across all successful backtests."""
        metrics_list = [r['metrics'] for r in successful_results if 'metrics' in r and r['metrics']]
        
        if not metrics_list:
            return {}
        
        # Extract numeric metrics
        total_trades = sum(m.get('total_trades', 0) for m in metrics_list if isinstance(m, dict))
        total_winning = sum(m.get('winning_trades', 0) for m in metrics_list if isinstance(m, dict))
        total_losing = sum(m.get('losing_trades', 0) for m in metrics_list if isinstance(m, dict))
        
        # Calculate averages
        valid_metrics = [m for m in metrics_list if isinstance(m, dict) and not m.get('error')]
        
        if not valid_metrics:
            return {}
        
        avg_win_rate = np.mean([m.get('win_rate', 0) for m in valid_metrics])
        avg_profit_factor = np.mean([m.get('profit_factor', 0) for m in valid_metrics 
                                    if m.get('profit_factor') != float('inf')])
        avg_sharpe = np.mean([m.get('sharpe_ratio', 0) for m in valid_metrics])
        avg_return_pct = np.mean([m.get('return_pct', 0) for m in valid_metrics])
        avg_max_drawdown = np.mean([m.get('max_drawdown', 0) for m in valid_metrics])
        
        return {
            'total_trades_all_symbols': total_trades,
            'total_winning_trades': total_winning,
            'total_losing_trades': total_losing,
            'average_win_rate': round(avg_win_rate, 2),
            'average_profit_factor': round(avg_profit_factor, 2),
            'average_sharpe_ratio': round(avg_sharpe, 2),
            'average_return_pct': round(avg_return_pct, 2),
            'average_max_drawdown': round(avg_max_drawdown, 2)
        }
    
    def _get_best_symbol(self, successful_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Get the best performing symbol based on return percentage."""
        if not successful_results:
            return None
        
        best = max(
            successful_results,
            key=lambda r: r['metrics'].get('return_pct', float('-inf')) 
                         if isinstance(r['metrics'], dict) else float('-inf')
        )
        
        return {
            'symbol': best['symbol'],
            'return_pct': best['metrics'].get('return_pct', 0) if isinstance(best['metrics'], dict) else 0,
            'win_rate': best['metrics'].get('win_rate', 0) if isinstance(best['metrics'], dict) else 0,
            'sharpe_ratio': best['metrics'].get('sharpe_ratio', 0) if isinstance(best['metrics'], dict) else 0
        }
    
    def _get_worst_symbol(self, successful_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Get the worst performing symbol based on return percentage."""
        if not successful_results:
            return None
        
        worst = min(
            successful_results,
            key=lambda r: r['metrics'].get('return_pct', float('inf'))
                         if isinstance(r['metrics'], dict) else float('inf')
        )
        
        return {
            'symbol': worst['symbol'],
            'return_pct': worst['metrics'].get('return_pct', 0) if isinstance(worst['metrics'], dict) else 0,
            'win_rate': worst['metrics'].get('win_rate', 0) if isinstance(worst['metrics'], dict) else 0,
            'sharpe_ratio': worst['metrics'].get('sharpe_ratio', 0) if isinstance(worst['metrics'], dict) else 0
        }


def run_parallel_backtests(symbols: List[str], exchange: str = EXCHANGE,
                          timeframe: str = TIMEFRAME, start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None, initial_capital: float = CAPITAL,
                          max_workers: Optional[int] = None) -> Dict[str, Any]:
    """
    Convenience function to run parallel backtests.
    
    Args:
        symbols: List of trading symbols
        exchange: Exchange name
        timeframe: Timeframe for backtesting
        start_date: Start date for backtest period
        end_date: End date for backtest period
        initial_capital: Initial capital for each backtest
        max_workers: Maximum number of worker processes
        
    Returns:
        Dictionary with aggregated results
    """
    backtester = ParallelBacktester(
        symbols=symbols,
        exchange=exchange,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        max_workers=max_workers
    )
    
    return backtester.run_parallel_backtests()
