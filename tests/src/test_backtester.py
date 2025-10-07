import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.backtester import run_backtest, calculate_performance_metrics, simulate_trades


class TestBacktester:
    """Test backtesting functionality."""

    def test_calculate_performance_metrics_basic(self):
        """Test basic performance metrics calculation."""
        # Create mock trade data
        trades = pd.DataFrame({
            'entry_price': [100, 110, 105],
            'exit_price': [105, 108, 107],
            'position_size': [1000, 1000, 1000],
            'direction': ['long', 'short', 'long']
        })

        capital = 10000
        metrics = calculate_performance_metrics(trades, capital)

        assert 'total_trades' in metrics
        assert 'winning_trades' in metrics
        assert 'losing_trades' in metrics
        assert 'win_rate' in metrics
        assert 'total_pnl' in metrics
        assert 'gross_profit' in metrics
        assert 'gross_loss' in metrics
        assert 'profit_factor' in metrics
        assert 'max_drawdown' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'final_capital' in metrics

        assert metrics['total_trades'] == 3
        assert metrics['win_rate'] == 100.0  # All trades are profitable
        assert metrics['final_capital'] == capital + metrics['total_pnl']

    def test_calculate_performance_metrics_with_losses(self):
        """Test metrics calculation with losing trades."""
        trades = pd.DataFrame({
            'entry_price': [100, 110, 105],
            'exit_price': [95, 115, 103],  # 2 wins, 1 loss
            'position_size': [1000, 1000, 1000],
            'direction': ['long', 'short', 'long']
        })

        capital = 10000
        metrics = calculate_performance_metrics(trades, capital)

        assert metrics['total_trades'] == 3
        assert metrics['winning_trades'] == 0
        assert metrics['losing_trades'] == 3
        assert metrics['win_rate'] == 0.0
        assert metrics['total_pnl'] < 0  # Overall loss

    def test_calculate_performance_metrics_empty_trades(self):
        """Test metrics calculation with no trades."""
        trades = pd.DataFrame()
        capital = 10000

        metrics = calculate_performance_metrics(trades, capital)

        assert metrics['total_trades'] == 0
        assert metrics['winning_trades'] == 0
        assert metrics['losing_trades'] == 0
        assert metrics['win_rate'] == 0.0
        assert metrics['total_pnl'] == 0.0
        assert metrics['final_capital'] == capital

    def test_simulate_trades_basic(self):
        """Test basic trade simulation."""
        # Create test data with signals
        data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10, freq='1min'),
            'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'high': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
            'volume': [1000] * 10,
            'Buy_Signal': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Buy at first candle
            'Sell_Signal': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # Sell at last candle
            'Stop_Loss_Buy': [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
            'Take_Profit_Buy': [103, 104, 105, 106, 107, 108, 109, 110, 111, 112]
        })

        trades = simulate_trades(data)

        assert isinstance(trades, pd.DataFrame)
        assert len(trades) == 1  # One complete trade

        trade = trades.iloc[0]
        assert trade['entry_price'] == 100.5  # Close price at buy signal
        assert trade['exit_price'] == 109.5   # Close price at sell signal
        assert trade['direction'] == 'long'
        assert trade['pnl'] > 0  # Should be profitable

    def test_simulate_trades_no_signals(self):
        """Test simulation with no trading signals."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5, freq='1min'),
            'open': [100, 101, 102, 103, 104],
            'high': [101, 102, 103, 104, 105],
            'low': [99, 100, 101, 102, 103],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'volume': [1000] * 5,
            'Buy_Signal': [0, 0, 0, 0, 0],
            'Sell_Signal': [0, 0, 0, 0, 0]
        })

        trades = simulate_trades(data)

        assert isinstance(trades, pd.DataFrame)
        assert len(trades) == 0  # No trades

    def test_simulate_trades_stop_loss_hit(self):
        """Test simulation with stop loss being hit."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5, freq='1min'),
            'open': [100, 101, 102, 103, 104],
            'high': [101, 102, 103, 104, 105],
            'low': [99, 100, 98, 97, 96],  # Price drops below stop loss at index 2
            'close': [100.5, 101.5, 98.5, 97.5, 96.5],
            'volume': [1000] * 5,
            'Buy_Signal': [1, 0, 0, 0, 0],  # Buy at first candle
            'Sell_Signal': [0, 0, 0, 0, 0],
            'Stop_Loss_Buy': [99, 99, 99, 99, 99],  # Stop loss at 99
            'Take_Profit_Buy': [110, 110, 110, 110, 110]
        })

        trades = simulate_trades(data)

        assert isinstance(trades, pd.DataFrame)
        assert len(trades) == 1

        trade = trades.iloc[0]
        assert trade['exit_price'] == 99.0  # Stop loss price
        assert trade['pnl'] < 0  # Should be a loss

    def test_simulate_trades_take_profit_hit(self):
        """Test simulation with take profit being hit."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5, freq='1min'),
            'open': [100, 101, 102, 103, 104],
            'high': [101, 102, 111, 112, 113],  # High reaches take profit at index 2
            'low': [99, 100, 101, 102, 103],
            'close': [100.5, 101.5, 110.5, 111.5, 112.5],
            'volume': [1000] * 5,
            'Buy_Signal': [1, 0, 0, 0, 0],  # Buy at first candle
            'Sell_Signal': [0, 0, 0, 0, 0],
            'Stop_Loss_Buy': [95, 95, 95, 95, 95],
            'Take_Profit_Buy': [110, 110, 110, 110, 110]  # Take profit at 110
        })

        trades = simulate_trades(data)

        assert isinstance(trades, pd.DataFrame)
        assert len(trades) == 1

        trade = trades.iloc[0]
        assert trade['exit_price'] == 110.0  # Take profit price (high)
        assert trade['pnl'] > 0  # Should be profitable

    @patch('src.backtester.fetch_historical_data')
    @patch('src.backtester.calculate_indicators')
    @patch('src.backtester.generate_signals')
    @patch('src.backtester.calculate_risk_management')
    def test_run_backtest_integration(self, mock_risk, mock_signals, mock_indicators, mock_fetch_data):
        """Test complete backtest integration."""
        # Mock historical data with signals already included
        mock_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='1min'),
            'open': np.random.randn(100) + 100,
            'high': np.random.randn(100) + 101,
            'low': np.random.randn(100) + 99,
            'close': np.random.randn(100) + 100,
            'volume': np.random.randint(1000, 10000, 100),
            'EMA9': np.random.randn(100) + 100.5,
            'EMA21': np.random.randn(100) + 100.3,
            'RSI': np.random.uniform(20, 80, 100),
            'BB_upper': np.random.randn(100) + 101.5,
            'BB_middle': np.random.randn(100) + 100.5,
            'BB_lower': np.random.randn(100) + 99.5,
            'ATR': np.random.uniform(0.1, 2.0, 100),
            'Buy_Signal': [1] + [0]*98 + [0],  # Buy at start, sell at end
            'Sell_Signal': [0]*99 + [1],
            'Position_Size_Absolute': [1000] * 100,
            'Stop_Loss_Buy': [98] * 100,
            'Take_Profit_Buy': [102] * 100
        })
        mock_fetch_data.return_value = mock_data

        # Run backtest
        results = run_backtest(
            symbol='EUR/USD',
            timeframe='1m',
            start_date='2023-01-01',
            end_date='2023-01-02'
        )

        assert 'backtest_results' in results
        assert 'trades' in results

        metrics = results['backtest_results']
        assert 'total_trades' in metrics
        assert 'win_rate' in metrics
        assert 'total_pnl' in metrics
        assert 'sharpe_ratio' in metrics

    def test_calculate_performance_metrics_edge_cases(self):
        """Test performance metrics with edge cases."""
        # Test with very small capital
        trades = pd.DataFrame({
            'entry_price': [100],
            'exit_price': [101],
            'position_size': [1],  # Very small position
            'direction': ['long']
        })

        metrics = calculate_performance_metrics(trades, 100)  # Small capital

        assert metrics['total_pnl'] == 0.01  # $0.01 profit (1% of position value)
        assert metrics['final_capital'] == 100.01

    def test_simulate_trades_multiple_signals(self):
        """Test simulation with multiple buy/sell signals."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10, freq='1min'),
            'open': range(100, 110),
            'high': range(101, 111),
            'low': range(99, 109),
            'close': range(100, 110),  # Simple increasing prices
            'volume': [1000] * 10,
            'Buy_Signal': [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # Two buy signals
            'Sell_Signal': [0, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # Two sell signals
            'Stop_Loss_Buy': [95] * 10,
            'Take_Profit_Buy': [120] * 10
        })

        trades = simulate_trades(data)

        assert len(trades) == 2  # Two complete trades

    def test_calculate_performance_metrics_risk_metrics(self):
        """Test calculation of risk-related metrics."""
        # Create trades with drawdown scenario
        trades = pd.DataFrame({
            'entry_price': [100, 95, 110],  # Prices going down then up
            'exit_price': [95, 110, 105],   # Losses then profits
            'position_size': [1000, 1000, 1000],
            'direction': ['long', 'long', 'short']
        })

        capital = 10000
        metrics = calculate_performance_metrics(trades, capital)

        # Check that drawdown is calculated
        assert 'max_drawdown' in metrics
        assert 'max_drawdown_percentage' in metrics
        assert metrics['max_drawdown'] >= 0
        assert metrics['max_drawdown_percentage'] >= 0

    def test_run_backtest_invalid_dates(self):
        """Test backtest with invalid date range."""
        with pytest.raises(ValueError):
            run_backtest(
                symbol='EUR/USD',
                timeframe='1m',
                start_date='2023-12-31',  # Start after end
                end_date='2023-01-01'
            )

    def test_simulate_trades_empty_data(self):
        """Test trade simulation with empty data."""
        data = pd.DataFrame()

        trades = simulate_trades(data)

        assert isinstance(trades, pd.DataFrame)
        assert len(trades) == 0


class TestBacktesterIntegration:
    """Integration tests for backtesting system."""

    @patch('src.backtester.fetch_historical_data')
    @patch('src.backtester.calculate_indicators')
    @patch('src.backtester.generate_signals')
    @patch('src.backtester.calculate_risk_management')
    def test_full_backtest_pipeline(self, mock_risk, mock_signals, mock_indicators, mock_fetch):
        """Test the complete backtest pipeline."""
        # Mock all dependencies
        base_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10, freq='1min'),
            'open': range(100, 110),
            'high': range(101, 111),
            'low': range(99, 109),
            'close': range(100, 110),
            'volume': [1000] * 10
        })

        mock_fetch.return_value = base_data

        # Mock indicators to add indicator columns
        def mock_indicators_func(data):
            return data.assign(
                EMA9=[100 + i*0.1 for i in range(10)],
                EMA21=[100 + i*0.05 for i in range(10)],
                RSI=[50 + i for i in range(10)],
                BB_upper=[105 + i*0.1 for i in range(10)],
                BB_middle=[100 + i*0.1 for i in range(10)],
                BB_lower=[95 + i*0.1 for i in range(10)],
                ATR=[1 + i*0.01 for i in range(10)]
            )
        mock_indicators.side_effect = mock_indicators_func

        # Mock signals to add signal columns
        def mock_signals_func(data):
            return data.assign(
                Buy_Signal=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                Sell_Signal=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
            )
        mock_signals.side_effect = mock_signals_func

        # Mock risk management to add risk columns
        def mock_risk_func(data):
            return data.assign(
                Position_Size_Absolute=[1000] * 10,
                Stop_Loss_Buy=[95 + i*0.1 for i in range(10)],
                Take_Profit_Buy=[120 + i*0.1 for i in range(10)]
            )
        mock_risk.side_effect = mock_risk_func

        results = run_backtest('EUR/USD', '1m', '2023-01-01', '2023-01-02')

        assert 'backtest_results' in results
        assert 'trades' in results
        assert isinstance(results['trades'], pd.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__])