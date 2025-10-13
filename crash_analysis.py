import pandas as pd
from datetime import datetime
from src.backtester import Backtester
from src.data_fetcher import fetch_historical_data
from src.indicators import calculate_indicators
from src.signal_generator import generate_signals, calculate_risk_management

print('=== COMPREHENSIVE BITCOIN CRASH BACKTEST ANALYSIS ===')
print()

# Fetch data for the crash period (same as main.py)
print('1. Fetching data for crash period (2025-09-01 to 2025-10-15)...')
data = fetch_historical_data(
    symbol='BTC/USDT',
    exchange_name='kraken',
    timeframe='1d',
    limit=1000,  # More data to ensure we get the full period
    start_date=datetime(2025, 9, 1)
)

print(f'Data fetched: {len(data)} rows')
print(f'Date range: {data.index.min()} to {data.index.max()}')

# Calculate indicators and signals
print()
print('2. Calculating indicators and generating signals...')
data = calculate_indicators(data)
data = generate_signals(data)
data = calculate_risk_management(data)

# Analyze the crash period
print()
print('3. Analyzing Bitcoin crash period...')
# Find crash period manually by looking at price drops
print('Looking for significant price drops (potential crash periods):')
prices = data['close'].values
for i in range(1, len(prices)):
    if prices[i] < prices[i-1] * 0.95:  # 5% drop
        row = data.iloc[i]
        drop_pct = ((prices[i]/prices[i-1])-1)*100
        print(f'  Price drop at row {i}: ${prices[i-1]:.2f} -> ${prices[i]:.2f} ({drop_pct:.1f}%)')
        print(f'    RSI: {row["RSI"]:.1f}, BB_lower: ${row["BB_lower"]:.2f}, Buy_Signal: {row["Buy_Signal"]}')

# Check all signals
total_buy_signals = data['Buy_Signal'].sum()
total_sell_signals = data['Sell_Signal'].sum()
print(f'Total signals in period: {total_buy_signals} BUY, {total_sell_signals} SELL')

# Show recent data (last 10 rows) to see crash
print()
print('Recent data (last 10 rows):')
recent_data = data.tail(10)
for i, (idx, row) in enumerate(recent_data.iterrows()):
    buy_sig = 'BUY' if row['Buy_Signal'] == 1 else ''
    sell_sig = 'SELL' if row['Sell_Signal'] == 1 else ''
    signal = f'{buy_sig}{sell_sig}' if buy_sig or sell_sig else 'HOLD'
    row_num = len(data) - 10 + i
    print(f'  Row {row_num}: ${row["close"]:.2f} (RSI: {row["RSI"]:.1f}, BB_lower: ${row["BB_lower"]:.2f}) [{signal}]')

# Run backtest
print()
print('4. Running backtest...')
backtester = Backtester('BTC/USDT', 'kraken', '1d', datetime(2025, 9, 1), datetime(2025, 10, 15))
result = backtester.run_backtest(df=data, strategy='traditional')

# Analyze results
print()
print('5. Backtest Results:')
if result['success'] and 'metrics' in result:
    metrics = result['metrics']
    trades_count = metrics.get('total_trades', 0)

    print(f'Total Trades: {trades_count}')
    if trades_count > 0:
        print(f'Win Rate: {metrics.get("win_rate", 0):.1f}%')
        print(f'Total P&L: ${metrics.get("total_pnl", 0):.2f}')
        print(f'Final Capital: ${metrics.get("final_capital", 0):.2f}')
        print(f'Return: {metrics.get("return_pct", 0):.2f}%')
        print(f'Max Drawdown: {metrics.get("max_drawdown", 0):.2f}%')

        # Show trades
        trades = result.get('trades', [])
        print(f'Trade Details ({len(trades)} trades):')
        for i, trade in enumerate(trades):
            entry_date = pd.Timestamp(trade['entry_time']).date() if isinstance(trade['entry_time'], (str, pd.Timestamp)) else 'Unknown'
            exit_date = pd.Timestamp(trade['exit_time']).date() if isinstance(trade['exit_time'], (str, pd.Timestamp)) else 'Unknown'
            print(f'  {i+1}. {trade["type"].upper()} {entry_date} @ ${trade["entry_price"]:.2f} -> {exit_date} @ ${trade["exit_price"]:.2f} (P&L: ${trade["pnl"]:.2f})')

        # Compare to buy-and-hold
        if len(data) > 1:
            start_price = data.iloc[0]['close']
            end_price = data.iloc[-1]['close']
            buy_hold_return = ((end_price - start_price) / start_price) * 100
            print(f'Buy-and-Hold Return: {buy_hold_return:.2f}%')
            outperformance = metrics.get('return_pct', 0) - buy_hold_return
            print(f'Strategy vs Buy-and-Hold: {outperformance:.2f}%')
    else:
        print('No trades executed during backtest period')
else:
    print(f'Backtest failed: {result.get("error", "Unknown error")}')

print()
print('=== ANALYSIS COMPLETE ===')