#!/usr/bin/env python3
"""
Debug script to check if signals are generated with best discovery config
"""

import sys
import os
import json
sys.path.insert(0, os.path.dirname(__file__))

from src.data_fetcher import fetch_historical_data
from src.indicators import calculate_indicators
from src.signal_generator import generate_signals

def main():
    print('🔍 Debugging Signal Generation with Best Discovery Configuration')

    # Load the best configuration
    try:
        with open('output/discovery_results.json', 'r') as f:
            discovery_data = json.load(f)
        best_config = discovery_data['top_configurations'][0]['configuration']
        print('✅ Loaded best configuration')
    except Exception as e:
        print(f'❌ Could not load discovery results: {e}')
        return

    # Fetch a small amount of data for testing
    print('📊 Fetching test data...')
    data = fetch_historical_data('BTC/USDT', '1h', limit=100)
    if data is None or data.empty:
        print('❌ No data fetched')
        return

    print(f'✅ Fetched {len(data)} rows of data')

    # Calculate indicators with best config
    print('📈 Calculating indicators...')
    data = calculate_indicators(data, config=best_config)

    # Check if configured EMA columns exist
    ema_periods = best_config.get('ema', {}).get('periods', [9, 21])
    ema_cols = [f'EMA{p}' for p in ema_periods]
    print(f'🔍 Checking for EMA columns: {ema_cols}')
    for col in ema_cols:
        if col in data.columns:
            print(f'✅ {col} exists')
        else:
            print(f'❌ {col} missing')

    # Generate signals
    print('📊 Generating signals...')
    data = generate_signals(data, config=best_config)

    # Check signal counts
    buy_signals = (data['Buy_Signal'] == 1).sum()
    sell_signals = (data['Sell_Signal'] == 1).sum()

    print(f'📊 Signal Summary:')
    print(f'   Buy Signals: {buy_signals}')
    print(f'   Sell Signals: {sell_signals}')
    print(f'   Total Signals: {buy_signals + sell_signals}')

    if buy_signals + sell_signals > 0:
        print('✅ Signals generated successfully!')
    else:
        print('❌ No signals generated')

        # Debug: Check indicator validity
        print('🔍 Debugging indicator validity...')
        ema_short_col = f'EMA{ema_periods[0]}' if len(ema_periods) > 0 else 'EMA9'
        ema_long_col = f'EMA{ema_periods[1]}' if len(ema_periods) > 1 else 'EMA21'

        valid_ema_short = data[ema_short_col].notna().sum() if ema_short_col in data.columns else 0
        valid_ema_long = data[ema_long_col].notna().sum() if ema_long_col in data.columns else 0
        valid_rsi = data['RSI'].notna().sum()
        valid_bb_lower = data['BB_lower'].notna().sum()
        valid_bb_upper = data['BB_upper'].notna().sum()

        print(f'   Valid {ema_short_col}: {valid_ema_short}/{len(data)}')
        print(f'   Valid {ema_long_col}: {valid_ema_long}/{len(data)}')
        print(f'   Valid RSI: {valid_rsi}/{len(data)}')
        print(f'   Valid BB_lower: {valid_bb_lower}/{len(data)}')
        print(f'   Valid BB_upper: {valid_bb_upper}/{len(data)}')

        # Check EMA crossover
        if ema_short_col in data.columns and ema_long_col in data.columns:
            crossover_buy = (data[ema_short_col] > data[ema_long_col]).sum()
            crossover_sell = (data[ema_short_col] < data[ema_long_col]).sum()
            print(f'   EMA Crossover Buy: {crossover_buy}')
            print(f'   EMA Crossover Sell: {crossover_sell}')

if __name__ == '__main__':
    main()