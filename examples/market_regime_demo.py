#!/usr/bin/env python3
"""
Market Regime Analysis Demo for TradPal

This demo showcases the market regime detection and multi-timeframe analysis
capabilities of the advanced ML trading system.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from services.mlops_service.market_regime_analysis import (
    MarketRegimeDetector, MultiTimeframeAnalyzer, AdaptiveStrategyManager,
    detect_market_regime, analyze_multi_timeframe, get_adaptive_strategy_config,
    MarketRegime
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_multi_timeframe_data(base_samples=500):
    """Create synthetic multi-timeframe trading data."""
    logger.info("Creating multi-timeframe synthetic data...")

    # Create base timeframe (1h)
    dates_1h = pd.date_range('2023-01-01', periods=base_samples, freq='H')
    data_1h = create_timeframe_data(dates_1h, volatility=0.02)

    # Create 4h timeframe (aggregate from 1h)
    dates_4h = pd.date_range('2023-01-01', periods=base_samples//4, freq='4H')
    data_4h = create_timeframe_data(dates_4h, volatility=0.03)

    # Create 1d timeframe (aggregate from 1h)
    dates_1d = pd.date_range('2023-01-01', periods=base_samples//24, freq='D')
    data_1d = create_timeframe_data(dates_1d, volatility=0.04)

    # Create 15m timeframe (more frequent)
    dates_15m = pd.date_range('2023-01-01', periods=base_samples*4, freq='15min')
    data_15m = create_timeframe_data(dates_15m, volatility=0.015)

    return {
        '1h': data_1h,
        '4h': data_4h,
        '1d': data_1d,
        '15m': data_15m
    }

def create_timeframe_data(dates, volatility=0.02):
    """Create synthetic OHLCV data for a specific timeframe."""
    n_samples = len(dates)

    # Generate base price with trends and cycles
    np.random.seed(42)
    base_price = 100

    # Create different market phases
    phase_length = n_samples // 4

    # Phase 1: Uptrend
    prices1 = [base_price]
    for i in range(1, phase_length):
        trend = 0.001  # Upward trend
        noise = np.random.normal(0, volatility)
        new_price = prices1[-1] * (1 + trend + noise)
        prices1.append(new_price)

    # Phase 2: Sideways
    prices2 = [prices1[-1]]
    for i in range(phase_length):
        noise = np.random.normal(0, volatility * 0.5)  # Lower volatility
        new_price = prices2[-1] * (1 + noise)
        prices2.append(new_price)

    # Phase 3: Downtrend
    prices3 = [prices2[-1]]
    for i in range(phase_length):
        trend = -0.001  # Downward trend
        noise = np.random.normal(0, volatility)
        new_price = prices3[-1] * (1 + trend + noise)
        prices3.append(new_price)

    # Phase 4: Volatile breakout
    prices4 = [prices3[-1]]
    for i in range(phase_length):
        trend = 0.002  # Strong uptrend
        noise = np.random.normal(0, volatility * 2)  # High volatility
        new_price = prices4[-1] * (1 + trend + noise)
        prices4.append(new_price)

    # Combine all phases
    prices = prices1 + prices2[1:] + prices3[1:] + prices4[1:]

    # Ensure we have the right number of samples
    if len(prices) > n_samples:
        prices = prices[:n_samples]
    elif len(prices) < n_samples:
        # Pad with last price
        prices.extend([prices[-1]] * (n_samples - len(prices)))

    # Create OHLCV data
    data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, volatility*0.5))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, volatility*0.5))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(100000, 1000000, n_samples)
    }, index=dates[:n_samples])

    return data

def demo_regime_detection():
    """Demonstrate market regime detection."""
    logger.info("=== Market Regime Detection Demo ===")

    # Create sample data
    data = create_timeframe_data(pd.date_range('2023-01-01', periods=300, freq='H'))

    # Detect regimes
    regime_detector = MarketRegimeDetector()
    regimes = regime_detector.detect_regime(data)

    logger.info(f"Detected regimes: {regimes.value_counts().to_dict()}")

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # Price chart
    ax1.plot(data.index, data['close'], label='Close Price', alpha=0.7)
    ax1.set_title('Price Data with Detected Regimes')
    ax1.set_ylabel('Price')
    ax1.legend()

    # Regime chart
    regime_colors = {
        MarketRegime.TREND_UP: 'green',
        MarketRegime.TREND_DOWN: 'red',
        MarketRegime.MEAN_REVERSION: 'blue',
        MarketRegime.HIGH_VOLATILITY: 'orange',
        MarketRegime.LOW_VOLATILITY: 'purple',
        MarketRegime.SIDEWAYS: 'gray',
        MarketRegime.BREAKOUT: 'cyan',
        MarketRegime.CONSOLIDATION: 'brown'
    }

    for regime in regimes.unique():
        mask = regimes == regime
        ax2.fill_between(data.index, 0, 1, where=mask,
                        color=regime_colors.get(regime, 'gray'),
                        alpha=0.3, label=f'{regime.value}')

    ax2.set_title('Market Regimes Over Time')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Regime')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig('regime_detection_demo.png', dpi=150, bbox_inches='tight')
    plt.show()

    return regimes

def demo_multi_timeframe_analysis():
    """Demonstrate multi-timeframe analysis."""
    logger.info("\n=== Multi-Timeframe Analysis Demo ===")

    # Create multi-timeframe data
    data_dict = create_multi_timeframe_data(200)

    # Perform analysis
    mtf_analyzer = MultiTimeframeAnalyzer()
    analysis = mtf_analyzer.analyze_multi_timeframe(data_dict)

    logger.info(f"Consensus regime: {analysis['consensus_regime'].value}")
    logger.info(".3f")
    logger.info(f"Timeframe strength: {analysis['timeframe_strength']}")

    # Plot regime alignment
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot each timeframe's regime
    timeframes = list(data_dict.keys())
    for i, tf in enumerate(timeframes[:4]):  # Max 4 subplots
        ax = axes[i//2, i%2]
        data = data_dict[tf]
        regimes = analysis['regime_results'][tf]

        ax.plot(data.index, data['close'], alpha=0.7, label='Price')
        ax.set_title(f'{tf} Timeframe - {regimes.iloc[-1].value}')
        ax.legend()

    plt.tight_layout()
    plt.savefig('multi_timeframe_demo.png', dpi=150, bbox_inches='tight')
    plt.show()

    return analysis

def demo_adaptive_strategy():
    """Demonstrate adaptive strategy configuration."""
    logger.info("\n=== Adaptive Strategy Demo ===")

    # Create multi-timeframe data
    data_dict = create_multi_timeframe_data(100)

    # Get adaptive configuration
    adaptive_manager = AdaptiveStrategyManager()
    config = adaptive_manager.get_adaptive_config(data_dict)

    logger.info("Adaptive Strategy Configuration:")
    logger.info(f"  Current Regime: {config['current_regime'].value}")
    logger.info(".2f")
    logger.info(".2f")
    logger.info(f"  Model Type: {config['model_type']}")
    logger.info(".3f")
    logger.info(".3f")
    logger.info(f"  Features: {config['features']}")

    # Show regime-specific configurations
    logger.info("\nRegime-specific configurations:")
    for regime, regime_config in adaptive_manager.strategy_configs.items():
        logger.info(f"  {regime.value}: {regime_config['model_type']}, position_size={regime_config['position_size']}")

    return config

def demo_regime_statistics():
    """Demonstrate regime statistics and transitions."""
    logger.info("\n=== Regime Statistics Demo ===")

    # Create longer sample data
    data = create_timeframe_data(pd.date_range('2023-01-01', periods=1000, freq='H'))
    regimes = detect_market_regime(data)

    # Calculate regime statistics
    regime_counts = regimes.value_counts()
    regime_percentages = (regime_counts / len(regimes) * 100).round(2)

    logger.info("Regime Distribution:")
    for regime, count in regime_counts.items():
        percentage = regime_percentages[regime]
        logger.info(f"  {regime.value}: {count} periods ({percentage}%)")

    # Calculate regime transitions
    transitions = {}
    prev_regime = None
    for regime in regimes:
        if prev_regime is not None:
            key = (prev_regime, regime)
            transitions[key] = transitions.get(key, 0) + 1
        prev_regime = regime

    logger.info("\nTop Regime Transitions:")
    sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
    for (from_regime, to_regime), count in sorted_transitions[:10]:
        logger.info(f"  {from_regime.value} → {to_regime.value}: {count} times")

    # Calculate average regime duration
    current_regime = None
    current_count = 0
    durations = []

    for regime in regimes:
        if regime == current_regime:
            current_count += 1
        else:
            if current_regime is not None:
                durations.append(current_count)
            current_regime = regime
            current_count = 1

    if durations:
        avg_duration = np.mean(durations)
        logger.info(".1f")

    return {
        'regime_counts': regime_counts,
        'transitions': transitions,
        'durations': durations
    }

def main():
    """Main demo function."""
    logger.info("TradPal Market Regime Analysis Demo")
    logger.info("=" * 50)

    try:
        # Regime detection demo
        regimes = demo_regime_detection()

        # Multi-timeframe analysis demo
        mtf_analysis = demo_multi_timeframe_analysis()

        # Adaptive strategy demo
        adaptive_config = demo_adaptive_strategy()

        # Regime statistics demo
        regime_stats = demo_regime_statistics()

        logger.info("\n" + "=" * 50)
        logger.info("Demo Complete!")
        logger.info("Generated files:")
        logger.info("- regime_detection_demo.png")
        logger.info("- multi_timeframe_demo.png")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise

if __name__ == '__main__':
    main()