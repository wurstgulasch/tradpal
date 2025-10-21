"""
Optimizer Service

Enhanced optimization service with:
- Walk-forward optimization with advanced metrics
- Information Coefficient tracking
- Bias-Variance tradeoff analysis
- Overfitting detection
"""

import sys
import os
from pathlib import Path
import argparse
import json
from datetime import datetime, timedelta

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.data_service.data_fetcher import fetch_historical_data
from services.core.indicators import calculate_indicators
from services.backtesting_service.optimization.walk_forward_optimizer import WalkForwardOptimizer, get_walk_forward_optimizer
from config.settings import SYMBOL, TIMEFRAME, LOOKBACK_DAYS


def run_walk_forward_optimization(symbol: str, timeframe: str, start_date: str,
                                  end_date: str, evaluation_metric: str = 'sharpe_ratio'):
    """
    Run walk-forward optimization with enhanced metrics.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        start_date: Start date
        end_date: End date
        evaluation_metric: Metric to optimize
    """
    print(f"🔍 Starting walk-forward optimization for {symbol} on {timeframe}")
    print(f"Evaluation metric: {evaluation_metric}")
    
    # Fetch data
    print(f"📈 Fetching data from {start_date} to {end_date}...")
    df = fetch_historical_data(
        symbol=symbol,
        timeframe=timeframe,
        limit=10000,
        start_date=datetime.fromisoformat(start_date) if start_date else None,
        show_progress=False
    )
    
    if df is None or len(df) == 0:
        print("❌ Failed to fetch data")
        return None
    
    print(f"✅ Fetched {len(df)} candles")
    
    # Calculate indicators
    print("🔧 Calculating indicators...")
    df = calculate_indicators(df)
    
    # Get optimizer
    optimizer = get_walk_forward_optimizer(symbol=symbol, timeframe=timeframe)
    
    # Create walk-forward windows
    print("📊 Creating walk-forward windows...")
    windows = optimizer.create_walk_forward_windows(
        df=df,
        initial_train_size=1000,
        test_size=200,
        step_size=50,
        min_samples=500
    )
    
    print(f"✅ Created {len(windows)} windows")
    
    # Define parameter grid for optimization
    parameter_grid = {
        'ema_short': [9, 12, 15, 20],
        'ema_long': [21, 26, 30, 50],
        'rsi_period': [14],
        'rsi_oversold': [25, 30, 35],
        'rsi_overbought': [65, 70, 75],
        'atr_period': [14],
        'risk_per_trade': [0.01, 0.02]
    }
    
    print(f"🎯 Parameter grid: {len([1 for _ in parameter_grid])} parameters")
    
    # Run optimization
    print("\n🚀 Starting optimization...")
    results = optimizer.optimize_strategy_parameters(
        df=df,
        parameter_grid=parameter_grid,
        evaluation_metric=evaluation_metric,
        min_trades=10
    )
    
    # Display enhanced metrics
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)
    
    analysis = results.get('analysis', {})
    
    print(f"\n📊 Performance Metrics:")
    print(f"  Average OOS Performance: {analysis.get('average_oos_performance', 0):.4f}")
    print(f"  Std OOS Performance: {analysis.get('std_oos_performance', 0):.4f}")
    print(f"  Average IS Performance: {analysis.get('average_is_performance', 0):.4f}")
    print(f"  Performance Decay: {analysis.get('performance_decay', 0):.4f}")
    
    # Enhanced metrics
    print(f"\n🎓 Advanced Metrics:")
    print(f"  Information Coefficient: {analysis.get('information_coefficient', 'N/A')}")
    print(f"  Overfitting Ratio: {analysis.get('overfitting_ratio', 'N/A')}")
    print(f"  Consistency Score: {analysis.get('consistency_score', 'N/A')}")
    
    # Bias-Variance analysis
    bias_variance = analysis.get('bias_variance', {})
    if bias_variance:
        print(f"\n⚖️  Bias-Variance Tradeoff:")
        print(f"  Bias: {bias_variance.get('bias', 'N/A')}")
        print(f"  Variance: {bias_variance.get('variance', 'N/A')}")
        print(f"  Total Error: {bias_variance.get('total_error', 'N/A')}")
        print(f"  Bias/Variance Ratio: {bias_variance.get('bias_variance_ratio', 'N/A')}")
        print(f"  Interpretation: {bias_variance.get('interpretation', 'N/A')}")
    
    # Robustness
    robustness = analysis.get('robustness', {})
    if robustness:
        print(f"\n💪 Robustness:")
        print(f"  Positive Windows: {robustness.get('positive_windows', 0)}")
        print(f"  Positive Ratio: {robustness.get('positive_ratio', 0):.2%}")
    
    # Save results
    results_dir = Path("output/walk_forward")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = results_dir / f"walk_forward_{symbol}_{timeframe}_{timestamp}.json"
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to {results_path}")
    
    # Generate interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    interpret_results(analysis)
    
    return results


def interpret_results(analysis: dict):
    """
    Provide human-readable interpretation of results.
    
    Args:
        analysis: Analysis dictionary from walk-forward optimization
    """
    performance_decay = analysis.get('performance_decay', 0)
    overfitting_ratio = analysis.get('overfitting_ratio')
    ic = analysis.get('information_coefficient')
    consistency = analysis.get('consistency_score')
    
    print("\n📝 Strategy Assessment:")
    
    # Overfitting check
    if overfitting_ratio is not None:
        if overfitting_ratio > 0.5:
            print("⚠️  HIGH OVERFITTING RISK")
            print("   Strategy performs significantly better in-sample than out-of-sample.")
            print("   Consider: simplifying model, adding regularization, or more data.")
        elif overfitting_ratio > 0.2:
            print("⚡ MODERATE OVERFITTING")
            print("   Some overfitting detected but within acceptable range.")
            print("   Monitor performance in live trading closely.")
        else:
            print("✅ LOW OVERFITTING")
            print("   Strategy generalizes well to unseen data.")
    
    # Information Coefficient
    if ic is not None:
        print(f"\n📊 Predictive Power (IC: {ic:.3f}):")
        if ic > 0.5:
            print("✅ STRONG predictive relationship between in-sample and out-of-sample")
        elif ic > 0.2:
            print("⚡ MODERATE predictive relationship")
        elif ic > 0:
            print("⚠️  WEAK predictive relationship")
        else:
            print("❌ NO or NEGATIVE predictive relationship")
    
    # Consistency
    if consistency is not None:
        print(f"\n🎯 Consistency Score: {consistency:.3f}")
        if consistency > 0.7:
            print("✅ HIGH consistency - results are stable across different periods")
        elif consistency > 0.4:
            print("⚡ MODERATE consistency - some variation in performance")
        else:
            print("⚠️  LOW consistency - highly variable performance")
    
    # Overall recommendation
    print("\n💡 Recommendation:")
    
    if overfitting_ratio and overfitting_ratio > 0.5:
        print("❌ NOT RECOMMENDED for live trading")
        print("   High overfitting risk - strategy unlikely to perform as expected")
    elif ic and ic < 0.1:
        print("⚠️  CAUTION ADVISED")
        print("   Weak predictive power - consider refining strategy")
    elif consistency and consistency < 0.3:
        print("⚠️  USE WITH CAUTION")
        print("   Inconsistent performance - may work in some conditions but not others")
    else:
        print("✅ ACCEPTABLE for further testing")
        print("   Strategy shows reasonable generalization")
        print("   Recommended: paper trading before live deployment")


def main():
    """Main optimizer service entry point."""
    parser = argparse.ArgumentParser(description="Optimizer Service")
    parser.add_argument('--symbol', type=str, default=SYMBOL, help='Trading symbol')
    parser.add_argument('--timeframe', type=str, default=TIMEFRAME, help='Timeframe')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--lookback-days', type=int, default=LOOKBACK_DAYS,
                       help='Days of historical data')
    parser.add_argument('--metric', type=str, default='sharpe_ratio',
                       choices=['sharpe_ratio', 'win_rate', 'profit_factor', 'total_return'],
                       help='Evaluation metric')
    
    args = parser.parse_args()
    
    # Calculate dates if not provided
    if not args.end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    else:
        end_date = args.end_date
    
    if not args.start_date:
        start_date = (datetime.now() - timedelta(days=args.lookback_days)).strftime('%Y-%m-%d')
    else:
        start_date = args.start_date
    
    print("=" * 70)
    print("OPTIMIZER SERVICE")
    print("=" * 70)
    print(f"Symbol: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Metric: {args.metric}")
    print("=" * 70)
    print()
    
    # Run optimization
    try:
        results = run_walk_forward_optimization(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=start_date,
            end_date=end_date,
            evaluation_metric=args.metric
        )
        
        if results:
            print("\n✅ Optimization complete!")
            return 0
        else:
            print("\n❌ Optimization failed")
            return 1
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
