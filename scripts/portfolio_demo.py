#!/usr/bin/env python3
"""
Multi-Asset Portfolio Management Demo Script

Demonstrates portfolio optimization, risk management, and performance tracking
for multi-asset trading portfolios.
"""

import sys
import os
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from config.settings import INITIAL_CAPITAL

# Import portfolio management
from src.multi_asset_portfolio import (
    get_portfolio_manager, create_sample_portfolio,
    ModernPortfolioTheoryOptimizer, RiskParityOptimizer, MinimumVarianceOptimizer
)

# Import data fetching
from src.data_fetcher import fetch_historical_data

# Import audit logging
from src.audit_logger import audit_logger


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Multi-Asset Portfolio Management Demo')

    parser.add_argument('--create-sample', action='store_true',
                       help='Create and demonstrate sample portfolio')
    parser.add_argument('--optimize', type=str, choices=['mpt', 'risk_parity', 'min_variance'],
                       default='risk_parity', help='Portfolio optimization method')
    parser.add_argument('--rebalance', action='store_true',
                       help='Demonstrate portfolio rebalancing')
    parser.add_argument('--simulate-trades', action='store_true',
                       help='Simulate trading activity')
    parser.add_argument('--risk-analysis', action='store_true',
                       help='Perform comprehensive risk analysis')
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate portfolio performance report')
    parser.add_argument('--save-results', action='store_true',
                       help='Save results to file')

    return parser.parse_args()


def create_sample_portfolio_demo():
    """Demonstrate creating a sample portfolio."""
    print("🏗️  Creating Sample Multi-Asset Portfolio...")
    print("=" * 50)

    # Create sample portfolio
    portfolio = create_sample_portfolio()

    print("✅ Sample portfolio created with assets:")
    for symbol, position in portfolio.positions.items():
        print(f"   {symbol}: {position.allocation_pct:.1%} allocation")

    print(f"\n📊 Initial Portfolio Value: ${portfolio.metrics.total_value:,.2f}")
    print(f"🎯 Target Weights: {portfolio.target_weights}")

    return portfolio


def optimize_portfolio_demo(portfolio, method: str = 'risk_parity'):
    """Demonstrate portfolio optimization."""
    print(f"\n🔬 Optimizing Portfolio using {method.upper()}...")
    print("=" * 50)

    # Create sample historical returns for demonstration
    assets = list(portfolio.positions.keys())
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')

    # Generate synthetic returns with realistic correlations
    np.random.seed(42)
    n_assets = len(assets)
    n_days = len(dates)

    print(f"Generating {n_days} days of data for {n_assets} assets...")

    # Simple uncorrelated returns for demo
    returns_data = []
    for i in range(n_assets):
        # Different mean returns for each asset
        mean_return = 0.0005 + (i * 0.0002)  # Slightly different means
        volatility = 0.02 + (i * 0.005)  # Slightly different volatilities
        asset_returns = np.random.normal(mean_return, volatility, n_days)
        returns_data.append(asset_returns)

    # Create returns DataFrame
    returns_df = pd.DataFrame(np.column_stack(returns_data), index=dates, columns=assets)

    # Update portfolio with historical data
    portfolio.update_asset_data(returns_df)

    # Optimize portfolio
    optimized_weights = portfolio.optimize_portfolio(
        optimization_method=method,
        constraints={
            'target_return': 0.02,  # 2% target annual return
            'risk_free_rate': 0.01,  # 1% risk-free rate
            'bounds': [(0, 0.4) for _ in assets]  # Max 40% per asset
        }
    )

    print("✅ Portfolio optimization completed:")
    for asset, weight in optimized_weights.items():
        print(f"   {asset}: {weight:.1%}")

    # Compare with equal weights
    equal_weights = {asset: 1/len(assets) for asset in assets}
    print(f"\n🔄 Previous equal weights: {equal_weights}")

    return portfolio


def simulate_trading_demo(portfolio):
    """Demonstrate trading simulation."""
    print("\n🎯 Simulating Trading Activity...")
    print("=" * 50)

    # Sample price updates
    price_updates = {
        'BTC/USDT': 45000.0,
        'ETH/USDT': 2800.0,
        'ADA/USDT': 0.45,
        'DOT/USDT': 8.50
    }

    print("📈 Updating prices...")
    portfolio.update_prices(price_updates)

    # Simulate some trades
    trades = [
        ('BTC/USDT', 'BUY', 45000.0, None),  # Use calculated position size
        ('ETH/USDT', 'BUY', 2800.0, None),
        ('ADA/USDT', 'BUY', 0.45, 10000),  # Fixed quantity
    ]

    executed_trades = []
    for symbol, signal, price, quantity in trades:
        print(f"\n🔄 Executing {signal} trade for {symbol} at ${price}")
        result = portfolio.execute_trade(symbol, signal, price, quantity)
        if result['success']:
            print("   ✅ Trade executed:")
            print(f"      Quantity: {result['quantity']:.4f}")
            print(f"      Trade Value: ${result['trade_value']:.2f}")
            print(f"      Realized P&L: ${result['realized_pnl']:.2f}")
            executed_trades.append(result)
        else:
            print(f"   ❌ Trade failed: {result.get('error', 'Unknown error')}")

    # Update prices again to see P&L
    updated_prices = {
        'BTC/USDT': 46500.0,  # +3.3%
        'ETH/USDT': 2750.0,   # -1.8%
        'ADA/USDT': 0.47,     # +4.4%
        'DOT/USDT': 8.50      # No change
    }

    print("\n📊 Updating prices to see P&L...")
    portfolio.update_prices(updated_prices)

    # Show updated portfolio
    metrics = portfolio.get_portfolio_metrics()
    print("\n💰 Updated Portfolio Metrics:")
    print(f"   Total Value: ${metrics.total_value:.2f}")
    print(f"   Total P&L: ${metrics.total_pnl:.2f}")
    print(f"   Daily P&L: ${metrics.daily_pnl:.2f}")

    return portfolio, executed_trades


def risk_analysis_demo(portfolio):
    """Demonstrate comprehensive risk analysis."""
    print("\n⚠️  Performing Risk Analysis...")
    print("=" * 50)

    risk_metrics = portfolio.get_risk_metrics()

    if risk_metrics:
        print("📊 Risk Metrics:")
        print(f"   Portfolio Volatility: {risk_metrics.get('volatility', 0):.2%}")
        print(f"   Sharpe Ratio: {risk_metrics.get('sharpe_ratio', 0):.2f}")
        print(f"   Maximum Drawdown: {risk_metrics.get('max_drawdown', 0):.2%}")
        print(f"   Value at Risk (95%): {risk_metrics.get('var_95', 0):.2%}")
        print(f"   Expected Shortfall: {risk_metrics.get('expected_shortfall', 0):.2%}")

        if 'diversification_ratio' in risk_metrics:
            print(f"   Diversification Ratio: {risk_metrics['diversification_ratio']:.2f}")
        if 'risk_parity_score' in risk_metrics:
            print(f"   Risk Parity Score: {risk_metrics['risk_parity_score']:.2f}")
    else:
        print("⚠️  No risk metrics available (need historical data)")

    return risk_metrics


def rebalancing_demo(portfolio):
    """Demonstrate portfolio rebalancing."""
    print("\n🔄 Demonstrating Portfolio Rebalancing...")
    print("=" * 50)

    print("📊 Current allocations:")
    total_value = portfolio.metrics.total_value
    for symbol, position in portfolio.positions.items():
        current_value = position.quantity * position.current_price
        current_weight = current_value / total_value if total_value > 0 else 0
        target_weight = portfolio.target_weights.get(symbol, 0)
        deviation = current_weight - target_weight
        print(f"   {symbol}: {current_weight:.1%} (target: {target_weight:.1%}, deviation: {deviation:+.1%})")

    # Perform rebalancing
    rebalance_result = portfolio.rebalance_portfolio()

    if rebalance_result['success']:
        print(f"\n✅ Rebalancing completed with {rebalance_result['trades_executed']} trades")
        print(f"   Total transaction cost: ${rebalance_result['total_cost']:.2f}")
    else:
        print(f"\n❌ Rebalancing failed: {rebalance_result.get('error', 'Unknown error')}")

    return rebalance_result


def generate_report_demo(portfolio):
    """Generate comprehensive portfolio report."""
    print("\n📋 Generating Portfolio Report...")
    print("=" * 50)

    report = portfolio.generate_report()

    print("📊 Portfolio Summary:")
    print(f"   Assets: {report['asset_count']}")
    print(f"   Total Value: ${report['portfolio_metrics']['total_value']:.2f}")
    print(f"   Total P&L: ${report['portfolio_metrics']['total_pnl']:.2f}")
    print(f"   Daily P&L: ${report['portfolio_metrics']['daily_pnl']:.2f}")

    print("\n📈 Risk Metrics:")
    risk = report['risk_metrics']
    if risk:
        print(f"   Volatility: {risk.get('volatility', 0):.2%}")
        print(f"   Sharpe Ratio: {risk.get('sharpe_ratio', 0):.2f}")
        print(f"   Max Drawdown: {risk.get('max_drawdown', 0):.2%}")
    else:
        print("   No risk metrics available")

    print("\n💼 Positions:")
    for pos in report['positions']:
        pnl_color = "🟢" if pos['unrealized_pnl'] >= 0 else "🔴"
        print(f"   {pos['symbol']}: {pos['quantity']:.4f} @ ${pos['entry_price']:.2f} "
              f"(Current: ${pos['current_price']:.2f}) {pnl_color}${pos['unrealized_pnl']:.2f}")

    return report


def save_results(results: dict, filename: str = None):
    """Save results to file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"portfolio_demo_results_{timestamp}.json"

    output_dir = Path("output/portfolio")
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / filename

    try:
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n💾 Results saved to: output/portfolio/{filename}")

    except Exception as e:
        print(f"\n❌ Failed to save results: {e}")


def main():
    """Main demonstration function."""
    args = parse_arguments()

    print("🤖 Multi-Asset Portfolio Management Demo")
    print("=" * 60)

    results = {
        'timestamp': datetime.now().isoformat(),
        'demonstrations': {}
    }

    # Create sample portfolio if needed
    if args.create_sample or not any([args.optimize, args.rebalance, args.simulate_trades,
                                     args.risk_analysis, args.generate_report]):
        portfolio = create_sample_portfolio_demo()
        results['demonstrations']['sample_portfolio'] = {
            'asset_count': len(portfolio.positions),
            'total_value': portfolio.metrics.total_value,
            'target_weights': portfolio.target_weights
        }
    else:
        # Create a basic portfolio for other operations
        portfolio = get_portfolio_manager()
        if not portfolio.positions:
            portfolio = create_sample_portfolio_demo()
            results['demonstrations']['sample_portfolio'] = {
                'asset_count': len(portfolio.positions),
                'total_value': portfolio.metrics.total_value,
                'target_weights': portfolio.target_weights
            }

    # Optimize portfolio
    if args.optimize:
        portfolio = optimize_portfolio_demo(portfolio, args.optimize)
        results['demonstrations']['optimization'] = {
            'method': args.optimize,
            'optimized_weights': portfolio.target_weights
        }

    # Simulate trading
    if args.simulate_trades:
        portfolio, trades = simulate_trading_demo(portfolio)
        results['demonstrations']['trading'] = {
            'trades_executed': len(trades),
            'final_portfolio_value': portfolio.metrics.total_value,
            'total_pnl': portfolio.metrics.total_pnl
        }

    # Perform risk analysis
    if args.risk_analysis:
        risk_metrics = risk_analysis_demo(portfolio)
        results['demonstrations']['risk_analysis'] = risk_metrics

    # Demonstrate rebalancing
    if args.rebalance:
        rebalance_result = rebalancing_demo(portfolio)
        results['demonstrations']['rebalancing'] = rebalance_result

    # Generate report
    if args.generate_report:
        report = generate_report_demo(portfolio)
        results['demonstrations']['report'] = report

    # Save results
    if args.save_results:
        save_results(results)

    print("\n🎉 Portfolio Management Demo Completed!")
    print("=" * 60)
    print("💡 Key Takeaways:")
    print("   • Multi-asset portfolios improve diversification")
    print("   • Risk parity ensures balanced risk contribution")
    print("   • Regular rebalancing maintains target allocations")
    print("   • Advanced optimization can improve risk-adjusted returns")
    print("   • Comprehensive risk monitoring is essential for success")


if __name__ == "__main__":
    main()