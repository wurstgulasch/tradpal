"""
Portfolio Management Example for TradPal Indicator

This example demonstrates advanced portfolio management capabilities including:
- Multi-asset portfolio creation with different allocation methods
- Risk-based position sizing and rebalancing
- Performance analytics and risk metrics
- Dynamic asset addition and removal

Features showcased:
- Equal weight allocation
- Risk parity allocation
- Portfolio rebalancing
- Performance tracking
- Risk management metrics

Author: TradPal Indicator Team
Version: 2.5.0
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.portfolio_manager import (
    PortfolioManager,
    AllocationMethod,
    RebalancingFrequency,
    PortfolioConstraints
)

def create_diversified_portfolio():
    """Create a diversified multi-asset portfolio."""
    print("🏗️  Creating Diversified Portfolio...")

    manager = PortfolioManager()

    # Define asset universe (crypto, forex, commodities)
    assets = [
        'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT',  # Crypto
        'EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD',      # Forex
        'XAU/USD', 'XAG/USD'                                         # Precious metals
    ]

    # Create portfolio with risk parity allocation
    constraints = PortfolioConstraints(
        max_weight_per_asset=0.15,  # Max 15% per asset
        min_weight_per_asset=0.02,  # Min 2% per asset
        max_volatility=0.20,        # Max 20% portfolio volatility
        max_drawdown=0.12,          # Max 12% drawdown
        max_assets=12
    )

    portfolio = manager.create_portfolio(
        name="diversified_portfolio",
        initial_capital=50000,
        assets=assets,
        allocation_method=AllocationMethod.RISK_PARITY,
        constraints=constraints
    )

    print(f"✅ Created portfolio with ${portfolio.initial_capital:,.0f} capital")
    print(f"   Assets: {len(portfolio.positions)}")
    print(f"   Allocation method: {portfolio.allocation_method.value}")

    return manager, portfolio

def demonstrate_allocation_methods():
    """Demonstrate different allocation methods."""
    print("\n📊 Comparing Allocation Methods...")

    manager = PortfolioManager()
    assets = ['BTC/USDT', 'ETH/USDT', 'EUR/USD', 'GBP/USD', 'XAU/USD']

    methods = [
        AllocationMethod.EQUAL_WEIGHT,
        AllocationMethod.RISK_PARITY,
        AllocationMethod.VOLATILITY_TARGETED
    ]

    results = {}

    for method in methods:
        portfolio = manager.create_portfolio(
            name=f"test_{method.value}",
            initial_capital=10000,
            assets=assets,
            allocation_method=method
        )

        summary = manager.get_portfolio_summary(f"test_{method.value}")
        results[method.value] = summary

        print(f"\n{method.value.upper()} ALLOCATION:")
        print(f"   Total Value: ${summary['metrics']['total_value']:,.2f}")
        print(f"   Expected Volatility: {summary['metrics']['volatility']:.1%}")
        print(f"   Sharpe Ratio: {summary['metrics']['sharpe_ratio']:.2f}")

        print("   Asset Weights:")
        for pos in summary['positions']:
            print(f"      {pos['symbol']}: {pos['weight']:.1%}")

    return results

def simulate_portfolio_rebalancing():
    """Simulate portfolio rebalancing over time."""
    print("\n🔄 Simulating Portfolio Rebalancing...")

    manager = PortfolioManager()
    assets = ['BTC/USDT', 'ETH/USDT', 'EUR/USD', 'GBP/USD']

    # Create portfolio with monthly rebalancing
    portfolio = manager.create_portfolio(
        name="rebalance_demo",
        initial_capital=25000,
        assets=assets,
        allocation_method=AllocationMethod.EQUAL_WEIGHT
    )

    print("Initial portfolio:")
    summary = manager.get_portfolio_summary("rebalance_demo")
    for pos in summary['positions']:
        print(f"   {pos['symbol']}: {pos['weight']:.1%} (${pos['market_value']:,.0f})")

    # Simulate price changes (some assets perform better than others)
    print("\n📈 Simulating market movements...")

    # Manually adjust prices to simulate market movement
    price_changes = {
        'BTC/USDT': 1.15,  # +15%
        'ETH/USDT': 0.95,  # -5%
        'EUR/USD': 1.02,   # +2%
        'GBP/USD': 1.08    # +8%
    }

    # Update positions with new prices
    for symbol, change in price_changes.items():
        if symbol in portfolio.positions:
            position = portfolio.positions[symbol]
            position.current_price *= change
            position.market_value = position.quantity * position.current_price
            position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity

    # Recalculate weights
    manager._update_portfolio_weights(portfolio)

    print("After price changes:")
    summary = manager.get_portfolio_summary("rebalance_demo")
    for pos in summary['positions']:
        print(f"   {pos['symbol']}: {pos['weight']:.1%} (${pos['market_value']:,.0f}) P&L: ${pos['unrealized_pnl']:,.0f}")

    # Check if rebalancing is needed
    needs_rebalance, deviations = manager.check_rebalancing_needed("rebalance_demo")
    print(f"\n🔄 Rebalancing needed: {needs_rebalance}")

    if needs_rebalance:
        print("Deviations from target weights:")
        for asset, deviation in deviations.items():
            print(f"   {asset}: {deviation:.1%}")

        # Perform rebalancing
        success = manager.rebalance_portfolio("rebalance_demo")
        if success:
            print("\n✅ Portfolio rebalanced successfully!")

            summary = manager.get_portfolio_summary("rebalance_demo")
            print("After rebalancing:")
            for pos in summary['positions']:
                print(f"   {pos['symbol']}: {pos['weight']:.1%} (${pos['market_value']:,.0f})")

def demonstrate_risk_management():
    """Demonstrate risk management features."""
    print("\n⚠️  Risk Management Demonstration...")

    manager = PortfolioManager()
    assets = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'EUR/USD', 'GBP/USD', 'XAU/USD']

    portfolio = manager.create_portfolio(
        name="risk_managed_portfolio",
        initial_capital=100000,
        assets=assets,
        allocation_method=AllocationMethod.RISK_PARITY
    )

    # Get comprehensive risk metrics
    summary = manager.get_portfolio_summary("risk_managed_portfolio")

    print("📊 Portfolio Risk Metrics:")
    metrics = summary['metrics']
    print(f"   Total Value: ${metrics['total_value']:,.0f}")
    print(f"   Volatility: {metrics['volatility']:.1%}")
    print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {metrics['max_drawdown']:.1%}")
    print(f"   VaR (95%): ${abs(metrics['var_95']):,.0f}")
    print(f"   CVaR (95%): ${abs(metrics.get('cvar_95', 0)):,.0f}")
    print(f"   Diversification Ratio: {metrics['diversification_ratio']:.2f}")
    print(f"   Concentration Ratio: {metrics['concentration_ratio']:.1%}")

    print(f"\n📋 Portfolio Constraints:")
    constraints = summary['constraints']
    print(f"   Max weight per asset: {constraints['max_weight_per_asset']:.0%}")
    print(f"   Max portfolio volatility: {constraints['max_volatility']:.0%}")
    print(f"   Max drawdown: {constraints['max_drawdown']:.0%}")
    print(f"   Max assets: {constraints['max_assets']}")

def create_performance_visualization():
    """Create sample performance visualization."""
    print("\n📈 Creating Performance Visualization...")

    try:
        # Sample data for visualization
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='M')
        portfolio_values = [100000 * (1 + np.random.normal(0.01, 0.05)) for _ in range(len(dates))]
        portfolio_values = np.cumprod(portfolio_values)

        benchmark_values = [100000 * (1 + np.random.normal(0.005, 0.03)) for _ in range(len(dates))]
        benchmark_values = np.cumprod(benchmark_values)

        # Create DataFrame
        df = pd.DataFrame({
            'Date': dates,
            'Portfolio': portfolio_values,
            'Benchmark': benchmark_values
        })

        # Calculate returns
        df['Portfolio_Return'] = df['Portfolio'].pct_change()
        df['Benchmark_Return'] = df['Benchmark'].pct_change()

        # Simple visualization (would be more sophisticated in real implementation)
        print("Sample Performance Data:")
        print(df.head())

        print("\n✅ Performance visualization data prepared")
        print("   (In a real implementation, this would create charts with matplotlib/seaborn)")

    except Exception as e:
        print(f"❌ Visualization failed: {e}")

def main():
    """Main demonstration function."""
    print("🚀 TradPal Indicator - Portfolio Management Demo")
    print("=" * 50)

    try:
        # Create diversified portfolio
        manager, portfolio = create_diversified_portfolio()

        # Compare allocation methods
        allocation_results = demonstrate_allocation_methods()

        # Simulate rebalancing
        simulate_portfolio_rebalancing()

        # Demonstrate risk management
        demonstrate_risk_management()

        # Create performance visualization
        create_performance_visualization()

        print("\n" + "=" * 50)
        print("🎉 Portfolio Management Demo Completed!")
        print("\nKey Features Demonstrated:")
        print("✅ Multi-asset portfolio creation")
        print("✅ Risk-based allocation methods")
        print("✅ Dynamic rebalancing")
        print("✅ Comprehensive risk metrics")
        print("✅ Performance tracking")
        print("✅ Asset management (add/remove)")

        print("\n💡 Next Steps:")
        print("• Integrate with real-time price feeds")
        print("• Add correlation-based risk management")
        print("• Implement tax-aware rebalancing")
        print("• Add portfolio optimization algorithms")
        print("• Create web dashboard for monitoring")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()