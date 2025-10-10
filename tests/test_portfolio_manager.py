#!/usr/bin/env python3
"""
Test script for Portfolio Manager functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.portfolio_manager import create_sample_portfolio, PortfolioManager, AllocationMethod

def test_portfolio_management():
    """Test portfolio management functionality."""
    print("📊 Testing Portfolio Management...")

    try:
        # Create portfolio manager
        manager = PortfolioManager()

        # Create sample portfolio
        assets = ['BTC/USDT', 'ETH/USDT', 'EUR/USD', 'GBP/USD']
        portfolio = manager.create_portfolio(
            name="sample_portfolio",
            initial_capital=10000,
            assets=assets,
            allocation_method=AllocationMethod.EQUAL_WEIGHT
        )

        print(f"✅ Created portfolio: {portfolio.name}")
        print(f"   Assets: {list(portfolio.positions.keys())}")
        print(f"   Initial Capital: ${portfolio.initial_capital}")

        # Update prices (simulated)
        manager.update_portfolio_prices("sample_portfolio")

        # Get portfolio summary
        summary = manager.get_portfolio_summary("sample_portfolio")
        print("\n📈 Portfolio Summary:")
        print(f"   Total Value: ${summary['metrics']['total_value']:.2f}")
        print(f"   Cumulative Return: {summary['metrics']['cumulative_return']:.2%}")
        print(f"   Sharpe Ratio: {summary['metrics']['sharpe_ratio']:.2f}")

        print("\n📊 Positions:")
        for pos in summary['positions']:
            print(f"   {pos['symbol']}: {pos['weight']:.1%} (${pos['market_value']:.2f})")

        # Check rebalancing
        needs_rebalance, deviations = manager.check_rebalancing_needed("sample_portfolio")
        print(f"\n🔄 Needs Rebalancing: {needs_rebalance}")

        # Test adding an asset
        print("\n➕ Testing asset addition...")
        success = manager.add_asset_to_portfolio("sample_portfolio", "ADA/USDT", 0.05)
        if success:
            print("✅ Successfully added ADA/USDT to portfolio")

            # Get updated summary
            summary = manager.get_portfolio_summary("sample_portfolio")
            print(f"   New total assets: {len(summary['positions'])}")
            print(f"   New total value: ${summary['metrics']['total_value']:.2f}")

        print("🎉 Portfolio management demo completed!")

    except Exception as e:
        print(f"❌ Portfolio management failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_portfolio_management()