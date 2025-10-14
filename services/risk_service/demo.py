#!/usr/bin/env python3
"""
Risk Service Demo Script

Demonstrates comprehensive risk management capabilities including:
- Position sizing calculations
- Risk assessment and monitoring
- Portfolio exposure management
- Real-time risk parameter adjustments
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from services.risk_service.service import (
    RiskService, RiskRequest, RiskParameters, PositionType, EventSystem
)


class RiskServiceDemo:
    """Demo class for Risk Service functionality."""

    def __init__(self):
        self.event_system = EventSystem()
        self.risk_service = RiskService(event_system=self.event_system)
        self.demo_data = self._generate_demo_data()

    def _generate_demo_data(self):
        """Generate synthetic market data for demonstration."""
        np.random.seed(42)  # For reproducible results

        # Generate 1 year of daily data
        dates = pd.date_range('2024-01-01', periods=365, freq='D')

        # Simulate BTC/USDT price data with realistic volatility
        base_price = 50000
        volatility = 0.03  # 3% daily volatility
        price_changes = np.random.normal(0.001, volatility, len(dates))
        prices = base_price * np.exp(np.cumsum(price_changes))

        # Calculate ATR (simplified)
        atr_values = np.full(len(dates), base_price * 0.02)  # 2% ATR

        # Generate returns for risk assessment
        returns = pd.Series(price_changes, index=dates)

        return {
            'dates': dates,
            'prices': prices,
            'atr_values': atr_values,
            'returns': returns,
            'volatility': volatility
        }

    async def demo_position_sizing(self):
        """Demonstrate position sizing calculations."""
        print("\n" + "="*60)
        print("RISK SERVICE DEMO: POSITION SIZING")
        print("="*60)

        # Different market scenarios
        scenarios = [
            {
                'name': 'Conservative BTC Trade',
                'symbol': 'BTC/USDT',
                'capital': 10000,
                'entry_price': 50000,
                'position_type': 'long',
                'atr_value': 1000,
                'volatility': 0.025,
                'risk_percentage': 0.005  # 0.5%
            },
            {
                'name': 'Aggressive ETH Trade',
                'symbol': 'ETH/USDT',
                'capital': 10000,
                'entry_price': 3000,
                'position_type': 'long',
                'atr_value': 60,
                'volatility': 0.04,
                'risk_percentage': 0.01  # 1%
            },
            {
                'name': 'Short Position ADA',
                'symbol': 'ADA/USDT',
                'capital': 5000,
                'entry_price': 0.50,
                'position_type': 'short',
                'atr_value': 0.01,
                'volatility': 0.06,
                'risk_percentage': 0.015  # 1.5%
            }
        ]

        for scenario in scenarios:
            print(f"\n--- {scenario['name']} ---")

            request = RiskRequest(
                symbol=scenario['symbol'],
                capital=scenario['capital'],
                entry_price=scenario['entry_price'],
                position_type=scenario['position_type'],
                atr_value=scenario['atr_value'],
                volatility=scenario['volatility'],
                risk_percentage=scenario['risk_percentage']
            )

            sizing = await self.risk_service.calculate_position_sizing(request)

            print(f"Symbol: {sizing.symbol}")
            print(".2f")
            print(".2f")
            print(".2f")
            print(".2f")
            print(".2f")
            print(".2f")
            print(".2f")
            print(".2f")
            print(".2f")

    async def demo_portfolio_risk_assessment(self):
        """Demonstrate portfolio risk assessment."""
        print("\n" + "="*60)
        print("RISK SERVICE DEMO: PORTFOLIO RISK ASSESSMENT")
        print("="*60)

        # Assess portfolio risk with different return scenarios
        scenarios = [
            {
                'name': 'Conservative Portfolio',
                'mean_return': 0.001,
                'volatility': 0.015,
                'description': 'Low volatility, steady returns'
            },
            {
                'name': 'Balanced Portfolio',
                'mean_return': 0.002,
                'volatility': 0.025,
                'description': 'Moderate risk and return'
            },
            {
                'name': 'Aggressive Portfolio',
                'mean_return': 0.003,
                'volatility': 0.045,
                'description': 'High volatility, high potential returns'
            },
            {
                'name': 'High Risk Portfolio',
                'mean_return': 0.001,
                'volatility': 0.08,
                'description': 'Very high volatility, inconsistent returns'
            }
        ]

        for scenario in scenarios:
            print(f"\n--- {scenario['name']} ---")
            print(f"Description: {scenario['description']}")

            # Generate synthetic returns
            np.random.seed(42)
            returns = pd.Series(
                np.random.normal(scenario['mean_return'], scenario['volatility'], 252),
                index=pd.date_range('2024-01-01', periods=252, freq='D')
            )

            assessment = await self.risk_service.assess_portfolio_risk(returns, 'daily')

            print(f"Risk Level: {assessment.risk_level.value}")
            print(".2f")
            print(f"Risk Score: {assessment.risk_score:.2f}")

            print("\nKey Risk Metrics:")
            metrics = assessment.metrics
            print(".4f")
            print(".4f")
            print(".4f")
            print(".4f")
            print(".4f")
            print(".4f")

            print(f"\nRecommendations ({len(assessment.recommendations)}):")
            for i, rec in enumerate(assessment.recommendations[:3], 1):
                print(f"{i}. {rec}")

    async def demo_portfolio_exposure_management(self):
        """Demonstrate portfolio exposure management."""
        print("\n" + "="*60)
        print("RISK SERVICE DEMO: PORTFOLIO EXPOSURE MANAGEMENT")
        print("="*60)

        # Create a diversified portfolio
        portfolio_positions = [
            {'symbol': 'BTC/USDT', 'allocation': 0.4, 'entry_price': 50000, 'atr': 1000},
            {'symbol': 'ETH/USDT', 'allocation': 0.3, 'entry_price': 3000, 'atr': 60},
            {'symbol': 'ADA/USDT', 'allocation': 0.15, 'entry_price': 0.50, 'atr': 0.01},
            {'symbol': 'SOL/USDT', 'allocation': 0.15, 'entry_price': 150, 'atr': 3}
        ]

        total_capital = 50000

        print("Building Portfolio Positions:")
        for pos in portfolio_positions:
            capital_allocation = total_capital * pos['allocation']
            risk_percentage = 0.01  # 1% risk per position

            request = RiskRequest(
                symbol=pos['symbol'],
                capital=capital_allocation,
                entry_price=pos['entry_price'],
                position_type='long',
                atr_value=pos['atr'],
                volatility=0.03,
                risk_percentage=risk_percentage
            )

            sizing = await self.risk_service.calculate_position_sizing(request)

            print(f"{pos['symbol']}: ${capital_allocation:.0f} capital -> "
                  f"${sizing.position_value:.0f} position (${sizing.risk_amount:.0f} risk)")

        # Get portfolio exposure
        exposure = await self.risk_service.get_portfolio_exposure()

        print("\nPortfolio Exposure Summary:")
        print(f"Total Positions: {exposure['total_positions']}")
        print(".2f")
        print(".2f")
        print(".2f")

        print("\nPosition Details:")
        for symbol, pos_data in exposure['positions'].items():
            print(f"{symbol}: ${pos_data['position_value']:.0f} ({pos_data['risk_amount']:.0f} risk)")

    async def demo_risk_parameter_adjustment(self):
        """Demonstrate dynamic risk parameter adjustment."""
        print("\n" + "="*60)
        print("RISK SERVICE DEMO: RISK PARAMETER ADJUSTMENT")
        print("="*60)

        # Show current parameters
        print("Current Risk Parameters:")
        params = self.risk_service.default_params
        print(f"Max Risk per Trade: {params.max_risk_per_trade:.1%}")
        print(f"Max Portfolio Risk: {params.max_portfolio_risk:.1%}")
        print(f"Max Leverage: {params.max_leverage:.1f}x")
        print(f"Min Leverage: {params.min_leverage:.1f}x")

        # Adjust parameters for different market conditions
        scenarios = [
            {
                'name': 'Bull Market - Higher Risk Tolerance',
                'updates': {
                    'max_risk_per_trade': 0.02,
                    'max_portfolio_risk': 0.08,
                    'max_leverage': 3.0
                }
            },
            {
                'name': 'Bear Market - Conservative Approach',
                'updates': {
                    'max_risk_per_trade': 0.005,
                    'max_portfolio_risk': 0.02,
                    'max_leverage': 1.5
                }
            },
            {
                'name': 'High Volatility - Reduced Exposure',
                'updates': {
                    'max_risk_per_trade': 0.01,
                    'max_portfolio_risk': 0.04,
                    'max_leverage': 2.0
                }
            }
        ]

        for scenario in scenarios:
            print(f"\n--- {scenario['name']} ---")

            await self.risk_service.update_risk_parameters(scenario['updates'])

            # Test position sizing with new parameters
            request = RiskRequest(
                symbol='BTC/USDT',
                capital=10000,
                entry_price=50000,
                position_type='long',
                atr_value=1000,
                volatility=0.03,
                risk_percentage=0.01
            )

            sizing = await self.risk_service.calculate_position_sizing(request)

            print(f"Position Size: ${sizing.position_value:.0f}")
            print(f"Risk Amount: ${sizing.risk_amount:.0f}")
            print(f"Leverage: {sizing.leverage:.1f}x")

    async def demo_real_time_risk_monitoring(self):
        """Demonstrate real-time risk monitoring."""
        print("\n" + "="*60)
        print("RISK SERVICE DEMO: REAL-TIME RISK MONITORING")
        print("="*60)

        # Simulate real-time price updates and risk monitoring
        print("Simulating real-time risk monitoring for BTC/USDT position...")

        # Initial position
        request = RiskRequest(
            symbol='BTC/USDT',
            capital=10000,
            entry_price=50000,
            position_type='long',
            atr_value=1000,
            volatility=0.03,
            risk_percentage=0.01
        )

        initial_sizing = await self.risk_service.calculate_position_sizing(request)

        print(f"Initial Position: ${initial_sizing.position_value:.0f} at ${request.entry_price:.0f}")
        print(f"Stop Loss: ${initial_sizing.stop_loss_price:.0f}")
        print(f"Take Profit: ${initial_sizing.take_profit_price:.0f}")

        # Simulate price movements
        price_changes = [0.02, -0.015, 0.03, -0.025, 0.01]  # 2%, -1.5%, 3%, -2.5%, 1%

        for i, change_pct in enumerate(price_changes, 1):
            current_price = request.entry_price * (1 + change_pct)
            pnl_pct = (current_price - request.entry_price) / request.entry_price

            print(f"\nUpdate {i}: Price ${current_price:.0f} ({change_pct:+.1%})")
            print(".1f")

            # Check if stop loss or take profit triggered
            if current_price <= initial_sizing.stop_loss_price:
                print("⚠️  STOP LOSS TRIGGERED!")
                break
            elif current_price >= initial_sizing.take_profit_price:
                print("✅ TAKE PROFIT TRIGGERED!")
                break
            else:
                print("Position still active")

    async def demo_risk_visualization(self):
        """Create risk visualization charts."""
        print("\n" + "="*60)
        print("RISK SERVICE DEMO: RISK VISUALIZATION")
        print("="*60)

        try:
            # Generate risk-return scatter plot data
            portfolios = []
            for i in range(50):
                # Random portfolio characteristics
                expected_return = np.random.uniform(0.001, 0.005)
                volatility = np.random.uniform(0.01, 0.08)
                sharpe = expected_return / volatility

                portfolios.append({
                    'expected_return': expected_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe,
                    'risk_level': 'Low' if sharpe > 1.5 else 'Medium' if sharpe > 0.5 else 'High'
                })

            df = pd.DataFrame(portfolios)

            # Create visualization
            plt.figure(figsize=(12, 8))

            # Risk-return scatter plot
            plt.subplot(2, 2, 1)
            colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
            for risk_level in df['risk_level'].unique():
                subset = df[df['risk_level'] == risk_level]
                plt.scatter(subset['volatility'], subset['expected_return'],
                           c=colors[risk_level], label=risk_level, alpha=0.7)

            plt.xlabel('Volatility (Risk)')
            plt.ylabel('Expected Return')
            plt.title('Risk-Return Profile')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Sharpe ratio distribution
            plt.subplot(2, 2, 2)
            plt.hist(df['sharpe_ratio'], bins=20, alpha=0.7, edgecolor='black')
            plt.axvline(x=1.0, color='red', linestyle='--', label='Sharpe = 1.0')
            plt.xlabel('Sharpe Ratio')
            plt.ylabel('Frequency')
            plt.title('Sharpe Ratio Distribution')
            plt.legend()

            # Risk level pie chart
            plt.subplot(2, 2, 3)
            risk_counts = df['risk_level'].value_counts()
            plt.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
                   colors=[colors[level] for level in risk_counts.index])
            plt.title('Portfolio Risk Distribution')

            # Volatility vs Sharpe
            plt.subplot(2, 2, 4)
            plt.scatter(df['volatility'], df['sharpe_ratio'], alpha=0.7)
            plt.xlabel('Volatility')
            plt.ylabel('Sharpe Ratio')
            plt.title('Volatility vs Sharpe Ratio')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('/tmp/risk_analysis_demo.png', dpi=150, bbox_inches='tight')
            print("Risk visualization saved to /tmp/risk_analysis_demo.png")

        except ImportError:
            print("Matplotlib/Seaborn not available for visualization demo")

    async def run_full_demo(self):
        """Run complete risk service demonstration."""
        print("TRADPAL RISK SERVICE DEMONSTRATION")
        print("===================================")
        print(f"Demo started at: {datetime.now()}")

        try:
            await self.demo_position_sizing()
            await self.demo_portfolio_risk_assessment()
            await self.demo_portfolio_exposure_management()
            await self.demo_risk_parameter_adjustment()
            await self.demo_real_time_risk_monitoring()
            await self.demo_risk_visualization()

            # Final health check
            health = await self.risk_service.health_check()
            print("\n" + "="*60)
            print("FINAL HEALTH CHECK")
            print("="*60)
            print(f"Service Status: {health['status']}")
            print(f"Portfolio Positions: {health['portfolio_positions']}")
            print(f"Risk History Entries: {len(health.get('risk_history', []))}")

        except Exception as e:
            print(f"Demo error: {e}")
            raise

        print(f"\nDemo completed at: {datetime.now()}")


async def main():
    """Main demo function."""
    demo = RiskServiceDemo()
    await demo.run_full_demo()


if __name__ == "__main__":
    asyncio.run(main())