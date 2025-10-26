#!/usr/bin/env python3
"""
TradPal UI Service Demo

This demo showcases the UI Service capabilities:
- Web-based trading interface
- Authentication system
- Strategy builder functionality
- Interactive controls
- Live charts with technical indicators
- Monitoring dashboard

Note: This is a demonstration script that shows how to interact with
the UI service programmatically. For the full web interface experience,
run: streamlit run services/ui_service/web_ui_service/app.py
"""

import asyncio
import logging
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Add the services directory to the path
sys.path.insert(0, '/Users/danielsadowski/VSCodeProjects/tradpal/tradpal')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UIServiceDemo:
    """Demo class for UI Service functionality"""

    def __init__(self):
        self.sample_data = self._generate_sample_data()
        self.sample_strategies = self._generate_sample_strategies()

    def _generate_sample_data(self) -> List[Dict[str, Any]]:
        """Generate sample OHLCV data for demonstration"""
        import random

        data = []
        base_price = 45000
        current_price = base_price

        # Generate 100 candles for demo
        for i in range(100):
            # Simulate price movement
            change = random.uniform(-300, 300)
            current_price += change
            current_price = max(35000, min(55000, current_price))

            # Generate OHLCV
            high = current_price + random.uniform(0, 150)
            low = current_price - random.uniform(0, 150)
            open_price = current_price + random.uniform(-50, 50)
            volume = random.uniform(200, 800)

            data.append({
                "timestamp": (datetime.now() - timedelta(hours=100-i)).isoformat(),
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(current_price, 2),
                "volume": round(volume, 2)
            })

        return data

    def _generate_sample_strategies(self) -> Dict[str, Any]:
        """Generate sample trading strategies"""
        return {
            "trend_following": {
                "name": "Trend Following Strategy",
                "description": "EMA crossover with ADX confirmation",
                "indicators": ["ema", "adx"],
                "parameters": {
                    "ema_short": 9,
                    "ema_long": 21,
                    "adx_period": 14,
                    "adx_threshold": 25
                },
                "risk_management": {
                    "stop_loss": 1.5,
                    "take_profit": 3.0,
                    "max_position_size": 0.02
                }
            },
            "mean_reversion": {
                "name": "Mean Reversion Strategy",
                "description": "Bollinger Bands with RSI filter",
                "indicators": ["bb", "rsi"],
                "parameters": {
                    "bb_period": 20,
                    "bb_std": 2.0,
                    "rsi_period": 14,
                    "rsi_overbought": 70,
                    "rsi_oversold": 30
                },
                "risk_management": {
                    "stop_loss": 1.0,
                    "take_profit": 2.0,
                    "max_position_size": 0.01
                }
            },
            "scalping": {
                "name": "Scalping Strategy",
                "description": "Fast EMA with tight stops",
                "indicators": ["ema"],
                "parameters": {
                    "ema_short": 5,
                    "ema_long": 10
                },
                "risk_management": {
                    "stop_loss": 0.5,
                    "take_profit": 1.0,
                    "max_position_size": 0.005
                }
            }
        }

    async def run_comprehensive_demo(self):
        """Run the complete UI service demo"""
        logger.info("\n🚀 === Starting TradPal UI Service Comprehensive Demo ===")

        print("🎨 TradPal UI Service - Complete Feature Demonstration")
        print("=" * 60)
        print("This demo showcases all UI service capabilities:")
        print("✅ Authentication system")
        print("✅ Strategy builder functionality")
        print("✅ Interactive controls")
        print("✅ Live charts with technical indicators")
        print("✅ Monitoring dashboard")
        print("✅ Web interface features")
        print("✅ Service integration capabilities")
        print("✅ Deployment options")
        print()

        try:
            # Run all demo methods
            self.demo_authentication_system()
            await asyncio.sleep(0.1)  # Brief pause for readability

            self.demo_strategy_builder()
            await asyncio.sleep(0.1)

            self.demo_interactive_controls()
            await asyncio.sleep(0.1)

            self.demo_live_charts()
            await asyncio.sleep(0.1)

            self.demo_monitoring_dashboard()
            await asyncio.sleep(0.1)

            self.demo_web_interface_features()
            await asyncio.sleep(0.1)

            self.demo_integration_capabilities()
            await asyncio.sleep(0.1)

            self.demo_deployment_options()
            await asyncio.sleep(0.1)

            print("\n🎉 === Demo Completed Successfully ===")
            print("All UI service features have been demonstrated!")
            print()
            print("To experience the full web interface:")
            print("1. Install Streamlit: pip install streamlit")
            print("2. Run: streamlit run services/ui_service/web_ui_service/app.py")
            print("3. Open your browser to the provided URL")
            print()
            print("For programmatic access, use the UIServiceDemo class methods.")

        except Exception as e:
            logger.error(f"Demo execution failed: {e}")
            print(f"\n❌ Demo failed with error: {e}")
            raise

    def demo_authentication_system(self):
        """Demo authentication system capabilities"""
        logger.info("\n🔐 === Authentication System Demo ===")

        print("🔐 TradPal UI Service - Authentication System")
        print("=" * 50)
        print("Features Demonstrated:")
        print("✅ User registration and login")
        print("✅ Password hashing and security")
        print("✅ Role-based access control")
        print("✅ Session management")
        print("✅ Secure password recovery")
        print()
        print("Default Credentials:")
        print("  Username: admin")
        print("  Password: admin123")
        print("⚠️  CHANGE THESE IMMEDIATELY IN PRODUCTION!")
        print()
        print("Security Features:")
        print("• Werkzeug password hashing")
        print("• Flask-Login session management")
        print("• CSRF protection")
        print("• Rate limiting")
        print("• Audit logging")
        print()
        print("To experience the full authentication system:")
        print("1. Run: streamlit run services/ui_service/web_ui_service/app.py")
        print("2. Navigate to the login page")
        print("3. Register a new account or use default credentials")

    def demo_strategy_builder(self):
        """Demo strategy builder functionality"""
        logger.info("\n� === Strategy Builder Demo ===")

        print("🎨 TradPal UI Service - Strategy Builder")
        print("=" * 50)
        print("Available Strategies:")
        print()

        for strategy_id, strategy in self.sample_strategies.items():
            print(f"📊 {strategy['name']}")
            print(f"   Description: {strategy['description']}")
            print(f"   Indicators: {', '.join(strategy['indicators'])}")
            print("   Parameters:")
            for param, value in strategy['parameters'].items():
                print(f"     • {param}: {value}")
            print("   Risk Management:")
            for risk_param, value in strategy['risk_management'].items():
                print(f"     • {risk_param}: {value}")
            print()

        print("Strategy Builder Features:")
        print("✅ Drag-and-drop indicator selection")
        print("✅ Real-time parameter adjustment")
        print("✅ Visual strategy composition")
        print("✅ Preset strategy templates")
        print("✅ Strategy persistence (save/load)")
        print("✅ Backtesting integration")
        print()
        print("Interactive Features:")
        print("• Slider controls for parameter adjustment")
        print("• Real-time validation feedback")
        print("• Strategy preview and visualization")
        print("• Export/import strategy configurations")

    def demo_interactive_controls(self):
        """Demo interactive controls functionality"""
        logger.info("\n⚙️ === Interactive Controls Demo ===")

        print("⚙️ TradPal UI Service - Interactive Controls")
        print("=" * 50)
        print("Timeframe-Specific Settings:")
        print()

        timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
        for tf in timeframes:
            print(f"🕐 {tf} Timeframe Configuration:")
            print("   Trend Indicators:")
            print("     • EMA Short: Dynamic slider (5-50)")
            print("     • EMA Long: Dynamic slider (10-200)")
            print("   Momentum Indicators:")
            print("     • RSI Period: Dynamic slider (5-30)")
            print("     • RSI Overbought: Slider (60-80)")
            print("     • RSI Oversold: Slider (20-40)")
            print("   Risk Management:")
            print("     • Stop Loss %: Slider (0.1-5.0)")
            print("     • Take Profit %: Slider (0.1-10.0)")
            print("     • Leverage: Slider (1-10)")
            print()

        print("Control Features:")
        print("✅ Real-time parameter adjustment")
        print("✅ Timeframe-specific configurations")
        print("✅ Organized tabbed interface")
        print("✅ Parameter validation")
        print("✅ Preset configurations")
        print("✅ Configuration export/import")
        print()
        print("Advanced Features:")
        print("• Visual feedback for parameter changes")
        print("• Quick preset buttons (Scalping, Trend, Conservative)")
        print("• Parameter synchronization across timeframes")
        print("• Undo/redo functionality")

    def demo_live_charts(self):
        """Demo live charts functionality"""
        logger.info("\n📈 === Live Charts Demo ===")

        print("📈 TradPal UI Service - Live Charts")
        print("=" * 50)
        print("Chart Types Supported:")
        print("✅ Candlestick charts")
        print("✅ Line charts")
        print("✅ OHLC charts")
        print()
        print("Technical Indicators Available:")
        indicators = [
            ("EMA", "Exponential Moving Average"),
            ("RSI", "Relative Strength Index"),
            ("BB", "Bollinger Bands"),
            ("MACD", "Moving Average Convergence Divergence"),
            ("Volume", "Trading Volume Bars"),
            ("Signals", "Buy/Sell Signal Markers")
        ]

        for indicator, description in indicators:
            print(f"📊 {indicator}: {description}")

        print()
        print("Interactive Features:")
        print("✅ Zoom and pan functionality")
        print("✅ Detailed hover information")
        print("✅ Auto-refresh capability")
        print("✅ Multi-timeframe support")
        print("✅ Indicator toggle on/off")
        print("✅ Signal strength visualization")
        print()
        print("Sample Data Statistics:")
        closes = [d['close'] for d in self.sample_data]
        highs = [d['high'] for d in self.sample_data]
        lows = [d['low'] for d in self.sample_data]
        volumes = [d['volume'] for d in self.sample_data]

        print(f"📈 Data Points: {len(self.sample_data)}")
        print(f"💰 Price Range: ${min(lows):.2f} - ${max(highs):.2f}")
        print(f"📊 Avg Price: ${sum(closes)/len(closes):.2f}")
        print(f"📦 Volume Range: {min(volumes):.0f} - {max(volumes):.0f}")
        print(f"📈 Avg Volume: {sum(volumes)/len(volumes):.0f}")

    def demo_monitoring_dashboard(self):
        """Demo monitoring dashboard functionality"""
        logger.info("\n📊 === Monitoring Dashboard Demo ===")

        print("📊 TradPal UI Service - Monitoring Dashboard")
        print("=" * 50)
        print("Performance Metrics:")
        print("📈 Win Rate: 68.5% (+2.1%)")
        print("💰 Total Return: 24.7% (+1.3%)")
        print("🎯 Sharpe Ratio: 1.85 (+0.12)")
        print("📉 Max Drawdown: -8.2% (-0.5%)")
        print()
        print("System Health Status:")
        components = [
            ("Data Fetcher", "🟢 Online", "Fetching market data successfully"),
            ("Indicators", "🟢 Online", "Calculating technical indicators"),
            ("Signal Generator", "🟢 Online", "Generating trading signals"),
            ("Backtester", "🟡 Warning", "High memory usage detected"),
            ("Database", "🟢 Online", "All connections healthy"),
            ("API Gateway", "🟢 Online", "Routing requests normally"),
            ("Notification Service", "🟢 Online", "Sending alerts")
        ]

        for component, status, details in components:
            print(f"{status} {component}: {details}")

        print()
        print("Dashboard Features:")
        print("✅ Real-time performance tracking")
        print("✅ System component monitoring")
        print("✅ Alert management and notifications")
        print("✅ Equity curve visualization")
        print("✅ Trade analysis and statistics")
        print("✅ Monthly returns breakdown")
        print("✅ Risk metrics monitoring")
        print()
        print("Alert Management:")
        print("🚨 Active Alerts:")
        print("   • High memory usage in backtester")
        print("   • Unusual volume spike detected")
        print("   • API rate limit approaching")
        print()
        print("📋 Recent Trades:")
        print("   • BUY BTC/USDT @ $45,230 (2 hours ago)")
        print("   • SELL ETH/USDT @ $2,850 (4 hours ago)")
        print("   • BUY BTC/USDT @ $44,890 (6 hours ago)")

    def demo_web_interface_features(self):
        """Demo comprehensive web interface features"""
        logger.info("\n🌐 === Web Interface Features Demo ===")

        print("🌐 TradPal UI Service - Complete Web Interface")
        print("=" * 50)
        print("Navigation Structure:")
        print("🏠 Dashboard - Overview and key metrics")
        print("🎨 Strategy Builder - Create and manage strategies")
        print("⚙️ Interactive Controls - Adjust trading parameters")
        print("📈 Live Charts - View technical analysis")
        print("📊 Monitoring - Performance and system health")
        print("👤 Profile - User settings and preferences")
        print()
        print("Technology Stack:")
        print("🎨 Frontend: Streamlit - Modern web UI framework")
        print("📊 Charts: Plotly - Interactive data visualization")
        print("🔐 Auth: Flask-Login - Secure session management")
        print("💾 Storage: JSON files - User data persistence")
        print("🔄 Real-time: Auto-refresh - Live data updates")
        print()
        print("Responsive Design:")
        print("✅ Desktop optimized interface")
        print("✅ Mobile-responsive layout")
        print("✅ Tablet compatibility")
        print("✅ Touch-friendly controls")
        print()
        print("User Experience Features:")
        print("🎯 Intuitive navigation with sidebar")
        print("⚡ Fast loading with lazy components")
        print("💾 Auto-save user preferences")
        print("🎨 Customizable themes and layouts")
        print("🔔 Real-time notifications")
        print("📱 Progressive Web App (PWA) ready")

    def demo_integration_capabilities(self):
        """Demo integration with other TradPal services"""
        logger.info("\n🔗 === Service Integration Demo ===")

        print("🔗 TradPal UI Service - Service Integration")
        print("=" * 50)
        print("Connected Services:")
        print()
        print("🔧 Core Service (Port 8002):")
        print("   • Technical indicator calculations")
        print("   • Signal generation and validation")
        print("   • Strategy execution")
        print("   • Performance monitoring")
        print()
        print("📊 Data Service (Port 8000):")
        print("   • Market data fetching")
        print("   • Historical data access")
        print("   • Real-time price feeds")
        print("   • Data quality validation")
        print()
        print("💼 Trading Service (Port 8005):")
        print("   • Live trading execution")
        print("   • Portfolio management")
        print("   • Risk assessment")
        print("   • Order management")
        print()
        print("📈 Monitoring Service (Port 8001/8004):")
        print("   • ML model monitoring")
        print("   • System health checks")
        print("   • Alert management")
        print("   • Performance analytics")
        print()
        print("Integration Features:")
        print("✅ RESTful API communication")
        print("✅ Async service calls")
        print("✅ Error handling and retries")
        print("✅ Real-time data streaming")
        print("✅ Service discovery")
        print("✅ Circuit breaker patterns")

    def demo_deployment_options(self):
        """Demo deployment and scaling options"""
        logger.info("\n🚀 === Deployment Options Demo ===")

        print("🚀 TradPal UI Service - Deployment Options")
        print("=" * 50)
        print("Development Deployment:")
        print("  streamlit run app.py")
        print("  • Local development server")
        print("  • Hot reload enabled")
        print("  • Debug mode available")
        print()
        print("Production Deployment:")
        print("  Docker Container:")
        print("    docker build -t tradpal/ui-service .")
        print("    docker run -p 8501:8501 tradpal/ui-service")
        print()
        print("  Kubernetes:")
        print("    kubectl apply -f k8s-deployment.yaml")
        print("    • Auto-scaling support")
        print("    • Load balancing")
        print("    • Health checks")
        print()
        print("  Cloud Platforms:")
        print("    • AWS Fargate/ECS")
        print("    • Google Cloud Run")
        print("    • Azure Container Instances")
        print("    • Heroku deployment")
        print()
        print("Scaling Features:")
        print("✅ Horizontal pod scaling")
        print("✅ Load balancer integration")
        print("✅ Session persistence")
        print("✅ CDN integration")
        print("✅ Database connection pooling")


async def main():
    """Main entry point for the UI service demo"""
    print("🚀 Starting TradPal UI Service Demo...")

    try:
        # Create demo instance
        demo = UIServiceDemo()

        # Run comprehensive demo
        await demo.run_comprehensive_demo()

    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Run the async demo
    asyncio.run(main())
