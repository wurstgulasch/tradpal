#!/usr/bin/env python3
"""
Trading Service Demo Script

Demonstrates the core functionality of the TradPal Trading Service:
- Trading session orchestration
- AI-powered trading signals
- Risk management integration
- Market regime awareness
- Performance monitoring

Usage:
    python demo_trading_service.py

Requirements:
    - Trading Service running on localhost:8002
    - Data Service running on localhost:8001
    - Redis running for event communication
"""

import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
import pandas as pd

# Import the trading service orchestrator
try:
    from services.trading_service.orchestrator import TradingServiceOrchestrator
    TRADING_SERVICE_AVAILABLE = True
except ImportError:
    print("❌ TradingServiceOrchestrator not available. Make sure the trading service is properly installed.")
    TRADING_SERVICE_AVAILABLE = False

# Fallback for demo purposes
class MockTradingServiceOrchestrator:
    """Mock orchestrator for demonstration when service is not available"""

    def __init__(self):
        self.is_initialized = False
        self.active_sessions = {}

    async def initialize(self):
        """Mock initialization"""
        print("🔧 Mock: Initializing Trading Service Orchestrator...")
        await asyncio.sleep(0.5)
        self.is_initialized = True
        print("✅ Mock: Trading Service Orchestrator initialized")

    async def start_trading_session(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Mock trading session start"""
        print(f"🚀 Mock: Starting trading session for {config.get('symbol', 'BTC/USDT')}")
        await asyncio.sleep(0.3)

        session_id = f"mock_session_{int(time.time())}"
        session = {
            "session_id": session_id,
            "symbol": config.get('symbol', 'BTC/USDT'),
            "config": config,
            "start_time": datetime.now().isoformat(),
            "status": "active",
            "services_initialized": ["ml_training", "reinforcement_learning", "market_regime", "risk_management", "trading_execution"]
        }

        self.active_sessions[session_id] = session
        return session

    async def execute_smart_trade(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock smart trade execution"""
        print(f"🤖 Mock: Executing smart trade for {symbol}")
        await asyncio.sleep(0.4)

        # Simulate AI decision making
        signals = {
            "ml_signal": {"action": "BUY", "confidence": 0.75},
            "rl_signal": {"action": "BUY", "confidence": 0.68},
            "regime": "bull_market"
        }

        risk_check = {
            "within_limits": True,
            "position_size": 0.02,
            "stop_loss": 0.02
        }

        trade_result = {
            "symbol": symbol,
            "action": "BUY",
            "quantity": 0.02,
            "price": market_data.get('close', [50000])[-1],
            "order_id": f"mock_order_{int(time.time())}",
            "signals": signals,
            "risk_check": risk_check,
            "timestamp": datetime.now().isoformat()
        }

        return trade_result

    async def get_trading_status(self) -> Dict[str, Any]:
        """Mock status check"""
        print("📊 Mock: Getting trading status")
        await asyncio.sleep(0.1)

        return {
            "orchestrator": {
                "initialized": self.is_initialized,
                "active_sessions": len(self.active_sessions)
            },
            "services": {
                "ml_training": {"status": "healthy"},
                "reinforcement_learning": {"status": "healthy"},
                "market_regime": {"status": "healthy"},
                "risk_management": {"status": "healthy"},
                "trading_execution": {"status": "healthy"},
                "backtesting": {"status": "healthy"}
            },
            "sessions": list(self.active_sessions.values())
        }

    async def get_performance_report(self, symbol: str = None) -> Dict[str, Any]:
        """Mock performance report"""
        print(f"📈 Mock: Getting performance report for {symbol or 'all'}")
        await asyncio.sleep(0.2)

        return {
            "symbol": symbol,
            "total_return": 0.156,
            "sharpe_ratio": 2.34,
            "max_drawdown": -0.085,
            "win_rate": 0.62,
            "total_trades": 45,
            "avg_trade_return": 0.034,
            "period": "30 days"
        }

    async def stop_all_trading(self) -> Dict[str, Any]:
        """Mock emergency stop"""
        print("🛑 Mock: Emergency stop - stopping all trading")
        await asyncio.sleep(0.2)

        stopped_sessions = list(self.active_sessions.keys())
        self.active_sessions.clear()

        return {
            "success": True,
            "stopped_sessions": stopped_sessions,
            "message": "All trading activities stopped"
        }


async def demo_trading_session_lifecycle(orchestrator):
    """Demonstrate complete trading session lifecycle"""
    print("\n" + "="*70)
    print("🚀 TRADING SESSION LIFECYCLE DEMO")
    print("="*70)

    # Configuration for trading session
    session_config = {
        "symbol": "BTC/USDT",
        "capital": 10000.0,
        "risk_per_trade": 0.02,
        "max_positions": 3,
        "strategy": "ai_ensemble",
        "timeframe": "1h",
        "paper_trading": True
    }

    print(f"\n📋 Session Configuration:")
    print(f"   Symbol: {session_config['symbol']}")
    print(f"   Capital: ${session_config['capital']}")
    print(f"   Risk per Trade: {session_config['risk_per_trade']:.1%}")
    print(f"   Strategy: {session_config['strategy']}")
    print(f"   Paper Trading: {session_config['paper_trading']}")

    # Start trading session
    print(f"\n🚀 Starting trading session...")
    start_time = time.time()
    session = await orchestrator.start_trading_session(session_config)
    session_time = time.time() - start_time

    print(f"Session started in {session_time:.2f}s")
    print(f"   Session ID: {session['session_id']}")
    print(f"   Services Initialized: {', '.join(session['services_initialized'])}")

    # Get status
    print(f"\n📊 Checking orchestrator status...")
    status = await orchestrator.get_trading_status()

    print(f"   Orchestrator Initialized: {status['orchestrator']['initialized']}")
    print(f"   Active Sessions: {status['orchestrator']['active_sessions']}")

    print("   Service Status:")
    for service_name, service_status in status['services'].items():
        status_indicator = "✅" if service_status.get('status') == 'healthy' else "❌"
        print(f"     {status_indicator} {service_name}: {service_status.get('status', 'unknown')}")

    return session


async def demo_smart_trading(orchestrator, session):
    """Demonstrate AI-powered smart trading"""
    print("\n" + "="*70)
    print("🤖 SMART TRADING DEMO")
    print("="*70)

    symbol = session['symbol']

    # Simulate market data
    market_data = {
        "close": [49500, 49800, 50200, 49900, 50300, 50100, 50500],
        "volume": [1200, 1100, 1300, 1250, 1400, 1350, 1500],
        "high": [49600, 49900, 50300, 50000, 50400, 50200, 50600],
        "low": [49400, 49700, 50100, 49800, 50200, 50000, 50400],
        "portfolio_value": 10000,
        "daily_loss": -50
    }

    print(f"\n📊 Market Data for {symbol}:")
    print(f"   Current Price: ${market_data['close'][-1]}")
    print(f"   Recent Prices: {market_data['close'][-5:]}")
    print(f"   Portfolio Value: ${market_data['portfolio_value']}")
    print(f"   Daily P&L: ${market_data['daily_loss']}")

    # Execute smart trade
    print(f"\n🤖 Executing smart trade analysis...")
    start_time = time.time()
    trade_result = await orchestrator.execute_smart_trade(symbol, market_data)
    trade_time = time.time() - start_time

    print(f"Smart trade executed in {trade_time:.2f}s")
    print(f"   Action: {trade_result['action']}")
    print(f"   Quantity: {trade_result['quantity']}")
    print(f"   Price: ${trade_result['price']}")

    # Show AI signals
    signals = trade_result.get('signals', {})
    if signals:
        print("   AI Signals:")
        if 'ml_signal' in signals:
            ml = signals['ml_signal']
            print(f"     🤖 ML: {ml.get('action', 'HOLD')} (confidence: {ml.get('confidence', 0):.2f})")
        if 'rl_signal' in signals:
            rl = signals['rl_signal']
            print(f"     🧠 RL: {rl.get('action', 'HOLD')} (confidence: {rl.get('confidence', 0):.2f})")
        if 'regime' in signals:
            print(f"     📈 Regime: {signals['regime'].replace('_', ' ').title()}")

    # Show risk check
    risk_check = trade_result.get('risk_check', {})
    if risk_check:
        print("   Risk Check:")
        print(f"     ✅ Within Limits: {risk_check.get('within_limits', False)}")
        print(f"     📏 Position Size: {risk_check.get('position_size', 0):.3f}")
        print(f"     🛡️  Stop Loss: {risk_check.get('stop_loss', 0):.1%}")

    return trade_result


async def demo_performance_monitoring(orchestrator):
    """Demonstrate performance monitoring and reporting"""
    print("\n" + "="*70)
    print("📈 PERFORMANCE MONITORING DEMO")
    print("="*70)

    # Get overall performance
    print(f"\n📊 Getting overall performance report...")
    performance = await orchestrator.get_performance_report()

    print("   Performance Metrics:")
    print(f"     💰 Total Return: {performance.get('total_return', 0):.2%}")
    print(f"     📊 Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}")
    print(f"     📉 Max Drawdown: {performance.get('max_drawdown', 0):.2%}")
    print(f"     🎯 Win Rate: {performance.get('win_rate', 0):.1%}")
    print(f"     📈 Total Trades: {performance.get('total_trades', 0)}")
    print(f"     📊 Avg Trade Return: {performance.get('avg_trade_return', 0):.2%}")
    print(f"     📅 Period: {performance.get('period', 'N/A')}")

    # Get symbol-specific performance
    print(f"\n📊 Getting BTC/USDT specific performance...")
    btc_performance = await orchestrator.get_performance_report("BTC/USDT")

    print("   BTC/USDT Performance:")
    print(f"     💰 Total Return: {btc_performance.get('total_return', 0):.2%}")
    print(f"     📊 Sharpe Ratio: {btc_performance.get('sharpe_ratio', 0):.2f}")
    print(f"     📉 Max Drawdown: {btc_performance.get('max_drawdown', 0):.2%}")


async def demo_emergency_stop(orchestrator):
    """Demonstrate emergency stop functionality"""
    print("\n" + "="*70)
    print("🛑 EMERGENCY STOP DEMO")
    print("="*70)

    print(f"\n🚨 Simulating emergency stop scenario...")

    # Execute emergency stop
    stop_result = await orchestrator.stop_all_trading()

    print(f"   ✅ Success: {stop_result.get('success', False)}")
    print(f"   📊 Stopped Sessions: {len(stop_result.get('stopped_sessions', []))}")
    print(f"   💬 Message: {stop_result.get('message', 'N/A')}")

    # Verify all sessions stopped
    final_status = await orchestrator.get_trading_status()
    print(f"   📊 Final Active Sessions: {final_status['orchestrator']['active_sessions']}")


async def demo_service_orchestration():
    """Demonstrate the power of service orchestration"""
    print("\n" + "="*70)
    print("🎼 SERVICE ORCHESTRATION DEMO")
    print("="*70)

    print("\n🔧 Service Orchestration Features:")
    print("   ✅ Multi-Agent AI Coordination")
    print("   ✅ Risk-Aware Decision Making")
    print("   ✅ Market Regime Awareness")
    print("   ✅ Real-time Performance Monitoring")
    print("   ✅ Automated Risk Controls")
    print("   ✅ Ensemble Signal Generation")
    print("   ✅ Adaptive Strategy Execution")

    print("\n🤖 AI Services Integration:")
    print("   🧠 ML Training Service - Advanced model training")
    print("   🎯 Reinforcement Learning - Market-aware RL agents")
    print("   📊 Market Regime Service - Multi-timeframe analysis")
    print("   🛡️  Risk Management Service - Position sizing & controls")
    print("   ⚡ Trading Execution Service - Order management")
    print("   📈 Backtesting Service - Historical validation")

    print("\n📊 Orchestration Benefits:")
    print("   🎯 Consistent Outperformance vs Buy&Hold")
    print("   🛡️  Automated Risk Management")
    print("   ⚡ Real-time Adaptation")
    print("   📈 Performance Transparency")
    print("   🔄 Continuous Learning")


async def main():
    """Main demo function"""
    print("🚀 TradPal Trading Service Demo")
    print("="*70)
    print("This demo showcases the core functionality of the Trading Service")
    print("including session orchestration, AI-powered trading, risk management,")
    print("performance monitoring, and emergency controls.")
    print("="*70)

    # Initialize orchestrator
    if TRADING_SERVICE_AVAILABLE:
        print("\n✅ Using real TradingServiceOrchestrator")
        orchestrator = TradingServiceOrchestrator()
        await orchestrator.initialize()
    else:
        print("\n⚠️  Using MockTradingServiceOrchestrator for demonstration")
        orchestrator = MockTradingServiceOrchestrator()
        await orchestrator.initialize()

    try:
        # Run all demos
        session = await demo_trading_session_lifecycle(orchestrator)
        trade_result = await demo_smart_trading(orchestrator, session)
        await demo_performance_monitoring(orchestrator)
        await demo_emergency_stop(orchestrator)
        await demo_service_orchestration()

        print("\n" + "="*70)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("The Trading Service provides:")
        print("• AI-powered trading orchestration")
        print("• Multi-agent signal generation")
        print("• Risk-aware execution")
        print("• Market regime awareness")
        print("• Real-time performance monitoring")
        print("• Automated emergency controls")
        print("\nFor production use, ensure all services are running:")
        print("- Trading Service: localhost:8002")
        print("- Data Service: localhost:8001")
        print("- Redis: localhost:6379")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Make sure the Trading Service and dependencies are running.")


if __name__ == "__main__":
    # Run the async demo
    asyncio.run(main())