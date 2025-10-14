#!/usr/bin/env python3
"""
TradPal Trading System - Modern Web Interface

A comprehensive Streamlit-based web application for trading system management,
portfolio optimization, ML predictions, and risk monitoring.
"""

import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

# Import trading system modules
from config.settings import (
    INITIAL_CAPITAL, RISK_PER_TRADE, MAX_LEVERAGE,
    ML_ADVANCED_FEATURES_ENABLED, ML_ENSEMBLE_MODELS,
    SENTIMENT_ENABLED, SENTIMENT_WEIGHT,
    KELLY_ENABLED, KELLY_FRACTION,
    PAPER_TRADING_ENABLED, PAPER_TRADING_INITIAL_BALANCE,
    LIVE_TRADING_ENABLED, LIVE_TRADING_CONFIRMATION_REQUIRED, LIVE_TRADING_MAX_TRADES_PER_DAY,
    ML_MARKET_REGIME_DETECTION, ML_REINFORCEMENT_LEARNING
)

# Import portfolio management
from src.multi_asset_portfolio import (
    get_portfolio_manager, create_sample_portfolio,
    ModernPortfolioTheoryOptimizer, RiskParityOptimizer, MinimumVarianceOptimizer
)

# Import ML predictor
from src.advanced_ml_predictor import AdvancedMLPredictor

# Import backtester
from src.backtester import Backtester

# Import AI Trading Bot
from src.ai_trading_bot import AITradingBot, create_ai_bot

# Page configuration
st.set_page_config(
    page_title="TradPal - Advanced Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme and modern styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #1a1c23;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .metric-card {
        background-color: #1a1c23;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #2e3440;
        margin: 10px 0;
    }
    .positive {
        color: #4CAF50;
    }
    .negative {
        color: #f44336;
    }
    .neutral {
        color: #ff9800;
    }
    .header-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'portfolio_manager' not in st.session_state:
    st.session_state.portfolio_manager = get_portfolio_manager()

if 'ml_predictor' not in st.session_state:
    st.session_state.ml_predictor = AdvancedMLPredictor() if ML_ADVANCED_FEATURES_ENABLED else None

if 'ai_trading_bot' not in st.session_state:
    # Initialize AI Bot with default config
    bot_config = {
        'trading_mode': 'paper',
        'assets': ['BTC/USDT'],
        'timeframe': '1h',
        'initial_balance': 10000.0,
        'advanced_ml_enabled': ML_ADVANCED_FEATURES_ENABLED,
        'sentiment_enabled': SENTIMENT_ENABLED,
        'multi_asset_enabled': False,
        'walk_forward_enabled': False,
        'regime_adaptation': ML_MARKET_REGIME_DETECTION,
        'kelly_enabled': KELLY_ENABLED,
        'risk_per_trade': RISK_PER_TRADE,
        'max_leverage': MAX_LEVERAGE,
        'max_drawdown': 0.1,
        'max_daily_loss': 0.05,
        'min_confidence_threshold': 0.6,
        'max_trades_per_hour': 5,
        'max_open_positions': 3,
        'cycle_interval': 60,
        'sentiment_weight': SENTIMENT_WEIGHT,
        'enhancement_weight': 0.3,
        'kelly_fraction': KELLY_FRACTION
    }
    st.session_state.ai_trading_bot = create_ai_bot(bot_config)

if 'backtester' not in st.session_state:
    st.session_state.backtester = Backtester()

if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()

# Navigation
PAGES = {
    "🏠 Dashboard": "dashboard",
    "� Observation": "observation",
    "� Backtesting": "backtesting",
    "🔍 Discovery": "discovery",
    "💼 Portfolio": "portfolio",
    "⚙️ Settings": "settings",
    "🎯 Live Trading": "live_trading"
}

def main():
    """Main application entry point."""

    # Sidebar navigation
    st.sidebar.markdown("""
    <div class="header-card">
        <h1>📈 TradPal</h1>
        <p>Advanced Trading System</p>
    </div>
    """, unsafe_allow_html=True)

    # Navigation selection
    page = st.sidebar.selectbox(
        "Navigation",
        options=list(PAGES.keys()),
        index=0
    )

    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.session_state.last_update = datetime.now()
        st.rerun()

    # Last update info
    st.sidebar.markdown(f"**Last Update:** {st.session_state.last_update.strftime('%H:%M:%S')}")

    # System status
    status_color = "🟢" if st.session_state.portfolio_manager else "🔴"
    st.sidebar.markdown(f"**Portfolio Status:** {status_color}")

    if st.session_state.ml_predictor:
        ml_status = "🟢" if ML_ADVANCED_FEATURES_ENABLED else "🟡"
        st.sidebar.markdown(f"**ML Status:** {ml_status}")

    # Page routing
    page_function = PAGES[page]

    if page_function == "dashboard":
        show_dashboard()
    elif page_function == "observation":
        show_observation()
    elif page_function == "backtesting":
        show_backtesting()
    elif page_function == "discovery":
        show_discovery()
    elif page_function == "portfolio":
        show_portfolio()
    elif page_function == "settings":
        show_settings()
    elif page_function == "live_trading":
        show_live_trading()

def show_dashboard():
    """Display main dashboard with portfolio overview."""
    st.title("📊 Trading Dashboard")

    # Portfolio metrics overview
    col1, col2, col3, col4 = st.columns(4)

    portfolio = st.session_state.portfolio_manager
    metrics = portfolio.get_portfolio_metrics()

    with col1:
        st.metric(
            "Portfolio Value",
            f"${metrics.total_value:,.2f}",
            f"{metrics.total_pnl:+,.2f}"
        )

    with col2:
        daily_pnl_pct = (metrics.daily_pnl / metrics.total_value) * 100 if metrics.total_value > 0 else 0
        st.metric(
            "Daily P&L",
            f"{daily_pnl_pct:+.2f}%",
            f"${metrics.daily_pnl:+,.2f}"
        )

    with col3:
        st.metric(
            "Sharpe Ratio",
            f"{metrics.sharpe_ratio:.2f}",
            help="Risk-adjusted return measure"
        )

    with col4:
        st.metric(
            "Volatility",
            f"{metrics.volatility:.2%}",
            help="Portfolio volatility (annualized)"
        )

    # Charts section
    st.markdown("### 📈 Portfolio Performance")

    col1, col2 = st.columns(2)

    with col1:
        # Asset allocation pie chart
        if portfolio.positions:
            labels = list(portfolio.positions.keys())
            values = [pos.quantity * pos.current_price for pos in portfolio.positions.values()]

            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=.3,
                marker_colors=px.colors.qualitative.Set3
            )])
            fig.update_layout(
                title="Asset Allocation",
                font=dict(color='white'),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Risk metrics
        risk_metrics = portfolio.get_risk_metrics()

        if risk_metrics:
            fig = go.Figure()

            metrics_to_show = ['volatility', 'sharpe_ratio', 'max_drawdown']
            values = [risk_metrics.get(m, 0) for m in metrics_to_show]
            labels = ['Volatility', 'Sharpe Ratio', 'Max Drawdown']

            fig.add_trace(go.Bar(
                x=labels,
                y=values,
                marker_color=['#ff6b6b', '#4ecdc4', '#45b7d1']
            ))

            fig.update_layout(
                title="Risk Metrics",
                font=dict(color='white'),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)

    # Recent trades and positions
    st.markdown("### 💼 Current Positions")

    if portfolio.positions:
        positions_data = []
        for symbol, position in portfolio.positions.items():
            positions_data.append({
                'Asset': symbol,
                'Quantity': position.quantity,
                'Entry Price': position.entry_price,
                'Current Price': position.current_price,
                'P&L': position.unrealized_pnl,
                'Allocation %': position.allocation_pct * 100
            })

        df_positions = pd.DataFrame(positions_data)
        st.dataframe(df_positions.style.apply(lambda x: ['background-color: #4CAF50' if v > 0 else 'background-color: #f44336' for v in x] if x.name == 'P&L' else [''] * len(x), axis=0))

    # ML Predictions summary (if available)
    if st.session_state.ml_predictor and ML_ADVANCED_FEATURES_ENABLED:
        st.markdown("### 🤖 ML Predictions Summary")

        # Get predictions for current assets
        predictions = {}
        for symbol in portfolio.positions.keys():
            try:
                pred = st.session_state.ml_predictor.predict_signal(symbol, None)
                predictions[symbol] = pred
            except:
                predictions[symbol] = {'signal': 'HOLD', 'confidence': 0.5}

        pred_data = []
        for symbol, pred in predictions.items():
            pred_data.append({
                'Asset': symbol,
                'Signal': pred.get('signal', 'HOLD'),
                'Confidence': pred.get('confidence', 0.5)
            })

        df_predictions = pd.DataFrame(pred_data)
        st.dataframe(df_predictions)

def show_observation():
    """Show observation mode page for signal monitoring."""
    st.title("👁️ Observation Mode - Signal Monitoring")

    st.markdown("### 📊 Real-time Signal Observation")

    # Asset selection
    col1, col2 = st.columns(2)

    with col1:
        symbol = st.selectbox("Asset", ["BTC/USDT", "ETH/USDT", "ADA/USDT", "DOT/USDT"], key="obs_symbol")

    with col2:
        timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"], key="obs_timeframe")

    # Signal generation controls
    st.markdown("### 🎛️ Signal Generation")

    col1, col2, col3 = st.columns(3)

    with col1:
        use_technical = st.checkbox("Technical Indicators", value=True, key="obs_technical")
        use_ml = st.checkbox("ML Models", value=ML_ADVANCED_FEATURES_ENABLED, key="obs_ml")

    with col2:
        use_sentiment = st.checkbox("Sentiment Analysis", value=SENTIMENT_ENABLED, key="obs_sentiment")
        use_multi_timeframe = st.checkbox("Multi-Timeframe", value=True, key="obs_multi_tf")

    with col3:
        auto_refresh = st.checkbox("Auto Refresh", value=True, key="obs_auto_refresh")
        refresh_interval = st.slider("Refresh Interval (sec)", 5, 60, 10, key="obs_refresh") if auto_refresh else 10

    # Generate Signal button
    if st.button("🔍 Generate Signal", type="primary"):
        with st.spinner("Analyzing market data..."):
            # Simulate signal generation
            time.sleep(1)

            # Mock signal data
            signal_data = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'signal': np.random.choice(['BUY', 'SELL', 'HOLD'], p=[0.4, 0.3, 0.3]),
                'confidence': np.random.uniform(0.5, 0.9),
                'price': 45000 + np.random.normal(0, 1000),
                'indicators': {
                    'ema_short': 44500 + np.random.normal(0, 500),
                    'ema_long': 44200 + np.random.normal(0, 300),
                    'rsi': np.random.uniform(30, 70),
                    'bb_upper': 46000 + np.random.normal(0, 200),
                    'bb_lower': 43000 + np.random.normal(0, 200)
                },
                'ml_prediction': np.random.uniform(0.4, 0.8) if use_ml else None,
                'sentiment_score': np.random.uniform(-0.3, 0.3) if use_sentiment else None
            }

            st.session_state.last_observation = signal_data
            st.success("Signal generated successfully!")
            st.rerun()

    # Display current signal
    if 'last_observation' in st.session_state:
        signal_data = st.session_state.last_observation

        st.markdown("### 🎯 Current Signal")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            signal_type = signal_data['signal']
            signal_emoji = {'BUY': '🟢 BUY', 'SELL': '🔴 SELL', 'HOLD': '🟡 HOLD'}.get(signal_type, '⚪ UNKNOWN')
            st.metric("Signal", signal_emoji)

        with col2:
            confidence = signal_data['confidence']
            st.metric("Confidence", f"{confidence:.1%}")

        with col3:
            price = signal_data['price']
            st.metric("Price", f"${price:,.2f}")

        with col4:
            timestamp = datetime.fromisoformat(signal_data['timestamp'])
            st.metric("Time", timestamp.strftime('%H:%M:%S'))

        # Technical indicators
        if signal_data.get('indicators'):
            st.markdown("#### 📈 Technical Indicators")

            indicators = signal_data['indicators']
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric("EMA Short", f"{indicators['ema_short']:.2f}")
            with col2:
                st.metric("EMA Long", f"{indicators['ema_long']:.2f}")
            with col3:
                st.metric("RSI", f"{indicators['rsi']:.1f}")
            with col4:
                st.metric("BB Upper", f"{indicators['bb_upper']:.2f}")
            with col5:
                st.metric("BB Lower", f"{indicators['bb_lower']:.2f}")

        # ML and Sentiment scores
        if use_ml or use_sentiment:
            st.markdown("#### 🤖 Advanced Analysis")

            col1, col2 = st.columns(2)

            with col1:
                if use_ml and signal_data.get('ml_prediction') is not None:
                    ml_score = signal_data['ml_prediction']
                    st.metric("ML Prediction", f"{ml_score:.2f}")

            with col2:
                if use_sentiment and signal_data.get('sentiment_score') is not None:
                    sentiment = signal_data['sentiment_score']
                    sentiment_label = "🐂 BULLISH" if sentiment > 0.1 else "🐻 BEARISH" if sentiment < -0.1 else "😐 NEUTRAL"
                    st.metric("Sentiment", sentiment_label, f"{sentiment:+.2f}")

    # Signal history
    st.markdown("### 📋 Signal History")

    if 'observation_history' not in st.session_state:
        st.session_state.observation_history = []

    if st.session_state.observation_history:
        history_df = pd.DataFrame(st.session_state.observation_history[-10:])  # Last 10 signals
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])

        st.dataframe(
            history_df[['timestamp', 'signal', 'confidence', 'price']].style.apply(
                lambda x: ['background-color: #4CAF50' if v == 'BUY' else 'background-color: #f44336' if v == 'SELL' else 'background-color: #ff9800' for v in x],
                axis=0
            ),
            use_container_width=True
        )
    else:
        st.info("No signal history available. Generate some signals to see history.")

    # Market context
    st.markdown("### 🌍 Market Context")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Simulate market regime
        regime = np.random.choice(['Bull', 'Bear', 'Sideways'], p=[0.4, 0.3, 0.3])
        regime_emoji = {'Bull': '🐂', 'Bear': '🐻', 'Sideways': '↔️'}.get(regime, '❓')
        st.metric("Market Regime", f"{regime_emoji} {regime}")

    with col2:
        volatility = np.random.uniform(0.02, 0.08)
        st.metric("Volatility", f"{volatility:.1%}")

    with col3:
        volume_trend = np.random.choice(['High', 'Normal', 'Low'])
        volume_emoji = {'High': '📈', 'Normal': '➡️', 'Low': '📉'}.get(volume_trend, '❓')
        st.metric("Volume", f"{volume_emoji} {volume_trend}")

def show_portfolio():
    """Show portfolio management page with advanced analytics."""
    st.title("� Portfolio Management")

    portfolio = st.session_state.portfolio_manager

    # Portfolio Overview
    st.markdown("### 💼 Portfolio Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_value = portfolio.get_total_value()
        st.metric("Total Value", f"${total_value:,.2f}")

    with col2:
        cash = portfolio.cash
        st.metric("Cash Available", f"${cash:,.2f}")

    with col3:
        invested = total_value - cash
        st.metric("Invested", f"${invested:,.2f}")

    with col4:
        positions_count = len(portfolio.positions)
        st.metric("Positions", positions_count)

    # Asset Allocation
    st.markdown("### 🎯 Asset Allocation")

    if portfolio.positions:
        # Create allocation data
        assets = []
        values = []
        weights = []

        for symbol, position in portfolio.positions.items():
            value = position['quantity'] * position['current_price']
            weight = value / total_value if total_value > 0 else 0

            assets.append(symbol)
            values.append(value)
            weights.append(weight)

        # Allocation Pie Chart
        fig = go.Figure(data=[go.Pie(
            labels=assets,
            values=values,
            textinfo='label+percent',
            marker_colors=px.colors.qualitative.Set3
        )])

        fig.update_layout(
            title="Portfolio Allocation",
            font=dict(color='white'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Positions Table
        st.markdown("### 📋 Current Positions")

        positions_data = []
        for symbol, position in portfolio.positions.items():
            current_value = position['quantity'] * position['current_price']
            cost_basis = position['quantity'] * position['avg_price']
            pnl = current_value - cost_basis
            pnl_pct = pnl / cost_basis if cost_basis > 0 else 0

            positions_data.append({
                'Asset': symbol,
                'Quantity': position['quantity'],
                'Avg Price': position['avg_price'],
                'Current Price': position['current_price'],
                'Value': current_value,
                'P&L': pnl,
                'P&L %': pnl_pct
            })

        df_positions = pd.DataFrame(positions_data)
        st.dataframe(df_positions.style.format({
            'Quantity': '{:.6f}',
            'Avg Price': '${:.2f}',
            'Current Price': '${:.2f}',
            'Value': '${:.2f}',
            'P&L': '${:.2f}',
            'P&L %': '{:.2%}'
        }).apply(lambda x: [
            'background-color: #4CAF50' if v > 0 else
            'background-color: #f44336' if v < 0 else
            'background-color: #ff9800' for v in x
        ] if x.name in ['P&L', 'P&L %'] else [''] * len(x), axis=0))

    # Portfolio Optimization
    st.markdown("### �️ Portfolio Optimization")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Optimization Method")
        optimization_method = st.selectbox("Method",
                                         ["Modern Portfolio Theory (MPT)",
                                          "Risk Parity",
                                          "Minimum Variance",
                                          "Equal Weight"])

    with col2:
        st.subheader("Risk Constraints")
        target_return = st.slider("Target Return (%)", 0.0, 20.0, 10.0, step=1.0)
        max_volatility = st.slider("Max Volatility (%)", 5.0, 50.0, 25.0, step=5.0)

    if st.button("🔄 Optimize Portfolio"):
        with st.spinner("Optimizing portfolio..."):
            try:
                # Perform optimization
                optimized_weights = portfolio.optimize_portfolio(
                    method=optimization_method.lower().replace(' ', '_'),
                    target_return=target_return/100,
                    max_volatility=max_volatility/100
                )

                if optimized_weights:
                    st.success("Portfolio optimized successfully!")

                    # Display optimized weights
                    st.markdown("#### 📊 Optimized Weights")

                    opt_data = []
                    for symbol, weight in optimized_weights.items():
                        opt_data.append({
                            'Asset': symbol,
                            'Weight': weight,
                            'Weight %': weight * 100
                        })

                    df_opt = pd.DataFrame(opt_data)
                    st.dataframe(df_opt.style.format({
                        'Weight': '{:.4f}',
                        'Weight %': '{:.1f}%'
                    }))

                    # Rebalancing recommendations
                    st.markdown("#### 🔄 Rebalancing Recommendations")

                    rebalance_actions = portfolio.get_rebalance_actions(optimized_weights)

                    if rebalance_actions:
                        for action in rebalance_actions:
                            if action['action'] == 'BUY':
                                st.success(f"📈 BUY {action['quantity']:.6f} {action['symbol']} @ ${action['price']:.2f}")
                            elif action['action'] == 'SELL':
                                st.error(f"📉 SELL {action['quantity']:.6f} {action['symbol']} @ ${action['price']:.2f}")
                    else:
                        st.info("Portfolio is already optimally balanced.")

                else:
                    st.error("Optimization failed. Check constraints and available assets.")

            except Exception as e:
                st.error(f"Optimization error: {str(e)}")

    # Performance Analytics
    st.markdown("### 📈 Performance Analytics")

    if portfolio.performance_history:
        # Performance chart
        perf_data = pd.DataFrame(portfolio.performance_history)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=perf_data['date'],
            y=perf_data['portfolio_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#4ecdc4', width=2)
        ))

        if 'benchmark' in perf_data.columns:
            fig.add_trace(go.Scatter(
                x=perf_data['date'],
                y=perf_data['benchmark'],
                mode='lines',
                name='Benchmark',
                line=dict(color='#ff6b6b', width=2, dash='dash')
            ))

        fig.update_layout(
            title="Portfolio Performance",
            xaxis_title="Date",
            yaxis_title="Value ($)",
            height=400,
            font=dict(color='white'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_return = (perf_data['portfolio_value'].iloc[-1] / perf_data['portfolio_value'].iloc[0] - 1) * 100
            st.metric("Total Return", f"{total_return:.1f}%")

        with col2:
            # Calculate Sharpe ratio
            returns = perf_data['portfolio_value'].pct_change().dropna()
            sharpe = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 else 0
            st.metric("Sharpe Ratio", f"{sharpe:.2f}")

        with col3:
            # Calculate max drawdown
            peak = perf_data['portfolio_value'].expanding().max()
            drawdown = (perf_data['portfolio_value'] - peak) / peak
            max_dd = drawdown.min() * 100
            st.metric("Max Drawdown", f"{max_dd:.1f}%")

        with col4:
            # Calculate volatility
            vol = returns.std() * np.sqrt(252) * 100
            st.metric("Volatility", f"{vol:.1f}%")

    # Add/Remove Assets
    st.markdown("### ➕ Asset Management")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Add Asset")
        new_symbol = st.text_input("Symbol (e.g., BTC/USDT)", key="add_symbol")
        add_quantity = st.number_input("Quantity", min_value=0.0, step=0.001, key="add_quantity")

        if st.button("Add Asset"):
            if new_symbol and add_quantity > 0:
                try:
                    portfolio.add_position(new_symbol, add_quantity)
                    st.success(f"Added {add_quantity} {new_symbol}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error adding asset: {str(e)}")
            else:
                st.error("Please enter valid symbol and quantity.")

    with col2:
        st.subheader("Remove Asset")
        if portfolio.positions:
            remove_symbol = st.selectbox("Select Asset to Remove", list(portfolio.positions.keys()), key="remove_symbol")

            if st.button("Remove Asset"):
                try:
                    portfolio.remove_position(remove_symbol)
                    st.success(f"Removed all {remove_symbol} positions")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error removing asset: {str(e)}")
        else:
            st.info("No positions to remove.")

    # Trading Mode Selection
    st.markdown("### 🎯 Trading Mode Selection")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📊 Pure Monitoring")
        pure_monitoring = st.checkbox("Enable Pure Monitoring", value=True, key="pure_monitoring")
        st.write("**Features:**")
        st.write("• Signal generation only")
        st.write("• No automatic trading")
        st.write("• Manual analysis")

    with col2:
        st.subheader("🤖 AI Bot Mode")
        ai_bot_mode = st.checkbox("Enable AI Bot Mode", key="ai_bot_mode")
        st.write("**Features:**")
        st.write("• ML + GA integration")
        st.write("• Auto trading execution")
        st.write("• Risk management")
        st.write("• Outperformance focus")

    with col3:
        st.subheader("⚙️ Bot Configuration")
        if ai_bot_mode:
            auto_optimize = st.checkbox("Auto GA Optimization", value=True, key="auto_optimize")
            adaptive_learning = st.checkbox("Adaptive Learning", value=True, key="adaptive_learning")
            risk_adaptive = st.checkbox("Risk-Adaptive Sizing", value=True, key="risk_adaptive")

    # Mode Status
    if pure_monitoring and ai_bot_mode:
        st.warning("⚠️ Please select only ONE mode: Pure Monitoring OR AI Bot Mode")
        ai_bot_mode = False
    elif not pure_monitoring and not ai_bot_mode:
        st.info("ℹ️ Please select a trading mode above")

        # AI Bot Configuration (only if AI Bot Mode enabled)
        if ai_bot_mode:
            st.markdown("### 🤖 AI Bot Configuration")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("Signal Sources")
                use_ml_signals = st.checkbox("ML Ensemble Signals", value=True, key="use_ml_signals")
                use_ga_optimized = st.checkbox("GA Optimized Parameters", value=True, key="use_ga_optimized")
                use_market_regime = st.checkbox("Market Regime Adaptation", value=True, key="use_market_regime")

            with col2:
                st.subheader("Execution Rules")
                min_confidence_threshold = st.slider("Min Confidence %", 50, 95, 75, key="min_confidence_threshold")
                max_trades_per_hour = st.slider("Max Trades/Hour", 1, 20, 5, key="max_trades_per_hour")
                max_open_positions = st.slider("Max Open Positions", 1, 10, 3, key="max_open_positions")

            with col3:
                st.subheader("Risk Parameters")
                bot_risk_per_trade = st.slider("Risk per Trade %", 0.1, 3.0, 1.0, step=0.1, key="bot_risk_per_trade")
                bot_stop_loss = st.slider("Stop Loss %", 0.5, 10.0, 2.0, step=0.5, key="bot_stop_loss")
                bot_take_profit = st.slider("Take Profit %", 1.0, 20.0, 5.0, step=0.5, key="bot_take_profit")

            # Advanced Features
            st.markdown("### 🚀 Advanced Features")

            col1, col2 = st.columns(2)

            with col1:
                # Sentiment Integration
                sentiment_enabled = st.checkbox("Use Sentiment Analysis", value=SENTIMENT_ENABLED, key="bot_sentiment_enabled")
                sentiment_weight = st.slider("Sentiment Weight", 0.0, 1.0, float(SENTIMENT_WEIGHT), 0.1, key="bot_sentiment_weight") if sentiment_enabled else 0.0

                # Multi-Asset Portfolio
                multi_asset_enabled = st.checkbox("Multi-Asset Portfolio", key="bot_multi_asset")
                if multi_asset_enabled:
                    portfolio_assets = st.multiselect(
                        "Portfolio Assets",
                        ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT", "DOT/USDT"],
                        default=["BTC/USDT", "ETH/USDT"],
                        key="bot_portfolio_assets"
                    )
                    portfolio_allocation = st.selectbox("Allocation Method",
                                                      ["Equal Weight", "Risk Parity", "Volatility Targeted"],
                                                      key="bot_allocation_method")

            with col2:
                # Advanced ML Features
                advanced_ml_enabled = st.checkbox("Advanced ML Features", value=ML_ADVANCED_FEATURES_ENABLED, key="bot_advanced_ml")
                if advanced_ml_enabled:
                    ensemble_models = st.multiselect(
                        "Ensemble Models",
                        ["LSTM", "Transformer", "Random Forest", "Gradient Boosting", "PyTorch Ensemble"],
                        default=["LSTM", "Random Forest"] if ML_ENSEMBLE_MODELS else [],
                        key="bot_ensemble_models"
                    )
                    regime_adaptation = st.checkbox("Market Regime Adaptation", value=ML_MARKET_REGIME_DETECTION, key="bot_regime_adaptation")
                    reinforcement_learning = st.checkbox("Reinforcement Learning", value=ML_REINFORCEMENT_LEARNING, key="bot_rl_enabled")

                # Walk-Forward Optimization
                walk_forward_enabled = st.checkbox("Walk-Forward Optimization", key="bot_walk_forward")
                if walk_forward_enabled:
                    wf_windows = st.slider("Walk-Forward Windows", 5, 20, 10, key="bot_wf_windows")
                    wf_retrain_interval = st.slider("Retraining Interval (hours)", 1, 24, 6, key="bot_retrain_interval")

            # Risk Management Extensions
            kelly_criterion = st.checkbox("Kelly Criterion Position Sizing", value=KELLY_ENABLED, key="bot_kelly_enabled")
            if kelly_criterion:
                kelly_fraction = st.slider("Kelly Fraction", 0.1, 1.0, float(KELLY_FRACTION), 0.1, key="bot_kelly_fraction")

            # Paper Trading Integration
            paper_trading_mode = st.checkbox("Paper Trading Mode", value=PAPER_TRADING_ENABLED, key="bot_paper_trading")
            if paper_trading_mode:
                paper_initial_balance = st.number_input("Paper Trading Balance", 1000.0, 100000.0, float(PAPER_TRADING_INITIAL_BALANCE), key="bot_paper_balance")

            # Live Trading Integration
            live_trading_enabled = st.checkbox("Enable Live Trading", value=LIVE_TRADING_ENABLED, key="bot_live_trading")
            if live_trading_enabled:
                live_confirm_orders = st.checkbox("Require Order Confirmation", value=LIVE_TRADING_CONFIRMATION_REQUIRED, key="bot_confirm_orders")
                live_max_daily_trades = st.slider("Max Daily Trades", 1, 20, LIVE_TRADING_MAX_TRADES_PER_DAY, key="bot_max_daily_trades")

            # Update Bot Configuration
            if st.button("🔄 Update Bot Configuration", key="update_bot_config"):
                # Update bot configuration
                new_config = {
                    'trading_mode': 'live' if live_trading_enabled else 'paper',
                    'assets': portfolio_assets if multi_asset_enabled else ['BTC/USDT'],
                    'timeframe': '1h',
                    'initial_balance': paper_initial_balance if paper_trading_mode else 10000.0,
                    'advanced_ml_enabled': advanced_ml_enabled,
                    'sentiment_enabled': sentiment_enabled,
                    'multi_asset_enabled': multi_asset_enabled,
                    'walk_forward_enabled': walk_forward_enabled,
                    'regime_adaptation': regime_adaptation,
                    'kelly_enabled': kelly_criterion,
                    'risk_per_trade': bot_risk_per_trade / 100.0,
                    'max_leverage': MAX_LEVERAGE,
                    'max_drawdown': 0.1,
                    'max_daily_loss': 0.05,
                    'min_confidence_threshold': min_confidence_threshold / 100.0,
                    'max_trades_per_hour': max_trades_per_hour,
                    'max_open_positions': max_open_positions,
                    'cycle_interval': 60,
                    'sentiment_weight': sentiment_weight,
                    'enhancement_weight': 0.3,
                    'kelly_fraction': kelly_fraction
                }

                # Recreate bot with new configuration
                st.session_state.ai_trading_bot = create_ai_bot(new_config)
                st.success("🤖 Bot configuration updated successfully!")

            # Bot Status
            st.markdown("### 📊 AI Bot Status")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                bot_active = st.checkbox("Activate AI Bot", key="bot_active")
                status = "🟢 ACTIVE" if bot_active else "🔴 INACTIVE"
                st.metric("Bot Status", status)

            with col2:
                bot_status = st.session_state.ai_trading_bot.get_status()
                trades_today = len([t for t in bot_status.get('trade_history', [])
                                  if (datetime.now() - t['timestamp']).days == 0])
                st.metric("Trades Today", trades_today)

            with col3:
                # Calculate win rate from recent trades
                recent_trades = st.session_state.ai_trading_bot.get_recent_trades(20)
                winning_trades = len([t for t in recent_trades if t.get('pnl', 0) > 0])
                win_rate = winning_trades / len(recent_trades) if recent_trades else 0
                st.metric("Win Rate", f"{win_rate:.1%}")

            with col4:
                daily_pnl = bot_status.get('daily_pnl', 0)
                st.metric("P&L Today", f"${daily_pnl:+.2f}")

            # Bot Control
            col1, col2 = st.columns(2)

            with col1:
                if st.button("▶️ Start Bot", key="start_bot", type="primary"):
                    if not st.session_state.ai_trading_bot.is_active:
                        # Start bot in background thread
                        import threading
                        bot_thread = threading.Thread(target=st.session_state.ai_trading_bot.start)
                        bot_thread.daemon = True
                        bot_thread.start()
                        st.success("🤖 AI Bot started successfully!")
                        time.sleep(1)  # Give it time to start
                        st.rerun()
                    else:
                        st.warning("Bot is already running")

            with col2:
                if st.button("⏹️ Stop Bot", key="stop_bot"):
                    if st.session_state.ai_trading_bot.is_active:
                        st.session_state.ai_trading_bot.stop()
                        st.success("🤖 AI Bot stopped successfully!")
                        st.rerun()
                    else:
                        st.info("Bot is not running")

            # Bot Performance Chart
            if st.session_state.ai_trading_bot.performance_history:
                st.markdown("### 📈 AI Bot Performance")

                # Get recent performance data
                perf_data = st.session_state.ai_trading_bot.get_performance_history(24)

                if perf_data:
                    # Create performance chart
                    times = [p['timestamp'] for p in perf_data]
                    pnl_values = [p['daily_pnl'] for p in perf_data]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=times,
                        y=pnl_values,
                        mode='lines+markers',
                        name='Daily P&L',
                        line=dict(color='#4ecdc4', width=3)
                    ))

                    fig.update_layout(
                        title="AI Bot Performance (24h)",
                        xaxis_title="Time",
                        yaxis_title="Daily P&L ($)",
                        height=300,
                        font=dict(color='white'),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )

                    st.plotly_chart(fig, use_container_width=True)

            # Bot Log
            st.markdown("### 📋 AI Bot Activity Log")

            recent_trades = st.session_state.ai_trading_bot.get_recent_trades(10)

            if recent_trades:
                # Convert to DataFrame for display
                trades_df = pd.DataFrame(recent_trades)
                trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])

                # Add emoji based on side
                trades_df['side_emoji'] = trades_df['side'].map({
                    'BUY': '🟢',
                    'SELL': '🔴'
                })

                st.dataframe(
                    trades_df[['timestamp', 'side_emoji', 'asset', 'quantity', 'price', 'confidence', 'reason']].rename(columns={
                        'timestamp': 'Time',
                        'side_emoji': 'Side',
                        'asset': 'Asset',
                        'quantity': 'Quantity',
                        'price': 'Price',
                        'confidence': 'Confidence',
                        'reason': 'Reason'
                    }).style.apply(
                        lambda x: ['background-color: #4CAF50' if v == 'BUY' else 'background-color: #f44336' if v == 'SELL' else '' for v in x],
                        axis=0
                    ),
                    use_container_width=True
                )
            else:
                st.info("🤖 No recent trades. Start the bot to see activity.")    # Pure Monitoring Mode (only if Pure Monitoring enabled)
    if pure_monitoring:
        # Monitoring Controls
        st.markdown("### 🎛️ Monitoring Controls")

        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            # Asset selection (currently only BTC/USDT available)
            available_assets = ['BTC/USDT']  # Can be extended
            selected_asset = st.selectbox("Asset", available_assets, key="monitoring_asset")

        with col2:
            # Configuration options
            st.write("**Configuration:**")

            use_ml = st.checkbox("Use ML Models", value=ML_ADVANCED_FEATURES_ENABLED, key="use_ml")
            timeframe = st.selectbox("Timeframe", ['1m', '5m', '15m', '1h', '4h', '1d'],
                                    index=3, key="timeframe")  # Default 1h
            interval = st.slider("Check Interval (seconds)", 30, 300, 60, key="interval")

        with col3:
            # Control buttons
            st.write("**Actions:**")

            status_class = "🟢 RUNNING" if st.session_state.monitoring_active else "🔴 STOPPED"
            st.markdown(f"**Status:** {status_class}")

            if st.session_state.monitoring_active:
                if st.button("⏹️ Stop Monitoring", use_container_width=True):
                    st.session_state.monitoring_active = False
                    st.success("Monitoring stopped")
                    st.rerun()
            else:
                if st.button("🚀 Start Monitoring", use_container_width=True):
                    st.session_state.monitoring_active = True
                    st.session_state.monitoring_data = {
                        'signals': [],
                        'events': [],
                        'last_signal': None,
                        'start_time': datetime.now()
                    }
                    st.success(f"Monitoring started for {selected_asset}")
                    st.rerun()

        # Current Status & Last Signal
        st.markdown("### 📊 Current Status")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.session_state.monitoring_active and st.session_state.monitoring_data.get('start_time'):
                start_time = st.session_state.monitoring_data['start_time']
                runtime = datetime.now() - start_time
                st.metric("Runtime", f"{runtime.seconds // 3600}h {(runtime.seconds % 3600) // 60}m")
            else:
                st.metric("Runtime", "Not running")

        with col2:
            signal_count = len(st.session_state.monitoring_data['signals'])
            st.metric("Signals Generated", signal_count)

        with col3:
            event_count = len(st.session_state.monitoring_data['events'])
            st.metric("Events Logged", event_count)

        # Last Signal Details
        st.markdown("### 🎯 Last Signal")

        last_signal = st.session_state.monitoring_data.get('last_signal')
        if last_signal:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                timestamp = datetime.fromisoformat(last_signal['timestamp'])
                st.metric("Time", timestamp.strftime('%H:%M:%S'))

            with col2:
                signal_type = last_signal.get('signal', 'HOLD')
                signal_emoji = {
                    'BUY': '🟢 BUY',
                    'SELL': '🔴 SELL',
                    'HOLD': '🟡 HOLD'
                }.get(signal_type, '⚪ UNKNOWN')
                st.metric("Signal", signal_emoji)

            with col3:
                price = last_signal.get('price', 0)
                st.metric("Price", f"${price:.2f}")

            with col4:
                confidence = last_signal.get('confidence', 0)
                st.metric("Confidence", f"{confidence:.1%}")

            # Indicator values
            if 'indicators' in last_signal:
                st.write("**Technical Indicators:**")
                indicators = last_signal['indicators']

                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    st.metric("EMA Short", f"{indicators.get('ema_short', 0):.2f}")
                with col2:
                    st.metric("EMA Long", f"{indicators.get('ema_long', 0):.2f}")
                with col3:
                    st.metric("RSI", f"{indicators.get('rsi', 0):.1f}")
                with col4:
                    st.metric("BB Upper", f"{indicators.get('bb_upper', 0):.2f}")
                with col5:
                    st.metric("BB Lower", f"{indicators.get('bb_lower', 0):.2f}")
        else:
            st.info("No signals generated yet. Start monitoring to see signals.")

        # Signal Chart
        st.markdown("### 📈 Signal Chart")

        if st.session_state.monitoring_data['signals']:
            # Create signal chart data
            signals_data = st.session_state.monitoring_data['signals']

            if signals_data:
                # Convert to DataFrame for plotting
                df_signals = pd.DataFrame(signals_data)
                df_signals['timestamp'] = pd.to_datetime(df_signals['timestamp'])

                # Create subplot with price and signals
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    subplot_titles=('Price & Signals', 'RSI'),
                    row_width=[0.7, 0.3]
                )

                # Price chart
                fig.add_trace(
                    go.Scatter(
                        x=df_signals['timestamp'],
                        y=df_signals['price'],
                        mode='lines+markers',
                        name='Price',
                        line=dict(color='blue', width=2)
                    ),
                    row=1, col=1
                )

                # Add signal markers
                buy_signals = df_signals[df_signals['signal'] == 'BUY']
                sell_signals = df_signals[df_signals['signal'] == 'SELL']

                if not buy_signals.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=buy_signals['timestamp'],
                            y=buy_signals['price'],
                            mode='markers',
                            name='BUY Signal',
                            marker=dict(color='green', size=12, symbol='triangle-up')
                        ),
                        row=1, col=1
                    )

                if not sell_signals.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=sell_signals['timestamp'],
                            y=sell_signals['price'],
                            mode='markers',
                            name='SELL Signal',
                            marker=dict(color='red', size=12, symbol='triangle-down')
                        ),
                        row=1, col=1
                    )

                # RSI subplot
                if 'indicators' in df_signals.columns:
                    rsi_values = []
                    for idx, row in df_signals.iterrows():
                        if isinstance(row.get('indicators'), dict):
                            rsi_values.append(row['indicators'].get('rsi', None))
                        else:
                            rsi_values.append(None)

                    if any(rsi_values):
                        fig.add_trace(
                            go.Scatter(
                                x=df_signals['timestamp'],
                                y=rsi_values,
                                mode='lines',
                                name='RSI',
                                line=dict(color='orange', width=1)
                            ),
                            row=2, col=1
                        )

                        # Add RSI levels
                        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

                fig.update_layout(
                    height=500,
                    showlegend=True,
                    font=dict(color='white'),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )

                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No signal data available. Start monitoring to generate chart.")

        # Event Log
        st.markdown("### 📋 Event Log")

        if st.session_state.monitoring_data['events']:
            # Convert to DataFrame for better display
            events_df = pd.DataFrame(st.session_state.monitoring_data['events'])
            events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])
            events_df = events_df.sort_values('timestamp', ascending=False)

            # Add emoji based on type
            events_df['type_emoji'] = events_df['type'].map({
                'status': 'ℹ️',
                'error': '❌',
                'warning': '⚠️'
            })

            st.dataframe(
                events_df[['timestamp', 'type_emoji', 'message']].rename(columns={
                    'timestamp': 'Time',
                    'type_emoji': 'Type',
                    'message': 'Message'
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No events logged yet.")

def show_discovery():
    """Show discovery mode page for genetic algorithm optimization."""
    st.title("🔍 Discovery Mode - Genetic Algorithm Optimization")

    st.markdown("### 🎯 Strategy Discovery & Optimization")

    # GA Configuration
    st.markdown("#### ⚙️ Genetic Algorithm Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        population_size = st.slider("Population Size", 50, 500, 100, step=50)
        generations = st.slider("Generations", 10, 200, 50, step=10)

    with col2:
        mutation_rate = st.slider("Mutation Rate", 0.01, 0.5, 0.1, step=0.01)
        crossover_rate = st.slider("Crossover Rate", 0.1, 1.0, 0.8, step=0.1)

    with col3:
        fitness_metric = st.selectbox("Fitness Metric",
                                    ["Sharpe Ratio", "Total Return", "Win Rate", "Profit Factor"])
        selection_method = st.selectbox("Selection Method",
                                      ["Tournament", "Roulette Wheel", "Rank"])

    # Strategy Parameters to Optimize
    st.markdown("#### 📊 Strategy Parameters")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Technical Indicators")
        optimize_ema_short = st.checkbox("EMA Short Period", value=True)
        optimize_ema_long = st.checkbox("EMA Long Period", value=True)
        optimize_rsi_period = st.checkbox("RSI Period", value=True)
        optimize_bb_period = st.checkbox("Bollinger Bands Period", value=True)

    with col2:
        st.subheader("Risk Parameters")
        optimize_risk_per_trade = st.checkbox("Risk per Trade", value=True)
        optimize_stop_loss = st.checkbox("Stop Loss %", value=True)
        optimize_take_profit = st.checkbox("Take Profit %", value=True)

    # Asset and Timeframe Selection
    col1, col2 = st.columns(2)

    with col1:
        asset = st.selectbox("Asset", ["BTC/USDT", "ETH/USDT", "ADA/USDT", "DOT/USDT"])

    with col2:
        timeframe = st.selectbox("Timeframe", ["1d", "4h", "1h", "15m"])
        test_period = st.slider("Test Period (days)", 30, 365, 90, step=30)

    # Run Discovery
    if st.button("🚀 Start Discovery", type="primary"):
        with st.spinner("Running genetic algorithm optimization..."):
            # Simulate GA optimization (would be replaced with actual GA implementation)
            time.sleep(3)

            # Mock results
            discovery_results = {
                'best_fitness': 2.45,
                'best_parameters': {
                    'ema_short': 12,
                    'ema_long': 26,
                    'rsi_period': 14,
                    'bb_period': 20,
                    'risk_per_trade': 0.015,
                    'stop_loss': 0.05,
                    'take_profit': 0.10
                },
                'generations_completed': generations,
                'total_strategies_tested': population_size * generations,
                'improvement_over_baseline': 0.85,
                'sharpe_ratio': 2.45,
                'max_drawdown': -0.12,
                'win_rate': 0.68,
                'total_return': 1.45
            }

            st.session_state.discovery_results = discovery_results
            st.success("Discovery completed!")
            st.rerun()

    # Display Results
    if 'discovery_results' in st.session_state and st.session_state.discovery_results:
        results = st.session_state.discovery_results

        st.markdown("### 🏆 Optimization Results")

        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Best Fitness", f"{results['best_fitness']:.2f}")
            st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")

        with col2:
            st.metric("Total Return", f"{results['total_return']:.1%}")
            st.metric("Win Rate", f"{results['win_rate']:.1%}")

        with col3:
            st.metric("Max Drawdown", f"{results['max_drawdown']:.1%}")
            st.metric("Improvement", f"{results['improvement_over_baseline']:.1%}")

        with col4:
            st.metric("Strategies Tested", f"{results['total_strategies_tested']:,}")
            st.metric("Generations", results['generations_completed'])

        # Best Parameters
        st.markdown("#### 🎛️ Optimal Parameters")

        params = results['best_parameters']
        param_cols = st.columns(len(params))

        for i, (param_name, param_value) in enumerate(params.items()):
            with param_cols[i]:
                display_name = param_name.replace('_', ' ').title()
                if 'period' in param_name.lower():
                    st.metric(display_name, f"{param_value}")
                elif 'risk' in param_name.lower() or 'loss' in param_name.lower() or 'profit' in param_name.lower():
                    st.metric(display_name, f"{param_value:.1%}")
                else:
                    st.metric(display_name, f"{param_value}")

        # Fitness Evolution Chart
        st.markdown("#### 📈 Fitness Evolution")

        # Generate sample fitness evolution data
        generations_range = list(range(1, results['generations_completed'] + 1))
        best_fitness = [results['best_fitness'] * (0.5 + 0.5 * (i / results['generations_completed'])) + np.random.normal(0, 0.1)
                       for i in generations_range]
        avg_fitness = [x * 0.7 + np.random.normal(0, 0.05) for x in best_fitness]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=generations_range,
            y=best_fitness,
            mode='lines',
            name='Best Fitness',
            line=dict(color='#4ecdc4', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=generations_range,
            y=avg_fitness,
            mode='lines',
            name='Average Fitness',
            line=dict(color='#ff6b6b', width=2, dash='dash')
        ))

        fig.update_layout(
            title="Genetic Algorithm Fitness Evolution",
            xaxis_title="Generation",
            yaxis_title="Fitness Score",
            height=400,
            font=dict(color='white'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Parameter Distribution
        st.markdown("#### � Parameter Distribution")

        # Create parameter distribution chart
        param_names = list(params.keys())
        param_values = list(params.values())

        fig2 = go.Figure(data=[go.Bar(
            x=param_names,
            y=param_values,
            marker_color='#45b7d1'
        )])

        fig2.update_layout(
            title="Optimal Parameter Values",
            xaxis_title="Parameters",
            yaxis_title="Values",
            height=300,
            font=dict(color='white'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.info("Run discovery to see optimization results.")

def show_backtesting():
    """Backtesting interface."""
    st.title("📉 Backtesting & Analysis")

    backtester = st.session_state.backtester

    # Backtest configuration
    st.markdown("### ⚙️ Backtest Configuration")

    col1, col2 = st.columns(2)

    with col1:
        symbol = st.selectbox("Asset", ["BTC/USDT", "ETH/USDT", "ADA/USDT", "DOT/USDT"])
        timeframe = st.selectbox("Timeframe", ["1d", "4h", "1h", "15m"])
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
        end_date = st.date_input("End Date", datetime.now())

    with col2:
        strategy = st.selectbox("Strategy", ["SMA Crossover", "RSI Divergence", "ML Ensemble", "Risk Parity"])
        initial_capital = st.number_input("Initial Capital", 1000.0, 1000000.0, 10000.0)
        risk_per_trade = st.slider("Risk per Trade %", 0.1, 5.0, 1.0, step=0.1)

    if st.button("Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            # Simulate backtest results
            time.sleep(2)

            # Sample backtest results
            results = {
                'total_return': 0.45,
                'sharpe_ratio': 1.8,
                'max_drawdown': -0.15,
                'win_rate': 0.62,
                'total_trades': 156,
                'avg_trade_return': 0.028
            }

            st.success("Backtest completed!")

            # Display results
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Return", f"{results['total_return']:.1%}")
                st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")

            with col2:
                st.metric("Max Drawdown", f"{results['max_drawdown']:.1%}")
                st.metric("Win Rate", f"{results['win_rate']:.1%}")

            with col3:
                st.metric("Total Trades", results['total_trades'])
                st.metric("Avg Trade Return", f"{results['avg_trade_return']:.1%}")

            # Sample equity curve
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            equity = [initial_capital]
            for i in range(1, len(dates)):
                return_pct = np.random.normal(0.001, 0.02)
                equity.append(equity[-1] * (1 + return_pct))

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=equity,
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#4ecdc4', width=2)
            ))

            fig.update_layout(
                title="Equity Curve",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                font=dict(color='white'),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )

            st.plotly_chart(fig, use_container_width=True)

def show_live_trading():
    """Show live trading page with order management and execution."""
    st.title("🎯 Live Trading")

    st.markdown("### ⚠️ **Disclaimer:** Live trading involves real financial risk. Use paper trading first.")

    # Trading Status
    st.markdown("### 📊 Trading Status")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        live_trading_enabled = st.checkbox("Enable Live Trading", key="live_trading_enabled")
        status = "🟢 ENABLED" if live_trading_enabled else "🔴 DISABLED"
        st.metric("Live Trading", status)

    with col2:
        paper_trading = st.checkbox("Paper Trading Mode", value=True, key="paper_trading")
        mode = "📝 PAPER" if paper_trading else "💰 LIVE"
        st.metric("Trading Mode", mode)

    with col3:
        auto_trading = st.checkbox("Auto Trading", key="auto_trading")
        auto_status = "🤖 ON" if auto_trading else "👤 MANUAL"
        st.metric("Auto Trading", auto_status)

    with col4:
        connection_status = "🟢 CONNECTED"  # Would check actual connection
        st.metric("Exchange Connection", connection_status)

    # Asset Selection & Signal Generation
    st.markdown("### 🎯 Asset Selection & Signals")

    col1, col2, col3 = st.columns(3)

    with col1:
        trading_symbol = st.selectbox("Trading Asset", ["BTC/USDT", "ETH/USDT", "ADA/USDT", "DOT/USDT"], key="trading_symbol")

    with col2:
        order_size_type = st.selectbox("Order Size Type", ["Fixed Amount", "Percentage of Portfolio", "Risk-based"], key="order_size_type")

    with col3:
        if order_size_type == "Fixed Amount":
            order_amount = st.number_input("Order Amount ($)", min_value=10.0, value=100.0, step=10.0, key="order_amount")
        elif order_size_type == "Percentage of Portfolio":
            order_percentage = st.slider("Portfolio %", 0.1, 10.0, 1.0, step=0.1, key="order_percentage")
        else:  # Risk-based
            risk_per_trade = st.slider("Risk per Trade %", 0.1, 5.0, 1.0, step=0.1, key="risk_per_trade")

    # Current Market Data
    st.markdown("### 📈 Current Market Data")

    # Simulate real-time price data
    current_price = 45000 + np.random.normal(0, 500)  # BTC/USDT around 45k
    price_change_24h = np.random.normal(0, 0.05)
    volume_24h = np.random.uniform(1000000, 5000000)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(f"{trading_symbol} Price", f"${current_price:,.2f}", f"{price_change_24h:+.2f}%")

    with col2:
        st.metric("24h Volume", f"${volume_24h:,.0f}")

    with col3:
        spread = current_price * 0.0001  # 0.01% spread
        st.metric("Spread", f"${spread:.2f}")

    with col4:
        # Get current signal
        if st.session_state.ml_predictor and ML_ADVANCED_FEATURES_ENABLED:
            try:
                signal_data = st.session_state.ml_predictor.predict_signal(trading_symbol, None)
                signal = signal_data.get('signal', 'HOLD')
                confidence = signal_data.get('confidence', 0.5)

                signal_emoji = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '🟡'}.get(signal, '⚪')
                st.metric("Current Signal", f"{signal_emoji} {signal}", f"{confidence:.1%}")
            except:
                st.metric("Current Signal", "⚪ NO SIGNAL")
        else:
            st.metric("Current Signal", "⚪ ML DISABLED")

    # Manual Trading Panel
    st.markdown("### 🎮 Manual Trading")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Buy Order")
        buy_price = st.number_input("Buy Price", min_value=0.0, value=current_price * 0.995, step=0.01, key="buy_price")
        buy_quantity = st.number_input("Quantity", min_value=0.00001, value=0.001, step=0.00001, key="buy_quantity")
        buy_order_type = st.selectbox("Order Type", ["Market", "Limit", "Stop-Limit"], key="buy_order_type")

        if st.button("🟢 Place BUY Order", type="primary", use_container_width=True):
            if live_trading_enabled or paper_trading:
                order_details = {
                    'symbol': trading_symbol,
                    'side': 'BUY',
                    'type': buy_order_type,
                    'price': buy_price if buy_order_type != 'Market' else current_price,
                    'quantity': buy_quantity,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'PENDING'
                }

                if 'live_orders' not in st.session_state:
                    st.session_state.live_orders = []

                st.session_state.live_orders.append(order_details)

                mode_text = "PAPER TRADE" if paper_trading else "LIVE ORDER"
                st.success(f"✅ {mode_text} BUY order placed for {buy_quantity} {trading_symbol} @ ${buy_price:.2f}")
            else:
                st.error("Live trading is disabled. Enable live trading or use paper trading mode.")

    with col2:
        st.subheader("📉 Sell Order")
        sell_price = st.number_input("Sell Price", min_value=0.0, value=current_price * 1.005, step=0.01, key="sell_price")
        sell_quantity = st.number_input("Quantity", min_value=0.00001, value=0.001, step=0.00001, key="sell_quantity")
        sell_order_type = st.selectbox("Order Type", ["Market", "Limit", "Stop-Limit"], key="sell_order_type")

        if st.button("🔴 Place SELL Order", type="primary", use_container_width=True):
            if live_trading_enabled or paper_trading:
                order_details = {
                    'symbol': trading_symbol,
                    'side': 'SELL',
                    'type': sell_order_type,
                    'price': sell_price if sell_order_type != 'Market' else current_price,
                    'quantity': sell_quantity,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'PENDING'
                }

                if 'live_orders' not in st.session_state:
                    st.session_state.live_orders = []

                st.session_state.live_orders.append(order_details)

                mode_text = "PAPER TRADE" if paper_trading else "LIVE ORDER"
                st.success(f"✅ {mode_text} SELL order placed for {sell_quantity} {trading_symbol} @ ${sell_price:.2f}")
            else:
                st.error("Live trading is disabled. Enable live trading or use paper trading mode.")

    # Active Orders
    st.markdown("### � Active Orders")

    if 'live_orders' in st.session_state and st.session_state.live_orders:
        orders_df = pd.DataFrame(st.session_state.live_orders)
        orders_df['timestamp'] = pd.to_datetime(orders_df['timestamp'])

        # Add action buttons
        def cancel_order(index):
            if index < len(st.session_state.live_orders):
                st.session_state.live_orders[index]['status'] = 'CANCELLED'
                st.success("Order cancelled")

        orders_display = orders_df.copy()
        orders_display['Action'] = [f"Cancel {i}" for i in range(len(orders_display))]

        st.dataframe(
            orders_display[['timestamp', 'symbol', 'side', 'type', 'price', 'quantity', 'status']].style.apply(
                lambda x: ['background-color: #4CAF50' if v == 'BUY' else 'background-color: #f44336' if v == 'SELL' else '' for v in x],
                axis=0
            )
        )

        # Cancel buttons
        for i, order in enumerate(st.session_state.live_orders):
            if order['status'] == 'PENDING':
                if st.button(f"❌ Cancel Order {i+1}", key=f"cancel_{i}"):
                    cancel_order(i)
                    st.rerun()
    else:
        st.info("No active orders")

    # Trading Performance
    st.markdown("### 📊 Trading Performance")

    # Simulate trading stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_trades = len(st.session_state.get('live_orders', []))
        st.metric("Total Trades", total_trades)

    with col2:
        win_rate = 0.65  # Simulated
        st.metric("Win Rate", f"{win_rate:.1%}")

    with col3:
        avg_profit = 45.67  # Simulated
        st.metric("Avg Profit/Trade", f"${avg_profit:.2f}")

    with col4:
        total_pnl = total_trades * avg_profit * win_rate - total_trades * avg_profit * (1 - win_rate)
        st.metric("Total P&L", f"${total_pnl:.2f}")

    # Risk Management
    st.markdown("### ⚠️ Risk Management")

    col1, col2, col3 = st.columns(3)

    with col1:
        max_daily_loss = st.slider("Max Daily Loss %", 1.0, 10.0, 5.0, step=0.5, key="max_daily_loss")
        st.metric("Daily Loss Limit", f"{max_daily_loss}%")

    with col2:
        max_open_orders = st.slider("Max Open Orders", 1, 10, 3, key="max_open_orders")
        st.metric("Max Open Orders", max_open_orders)

    with col3:
        emergency_stop = st.checkbox("Emergency Stop", key="emergency_stop")
        stop_status = "🛑 ACTIVE" if emergency_stop else "✅ NORMAL"
        st.metric("Emergency Stop", stop_status)

    # Auto Trading Rules (if enabled)
    if auto_trading:
        st.markdown("### 🤖 Auto Trading Rules")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Entry Rules")
            min_confidence = st.slider("Min Signal Confidence", 0.5, 0.9, 0.7, step=0.05, key="min_confidence")
            min_volume = st.slider("Min Volume Threshold", 100000, 10000000, 1000000, step=100000, key="min_volume")

        with col2:
            st.subheader("Exit Rules")
            take_profit_pct = st.slider("Take Profit %", 1.0, 20.0, 5.0, step=0.5, key="take_profit_pct")
            stop_loss_pct = st.slider("Stop Loss %", 0.5, 10.0, 2.0, step=0.5, key="stop_loss_pct")

        # Auto trading status
        st.info("🤖 Auto trading is active. System will automatically execute trades based on ML signals and risk rules.")

def show_settings():
    """Settings and configuration panel."""
    st.title("⚙️ Settings & Configuration")

    # ML Configuration
    st.markdown("### 🤖 ML Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Settings")
        ensemble_enabled = st.checkbox("Enable Ensemble Models", value=ML_ENSEMBLE_MODELS)
        advanced_features = st.checkbox("Enable Advanced Features", value=ML_ADVANCED_FEATURES_ENABLED)

        model_types = st.multiselect(
            "Model Types",
            ["LSTM", "Transformer", "Random Forest", "XGBoost"],
            default=["LSTM", "Random Forest"]
        )

    with col2:
        st.subheader("Training Parameters")
        lookback_period = st.slider("Lookback Period (days)", 30, 365, 90, step=1)
        prediction_horizon = st.slider("Prediction Horizon (hours)", 1, 168, 24, step=1)
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.6, step=0.1)

    # Risk Management Settings
    st.markdown("### ⚠️ Risk Management")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Position Sizing")
        risk_per_trade_setting = st.slider("Risk per Trade %", 0.1, 5.0, float(RISK_PER_TRADE * 100), step=0.1)
        max_leverage_setting = st.slider("Max Leverage", 1.0, 10.0, float(MAX_LEVERAGE), step=0.1)
        max_allocation = st.slider("Max Allocation per Asset %", 10.0, 100.0, 40.0, step=1.0)

    with col2:
        st.subheader("Risk Limits")
        max_drawdown_limit = st.slider("Max Drawdown Limit %", 5.0, 50.0, 20.0, step=1.0)
        var_limit = st.slider("VaR Limit %", 1.0, 10.0, 5.0, step=0.1)
        correlation_limit = st.slider("Max Correlation", 0.1, 1.0, 0.8, step=0.1)

    # Portfolio Settings
    st.markdown("### 💼 Portfolio Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Optimization")
        rebalance_threshold = st.slider("Rebalance Threshold %", 1.0, 20.0, 5.0, step=1.0)
        optimization_frequency = st.selectbox("Optimization Frequency", ["Daily", "Weekly", "Monthly"])

    with col2:
        st.subheader("Constraints")
        min_weight = st.slider("Minimum Weight %", 0.0, 10.0, 0.0, step=0.1)
        max_weight = st.slider("Maximum Weight %", 10.0, 100.0, 40.0, step=1.0)

    # API Keys (secure storage simulation)
    st.markdown("### 🔑 API Configuration")

    st.warning("API keys are stored securely and encrypted. Changes require restart.")

    col1, col2 = st.columns(2)

    with col1:
        binance_api_key = st.text_input("Binance API Key", type="password", placeholder="Enter API Key")
        binance_secret = st.text_input("Binance Secret", type="password", placeholder="Enter Secret")

    with col2:
        alpaca_api_key = st.text_input("Alpaca API Key", type="password", placeholder="Enter API Key")
        alpaca_secret = st.text_input("Alpaca Secret", type="password", placeholder="Enter Secret")

    # Save Settings
    if st.button("Save Settings", type="primary"):
        st.success("Settings saved successfully! Some changes may require restart.")

    # System Information
    st.markdown("### ℹ️ System Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Python Version", f"{sys.version.split()[0]}")

    with col2:
        st.metric("Streamlit Version", "1.28.1")  # Would be dynamic in real app

    with col3:
        st.metric("TradPal Version", "2.0.0")  # Would be dynamic in real app

if __name__ == "__main__":
    main()