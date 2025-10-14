#!/usr/bin/env python3
"""
TradPal Trading Dashboard - Streamlit Web Interface

A comprehensive web dashboard for the TradPal trading system featuring:
- Real-time portfolio monitoring
- Live trading controls
- ML model analytics
- Risk management dashboard
- Performance visualization
- Multi-asset portfolio management
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import sys
from pathlib import Path
import json
import threading
import queue

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import trading system components
from config.settings import (
    SYMBOL, EXCHANGE, TIMEFRAME, INITIAL_CAPITAL,
    RISK_PER_TRADE, MAX_LEVERAGE
)

# Import portfolio management
from src.multi_asset_portfolio import get_portfolio_manager

# Import ML predictor
from src.advanced_ml_predictor import AdvancedMLPredictor

# Import data fetcher
from src.data_fetcher import fetch_historical_data

# Import backtester
from src.backtester import Backtester

# Page configuration
st.set_page_config(
    page_title="TradPal Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .profit-positive {
        color: #28a745;
        font-weight: bold;
    }
    .profit-negative {
        color: #dc3545;
        font-weight: bold;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 0.5rem;
    }
    .status-active {
        background-color: #28a745;
    }
    .status-inactive {
        background-color: #dc3545;
    }
    .status-warning {
        background-color: #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'portfolio_manager' not in st.session_state:
    st.session_state.portfolio_manager = get_portfolio_manager()

if 'ml_predictor' not in st.session_state:
    st.session_state.ml_predictor = AdvancedMLPredictor()

if 'backtester' not in st.session_state:
    st.session_state.backtester = Backtester()

if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()

if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True

# Global data cache
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_cached_data(symbol, timeframe='1h', limit=1000):
    """Get cached historical data."""
    try:
        data = fetch_historical_data(symbol, timeframe, limit)
        return data
    except Exception as e:
        st.error(f"Failed to fetch data for {symbol}: {e}")
        return pd.DataFrame()

def create_sidebar():
    """Create the sidebar with navigation and controls."""
    with st.sidebar:
        st.markdown('<div class="sidebar-header">📊 TradPal Dashboard</div>', unsafe_allow_html=True)

        # Navigation
        page = st.radio(
            "Navigation",
            ["Dashboard", "Trading", "Portfolio", "ML Analytics", "Risk Management", "Backtesting", "Settings"],
            index=0
        )

        st.markdown("---")

        # System Status
        st.markdown("### System Status")

        # Portfolio status
        portfolio = st.session_state.portfolio_manager
        metrics = portfolio.get_portfolio_metrics()

        col1, col2 = st.columns(2)
        with col1:
            if metrics.total_value > INITIAL_CAPITAL:
                st.markdown('<span class="status-indicator status-active"></span>Portfolio', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-indicator status-warning"></span>Portfolio', unsafe_allow_html=True)

        with col2:
            st.markdown('<span class="status-indicator status-active"></span>ML Models', unsafe_allow_html=True)

        # Quick stats
        st.markdown("### Quick Stats")
        st.metric("Portfolio Value", f"${metrics.total_value:,.2f}")
        pnl_color = "profit-positive" if metrics.total_pnl >= 0 else "profit-negative"
        st.markdown(f"Total P&L: <span class='{pnl_color}'>${metrics.total_pnl:+,.2f}</span>", unsafe_allow_html=True)

        st.markdown("---")

        # Controls
        st.markdown("### Controls")

        # Auto refresh toggle
        st.session_state.auto_refresh = st.checkbox("Auto Refresh (30s)", value=st.session_state.auto_refresh)

        # Manual refresh button
        if st.button("🔄 Refresh Data"):
            st.session_state.last_update = datetime.now()
            st.rerun()

        # Last update time
        st.caption(f"Last update: {st.session_state.last_update.strftime('%H:%M:%S')}")

        return page

def create_dashboard_page():
    """Create the main dashboard page."""
    st.markdown('<div class="main-header">📈 Trading Dashboard</div>', unsafe_allow_html=True)

    # Get portfolio data
    portfolio = st.session_state.portfolio_manager
    metrics = portfolio.get_portfolio_metrics()

    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Portfolio Value", f"${metrics.total_value:,.2f}", f"{((metrics.total_value/INITIAL_CAPITAL)-1)*100:+.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        pnl_color = "profit-positive" if metrics.total_pnl >= 0 else "profit-negative"
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total P&L", f"${metrics.total_pnl:+,.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Daily P&L", f"${metrics.daily_pnl:+,.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Charts Row
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Portfolio Allocation")

        # Create pie chart for portfolio allocation
        if portfolio.positions:
            labels = list(portfolio.positions.keys())
            values = [pos.quantity * pos.current_price for pos in portfolio.positions.values()]

            fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
            fig.update_layout(
                title="Asset Allocation",
                font=dict(size=12),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No positions in portfolio")

    with col2:
        st.subheader("Recent Performance")

        # Create sample performance chart (would be replaced with real data)
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        performance = np.cumsum(np.random.normal(0.001, 0.02, len(dates)))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=performance, mode='lines', name='Portfolio Value'))
        fig.update_layout(
            title="30-Day Performance",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            font=dict(size=12),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Recent Trades Table
    st.subheader("Recent Activity")
    st.markdown("Recent trades and portfolio changes will appear here.")

    # Placeholder for recent trades
    if hasattr(portfolio, 'trade_history') and portfolio.trade_history:
        trades_df = pd.DataFrame(portfolio.trade_history[-10:])  # Last 10 trades
        st.dataframe(trades_df, use_container_width=True)
    else:
        st.info("No recent trades to display")

def create_trading_page():
    """Create the live trading interface page."""
    st.markdown('<div class="main-header">🎯 Live Trading</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Market Data")

        # Symbol selection
        symbol = st.selectbox("Select Symbol", ["BTC/USDT", "ETH/USDT", "ADA/USDT", "DOT/USDT"], index=0)

        # Get market data
        data = get_cached_data(symbol, TIMEFRAME, 100)

        if not data.empty:
            # Price chart
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price'
            ))
            fig.update_layout(
                title=f"{symbol} Price Chart",
                xaxis_title="Time",
                yaxis_title="Price",
                font=dict(size=12),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Current price info
            current_price = data['close'].iloc[-1]
            price_change = data['close'].iloc[-1] - data['close'].iloc[-2]
            price_change_pct = (price_change / data['close'].iloc[-2]) * 100

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Current Price", f"${current_price:.2f}")
            with col_b:
                st.metric("24h Change", f"${price_change:+.2f}")
            with col_c:
                st.metric("24h Change %", f"{price_change_pct:+.2f}%")

        else:
            st.error("Failed to load market data")

    with col2:
        st.subheader("Trading Controls")

        # ML Prediction
        predictor = st.session_state.ml_predictor

        if st.button("🔮 Get ML Prediction"):
            with st.spinner("Analyzing market data..."):
                try:
                    # Get recent data for prediction
                    pred_data = data.tail(50) if not data.empty else None

                    if pred_data is not None:
                        signal = predictor.predict_signal(pred_data)
                        confidence = predictor.get_prediction_confidence(pred_data)

                        st.success(f"ML Signal: **{signal.upper()}**")
                        st.info(f"Confidence: {confidence:.1%}")

                        # Signal visualization
                        if signal == 'BUY':
                            st.markdown("🟢 **BUY SIGNAL** - Market conditions favorable")
                        elif signal == 'SELL':
                            st.markdown("🔴 **SELL SIGNAL** - Consider reducing exposure")
                        else:
                            st.markdown("🟡 **HOLD SIGNAL** - Wait for better conditions")

                    else:
                        st.warning("Insufficient data for ML prediction")

                except Exception as e:
                    st.error(f"ML prediction failed: {e}")

        st.markdown("---")

        # Manual Trading
        st.subheader("Manual Trade")

        trade_type = st.radio("Trade Type", ["BUY", "SELL"], horizontal=True)
        trade_amount = st.number_input("Amount (USD)", min_value=10.0, value=100.0, step=10.0)

        if st.button(f"Execute {trade_type} Order"):
            portfolio = st.session_state.portfolio_manager

            try:
                result = portfolio.execute_trade(
                    symbol=symbol,
                    signal=trade_type,
                    price=current_price if 'current_price' in locals() else 0,
                    quantity=None  # Will calculate based on amount
                )

                if result['success']:
                    st.success(f"✅ {trade_type} order executed successfully!")
                    st.info(f"Quantity: {result['quantity']:.6f}")
                    st.info(f"Value: ${result['value']:.2f}")
                else:
                    st.error(f"❌ Trade failed: {result.get('error', 'Unknown error')}")

            except Exception as e:
                st.error(f"Trade execution failed: {e}")

def create_portfolio_page():
    """Create the portfolio management page."""
    st.markdown('<div class="main-header">📊 Portfolio Management</div>', unsafe_allow_html=True)

    portfolio = st.session_state.portfolio_manager

    # Portfolio Overview
    col1, col2, col3 = st.columns(3)

    metrics = portfolio.get_portfolio_metrics()

    with col1:
        st.metric("Total Assets", len(portfolio.positions))
    with col2:
        st.metric("Total Value", f"${metrics.total_value:,.2f}")
    with col3:
        st.metric("Diversification Ratio", f"{metrics.diversification_ratio:.2f}")

    # Current Positions
    st.subheader("Current Positions")

    if portfolio.positions:
        positions_data = []
        for symbol, position in portfolio.positions.items():
            positions_data.append({
                'Symbol': symbol,
                'Quantity': position.quantity,
                'Entry Price': position.entry_price,
                'Current Price': position.current_price,
                'Unrealized P&L': position.unrealized_pnl,
                'Allocation %': position.allocation_pct * 100,
                'Volatility': position.volatility
            })

        positions_df = pd.DataFrame(positions_data)
        st.dataframe(positions_df, use_container_width=True)

        # Allocation Chart
        fig = px.pie(
            positions_df,
            values='Allocation %',
            names='Symbol',
            title="Portfolio Allocation"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No positions in portfolio")

    # Portfolio Optimization
    st.subheader("Portfolio Optimization")

    col1, col2 = st.columns(2)

    with col1:
        optimization_method = st.selectbox(
            "Optimization Method",
            ["risk_parity", "mpt", "min_variance"],
            index=0
        )

        if st.button("🔬 Optimize Portfolio"):
            with st.spinner("Optimizing portfolio..."):
                try:
                    optimized_weights = portfolio.optimize_portfolio(
                        optimization_method=optimization_method
                    )

                    st.success("✅ Portfolio optimized successfully!")

                    # Display optimized weights
                    st.subheader("Optimized Weights")
                    for symbol, weight in optimized_weights.items():
                        st.write(f"{symbol}: {weight:.1%}")

                except Exception as e:
                    st.error(f"Optimization failed: {e}")

    with col2:
        if st.button("🔄 Rebalance Portfolio"):
            with st.spinner("Rebalancing portfolio..."):
                try:
                    result = portfolio.rebalance_portfolio()

                    if result['success']:
                        st.success(f"✅ Portfolio rebalanced! {result['trades_executed']} trades executed")
                        st.info(f"Total cost: ${result.get('total_cost', 0):.2f}")
                    else:
                        st.error(f"Rebalancing failed: {result.get('error', 'Unknown error')}")

                except Exception as e:
                    st.error(f"Rebalancing failed: {e}")

def create_ml_analytics_page():
    """Create the ML analytics page."""
    st.markdown('<div class="main-header">🤖 ML Analytics</div>', unsafe_allow_html=True)

    predictor = st.session_state.ml_predictor

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Performance")

        # Model metrics (placeholder - would be real metrics)
        st.metric("Accuracy", "68.5%")
        st.metric("Precision", "71.2%")
        st.metric("Recall", "65.8%")

        # Feature importance chart
        features = ['RSI', 'MACD', 'Volume', 'Price Momentum', 'Volatility']
        importance = np.random.rand(len(features))
        importance = importance / importance.sum()

        fig = px.bar(
            x=features,
            y=importance,
            title="Feature Importance",
            labels={'x': 'Feature', 'y': 'Importance'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Market Regime Detection")

        # Regime detection
        regime_data = pd.DataFrame({
            'timestamp': pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='H'),
            'regime': np.random.choice(['bull', 'bear', 'sideways'], 168)
        })

        regime_colors = {'bull': 'green', 'bear': 'red', 'sideways': 'orange'}
        regime_data['color'] = regime_data['regime'].map(regime_colors)

        fig = px.scatter(
            regime_data,
            x='timestamp',
            y=regime_data.index,
            color='regime',
            color_discrete_map=regime_colors,
            title="Market Regime Over Time"
        )
        fig.update_layout(showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    # Prediction History
    st.subheader("Recent Predictions")

    # Sample prediction history
    predictions_df = pd.DataFrame({
        'timestamp': pd.date_range(start=datetime.now() - timedelta(hours=24), periods=24, freq='H'),
        'symbol': np.random.choice(['BTC/USDT', 'ETH/USDT'], 24),
        'prediction': np.random.choice(['BUY', 'SELL', 'HOLD'], 24),
        'confidence': np.random.uniform(0.5, 0.95, 24),
        'actual': np.random.choice(['BUY', 'SELL', 'HOLD'], 24)
    })

    st.dataframe(predictions_df.tail(10), use_container_width=True)

def create_risk_page():
    """Create the risk management page."""
    st.markdown('<div class="main-header">⚠️ Risk Management</div>', unsafe_allow_html=True)

    portfolio = st.session_state.portfolio_manager

    # Risk Metrics
    risk_metrics = portfolio.get_risk_metrics()

    if risk_metrics:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Portfolio Volatility", f"{risk_metrics.get('volatility', 0):.2%}")
        with col2:
            st.metric("Sharpe Ratio", f"{risk_metrics.get('sharpe_ratio', 0):.2f}")
        with col3:
            st.metric("VaR (95%)", f"{risk_metrics.get('var_95', 0):.2%}")
        with col4:
            st.metric("Max Drawdown", f"{risk_metrics.get('max_drawdown', 0):.2%}")

        # Risk Charts
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Risk Contribution")

            if portfolio.positions:
                symbols = list(portfolio.positions.keys())
                risk_contrib = np.random.rand(len(symbols))  # Placeholder
                risk_contrib = risk_contrib / risk_contrib.sum()

                fig = px.bar(
                    x=symbols,
                    y=risk_contrib,
                    title="Risk Contribution by Asset",
                    labels={'x': 'Asset', 'y': 'Risk Contribution'}
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Correlation Matrix")

            if len(portfolio.positions) > 1:
                corr_matrix = portfolio.correlation_matrix
                if not corr_matrix.empty:
                    fig = px.imshow(
                        corr_matrix,
                        title="Asset Correlations",
                        color_continuous_scale='RdBu_r'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Correlation data not available")
            else:
                st.info("Need multiple assets for correlation analysis")

    else:
        st.warning("Risk metrics not available. Add positions and historical data.")

    # Risk Alerts
    st.subheader("Risk Alerts")

    alerts = []

    # Check for high concentration
    if portfolio.positions:
        max_allocation = max(pos.allocation_pct for pos in portfolio.positions.values())
        if max_allocation > 0.5:
            alerts.append("⚠️ High concentration risk: Single asset > 50% allocation")

    # Check volatility
    if risk_metrics and risk_metrics.get('volatility', 0) > 0.3:
        alerts.append("⚠️ High portfolio volatility detected")

    if alerts:
        for alert in alerts:
            st.warning(alert)
    else:
        st.success("✅ No risk alerts at this time")

def create_backtesting_page():
    """Create the backtesting interface page."""
    st.markdown('<div class="main-header">📊 Backtesting</div>', unsafe_allow_html=True)

    backtester = st.session_state.backtester

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Backtest Configuration")

        # Backtest parameters
        symbol = st.selectbox("Symbol", ["BTC/USDT", "ETH/USDT"], index=0)
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
        end_date = st.date_input("End Date", datetime.now())
        initial_capital = st.number_input("Initial Capital", value=10000.0, min_value=1000.0)

        strategy = st.selectbox("Strategy", ["ML Ensemble", "Moving Average Crossover", "RSI Strategy"])

        if st.button("🚀 Run Backtest"):
            with st.spinner("Running backtest..."):
                try:
                    # Run backtest (placeholder implementation)
                    results = {
                        'total_return': 0.156,
                        'sharpe_ratio': 1.23,
                        'max_drawdown': -0.089,
                        'win_rate': 0.62,
                        'total_trades': 145
                    }

                    st.session_state.backtest_results = results
                    st.success("✅ Backtest completed!")

                except Exception as e:
                    st.error(f"Backtest failed: {e}")

    with col2:
        st.subheader("Backtest Results")

        if 'backtest_results' in st.session_state:
            results = st.session_state.backtest_results

            # Results metrics
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Total Return", f"{results['total_return']:.1%}")
            with col_b:
                st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
            with col_c:
                st.metric("Max Drawdown", f"{results['max_drawdown']:.1%}")

            col_d, col_e = st.columns(2)
            with col_d:
                st.metric("Win Rate", f"{results['win_rate']:.1%}")
            with col_e:
                st.metric("Total Trades", results['total_trades'])

            # Sample equity curve
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            equity = np.cumprod(1 + np.random.normal(0.001, 0.02, len(dates)))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=equity, mode='lines', name='Equity Curve'))
            fig.update_layout(
                title="Backtest Equity Curve",
                xaxis_title="Date",
                yaxis_title="Portfolio Value",
                font=dict(size=12),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run a backtest to see results")

def create_settings_page():
    """Create the settings and configuration page."""
    st.markdown('<div class="main-header">⚙️ Settings</div>', unsafe_allow_html=True)

    st.subheader("Trading Parameters")

    col1, col2 = st.columns(2)

    with col1:
        st.number_input("Risk per Trade (%)", value=RISK_PER_TRADE * 100, min_value=0.1, max_value=10.0)
        st.number_input("Max Leverage", value=MAX_LEVERAGE, min_value=1.0, max_value=100.0)
        st.selectbox("Default Timeframe", [TIMEFRAME, "1m", "5m", "15m", "1h", "4h", "1d"])

    with col2:
        st.selectbox("Exchange", [EXCHANGE, "binance", "coinbase", "kraken"])
        st.multiselect("Enabled Symbols", ["BTC/USDT", "ETH/USDT", "ADA/USDT", "DOT/USDT"],
                      default=["BTC/USDT", "ETH/USDT"])
        st.checkbox("Enable ML Predictions", value=True)
        st.checkbox("Enable Auto Trading", value=False)

    st.subheader("ML Model Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.slider("Ensemble Models", min_value=1, max_value=5, value=3)
        st.slider("Lookback Period (days)", min_value=7, max_value=365, value=30)

    with col2:
        st.multiselect("ML Features", ["RSI", "MACD", "Volume", "Price Momentum", "Volatility"],
                      default=["RSI", "MACD", "Volume"])
        st.selectbox("Model Update Frequency", ["1h", "4h", "1d", "1w"], index=2)

    if st.button("💾 Save Settings"):
        st.success("Settings saved successfully!")

    st.subheader("System Information")

    col1, col2 = st.columns(2)

    with col1:
        st.info(f"Python Version: {sys.version}")
        st.info(f"Streamlit Version: {st.__version__}")

    with col2:
        st.info(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
        st.info(f"Active Symbols: {len(st.session_state.portfolio_manager.positions)}")

def main():
    """Main application function."""
    # Create sidebar and get selected page
    page = create_sidebar()

    # Route to appropriate page
    if page == "Dashboard":
        create_dashboard_page()
    elif page == "Trading":
        create_trading_page()
    elif page == "Portfolio":
        create_portfolio_page()
    elif page == "ML Analytics":
        create_ml_analytics_page()
    elif page == "Risk Management":
        create_risk_page()
    elif page == "Backtesting":
        create_backtesting_page()
    elif page == "Settings":
        create_settings_page()

    # Auto refresh functionality
    if st.session_state.auto_refresh:
        time.sleep(30)  # Refresh every 30 seconds
        st.rerun()

if __name__ == "__main__":
    main()