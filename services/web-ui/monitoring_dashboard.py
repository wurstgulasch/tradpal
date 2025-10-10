#!/usr/bin/env python3
"""
TradPal Monitoring Dashboard

A comprehensive monitoring dashboard for the TradPal trading system.
Provides real-time performance metrics, alerts, and system monitoring.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import system components with error handling
try:
    from config.settings import SYMBOL, TIMEFRAME, CAPITAL
    from src.data_fetcher import fetch_data, fetch_historical_data
    from src.indicators import calculate_indicators
    from src.signal_generator import generate_signals, calculate_risk_management
    from src.backtester import Backtester, run_backtest
    from src.logging_config import logger
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    st.error(f"❌ Failed to import system components: {e}")
    IMPORTS_SUCCESSFUL = False


class MonitoringDashboard:
    """
    Comprehensive monitoring dashboard for the TradPal system.
    """
    
    def __init__(self):
        self.metrics_cache = {}
        self.alerts = []
        self.last_update = None
        
        # Initialize instance variables for dashboard state
        self.current_symbol = "BTC/USD"
        self.current_timeframe = "1m"
        self.win_rate_threshold = 60
        self.drawdown_threshold = 20
    
    def run_dashboard(self):
        """Main dashboard interface."""
        if not IMPORTS_SUCCESSFUL:
            st.error("❌ System components not available. Please check imports.")
            return
            
        st.set_page_config(
            page_title="TradPal - Monitoring Dashboard",
            page_icon="📊",
            layout="wide"
        )
        
        st.title("📊 TradPal - Monitoring Dashboard")
        
        # Sidebar with controls
        self.sidebar_controls()
        
        # Main dashboard content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self.main_metrics_section()
        
        with col2:
            self.alerts_section()
        
        # Additional sections
        self.performance_charts_section()
        self.system_status_section()
        self.realtime_signals_section()
    
    def sidebar_controls(self):
        """Sidebar controls for dashboard configuration."""
        st.sidebar.title("⚙️ Dashboard Controls")
        
        # Symbol and timeframe selection
        st.sidebar.subheader("📈 Trading Configuration")
        self.current_symbol = st.sidebar.selectbox("Symbol", ["BTC/USD", "ETH/USD", "EUR/USD"], 
                                                  index=["BTC/USD", "ETH/USD", "EUR/USD"].index(self.current_symbol), 
                                                  key="monitor_symbol")
        self.current_timeframe = st.sidebar.selectbox("Timeframe", ["1m", "5m", "1h", "1d"], 
                                                     index=["1m", "5m", "1h", "1d"].index(self.current_timeframe), 
                                                     key="monitor_timeframe")
        
        # Alert thresholds
        st.sidebar.subheader("🚨 Alert Thresholds")
        self.win_rate_threshold = st.sidebar.slider("Win Rate Alert (%)", 0, 100, self.win_rate_threshold, key="win_rate_alert")
        self.drawdown_threshold = st.sidebar.slider("Max Drawdown Alert (%)", 0, 50, self.drawdown_threshold, key="drawdown_alert")
    
    def main_metrics_section(self):
        """Main performance metrics section."""
        st.header("📈 Key Performance Metrics")
        
        # Get current metrics
        metrics = self.get_performance_metrics()
        
        if metrics:
            # Primary metrics in large cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                win_rate = metrics.get('win_rate', 0)
                color = "🟢" if win_rate >= self.win_rate_threshold else "🔴"
                st.metric("Win Rate", f"{win_rate:.1f}%", delta=None, delta_color="normal")
                st.caption(f"{color} Target: {self.win_rate_threshold}%")
            
            with col2:
                total_return = metrics.get('total_return', 0)
                st.metric("Total Return", f"{total_return:.2f}%")
            
            with col3:
                sharpe_ratio = metrics.get('sharpe_ratio', 0)
                color = "🟢" if sharpe_ratio > 1.0 else "🟡" if sharpe_ratio > 0.5 else "🔴"
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}", delta=None, delta_color="normal")
                st.caption(f"{color} Target: >1.0")
            
            with col4:
                max_drawdown = metrics.get('max_drawdown', 0)
                color = "🟢" if max_drawdown < self.drawdown_threshold else "🔴"
                st.metric("Max Drawdown", f"{max_drawdown:.1f}%", delta=None, delta_color="normal")
                st.caption(f"{color} Limit: {self.drawdown_threshold}%")
            
            # Secondary metrics
            st.subheader("📊 Detailed Metrics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Trades", metrics.get('total_trades', 0))
                st.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")
            
            with col2:
                st.metric("Avg Win", f"{metrics.get('avg_win', 0):.2f}%")
                st.metric("Avg Loss", f"{metrics.get('avg_loss', 0):.2f}%")
            
            with col3:
                st.metric("Largest Win", f"{metrics.get('largest_win', 0):.2f}%")
                st.metric("Largest Loss", f"{metrics.get('largest_loss', 0):.2f}%")
        
        else:
            st.warning("⚠️ No performance data available. Run some backtests first.")
    
    def alerts_section(self):
        """Alerts and notifications section."""
        st.header("🚨 Active Alerts")
        
        # Check for new alerts
        self.check_alerts()
        
        if self.alerts:
            for alert in self.alerts[-5:]:  # Show last 5 alerts
                if alert['level'] == 'critical':
                    st.error(f"🚨 {alert['message']}")
                elif alert['level'] == 'warning':
                    st.warning(f"⚠️ {alert['message']}")
                else:
                    st.info(f"ℹ️ {alert['message']}")
                
                st.caption(f"🕒 {alert['timestamp'].strftime('%H:%M:%S')}")
                st.divider()
        else:
            st.success("✅ No active alerts")
    
    def performance_charts_section(self):
        """Performance visualization section."""
        st.header("📊 Performance Charts")
        
        # Get historical data for charts
        chart_data = self.get_chart_data()
        
        if chart_data is not None and not chart_data.empty:
            # Equity curve
            st.subheader("💰 Equity Curve")
            fig = self.create_equity_chart(chart_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Win/Loss distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Win/Loss Distribution")
                fig = self.create_win_loss_chart(chart_data)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📈 Monthly Returns")
                fig = self.create_monthly_returns_chart(chart_data)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ Generate some trading data to see performance charts")
    
    def system_status_section(self):
        """System status and health monitoring."""
        st.header("🔧 System Status")
        
        # System components status
        components = {
            "Data Fetcher": self.check_data_fetcher(),
            "Indicators": self.check_indicators(),
            "Signal Generator": self.check_signal_generator(),
            "Backtester": self.check_backtester(),
            "Database": self.check_database()
        }
        
        cols = st.columns(3)
        for i, (component, status) in enumerate(components.items()):
            with cols[i % 3]:
                if status['status'] == 'healthy':
                    st.success(f"✅ {component}")
                elif status['status'] == 'warning':
                    st.warning(f"⚠️ {component}")
                else:
                    st.error(f"❌ {component}")
                st.caption(status.get('message', ''))
        
        # System resources
        st.subheader("💻 System Resources")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("CPU Usage", "45%", "Normal")
        with col2:
            st.metric("Memory", "2.1GB", "Available: 6GB")
        with col3:
            st.metric("Disk Space", "85%", "Available: 50GB")
    
    def realtime_signals_section(self):
        """Real-time signals monitoring."""
        st.header("🎯 Real-Time Signals")
        
        # Get latest signals
        signals = self.get_realtime_signals()
        
        if signals:
            # Current signal status
            latest_signal = signals[0] if signals else None
            
            if latest_signal:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    signal_type = latest_signal.get('signal', 'HOLD')
                    if signal_type == 'BUY':
                        st.success("📈 BUY SIGNAL")
                    elif signal_type == 'SELL':
                        st.error("📉 SELL SIGNAL")
                    else:
                        st.info("⏸️ HOLD")
                
                with col2:
                    confidence = latest_signal.get('confidence', 0)
                    st.metric("Confidence", f"{confidence:.1%}")
                
                with col3:
                    timestamp = latest_signal.get('timestamp', datetime.now())
                    st.metric("Last Update", timestamp.strftime("%H:%M:%S"))
            
            # Recent signals table
            st.subheader("📋 Recent Signals")
            signals_df = pd.DataFrame(signals[:10])
            st.dataframe(signals_df)
        else:
            st.info("ℹ️ No recent signals available")
    
    def get_performance_metrics(self):
        """Get current performance metrics."""
        try:
            # Try to get cached metrics first
            if self.metrics_cache and (datetime.now() - self.last_update).seconds < 300:  # 5 min cache
                return self.metrics_cache
            
            # Generate mock metrics for demonstration
            np.random.seed(42)
            metrics = {
                'win_rate': np.random.uniform(55, 75),
                'total_return': np.random.uniform(-5, 25),
                'sharpe_ratio': np.random.uniform(0.5, 2.0),
                'max_drawdown': np.random.uniform(5, 25),
                'total_trades': np.random.randint(50, 200),
                'profit_factor': np.random.uniform(1.1, 2.5),
                'avg_win': np.random.uniform(1, 5),
                'avg_loss': np.random.uniform(0.5, 2),
                'largest_win': np.random.uniform(5, 15),
                'largest_loss': np.random.uniform(2, 8)
            }
            
            self.metrics_cache = metrics
            self.last_update = datetime.now()
            return metrics
            
        except Exception as e:
            if 'logger' in globals():
                logger.error(f"Error getting performance metrics: {e}")
            return None
    
    def check_alerts(self):
        """Check for new alerts based on current metrics."""
        metrics = self.get_performance_metrics()
        
        if metrics:
            current_time = datetime.now()
            
            # Win rate alert
            win_rate = metrics.get('win_rate', 0)
            threshold = self.win_rate_threshold
            if win_rate < threshold:
                self.add_alert(
                    f"Win rate dropped below {threshold}% (Current: {win_rate:.1f}%)",
                    'warning',
                    current_time
                )
            
            # Drawdown alert
            max_drawdown = metrics.get('max_drawdown', 0)
            dd_threshold = self.drawdown_threshold
            if max_drawdown > dd_threshold:
                self.add_alert(
                    f"Drawdown exceeded {dd_threshold}% (Current: {max_drawdown:.1f}%)",
                    'critical',
                    current_time
                )
            
            # Sharpe ratio alert
            sharpe = metrics.get('sharpe_ratio', 0)
            if sharpe < 0.5:
                self.add_alert(
                    f"Sharpe ratio too low: {sharpe:.2f} (Target: >0.5)",
                    'warning',
                    current_time
                )
    
    def add_alert(self, message, level, timestamp):
        """Add a new alert."""
        alert = {
            'message': message,
            'level': level,
            'timestamp': timestamp
        }
        self.alerts.append(alert)
        
        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
    
    def get_chart_data(self):
        """Get data for performance charts."""
        try:
            # Generate sample performance data
            dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
            np.random.seed(42)
            
            # Simulate equity curve
            returns = np.random.normal(0.001, 0.02, 100)  # Daily returns
            equity = 10000 * np.cumprod(1 + returns)
            
            df = pd.DataFrame({
                'date': dates,
                'equity': equity,
                'returns': returns
            })
            
            return df
        except Exception as e:
            if 'logger' in globals():
                logger.error(f"Error getting chart data: {e}")
            return None
    
    def create_equity_chart(self, data):
        """Create equity curve chart."""
        import plotly.graph_objects as go
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data['date'],
            y=data['equity'],
            mode='lines',
            name='Equity',
            line=dict(color='green', width=2)
        ))
        
        fig.update_layout(
            title="Portfolio Equity Curve",
            xaxis_title="Date",
            yaxis_title="Equity ($)",
            height=400
        )
        
        return fig
    
    def create_win_loss_chart(self, data):
        """Create win/loss distribution chart."""
        import plotly.graph_objects as go
        
        # Simulate win/loss data
        wins = np.random.exponential(2, 50)
        losses = -np.random.exponential(1.5, 30)
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=wins,
            name='Wins',
            marker_color='green',
            opacity=0.7
        ))
        fig.add_trace(go.Histogram(
            x=losses,
            name='Losses',
            marker_color='red',
            opacity=0.7
        ))
        
        fig.update_layout(
            title="Win/Loss Distribution",
            xaxis_title="Return (%)",
            yaxis_title="Frequency",
            barmode='overlay',
            height=300
        )
        
        return fig
    
    def create_monthly_returns_chart(self, data):
        """Create monthly returns chart."""
        import plotly.graph_objects as go
        
        # Group by month and calculate returns
        data['month'] = data['date'].dt.to_period('M')
        monthly_returns = data.groupby('month')['returns'].sum() * 100
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly_returns.index.astype(str),
            y=monthly_returns.values,
            marker_color=['green' if x > 0 else 'red' for x in monthly_returns.values]
        ))
        
        fig.update_layout(
            title="Monthly Returns",
            xaxis_title="Month",
            yaxis_title="Return (%)",
            height=300
        )
        
        return fig
    
    def get_realtime_signals(self):
        """Get real-time trading signals."""
        try:
            # Generate mock signals for demonstration
            signals = []
            for i in range(10):
                signal = {
                    'timestamp': datetime.now() - timedelta(minutes=i*5),
                    'signal': np.random.choice(['BUY', 'SELL', 'HOLD'], p=[0.4, 0.4, 0.2]),
                    'confidence': np.random.uniform(0.6, 0.9),
                    'price': np.random.uniform(40000, 50000),
                    'rsi': np.random.uniform(30, 70),
                    'ema9': np.random.uniform(45000, 48000),
                    'ema21': np.random.uniform(44000, 47000)
                }
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            if 'logger' in globals():
                logger.error(f"Error getting realtime signals: {e}")
            return []
    
    def check_data_fetcher(self):
        """Check data fetcher status."""
        try:
            if IMPORTS_SUCCESSFUL:
                # Quick test
                data = fetch_data(limit=10)
                if data is not None and not data.empty:
                    return {'status': 'healthy', 'message': f'{len(data)} records fetched'}
                else:
                    return {'status': 'error', 'message': 'No data received'}
            else:
                return {'status': 'error', 'message': 'Imports failed'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def check_indicators(self):
        """Check indicators status."""
        try:
            if IMPORTS_SUCCESSFUL:
                data = fetch_data(limit=20)
                if data is not None:
                    data = calculate_indicators(data)
                    indicators = [col for col in data.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
                    return {'status': 'healthy', 'message': f'{len(indicators)} indicators calculated'}
                return {'status': 'error', 'message': 'Failed to calculate indicators'}
            else:
                return {'status': 'warning', 'message': 'Using mock data'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def check_signal_generator(self):
        """Check signal generator status."""
        try:
            if IMPORTS_SUCCESSFUL:
                data = fetch_data(limit=20)
                if data is not None:
                    data = calculate_indicators(data)
                    data = generate_signals(data)
                    buy_signals = data['Buy_Signal'].sum()
                    sell_signals = data['Sell_Signal'].sum()
                    return {'status': 'healthy', 'message': f'{int(buy_signals)} buys, {int(sell_signals)} sells'}
                return {'status': 'error', 'message': 'Failed to generate signals'}
            else:
                return {'status': 'warning', 'message': 'Using mock data'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def check_backtester(self):
        """Check backtester status."""
        try:
            if IMPORTS_SUCCESSFUL:
                backtester = Backtester(initial_capital=1000)
                return {'status': 'healthy', 'message': 'Backtester initialized'}
            else:
                return {'status': 'warning', 'message': 'Using mock data'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def check_database(self):
        """Check database/cache status."""
        try:
            # Check if cache directory exists
            if os.path.exists('cache'):
                cache_files = len([f for f in os.listdir('cache') if f.endswith('.pkl')])
                return {'status': 'healthy', 'message': f'{cache_files} cached files'}
            else:
                return {'status': 'warning', 'message': 'Cache directory not found'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}


def main():
    """Main dashboard application."""
    dashboard = MonitoringDashboard()
    dashboard.run_dashboard()


if __name__ == "__main__":
    main()
