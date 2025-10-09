#!/usr/bin/env python3
"""
TradPal Indicator Web UI - Main Application

Comprehensive web interface with authentication, strategy builder, and interactive controls.
Uses Streamlit for UI, Plotly for charts, and Flask-Login for authentication.
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import authentication module
try:
    from auth import check_authentication, login_page
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False

# Import UI components
try:
    from strategy_builder import StrategyBuilderUI
    from interactive_controls import InteractiveControlsUI
    from live_charts import LiveChartsUI
    COMPONENTS_AVAILABLE = True
except ImportError:
    COMPONENTS_AVAILABLE = False

# Import monitoring dashboard
try:
    from monitoring_dashboard import MonitoringDashboard
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="TradPal Indicator - Web UI",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Session state initialization
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'page' not in st.session_state:
        st.session_state.page = 'Dashboard'
    
    # Check authentication
    if AUTH_AVAILABLE and not st.session_state.authenticated:
        login_page()
        return
    
    # Main UI
    st.title("📊 TradPal Indicator - Trading System")
    
    # Sidebar navigation
    with st.sidebar:
        st.title("🧭 Navigation")
        
        # Show username if authenticated
        if st.session_state.get('username'):
            st.success(f"👤 Logged in as: {st.session_state.username}")
            if st.button("🚪 Logout"):
                st.session_state.authenticated = False
                st.session_state.username = None
                st.rerun()
        
        st.divider()
        
        # Navigation menu
        pages = {
            "📊 Dashboard": "dashboard",
            "🎨 Strategy Builder": "strategy_builder",
            "⚙️ Interactive Controls": "controls",
            "📈 Live Charts": "charts",
            "🔧 System Monitor": "monitor"
        }
        
        for page_name, page_id in pages.items():
            if st.button(page_name, key=f"nav_{page_id}", use_container_width=True):
                st.session_state.page = page_id
        
        st.divider()
        
        # Quick settings
        st.subheader("⚡ Quick Settings")
        st.selectbox("Theme", ["Dark", "Light"], key="theme")
        st.selectbox("Refresh Rate", ["5s", "10s", "30s", "1m"], index=2, key="refresh_rate")
    
    # Route to selected page
    current_page = st.session_state.page
    
    if current_page == "dashboard":
        show_dashboard()
    elif current_page == "strategy_builder":
        show_strategy_builder()
    elif current_page == "controls":
        show_interactive_controls()
    elif current_page == "charts":
        show_live_charts()
    elif current_page == "monitor":
        show_monitoring()
    else:
        show_dashboard()


def show_dashboard():
    """Show main dashboard."""
    st.header("📊 Dashboard Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Strategies", "3", "+1")
    with col2:
        st.metric("Total Return", "23.4%", "+2.3%")
    with col3:
        st.metric("Win Rate", "68.5%", "+1.2%")
    with col4:
        st.metric("Active Signals", "2", "0")
    
    st.divider()
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("▶️ Run Backtest", use_container_width=True):
            st.info("Starting backtest...")
    with col2:
        if st.button("📊 Generate Signals", use_container_width=True):
            st.info("Generating signals...")
    with col3:
        if st.button("🔄 Optimize Strategy", use_container_width=True):
            st.info("Running optimization...")
    
    st.divider()
    
    # Recent activity
    st.subheader("📋 Recent Activity")
    st.info("Dashboard is ready. Use the navigation menu to access different features.")


def show_strategy_builder():
    """Show strategy builder interface."""
    if COMPONENTS_AVAILABLE:
        builder = StrategyBuilderUI()
        builder.render()
    else:
        st.error("❌ Strategy Builder component not available. Please check installation.")


def show_interactive_controls():
    """Show interactive controls interface."""
    if COMPONENTS_AVAILABLE:
        controls = InteractiveControlsUI()
        controls.render()
    else:
        st.error("❌ Interactive Controls component not available. Please check installation.")


def show_live_charts():
    """Show live charts interface."""
    if COMPONENTS_AVAILABLE:
        charts = LiveChartsUI()
        charts.render()
    else:
        st.error("❌ Live Charts component not available. Please check installation.")


def show_monitoring():
    """Show monitoring dashboard."""
    if MONITORING_AVAILABLE:
        dashboard = MonitoringDashboard()
        dashboard.run_dashboard()
    else:
        st.error("❌ Monitoring Dashboard not available. Please check installation.")


if __name__ == "__main__":
    main()
