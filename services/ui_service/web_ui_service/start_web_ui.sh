#!/bin/bash
# Quick start script for TradPal Web UI

echo "=================================="
echo "TradPal - Web UI Launcher"
echo "=================================="
echo ""

# Check if we're in the correct directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found!"
    echo "Please run this script from the services/web-ui directory:"
    echo "  cd services/web-ui"
    echo "  ./start_web_ui.sh"
    exit 1
fi

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "⚠️  Streamlit is not installed."
    echo "Installing required dependencies..."
    pip install streamlit plotly flask flask-login werkzeug -q
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies."
        echo "Please install manually: pip install streamlit plotly flask flask-login werkzeug"
        exit 1
    fi
    echo "✅ Dependencies installed successfully!"
fi

echo "🚀 Starting TradPal Web UI..."
echo ""
echo "Access the Web UI at: http://localhost:8501"
echo "Default credentials:"
echo "  Username: admin"
echo "  Password: admin123"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "=================================="
echo ""

# Start streamlit
streamlit run app.py --server.headless true
