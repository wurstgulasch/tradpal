#!/usr/bin/env python3
"""
TradPal Indicator Tutorials Launcher

Starts the interactive Streamlit tutorials for the TradPal Indicator system.
"""

import subprocess
import sys
import os
from pathlib import Path


def check_streamlit_installation():
    """Check if Streamlit is installed."""
    try:
        import streamlit
        return True
    except ImportError:
        print("❌ Streamlit is not installed.")
        print("📦 Installing Streamlit...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
            print("✅ Streamlit installed successfully!")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install Streamlit. Please install manually: pip install streamlit")
            return False


def check_system_requirements():
    """Check basic system requirements."""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 10):
        print(f"❌ Python {sys.version_info.major}.{sys.version_info.minor} detected. Python 3.10+ required.")
        return False
    else:
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected.")
    
    # Check if tutorials.py exists
    if not Path("tutorials.py").exists():
        print("❌ tutorials.py not found in current directory.")
        return False
    else:
        print("✅ Tutorial file found.")
    
    return True


def start_tutorials(port: int = 8501):
    """Start the Streamlit tutorials."""
    print("🚀 Starting TradPal Indicator Tutorials...")
    print(f"📱 Tutorials will be available at: http://localhost:{port}")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Set environment variables for better Streamlit experience
        env = os.environ.copy()
        env["STREAMLIT_SERVER_HEADLESS"] = "true"
        env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
        
        # Start Streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", "tutorials.py", "--server.port", str(port)]
        subprocess.run(cmd, env=env)
        
    except KeyboardInterrupt:
        print("\n🛑 Tutorials stopped by user.")
    except Exception as e:
        print(f"❌ Error starting tutorials: {e}")
        return False
    
    return True


def main():
    """Main launcher function."""
    print("🚀 TradPal Indicator - Interactive Tutorials")
    print("=" * 50)
    
    # Check requirements
    if not check_system_requirements():
        print("❌ System requirements not met. Please fix the issues above.")
        sys.exit(1)
    
    # Check Streamlit
    if not check_streamlit_installation():
        sys.exit(1)
    
    # Start tutorials
    success = start_tutorials()
    
    if success:
        print("✅ Tutorials completed successfully!")
    else:
        print("❌ Tutorials failed to start.")
        sys.exit(1)


if __name__ == "__main__":
    main()
