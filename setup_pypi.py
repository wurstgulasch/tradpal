#!/usr/bin/env python3
"""
PyPI Setup and Publishing Script for TradPal

This script helps with:
- Building the package for PyPI
- Testing the package installation
- Publishing to PyPI or TestPyPI
- Managing package versions

Usage:
    python setup_pypi.py build        # Build package
    python setup_pypi.py test         # Test package in virtual environment
    python setup_pypi.py publish      # Publish to PyPI
    python setup_pypi.py test-publish # Publish to TestPyPI
    python setup_pypi.py check        # Check package with twine
"""

import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path

def run_command(cmd, cwd=None, check=True):
    """Run a shell command and return the result."""
    print(f"🔧 Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"❌ Command failed with return code {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        sys.exit(1)
    return result

def check_dependencies():
    """Check if required tools are installed."""
    required_tools = ['python', 'pip', 'twine', 'build']
    missing_tools = []

    for tool in required_tools:
        if shutil.which(tool) is None:
            missing_tools.append(tool)

    if missing_tools:
        print(f"❌ Missing required tools: {', '.join(missing_tools)}")
        print("💡 Install with: pip install build twine")
        sys.exit(1)

    print("✅ All required tools are available")

def build_package():
    """Build the package for distribution."""
    print("📦 Building TradPal package...")

    # Clean previous builds
    if os.path.exists('dist'):
        shutil.rmtree('dist')
    if os.path.exists('build'):
        shutil.rmtree('build')

    # Build package
    run_command(['python', '-m', 'build'])

    # Check what was built
    dist_files = list(Path('dist').glob('*'))
    if not dist_files:
        print("❌ No distribution files were created")
        sys.exit(1)

    print("✅ Package built successfully:")
    for f in dist_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(".2f")

    return dist_files

def check_package():
    """Check the package with twine."""
    print("🔍 Checking package with twine...")

    if not os.path.exists('dist'):
        print("❌ No dist directory found. Run 'build' first.")
        sys.exit(1)

    run_command(['twine', 'check', 'dist/*'])
    print("✅ Package check passed")

def test_package():
    """Test the package in a virtual environment."""
    print("🧪 Testing package installation...")

    import tempfile
    import venv

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Using temporary directory: {temp_dir}")

        # Create virtual environment
        venv_path = os.path.join(temp_dir, 'test_env')
        venv.create(venv_path, with_pip=True)

        # Get pip path
        if sys.platform == 'win32':
            pip_path = os.path.join(venv_path, 'Scripts', 'pip.exe')
            python_path = os.path.join(venv_path, 'Scripts', 'python.exe')
        else:
            pip_path = os.path.join(venv_path, 'bin', 'pip')
            python_path = os.path.join(venv_path, 'bin', 'python')

        # Install package
        wheel_files = list(Path('dist').glob('*.whl'))
        if not wheel_files:
            print("❌ No wheel file found in dist/")
            sys.exit(1)

        wheel_file = str(wheel_files[0])
        run_command([pip_path, 'install', wheel_file])

        # Test basic import
        test_script = '''
import sys
try:
    from src.main import main
    print("✅ Main module imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

try:
    from src.data_fetcher import fetch_historical_data
    print("✅ Data fetcher imported successfully")
except ImportError as e:
    print(f"❌ Data fetcher import failed: {e}")
    sys.exit(1)

try:
    from src.indicators import calculate_indicators
    print("✅ Indicators module imported successfully")
except ImportError as e:
    print(f"❌ Indicators import failed: {e}")
    sys.exit(1)

print("🎉 All core modules imported successfully!")
'''

        test_file = os.path.join(temp_dir, 'test_import.py')
        with open(test_file, 'w') as f:
            f.write(test_script)

        run_command([python_path, test_file])
        print("✅ Package installation test passed")

def publish_to_pypi(test_pypi=False):
    """Publish the package to PyPI or TestPyPI."""
    if test_pypi:
        print("🧪 Publishing to TestPyPI...")
        repo_url = 'https://test.pypi.org/legacy/'
        token_env = 'TEST_PYPI_API_TOKEN'
    else:
        print("🚀 Publishing to PyPI...")
        repo_url = None
        token_env = 'PYPI_API_TOKEN'

    # Check for API token
    token = os.getenv(token_env)
    if not token:
        print(f"❌ {token_env} environment variable not set")
        print("💡 Set your PyPI API token:")
        print(f"   export {token_env}='your_api_token_here'")
        print("   Get token from: https://pypi.org/manage/account/token/")
        sys.exit(1)

    # Build package first
    build_package()
    check_package()

    # Upload to PyPI
    cmd = ['twine', 'upload', 'dist/*']
    if test_pypi:
        cmd.extend(['--repository-url', repo_url])

    # Set environment for token
    env = os.environ.copy()
    env['TWINE_USERNAME'] = '__token__'
    env['TWINE_PASSWORD'] = token

    run_command(cmd, env=env)
    print(f"✅ Package published successfully to {'TestPyPI' if test_pypi else 'PyPI'}!")

def show_package_info():
    """Show information about the package."""
    print("📋 TradPal Package Information")
    print("=" * 50)

    try:
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                print("❌ tomli/tomllib not available")
                print("💡 Install with: pip install tomli")
                return

        with open('pyproject.toml', 'rb') as f:
            config = tomllib.load(f)

        project = config.get('project', {})
        print(f"Name: {project.get('name', 'N/A')}")
        print(f"Version: {project.get('version', 'N/A')}")
        print(f"Description: {project.get('description', 'N/A')}")
        print(f"Python: {project.get('requires-python', 'N/A')}")

        deps = project.get('dependencies', [])
        print(f"Core Dependencies: {len(deps)}")

        optional_deps = project.get('optional-dependencies', {})
        print(f"Optional Groups: {list(optional_deps.keys())}")

    except ImportError:
        print("❌ tomllib not available (Python < 3.11)")
        print("💡 Install with: pip install tomli")

    # Check dist files
    if os.path.exists('dist'):
        dist_files = list(Path('dist').glob('*'))
        print(f"\\nBuilt Distributions: {len(dist_files)}")
        for f in dist_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(".2f")
    else:
        print("\\nNo built distributions found")

def main():
    parser = argparse.ArgumentParser(
        description='PyPI Setup and Publishing Script for TradPal',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python setup_pypi.py build          # Build package
  python setup_pypi.py check          # Check package with twine
  python setup_pypi.py test           # Test package installation
  python setup_pypi.py publish        # Publish to PyPI
  python setup_pypi.py test-publish   # Publish to TestPyPI
  python setup_pypi.py info           # Show package information
        '''
    )

    parser.add_argument(
        'action',
        choices=['build', 'check', 'test', 'publish', 'test-publish', 'info'],
        help='Action to perform'
    )

    args = parser.parse_args()

    print("🚀 TradPal - PyPI Setup Script")
    print("=" * 50)

    if args.action in ['build', 'check', 'test', 'publish', 'test-publish']:
        check_dependencies()

    if args.action == 'build':
        build_package()
    elif args.action == 'check':
        check_package()
    elif args.action == 'test':
        test_package()
    elif args.action == 'publish':
        publish_to_pypi(test_pypi=False)
    elif args.action == 'test-publish':
        publish_to_pypi(test_pypi=True)
    elif args.action == 'info':
        show_package_info()

    print("\\n✨ Done!")

if __name__ == "__main__":
    main()