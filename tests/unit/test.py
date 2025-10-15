#!/usr/bin/env python3
"""
Alternative Test Runner for TradPal Project

This is a simple wrapper around pytest. For full pytest functionality,
use pytest directly:

    pytest                    # Run all tests
    pytest tests/             # Run all tests in tests directory
    pytest -v                 # Verbose output
    pytest --cov=src          # With coverage
    pytest -k "test_name"      # Run specific test
    pytest tests/test_edge_cases.py::TestClass::test_method  # Run specific test method

This script is kept for convenience and CI/CD integration.
"""

import subprocess
import sys
import os

def main():
    """Run tests using pytest directly."""
    # Change to the parent directory (project root)
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Build pytest command
    cmd = [sys.executable, '-m', 'pytest']

    # Add any additional arguments passed to this script
    cmd.extend(sys.argv[1:])

    # If no arguments provided, run all tests
    if len(sys.argv) == 1:
        cmd.append('tests/')

    print(f"Running: {' '.join(cmd)}")
    print()

    # Run pytest
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()