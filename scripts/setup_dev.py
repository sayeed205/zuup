#!/usr/bin/env python3
"""
Development environment setup script.

This script sets up the development environment for the Zuup download manager project.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str, check: bool = True) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"⚠️  {description} completed with warnings")
            if result.stderr:
                print(f"   Stderr: {result.stderr.strip()}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Command: {' '.join(cmd)}")
        print(f"   Error: {e.stderr.strip() if e.stderr else 'Unknown error'}")
        return False


def check_system_dependencies() -> bool:
    """Check if required system dependencies are installed."""
    print("🔍 Checking system dependencies...")
    
    # Check uv
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        print("✅ uv is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ uv is not installed. Please install it first:")
        print("   curl -LsSf https://astral.sh/uv/install.sh | sh")
        return False
    
    # Check Python version
    if sys.version_info < (3, 10):
        print(f"❌ Python 3.10+ required, found {sys.version_info.major}.{sys.version_info.minor}")
        return False
    else:
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} is compatible")
    
    return True


def setup_development_environment() -> bool:
    """Set up the development environment."""
    success = True
    
    # Install dependencies
    if not run_command(["uv", "sync", "--dev"], "Installing dependencies"):
        return False

    # Create necessary directories
    directories = [
        "logs",
        "downloads", 
        "temp",
        ".mypy_cache",
        ".ruff_cache"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("✅ Created necessary directories")

    # Run initial checks
    print("\n🔧 Running initial code quality checks...")
    
    # Type checking
    if not run_command(["uv", "run", "mypy", "src/zuup/"], "Running type check", check=False):
        print("⚠️  Type checking found issues (this is normal for initial setup)")
        success = False

    # Linting
    if not run_command(["uv", "run", "ruff", "check", "src/"], "Running linter", check=False):
        print("⚠️  Linting found issues (this is normal for initial setup)")
        success = False

    # Formatting check
    if not run_command(["uv", "run", "ruff", "format", "--check", "src/"], "Checking code formatting", check=False):
        print("⚠️  Code formatting issues found (this is normal for initial setup)")
        success = False

    return success


def main() -> None:
    """Set up development environment."""
    print("🚀 Setting up Zuup development environment\n")

    # Check system dependencies
    if not check_system_dependencies():
        sys.exit(1)

    # Setup development environment
    setup_success = setup_development_environment()

    print("\n" + "="*60)
    if setup_success:
        print("🎉 Development environment setup complete!")
    else:
        print("⚠️  Development environment setup completed with warnings")
    
    print("\n📋 Available development commands:")
    print("  uv run zuup --help                    # Show CLI help")
    print("  uv run mypy src/zuup/                 # Run type checking")
    print("  uv run ruff check src/                # Run linting")
    print("  uv run ruff format src/               # Format code")
    print("  uv run python -m zuup.main           # Run application")
    print("  uv run python examples/sample_downloads.py  # Run examples")
    
    print("\n🛠️  Development workflow:")
    print("  1. Make code changes")
    print("  2. Run: uv run ruff format src/      # Format code")
    print("  3. Run: uv run ruff check src/       # Check linting")
    print("  4. Run: uv run mypy src/zuup/        # Check types")
    print("  5. Test manually with examples/")


if __name__ == "__main__":
    main()
