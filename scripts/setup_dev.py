#!/usr/bin/env python3
"""
Development environment setup script.

This script sets up the development environment for the Zuup download manager project.
"""

import subprocess
import sys


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Command: {' '.join(cmd)}")
        print(f"   Error: {e.stderr}")
        return False


def main() -> None:
    """Set up development environment."""
    print("🚀 Setting up Zuup development environment")

    # Check if uv is installed
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        print("✅ uv is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ uv is not installed. Please install it first:")
        print("   curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)

    # Install dependencies
    if not run_command(["uv", "sync", "--dev"], "Installing dependencies"):
        sys.exit(1)

    # Development tools are ready

    # Run initial type checking
    if not run_command(["uv", "run", "mypy", "src/"], "Running initial type check"):
        print("⚠️  Type checking failed, but continuing setup...")

    # Run initial linting
    if not run_command(
        ["uv", "run", "ruff", "check", "src/"], "Running initial linting"
    ):
        print("⚠️  Linting found issues, but continuing setup...")

    print("\n🎉 Development environment setup complete!")
    print("\nNext steps:")
    print("  1. Activate the virtual environment: source .venv/bin/activate")
    print("  2. Start development: uv run zuup start --gui")
    print("  3. Run tests: uv run pytest")
    print("  4. Check code quality: uv run ruff check src/")


if __name__ == "__main__":
    main()
