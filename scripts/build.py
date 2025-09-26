#!/usr/bin/env python3
"""
Build script for the Zuup download manager project.

This script handles building, packaging, and distribution tasks.
"""

import shutil
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


def clean_build() -> None:
    """Clean build artifacts and cache directories."""
    print("🧹 Cleaning build artifacts...")

    # Build and distribution directories
    build_dirs = ["build", "dist", "*.egg-info", "src/*.egg-info"]

    # Cache directories
    cache_dirs = [
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "__pycache__",
        "**/__pycache__",
        "**/*.pyc",
        "**/*.pyo",
    ]

    all_patterns = build_dirs + cache_dirs

    for pattern in all_patterns:
        for path in Path().glob(pattern):
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                    print(f"   Removed directory: {path}")
                elif path.is_file():
                    path.unlink()
                    print(f"   Removed file: {path}")
            except Exception as e:
                print(f"   Warning: Could not remove {path}: {e}")


def run_type_check() -> bool:
    """Run mypy type checking."""
    return run_command(
        ["uv", "run", "mypy", "src/zuup/"], "Type checking with mypy", check=False
    )


def run_linting() -> bool:
    """Run ruff linting."""
    return run_command(
        ["uv", "run", "ruff", "check", "src/"], "Linting with ruff", check=False
    )


def run_formatting_check() -> bool:
    """Check code formatting."""
    return run_command(
        ["uv", "run", "ruff", "format", "--check", "src/"],
        "Checking code formatting",
        check=False,
    )


def fix_formatting() -> bool:
    """Fix code formatting."""
    return run_command(
        ["uv", "run", "ruff", "format", "src/"], "Fixing code formatting", check=False
    )


def run_security_check() -> bool:
    """Run security checks with bandit."""
    return run_command(
        ["uv", "run", "bandit", "-r", "src/", "-f", "text"],
        "Running security checks",
        check=False,
    )


def run_all_checks(fix_format: bool = False) -> bool:
    """Run all code quality checks."""
    print("🔍 Running comprehensive code quality checks...\n")

    success = True

    # Fix formatting if requested
    if fix_format:
        if not fix_formatting():
            success = False
    else:
        if not run_formatting_check():
            success = False

    # Run linting
    if not run_linting():
        success = False

    # Run type checking
    if not run_type_check():
        success = False

    return success


def build_package() -> bool:
    """Build the package."""
    return run_command(["uv", "build"], "Building package")


def install_dev_package() -> bool:
    """Install package in development mode."""
    return run_command(
        ["uv", "pip", "install", "-e", "."], "Installing in development mode"
    )


def generate_requirements() -> bool:
    """Generate requirements.txt for deployment."""
    return run_command(
        ["uv", "pip", "compile", "pyproject.toml", "-o", "requirements.txt"],
        "Generating requirements.txt",
    )


def main() -> None:
    """Main build script."""
    import argparse

    parser = argparse.ArgumentParser(description="Build script for Zuup")
    parser.add_argument(
        "--clean", action="store_true", help="Clean build artifacts and caches"
    )
    parser.add_argument("--format", action="store_true", help="Fix code formatting")
    parser.add_argument("--check", action="store_true", help="Run code quality checks")
    parser.add_argument(
        "--type-check", action="store_true", help="Run type checking only"
    )
    parser.add_argument("--lint", action="store_true", help="Run linting only")
    parser.add_argument("--security", action="store_true", help="Run security checks")
    parser.add_argument("--build", action="store_true", help="Build package")
    parser.add_argument(
        "--install-dev", action="store_true", help="Install in development mode"
    )
    parser.add_argument(
        "--requirements", action="store_true", help="Generate requirements.txt"
    )
    parser.add_argument(
        "--all", action="store_true", help="Run all steps (clean, check, build)"
    )

    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        return

    success = True

    # Clean step
    if args.all or args.clean:
        clean_build()

    # Individual checks
    if args.format:
        if not fix_formatting():
            success = False

    if args.type_check:
        if not run_type_check():
            success = False

    if args.lint:
        if not run_linting():
            success = False

    if args.security:
        if not run_security_check():
            success = False

    # Comprehensive checks
    if args.all or args.check:
        if not run_all_checks(fix_format=args.all):
            success = False

    # Development installation
    if args.install_dev:
        if not install_dev_package():
            success = False

    # Requirements generation
    if args.requirements:
        if not generate_requirements():
            success = False

    # Build step
    if args.all or args.build:
        if not build_package():
            success = False

    print("\n" + "=" * 60)
    if success:
        print("🎉 Build process completed successfully!")
    else:
        print("⚠️  Build process completed with issues")
        sys.exit(1)

    if args.all:
        print("\n📦 Build artifacts:")
        dist_dir = Path("dist")
        if dist_dir.exists():
            for file in dist_dir.iterdir():
                print(f"   {file}")


if __name__ == "__main__":
    main()
