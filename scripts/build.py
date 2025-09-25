#!/usr/bin/env python3
"""
Build script for the Zuup download manager project.

This script handles building, packaging, and distribution tasks.
"""

from pathlib import Path
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


def clean_build() -> None:
    """Clean build artifacts."""
    print("🧹 Cleaning build artifacts...")

    # Remove build directories
    for path in ["build", "dist", "*.egg-info"]:
        for p in Path().glob(path):
            if p.is_dir():
                import shutil

                shutil.rmtree(p)
                print(f"   Removed {p}")
            elif p.is_file():
                p.unlink()
                print(f"   Removed {p}")


def run_checks() -> bool:
    """Run all code quality checks."""
    print("🔍 Running code quality checks...")

    checks = [
        (["uv", "run", "mypy", "src/"], "Type checking"),
        (["uv", "run", "ruff", "check", "src/"], "Linting"),
        (["uv", "run", "ruff", "format", "--check", "src/"], "Format checking"),
        (["uv", "run", "pytest"], "Running tests"),
    ]

    for cmd, description in checks:
        if not run_command(cmd, description):
            return False

    return True


def build_package() -> bool:
    """Build the package."""
    return run_command(["uv", "build"], "Building package")


def main() -> None:
    """Main build script."""
    import argparse

    parser = argparse.ArgumentParser(description="Build script for Zuup")
    parser.add_argument("--clean", action="store_true", help="Clean build artifacts")
    parser.add_argument("--check", action="store_true", help="Run code quality checks")
    parser.add_argument("--build", action="store_true", help="Build package")
    parser.add_argument("--all", action="store_true", help="Run all steps")

    args = parser.parse_args()

    if not any([args.clean, args.check, args.build, args.all]):
        parser.print_help()
        return

    if args.all or args.clean:
        clean_build()

    if args.all or args.check:
        if not run_checks():
            print("❌ Code quality checks failed")
            sys.exit(1)

    if args.all or args.build:
        if not build_package():
            print("❌ Package build failed")
            sys.exit(1)

    print("🎉 Build completed successfully!")


if __name__ == "__main__":
    main()
