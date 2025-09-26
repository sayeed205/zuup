#!/usr/bin/env python3
"""
Development tools and utilities for the Zuup project.

This script provides various development utilities for code quality,
testing, and project maintenance.
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def run_command(
    cmd: List[str], 
    description: str, 
    check: bool = True,
    capture_output: bool = True
) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(
            cmd, 
            check=check, 
            capture_output=capture_output, 
            text=True
        )
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            if not capture_output and result.stdout:
                print(result.stdout)
            return True
        else:
            print(f"⚠️  {description} completed with warnings")
            if capture_output and result.stderr:
                print(f"   Stderr: {result.stderr.strip()}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Command: {' '.join(cmd)}")
        if capture_output and e.stderr:
            print(f"   Error: {e.stderr.strip()}")
        return False


def check_dependencies() -> bool:
    """Check if all development dependencies are available."""
    print("🔍 Checking development dependencies...")
    
    dependencies = [
        ("uv", ["uv", "--version"]),
        ("mypy", ["uv", "run", "mypy", "--version"]),
        ("ruff", ["uv", "run", "ruff", "--version"]),
    ]
    
    all_available = True
    for name, cmd in dependencies:
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"✅ {name} is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"❌ {name} is not available")
            all_available = False
    
    return all_available


def run_type_check_detailed() -> bool:
    """Run detailed type checking with mypy."""
    print("🔍 Running detailed type checking...")
    
    # Check main source code
    success = run_command(
        ["uv", "run", "mypy", "src/zuup/", "--show-error-codes", "--pretty"],
        "Type checking main source code",
        check=False
    )
    
    # Check examples (less strict)
    example_success = run_command(
        ["uv", "run", "mypy", "examples/", "--ignore-missing-imports"],
        "Type checking examples",
        check=False
    )
    
    return success and example_success


def run_linting_detailed() -> bool:
    """Run detailed linting with ruff."""
    print("🔍 Running detailed linting...")
    
    # Check for issues
    check_success = run_command(
        ["uv", "run", "ruff", "check", "src/", "--show-fixes"],
        "Checking for linting issues",
        check=False
    )
    
    # Check formatting
    format_success = run_command(
        ["uv", "run", "ruff", "format", "--check", "src/"],
        "Checking code formatting",
        check=False
    )
    
    return check_success and format_success


def fix_code_issues() -> bool:
    """Automatically fix code issues where possible."""
    print("🔧 Fixing code issues automatically...")
    
    # Fix linting issues
    lint_success = run_command(
        ["uv", "run", "ruff", "check", "src/", "--fix"],
        "Fixing linting issues",
        check=False
    )
    
    # Fix formatting
    format_success = run_command(
        ["uv", "run", "ruff", "format", "src/"],
        "Fixing code formatting",
        check=False
    )
    
    return lint_success and format_success


def generate_type_stubs() -> bool:
    """Generate type stubs for better type checking."""
    print("📝 Generating type stubs...")
    
    # Create stubs directory
    stubs_dir = Path("stubs")
    stubs_dir.mkdir(exist_ok=True)
    
    # Generate stubs for key dependencies that might not have them
    stub_packages = ["libtorrent", "pycurl"]
    
    success = True
    for package in stub_packages:
        stub_success = run_command(
            ["uv", "run", "stubgen", "-p", package, "-o", "stubs/"],
            f"Generating stubs for {package}",
            check=False
        )
        if not stub_success:
            success = False
    
    return success


def run_code_metrics() -> bool:
    """Run code metrics and complexity analysis."""
    print("📊 Running code metrics...")
    
    # Count lines of code
    try:
        result = subprocess.run(
            ["find", "src/", "-name", "*.py", "-exec", "wc", "-l", "{}", "+"],
            capture_output=True,
            text=True,
            check=True
        )
        lines = result.stdout.strip().split('\n')
        if lines:
            total_line = lines[-1]
            print(f"📈 Total lines of code: {total_line.split()[0]}")
    except subprocess.CalledProcessError:
        print("⚠️  Could not count lines of code")
    
    # Run complexity analysis with radon if available
    complexity_success = run_command(
        ["uv", "run", "radon", "cc", "src/", "-a"],
        "Running complexity analysis",
        check=False
    )
    
    return complexity_success


def validate_project_structure() -> bool:
    """Validate project structure and required files."""
    print("🏗️  Validating project structure...")
    
    required_files = [
        "pyproject.toml",
        "README.md",
        "src/zuup/__init__.py",
        "src/zuup/main.py",
        "mypy.ini",
    ]
    
    required_dirs = [
        "src/zuup/core",
        "src/zuup/engines", 
        "src/zuup/config",
        "src/zuup/storage",
        "src/zuup/api",
        "src/zuup/gui",
        "examples",
        "scripts",
    ]
    
    all_valid = True
    
    # Check files
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
            all_valid = False
    
    # Check directories
    for dir_path in required_dirs:
        if Path(dir_path).is_dir():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ Missing directory: {dir_path}/")
            all_valid = False
    
    return all_valid


def main() -> None:
    """Main development tools script."""
    import argparse

    parser = argparse.ArgumentParser(description="Development tools for Zuup")
    parser.add_argument("--check-deps", action="store_true", help="Check development dependencies")
    parser.add_argument("--type-check", action="store_true", help="Run detailed type checking")
    parser.add_argument("--lint", action="store_true", help="Run detailed linting")
    parser.add_argument("--fix", action="store_true", help="Fix code issues automatically")
    parser.add_argument("--stubs", action="store_true", help="Generate type stubs")
    parser.add_argument("--metrics", action="store_true", help="Run code metrics")
    parser.add_argument("--validate", action="store_true", help="Validate project structure")
    parser.add_argument("--all", action="store_true", help="Run all checks")

    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        return

    success = True

    if args.all or args.check_deps:
        if not check_dependencies():
            success = False

    if args.all or args.validate:
        if not validate_project_structure():
            success = False

    if args.fix:
        if not fix_code_issues():
            success = False

    if args.all or args.type_check:
        if not run_type_check_detailed():
            success = False

    if args.all or args.lint:
        if not run_linting_detailed():
            success = False

    if args.stubs:
        if not generate_type_stubs():
            success = False

    if args.metrics:
        if not run_code_metrics():
            success = False

    print("\n" + "="*60)
    if success:
        print("🎉 All development tools completed successfully!")
    else:
        print("⚠️  Some development tools completed with issues")
        sys.exit(1)


if __name__ == "__main__":
    main()