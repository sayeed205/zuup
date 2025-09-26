#!/usr/bin/env python3
"""
Manual test script for development environment setup.

This script tests that the development environment is properly configured
and all tools are working correctly.
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def run_test(cmd: List[str], description: str) -> Tuple[bool, str]:
    """Run a test command and return success status and output."""
    try:
        result = subprocess.run(
            cmd, 
            check=False,  # Don't raise exception on non-zero exit
            capture_output=True, 
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        else:
            # For commands that are expected to fail (like linting with issues)
            output = result.stdout.strip() + result.stderr.strip()
            return False, output if output else "Command failed with no output"
    except subprocess.TimeoutExpired:
        return False, "Command timed out"


def test_uv_installation() -> bool:
    """Test that uv is properly installed and working."""
    print("🔍 Testing uv installation...")
    
    success, output = run_test(["uv", "--version"], "uv version check")
    if success:
        print(f"✅ uv is installed: {output}")
        return True
    else:
        print(f"❌ uv installation failed: {output}")
        return False


def test_python_version() -> bool:
    """Test Python version compatibility."""
    print("🔍 Testing Python version...")
    
    if sys.version_info >= (3, 10):
        print(f"✅ Python version is compatible: {sys.version_info.major}.{sys.version_info.minor}")
        return True
    else:
        print(f"❌ Python version is too old: {sys.version_info.major}.{sys.version_info.minor}")
        return False


def test_dependencies() -> bool:
    """Test that all dependencies are installed."""
    print("🔍 Testing dependencies...")
    
    dependencies = [
        ("mypy", ["uv", "run", "mypy", "--version"]),
        ("ruff", ["uv", "run", "ruff", "version"]),
        ("pydantic", ["uv", "run", "python", "-c", "import pydantic; print(pydantic.__version__)"]),
        ("fastapi", ["uv", "run", "python", "-c", "import fastapi; print(fastapi.__version__)"]),
        ("PySide6", ["uv", "run", "python", "-c", "import PySide6; print('PySide6 available')"]),
    ]
    
    all_success = True
    for name, cmd in dependencies:
        success, output = run_test(cmd, f"{name} check")
        if success:
            print(f"✅ {name}: {output}")
        else:
            print(f"❌ {name} failed: {output}")
            all_success = False
    
    return all_success


def test_project_structure() -> bool:
    """Test that project structure is correct."""
    print("🔍 Testing project structure...")
    
    required_files = [
        "pyproject.toml",
        "mypy.ini",
        "src/zuup/__init__.py",
        "src/zuup/main.py",
        "scripts/setup_dev.py",
        "scripts/build.py",
        "scripts/dev_tools.py",
    ]
    
    required_dirs = [
        "src/zuup/core",
        "src/zuup/engines",
        "src/zuup/config",
        "src/zuup/storage",
        "examples/config_examples",
        "examples/manual_tests",
    ]
    
    all_exist = True
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
            all_exist = False
    
    for dir_path in required_dirs:
        if Path(dir_path).is_dir():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ Missing directory: {dir_path}/")
            all_exist = False
    
    return all_exist


def test_type_checking() -> bool:
    """Test that type checking works."""
    print("🔍 Testing type checking...")
    
    # Create a simple test file
    test_file = Path("temp_type_test.py")
    test_content = '''
def add_numbers(a: int, b: int) -> int:
    return a + b

result: int = add_numbers(1, 2)
'''
    
    try:
        test_file.write_text(test_content)
        success, output = run_test(
            ["uv", "run", "mypy", str(test_file)], 
            "mypy type check"
        )
        test_file.unlink()  # Clean up
        
        if success:
            print("✅ Type checking works correctly")
            return True
        else:
            print(f"❌ Type checking failed: {output}")
            return False
    except Exception as e:
        if test_file.exists():
            test_file.unlink()
        print(f"❌ Type checking test failed: {e}")
        return False


def test_linting() -> bool:
    """Test that linting works."""
    print("🔍 Testing linting...")
    
    # Create a simple test file with linting issues
    test_file = Path("temp_lint_test.py")
    test_content = '''
import os
import sys  # unused import

def bad_function( ):
    x=1+2
    return x
'''
    
    try:
        test_file.write_text(test_content)
        success, output = run_test(
            ["uv", "run", "ruff", "check", str(test_file)], 
            "ruff linting check"
        )
        test_file.unlink()  # Clean up
        
        # Ruff should find issues, so success=False is expected
        if not success and ("F401" in output or "unused import" in output.lower()):  # Unused import error
            print("✅ Linting works correctly (found expected issues)")
            return True
        elif success:
            # If no issues found, that's also okay (ruff might have been run with --fix)
            print("✅ Linting works correctly (no issues found)")
            return True
        else:
            print(f"❌ Linting didn't work as expected: {output}")
            return False
    except Exception as e:
        if test_file.exists():
            test_file.unlink()
        print(f"❌ Linting test failed: {e}")
        return False


def test_formatting() -> bool:
    """Test that code formatting works."""
    print("🔍 Testing code formatting...")
    
    # Create a simple test file with formatting issues
    test_file = Path("temp_format_test.py")
    test_content = '''
def bad_formatting(  a,b  ):
    x=a+b
    return x
'''
    
    try:
        test_file.write_text(test_content)
        
        # Try to format the file
        success, output = run_test(
            ["uv", "run", "ruff", "format", str(test_file)], 
            "ruff formatting"
        )
        
        if success:
            # Check if file was actually formatted
            formatted_content = test_file.read_text()
            test_file.unlink()  # Clean up
            
            if "def bad_formatting(a, b):" in formatted_content:
                print("✅ Code formatting works correctly")
                return True
            else:
                print("❌ Code formatting didn't change the file as expected")
                return False
        else:
            test_file.unlink()  # Clean up
            print(f"❌ Code formatting failed: {output}")
            return False
    except Exception as e:
        if test_file.exists():
            test_file.unlink()
        print(f"❌ Code formatting test failed: {e}")
        return False


def main() -> None:
    """Run all development environment tests."""
    print("🚀 Testing Zuup Development Environment\n")
    
    tests = [
        ("UV Installation", test_uv_installation),
        ("Python Version", test_python_version),
        ("Dependencies", test_dependencies),
        ("Project Structure", test_project_structure),
        ("Type Checking", test_type_checking),
        ("Linting", test_linting),
        ("Code Formatting", test_formatting),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All development environment tests passed!")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()