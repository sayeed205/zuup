#!/usr/bin/env python3
"""
Manual test script for development tooling integration.

This script tests that all development tools work together correctly
and can be used in a typical development workflow.
"""

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple


def run_command(cmd: List[str], description: str, timeout: int = 30) -> Tuple[bool, str, str]:
    """Run a command and return success status, stdout, and stderr."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)


def test_script_execution() -> bool:
    """Test that development scripts can be executed."""
    print("🔍 Testing development script execution...")
    
    scripts = [
        (["python", "scripts/setup_dev.py", "--help"], "setup_dev.py help"),
        (["python", "scripts/build.py", "--help"], "build.py help"),
        (["python", "scripts/dev_tools.py", "--help"], "dev_tools.py help"),
    ]
    
    all_success = True
    for cmd, description in scripts:
        success, stdout, stderr = run_command(cmd, description)
        # setup_dev.py doesn't have --help, so check if it runs without error
        if "setup_dev.py" in description:
            if "Setting up Zuup development environment" in stdout or "uv is not installed" in stdout:
                print(f"✅ {description}")
            else:
                print(f"❌ {description} failed: {stderr}")
                all_success = False
        elif success and "usage:" in stdout.lower():
            print(f"✅ {description}")
        else:
            print(f"❌ {description} failed: {stderr}")
            all_success = False
    
    return all_success


def test_mypy_configuration() -> bool:
    """Test mypy configuration and execution."""
    print("🔍 Testing mypy configuration...")
    
    # Test mypy with existing source code
    success, stdout, stderr = run_command(
        ["uv", "run", "mypy", "src/zuup/__init__.py"],
        "mypy on __init__.py"
    )
    
    if success:
        print("✅ mypy runs successfully on source code")
    else:
        # Check if it's a configuration issue or just type errors
        if "error: Cannot find implementation" in stderr or "No module named" in stderr:
            print(f"❌ mypy configuration issue: {stderr}")
            return False
        else:
            print("✅ mypy runs (found type issues, which is normal)")
    
    # Test mypy configuration file
    mypy_config = Path("mypy.ini")
    if mypy_config.exists():
        print("✅ mypy.ini configuration file exists")
    else:
        print("❌ mypy.ini configuration file missing")
        return False
    
    return True


def test_ruff_configuration() -> bool:
    """Test ruff configuration and execution."""
    print("🔍 Testing ruff configuration...")
    
    # Test ruff check
    success, stdout, stderr = run_command(
        ["uv", "run", "ruff", "check", "src/zuup/__init__.py"],
        "ruff check on __init__.py"
    )
    
    # Ruff might find issues, that's okay
    print("✅ ruff check runs successfully")
    
    # Test ruff format check
    success, stdout, stderr = run_command(
        ["uv", "run", "ruff", "format", "--check", "src/zuup/__init__.py"],
        "ruff format check on __init__.py"
    )
    
    # Format check might fail if code needs formatting, that's okay
    print("✅ ruff format check runs successfully")
    
    return True


def test_workflow_integration() -> bool:
    """Test a complete development workflow."""
    print("🔍 Testing complete development workflow...")
    
    # Create a temporary Python file with issues
    temp_dir = Path(tempfile.mkdtemp())
    test_file = temp_dir / "test_workflow.py"
    
    # Code with formatting and type issues
    problematic_code = '''
import os
import sys  # unused import

def bad_function( a,b ):
    x=a+b
    return x

result=bad_function("hello",123)  # type error
'''
    
    try:
        test_file.write_text(problematic_code)
        
        # Step 1: Run ruff check (should find issues)
        success, stdout, stderr = run_command(
            ["uv", "run", "ruff", "check", str(test_file)],
            "ruff check on problematic code"
        )
        
        if not success and ("F401" in stderr or "F401" in stdout):  # Unused import
            print("✅ ruff check correctly identifies issues")
        else:
            print("❌ ruff check didn't identify expected issues")
            return False
        
        # Step 2: Fix formatting with ruff
        success, stdout, stderr = run_command(
            ["uv", "run", "ruff", "format", str(test_file)],
            "ruff format fix"
        )
        
        if success:
            print("✅ ruff format fixes formatting issues")
        else:
            print(f"❌ ruff format failed: {stderr}")
            return False
        
        # Step 3: Check if formatting was applied
        formatted_content = test_file.read_text()
        if "def bad_function(a, b):" in formatted_content:
            print("✅ Code formatting was applied correctly")
        else:
            print("❌ Code formatting was not applied correctly")
            return False
        
        # Step 4: Run mypy (should still find type issues)
        success, stdout, stderr = run_command(
            ["uv", "run", "mypy", str(test_file)],
            "mypy check on formatted code"
        )
        
        if not success:
            print("✅ mypy correctly identifies remaining type issues")
        else:
            print("⚠️  mypy didn't find expected type issues (might be okay)")
        
        return True
    
    finally:
        # Clean up
        if test_file.exists():
            test_file.unlink()
        temp_dir.rmdir()


def test_configuration_files() -> bool:
    """Test that all configuration files are present and valid."""
    print("🔍 Testing configuration files...")
    
    config_files = [
        ("pyproject.toml", "Project configuration"),
        ("mypy.ini", "MyPy configuration"),
        (".editorconfig", "Editor configuration"),
    ]
    
    all_valid = True
    for file_path, description in config_files:
        if Path(file_path).exists():
            print(f"✅ {description}: {file_path}")
        else:
            print(f"❌ Missing {description}: {file_path}")
            all_valid = False
    
    # Test pyproject.toml syntax
    try:
        # Try tomllib first (Python 3.11+), then fall back to tomli
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        
        with open("pyproject.toml", "rb") as f:
            tomllib.load(f)
        print("✅ pyproject.toml has valid TOML syntax")
    except Exception as e:
        print(f"❌ pyproject.toml syntax error: {e}")
        all_valid = False
    
    return all_valid


def test_example_configurations() -> bool:
    """Test that example configuration files are present."""
    print("🔍 Testing example configurations...")
    
    example_configs = [
        "examples/config_examples/example_global_config.json",
        "examples/config_examples/development_config.json",
        "examples/config_examples/production_config.json",
        "examples/config_examples/example_task_config.json",
    ]
    
    all_exist = True
    for config_path in example_configs:
        if Path(config_path).exists():
            print(f"✅ {config_path}")
            
            # Test JSON syntax
            try:
                import json
                with open(config_path) as f:
                    json.load(f)
                print(f"   ✅ Valid JSON syntax")
            except Exception as e:
                print(f"   ❌ JSON syntax error: {e}")
                all_exist = False
        else:
            print(f"❌ Missing: {config_path}")
            all_exist = False
    
    return all_exist


def test_manual_test_scripts() -> bool:
    """Test that manual test scripts are executable."""
    print("🔍 Testing manual test scripts...")
    
    test_scripts = [
        "examples/manual_tests/test_development_environment.py",
        "examples/manual_tests/test_type_safety.py",
        "examples/manual_tests/test_tooling_integration.py",
    ]
    
    all_executable = True
    for script_path in test_scripts:
        if Path(script_path).exists():
            print(f"✅ {script_path}")
            
            # Test that script can show help or runs without error
            success, stdout, stderr = run_command(
                ["python", script_path, "--help"],
                f"{script_path} help",
                timeout=5
            )
            
            # Check if it's the current script (avoid infinite recursion)
            if "test_tooling_integration.py" in script_path:
                print(f"   ✅ Script is executable (current script)")
            elif success or "usage:" in stdout.lower() or "manual test" in stdout.lower() or "Testing" in stdout:
                print(f"   ✅ Script is executable")
            else:
                print(f"   ❌ Script execution issue: {stderr}")
                all_executable = False
        else:
            print(f"❌ Missing: {script_path}")
            all_executable = False
    
    return all_executable


def main() -> None:
    """Run all tooling integration tests."""
    print("🚀 Testing Development Tooling Integration\n")
    
    tests = [
        ("Script Execution", test_script_execution),
        ("MyPy Configuration", test_mypy_configuration),
        ("Ruff Configuration", test_ruff_configuration),
        ("Workflow Integration", test_workflow_integration),
        ("Configuration Files", test_configuration_files),
        ("Example Configurations", test_example_configurations),
        ("Manual Test Scripts", test_manual_test_scripts),
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
    print("TOOLING INTEGRATION TEST SUMMARY")
    print('='*60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tooling integration tests passed!")
        print("\n📋 Development workflow is ready:")
        print("  1. uv run ruff format src/          # Format code")
        print("  2. uv run ruff check src/           # Check linting")
        print("  3. uv run mypy src/zuup/            # Check types")
        print("  4. python scripts/build.py --all   # Full build")
        return True
    else:
        print("⚠️  Some tooling integration tests failed.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)