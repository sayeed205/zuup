#!/usr/bin/env python3
"""
Manual test script for type safety verification.

This script tests various type safety scenarios to ensure the development
environment properly catches type errors.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple


def create_test_file(content: str) -> Path:
    """Create a temporary test file with the given content."""
    temp_file = Path(tempfile.mktemp(suffix=".py"))
    temp_file.write_text(content)
    return temp_file


def run_mypy_test(content: str, description: str) -> Tuple[bool, str]:
    """Run mypy on test content and return results."""
    test_file = create_test_file(content)

    try:
        result = subprocess.run(
            ["uv", "run", "mypy", str(test_file)],
            capture_output=True,
            text=True,
            timeout=10,
        )

        test_file.unlink()  # Clean up

        # For type error tests, we expect mypy to fail (find errors)
        # For valid code tests, we expect mypy to succeed
        return result.returncode == 0, result.stdout + result.stderr

    except Exception as e:
        if test_file.exists():
            test_file.unlink()
        return False, str(e)


def test_basic_type_annotations() -> bool:
    """Test basic type annotation checking."""
    print("🔍 Testing basic type annotations...")

    # Valid code - should pass
    valid_code = """
def add_numbers(a: int, b: int) -> int:
    return a + b

def greet(name: str) -> str:
    return f"Hello, {name}!"

result: int = add_numbers(5, 3)
message: str = greet("World")
"""

    success, output = run_mypy_test(valid_code, "valid type annotations")
    if success:
        print("✅ Valid type annotations pass mypy")
    else:
        print(f"❌ Valid type annotations failed mypy: {output}")
        return False

    # Invalid code - should fail
    invalid_code = """
def add_numbers(a: int, b: int) -> int:
    return a + b

# Type error: passing string to int parameter
result: int = add_numbers("5", 3)
"""

    success, output = run_mypy_test(invalid_code, "invalid type usage")
    if not success and "Argument 1" in output:
        print("✅ Invalid type usage correctly caught by mypy")
    else:
        print(f"❌ Invalid type usage not caught by mypy: {output}")
        return False

    return True


def test_pydantic_integration() -> bool:
    """Test Pydantic model type checking."""
    print("🔍 Testing Pydantic integration...")

    # Valid Pydantic usage
    valid_pydantic = """
from pydantic import BaseModel
from typing import Optional

class User(BaseModel):
    name: str
    age: int
    email: Optional[str] = None

# Valid usage
user = User(name="John", age=30)
user_with_email = User(name="Jane", age=25, email="jane@example.com")
"""

    success, output = run_mypy_test(valid_pydantic, "valid Pydantic usage")
    if success:
        print("✅ Valid Pydantic usage passes mypy")
    else:
        print(f"❌ Valid Pydantic usage failed mypy: {output}")
        return False

    # Invalid Pydantic usage
    invalid_pydantic = """
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

# Type error: missing required field
user = User(name="John")
"""

    success, output = run_mypy_test(invalid_pydantic, "invalid Pydantic usage")
    if not success:
        print("✅ Invalid Pydantic usage correctly caught by mypy")
    else:
        print(f"❌ Invalid Pydantic usage not caught by mypy: {output}")
        return False

    return True


def test_generic_types() -> bool:
    """Test generic type checking."""
    print("🔍 Testing generic types...")

    # Valid generic usage
    valid_generics = """
from typing import List, Dict, Optional, TypeVar, Generic

T = TypeVar('T')

class Container(Generic[T]):
    def __init__(self, value: T) -> None:
        self.value = value
    
    def get(self) -> T:
        return self.value

# Valid usage
numbers: List[int] = [1, 2, 3]
mapping: Dict[str, int] = {"a": 1, "b": 2}
container: Container[str] = Container("hello")
"""

    success, output = run_mypy_test(valid_generics, "valid generic usage")
    if success:
        print("✅ Valid generic usage passes mypy")
    else:
        print(f"❌ Valid generic usage failed mypy: {output}")
        return False

    # Invalid generic usage
    invalid_generics = """
from typing import List

# Type error: mixing types in list
numbers: List[int] = [1, 2, "three"]
"""

    success, output = run_mypy_test(invalid_generics, "invalid generic usage")
    if not success and ("List[int]" in output or "incompatible" in output):
        print("✅ Invalid generic usage correctly caught by mypy")
    else:
        print(f"❌ Invalid generic usage not caught by mypy: {output}")
        return False

    return True


def test_protocol_types() -> bool:
    """Test Protocol type checking."""
    print("🔍 Testing Protocol types...")

    # Valid Protocol usage
    valid_protocol = """
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> str: ...

class Circle:
    def draw(self) -> str:
        return "Drawing circle"

class Square:
    def draw(self) -> str:
        return "Drawing square"

def render(shape: Drawable) -> str:
    return shape.draw()

# Valid usage
circle = Circle()
square = Square()
result1 = render(circle)
result2 = render(square)
"""

    success, output = run_mypy_test(valid_protocol, "valid Protocol usage")
    if success:
        print("✅ Valid Protocol usage passes mypy")
    else:
        print(f"❌ Valid Protocol usage failed mypy: {output}")
        return False

    # Invalid Protocol usage
    invalid_protocol = """
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> str: ...

class InvalidShape:
    def paint(self) -> str:  # Wrong method name
        return "Painting"

def render(shape: Drawable) -> str:
    return shape.draw()

# Type error: InvalidShape doesn't implement Drawable protocol
shape = InvalidShape()
result = render(shape)
"""

    success, output = run_mypy_test(invalid_protocol, "invalid Protocol usage")
    if not success:
        print("✅ Invalid Protocol usage correctly caught by mypy")
    else:
        print(f"❌ Invalid Protocol usage not caught by mypy: {output}")
        return False

    return True


def test_async_types() -> bool:
    """Test async/await type checking."""
    print("🔍 Testing async types...")

    # Valid async usage
    valid_async = """
import asyncio
from typing import AsyncIterator

async def fetch_data() -> str:
    await asyncio.sleep(0.1)
    return "data"

async def generate_numbers() -> AsyncIterator[int]:
    for i in range(3):
        yield i

async def main() -> None:
    data: str = await fetch_data()
    async for num in generate_numbers():
        print(num)
"""

    success, output = run_mypy_test(valid_async, "valid async usage")
    if success:
        print("✅ Valid async usage passes mypy")
    else:
        print(f"❌ Valid async usage failed mypy: {output}")
        return False

    return True


def test_strict_mode_features() -> bool:
    """Test strict mode type checking features."""
    print("🔍 Testing strict mode features...")

    # Code that should fail in strict mode
    strict_violations = """
# Missing return type annotation
def process_data(data):
    return data.upper()

# Missing parameter type annotation
def calculate(x, y: int) -> int:
    return x + y

# Implicit Any
from typing import Any
def handle_any(data: Any) -> Any:
    return data
"""

    success, output = run_mypy_test(strict_violations, "strict mode violations")
    if not success:
        print("✅ Strict mode violations correctly caught by mypy")
    else:
        print(f"❌ Strict mode violations not caught by mypy: {output}")
        return False

    return True


def main() -> None:
    """Run all type safety tests."""
    print("🚀 Testing Type Safety Configuration\n")

    tests = [
        ("Basic Type Annotations", test_basic_type_annotations),
        ("Pydantic Integration", test_pydantic_integration),
        ("Generic Types", test_generic_types),
        ("Protocol Types", test_protocol_types),
        ("Async Types", test_async_types),
        ("Strict Mode Features", test_strict_mode_features),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{'=' * 50}")
        print(f"Running: {test_name}")
        print("=" * 50)

        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print(f"\n{'=' * 60}")
    print("TYPE SAFETY TEST SUMMARY")
    print("=" * 60)

    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1

    print(f"\nResults: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("🎉 All type safety tests passed!")
        return True
    else:
        print("⚠️  Some type safety tests failed.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
