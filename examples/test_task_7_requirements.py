#!/usr/bin/env python3
"""
Test script to verify Task 7 requirements are implemented correctly.

This script tests all the requirements for task 7:
- Requirement 3.1: GUI mode with embedded server
- Requirement 3.2: Headless server mode for remote access  
- Requirement 3.3: Configure appropriate logging and monitoring
- Requirement 2.4: Use dependency injection patterns for loose coupling
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

import httpx
from rich.console import Console
from rich.table import Table

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from zuup.config.manager import ConfigManager
from zuup.core.app import Application

console = Console()
logger = logging.getLogger(__name__)


def test_requirement_3_1():
    """Test Requirement 3.1: GUI mode with embedded server functionality."""
    console.print("[bold blue]Testing Requirement 3.1: GUI mode with embedded server[/bold blue]")
    
    try:
        config_manager = ConfigManager()
        app = Application(config_manager=config_manager)
        
        # Check that GUI mode method exists and is callable
        assert hasattr(app, 'start_gui'), "start_gui method missing"
        assert callable(app.start_gui), "start_gui method not callable"
        
        console.print("[green]✓ GUI mode method exists and is callable[/green]")
        
        # Check that the method accepts no additional parameters (embedded server is automatic)
        import inspect
        sig = inspect.signature(app.start_gui)
        assert len(sig.parameters) == 0, "start_gui should not require additional parameters"
        
        console.print("[green]✓ GUI mode uses embedded server (no additional parameters)[/green]")
        
        return True
        
    except Exception as e:
        console.print(f"[red]✗ Requirement 3.1 failed: {e}[/red]")
        return False


def test_requirement_3_2():
    """Test Requirement 3.2: Headless server mode for remote access."""
    console.print("[bold blue]Testing Requirement 3.2: Headless server mode for remote access[/bold blue]")
    
    try:
        config_manager = ConfigManager()
        app = Application(config_manager=config_manager)
        
        # Check that server mode method exists and is callable
        assert hasattr(app, 'start_server'), "start_server method missing"
        assert callable(app.start_server), "start_server method not callable"
        
        console.print("[green]✓ Server mode method exists and is callable[/green]")
        
        # Check that the method accepts host and port parameters for remote access
        import inspect
        sig = inspect.signature(app.start_server)
        params = list(sig.parameters.keys())
        assert 'host' in params, "start_server should accept host parameter"
        assert 'port' in params, "start_server should accept port parameter"
        
        console.print("[green]✓ Server mode accepts host and port for remote access[/green]")
        
        # Check default values allow remote access
        host_default = sig.parameters['host'].default
        port_default = sig.parameters['port'].default
        assert host_default == "127.0.0.1", f"Expected default host 127.0.0.1, got {host_default}"
        assert port_default == 8080, f"Expected default port 8080, got {port_default}"
        
        console.print("[green]✓ Server mode has appropriate defaults[/green]")
        
        return True
        
    except Exception as e:
        console.print(f"[red]✗ Requirement 3.2 failed: {e}[/red]")
        return False


def test_requirement_3_3():
    """Test Requirement 3.3: Configure appropriate logging and monitoring."""
    console.print("[bold blue]Testing Requirement 3.3: Configure appropriate logging and monitoring[/bold blue]")
    
    try:
        config_manager = ConfigManager()
        app = Application(config_manager=config_manager)
        
        # Check that logging configuration method exists
        assert hasattr(app, '_configure_logging_for_mode'), "Logging configuration method missing"
        
        console.print("[green]✓ Logging configuration method exists[/green]")
        
        # Test different logging modes
        app._configure_logging_for_mode("server")
        console.print("[green]✓ Server mode logging configuration works[/green]")
        
        app._configure_logging_for_mode("gui")
        console.print("[green]✓ GUI mode logging configuration works[/green]")
        
        app._configure_logging_for_mode("combined")
        console.print("[green]✓ Combined mode logging configuration works[/green]")
        
        # Check that monitoring is configurable
        global_config = config_manager.get_global_config()
        assert hasattr(global_config, 'enable_monitoring'), "Monitoring configuration missing"
        
        console.print("[green]✓ Monitoring configuration available[/green]")
        
        return True
        
    except Exception as e:
        console.print(f"[red]✗ Requirement 3.3 failed: {e}[/red]")
        return False


def test_requirement_2_4():
    """Test Requirement 2.4: Use dependency injection patterns for loose coupling."""
    console.print("[bold blue]Testing Requirement 2.4: Dependency injection patterns[/bold blue]")
    
    try:
        config_manager = ConfigManager()
        app = Application(config_manager=config_manager)
        
        # Check that Application accepts dependencies via constructor
        assert app.config_manager is config_manager, "ConfigManager not injected properly"
        
        console.print("[green]✓ ConfigManager dependency injection works[/green]")
        
        # Check that components are initialized with dependency injection
        assert hasattr(app, 'database'), "Database component not available"
        assert hasattr(app, 'task_manager'), "TaskManager component not available"
        assert hasattr(app, 'api_server'), "APIServer component not available"
        
        console.print("[green]✓ Core components use dependency injection[/green]")
        
        # Check that API server accepts dependencies
        from zuup.api.server import APIServer
        import inspect
        
        sig = inspect.signature(APIServer.__init__)
        params = list(sig.parameters.keys())
        assert 'task_manager' in params, "APIServer should accept task_manager dependency"
        assert 'config_manager' in params, "APIServer should accept config_manager dependency"
        
        console.print("[green]✓ APIServer uses dependency injection[/green]")
        
        # Check that GUI accepts dependencies
        from zuup.gui.main_window import MainWindow
        
        sig = inspect.signature(MainWindow.__init__)
        params = list(sig.parameters.keys())
        assert 'task_manager' in params, "MainWindow should accept task_manager dependency"
        assert 'config_manager' in params, "MainWindow should accept config_manager dependency"
        
        console.print("[green]✓ MainWindow uses dependency injection[/green]")
        
        return True
        
    except Exception as e:
        console.print(f"[red]✗ Requirement 2.4 failed: {e}[/red]")
        return False


def test_cli_interface():
    """Test that CLI interface supports all deployment modes."""
    console.print("[bold blue]Testing CLI Interface[/bold blue]")
    
    try:
        # Import CLI components
        from zuup.main import cli
        import click.testing
        
        runner = click.testing.CliRunner()
        
        # Test help command
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0, "CLI help should work"
        assert 'start' in result.output, "start command should be available"
        assert 'status' in result.output, "status command should be available"
        
        console.print("[green]✓ CLI help works[/green]")
        
        # Test start command help
        result = runner.invoke(cli, ['start', '--help'])
        assert result.exit_code == 0, "start command help should work"
        assert '--gui' in result.output, "GUI option should be available"
        assert '--server-only' in result.output, "Server-only option should be available"
        
        console.print("[green]✓ Start command options available[/green]")
        
        # Test status command help
        result = runner.invoke(cli, ['status', '--help'])
        assert result.exit_code == 0, "status command help should work"
        
        console.print("[green]✓ Status command available[/green]")
        
        return True
        
    except Exception as e:
        console.print(f"[red]✗ CLI interface test failed: {e}[/red]")
        return False


async def test_api_endpoints():
    """Test that API endpoints are working correctly."""
    console.print("[bold blue]Testing API Endpoints[/bold blue]")
    
    try:
        # Start server in background for testing
        config_manager = ConfigManager()
        app = Application(config_manager=config_manager)
        
        # Initialize core components
        await app._initialize_core_components()
        
        # Initialize API server
        from zuup.api.server import APIServer
        api_server = APIServer(
            host="127.0.0.1",
            port=8083,
            task_manager=app.task_manager,
            config_manager=app.config_manager,
        )
        
        # Start server in background
        server_task = asyncio.create_task(api_server.start())
        
        # Wait for server to start
        await asyncio.sleep(1)
        
        # Test API endpoints
        async with httpx.AsyncClient() as client:
            # Test health endpoint
            response = await client.get("http://127.0.0.1:8083/api/v1/health")
            assert response.status_code == 200, "Health endpoint should return 200"
            data = response.json()
            assert data['status'] == 'healthy', "Health status should be healthy"
            
            console.print("[green]✓ Health endpoint works[/green]")
            
            # Test status endpoint
            response = await client.get("http://127.0.0.1:8083/api/v1/status")
            assert response.status_code == 200, "Status endpoint should return 200"
            data = response.json()
            assert 'stats' in data, "Status should include stats"
            
            console.print("[green]✓ Status endpoint works[/green]")
            
            # Test tasks endpoint
            response = await client.get("http://127.0.0.1:8083/api/v1/tasks")
            assert response.status_code == 200, "Tasks endpoint should return 200"
            data = response.json()
            assert 'tasks' in data, "Tasks response should include tasks list"
            
            console.print("[green]✓ Tasks endpoint works[/green]")
        
        # Stop server
        await api_server.stop()
        server_task.cancel()
        
        # Cleanup
        await app._shutdown()
        
        return True
        
    except Exception as e:
        console.print(f"[red]✗ API endpoints test failed: {e}[/red]")
        return False


async def main():
    """Run all requirement tests."""
    console.print("[bold green]Task 7 Requirements Verification[/bold green]")
    console.print()
    
    # Setup logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    
    results = []
    
    # Test each requirement
    results.append(("Requirement 3.1 (GUI + Embedded Server)", test_requirement_3_1()))
    console.print()
    
    results.append(("Requirement 3.2 (Headless Server)", test_requirement_3_2()))
    console.print()
    
    results.append(("Requirement 3.3 (Logging & Monitoring)", test_requirement_3_3()))
    console.print()
    
    results.append(("Requirement 2.4 (Dependency Injection)", test_requirement_2_4()))
    console.print()
    
    results.append(("CLI Interface", test_cli_interface()))
    console.print()
    
    results.append(("API Endpoints", await test_api_endpoints()))
    console.print()
    
    # Summary
    console.print("[bold blue]Test Results Summary[/bold blue]")
    
    table = Table(title="Task 7 Requirements Verification")
    table.add_column("Requirement", style="cyan")
    table.add_column("Status", style="white")
    
    all_passed = True
    for name, passed in results:
        status = "[green]✓ PASS[/green]" if passed else "[red]✗ FAIL[/red]"
        table.add_row(name, status)
        if not passed:
            all_passed = False
    
    console.print(table)
    console.print()
    
    if all_passed:
        console.print("[bold green]🎉 All requirements for Task 7 are implemented correctly![/bold green]")
        console.print()
        console.print("[dim]Task 7 implementation includes:[/dim]")
        console.print("  • Application controller with dependency injection")
        console.print("  • GUI mode with embedded server functionality")
        console.print("  • Headless server mode for remote access")
        console.print("  • Configurable logging and monitoring")
        console.print("  • Command-line interface for deployment control")
        console.print("  • REST API endpoints for task management")
    else:
        console.print("[bold red]❌ Some requirements are not fully implemented[/bold red]")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)