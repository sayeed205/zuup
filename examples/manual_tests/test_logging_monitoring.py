#!/usr/bin/env python3
"""Manual testing script for logging and monitoring infrastructure."""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys
import tempfile

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from zuup.utils.logging import (
    setup_logging, 
    setup_debug_logging, 
    get_download_logger, 
    log_system_info,
    LogCapture,
    StructuredFormatter
)
from zuup.utils.monitoring import (
    MetricsCollector, 
    PerformanceMonitor, 
    initialize_monitoring,
    get_metrics_collector
)
from zuup.utils.debugging import (
    start_debug_session, 
    end_debug_session, 
    debug_trace, 
    debug_context,
    TaskDebugger,
    initialize_error_reporting,
    report_error,
    get_profiler
)
from zuup.storage.models import (
    DownloadTask, 
    ProgressInfo, 
    TaskStatus, 
    EngineType, 
    TaskConfig
)


class TestDownloadSimulator:
    """Simulates download behavior for testing monitoring."""
    
    def __init__(self, task: DownloadTask, metrics_collector: MetricsCollector):
        self.task = task
        self.metrics_collector = metrics_collector
        self.logger = get_download_logger(task.id, task.engine_type)
        self.debugger = TaskDebugger(task)
        self.running = False
    
    @debug_trace
    async def simulate_download(self, duration: float = 10.0, 
                              simulate_errors: bool = False) -> None:
        """Simulate a download with progress updates."""
        self.running = True
        self.metrics_collector.start_task_monitoring(self.task)
        self.debugger.log_event("download_started")
        
        total_bytes = 10 * 1024 * 1024  # 10MB
        downloaded = 0
        start_time = time.time()
        
        try:
            while self.running and downloaded < total_bytes:
                # Simulate download progress
                chunk_size = min(1024 * 100, total_bytes - downloaded)  # 100KB chunks
                downloaded += chunk_size
                
                elapsed = time.time() - start_time
                speed = downloaded / elapsed if elapsed > 0 else 0
                
                # Create progress info
                progress = ProgressInfo(
                    downloaded_bytes=downloaded,
                    total_bytes=total_bytes,
                    download_speed=speed,
                    status=TaskStatus.DOWNLOADING
                )
                
                # Update task and metrics
                self.task.update_progress(progress)
                self.metrics_collector.update_task_progress(self.task.id, progress)
                self.debugger.log_progress_update(progress)
                
                # Log progress
                self.logger.log_progress(progress, self.task.url)
                
                # Simulate error occasionally
                if simulate_errors and downloaded > total_bytes * 0.3 and downloaded < total_bytes * 0.4:
                    error = ConnectionError("Simulated network error")
                    self.debugger.log_error(error, "network_simulation")
                    self.metrics_collector.update_task_error(self.task.id, str(error))
                    report_error(error, "download_simulation", self.task.id)
                    
                    # Simulate retry
                    await asyncio.sleep(1)
                    self.debugger.log_retry(1, "network_error_recovery")
                    self.metrics_collector.update_task_retry(self.task.id)
                
                await asyncio.sleep(0.1)  # Simulate download time
                
                if elapsed >= duration:
                    break
            
            # Mark as completed
            if downloaded >= total_bytes:
                self.task.mark_completed()
                self.debugger.log_completion(TaskStatus.COMPLETED)
                self.logger.log_completion(downloaded, time.time() - start_time, self.task.url)
            
            self.metrics_collector.complete_task_monitoring(self.task.id, self.task.status)
            
        except Exception as e:
            self.task.mark_failed(str(e))
            self.debugger.log_error(e, "download_simulation")
            self.metrics_collector.update_task_error(self.task.id, str(e))
            report_error(e, "download_simulation", self.task.id)
            raise
        finally:
            self.running = False


def test_basic_logging():
    """Test basic logging functionality."""
    print("\n=== Testing Basic Logging ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        log_dir = Path(temp_dir)
        
        # Test standard logging
        setup_logging(
            level="DEBUG",
            log_file=log_dir / "test.log",
            rich_console=True
        )
        
        logger = logging.getLogger("test.basic")
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        # Test structured logging
        setup_logging(
            level="INFO",
            log_file=log_dir / "structured.log",
            structured_logging=True
        )
        
        logger.info("Structured log message", extra={
            'task_id': 'test-123',
            'download_speed': 1024000,
            'progress_percentage': 45.5
        })
        
        # Test log capture
        with LogCapture("test.capture") as capture:
            test_logger = logging.getLogger("test.capture")
            test_logger.info("Captured message 1")
            test_logger.warning("Captured message 2")
        
        messages = capture.get_messages()
        print(f"Captured {len(messages)} messages: {messages}")
        
        # Verify log files exist
        assert (log_dir / "test.log").exists(), "Standard log file not created"
        assert (log_dir / "structured.log").exists(), "Structured log file not created"
        
        print("✓ Basic logging tests passed")


def test_download_logger():
    """Test download-specific logger."""
    print("\n=== Testing Download Logger ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        setup_logging(log_file=Path(temp_dir) / "download.log")
        
        # Create test task
        task = DownloadTask(
            url="https://example.com/file.zip",
            destination=Path("/tmp/file.zip"),
            engine_type=EngineType.HTTP
        )
        
        logger = get_download_logger(task.id, task.engine_type)
        
        # Test progress logging
        progress = ProgressInfo(
            downloaded_bytes=5 * 1024 * 1024,
            total_bytes=10 * 1024 * 1024,
            download_speed=1024 * 1024,  # 1 MB/s
            status=TaskStatus.DOWNLOADING
        )
        
        logger.log_progress(progress, task.url)
        
        # Test error logging
        try:
            raise ValueError("Test error")
        except ValueError as e:
            logger.log_error(e, "TEST_ERROR", task.url)
        
        # Test completion logging
        logger.log_completion(10 * 1024 * 1024, 10.5, task.url)
        
        print("✓ Download logger tests passed")


def test_metrics_collection():
    """Test metrics collection system."""
    print("\n=== Testing Metrics Collection ===")
    
    metrics = initialize_monitoring()
    
    # Create test tasks
    tasks = [
        DownloadTask(
            url="https://example.com/file1.zip",
            destination=Path("/tmp/file1.zip"),
            engine_type=EngineType.HTTP,
            file_size=10 * 1024 * 1024
        ),
        DownloadTask(
            url="magnet:?xt=urn:btih:test",
            destination=Path("/tmp/torrent"),
            engine_type=EngineType.TORRENT,
            file_size=100 * 1024 * 1024
        )
    ]
    
    # Start monitoring tasks
    for task in tasks:
        metrics.start_task_monitoring(task)
    
    # Simulate progress updates
    for i in range(5):
        for task in tasks:
            progress = ProgressInfo(
                downloaded_bytes=i * 2 * 1024 * 1024,
                total_bytes=task.file_size,
                download_speed=(i + 1) * 1024 * 1024,
                status=TaskStatus.DOWNLOADING
            )
            
            if task.engine_type == EngineType.TORRENT:
                progress.upload_speed = i * 512 * 1024
                progress.peers_connected = 10 + i
                progress.ratio = 0.5 + (i * 0.1)
            
            metrics.update_task_progress(task.id, progress)
    
    # Test error reporting
    metrics.update_task_error(tasks[0].id, "Connection timeout")
    metrics.update_task_retry(tasks[0].id)
    
    # Complete tasks
    for task in tasks:
        metrics.complete_task_monitoring(task.id, TaskStatus.COMPLETED)
    
    # Collect system metrics
    system_metrics = metrics.collect_system_metrics()
    print(f"System metrics: {system_metrics.to_dict()}")
    
    # Test metrics export
    with tempfile.TemporaryDirectory() as temp_dir:
        export_file = Path(temp_dir) / "metrics.json"
        metrics.export_metrics(export_file)
        assert export_file.exists(), "Metrics export file not created"
    
    print("✓ Metrics collection tests passed")


async def test_performance_monitoring():
    """Test performance monitoring system."""
    print("\n=== Testing Performance Monitoring ===")
    
    metrics = get_metrics_collector()
    monitor = PerformanceMonitor(metrics, interval=1.0)
    
    # Start monitoring
    await monitor.start()
    
    # Create and run simulated downloads
    tasks = [
        DownloadTask(
            url=f"https://example.com/file{i}.zip",
            destination=Path(f"/tmp/file{i}.zip"),
            engine_type=EngineType.HTTP,
            file_size=5 * 1024 * 1024
        )
        for i in range(3)
    ]
    
    simulators = [
        TestDownloadSimulator(task, metrics)
        for task in tasks
    ]
    
    # Run downloads concurrently
    download_tasks = [
        asyncio.create_task(sim.simulate_download(duration=3.0, simulate_errors=(i == 1)))
        for i, sim in enumerate(simulators)
    ]
    
    await asyncio.gather(*download_tasks)
    
    # Stop monitoring
    await monitor.stop()
    
    print("✓ Performance monitoring tests passed")


def test_debugging_system():
    """Test debugging and error reporting system."""
    print("\n=== Testing Debugging System ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        debug_dir = Path(temp_dir)
        
        # Initialize error reporting
        initialize_error_reporting(debug_dir)
        
        # Start debug session
        session = start_debug_session("test_session")
        
        @debug_trace
        def test_function(x: int, y: int) -> int:
            if x < 0:
                raise ValueError("x must be positive")
            return x + y
        
        @debug_trace
        async def async_test_function(delay: float) -> str:
            await asyncio.sleep(delay)
            return f"Completed after {delay}s"
        
        # Test function tracing
        result = test_function(5, 3)
        assert result == 8
        
        # Test async function tracing
        async def run_async_test():
            return await async_test_function(0.1)
        
        result = asyncio.run(run_async_test())
        assert "Completed after 0.1s" in result
        
        # Test error handling
        try:
            test_function(-1, 5)
        except ValueError as e:
            report_error(e, "test_function_validation")
        
        # Test debug context
        with debug_context("test_context"):
            time.sleep(0.1)  # Simulate some work
        
        # Test task debugger
        task = DownloadTask(
            url="https://example.com/test.zip",
            destination=Path("/tmp/test.zip"),
            engine_type=EngineType.HTTP
        )
        
        debugger = TaskDebugger(task)
        debugger.log_event("test_event", {"test_data": "value"})
        
        progress = ProgressInfo(
            downloaded_bytes=1024,
            total_bytes=2048,
            download_speed=512,
            status=TaskStatus.DOWNLOADING
        )
        debugger.log_progress_update(progress)
        
        # Save debug data
        session_file = end_debug_session(debug_dir)
        task_file = debugger.save_debug_log(debug_dir)
        
        assert session_file and session_file.exists(), "Debug session file not created"
        assert task_file.exists(), "Task debug file not created"
        
        print("✓ Debugging system tests passed")


def test_profiler():
    """Test performance profiler."""
    print("\n=== Testing Performance Profiler ===")
    
    profiler = get_profiler()
    profiler.reset()
    
    # Profile some operations
    with profiler.profile("test_operation_1"):
        time.sleep(0.1)
    
    with profiler.profile("test_operation_2"):
        time.sleep(0.05)
    
    with profiler.profile("test_operation_1"):
        time.sleep(0.08)
    
    # Get statistics
    stats = profiler.get_all_stats()
    print(f"Profiler stats: {stats}")
    
    assert "test_operation_1" in stats
    assert "test_operation_2" in stats
    assert stats["test_operation_1"]["count"] == 2
    assert stats["test_operation_2"]["count"] == 1
    
    print("✓ Performance profiler tests passed")


async def run_comprehensive_test():
    """Run comprehensive test of all logging and monitoring features."""
    print("\n=== Running Comprehensive Test ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        log_dir = Path(temp_dir)
        
        # Setup comprehensive logging
        setup_debug_logging(log_dir)
        
        # Initialize monitoring and debugging
        metrics = initialize_monitoring()
        initialize_error_reporting(log_dir)
        
        # Start debug session
        session = start_debug_session("comprehensive_test")
        
        # Start performance monitoring
        monitor = PerformanceMonitor(metrics, interval=0.5)
        await monitor.start()
        
        try:
            # Create multiple download tasks
            tasks = []
            for i in range(4):
                if i % 2 == 0:
                    # HTTP tasks
                    task = DownloadTask(
                        url=f"https://example.com/file{i}.zip",
                        destination=Path(f"/tmp/file{i}.zip"),
                        engine_type=EngineType.HTTP,
                        file_size=(i + 1) * 5 * 1024 * 1024
                    )
                else:
                    # Torrent tasks - use magnet URLs
                    task = DownloadTask(
                        url=f"magnet:?xt=urn:btih:{'a' * 40}&dn=file{i}",
                        destination=Path(f"/tmp/torrent{i}"),
                        engine_type=EngineType.TORRENT,
                        file_size=(i + 1) * 5 * 1024 * 1024
                    )
                tasks.append(task)
            
            # Create simulators
            simulators = [
                TestDownloadSimulator(task, metrics)
                for task in tasks
            ]
            
            # Run downloads with different behaviors
            download_tasks = []
            for i, sim in enumerate(simulators):
                # Simulate errors in some downloads
                simulate_errors = i in [1, 3]
                duration = 2.0 + (i * 0.5)
                
                task = asyncio.create_task(
                    sim.simulate_download(duration=duration, simulate_errors=simulate_errors)
                )
                download_tasks.append(task)
            
            # Wait for all downloads
            await asyncio.gather(*download_tasks, return_exceptions=True)
            
            # Collect final metrics
            final_metrics = metrics.collect_system_metrics()
            print(f"Final system metrics: {final_metrics.to_dict()}")
            
            # Export all data
            metrics.export_metrics(log_dir / "final_metrics.json")
            
        finally:
            # Cleanup
            await monitor.stop()
            end_debug_session(log_dir)
        
        # Verify all files were created
        expected_files = [
            "zuup_debug.log",
            "final_metrics.json",
            "comprehensive_test.json"
        ]
        
        for filename in expected_files:
            file_path = log_dir / filename
            if file_path.exists():
                print(f"✓ Created {filename} ({file_path.stat().st_size} bytes)")
            else:
                print(f"✗ Missing {filename}")
        
        print("✓ Comprehensive test completed")


def main():
    """Run all manual tests."""
    print("Starting manual tests for logging and monitoring infrastructure...")
    
    try:
        # Run synchronous tests
        test_basic_logging()
        test_download_logger()
        test_metrics_collection()
        test_debugging_system()
        test_profiler()
        
        # Run asynchronous tests
        asyncio.run(test_performance_monitoring())
        asyncio.run(run_comprehensive_test())
        
        print("\n🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())