#!/usr/bin/env python3
"""Interactive demo script for logging and monitoring features."""

import asyncio
import logging
import random
import time
from datetime import datetime
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from zuup.utils.logging import (
    setup_logging, 
    setup_debug_logging, 
    get_download_logger,
    log_system_info
)
from zuup.utils.monitoring import (
    initialize_monitoring, 
    PerformanceMonitor,
    get_metrics_collector
)
from zuup.utils.debugging import (
    start_debug_session, 
    end_debug_session,
    initialize_error_reporting,
    report_error,
    get_profiler
)
from zuup.storage.models import (
    DownloadTask, 
    ProgressInfo, 
    TaskStatus, 
    EngineType
)


class InteractiveDemo:
    """Interactive demonstration of logging and monitoring features."""
    
    def __init__(self):
        self.output_dir = Path.home() / "zuup_demo_logs"
        self.output_dir.mkdir(exist_ok=True)
        
        self.metrics = None
        self.monitor = None
        self.active_tasks = {}
        
        print(f"Demo output directory: {self.output_dir}")
    
    def setup_logging_demo(self):
        """Demonstrate logging setup options."""
        print("\n=== Logging Setup Demo ===")
        
        print("1. Setting up basic console logging...")
        setup_logging(level="INFO", rich_console=True)
        
        logger = logging.getLogger("demo.basic")
        logger.info("Basic console logging is now active")
        logger.warning("This is a warning message")
        logger.error("This is an error message")
        
        print("\n2. Setting up file logging with rotation...")
        setup_logging(
            level="DEBUG",
            log_file=self.output_dir / "demo.log",
            rich_console=True,
            structured_logging=False
        )
        
        logger.debug("Debug message (visible in file only)")
        logger.info("Info message (visible in console and file)")
        
        print("\n3. Setting up structured JSON logging...")
        setup_logging(
            level="INFO",
            log_file=self.output_dir / "structured_demo.log",
            structured_logging=True
        )
        
        logger.info("Structured log entry", extra={
            'task_id': 'demo-123',
            'download_speed': 1024000,
            'progress_percentage': 75.5,
            'engine_type': 'http'
        })
        
        print("\n4. Setting up debug logging for development...")
        setup_debug_logging(self.output_dir)
        log_system_info()
        
        print("✓ Logging setup complete. Check files in:", self.output_dir)
    
    def demonstrate_download_logging(self):
        """Demonstrate download-specific logging."""
        print("\n=== Download Logging Demo ===")
        
        # Create sample download task
        task = DownloadTask(
            url="https://example.com/large-file.zip",
            destination=Path("/tmp/large-file.zip"),
            engine_type=EngineType.HTTP,
            file_size=100 * 1024 * 1024  # 100MB
        )
        
        logger = get_download_logger(task.id, task.engine_type)
        
        print(f"Created download logger for task: {task.id}")
        
        # Simulate download progress
        total_size = 100 * 1024 * 1024
        downloaded = 0
        
        print("Simulating download progress...")
        for i in range(10):
            downloaded += 10 * 1024 * 1024  # 10MB chunks
            speed = random.uniform(1024*1024, 5*1024*1024)  # 1-5 MB/s
            
            progress = ProgressInfo(
                downloaded_bytes=downloaded,
                total_bytes=total_size,
                download_speed=speed,
                status=TaskStatus.DOWNLOADING
            )
            
            logger.log_progress(progress, task.url)
            time.sleep(0.5)  # Pause for demonstration
        
        # Simulate completion
        logger.log_completion(total_size, 5.0, task.url)
        
        print("✓ Download logging demonstration complete")
    
    def setup_monitoring_demo(self):
        """Demonstrate monitoring system setup."""
        print("\n=== Monitoring Setup Demo ===")
        
        # Initialize monitoring
        self.metrics = initialize_monitoring(history_size=500)
        print("✓ Metrics collector initialized")
        
        # Initialize error reporting
        initialize_error_reporting(self.output_dir)
        print("✓ Error reporting initialized")
        
        # Start debug session
        start_debug_session("interactive_demo")
        print("✓ Debug session started")
        
        print("Monitoring system is now active")
    
    async def demonstrate_live_monitoring(self):
        """Demonstrate live monitoring of multiple downloads."""
        print("\n=== Live Monitoring Demo ===")
        
        if not self.metrics:
            print("Please run setup_monitoring_demo() first")
            return
        
        # Start performance monitor
        self.monitor = PerformanceMonitor(self.metrics, interval=2.0)
        await self.monitor.start()
        print("✓ Performance monitor started")
        
        # Create multiple download tasks
        tasks = [
            DownloadTask(
                url=f"https://example.com/file{i}.zip",
                destination=Path(f"/tmp/file{i}.zip"),
                engine_type=random.choice(list(EngineType)),
                file_size=random.randint(10, 100) * 1024 * 1024
            )
            for i in range(3)
        ]
        
        print(f"Created {len(tasks)} download tasks")
        
        # Start monitoring tasks
        for task in tasks:
            self.metrics.start_task_monitoring(task)
            self.active_tasks[task.id] = task
        
        # Simulate concurrent downloads
        print("Simulating concurrent downloads for 15 seconds...")
        start_time = time.time()
        
        while time.time() - start_time < 15:
            for task in tasks:
                if task.id not in self.active_tasks:
                    continue
                
                # Simulate progress
                elapsed = time.time() - start_time
                progress_ratio = min(elapsed / 10.0, 1.0)  # Complete in 10 seconds
                
                downloaded = int(task.file_size * progress_ratio)
                speed = random.uniform(512*1024, 3*1024*1024)  # 0.5-3 MB/s
                
                progress = ProgressInfo(
                    downloaded_bytes=downloaded,
                    total_bytes=task.file_size,
                    download_speed=speed,
                    status=TaskStatus.DOWNLOADING if progress_ratio < 1.0 else TaskStatus.COMPLETED
                )
                
                # Add torrent-specific data for torrent tasks
                if task.engine_type == EngineType.TORRENT:
                    progress.upload_speed = random.uniform(100*1024, 1024*1024)
                    progress.peers_connected = random.randint(5, 25)
                    progress.ratio = random.uniform(0.1, 2.0)
                
                self.metrics.update_task_progress(task.id, progress)
                
                # Simulate occasional errors
                if random.random() < 0.05:  # 5% chance
                    error_msg = random.choice([
                        "Connection timeout",
                        "Server returned 503",
                        "Network unreachable"
                    ])
                    self.metrics.update_task_error(task.id, error_msg)
                    report_error(Exception(error_msg), "simulation", task.id)
                
                # Complete task if done
                if progress_ratio >= 1.0:
                    self.metrics.complete_task_monitoring(task.id, TaskStatus.COMPLETED)
                    if task.id in self.active_tasks:
                        del self.active_tasks[task.id]
            
            # Collect and display system metrics
            system_metrics = self.metrics.collect_system_metrics()
            print(f"Active: {system_metrics.active_downloads}, "
                  f"Total Speed: {system_metrics.total_download_speed/1024/1024:.1f} MB/s")
            
            await asyncio.sleep(1)
        
        # Stop monitoring
        await self.monitor.stop()
        print("✓ Live monitoring demonstration complete")
    
    def demonstrate_debugging_features(self):
        """Demonstrate debugging and error reporting features."""
        print("\n=== Debugging Features Demo ===")
        
        profiler = get_profiler()
        
        # Demonstrate function profiling
        print("1. Profiling operations...")
        
        with profiler.profile("file_operation"):
            time.sleep(0.1)  # Simulate file I/O
        
        with profiler.profile("network_operation"):
            time.sleep(0.2)  # Simulate network call
        
        with profiler.profile("file_operation"):
            time.sleep(0.05)  # Another file operation
        
        # Show profiling results
        stats = profiler.get_all_stats()
        print("Profiling results:")
        for operation, data in stats.items():
            print(f"  {operation}: {data['count']} calls, "
                  f"avg {data['average_time']*1000:.1f}ms")
        
        # Demonstrate error reporting
        print("\n2. Error reporting...")
        
        try:
            # Simulate various types of errors
            raise ConnectionError("Failed to connect to server")
        except Exception as e:
            report_error(e, "network_demo", additional_data={
                'server': 'example.com',
                'port': 443,
                'retry_count': 3
            })
        
        try:
            raise ValueError("Invalid configuration parameter")
        except Exception as e:
            report_error(e, "config_demo")
        
        print("✓ Debugging features demonstration complete")
    
    def export_demo_data(self):
        """Export all collected demo data."""
        print("\n=== Exporting Demo Data ===")
        
        if self.metrics:
            # Export metrics
            metrics_file = self.output_dir / "demo_metrics.json"
            self.metrics.export_metrics(metrics_file)
            print(f"✓ Metrics exported to: {metrics_file}")
        
        # End debug session
        session_file = end_debug_session(self.output_dir)
        if session_file:
            print(f"✓ Debug session saved to: {session_file}")
        
        # List all generated files
        print("\nGenerated files:")
        for file_path in sorted(self.output_dir.glob("*")):
            size = file_path.stat().st_size
            print(f"  {file_path.name} ({size} bytes)")
        
        print(f"\nAll demo data saved to: {self.output_dir}")
    
    def interactive_menu(self):
        """Show interactive menu for demo features."""
        while True:
            print("\n" + "="*50)
            print("Zuup Logging & Monitoring Interactive Demo")
            print("="*50)
            print("1. Logging Setup Demo")
            print("2. Download Logging Demo")
            print("3. Monitoring Setup Demo")
            print("4. Live Monitoring Demo (async)")
            print("5. Debugging Features Demo")
            print("6. Export Demo Data")
            print("7. View Log Files")
            print("0. Exit")
            
            try:
                choice = input("\nSelect option (0-7): ").strip()
                
                if choice == "0":
                    break
                elif choice == "1":
                    self.setup_logging_demo()
                elif choice == "2":
                    self.demonstrate_download_logging()
                elif choice == "3":
                    self.setup_monitoring_demo()
                elif choice == "4":
                    asyncio.run(self.demonstrate_live_monitoring())
                elif choice == "5":
                    self.demonstrate_debugging_features()
                elif choice == "6":
                    self.export_demo_data()
                elif choice == "7":
                    self.view_log_files()
                else:
                    print("Invalid option. Please try again.")
                
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                input("Press Enter to continue...")
    
    def view_log_files(self):
        """View contents of generated log files."""
        print("\n=== Log Files Viewer ===")
        
        log_files = list(self.output_dir.glob("*.log"))
        
        if not log_files:
            print("No log files found. Run some demos first.")
            return
        
        print("Available log files:")
        for i, file_path in enumerate(log_files, 1):
            size = file_path.stat().st_size
            print(f"{i}. {file_path.name} ({size} bytes)")
        
        try:
            choice = input(f"\nSelect file to view (1-{len(log_files)}): ").strip()
            file_index = int(choice) - 1
            
            if 0 <= file_index < len(log_files):
                file_path = log_files[file_index]
                print(f"\n--- Contents of {file_path.name} ---")
                
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    
                # Show last 20 lines
                for line in lines[-20:]:
                    print(line.rstrip())
                
                if len(lines) > 20:
                    print(f"\n(Showing last 20 lines of {len(lines)} total)")
            else:
                print("Invalid selection.")
                
        except (ValueError, IndexError):
            print("Invalid input.")


def main():
    """Main entry point for interactive demo."""
    print("Welcome to the Zuup Logging & Monitoring Interactive Demo!")
    print("This demo will show you all the logging and monitoring features.")
    
    demo = InteractiveDemo()
    
    try:
        demo.interactive_menu()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    finally:
        # Cleanup
        if demo.monitor:
            asyncio.run(demo.monitor.stop())
        
        print("Thank you for trying the demo!")
        print(f"Demo files are saved in: {demo.output_dir}")


if __name__ == "__main__":
    main()