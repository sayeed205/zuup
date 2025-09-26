"""Monitoring and metrics collection system for download manager."""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
import json
import threading
import logging

from ..storage.models import ProgressInfo, TaskStatus, EngineType, DownloadTask


@dataclass
class DownloadMetrics:
    """Metrics for a single download task."""
    
    task_id: str
    engine_type: EngineType
    url: str
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Size metrics
    total_bytes: Optional[int] = None
    downloaded_bytes: int = 0
    uploaded_bytes: int = 0  # For torrents
    
    # Speed metrics (bytes per second)
    current_download_speed: float = 0.0
    current_upload_speed: float = 0.0
    peak_download_speed: float = 0.0
    peak_upload_speed: float = 0.0
    average_download_speed: float = 0.0
    average_upload_speed: float = 0.0
    
    # Connection metrics
    active_connections: int = 0
    max_connections: int = 0
    
    # Torrent-specific metrics
    peers_connected: int = 0
    seeds_connected: int = 0
    share_ratio: float = 0.0
    
    # Error metrics
    retry_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    
    # Performance metrics
    cpu_usage: float = 0.0
    memory_usage: int = 0  # bytes
    
    def duration(self) -> timedelta:
        """Calculate download duration."""
        end = self.end_time or datetime.now()
        return end - self.start_time
    
    def completion_percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total_bytes and self.total_bytes > 0:
            return (self.downloaded_bytes / self.total_bytes) * 100
        return 0.0
    
    def eta(self) -> Optional[timedelta]:
        """Estimate time to completion."""
        if (self.total_bytes and 
            self.current_download_speed > 0 and 
            self.downloaded_bytes < self.total_bytes):
            remaining_bytes = self.total_bytes - self.downloaded_bytes
            seconds_remaining = remaining_bytes / self.current_download_speed
            return timedelta(seconds=seconds_remaining)
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            'task_id': self.task_id,
            'engine_type': self.engine_type.value,
            'url': self.url,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration().total_seconds(),
            'total_bytes': self.total_bytes,
            'downloaded_bytes': self.downloaded_bytes,
            'uploaded_bytes': self.uploaded_bytes,
            'completion_percentage': self.completion_percentage(),
            'current_download_speed': self.current_download_speed,
            'current_upload_speed': self.current_upload_speed,
            'peak_download_speed': self.peak_download_speed,
            'peak_upload_speed': self.peak_upload_speed,
            'average_download_speed': self.average_download_speed,
            'average_upload_speed': self.average_upload_speed,
            'active_connections': self.active_connections,
            'max_connections': self.max_connections,
            'peers_connected': self.peers_connected,
            'seeds_connected': self.seeds_connected,
            'share_ratio': self.share_ratio,
            'retry_count': self.retry_count,
            'error_count': self.error_count,
            'last_error': self.last_error,
            'eta_seconds': self.eta().total_seconds() if self.eta() else None,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
        }


@dataclass
class SystemMetrics:
    """System-wide metrics."""
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Download statistics
    active_downloads: int = 0
    completed_downloads: int = 0
    failed_downloads: int = 0
    total_downloads: int = 0
    
    # Speed statistics (bytes per second)
    total_download_speed: float = 0.0
    total_upload_speed: float = 0.0
    
    # Resource usage
    cpu_usage: float = 0.0
    memory_usage: int = 0  # bytes
    disk_usage: int = 0  # bytes
    network_usage: Dict[str, float] = field(default_factory=dict)  # interface -> bytes/sec
    
    # Engine statistics
    engine_stats: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert system metrics to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'active_downloads': self.active_downloads,
            'completed_downloads': self.completed_downloads,
            'failed_downloads': self.failed_downloads,
            'total_downloads': self.total_downloads,
            'total_download_speed': self.total_download_speed,
            'total_upload_speed': self.total_upload_speed,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'disk_usage': self.disk_usage,
            'network_usage': dict(self.network_usage),
            'engine_stats': {k: dict(v) for k, v in self.engine_stats.items()},
        }


class MetricsCollector:
    """Collects and manages download and system metrics."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.download_metrics: Dict[str, DownloadMetrics] = {}
        self.system_metrics_history: deque[SystemMetrics] = deque(maxlen=history_size)
        self.speed_history: Dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=100))
        
        self._lock = threading.RLock()
        self._callbacks: List[Callable[[str, DownloadMetrics], None]] = []
        self._system_callbacks: List[Callable[[SystemMetrics], None]] = []
        
        self.logger = logging.getLogger("zuup.monitoring")
    
    def start_task_monitoring(self, task: DownloadTask) -> None:
        """Start monitoring a download task."""
        with self._lock:
            metrics = DownloadMetrics(
                task_id=task.id,
                engine_type=task.engine_type,
                url=task.url,
                start_time=datetime.now(),
                total_bytes=task.file_size
            )
            self.download_metrics[task.id] = metrics
            
            self.logger.info(f"Started monitoring task {task.id} ({task.engine_type.value})")
    
    def update_task_progress(self, task_id: str, progress: ProgressInfo) -> None:
        """Update progress metrics for a task."""
        with self._lock:
            if task_id not in self.download_metrics:
                self.logger.warning(f"Metrics not found for task {task_id}")
                return
            
            metrics = self.download_metrics[task_id]
            
            # Update basic metrics
            metrics.downloaded_bytes = progress.downloaded_bytes
            metrics.total_bytes = progress.total_bytes or metrics.total_bytes
            metrics.current_download_speed = progress.download_speed
            
            # Update peak speeds
            if progress.download_speed > metrics.peak_download_speed:
                metrics.peak_download_speed = progress.download_speed
            
            # Update torrent-specific metrics
            if progress.upload_speed is not None:
                metrics.current_upload_speed = progress.upload_speed
                if progress.upload_speed > metrics.peak_upload_speed:
                    metrics.peak_upload_speed = progress.upload_speed
            
            if progress.peers_connected is not None:
                metrics.peers_connected = progress.peers_connected
            
            if progress.seeds_connected is not None:
                metrics.seeds_connected = progress.seeds_connected
            
            if progress.ratio is not None:
                metrics.share_ratio = progress.ratio
            
            # Calculate average speeds
            duration = metrics.duration().total_seconds()
            if duration > 0:
                metrics.average_download_speed = metrics.downloaded_bytes / duration
                if metrics.uploaded_bytes > 0:
                    metrics.average_upload_speed = metrics.uploaded_bytes / duration
            
            # Store speed history for trend analysis
            self.speed_history[task_id].append(progress.download_speed)
            
            # Notify callbacks
            for callback in self._callbacks:
                try:
                    callback(task_id, metrics)
                except Exception as e:
                    self.logger.error(f"Error in metrics callback: {e}")
    
    def update_task_error(self, task_id: str, error: str) -> None:
        """Update error metrics for a task."""
        with self._lock:
            if task_id in self.download_metrics:
                metrics = self.download_metrics[task_id]
                metrics.error_count += 1
                metrics.last_error = error
                
                self.logger.warning(f"Task {task_id} error: {error}")
    
    def update_task_retry(self, task_id: str) -> None:
        """Update retry count for a task."""
        with self._lock:
            if task_id in self.download_metrics:
                metrics = self.download_metrics[task_id]
                metrics.retry_count += 1
                
                self.logger.info(f"Task {task_id} retry #{metrics.retry_count}")
    
    def complete_task_monitoring(self, task_id: str, status: TaskStatus) -> None:
        """Complete monitoring for a task."""
        with self._lock:
            if task_id in self.download_metrics:
                metrics = self.download_metrics[task_id]
                metrics.end_time = datetime.now()
                
                self.logger.info(
                    f"Completed monitoring task {task_id}: {status.value} "
                    f"({metrics.downloaded_bytes} bytes in {metrics.duration()})"
                )
    
    def get_task_metrics(self, task_id: str) -> Optional[DownloadMetrics]:
        """Get metrics for a specific task."""
        with self._lock:
            return self.download_metrics.get(task_id)
    
    def get_all_task_metrics(self) -> Dict[str, DownloadMetrics]:
        """Get metrics for all tasks."""
        with self._lock:
            return self.download_metrics.copy()
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            import psutil
        except ImportError:
            self.logger.warning("psutil not available, system metrics will be limited")
            psutil = None
        
        with self._lock:
            metrics = SystemMetrics()
            
            # Count downloads by status
            for task_metrics in self.download_metrics.values():
                if task_metrics.end_time is None:
                    metrics.active_downloads += 1
                elif task_metrics.error_count > 0:
                    metrics.failed_downloads += 1
                else:
                    metrics.completed_downloads += 1
            
            metrics.total_downloads = len(self.download_metrics)
            
            # Calculate total speeds
            for task_metrics in self.download_metrics.values():
                if task_metrics.end_time is None:  # Active downloads only
                    metrics.total_download_speed += task_metrics.current_download_speed
                    metrics.total_upload_speed += task_metrics.current_upload_speed
            
            # Collect system resource metrics
            if psutil:
                metrics.cpu_usage = psutil.cpu_percent()
                metrics.memory_usage = psutil.virtual_memory().used
                metrics.disk_usage = psutil.disk_usage('/').used
                
                # Network usage (simplified)
                net_io = psutil.net_io_counters()
                metrics.network_usage = {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv,
                }
            
            # Engine statistics
            for task_metrics in self.download_metrics.values():
                engine = task_metrics.engine_type.value
                if task_metrics.end_time is None:
                    metrics.engine_stats[engine]['active'] += 1
                elif task_metrics.error_count > 0:
                    metrics.engine_stats[engine]['failed'] += 1
                else:
                    metrics.engine_stats[engine]['completed'] += 1
            
            # Store in history
            self.system_metrics_history.append(metrics)
            
            # Notify system callbacks
            for callback in self._system_callbacks:
                try:
                    callback(metrics)
                except Exception as e:
                    self.logger.error(f"Error in system metrics callback: {e}")
            
            return metrics
    
    def get_speed_trend(self, task_id: str, window_size: int = 10) -> List[float]:
        """Get recent speed trend for a task."""
        with self._lock:
            history = self.speed_history.get(task_id, deque())
            return list(history)[-window_size:]
    
    def get_system_metrics_history(self, duration: timedelta) -> List[SystemMetrics]:
        """Get system metrics history for a given duration."""
        cutoff_time = datetime.now() - duration
        with self._lock:
            return [
                metrics for metrics in self.system_metrics_history
                if metrics.timestamp >= cutoff_time
            ]
    
    def export_metrics(self, file_path: Path) -> None:
        """Export all metrics to a JSON file."""
        with self._lock:
            data = {
                'export_time': datetime.now().isoformat(),
                'download_metrics': {
                    task_id: metrics.to_dict()
                    for task_id, metrics in self.download_metrics.items()
                },
                'system_metrics_history': [
                    metrics.to_dict() for metrics in self.system_metrics_history
                ],
                'speed_history': {
                    task_id: list(speeds)
                    for task_id, speeds in self.speed_history.items()
                }
            }
            
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"Exported metrics to {file_path}")
    
    def add_callback(self, callback: Callable[[str, DownloadMetrics], None]) -> None:
        """Add a callback for task metrics updates."""
        self._callbacks.append(callback)
    
    def add_system_callback(self, callback: Callable[[SystemMetrics], None]) -> None:
        """Add a callback for system metrics updates."""
        self._system_callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[str, DownloadMetrics], None]) -> None:
        """Remove a task metrics callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def remove_system_callback(self, callback: Callable[[SystemMetrics], None]) -> None:
        """Remove a system metrics callback."""
        if callback in self._system_callbacks:
            self._system_callbacks.remove(callback)
    
    def clear_completed_tasks(self, older_than: timedelta = timedelta(hours=24)) -> None:
        """Clear metrics for completed tasks older than specified time."""
        cutoff_time = datetime.now() - older_than
        
        with self._lock:
            to_remove = []
            for task_id, metrics in self.download_metrics.items():
                if (metrics.end_time and 
                    metrics.end_time < cutoff_time and 
                    metrics.error_count == 0):
                    to_remove.append(task_id)
            
            for task_id in to_remove:
                del self.download_metrics[task_id]
                if task_id in self.speed_history:
                    del self.speed_history[task_id]
            
            if to_remove:
                self.logger.info(f"Cleared metrics for {len(to_remove)} completed tasks")


class PerformanceMonitor:
    """Monitors system performance during downloads."""
    
    def __init__(self, metrics_collector: MetricsCollector, interval: float = 5.0):
        self.metrics_collector = metrics_collector
        self.interval = interval
        self.running = False
        self._task: Optional[asyncio.Task] = None
        
        self.logger = logging.getLogger("zuup.performance")
    
    async def start(self) -> None:
        """Start performance monitoring."""
        if self.running:
            return
        
        self.running = True
        self._task = asyncio.create_task(self._monitor_loop())
        self.logger.info("Performance monitoring started")
    
    async def stop(self) -> None:
        """Stop performance monitoring."""
        if not self.running:
            return
        
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Performance monitoring stopped")
    
    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.running:
            try:
                # Collect system metrics
                system_metrics = self.metrics_collector.collect_system_metrics()
                
                # Log performance warnings
                if system_metrics.cpu_usage > 80:
                    self.logger.warning(f"High CPU usage: {system_metrics.cpu_usage:.1f}%")
                
                if system_metrics.memory_usage > 0:
                    try:
                        import psutil
                        memory_percent = (system_metrics.memory_usage / psutil.virtual_memory().total) * 100
                        if memory_percent > 80:
                            self.logger.warning(f"High memory usage: {memory_percent:.1f}%")
                    except ImportError:
                        pass
                
                # Check for stalled downloads
                for task_id, metrics in self.metrics_collector.get_all_task_metrics().items():
                    if (metrics.end_time is None and 
                        metrics.current_download_speed == 0 and 
                        metrics.duration().total_seconds() > 60):
                        self.logger.warning(f"Task {task_id} appears stalled")
                
                await asyncio.sleep(self.interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(self.interval)


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def initialize_monitoring(history_size: int = 1000) -> MetricsCollector:
    """Initialize the global monitoring system."""
    global _metrics_collector
    _metrics_collector = MetricsCollector(history_size)
    return _metrics_collector