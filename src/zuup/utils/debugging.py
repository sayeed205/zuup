"""Debugging utilities and error reporting tools."""

import asyncio
import functools
import inspect
import logging
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
import json

from ..storage.models import DownloadTask, ProgressInfo, TaskStatus, EngineType


F = TypeVar('F', bound=Callable[..., Any])


class DebugInfo:
    """Container for debug information."""
    
    def __init__(self):
        self.timestamp = datetime.now()
        self.function_calls: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, Any]] = []
        self.performance_data: Dict[str, Any] = {}
        self.system_state: Dict[str, Any] = {}
    
    def add_function_call(self, func_name: str, args: tuple, kwargs: dict, 
                         duration: float, result: Any = None, error: Exception = None) -> None:
        """Add function call information."""
        call_info = {
            'timestamp': datetime.now().isoformat(),
            'function': func_name,
            'args': str(args)[:200],  # Truncate long args
            'kwargs': str(kwargs)[:200],
            'duration_ms': duration * 1000,
            'success': error is None,
        }
        
        if error:
            call_info['error'] = {
                'type': type(error).__name__,
                'message': str(error),
                'traceback': traceback.format_exception(type(error), error, error.__traceback__)
            }
        
        self.function_calls.append(call_info)
    
    def add_error(self, error: Exception, context: str = "") -> None:
        """Add error information."""
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'type': type(error).__name__,
            'message': str(error),
            'context': context,
            'traceback': traceback.format_exception(type(error), error, error.__traceback__)
        }
        self.errors.append(error_info)
    
    def set_system_state(self, state: Dict[str, Any]) -> None:
        """Set current system state."""
        self.system_state = {
            'timestamp': datetime.now().isoformat(),
            **state
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert debug info to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'function_calls': self.function_calls,
            'errors': self.errors,
            'performance_data': self.performance_data,
            'system_state': self.system_state,
        }
    
    def save_to_file(self, file_path: Path) -> None:
        """Save debug info to file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class DebugSession:
    """Debug session manager."""
    
    def __init__(self, session_name: str = ""):
        self.session_name = session_name or f"debug_{int(time.time())}"
        self.debug_info = DebugInfo()
        self.logger = logging.getLogger(f"zuup.debug.{self.session_name}")
        self.active = True
    
    def log_function_call(self, func_name: str, args: tuple, kwargs: dict,
                         duration: float, result: Any = None, error: Exception = None) -> None:
        """Log a function call."""
        if not self.active:
            return
        
        self.debug_info.add_function_call(func_name, args, kwargs, duration, result, error)
        
        if error:
            self.logger.error(f"{func_name} failed after {duration*1000:.2f}ms: {error}")
        else:
            self.logger.debug(f"{func_name} completed in {duration*1000:.2f}ms")
    
    def log_error(self, error: Exception, context: str = "") -> None:
        """Log an error."""
        if not self.active:
            return
        
        self.debug_info.add_error(error, context)
        self.logger.error(f"Error in {context}: {error}", exc_info=True)
    
    def update_system_state(self, **kwargs) -> None:
        """Update system state information."""
        if not self.active:
            return
        
        self.debug_info.set_system_state(kwargs)
    
    def save_session(self, output_dir: Path) -> Path:
        """Save debug session to file."""
        output_file = output_dir / f"{self.session_name}.json"
        self.debug_info.save_to_file(output_file)
        self.logger.info(f"Debug session saved to {output_file}")
        return output_file
    
    def close(self) -> None:
        """Close debug session."""
        self.active = False


# Global debug session
_current_debug_session: Optional[DebugSession] = None


def start_debug_session(session_name: str = "") -> DebugSession:
    """Start a new debug session."""
    global _current_debug_session
    _current_debug_session = DebugSession(session_name)
    return _current_debug_session


def get_debug_session() -> Optional[DebugSession]:
    """Get current debug session."""
    return _current_debug_session


def end_debug_session(output_dir: Path) -> Optional[Path]:
    """End current debug session and save to file."""
    global _current_debug_session
    if _current_debug_session:
        output_file = _current_debug_session.save_session(output_dir)
        _current_debug_session.close()
        _current_debug_session = None
        return output_file
    return None


def debug_trace(func: F) -> F:
    """Decorator to trace function calls for debugging."""
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        session = get_debug_session()
        if not session:
            return func(*args, **kwargs)
        
        func_name = f"{func.__module__}.{func.__qualname__}"
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            session.log_function_call(func_name, args, kwargs, duration, result)
            return result
        except Exception as e:
            duration = time.time() - start_time
            session.log_function_call(func_name, args, kwargs, duration, error=e)
            raise
    
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        session = get_debug_session()
        if not session:
            return await func(*args, **kwargs)
        
        func_name = f"{func.__module__}.{func.__qualname__}"
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            session.log_function_call(func_name, args, kwargs, duration, result)
            return result
        except Exception as e:
            duration = time.time() - start_time
            session.log_function_call(func_name, args, kwargs, duration, error=e)
            raise
    
    if inspect.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


@contextmanager
def debug_context(context_name: str):
    """Context manager for debugging specific code blocks."""
    session = get_debug_session()
    if session:
        session.logger.debug(f"Entering {context_name}")
        start_time = time.time()
    
    try:
        yield
    except Exception as e:
        if session:
            session.log_error(e, context_name)
        raise
    finally:
        if session:
            duration = time.time() - start_time
            session.logger.debug(f"Exiting {context_name} after {duration*1000:.2f}ms")


class TaskDebugger:
    """Specialized debugger for download tasks."""
    
    def __init__(self, task: DownloadTask):
        self.task = task
        self.logger = logging.getLogger(f"zuup.debug.task.{task.id}")
        self.events: List[Dict[str, Any]] = []
        self.start_time = time.time()
    
    def log_event(self, event_type: str, data: Dict[str, Any] = None) -> None:
        """Log a task event."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'elapsed_seconds': time.time() - self.start_time,
            'event_type': event_type,
            'task_id': self.task.id,
            'task_status': self.task.status.value,
            'data': data or {}
        }
        
        self.events.append(event)
        self.logger.debug(f"Task event: {event_type} - {data}")
    
    def log_progress_update(self, progress: ProgressInfo) -> None:
        """Log progress update."""
        self.log_event('progress_update', {
            'downloaded_bytes': progress.downloaded_bytes,
            'total_bytes': progress.total_bytes,
            'download_speed': progress.download_speed,
            'progress_percentage': progress.progress_percentage,
            'status': progress.status.value,
        })
    
    def log_error(self, error: Exception, context: str = "") -> None:
        """Log task error."""
        self.log_event('error', {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'traceback': traceback.format_exception(type(error), error, error.__traceback__)
        })
    
    def log_retry(self, attempt: int, reason: str = "") -> None:
        """Log retry attempt."""
        self.log_event('retry', {
            'attempt': attempt,
            'reason': reason
        })
    
    def log_completion(self, final_status: TaskStatus) -> None:
        """Log task completion."""
        self.log_event('completion', {
            'final_status': final_status.value,
            'total_duration': time.time() - self.start_time,
            'final_size': self.task.progress.downloaded_bytes
        })
    
    def save_debug_log(self, output_dir: Path) -> Path:
        """Save task debug log to file."""
        output_file = output_dir / f"task_{self.task.id}_debug.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        debug_data = {
            'task_info': {
                'id': self.task.id,
                'url': self.task.url,
                'engine_type': self.task.engine_type.value,
                'destination': str(self.task.destination),
                'created_at': self.task.created_at.isoformat(),
            },
            'events': self.events,
            'total_duration': time.time() - self.start_time,
        }
        
        with open(output_file, 'w') as f:
            json.dump(debug_data, f, indent=2, default=str)
        
        self.logger.info(f"Task debug log saved to {output_file}")
        return output_file


class ErrorReporter:
    """Centralized error reporting system."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("zuup.error_reporter")
        
        # Error statistics
        self.error_counts: Dict[str, int] = {}
        self.recent_errors: List[Dict[str, Any]] = []
    
    def report_error(self, error: Exception, context: str = "", 
                    task_id: str = "", additional_data: Dict[str, Any] = None) -> None:
        """Report an error with full context."""
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        error_report = {
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'error_message': str(error),
            'context': context,
            'task_id': task_id,
            'traceback': traceback.format_exception(type(error), error, error.__traceback__),
            'additional_data': additional_data or {},
            'system_info': self._get_system_info(),
        }
        
        self.recent_errors.append(error_report)
        
        # Keep only recent errors (last 100)
        if len(self.recent_errors) > 100:
            self.recent_errors = self.recent_errors[-100:]
        
        # Save individual error report
        error_file = self.output_dir / f"error_{int(time.time())}_{error_type.lower()}.json"
        error_file.parent.mkdir(parents=True, exist_ok=True)
        with open(error_file, 'w') as f:
            json.dump(error_report, f, indent=2, default=str)
        
        self.logger.error(f"Error reported: {error_type} in {context} (saved to {error_file})")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get current system information."""
        info = {
            'python_version': sys.version,
            'platform': sys.platform,
        }
        
        try:
            import psutil
            info.update({
                'cpu_count': psutil.cpu_count(),
                'memory_total': psutil.virtual_memory().total,
                'memory_available': psutil.virtual_memory().available,
                'disk_free': psutil.disk_usage('/').free,
            })
        except ImportError:
            pass
        
        return info
    
    def generate_error_summary(self) -> Dict[str, Any]:
        """Generate error summary report."""
        return {
            'timestamp': datetime.now().isoformat(),
            'error_counts': self.error_counts,
            'total_errors': sum(self.error_counts.values()),
            'recent_errors_count': len(self.recent_errors),
            'most_common_errors': sorted(
                self.error_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10],
        }
    
    def save_error_summary(self) -> Path:
        """Save error summary to file."""
        summary_file = self.output_dir / "error_summary.json"
        summary = self.generate_error_summary()
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Error summary saved to {summary_file}")
        return summary_file


# Global error reporter
_error_reporter: Optional[ErrorReporter] = None


def initialize_error_reporting(output_dir: Path) -> ErrorReporter:
    """Initialize global error reporting."""
    global _error_reporter
    _error_reporter = ErrorReporter(output_dir)
    return _error_reporter


def report_error(error: Exception, context: str = "", 
                task_id: str = "", additional_data: Dict[str, Any] = None) -> None:
    """Report an error using the global error reporter."""
    if _error_reporter:
        _error_reporter.report_error(error, context, task_id, additional_data)


def get_error_reporter() -> Optional[ErrorReporter]:
    """Get the global error reporter."""
    return _error_reporter


class PerformanceProfiler:
    """Simple performance profiler for download operations."""
    
    def __init__(self):
        self.profiles: Dict[str, List[float]] = {}
        self.active_timers: Dict[str, float] = {}
    
    @contextmanager
    def profile(self, operation_name: str):
        """Context manager for profiling operations."""
        start_time = time.time()
        self.active_timers[operation_name] = start_time
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            
            if operation_name not in self.profiles:
                self.profiles[operation_name] = []
            
            self.profiles[operation_name].append(duration)
            
            if operation_name in self.active_timers:
                del self.active_timers[operation_name]
    
    def get_stats(self, operation_name: str) -> Dict[str, float]:
        """Get statistics for an operation."""
        if operation_name not in self.profiles:
            return {}
        
        times = self.profiles[operation_name]
        return {
            'count': len(times),
            'total_time': sum(times),
            'average_time': sum(times) / len(times),
            'min_time': min(times),
            'max_time': max(times),
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations."""
        return {
            operation: self.get_stats(operation)
            for operation in self.profiles.keys()
        }
    
    def reset(self) -> None:
        """Reset all profiling data."""
        self.profiles.clear()
        self.active_timers.clear()


# Global profiler
_profiler = PerformanceProfiler()


def get_profiler() -> PerformanceProfiler:
    """Get the global performance profiler."""
    return _profiler