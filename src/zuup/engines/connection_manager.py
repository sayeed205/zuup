"""ConnectionManager for multi-connection coordination using pycurl.CurlMulti."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import logging
import time
from typing import Any, Optional
from pathlib import Path

import pycurl

from .curl_worker import CurlWorker
from .pycurl_models import (
    DownloadSegment,
    HttpFtpConfig,
    WorkerProgress,
    WorkerStatus,
)
from .pycurl_logging import LogLevel
from .connection_pool import get_global_pool
from .adaptive_scaling import AdaptiveConnectionScaler, create_balanced_scaler
from .bandwidth_manager import EnhancedBandwidthManager, AllocationAlgorithm

logger = logging.getLogger(__name__)


# Legacy BandwidthManager class removed - now using EnhancedBandwidthManager


class ConnectionManager:
    """Manages multiple curl connections for parallel downloads with performance optimizations."""

    def __init__(
        self, 
        config: HttpFtpConfig,
        enable_adaptive_scaling: bool = True,
        enable_connection_pooling: bool = True,
        bandwidth_algorithm: AllocationAlgorithm = AllocationAlgorithm.ADAPTIVE,
    ) -> None:
        """
        Initialize ConnectionManager with performance optimizations.

        Args:
            config: Configuration for HTTP/FTP downloads
            enable_adaptive_scaling: Enable adaptive connection scaling
            enable_connection_pooling: Enable connection pooling
            bandwidth_algorithm: Bandwidth allocation algorithm
        """
        self.config = config
        self.curl_multi: pycurl.CurlMulti | None = None
        self.curl_share: pycurl.CurlShare | None = None
        self.active_workers: dict[str, CurlWorker] = {}

        # Performance optimizations
        self.enable_adaptive_scaling = enable_adaptive_scaling
        self.enable_connection_pooling = enable_connection_pooling

        # Enhanced bandwidth management
        total_bandwidth = getattr(config, "download_speed_limit", None)
        self.bandwidth_manager = EnhancedBandwidthManager(
            total_bandwidth=total_bandwidth,
            algorithm=bandwidth_algorithm,
        )

        # Adaptive connection scaling
        if enable_adaptive_scaling:
            self.connection_scaler = create_balanced_scaler(
                min_conn=1, max_conn=config.max_connections
            )
        else:
            self.connection_scaler = None

        # Connection management
        self.max_connections = config.max_connections
        self.current_connections = 0
        self._shutdown_requested = False

        # Performance tracking
        self._worker_progress: dict[str, WorkerProgress] = {}
        self._last_progress_update = time.time()
        self._performance_samples: list[dict[str, Any]] = []

        # Connection pool
        if enable_connection_pooling:
            self.connection_pool = get_global_pool()
        else:
            self.connection_pool = None

        logger.info(
            f"Initialized ConnectionManager with max {self.max_connections} connections, "
            f"adaptive_scaling={enable_adaptive_scaling}, "
            f"connection_pooling={enable_connection_pooling}, "
            f"bandwidth_algorithm={bandwidth_algorithm.value}"
        )

    async def __aenter__(self) -> ConnectionManager:
        """Async context manager entry."""
        await self._initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.cleanup()

    async def _initialize(self) -> None:
        """Initialize curl multi and share handles."""
        try:
            # Initialize CurlMulti for managing multiple transfers
            self.curl_multi = pycurl.CurlMulti()

            # Configure multi handle options
            self.curl_multi.setopt(pycurl.M_MAXCONNECTS, self.max_connections)

            # Initialize CurlShare for sharing data between handles
            self.curl_share = pycurl.CurlShare()
            self._setup_curl_share()

            logger.debug("Initialized curl multi and share handles")

        except Exception as e:
            logger.error(f"Failed to initialize ConnectionManager: {e}")
            raise

    def _setup_curl_share(self) -> None:
        """Configure shared data between curl handles."""
        if not self.curl_share:
            return

        try:
            # Share cookies between handles
            self.curl_share.setopt(pycurl.SH_SHARE, pycurl.LOCK_DATA_COOKIE)

            # Share DNS cache between handles
            self.curl_share.setopt(pycurl.SH_SHARE, pycurl.LOCK_DATA_DNS)

            # Share SSL session cache between handles
            self.curl_share.setopt(pycurl.SH_SHARE, pycurl.LOCK_DATA_SSL_SESSION)

            logger.debug(
                "Configured curl handle sharing for cookies, DNS, and SSL sessions"
            )

        except Exception as e:
            logger.warning(f"Failed to setup curl sharing: {e}")

    async def create_workers(
        self, 
        segments: list[DownloadSegment], 
        log_level: LogLevel = LogLevel.BASIC,
        log_dir: Optional[Path] = None,
    ) -> list[CurlWorker]:
        """
        Create workers for downloading segments.

        Args:
            segments: List of download segments
            log_level: Logging verbosity level for workers
            log_dir: Optional directory for debug logs

        Returns:
            List of created workers

        Raises:
            ValueError: If too many segments for available connections
        """
        if len(segments) > self.max_connections:
            logger.warning(
                f"Requested {len(segments)} segments but only {self.max_connections} "
                f"connections available. Using first {self.max_connections} segments."
            )
            segments = segments[: self.max_connections]

        workers = []

        try:
            for segment in segments:
                # Create worker with shared curl handle and logging configuration
                worker = CurlWorker(
                    segment=segment, 
                    config=self.config, 
                    curl_share=self.curl_share,
                    log_level=log_level,
                    log_dir=log_dir,
                )

                # Set progress callback
                worker.set_progress_callback(self._on_worker_progress)

                workers.append(worker)
                self.active_workers[worker.worker_id] = worker

                logger.debug(
                    f"Created worker {worker.worker_id} for segment {segment.id} with log level {log_level.value}"
                )

            # Allocate bandwidth among workers
            self._allocate_bandwidth(workers)

            logger.info(f"Created {len(workers)} workers for parallel download with {log_level.value} logging")
            return workers

        except Exception as e:
            # Cleanup any created workers on failure
            for worker in workers:
                self.active_workers.pop(worker.worker_id, None)
            logger.error(f"Failed to create workers: {e}")
            raise

    def _allocate_bandwidth(self, workers: list[CurlWorker]) -> None:
        """
        Allocate bandwidth among workers using enhanced bandwidth management.

        Args:
            workers: List of workers to allocate bandwidth for
        """
        # Add workers to bandwidth manager if not already present
        for worker in workers:
            if worker.worker_id not in self.bandwidth_manager.allocations:
                self.bandwidth_manager.add_worker(worker.worker_id)

        # Get current worker speeds
        worker_speeds = {}
        for worker in workers:
            progress = worker.get_progress()
            worker_speeds[worker.worker_id] = progress.download_speed

        # Allocate bandwidth using enhanced algorithm
        allocations = self.bandwidth_manager.allocate_bandwidth(worker_speeds)

        # Apply allocations to workers
        for worker in workers:
            limit = allocations.get(worker.worker_id, 0)
            if limit > 0:
                logger.debug(
                    f"Allocated {limit} bytes/sec to worker {worker.worker_id}"
                )
                # Apply bandwidth limit to worker if supported
                if hasattr(worker, 'set_bandwidth_limit'):
                    worker.set_bandwidth_limit(limit)

    async def monitor_workers(
        self, workers: list[CurlWorker]
    ) -> AsyncIterator[dict[str, WorkerProgress]]:
        """
        Monitor worker progress and yield updates.

        Args:
            workers: List of workers to monitor

        Yields:
            Dictionary mapping worker_id to progress information
        """
        if not workers:
            return

        logger.info(f"Starting monitoring of {len(workers)} workers")

        # Start all worker downloads
        download_tasks = []
        for worker in workers:
            task = asyncio.create_task(worker.download_segment())
            download_tasks.append(task)

        try:
            # Monitor progress while downloads are active
            last_optimization_update = time.time()
            optimization_interval = 10.0  # Update optimizations every 10 seconds
            
            while not self._shutdown_requested and any(
                not task.done() for task in download_tasks
            ):
                current_time = time.time()
                
                # Collect current progress from all workers
                current_progress = {}
                worker_speeds = {}

                for worker in workers:
                    progress = worker.get_progress()
                    current_progress[worker.worker_id] = progress
                    worker_speeds[worker.worker_id] = progress.download_speed

                # Update bandwidth allocation based on current speeds
                allocations = self.bandwidth_manager.allocate_bandwidth(worker_speeds)
                
                # Apply bandwidth limits to workers
                for worker in workers:
                    limit = allocations.get(worker.worker_id, 0)
                    if hasattr(worker, 'set_bandwidth_limit') and limit > 0:
                        worker.set_bandwidth_limit(limit)

                # Periodic optimization updates
                if current_time - last_optimization_update >= optimization_interval:
                    # Update adaptive scaling
                    self._update_adaptive_scaling()
                    
                    # Collect performance metrics
                    metrics = self._collect_performance_metrics()
                    self._performance_samples.append(metrics)
                    
                    # Keep only recent samples (last 100)
                    if len(self._performance_samples) > 100:
                        self._performance_samples = self._performance_samples[-100:]
                    
                    last_optimization_update = current_time

                # Yield progress update
                yield current_progress

                # Wait before next update
                await asyncio.sleep(1.0)

            # Wait for all downloads to complete
            if download_tasks:
                results = await asyncio.gather(*download_tasks, return_exceptions=True)

                # Log results
                for i, result in enumerate(results):
                    worker = workers[i]
                    if isinstance(result, Exception):
                        logger.error(f"Worker {worker.worker_id} failed: {result}")
                    else:
                        success = (
                            result.get("success", False)
                            if isinstance(result, dict)
                            else False
                        )
                        if success:
                            logger.info(
                                f"Worker {worker.worker_id} completed successfully"
                            )
                        else:
                            error = (
                                result.get("error", "Unknown error")
                                if isinstance(result, dict)
                                else str(result)
                            )
                            logger.warning(f"Worker {worker.worker_id} failed: {error}")

        except Exception as e:
            logger.error(f"Error monitoring workers: {e}")
            # Cancel remaining tasks
            for task in download_tasks:
                if not task.done():
                    task.cancel()
            raise
        finally:
            # Cleanup workers
            for worker in workers:
                self.active_workers.pop(worker.worker_id, None)

    def _on_worker_progress(self, progress: WorkerProgress) -> None:
        """
        Handle progress updates from workers with performance optimization.

        Args:
            progress: Worker progress information
        """
        self._worker_progress[progress.worker_id] = progress

        # Update adaptive scaling with performance data
        if self.connection_scaler:
            # Calculate response time (simplified - would need actual measurement)
            response_time = 1.0  # Placeholder
            had_error = progress.status == WorkerStatus.FAILED
            connection_successful = progress.status != WorkerStatus.FAILED

            self.connection_scaler.add_performance_sample(
                response_time=response_time,
                download_speed=progress.download_speed,
                had_error=had_error,
                connection_successful=connection_successful,
            )

        # Log significant progress updates
        if progress.status in (WorkerStatus.COMPLETED, WorkerStatus.FAILED):
            logger.info(
                f"Worker {progress.worker_id} {progress.status.value}: "
                f"{progress.downloaded_bytes} bytes"
            )

    def _update_adaptive_scaling(self) -> None:
        """Update adaptive connection scaling based on performance."""
        if not self.connection_scaler:
            return

        # Get scaling recommendation
        recommendation = self.connection_scaler.get_scaling_recommendation()
        
        if recommendation["would_change"]:
            new_connections = recommendation["recommended_connections"]
            reason = recommendation["reason"]
            
            logger.info(
                f"Adaptive scaling recommendation: {self.max_connections} -> {new_connections} ({reason})"
            )
            
            # Apply scaling decision
            self.connection_scaler.apply_scaling_decision()
            
            # Update max connections (this would affect future worker creation)
            old_max = self.max_connections
            self.max_connections = new_connections
            
            # Update curl multi handle if initialized
            if self.curl_multi:
                self.curl_multi.setopt(pycurl.M_MAXCONNECTS, new_connections)
            
            logger.info(f"Scaled max connections from {old_max} to {new_connections}")

    def _collect_performance_metrics(self) -> dict[str, Any]:
        """Collect comprehensive performance metrics."""
        current_time = time.time()
        
        # Worker metrics
        worker_metrics = {}
        total_speed = 0.0
        active_workers = 0
        
        for worker_id, progress in self._worker_progress.items():
            if progress.status == WorkerStatus.DOWNLOADING:
                active_workers += 1
                total_speed += progress.download_speed
            
            worker_metrics[worker_id] = {
                "status": progress.status.value,
                "downloaded_bytes": progress.downloaded_bytes,
                "total_bytes": progress.total_bytes,
                "download_speed": progress.download_speed,
                "progress_percentage": progress.progress_percentage,
            }
        
        # Connection metrics
        connection_metrics = {
            "active_workers": active_workers,
            "max_connections": self.max_connections,
            "total_download_speed": total_speed,
            "average_speed_per_worker": total_speed / max(1, active_workers),
        }
        
        # Bandwidth metrics
        bandwidth_stats = self.bandwidth_manager.get_allocation_stats()
        
        # Adaptive scaling metrics
        scaling_stats = {}
        if self.connection_scaler:
            scaling_stats = self.connection_scaler.get_scaling_stats()
        
        return {
            "timestamp": current_time,
            "workers": worker_metrics,
            "connections": connection_metrics,
            "bandwidth": bandwidth_stats,
            "scaling": scaling_stats,
        }

    async def pause_workers(self, worker_ids: list[str] | None = None) -> None:
        """
        Pause specified workers or all active workers.

        Args:
            worker_ids: List of worker IDs to pause (None for all)
        """
        target_workers = (
            [
                self.active_workers[wid]
                for wid in worker_ids
                if wid in self.active_workers
            ]
            if worker_ids
            else list(self.active_workers.values())
        )

        for worker in target_workers:
            worker.pause()
            logger.debug(f"Paused worker {worker.worker_id}")

        logger.info(f"Paused {len(target_workers)} workers")

    async def resume_workers(self, worker_ids: list[str] | None = None) -> None:
        """
        Resume specified workers or all paused workers.

        Args:
            worker_ids: List of worker IDs to resume (None for all)
        """
        target_workers = (
            [
                self.active_workers[wid]
                for wid in worker_ids
                if wid in self.active_workers
            ]
            if worker_ids
            else list(self.active_workers.values())
        )

        for worker in target_workers:
            worker.resume()
            logger.debug(f"Resumed worker {worker.worker_id}")

        logger.info(f"Resumed {len(target_workers)} workers")

    async def cancel_workers(self, worker_ids: list[str] | None = None) -> None:
        """
        Cancel specified workers or all active workers.

        Args:
            worker_ids: List of worker IDs to cancel (None for all)
        """
        target_workers = (
            [
                self.active_workers[wid]
                for wid in worker_ids
                if wid in self.active_workers
            ]
            if worker_ids
            else list(self.active_workers.values())
        )

        for worker in target_workers:
            worker.cancel()
            logger.debug(f"Cancelled worker {worker.worker_id}")

        # Remove cancelled workers from active list
        for worker in target_workers:
            self.active_workers.pop(worker.worker_id, None)

        logger.info(f"Cancelled {len(target_workers)} workers")

    def get_worker_progress(self, worker_id: str) -> WorkerProgress | None:
        """
        Get progress for a specific worker.

        Args:
            worker_id: Worker identifier

        Returns:
            Worker progress or None if not found
        """
        return self._worker_progress.get(worker_id)

    def get_all_progress(self) -> dict[str, WorkerProgress]:
        """
        Get progress for all workers.

        Returns:
            Dictionary mapping worker_id to progress
        """
        return self._worker_progress.copy()

    def scale_connections(self, new_max: int) -> None:
        """
        Dynamically scale the maximum number of connections.

        Args:
            new_max: New maximum connection count
        """
        if new_max <= 0:
            raise ValueError("Maximum connections must be positive")

        old_max = self.max_connections
        self.max_connections = new_max

        # Update curl multi handle if initialized
        if self.curl_multi:
            self.curl_multi.setopt(pycurl.M_MAXCONNECTS, new_max)

        logger.info(f"Scaled connections from {old_max} to {new_max}")

        # If reducing connections and we have too many active workers,
        # we should handle this gracefully (implementation depends on use case)
        if new_max < len(self.active_workers):
            logger.warning(
                f"Current active workers ({len(self.active_workers)}) exceed "
                f"new limit ({new_max}). Consider cancelling some workers."
            )

    def get_connection_stats(self) -> dict[str, Any]:
        """
        Get connection statistics.

        Returns:
            Dictionary with connection statistics
        """
        active_count = len(self.active_workers)

        # Calculate total speeds
        total_download_speed = sum(
            progress.download_speed for progress in self._worker_progress.values()
        )

        # Count workers by status
        status_counts = {}
        for progress in self._worker_progress.values():
            status = progress.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "max_connections": self.max_connections,
            "active_workers": active_count,
            "total_download_speed": total_download_speed,
            "worker_status_counts": status_counts,
            "bandwidth_limit": self.bandwidth_manager.total_limit,
        }

    async def cleanup(self) -> None:
        """Clean up resources and cancel active workers."""
        logger.info("Cleaning up ConnectionManager")

        self._shutdown_requested = True

        # Cancel all active workers
        if self.active_workers:
            await self.cancel_workers()

        # Clean up curl handles
        if self.curl_multi:
            try:
                self.curl_multi.close()
            except Exception as e:
                logger.warning(f"Error closing curl multi handle: {e}")
            finally:
                self.curl_multi = None

        if self.curl_share:
            try:
                self.curl_share.close()
            except Exception as e:
                logger.warning(f"Error closing curl share handle: {e}")
            finally:
                self.curl_share = None

        # Clean up performance optimization components
        if hasattr(self, 'bandwidth_manager'):
            # Remove all workers from bandwidth manager
            for worker_id in list(self.bandwidth_manager.allocations.keys()):
                self.bandwidth_manager.remove_worker(worker_id)

        # Clear tracking data
        self.active_workers.clear()
        self._worker_progress.clear()
        self._performance_samples.clear()

        logger.info("ConnectionManager cleanup completed")

    def __del__(self) -> None:
        """Cleanup when manager is destroyed."""
        if self.curl_multi or self.curl_share or self.active_workers:
            logger.warning("ConnectionManager destroyed without proper cleanup")
    def set_worker_log_level(self, log_level: LogLevel, worker_ids: Optional[list[str]] = None) -> None:
        """
        Set logging level for specified workers or all workers.
        
        Args:
            log_level: New logging level
            worker_ids: Optional list of worker IDs (None for all workers)
        """
        target_workers = []
        
        if worker_ids:
            target_workers = [
                worker for worker_id, worker in self.active_workers.items()
                if worker_id in worker_ids
            ]
        else:
            target_workers = list(self.active_workers.values())
        
        for worker in target_workers:
            worker.set_log_level(log_level)
        
        logger.info(f"Updated log level to {log_level.value} for {len(target_workers)} workers")
    
    def get_debug_summary(self) -> dict[str, Any]:
        """
        Get comprehensive debug summary from all workers.
        
        Returns:
            Debug summary with worker information
        """
        summary = {
            "connection_manager": {
                "active_workers": len(self.active_workers),
                "max_connections": self.max_connections,
                "bandwidth_limit": self.bandwidth_manager.total_limit,
                "connection_stats": self.get_connection_stats(),
            },
            "workers": {},
            "aggregated_stats": {
                "total_errors": 0,
                "total_retries": 0,
                "total_connection_attempts": 0,
                "successful_connections": 0,
                "failed_connections": 0,
            }
        }
        
        for worker_id, worker in self.active_workers.items():
            worker_debug = worker.get_debug_summary()
            summary["workers"][worker_id] = worker_debug
            
            # Aggregate statistics
            summary["aggregated_stats"]["total_errors"] += worker_debug.get("error_count", 0)
            summary["aggregated_stats"]["total_retries"] += worker_debug.get("retry_count", 0)
            
            conn_stats = worker_debug.get("connection_stats", {})
            summary["aggregated_stats"]["total_connection_attempts"] += conn_stats.get("attempts", 0)
            summary["aggregated_stats"]["successful_connections"] += conn_stats.get("successful", 0)
            summary["aggregated_stats"]["failed_connections"] += conn_stats.get("failed", 0)
        
        return summary
    
    def get_performance_metrics(self) -> dict[str, Any]:
        """
        Get performance metrics from all workers.
        
        Returns:
            Performance metrics summary
        """
        metrics = {
            "workers": {},
            "aggregated": {
                "total_downloaded": 0,
                "average_speed": 0.0,
                "total_duration": 0.0,
                "efficiency_scores": [],
            }
        }
        
        total_speed = 0.0
        worker_count = 0
        
        for worker_id, worker in self.active_workers.items():
            worker_metrics = worker.get_performance_metrics()
            metrics["workers"][worker_id] = worker_metrics
            
            # Aggregate metrics
            transfer_info = worker_metrics.get("transfer", {})
            timing_info = worker_metrics.get("timing", {})
            
            metrics["aggregated"]["total_downloaded"] += transfer_info.get("downloaded_bytes", 0)
            
            if transfer_info.get("download_speed", 0) > 0:
                total_speed += transfer_info["download_speed"]
                worker_count += 1
            
            if timing_info.get("duration", 0) > 0:
                metrics["aggregated"]["total_duration"] = max(
                    metrics["aggregated"]["total_duration"],
                    timing_info["duration"]
                )
            
            if transfer_info.get("efficiency", 0) > 0:
                metrics["aggregated"]["efficiency_scores"].append(transfer_info["efficiency"])
        
        # Calculate averages
        if worker_count > 0:
            metrics["aggregated"]["average_speed"] = total_speed / worker_count
        
        if metrics["aggregated"]["efficiency_scores"]:
            metrics["aggregated"]["average_efficiency"] = (
                sum(metrics["aggregated"]["efficiency_scores"]) / 
                len(metrics["aggregated"]["efficiency_scores"])
            )
        
        return metrics
    
    def generate_debug_commands(self) -> dict[str, str]:
        """
        Generate equivalent curl commands for all workers for debugging.
        
        Returns:
            Dictionary mapping worker IDs to curl commands
        """
        commands = {}
        
        for worker_id, worker in self.active_workers.items():
            try:
                commands[worker_id] = worker.generate_debug_curl_command()
            except Exception as e:
                commands[worker_id] = f"Error generating command: {e}"
        
        return commands
    
    def diagnose_connection_issues(self) -> dict[str, Any]:
        """
        Diagnose connection issues across all workers.
        
        Returns:
            Diagnosis information with suggestions
        """
        from .pycurl_logging import CurlDebugUtilities
        
        diagnosis = {
            "overall_health": "healthy",
            "issues_found": [],
            "worker_diagnoses": {},
            "recommendations": [],
        }
        
        failed_workers = 0
        total_workers = len(self.active_workers)
        
        for worker_id, worker in self.active_workers.items():
            worker_debug = worker.get_debug_summary()
            
            # Check for worker issues
            if worker_debug.get("error_count", 0) > 0:
                failed_workers += 1
                
                # Get last error for diagnosis
                last_error = worker_debug.get("last_error")
                if last_error:
                    # Try to extract curl code if available
                    curl_code = None
                    if "curl code" in last_error.lower():
                        try:
                            # Extract curl code from error message
                            import re
                            match = re.search(r'curl code (\d+)', last_error.lower())
                            if match:
                                curl_code = int(match.group(1))
                        except:
                            pass
                    
                    if curl_code:
                        worker_diagnosis = CurlDebugUtilities.diagnose_connection_error(
                            curl_code, worker.segment.url
                        )
                        diagnosis["worker_diagnoses"][worker_id] = worker_diagnosis
        
        # Determine overall health
        failure_rate = failed_workers / total_workers if total_workers > 0 else 0
        
        if failure_rate == 0:
            diagnosis["overall_health"] = "healthy"
        elif failure_rate < 0.25:
            diagnosis["overall_health"] = "minor_issues"
            diagnosis["issues_found"].append(f"{failed_workers}/{total_workers} workers have errors")
        elif failure_rate < 0.75:
            diagnosis["overall_health"] = "degraded"
            diagnosis["issues_found"].append(f"High failure rate: {failed_workers}/{total_workers} workers failing")
        else:
            diagnosis["overall_health"] = "critical"
            diagnosis["issues_found"].append(f"Critical failure rate: {failed_workers}/{total_workers} workers failing")
        
        # Generate recommendations
        if failure_rate > 0:
            diagnosis["recommendations"].extend([
                "Check network connectivity",
                "Verify server availability",
                "Review authentication credentials",
                "Consider reducing concurrent connections",
                "Check firewall and proxy settings",
            ])
        
        if failure_rate > 0.5:
            diagnosis["recommendations"].extend([
                "Consider switching to single-connection mode",
                "Increase timeout values",
                "Check for server-side rate limiting",
            ])
        
        return diagnosis

    def get_performance_optimization_stats(self) -> dict[str, Any]:
        """Get comprehensive performance optimization statistics."""
        stats = {
            "connection_manager": {
                "adaptive_scaling_enabled": self.enable_adaptive_scaling,
                "connection_pooling_enabled": self.enable_connection_pooling,
                "current_connections": len(self.active_workers),
                "max_connections": self.max_connections,
            },
            "bandwidth_management": self.bandwidth_manager.get_allocation_stats(),
            "performance_analysis": self.bandwidth_manager.get_performance_analysis(),
        }

        # Add adaptive scaling stats if enabled
        if self.connection_scaler:
            stats["adaptive_scaling"] = self.connection_scaler.get_scaling_stats()
            stats["scaling_recommendation"] = self.connection_scaler.get_scaling_recommendation()

        # Add connection pool stats if enabled
        if self.connection_pool:
            stats["connection_pool"] = self.connection_pool.get_stats()

        # Add recent performance samples
        if self._performance_samples:
            stats["recent_performance"] = self._performance_samples[-10:]  # Last 10 samples

        return stats

    def optimize_performance(self) -> dict[str, Any]:
        """Perform performance optimization and return recommendations."""
        optimization_results = {
            "timestamp": time.time(),
            "optimizations_applied": [],
            "recommendations": [],
        }

        # Bandwidth optimization
        bandwidth_optimization = self.bandwidth_manager.optimize_allocation()
        optimization_results["bandwidth_optimization"] = bandwidth_optimization
        optimization_results["recommendations"].extend(bandwidth_optimization["suggestions"])

        # Adaptive scaling optimization
        if self.connection_scaler:
            scaling_recommendation = self.connection_scaler.get_scaling_recommendation()
            if scaling_recommendation["would_change"]:
                # Apply scaling recommendation
                new_connections, reason = self.connection_scaler.apply_scaling_decision()
                optimization_results["optimizations_applied"].append(
                    f"Scaled connections to {new_connections} ({reason})"
                )
                
                # Update max connections
                self.max_connections = new_connections
                if self.curl_multi:
                    self.curl_multi.setopt(pycurl.M_MAXCONNECTS, new_connections)

        # Connection pool optimization
        if self.connection_pool:
            pool_stats = self.connection_pool.get_stats()
            if pool_stats["reuse_rate"] < 0.3:
                optimization_results["recommendations"].append(
                    "Low connection reuse rate - consider adjusting pool settings"
                )

        return optimization_results

    def set_bandwidth_algorithm(self, algorithm: AllocationAlgorithm) -> None:
        """Change the bandwidth allocation algorithm."""
        old_algorithm = self.bandwidth_manager.algorithm
        self.bandwidth_manager.algorithm = algorithm
        
        logger.info(f"Changed bandwidth algorithm from {old_algorithm.value} to {algorithm.value}")

    def enable_aggressive_scaling(self) -> None:
        """Enable aggressive adaptive scaling."""
        if self.connection_scaler:
            self.connection_scaler.aggressive_scaling = True
            logger.info("Enabled aggressive adaptive scaling")

    def disable_aggressive_scaling(self) -> None:
        """Disable aggressive adaptive scaling."""
        if self.connection_scaler:
            self.connection_scaler.aggressive_scaling = False
            logger.info("Disabled aggressive adaptive scaling")

    def force_connection_count(self, count: int, reason: str = "manual") -> None:
        """Force a specific connection count."""
        count = max(1, min(count, 32))  # Reasonable bounds
        
        if self.connection_scaler:
            self.connection_scaler.force_scale_to(count, reason)
        
        self.max_connections = count
        if self.curl_multi:
            self.curl_multi.setopt(pycurl.M_MAXCONNECTS, count)
        
        logger.info(f"Forced connection count to {count} ({reason})")

    def get_optimization_recommendations(self) -> list[str]:
        """Get performance optimization recommendations."""
        recommendations = []
        
        # Analyze current performance
        if self._performance_samples:
            recent_samples = self._performance_samples[-10:]
            
            # Check for consistent low speeds
            avg_speeds = [s["connections"]["total_download_speed"] for s in recent_samples]
            if avg_speeds and sum(avg_speeds) / len(avg_speeds) < 100 * 1024:  # < 100KB/s
                recommendations.append("Low download speeds detected - consider checking network connectivity")
            
            # Check for high error rates
            error_rates = []
            for sample in recent_samples:
                failed_workers = sum(
                    1 for w in sample["workers"].values() 
                    if w["status"] == "failed"
                )
                total_workers = len(sample["workers"])
                if total_workers > 0:
                    error_rates.append(failed_workers / total_workers)
            
            if error_rates and sum(error_rates) / len(error_rates) > 0.2:
                recommendations.append("High error rate detected - consider reducing connection count")

        # Bandwidth management recommendations
        bandwidth_optimization = self.bandwidth_manager.optimize_allocation()
        recommendations.extend(bandwidth_optimization["suggestions"])

        # Connection pool recommendations
        if self.connection_pool:
            pool_stats = self.connection_pool.get_stats()
            if pool_stats["reuse_rate"] < 0.2:
                recommendations.append("Consider increasing connection pool size for better reuse")

        return recommendations