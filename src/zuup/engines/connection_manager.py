"""ConnectionManager for multi-connection coordination using pycurl.CurlMulti."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import logging
import time
from typing import Any

import pycurl

from .curl_worker import CurlWorker
from .pycurl_models import (
    DownloadSegment,
    HttpFtpConfig,
    WorkerProgress,
    WorkerStatus,
)

logger = logging.getLogger(__name__)


class BandwidthManager:
    """Manages bandwidth allocation across multiple workers."""

    def __init__(self, total_limit: int | None = None) -> None:
        """
        Initialize bandwidth manager.

        Args:
            total_limit: Total bandwidth limit in bytes per second (None for unlimited)
        """
        self.total_limit = total_limit
        self.worker_allocations: dict[str, int] = {}
        self.worker_speeds: dict[str, float] = {}
        self._last_update = time.time()

    def allocate_bandwidth(self, workers: list[CurlWorker]) -> dict[str, int]:
        """
        Allocate bandwidth among active workers.

        Args:
            workers: List of active workers

        Returns:
            Dictionary mapping worker_id to allocated bandwidth in bytes/sec
        """
        if not self.total_limit or not workers:
            # No limit or no workers - return unlimited for all
            return {worker.worker_id: 0 for worker in workers}

        # Equal allocation among workers
        per_worker_limit = self.total_limit // len(workers)
        
        allocations = {}
        for worker in workers:
            # Ensure minimum allocation of 1KB/s per worker
            allocation = max(per_worker_limit, 1024)
            allocations[worker.worker_id] = allocation
            
        self.worker_allocations = allocations
        return allocations

    def update_allocation(self, worker_speeds: dict[str, float]) -> None:
        """
        Update bandwidth allocation based on current worker speeds.

        Args:
            worker_speeds: Dictionary mapping worker_id to current speed
        """
        self.worker_speeds = worker_speeds.copy()
        current_time = time.time()
        
        # Update allocations every 5 seconds
        if current_time - self._last_update >= 5.0:
            self._rebalance_bandwidth()
            self._last_update = current_time

    def _rebalance_bandwidth(self) -> None:
        """Rebalance bandwidth based on worker performance."""
        if not self.total_limit or not self.worker_speeds:
            return

        # Calculate total current usage
        total_usage = sum(self.worker_speeds.values())
        
        if total_usage <= self.total_limit:
            # Under limit - allow workers to use more if needed
            return

        # Over limit - need to throttle
        for worker_id, current_speed in self.worker_speeds.items():
            if worker_id in self.worker_allocations:
                # Reduce allocation proportionally
                reduction_factor = self.total_limit / total_usage
                new_allocation = int(current_speed * reduction_factor)
                self.worker_allocations[worker_id] = max(new_allocation, 1024)

    def get_worker_limit(self, worker_id: str) -> int:
        """
        Get bandwidth limit for a specific worker.

        Args:
            worker_id: Worker identifier

        Returns:
            Bandwidth limit in bytes per second (0 for unlimited)
        """
        return self.worker_allocations.get(worker_id, 0)

    def enforce_limits(self) -> None:
        """Enforce bandwidth limits on workers."""
        # This would be called by the ConnectionManager to apply limits
        # Implementation depends on how pycurl handles bandwidth limiting
        pass


class ConnectionManager:
    """Manages multiple curl connections for parallel downloads."""

    def __init__(self, config: HttpFtpConfig) -> None:
        """
        Initialize ConnectionManager.

        Args:
            config: Configuration for HTTP/FTP downloads
        """
        self.config = config
        self.curl_multi: pycurl.CurlMulti | None = None
        self.curl_share: pycurl.CurlShare | None = None
        self.active_workers: dict[str, CurlWorker] = {}
        self.bandwidth_manager = BandwidthManager(
            config.download_speed_limit if hasattr(config, 'download_speed_limit') else None
        )
        
        # Connection management
        self.max_connections = config.max_connections
        self.current_connections = 0
        self._shutdown_requested = False
        
        # Performance tracking
        self._worker_progress: dict[str, WorkerProgress] = {}
        self._last_progress_update = time.time()
        
        logger.info(f"Initialized ConnectionManager with max {self.max_connections} connections")

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
            
            logger.debug("Configured curl handle sharing for cookies, DNS, and SSL sessions")
            
        except Exception as e:
            logger.warning(f"Failed to setup curl sharing: {e}")

    async def create_workers(self, segments: list[DownloadSegment]) -> list[CurlWorker]:
        """
        Create workers for downloading segments.

        Args:
            segments: List of download segments

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
            segments = segments[:self.max_connections]

        workers = []
        
        try:
            for segment in segments:
                # Create worker with shared curl handle
                worker = CurlWorker(
                    segment=segment,
                    config=self.config,
                    curl_share=self.curl_share
                )
                
                # Set progress callback
                worker.set_progress_callback(self._on_worker_progress)
                
                workers.append(worker)
                self.active_workers[worker.worker_id] = worker
                
                logger.debug(f"Created worker {worker.worker_id} for segment {segment.id}")

            # Allocate bandwidth among workers
            self._allocate_bandwidth(workers)
            
            logger.info(f"Created {len(workers)} workers for parallel download")
            return workers
            
        except Exception as e:
            # Cleanup any created workers on failure
            for worker in workers:
                self.active_workers.pop(worker.worker_id, None)
            logger.error(f"Failed to create workers: {e}")
            raise

    def _allocate_bandwidth(self, workers: list[CurlWorker]) -> None:
        """
        Allocate bandwidth among workers.

        Args:
            workers: List of workers to allocate bandwidth for
        """
        allocations = self.bandwidth_manager.allocate_bandwidth(workers)
        
        for worker in workers:
            limit = allocations.get(worker.worker_id, 0)
            if limit > 0:
                logger.debug(f"Allocated {limit} bytes/sec to worker {worker.worker_id}")
            # Note: Actual bandwidth limiting would be implemented in the worker
            # using curl options like CURLOPT_MAX_RECV_SPEED_LARGE

    async def monitor_workers(self, workers: list[CurlWorker]) -> AsyncIterator[dict[str, WorkerProgress]]:
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
            while not self._shutdown_requested and any(not task.done() for task in download_tasks):
                # Collect current progress from all workers
                current_progress = {}
                worker_speeds = {}
                
                for worker in workers:
                    progress = worker.get_progress()
                    current_progress[worker.worker_id] = progress
                    worker_speeds[worker.worker_id] = progress.download_speed

                # Update bandwidth allocation based on current speeds
                self.bandwidth_manager.update_allocation(worker_speeds)
                
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
                        success = result.get('success', False) if isinstance(result, dict) else False
                        if success:
                            logger.info(f"Worker {worker.worker_id} completed successfully")
                        else:
                            error = result.get('error', 'Unknown error') if isinstance(result, dict) else str(result)
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
        Handle progress updates from workers.

        Args:
            progress: Worker progress information
        """
        self._worker_progress[progress.worker_id] = progress
        
        # Log significant progress updates
        if progress.status in (WorkerStatus.COMPLETED, WorkerStatus.FAILED):
            logger.info(
                f"Worker {progress.worker_id} {progress.status.value}: "
                f"{progress.downloaded_bytes} bytes"
            )

    async def pause_workers(self, worker_ids: list[str] | None = None) -> None:
        """
        Pause specified workers or all active workers.

        Args:
            worker_ids: List of worker IDs to pause (None for all)
        """
        target_workers = (
            [self.active_workers[wid] for wid in worker_ids if wid in self.active_workers]
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
            [self.active_workers[wid] for wid in worker_ids if wid in self.active_workers]
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
            [self.active_workers[wid] for wid in worker_ids if wid in self.active_workers]
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

        # Clear tracking data
        self.active_workers.clear()
        self._worker_progress.clear()
        
        logger.info("ConnectionManager cleanup completed")

    def __del__(self) -> None:
        """Cleanup when manager is destroyed."""
        if self.curl_multi or self.curl_share or self.active_workers:
            logger.warning("ConnectionManager destroyed without proper cleanup")