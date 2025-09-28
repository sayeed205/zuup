"""Adaptive connection scaling based on server performance and network conditions."""

from __future__ import annotations

from collections import deque
import logging
import statistics
import time
from typing import Any

logger = logging.getLogger(__name__)


class ServerPerformanceMetrics:
    """Tracks server performance metrics for adaptive scaling decisions."""

    def __init__(self, window_size: int = 30) -> None:
        """
        Initialize performance metrics tracker.

        Args:
            window_size: Number of samples to keep in sliding window
        """
        self.window_size = window_size

        # Sliding windows for metrics
        self._response_times: deque[float] = deque(maxlen=window_size)
        self._download_speeds: deque[float] = deque(maxlen=window_size)
        self._error_rates: deque[float] = deque(maxlen=window_size)
        self._connection_success_rates: deque[float] = deque(maxlen=window_size)

        # Timestamps for rate calculations
        self._last_update = time.time()
        self._sample_count = 0

        logger.debug(
            f"Initialized ServerPerformanceMetrics with window_size={window_size}"
        )

    def add_sample(
        self,
        response_time: float,
        download_speed: float,
        had_error: bool = False,
        connection_successful: bool = True,
    ) -> None:
        """
        Add a performance sample.

        Args:
            response_time: Response time in seconds
            download_speed: Download speed in bytes per second
            had_error: Whether this sample had an error
            connection_successful: Whether connection was successful
        """
        current_time = time.time()

        self._response_times.append(response_time)
        self._download_speeds.append(download_speed)
        self._error_rates.append(1.0 if had_error else 0.0)
        self._connection_success_rates.append(1.0 if connection_successful else 0.0)

        self._sample_count += 1
        self._last_update = current_time

        logger.debug(
            f"Added performance sample: response_time={response_time:.3f}s, "
            f"speed={download_speed:.0f} B/s, error={had_error}, success={connection_successful}"
        )

    def get_average_response_time(self) -> float:
        """Get average response time."""
        return statistics.mean(self._response_times) if self._response_times else 0.0

    def get_average_download_speed(self) -> float:
        """Get average download speed."""
        return statistics.mean(self._download_speeds) if self._download_speeds else 0.0

    def get_error_rate(self) -> float:
        """Get error rate (0.0 to 1.0)."""
        return statistics.mean(self._error_rates) if self._error_rates else 0.0

    def get_connection_success_rate(self) -> float:
        """Get connection success rate (0.0 to 1.0)."""
        return (
            statistics.mean(self._connection_success_rates)
            if self._connection_success_rates
            else 1.0
        )

    def get_speed_stability(self) -> float:
        """
        Get speed stability metric (lower is more stable).

        Returns:
            Coefficient of variation for download speeds
        """
        if len(self._download_speeds) < 2:
            return 0.0

        try:
            mean_speed = statistics.mean(self._download_speeds)
            if mean_speed == 0:
                return float("inf")

            stdev_speed = statistics.stdev(self._download_speeds)
            return stdev_speed / mean_speed  # Coefficient of variation
        except statistics.StatisticsError:
            return 0.0

    def get_performance_score(self) -> float:
        """
        Calculate overall performance score (0.0 to 1.0, higher is better).

        Returns:
            Performance score based on multiple metrics
        """
        if not self._download_speeds:
            return 0.5  # Neutral score with no data

        # Normalize metrics to 0-1 scale
        speed_score = min(
            1.0, self.get_average_download_speed() / (10 * 1024 * 1024)
        )  # 10MB/s = 1.0
        response_score = max(
            0.0, 1.0 - (self.get_average_response_time() / 10.0)
        )  # 10s = 0.0
        error_score = 1.0 - self.get_error_rate()
        connection_score = self.get_connection_success_rate()
        stability_score = max(0.0, 1.0 - min(1.0, self.get_speed_stability()))

        # Weighted average
        weights = {
            "speed": 0.3,
            "response": 0.2,
            "error": 0.2,
            "connection": 0.2,
            "stability": 0.1,
        }

        score = (
            weights["speed"] * speed_score
            + weights["response"] * response_score
            + weights["error"] * error_score
            + weights["connection"] * connection_score
            + weights["stability"] * stability_score
        )

        return max(0.0, min(1.0, score))

    def has_sufficient_data(self, min_samples: int = 5) -> bool:
        """Check if we have sufficient data for reliable metrics."""
        return len(self._download_speeds) >= min_samples

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "sample_count": self._sample_count,
            "window_size": self.window_size,
            "current_samples": len(self._download_speeds),
            "avg_response_time": self.get_average_response_time(),
            "avg_download_speed": self.get_average_download_speed(),
            "error_rate": self.get_error_rate(),
            "connection_success_rate": self.get_connection_success_rate(),
            "speed_stability": self.get_speed_stability(),
            "performance_score": self.get_performance_score(),
            "has_sufficient_data": self.has_sufficient_data(),
        }


class AdaptiveConnectionScaler:
    """Manages adaptive scaling of connections based on server performance."""

    def __init__(
        self,
        min_connections: int = 1,
        max_connections: int = 8,
        scale_up_threshold: float = 0.7,
        scale_down_threshold: float = 0.4,
        scale_interval: float = 30.0,  # seconds
        aggressive_scaling: bool = False,
    ) -> None:
        """
        Initialize adaptive connection scaler.

        Args:
            min_connections: Minimum number of connections
            max_connections: Maximum number of connections
            scale_up_threshold: Performance threshold for scaling up
            scale_down_threshold: Performance threshold for scaling down
            scale_interval: Minimum interval between scaling decisions
            aggressive_scaling: Enable more aggressive scaling
        """
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.scale_interval = scale_interval
        self.aggressive_scaling = aggressive_scaling

        # Current state
        self.current_connections = min_connections
        self.last_scale_time = 0.0
        self.scale_history: deque[tuple[float, int, str]] = deque(maxlen=50)

        # Performance tracking
        self.performance_metrics = ServerPerformanceMetrics()

        # Scaling statistics
        self._scale_up_count = 0
        self._scale_down_count = 0

        logger.info(
            f"Initialized AdaptiveConnectionScaler: {min_connections}-{max_connections} connections, "
            f"thresholds: up={scale_up_threshold}, down={scale_down_threshold}, "
            f"interval={scale_interval}s, aggressive={aggressive_scaling}"
        )

    def add_performance_sample(
        self,
        response_time: float,
        download_speed: float,
        had_error: bool = False,
        connection_successful: bool = True,
    ) -> None:
        """
        Add a performance sample for scaling decisions.

        Args:
            response_time: Response time in seconds
            download_speed: Download speed in bytes per second
            had_error: Whether this sample had an error
            connection_successful: Whether connection was successful
        """
        self.performance_metrics.add_sample(
            response_time, download_speed, had_error, connection_successful
        )

    def should_scale(self) -> bool:
        """Check if enough time has passed since last scaling decision."""
        current_time = time.time()
        return (current_time - self.last_scale_time) >= self.scale_interval

    def evaluate_scaling_decision(self) -> tuple[int, str]:
        """
        Evaluate whether to scale connections up or down.

        Returns:
            Tuple of (new_connection_count, reason)
        """
        if not self.should_scale():
            return self.current_connections, "too_soon"

        if not self.performance_metrics.has_sufficient_data():
            return self.current_connections, "insufficient_data"

        performance_score = self.performance_metrics.get_performance_score()
        error_rate = self.performance_metrics.get_error_rate()
        connection_success_rate = self.performance_metrics.get_connection_success_rate()

        # Determine scaling action
        new_connections = self.current_connections
        reason = "no_change"

        # Scale down conditions
        if (
            error_rate > 0.3  # High error rate
            or connection_success_rate < 0.7  # Low connection success
            or performance_score < self.scale_down_threshold
        ):
            if self.current_connections > self.min_connections:
                if self.aggressive_scaling:
                    # Aggressive scaling: reduce by 50% or at least 1
                    reduction = max(1, self.current_connections // 2)
                    new_connections = max(
                        self.min_connections, self.current_connections - reduction
                    )
                    reason = "aggressive_scale_down"
                else:
                    # Conservative scaling: reduce by 1
                    new_connections = self.current_connections - 1
                    reason = "scale_down"

        # Scale up conditions (only if not scaling down)
        elif (
            performance_score > self.scale_up_threshold
            and error_rate < 0.1
            and connection_success_rate > 0.9
        ):
            if self.current_connections < self.max_connections:
                if self.aggressive_scaling:
                    # Aggressive scaling: increase by 50% or at least 1
                    increase = max(1, self.current_connections // 2)
                    new_connections = min(
                        self.max_connections, self.current_connections + increase
                    )
                    reason = "aggressive_scale_up"
                else:
                    # Conservative scaling: increase by 1
                    new_connections = self.current_connections + 1
                    reason = "scale_up"

        # Additional conditions for scaling down
        if new_connections == self.current_connections:
            # Check for server overload indicators
            avg_response_time = self.performance_metrics.get_average_response_time()
            speed_stability = self.performance_metrics.get_speed_stability()

            if (
                avg_response_time > 10.0  # Very slow responses
                or speed_stability > 1.0  # Very unstable speeds
            ) and self.current_connections > self.min_connections:
                new_connections = max(
                    self.min_connections, self.current_connections - 1
                )
                reason = "server_overload"

        return new_connections, reason

    def apply_scaling_decision(self) -> tuple[int, str]:
        """
        Apply scaling decision and update state.

        Returns:
            Tuple of (new_connection_count, reason)
        """
        new_connections, reason = self.evaluate_scaling_decision()

        if new_connections != self.current_connections:
            old_connections = self.current_connections
            self.current_connections = new_connections
            self.last_scale_time = time.time()

            # Update statistics
            if new_connections > old_connections:
                self._scale_up_count += 1
            else:
                self._scale_down_count += 1

            # Record in history
            self.scale_history.append((self.last_scale_time, new_connections, reason))

            logger.info(
                f"Scaled connections: {old_connections} -> {new_connections} ({reason})"
            )

        return new_connections, reason

    def get_recommended_connections(self) -> int:
        """Get current recommended connection count."""
        return self.current_connections

    def force_scale_to(self, connections: int, reason: str = "manual") -> None:
        """
        Force scaling to a specific connection count.

        Args:
            connections: Target connection count
            reason: Reason for forced scaling
        """
        connections = max(self.min_connections, min(self.max_connections, connections))

        if connections != self.current_connections:
            old_connections = self.current_connections
            self.current_connections = connections
            self.last_scale_time = time.time()

            # Record in history
            self.scale_history.append((self.last_scale_time, connections, reason))

            logger.info(
                f"Forced scaling: {old_connections} -> {connections} ({reason})"
            )

    def reset_performance_data(self) -> None:
        """Reset performance metrics (useful when changing servers)."""
        self.performance_metrics = ServerPerformanceMetrics()
        logger.debug("Reset performance metrics")

    def get_scaling_stats(self) -> dict[str, Any]:
        """Get scaling statistics."""
        return {
            "current_connections": self.current_connections,
            "min_connections": self.min_connections,
            "max_connections": self.max_connections,
            "scale_up_count": self._scale_up_count,
            "scale_down_count": self._scale_down_count,
            "total_scaling_events": self._scale_up_count + self._scale_down_count,
            "last_scale_time": self.last_scale_time,
            "time_since_last_scale": time.time() - self.last_scale_time,
            "can_scale_now": self.should_scale(),
            "performance_metrics": self.performance_metrics.get_stats(),
            "recent_history": list(self.scale_history)[-10:],  # Last 10 events
        }

    def get_scaling_recommendation(self) -> dict[str, Any]:
        """
        Get detailed scaling recommendation without applying it.

        Returns:
            Dictionary with recommendation details
        """
        new_connections, reason = self.evaluate_scaling_decision()
        performance_score = self.performance_metrics.get_performance_score()

        return {
            "current_connections": self.current_connections,
            "recommended_connections": new_connections,
            "would_change": new_connections != self.current_connections,
            "reason": reason,
            "performance_score": performance_score,
            "can_scale_now": self.should_scale(),
            "has_sufficient_data": self.performance_metrics.has_sufficient_data(),
            "metrics_summary": {
                "error_rate": self.performance_metrics.get_error_rate(),
                "connection_success_rate": self.performance_metrics.get_connection_success_rate(),
                "avg_response_time": self.performance_metrics.get_average_response_time(),
                "avg_download_speed": self.performance_metrics.get_average_download_speed(),
                "speed_stability": self.performance_metrics.get_speed_stability(),
            },
        }


def create_conservative_scaler(
    min_conn: int = 1, max_conn: int = 4
) -> AdaptiveConnectionScaler:
    """Create a conservative scaler for stable servers."""
    return AdaptiveConnectionScaler(
        min_connections=min_conn,
        max_connections=max_conn,
        scale_up_threshold=0.8,
        scale_down_threshold=0.3,
        scale_interval=60.0,
        aggressive_scaling=False,
    )


def create_aggressive_scaler(
    min_conn: int = 1, max_conn: int = 16
) -> AdaptiveConnectionScaler:
    """Create an aggressive scaler for high-performance scenarios."""
    return AdaptiveConnectionScaler(
        min_connections=min_conn,
        max_connections=max_conn,
        scale_up_threshold=0.6,
        scale_down_threshold=0.5,
        scale_interval=15.0,
        aggressive_scaling=True,
    )


def create_balanced_scaler(
    min_conn: int = 1, max_conn: int = 8
) -> AdaptiveConnectionScaler:
    """Create a balanced scaler for general use."""
    return AdaptiveConnectionScaler(
        min_connections=min_conn,
        max_connections=max_conn,
        scale_up_threshold=0.7,
        scale_down_threshold=0.4,
        scale_interval=30.0,
        aggressive_scaling=False,
    )
