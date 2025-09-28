"""Quality controller for adaptive format selection during downloads."""

from enum import Enum
import logging

from .format_selector import FormatSelector, QualityTier
from .media_models import (
    DownloadProgress,
    DownloadStatus,
    FormatPreferences,
    MediaFormat,
)

logger = logging.getLogger(__name__)


class AdaptationTrigger(Enum):
    """Triggers for quality adaptation."""

    SLOW_SPEED = "slow_speed"
    HIGH_ERROR_RATE = "high_error_rate"
    NETWORK_CONGESTION = "network_congestion"
    USER_REQUEST = "user_request"
    BANDWIDTH_LIMIT = "bandwidth_limit"


class QualityController:
    """Controls adaptive quality selection during downloads."""

    def __init__(self, format_selector: FormatSelector) -> None:
        """
        Initialize quality controller.

        Args:
            format_selector: Format selector instance
        """
        self.format_selector = format_selector
        self.adaptation_history: dict[str, list[tuple[str, QualityTier]]] = {}
        self.performance_metrics: dict[str, dict[str, float]] = {}

        # Adaptation thresholds
        self.min_speed_threshold = 100 * 1024  # 100 KB/s
        self.adaptation_cooldown = 30.0  # seconds
        self.max_adaptations_per_download = 3

        logger.info("QualityController initialized")

    def should_adapt_quality(
        self,
        download_id: str,
        current_progress: DownloadProgress,
        current_format: MediaFormat,
        available_formats: list[MediaFormat],
    ) -> tuple[bool, AdaptationTrigger | None]:
        """
        Determine if quality adaptation is needed.

        Args:
            download_id: Unique download identifier
            current_progress: Current download progress
            current_format: Currently selected format
            available_formats: All available formats

        Returns:
            Tuple of (should_adapt, trigger_reason)
        """
        # Check if we've already adapted too many times
        if self._get_adaptation_count(download_id) >= self.max_adaptations_per_download:
            return False, None

        # Check if we're in cooldown period
        if self._is_in_cooldown(download_id):
            return False, None

        # Check for slow download speed
        if (
            current_progress.download_speed is not None
            and current_progress.download_speed < self.min_speed_threshold
        ):
            # Only adapt if lower quality alternatives exist
            lower_quality_formats = self.format_selector.get_quality_alternatives(
                available_formats, current_format, "lower"
            )

            if lower_quality_formats:
                logger.info(
                    f"Slow speed detected ({current_progress.download_speed:.0f} B/s), "
                    f"considering quality adaptation for {download_id}"
                )
                return True, AdaptationTrigger.SLOW_SPEED

        # Check for high error rate (would need error tracking from downloader)
        # This is a placeholder for future implementation

        return False, None

    def adapt_quality(
        self,
        download_id: str,
        current_format: MediaFormat,
        available_formats: list[MediaFormat],
        preferences: FormatPreferences,
        trigger: AdaptationTrigger,
    ) -> MediaFormat | None:
        """
        Adapt quality based on current conditions.

        Args:
            download_id: Unique download identifier
            current_format: Currently selected format
            available_formats: All available formats
            preferences: User preferences
            trigger: Reason for adaptation

        Returns:
            New format to use, or None if no adaptation possible
        """
        logger.info(f"Adapting quality for {download_id} due to {trigger.value}")

        # Get appropriate alternatives based on trigger
        if trigger == AdaptationTrigger.SLOW_SPEED:
            alternatives = self._get_lower_quality_alternatives(
                current_format, available_formats, preferences
            )
        elif trigger == AdaptationTrigger.BANDWIDTH_LIMIT:
            alternatives = self._get_bandwidth_efficient_alternatives(
                current_format, available_formats, preferences
            )
        else:
            alternatives = self.format_selector.get_quality_alternatives(
                available_formats, current_format, "lower"
            )

        if not alternatives:
            logger.warning(f"No quality alternatives available for {download_id}")
            return None

        # Select best alternative
        selected_alternative = alternatives[0]

        # Record adaptation
        self._record_adaptation(download_id, current_format, selected_alternative)

        logger.info(
            f"Quality adapted from {current_format.format_id} to "
            f"{selected_alternative.format_id} for {download_id}"
        )

        return selected_alternative

    def get_optimal_starting_quality(
        self,
        available_formats: list[MediaFormat],
        preferences: FormatPreferences,
        estimated_bandwidth: float | None = None,
    ) -> MediaFormat:
        """
        Get optimal starting quality based on conditions.

        Args:
            available_formats: All available formats
            preferences: User preferences
            estimated_bandwidth: Estimated bandwidth in bytes/second

        Returns:
            Optimal starting format
        """
        logger.info("Determining optimal starting quality")

        # If bandwidth is limited, start with conservative quality
        if estimated_bandwidth and estimated_bandwidth < 500 * 1024:  # < 500 KB/s
            logger.info("Limited bandwidth detected, starting with medium quality")

            # Create conservative preferences
            conservative_prefs = preferences.model_copy()
            conservative_prefs.target_quality = "medium"
            conservative_prefs.max_height = 720

            try:
                return self.format_selector.select_optimal_format(
                    available_formats, conservative_prefs, QualityTier.MEDIUM
                )
            except Exception:
                pass

        # Use normal selection
        return self.format_selector.select_optimal_format(
            available_formats, preferences, adaptive=True
        )

    def analyze_download_performance(
        self, download_id: str, progress_history: list[DownloadProgress]
    ) -> dict[str, float]:
        """
        Analyze download performance for future optimizations.

        Args:
            download_id: Download identifier
            progress_history: History of progress updates

        Returns:
            Performance metrics dictionary
        """
        if not progress_history:
            return {}

        metrics = {}

        # Calculate average speed
        speeds = [p.download_speed for p in progress_history if p.download_speed]
        if speeds:
            metrics["avg_speed"] = sum(speeds) / len(speeds)
            metrics["min_speed"] = min(speeds)
            metrics["max_speed"] = max(speeds)

        # Calculate speed stability (coefficient of variation)
        if len(speeds) > 1:
            import statistics

            mean_speed = statistics.mean(speeds)
            if mean_speed > 0:
                std_dev = statistics.stdev(speeds)
                metrics["speed_stability"] = 1 - (std_dev / mean_speed)

        # Calculate completion rate
        completed_progress = [
            p for p in progress_history if p.status == DownloadStatus.FINISHED
        ]
        metrics["completion_rate"] = len(completed_progress) / len(progress_history)

        # Store metrics for future reference
        self.performance_metrics[download_id] = metrics

        logger.info(f"Performance analysis for {download_id}: {metrics}")
        return metrics

    def get_recommended_quality_for_connection(
        self,
        available_formats: list[MediaFormat],
        connection_speed: float,  # bytes per second
        preferences: FormatPreferences,
    ) -> MediaFormat:
        """
        Recommend quality based on connection speed.

        Args:
            available_formats: Available formats
            connection_speed: Connection speed in bytes/second
            preferences: User preferences

        Returns:
            Recommended format
        """
        logger.info(
            f"Recommending quality for connection speed: {connection_speed:.0f} B/s"
        )

        # Define speed-to-quality mapping (rough estimates)
        if connection_speed >= 5 * 1024 * 1024:  # >= 5 MB/s
            target_quality = QualityTier.HIGH
            max_height = 1080
        elif connection_speed >= 2 * 1024 * 1024:  # >= 2 MB/s
            target_quality = QualityTier.MEDIUM
            max_height = 720
        elif connection_speed >= 500 * 1024:  # >= 500 KB/s
            target_quality = QualityTier.LOW
            max_height = 480
        else:
            target_quality = QualityTier.VERY_LOW
            max_height = 360

        # Create speed-optimized preferences
        speed_prefs = preferences.model_copy()
        speed_prefs.target_quality = target_quality.value
        speed_prefs.max_height = min(speed_prefs.max_height or max_height, max_height)

        try:
            return self.format_selector.select_optimal_format(
                available_formats,
                speed_prefs,
                target_quality,
                adaptive=True,
                connection_speed=connection_speed,
            )
        except Exception as e:
            logger.warning(f"Speed-based selection failed: {e}, using fallback")
            return available_formats[0]

    def _get_adaptation_count(self, download_id: str) -> int:
        """Get number of adaptations for a download."""
        return len(self.adaptation_history.get(download_id, []))

    def _is_in_cooldown(self, download_id: str) -> bool:
        """Check if download is in adaptation cooldown period."""
        history = self.adaptation_history.get(download_id, [])
        if not history:
            return False

        # Check if last adaptation was recent
        import time

        last_adaptation_time = getattr(self, f"_last_adaptation_{download_id}", 0)
        return time.time() - last_adaptation_time < self.adaptation_cooldown

    def _record_adaptation(
        self, download_id: str, from_format: MediaFormat, to_format: MediaFormat
    ) -> None:
        """Record a quality adaptation."""
        if download_id not in self.adaptation_history:
            self.adaptation_history[download_id] = []

        # Determine quality tiers
        from_quality = self._format_to_quality_tier(from_format)
        to_quality = self._format_to_quality_tier(to_format)

        self.adaptation_history[download_id].append(
            (from_format.format_id, from_quality)
        )

        # Record timestamp
        import time

        setattr(self, f"_last_adaptation_{download_id}", time.time())

    def _format_to_quality_tier(self, fmt: MediaFormat) -> QualityTier:
        """Convert format to quality tier."""
        height = self._get_format_height(fmt)
        if height is None:
            return QualityTier.MEDIUM

        if height >= 2160:
            return QualityTier.ULTRA_HIGH
        elif height >= 1080:
            return QualityTier.HIGH
        elif height >= 720:
            return QualityTier.MEDIUM
        elif height >= 480:
            return QualityTier.LOW
        else:
            return QualityTier.VERY_LOW

    def _get_format_height(self, fmt: MediaFormat) -> int | None:
        """Extract height from format resolution."""
        if not fmt.resolution:
            return None

        try:
            if "x" in fmt.resolution:
                return int(fmt.resolution.split("x")[1])
            elif "p" in fmt.resolution:
                return int(fmt.resolution.replace("p", ""))
            else:
                return int(fmt.resolution)
        except (ValueError, IndexError):
            return None

    def _get_lower_quality_alternatives(
        self,
        current_format: MediaFormat,
        available_formats: list[MediaFormat],
        preferences: FormatPreferences,
    ) -> list[MediaFormat]:
        """Get lower quality alternatives optimized for slow connections."""
        alternatives = self.format_selector.get_quality_alternatives(
            available_formats, current_format, "lower"
        )

        # Filter alternatives that are significantly lower quality
        current_height = self._get_format_height(current_format)
        if current_height:
            # Prefer alternatives that are at least 25% lower resolution
            min_height = int(current_height * 0.75)
            alternatives = [
                fmt
                for fmt in alternatives
                if self._get_format_height(fmt)
                and self._get_format_height(fmt) <= min_height
            ]

        return alternatives

    def _get_bandwidth_efficient_alternatives(
        self,
        current_format: MediaFormat,
        available_formats: list[MediaFormat],
        preferences: FormatPreferences,
    ) -> list[MediaFormat]:
        """Get bandwidth-efficient alternatives."""
        alternatives = self.format_selector.get_quality_alternatives(
            available_formats, current_format, "lower"
        )

        # Sort by estimated bandwidth efficiency (smaller filesize per quality unit)
        def efficiency_score(fmt: MediaFormat) -> float:
            height = self._get_format_height(fmt) or 480
            filesize = fmt.filesize or fmt.filesize_approx or (height * height * 0.1)
            return height / filesize if filesize > 0 else 0

        alternatives.sort(key=efficiency_score, reverse=True)
        return alternatives

    def cleanup_download_data(self, download_id: str) -> None:
        """Clean up data for completed download."""
        self.adaptation_history.pop(download_id, None)
        self.performance_metrics.pop(download_id, None)

        # Clean up timestamp attributes
        timestamp_attr = f"_last_adaptation_{download_id}"
        if hasattr(self, timestamp_attr):
            delattr(self, timestamp_attr)

        logger.debug(f"Cleaned up quality controller data for {download_id}")
