"""AlertManager for structured alert processing from libtorrent."""

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime
import logging
from typing import Any

try:
    import libtorrent as lt
except ImportError:
    # Handle case where libtorrent is not available
    lt = None

from .torrent_models import (
    AlertCategory,
    AlertSeverity,
    AlertType,
    TorrentAlert,
)

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Manages libtorrent alert processing and converts them to structured events.

    Provides real-time alert streaming with filtering, categorization, and
    conversion from libtorrent's native alert system to structured TorrentAlert objects.
    """

    def __init__(self, session: Any) -> None:
        """
        Initialize AlertManager with libtorrent session.

        Args:
            session: libtorrent session instance

        Raises:
            ImportError: If libtorrent is not available
            ValueError: If session is None
        """
        if lt is None:
            raise ImportError(
                "libtorrent is not available. Install with: pip install libtorrent"
            )

        if session is None:
            raise ValueError("Session cannot be None")

        self.session = session
        self._is_running = False
        self._alert_filters: set[AlertType] = set()
        self._severity_filter = AlertSeverity.DEBUG  # Include all by default
        self._category_filters: set[AlertCategory] = set()
        self._alert_counter = 0

        # Alert type mappings from libtorrent to our enum
        self._alert_type_mappings = self._create_alert_type_mappings()

        logger.info("AlertManager initialized")

    def setup_alert_mask(self) -> None:
        """
        Set up alert mask for comprehensive monitoring.

        Configures libtorrent to generate alerts for all relevant events
        including torrent status changes, peer events, tracker updates, etc.
        """
        if not self.session:
            raise RuntimeError("Session not initialized")

        logger.debug("Setting up comprehensive alert mask")

        # Enable comprehensive alert monitoring using libtorrent 2.x API
        alert_mask = (
            lt.alert_category.error
            | lt.alert_category.peer
            | lt.alert_category.port_mapping
            | lt.alert_category.storage
            | lt.alert_category.tracker
            | lt.alert_category.status
            | lt.alert_category.ip_block
            | lt.alert_category.performance_warning
            | lt.alert_category.dht
            | lt.alert_category.stats
            | lt.alert_category.session_log
            | lt.alert_category.torrent_log
            | lt.alert_category.peer_log
            | lt.alert_category.incoming_request
            | lt.alert_category.dht_log
            | lt.alert_category.dht_operation
            | lt.alert_category.port_mapping_log
            | lt.alert_category.picker_log
        )

        self.session.set_alert_mask(alert_mask)
        logger.debug("Alert mask configured for comprehensive monitoring")

    async def process_alerts(self) -> AsyncIterator[TorrentAlert]:
        """
        Process libtorrent alerts and yield structured events.

        This is the main method for real-time alert processing. It continuously
        polls libtorrent for new alerts, converts them to structured format,
        applies filtering, and yields them for consumption.

        Yields:
            TorrentAlert: Structured alert objects

        Raises:
            RuntimeError: If session is not initialized
        """
        if not self.session:
            raise RuntimeError("Session not initialized")

        self._is_running = True
        logger.info("Starting alert processing")

        try:
            while self._is_running:
                # Pop all available alerts from libtorrent
                alerts = self.session.pop_alerts()

                if not alerts:
                    # No alerts available, wait briefly to prevent busy waiting
                    await asyncio.sleep(0.1)
                    continue

                # Process each alert
                for alert in alerts:
                    try:
                        structured_alert = self.convert_alert(alert)
                        if structured_alert and self._should_include_alert(structured_alert):
                            self._alert_counter += 1
                            structured_alert.alert_id = self._alert_counter
                            yield structured_alert

                    except Exception as e:
                        logger.error(f"Error processing alert {type(alert).__name__}: {e}")
                        # Continue processing other alerts even if one fails

                # Brief pause to prevent excessive CPU usage
                await asyncio.sleep(0.05)

        except asyncio.CancelledError:
            logger.info("Alert processing cancelled")
            raise
        except Exception as e:
            logger.error(f"Alert processing error: {e}")
            raise
        finally:
            self._is_running = False
            logger.info("Alert processing stopped")

    def convert_alert(self, alert: Any) -> TorrentAlert | None:
        """
        Convert libtorrent alert to structured format.

        Args:
            alert: libtorrent alert object

        Returns:
            Structured TorrentAlert object or None if conversion fails
        """
        try:
            # Get alert type
            alert_type = self._get_alert_type(alert)
            if not alert_type:
                logger.debug(f"Unsupported alert type: {type(alert).__name__}")
                return None

            # Extract torrent ID if available
            torrent_id = self._extract_torrent_id(alert)

            # Get alert message
            message = self._get_alert_message(alert)

            # Determine severity and category
            severity = self._get_alert_severity(alert)
            category = self._categorize_alert(alert)

            # Extract additional data
            data = self._extract_alert_data(alert)

            return TorrentAlert(
                alert_type=alert_type,
                torrent_id=torrent_id,
                message=message,
                timestamp=datetime.now(),
                severity=severity,
                category=category,
                data=data,
                source="libtorrent"
            )

        except Exception as e:
            logger.error(f"Failed to convert alert {type(alert).__name__}: {e}")
            return None

    def set_alert_filters(
        self,
        alert_types: set[AlertType] | None = None,
        severity_filter: AlertSeverity | None = None,
        categories: set[AlertCategory] | None = None
    ) -> None:
        """
        Set alert filtering criteria.

        Args:
            alert_types: Set of alert types to include (None = include all)
            severity_filter: Minimum severity level to include
            categories: Set of categories to include (None = include all)
        """
        if alert_types is not None:
            self._alert_filters = alert_types
            logger.debug(f"Alert type filter set: {len(alert_types)} types")

        if severity_filter is not None:
            self._severity_filter = severity_filter
            logger.debug(f"Severity filter set: {severity_filter.value}")

        if categories is not None:
            self._category_filters = categories
            logger.debug(f"Category filter set: {len(categories)} categories")

    def clear_filters(self) -> None:
        """Clear all alert filters."""
        self._alert_filters.clear()
        self._category_filters.clear()
        self._severity_filter = AlertSeverity.DEBUG
        logger.debug("All alert filters cleared")

    def stop_processing(self) -> None:
        """Stop alert processing."""
        self._is_running = False
        logger.info("Alert processing stop requested")

    def get_alert_stats(self) -> dict[str, Any]:
        """
        Get alert processing statistics.

        Returns:
            Dictionary containing alert processing statistics
        """
        return {
            "is_running": self._is_running,
            "total_alerts_processed": self._alert_counter,
            "active_filters": {
                "alert_types": len(self._alert_filters),
                "severity_filter": self._severity_filter.value,
                "categories": len(self._category_filters)
            }
        }

    def _create_alert_type_mappings(self) -> dict[str, AlertType]:
        """Create mappings from libtorrent alert types to our AlertType enum."""
        if lt is None:
            return {}

        mappings = {}

        # Torrent lifecycle alerts
        mappings["torrent_added_alert"] = AlertType.TORRENT_ADDED
        mappings["torrent_removed_alert"] = AlertType.TORRENT_REMOVED
        mappings["torrent_finished_alert"] = AlertType.TORRENT_FINISHED
        mappings["torrent_paused_alert"] = AlertType.TORRENT_PAUSED
        mappings["torrent_resumed_alert"] = AlertType.TORRENT_RESUMED
        mappings["torrent_error_alert"] = AlertType.TORRENT_ERROR
        mappings["torrent_need_cert_alert"] = AlertType.TORRENT_NEED_CERT

        # Download progress alerts
        mappings["piece_finished_alert"] = AlertType.PIECE_FINISHED
        mappings["block_finished_alert"] = AlertType.BLOCK_FINISHED
        mappings["block_downloading_alert"] = AlertType.BLOCK_DOWNLOADING
        mappings["block_timeout_alert"] = AlertType.BLOCK_TIMEOUT

        # Tracker alerts
        mappings["tracker_announce_alert"] = AlertType.TRACKER_ANNOUNCE
        mappings["tracker_error_alert"] = AlertType.TRACKER_ERROR
        mappings["tracker_warning_alert"] = AlertType.TRACKER_WARNING
        mappings["tracker_reply_alert"] = AlertType.TRACKER_REPLY

        # DHT alerts
        mappings["dht_announce_alert"] = AlertType.DHT_ANNOUNCE
        mappings["dht_get_peers_alert"] = AlertType.DHT_GET_PEERS
        mappings["dht_bootstrap_alert"] = AlertType.DHT_BOOTSTRAP
        mappings["dht_error_alert"] = AlertType.DHT_ERROR

        # Peer alerts
        mappings["peer_connect_alert"] = AlertType.PEER_CONNECT
        mappings["peer_disconnected_alert"] = AlertType.PEER_DISCONNECT
        mappings["peer_ban_alert"] = AlertType.PEER_BAN
        mappings["peer_unsnubbed_alert"] = AlertType.PEER_UNSNUBBED
        mappings["peer_snubbed_alert"] = AlertType.PEER_SNUBBED
        mappings["peer_error_alert"] = AlertType.PEER_ERROR

        # Metadata alerts
        mappings["metadata_received_alert"] = AlertType.METADATA_RECEIVED
        mappings["metadata_failed_alert"] = AlertType.METADATA_FAILED

        # Storage alerts
        mappings["save_resume_data_alert"] = AlertType.SAVE_RESUME_DATA
        mappings["save_resume_data_failed_alert"] = AlertType.SAVE_RESUME_DATA_FAILED
        mappings["storage_moved_alert"] = AlertType.STORAGE_MOVED
        mappings["storage_moved_failed_alert"] = AlertType.STORAGE_MOVED_FAILED
        mappings["file_error_alert"] = AlertType.FILE_ERROR

        # Session alerts
        mappings["session_stats_alert"] = AlertType.SESSION_STATS
        mappings["listen_failed_alert"] = AlertType.LISTEN_FAILED
        mappings["listen_succeeded_alert"] = AlertType.LISTEN_SUCCEEDED
        mappings["portmap_alert"] = AlertType.PORTMAP
        mappings["portmap_error_alert"] = AlertType.PORTMAP_ERROR

        # Performance alerts
        mappings["performance_alert"] = AlertType.PERFORMANCE_ALERT
        mappings["stats_alert"] = AlertType.STATS
        mappings["cache_flushed_alert"] = AlertType.CACHE_FLUSHED

        return mappings

    def _get_alert_type(self, alert: Any) -> AlertType | None:
        """Get AlertType from libtorrent alert."""
        alert_class_name = type(alert).__name__
        # Handle both real libtorrent alerts and mock alerts
        if hasattr(alert, 'alert_type'):
            alert_class_name = alert.alert_type
        return self._alert_type_mappings.get(alert_class_name)

    def _extract_torrent_id(self, alert: Any) -> str | None:
        """Extract torrent ID (info hash) from alert if available."""
        try:
            # Check if alert has torrent handle
            if hasattr(alert, 'handle') and alert.handle.is_valid():
                info_hash = alert.handle.info_hash()
                return str(info_hash)

            # Some alerts might have torrent_name or other identifiers
            if hasattr(alert, 'torrent_name'):
                return str(alert.torrent_name)

        except Exception as e:
            logger.debug(f"Could not extract torrent ID from alert: {e}")

        return None

    def _get_alert_message(self, alert: Any) -> str:
        """Get message from libtorrent alert."""
        try:
            if hasattr(alert, 'message'):
                return str(alert.message())
            elif hasattr(alert, 'what'):
                return str(alert.what())
            else:
                return f"{type(alert).__name__}"
        except Exception:
            return f"Alert: {type(alert).__name__}"

    def _get_alert_severity(self, alert: Any) -> AlertSeverity:
        """Determine alert severity based on alert type and content."""
        alert_class_name = type(alert).__name__
        # Handle mock alerts
        if hasattr(alert, 'alert_type'):
            alert_class_name = alert.alert_type

        alert_name_lower = alert_class_name.lower()

        # Critical alerts
        if any(keyword in alert_name_lower for keyword in ['error', 'failed', 'ban']):
            return AlertSeverity.ERROR

        # Warning alerts
        if any(keyword in alert_name_lower for keyword in ['warning', 'timeout', 'disconnect']):
            return AlertSeverity.WARNING

        # Info alerts
        if any(keyword in alert_name_lower for keyword in [
            'finished', 'added', 'removed', 'connect', 'announce', 'received'
        ]):
            return AlertSeverity.INFO

        # Debug alerts (performance, stats, etc.)
        if any(keyword in alert_name_lower for keyword in [
            'stats', 'performance', 'log', 'block_downloading'
        ]):
            return AlertSeverity.DEBUG

        return AlertSeverity.INFO  # Default

    def _categorize_alert(self, alert: Any) -> AlertCategory:
        """Categorize alert based on its type."""
        alert_class_name = type(alert).__name__
        # Handle mock alerts
        if hasattr(alert, 'alert_type'):
            alert_class_name = alert.alert_type

        alert_name_lower = alert_class_name.lower()

        # Define category mappings to reduce return statements
        category_mappings = [
            (['torrent'], AlertCategory.TORRENT),
            (['tracker'], AlertCategory.TRACKER),
            (['peer'], AlertCategory.PEER),
            (['dht'], AlertCategory.DHT),
            (['storage', 'file', 'save'], AlertCategory.STORAGE),
            (['session', 'listen', 'portmap'], AlertCategory.SESSION),
            (['performance', 'stats', 'cache'], AlertCategory.PERFORMANCE),
            (['error', 'failed'], AlertCategory.ERROR),
        ]

        for keywords, category in category_mappings:
            if any(keyword in alert_name_lower for keyword in keywords):
                return category

        return AlertCategory.TORRENT  # Default

    def _extract_alert_data(self, alert: Any) -> dict[str, Any]:
        """Extract additional data from alert."""
        data = {}

        try:
            # Common alert attributes to extract
            attributes = [
                'url', 'endpoint', 'error_code', 'error', 'piece', 'block',
                'bytes_downloaded', 'bytes_uploaded', 'num_peers', 'interval',
                'min_interval', 'complete', 'incomplete', 'downloaded'
            ]

            for attr in attributes:
                if hasattr(alert, attr):
                    value = getattr(alert, attr)
                    # Convert complex objects to strings
                    if hasattr(value, '__str__'):
                        data[attr] = str(value)
                    else:
                        data[attr] = value

        except Exception as e:
            logger.debug(f"Error extracting alert data: {e}")

        return data

    def _should_include_alert(self, alert: TorrentAlert) -> bool:
        """Check if alert should be included based on filters."""
        # Check alert type filter
        if self._alert_filters and alert.alert_type not in self._alert_filters:
            return False

        # Check severity filter - include alerts at or above the filter level
        severity_levels = {
            AlertSeverity.DEBUG: 0,
            AlertSeverity.INFO: 1,
            AlertSeverity.WARNING: 2,
            AlertSeverity.ERROR: 3,
            AlertSeverity.CRITICAL: 4
        }

        # Alert must be at or above the minimum severity level
        if severity_levels[alert.severity] < severity_levels[self._severity_filter]:
            return False

        # Check category filter - if categories are specified, alert must be in one of them
        if self._category_filters and alert.category not in self._category_filters:
            return False

        return True
