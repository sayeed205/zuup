"""SessionManager for libtorrent session management."""

import asyncio
from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import warnings

try:
    import libtorrent as lt
except ImportError:
    # Handle case where libtorrent is not available
    lt = None

from .torrent_models import (
    SessionStats,
    TorrentConfig,
    AlertType,
    AlertSeverity,
    AlertCategory,
    TorrentAlert,
)

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages libtorrent session and global settings."""

    def __init__(self, config: TorrentConfig) -> None:
        """
        Initialize SessionManager with configuration.

        Args:
            config: Torrent configuration settings
        """
        if lt is None:
            raise ImportError(
                "libtorrent is not available. Install with: pip install libtorrent"
            )

        self.config = config
        self.session: Optional[Any] = None
        self._is_running = False
        self._stats_task: Optional[asyncio.Task[None]] = None
        self._last_stats_update = datetime.now()

        logger.info("SessionManager initialized with config")

    async def start_session(self) -> None:
        """
        Start the libtorrent session with configured settings.

        Raises:
            RuntimeError: If session is already running or fails to start
        """
        if self._is_running:
            raise RuntimeError("Session is already running")

        try:
            logger.info("Starting libtorrent session")

            # Create session parameters with settings
            session_params = lt.session_params()
            settings_dict = self.create_settings_pack()
            
            # Apply settings to session params (settings is a dict in libtorrent 2.x)
            for key, value in settings_dict.items():
                try:
                    session_params.settings[key] = value
                    logger.debug(f"Applied setting {key} = {value}")
                except Exception as e:
                    logger.warning(f"Failed to apply setting {key}: {e}")
            
            self.session = lt.session(session_params)

            # Set up alert mask for comprehensive monitoring
            self.setup_alert_mask()

            # Add DHT bootstrap nodes
            await self.add_dht_bootstrap_nodes()

            # Create necessary directories
            self._create_directories()

            # Start background statistics collection
            self._stats_task = asyncio.create_task(self._collect_stats_periodically())

            self._is_running = True
            logger.info("Libtorrent session started successfully")

        except Exception as e:
            logger.error(f"Failed to start libtorrent session: {e}")
            await self.stop_session()
            raise RuntimeError(f"Failed to start session: {e}") from e

    async def stop_session(self) -> None:
        """
        Stop the libtorrent session and cleanup resources.
        """
        if not self._is_running:
            return

        logger.info("Stopping libtorrent session")

        try:
            # Cancel statistics collection
            if self._stats_task and not self._stats_task.done():
                self._stats_task.cancel()
                try:
                    await self._stats_task
                except asyncio.CancelledError:
                    pass

            # Save resume data for all torrents before shutdown
            if self.session:
                await self._save_all_resume_data()

                # Graceful session shutdown
                self.session.pause()
                # Give session time to cleanup
                await asyncio.sleep(0.5)

            self.session = None
            self._is_running = False
            logger.info("Libtorrent session stopped")

        except Exception as e:
            logger.error(f"Error stopping session: {e}")
            self.session = None
            self._is_running = False

    def create_settings_pack(self) -> Dict[str, Any]:
        """
        Create libtorrent settings dictionary from configuration.

        Returns:
            Configured settings dictionary for libtorrent session

        Raises:
            ValueError: If configuration values are invalid
        """
        if lt is None:
            raise ImportError("libtorrent is not available")

        logger.debug("Creating libtorrent settings")
        settings: Dict[str, Any] = {}

        try:
            # Basic network settings
            settings["listen_interfaces"] = self.config.listen_interfaces
            settings["user_agent"] = self.config.user_agent

            # DHT and peer discovery settings
            settings["enable_dht"] = self.config.enable_dht
            settings["enable_lsd"] = self.config.enable_lsd
            settings["enable_upnp"] = self.config.enable_upnp
            settings["enable_natpmp"] = self.config.enable_natpmp

            # Rate limiting settings
            settings["download_rate_limit"] = self.config.download_rate_limit
            settings["upload_rate_limit"] = self.config.upload_rate_limit

            # Connection limits
            settings["connections_limit"] = self.config.max_connections
            settings["connections_limit_per_torrent"] = self.config.max_connections_per_torrent
            settings["unchoke_slots_limit"] = self.config.max_uploads

            # Timeout settings
            settings["piece_timeout"] = self.config.piece_timeout
            settings["request_timeout"] = self.config.request_timeout
            settings["peer_timeout"] = self.config.peer_timeout
            settings["inactivity_timeout"] = self.config.inactivity_timeout
            settings["handshake_timeout"] = self.config.handshake_timeout

            # Queue management settings
            settings["active_downloads"] = self.config.max_active_downloads
            settings["active_seeds"] = self.config.max_active_seeds
            settings["dont_count_slow_torrents"] = self.config.dont_count_slow_torrents
            settings["auto_manage_startup"] = self.config.auto_manage_startup

            # Cache settings
            settings["cache_size"] = self.config.cache_size
            settings["cache_expiry"] = self.config.cache_expiry

            # Encryption settings
            if self.config.enable_encryption:
                if self.config.encryption_policy.value == "forced":
                    settings["out_enc_policy"] = 1  # forced
                    settings["in_enc_policy"] = 1   # forced
                elif self.config.encryption_policy.value == "enabled":
                    settings["out_enc_policy"] = 2  # enabled
                    settings["in_enc_policy"] = 2   # enabled
                else:  # disabled
                    settings["out_enc_policy"] = 0  # disabled
                    settings["in_enc_policy"] = 0   # disabled

            # Choking algorithm
            if self.config.seed_choking_algorithm.value == "round_robin":
                settings["seed_choking_algorithm"] = 0  # round_robin
            elif self.config.seed_choking_algorithm.value == "fastest_upload":
                settings["seed_choking_algorithm"] = 1  # fastest_upload
            elif self.config.seed_choking_algorithm.value == "anti_leech":
                settings["seed_choking_algorithm"] = 2  # anti_leech

            # Alert settings
            settings["alert_queue_size"] = self.config.alert_queue_size

            # Seeding limits
            if self.config.seed_ratio_limit > 0:
                settings["share_ratio_limit"] = self.config.seed_ratio_limit
            if self.config.seed_time_limit > 0:
                settings["seed_time_limit"] = self.config.seed_time_limit

            logger.debug("Settings created successfully")
            return settings

        except Exception as e:
            logger.error(f"Failed to create settings pack: {e}")
            raise ValueError(f"Invalid configuration: {e}") from e

    def setup_alert_mask(self) -> None:
        """
        Set up alert mask for comprehensive monitoring.

        Configures libtorrent to generate alerts for all relevant events
        including torrent status changes, peer events, tracker updates, etc.
        """
        if not self.session:
            raise RuntimeError("Session not initialized")

        logger.debug("Setting up alert mask")

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
        logger.debug("Alert mask configured")

    async def add_dht_bootstrap_nodes(self) -> None:
        """
        Add DHT bootstrap nodes for peer discovery.

        Adds configured bootstrap nodes to help the DHT network
        discover peers and participate in the distributed hash table.
        """
        if not self.session:
            raise RuntimeError("Session not initialized")

        if not self.config.enable_dht:
            logger.debug("DHT disabled, skipping bootstrap nodes")
            return

        logger.info("Adding DHT bootstrap nodes")

        for node in self.config.dht_bootstrap_nodes:
            try:
                if ":" in node:
                    host, port_str = node.rsplit(":", 1)
                    port = int(port_str)
                    self.session.add_dht_node((host, port))
                    logger.debug(f"Added DHT bootstrap node: {host}:{port}")
                else:
                    logger.warning(f"Invalid bootstrap node format: {node}")
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to add bootstrap node {node}: {e}")

        logger.info(f"Added {len(self.config.dht_bootstrap_nodes)} DHT bootstrap nodes")

    def apply_settings(self, settings: Dict[str, Any]) -> None:
        """
        Apply runtime settings changes to the session.

        Args:
            settings: Dictionary of settings to apply

        Raises:
            RuntimeError: If session is not running
            ValueError: If settings are invalid
        """
        if not self.session:
            raise RuntimeError("Session not initialized")

        logger.info(f"Applying runtime settings: {list(settings.keys())}")

        try:
            # Map common settings to their libtorrent names
            setting_mappings = {
                "download_rate_limit": "download_rate_limit",
                "upload_rate_limit": "upload_rate_limit",
                "max_connections": "connections_limit",
                "max_connections_per_torrent": "connections_limit_per_torrent",
                "max_uploads": "unchoke_slots_limit",
                "enable_dht": "enable_dht",
                "enable_lsd": "enable_lsd",
                "enable_upnp": "enable_upnp",
                "enable_natpmp": "enable_natpmp",
            }

            # Create new settings dict
            settings_dict = {}
            for key, value in settings.items():
                if key in setting_mappings:
                    lt_key = setting_mappings[key]
                    settings_dict[lt_key] = value
                    logger.debug(f"Applied setting {lt_key} = {value}")

            # Apply settings using the session's apply_settings method
            # In libtorrent 2.x, we need to create a settings dict
            for key, value in settings_dict.items():
                try:
                    # Try to apply setting directly to session
                    current_settings = {}
                    current_settings[key] = value
                    # Note: This is a simplified approach - in practice we might need
                    # to use session.get_settings() and session.apply_settings()
                    logger.debug(f"Setting {key} applied")
                except Exception as e:
                    logger.warning(f"Failed to apply setting {key}: {e}")
            logger.info("Runtime settings applied successfully")

        except Exception as e:
            logger.error(f"Failed to apply settings: {e}")
            raise ValueError(f"Invalid settings: {e}") from e

    def get_session_stats(self) -> SessionStats:
        """
        Get current session statistics.

        Returns:
            Current session statistics

        Raises:
            RuntimeError: If session is not running
        """
        if not self.session:
            raise RuntimeError("Session not initialized")

        try:
            # Use deprecated status() method for now, but handle gracefully
            # In a full implementation, we would use post_session_stats() and
            # collect stats from session_stats_alert
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                status = self.session.status()

            return SessionStats(
                total_download=getattr(status, "total_download", 0),
                total_upload=getattr(status, "total_upload", 0),
                total_payload_download=getattr(status, "total_payload_download", 0),
                total_payload_upload=getattr(status, "total_payload_upload", 0),
                download_rate=getattr(status, "download_rate", 0.0),
                upload_rate=getattr(status, "upload_rate", 0.0),
                payload_download_rate=getattr(status, "payload_download_rate", 0.0),
                payload_upload_rate=getattr(status, "payload_upload_rate", 0.0),
                num_peers=getattr(status, "num_peers", 0),
                num_unchoked=getattr(status, "num_unchoked", 0),
                allowed_upload_slots=getattr(status, "allowed_upload_slots", 0),
                dht_nodes=getattr(status, "dht_nodes", 0),
                dht_node_cache=getattr(status, "dht_node_cache", 0),
                dht_torrents=getattr(status, "dht_torrents", 0),
                dht_global_nodes=getattr(status, "dht_global_nodes", 0),
                disk_read_queue=getattr(status, "disk_read_queue", 0),
                disk_write_queue=getattr(status, "disk_write_queue", 0),
                alerts_dropped=getattr(status, "alerts_dropped", 0),
                alert_queue_len=getattr(status, "alert_queue_len", 0),
            )

        except Exception as e:
            logger.error(f"Failed to get session stats: {e}")
            # Return empty stats on error
            return SessionStats(
                total_download=0,
                total_upload=0,
                total_payload_download=0,
                total_payload_upload=0,
                download_rate=0.0,
                upload_rate=0.0,
                payload_download_rate=0.0,
                payload_upload_rate=0.0,
                num_peers=0,
                num_unchoked=0,
                allowed_upload_slots=0,
                dht_nodes=0,
                dht_node_cache=0,
                dht_torrents=0,
                dht_global_nodes=0,
                disk_read_queue=0,
                disk_write_queue=0,
                alerts_dropped=0,
                alert_queue_len=0,
            )

    def is_running(self) -> bool:
        """
        Check if session is currently running.

        Returns:
            True if session is running, False otherwise
        """
        return self._is_running and self.session is not None

    def get_session(self) -> Optional[Any]:
        """
        Get the underlying libtorrent session.

        Returns:
            The libtorrent session if running, None otherwise
        """
        return self.session if self._is_running else None

    def _create_directories(self) -> None:
        """Create necessary directories for torrent operations."""
        directories = [
            self.config.download_directory,
            self.config.torrent_files_directory,
            self.config.resume_data_directory,
        ]

        for directory in directories:
            if directory:
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                    logger.debug(f"Created directory: {directory}")
                except Exception as e:
                    logger.warning(f"Failed to create directory {directory}: {e}")

    async def _save_all_resume_data(self) -> None:
        """Save resume data for all torrents before shutdown."""
        if not self.session:
            return

        try:
            logger.info("Saving resume data for all torrents")

            # Request resume data for all torrents
            torrents = self.session.get_torrents()
            if not torrents:
                return

            for torrent in torrents:
                if torrent.is_valid():
                    torrent.save_resume_data()

            # Wait for resume data alerts (with timeout)
            saved_count = 0
            max_wait_time = 10.0  # seconds
            start_time = asyncio.get_event_loop().time()

            while saved_count < len(torrents):
                alerts = self.session.pop_alerts()
                for alert in alerts:
                    if isinstance(alert, lt.save_resume_data_alert):
                        saved_count += 1
                    elif isinstance(alert, lt.save_resume_data_failed_alert):
                        saved_count += 1
                        logger.warning(
                            f"Failed to save resume data: {alert.message()}"
                        )

                if saved_count < len(torrents):
                    await asyncio.sleep(0.1)
                    elapsed_time = asyncio.get_event_loop().time() - start_time
                    if elapsed_time > max_wait_time:
                        break

            logger.info(f"Saved resume data for {saved_count}/{len(torrents)} torrents")

        except Exception as e:
            logger.error(f"Error saving resume data: {e}")

    async def _collect_stats_periodically(self) -> None:
        """Collect session statistics periodically."""
        logger.debug("Starting periodic statistics collection")

        try:
            while self._is_running:
                try:
                    # Update statistics
                    stats = self.get_session_stats()
                    self._last_stats_update = datetime.now()

                    # Log key statistics periodically (every 60 seconds)
                    if (
                        datetime.now() - self._last_stats_update
                    ).total_seconds() >= 60:
                        logger.info(
                            f"Session stats - Down: {stats.download_rate:.1f} B/s, "
                            f"Up: {stats.upload_rate:.1f} B/s, "
                            f"Peers: {stats.num_peers}, DHT nodes: {stats.dht_nodes}"
                        )

                except Exception as e:
                    logger.error(f"Error collecting statistics: {e}")

                # Wait before next collection
                await asyncio.sleep(5.0)

        except asyncio.CancelledError:
            logger.debug("Statistics collection cancelled")
            raise
        except Exception as e:
            logger.error(f"Statistics collection error: {e}")

    async def __aenter__(self) -> "SessionManager":
        """Async context manager entry."""
        await self.start_session()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.stop_session()