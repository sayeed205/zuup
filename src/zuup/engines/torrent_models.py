"""Comprehensive torrent data models and type definitions for libtorrent integration."""

from datetime import datetime, timedelta
from enum import Enum
import ipaddress
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator

# Constants for validation
MAX_TORRENT_PRIORITY = 255
MAX_FILE_PRIORITY = 7
MAX_PORT_NUMBER = 65535
MIN_PORT_NUMBER = 1
PEER_ID_PREFIX_LENGTH = 8
PROGRESS_TOLERANCE = 0.01

# Health calculation constants
GOOD_SEED_RATIO_THRESHOLD = 0.5
MIN_SEED_RATIO_THRESHOLD = 0.2
GOOD_AVAILABILITY_THRESHOLD = 2.0
MIN_AVAILABILITY_THRESHOLD = 1.0
GOOD_COPIES_THRESHOLD = 2.0
MIN_COPIES_THRESHOLD = 1.0
HEALTH_THRESHOLD = 0.5
SEED_SHORTAGE_THRESHOLD = 0.1
STALL_AVAILABILITY_THRESHOLD = 0.5

# ============================================================================
# Core Torrent State and Status Models
# ============================================================================


class TorrentState(Enum):
    """Torrent state enumeration matching libtorrent states."""

    QUEUED_FOR_CHECKING = "queued_for_checking"
    CHECKING_FILES = "checking_files"
    DOWNLOADING_METADATA = "downloading_metadata"
    DOWNLOADING = "downloading"
    FINISHED = "finished"
    SEEDING = "seeding"
    ALLOCATING = "allocating"
    CHECKING_RESUME_DATA = "checking_resume_data"


class TorrentInfo(BaseModel):
    """Torrent information for adding torrents to session."""

    source: str  # File path or magnet link
    save_path: Path
    name: str | None = None
    priority: int = 1
    sequential_download: bool = False
    file_priorities: list[int] = Field(default_factory=list)
    trackers: list[str] = Field(default_factory=list)

    # Advanced torrent settings
    auto_managed: bool = True
    paused: bool = False
    duplicate_is_error: bool = True
    storage_mode: str = "sparse"  # sparse, allocate, compact

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v: int) -> int:
        """Validate priority is within valid range."""
        if not (0 <= v <= MAX_TORRENT_PRIORITY):
            raise ValueError(f"Priority must be between 0 and {MAX_TORRENT_PRIORITY}")
        return v

    @field_validator("file_priorities")
    @classmethod
    def validate_file_priorities(cls, v: list[int]) -> list[int]:
        """Validate file priorities are within valid range."""
        for priority in v:
            if not (0 <= priority <= MAX_FILE_PRIORITY):
                raise ValueError(
                    f"File priorities must be between 0 and {MAX_FILE_PRIORITY}"
                )
        return v

    @field_validator("storage_mode")
    @classmethod
    def validate_storage_mode(cls, v: str) -> str:
        """Validate storage mode."""
        valid_modes = {"sparse", "allocate", "compact"}
        if v not in valid_modes:
            raise ValueError(f"Storage mode must be one of {valid_modes}")
        return v


class TorrentStatus(BaseModel):
    """Comprehensive torrent status information."""

    # Basic identification
    info_hash: str
    name: str
    state: TorrentState

    # Progress information
    progress: float = Field(ge=0.0, le=1.0)  # 0.0 to 1.0
    downloaded_bytes: int = Field(ge=0)
    uploaded_bytes: int = Field(ge=0)
    total_bytes: int = Field(ge=0)
    total_wanted_bytes: int = Field(ge=0)  # Bytes of files we want to download

    # Speed information
    download_speed: float = Field(ge=0.0)  # bytes per second
    upload_speed: float = Field(ge=0.0)  # bytes per second
    payload_download_speed: float = Field(ge=0.0)  # actual data, not protocol overhead
    payload_upload_speed: float = Field(ge=0.0)

    # Peer and seed information
    num_peers: int = Field(ge=0)
    num_seeds: int = Field(ge=0)
    num_complete: int = Field(ge=0)  # Seeds in swarm
    num_incomplete: int = Field(ge=0)  # Peers in swarm
    list_peers: int = Field(ge=0)  # Peers we know about
    list_seeds: int = Field(ge=0)  # Seeds we know about
    connect_candidates: int = Field(ge=0)  # Peers we can connect to

    # Ratio and sharing
    ratio: float = Field(ge=0.0)
    all_time_upload: int = Field(ge=0)
    all_time_download: int = Field(ge=0)

    # Time information
    eta: timedelta | None = None
    active_time: int = Field(ge=0)  # seconds
    finished_time: int = Field(ge=0)  # seconds
    seeding_time: int = Field(ge=0)  # seconds
    time_since_download: int = Field(ge=0)  # seconds
    time_since_upload: int = Field(ge=0)  # seconds

    # Piece information
    pieces_have: int = Field(ge=0)
    pieces_total: int = Field(ge=0)
    availability: float = Field(ge=0.0)  # Average availability of pieces
    distributed_copies: float = Field(ge=0.0)  # Number of distributed copies

    # Queue and priority
    queue_position: int = Field(ge=-1)  # -1 if not queued
    priority: int = Field(ge=0, le=255)

    # Error information
    error: str | None = None
    error_file: str | None = None

    # Advanced status
    is_auto_managed: bool = True
    is_paused: bool = False
    is_finished: bool = False
    is_seeding: bool = False
    has_metadata: bool = False
    moving_storage: bool = False
    announcing_to_trackers: bool = False
    announcing_to_lsd: bool = False
    announcing_to_dht: bool = False

    @model_validator(mode="after")
    def validate_status_consistency(self) -> "TorrentStatus":
        """Validate status field consistency."""
        # Ensure downloaded bytes doesn't exceed total bytes
        if self.downloaded_bytes > self.total_bytes:
            raise ValueError("Downloaded bytes cannot exceed total bytes")

        # Ensure pieces consistency
        if self.pieces_have > self.pieces_total:
            raise ValueError("Pieces have cannot exceed pieces total")

        # Validate progress consistency with pieces
        if self.pieces_total > 0:
            calculated_progress = self.pieces_have / self.pieces_total
            # Allow small floating point differences
            if abs(self.progress - calculated_progress) > PROGRESS_TOLERANCE:
                # Use pieces-based calculation as authoritative
                self.progress = calculated_progress

        return self


# ============================================================================
# Peer Information Models
# ============================================================================


class ConnectionType(Enum):
    """Peer connection type."""

    BITTORRENT = "bittorrent"
    HTTP_SEED = "http_seed"
    WEB_SEED = "web_seed"


class PeerFlag(Enum):
    """Peer flags indicating various states."""

    INTERESTING = "interesting"  # We're interested in this peer
    CHOKED = "choked"  # We're choked by this peer
    REMOTE_INTERESTED = "remote_interested"  # Peer is interested in us
    REMOTE_CHOKED = "remote_choked"  # We're choking this peer
    SUPPORTS_EXTENSIONS = "supports_extensions"  # Peer supports extensions
    LOCAL_CONNECTION = "local_connection"  # Connection initiated by us
    HANDSHAKE = "handshake"  # Handshake not yet complete
    CONNECTING = "connecting"  # Currently connecting
    ON_PAROLE = "on_parole"  # Peer is on parole
    SEED = "seed"  # Peer is a seed
    OPTIMISTIC_UNCHOKE = "optimistic_unchoke"  # Optimistically unchoked
    SNUBBED = "snubbed"  # Peer is snubbed
    UPLOAD_ONLY = "upload_only"  # Peer is upload only
    ENDGAME_MODE = "endgame_mode"  # In endgame mode
    HOLEPUNCHED = "holepunched"  # Hole punched connection


class PeerInfo(BaseModel):
    """Detailed peer information."""

    # Connection information
    ip: str
    port: int
    client: str  # Peer client identification
    connection_type: ConnectionType
    flags: list[PeerFlag] = Field(default_factory=list)

    # Transfer statistics
    download_speed: float = Field(ge=0.0)  # bytes per second
    upload_speed: float = Field(ge=0.0)  # bytes per second
    payload_download_speed: float = Field(ge=0.0)  # actual data speed
    payload_upload_speed: float = Field(ge=0.0)  # actual data speed
    total_download: int = Field(ge=0)  # total bytes downloaded from peer
    total_upload: int = Field(ge=0)  # total bytes uploaded to peer

    # Progress and pieces
    progress: float = Field(ge=0.0, le=1.0)  # peer's download progress
    pieces: list[bool] = Field(default_factory=list)  # pieces peer has

    # Connection quality
    rtt: int = Field(ge=0)  # round trip time in milliseconds
    num_hashfails: int = Field(ge=0)  # number of hash failures
    download_queue_length: int = Field(ge=0)  # pending requests to peer
    upload_queue_length: int = Field(ge=0)  # pending requests from peer

    # Geographic information (optional)
    country: str | None = None  # ISO country code

    # Connection timing
    last_request: datetime | None = None
    last_active: datetime | None = None
    connection_time: datetime | None = None

    @field_validator("ip")
    @classmethod
    def validate_ip(cls, v: str) -> str:
        """Validate IP address format."""
        try:
            ipaddress.ip_address(v)
            return v
        except ValueError as e:
            raise ValueError(f"Invalid IP address: {e}") from e

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate port number."""
        if not (MIN_PORT_NUMBER <= v <= MAX_PORT_NUMBER):
            raise ValueError(
                f"Port must be between {MIN_PORT_NUMBER} and {MAX_PORT_NUMBER}"
            )
        return v


# ============================================================================
# Tracker Information Models
# ============================================================================


class TrackerStatus(Enum):
    """Tracker status enumeration."""

    WORKING = "working"
    UPDATING = "updating"
    NOT_CONTACTED = "not_contacted"
    NOT_WORKING = "not_working"


class TrackerInfo(BaseModel):
    """Tracker information and statistics."""

    # Basic tracker information
    url: str
    tier: int = Field(ge=0)  # tracker tier (0 is highest priority)
    status: TrackerStatus = TrackerStatus.NOT_CONTACTED

    # Announce timing
    last_announce: datetime | None = None
    next_announce: datetime | None = None
    announce_interval: int = Field(ge=0)  # seconds
    min_announce_interval: int = Field(ge=0)  # seconds

    # Scrape information
    scrape_complete: int = Field(ge=0)  # seeds reported by tracker
    scrape_incomplete: int = Field(ge=0)  # peers reported by tracker
    scrape_downloaded: int = Field(ge=0)  # completed downloads reported

    # Status information
    message: str = ""  # last message from tracker
    error_code: int = 0  # HTTP error code or 0 if no error

    # Statistics
    fails: int = Field(ge=0)  # number of consecutive failures
    fail_limit: int = Field(ge=0)  # failure limit before considered failed
    updating: bool = False  # currently announcing
    start_sent: bool = False  # start event sent
    complete_sent: bool = False  # complete event sent

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate tracker URL."""
        parsed = urlparse(v)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("Tracker URL must include scheme and host")

        valid_schemes = {"http", "https", "udp"}
        if parsed.scheme.lower() not in valid_schemes:
            raise ValueError(f"Tracker scheme must be one of {valid_schemes}")

        return v


# ============================================================================
# File Management Models
# ============================================================================


class FilePriority(Enum):
    """File download priority levels."""

    DONT_DOWNLOAD = 0
    LOW = 1
    NORMAL = 4
    HIGH = 7


class TorrentFile(BaseModel):
    """Individual file within a torrent."""

    index: int = Field(ge=0)  # file index in torrent
    path: Path  # relative path within torrent
    size: int = Field(ge=0)  # file size in bytes
    priority: FilePriority = FilePriority.NORMAL
    progress: float = Field(ge=0.0, le=1.0)  # download progress
    mtime: datetime | None = None  # modification time

    # File-specific information
    offset: int = Field(ge=0)  # byte offset in torrent
    executable: bool = False  # file is executable
    hidden: bool = False  # file is hidden
    symlink: bool = False  # file is a symlink

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: Path) -> Path:
        """Validate file path."""
        if v.is_absolute():
            raise ValueError("File path must be relative")

        # Check for path traversal attempts
        if ".." in v.parts:
            raise ValueError("File path cannot contain '..' components")

        return v


class FileProgress(BaseModel):
    """File download progress information."""

    file_index: int = Field(ge=0)
    bytes_downloaded: int = Field(ge=0)
    total_bytes: int = Field(ge=0)
    progress: float = Field(ge=0.0, le=1.0)
    pieces_have: list[bool] = Field(default_factory=list)  # pieces for this file

    # Progress timing
    last_update: datetime = Field(default_factory=datetime.now)
    download_speed: float = Field(ge=0.0)  # current download speed

    @model_validator(mode="after")
    def validate_progress_consistency(self) -> "FileProgress":
        """Validate progress consistency."""
        if self.bytes_downloaded > self.total_bytes:
            raise ValueError("Downloaded bytes cannot exceed total bytes")

        if self.total_bytes > 0:
            calculated_progress = self.bytes_downloaded / self.total_bytes
            # Allow small floating point differences
            if abs(self.progress - calculated_progress) > PROGRESS_TOLERANCE:
                self.progress = calculated_progress

        return self


# ============================================================================
# Alert System Models
# ============================================================================


class AlertType(Enum):
    """Torrent alert types."""

    # Torrent lifecycle alerts
    TORRENT_ADDED = "torrent_added"
    TORRENT_REMOVED = "torrent_removed"
    TORRENT_FINISHED = "torrent_finished"
    TORRENT_PAUSED = "torrent_paused"
    TORRENT_RESUMED = "torrent_resumed"
    TORRENT_ERROR = "torrent_error"
    TORRENT_NEED_CERT = "torrent_need_cert"

    # Download progress alerts
    PIECE_FINISHED = "piece_finished"
    BLOCK_FINISHED = "block_finished"
    BLOCK_DOWNLOADING = "block_downloading"
    BLOCK_TIMEOUT = "block_timeout"

    # Tracker alerts
    TRACKER_ANNOUNCE = "tracker_announce"
    TRACKER_ERROR = "tracker_error"
    TRACKER_WARNING = "tracker_warning"
    TRACKER_REPLY = "tracker_reply"

    # DHT alerts
    DHT_ANNOUNCE = "dht_announce"
    DHT_GET_PEERS = "dht_get_peers"
    DHT_BOOTSTRAP = "dht_bootstrap"
    DHT_ERROR = "dht_error"

    # Peer alerts
    PEER_CONNECT = "peer_connect"
    PEER_DISCONNECT = "peer_disconnect"
    PEER_BAN = "peer_ban"
    PEER_UNSNUBBED = "peer_unsnubbed"
    PEER_SNUBBED = "peer_snubbed"
    PEER_ERROR = "peer_error"

    # Metadata alerts
    METADATA_RECEIVED = "metadata_received"
    METADATA_FAILED = "metadata_failed"

    # Storage alerts
    SAVE_RESUME_DATA = "save_resume_data"
    SAVE_RESUME_DATA_FAILED = "save_resume_data_failed"
    STORAGE_MOVED = "storage_moved"
    STORAGE_MOVED_FAILED = "storage_moved_failed"
    FILE_ERROR = "file_error"

    # Session alerts
    SESSION_STATS = "session_stats"
    LISTEN_FAILED = "listen_failed"
    LISTEN_SUCCEEDED = "listen_succeeded"
    PORTMAP = "portmap"
    PORTMAP_ERROR = "portmap_error"

    # Performance alerts
    PERFORMANCE_ALERT = "performance_alert"
    STATS = "stats"
    CACHE_FLUSHED = "cache_flushed"


class AlertSeverity(Enum):
    """Alert severity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertCategory(Enum):
    """Alert categories for filtering and routing."""

    TORRENT = "torrent"
    TRACKER = "tracker"
    PEER = "peer"
    DHT = "dht"
    STORAGE = "storage"
    SESSION = "session"
    PERFORMANCE = "performance"
    ERROR = "error"


class TorrentAlert(BaseModel):
    """Structured torrent alert information."""

    alert_type: AlertType
    torrent_id: str | None = None  # info_hash if torrent-specific
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)
    severity: AlertSeverity = AlertSeverity.INFO
    category: AlertCategory
    data: dict[str, Any] = Field(default_factory=dict)  # additional alert-specific data

    # Alert source information
    source: str = "libtorrent"  # alert source
    alert_id: int | None = None  # libtorrent alert ID

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        """Validate alert message is not empty."""
        if not v.strip():
            raise ValueError("Alert message cannot be empty")
        return v.strip()


# ============================================================================
# Configuration Models
# ============================================================================


class ChokingAlgorithm(Enum):
    """Choking algorithm options."""

    ROUND_ROBIN = "round_robin"
    FASTEST_UPLOAD = "fastest_upload"
    ANTI_LEECH = "anti_leech"


class EncryptionPolicy(Enum):
    """Encryption policy options."""

    FORCED = "forced"  # Only encrypted connections
    ENABLED = "enabled"  # Prefer encrypted, allow unencrypted
    DISABLED = "disabled"  # No encryption


class AllocationMode(Enum):
    """File allocation mode options."""

    SPARSE = "sparse"  # Sparse files (default)
    ALLOCATE = "allocate"  # Pre-allocate full size
    COMPACT = "compact"  # Compact allocation


class TorrentConfig(BaseModel):
    """Comprehensive torrent engine configuration."""

    # Session settings
    listen_interfaces: str = "0.0.0.0:6881"
    dht_bootstrap_nodes: list[str] = Field(
        default_factory=lambda: [
            "router.bittorrent.com:6881",
            "dht.transmissionbt.com:6881",
            "router.utorrent.com:6881",
        ]
    )
    enable_dht: bool = True
    enable_lsd: bool = True  # Local Service Discovery
    enable_upnp: bool = True
    enable_natpmp: bool = True

    # Download settings
    download_rate_limit: int = Field(ge=0, default=0)  # 0 = unlimited
    upload_rate_limit: int = Field(ge=0, default=0)  # 0 = unlimited
    max_connections: int = Field(ge=1, default=200)
    max_connections_per_torrent: int = Field(ge=1, default=50)
    max_uploads: int = Field(ge=1, default=4)
    max_uploads_per_torrent: int = Field(ge=1, default=2)

    # Seeding settings
    seed_time_limit: int = Field(ge=0, default=0)  # 0 = unlimited (seconds)
    seed_ratio_limit: float = Field(ge=0.0, default=0.0)  # 0 = unlimited
    share_ratio_limit: float = Field(ge=0.0, default=2.0)
    seed_choking_algorithm: ChokingAlgorithm = ChokingAlgorithm.FASTEST_UPLOAD

    # File settings
    download_directory: Path
    torrent_files_directory: Path | None = None
    resume_data_directory: Path | None = None
    allocation_mode: AllocationMode = AllocationMode.SPARSE

    # Network settings
    user_agent: str = "Zuup/1.0"
    peer_id_prefix: str = "-ZU1000-"
    enable_encryption: bool = True
    encryption_policy: EncryptionPolicy = EncryptionPolicy.ENABLED

    # Advanced settings
    piece_timeout: int = Field(ge=1, default=20)  # seconds
    request_timeout: int = Field(ge=1, default=60)  # seconds
    peer_timeout: int = Field(ge=1, default=120)  # seconds
    inactivity_timeout: int = Field(ge=1, default=600)  # seconds
    handshake_timeout: int = Field(ge=1, default=10)  # seconds

    # Queue settings
    max_active_downloads: int = Field(ge=1, default=3)
    max_active_seeds: int = Field(ge=1, default=5)
    dont_count_slow_torrents: bool = True
    auto_manage_startup: int = Field(ge=0, default=60)  # seconds

    # Cache settings
    cache_size: int = Field(ge=0, default=1024)  # 16KB blocks
    cache_expiry: int = Field(ge=1, default=60)  # seconds

    # Alert settings
    alert_queue_size: int = Field(ge=100, default=1000)

    @field_validator("listen_interfaces")
    @classmethod
    def validate_listen_interfaces(cls, v: str) -> str:
        """Validate listen interfaces format."""
        # Basic validation - should be in format "ip:port" or "ip:port,ip:port"
        interfaces = v.split(",")
        for interface in interfaces:
            if ":" not in interface.strip():
                raise ValueError("Listen interface must be in format 'ip:port'")
        return v

    @field_validator("peer_id_prefix")
    @classmethod
    def validate_peer_id_prefix(cls, v: str) -> str:
        """Validate peer ID prefix format."""
        if len(v) != PEER_ID_PREFIX_LENGTH:
            raise ValueError(
                f"Peer ID prefix must be exactly {PEER_ID_PREFIX_LENGTH} characters"
            )
        if not v.startswith("-") or not v.endswith("-"):
            raise ValueError("Peer ID prefix must start and end with '-'")
        return v

    @model_validator(mode="after")
    def validate_config_consistency(self) -> "TorrentConfig":
        """Validate configuration consistency."""
        # Ensure max connections per torrent doesn't exceed global max
        self.max_connections_per_torrent = min(
            self.max_connections_per_torrent, self.max_connections
        )

        # Ensure max uploads per torrent doesn't exceed global max
        self.max_uploads_per_torrent = min(
            self.max_uploads_per_torrent, self.max_uploads
        )

        # Set default directories if not provided
        if self.torrent_files_directory is None:
            self.torrent_files_directory = self.download_directory / ".torrents"

        if self.resume_data_directory is None:
            self.resume_data_directory = self.download_directory / ".resume"

        return self


# ============================================================================
# Statistics and Monitoring Models
# ============================================================================


class SessionStats(BaseModel):
    """Session-wide statistics."""

    # Transfer statistics
    total_download: int = Field(ge=0)
    total_upload: int = Field(ge=0)
    total_payload_download: int = Field(ge=0)
    total_payload_upload: int = Field(ge=0)

    # Speed statistics
    download_rate: float = Field(ge=0.0)
    upload_rate: float = Field(ge=0.0)
    payload_download_rate: float = Field(ge=0.0)
    payload_upload_rate: float = Field(ge=0.0)

    # Connection statistics
    num_peers: int = Field(ge=0)
    num_unchoked: int = Field(ge=0)
    allowed_upload_slots: int = Field(ge=0)

    # DHT statistics
    dht_nodes: int = Field(ge=0)
    dht_node_cache: int = Field(ge=0)
    dht_torrents: int = Field(ge=0)
    dht_global_nodes: int = Field(ge=0)

    # Disk statistics
    disk_read_queue: int = Field(ge=0)
    disk_write_queue: int = Field(ge=0)

    # Alert statistics
    alerts_dropped: int = Field(ge=0)
    alert_queue_len: int = Field(ge=0)


class SwarmHealth(BaseModel):
    """Swarm health analysis."""

    seed_peer_ratio: float = Field(ge=0.0)  # seeds / (seeds + peers)
    availability: float = Field(ge=0.0)  # average piece availability
    distributed_copies: float = Field(ge=0.0)  # number of complete copies

    # Health indicators
    is_healthy: bool = True  # overall health assessment
    health_score: float = Field(
        ge=0.0, le=1.0, default=0.0
    )  # 0.0 = unhealthy, 1.0 = very healthy

    # Recommendations
    needs_more_seeds: bool = False
    needs_more_peers: bool = False
    is_stalled: bool = False

    @model_validator(mode="after")
    def calculate_health_metrics(self) -> "SwarmHealth":
        """Calculate derived health metrics."""
        # Calculate health score based on various factors
        score = 0.0

        # Seed/peer ratio contributes 40% to health
        if self.seed_peer_ratio > GOOD_SEED_RATIO_THRESHOLD:
            score += 0.4
        elif self.seed_peer_ratio > MIN_SEED_RATIO_THRESHOLD:
            score += 0.2

        # Availability contributes 30% to health
        if self.availability > GOOD_AVAILABILITY_THRESHOLD:
            score += 0.3
        elif self.availability > MIN_AVAILABILITY_THRESHOLD:
            score += 0.15

        # Distributed copies contributes 30% to health
        if self.distributed_copies > GOOD_COPIES_THRESHOLD:
            score += 0.3
        elif self.distributed_copies > MIN_COPIES_THRESHOLD:
            score += 0.15

        self.health_score = min(score, 1.0)
        self.is_healthy = self.health_score > HEALTH_THRESHOLD

        # Set recommendations
        self.needs_more_seeds = self.seed_peer_ratio < SEED_SHORTAGE_THRESHOLD
        self.needs_more_peers = self.availability < MIN_AVAILABILITY_THRESHOLD
        self.is_stalled = (
            self.availability < STALL_AVAILABILITY_THRESHOLD
            and self.distributed_copies < MIN_COPIES_THRESHOLD
        )

        return self
