pub mod api;
pub mod bandwidth;
pub mod callbacks;
pub mod checksum;
pub mod config;
pub mod download;
pub mod engine;
pub mod error;
pub mod event;
pub mod file;
pub mod logging;
pub mod media;
pub mod metalink;
pub mod metrics;
pub mod network;
pub mod protocol;
pub mod protocols;
pub mod session;
pub mod types;

// Re-export the high-level API
pub use api::{DownloadResult, Zuup, ZuupBuilder};
pub use bandwidth::{BandwidthLimit, BandwidthManager, BandwidthStats, DownloadBandwidthTracker};
pub use callbacks::{
    BandwidthCallback, BandwidthStats as CallbackBandwidthStats, CallbackBuilder, CallbackManager,
    CompletionCallback, DownloadResult as CallbackDownloadResult, ErrorCallback,
    ErrorRecoveryAction, ProgressCallback, ProgressMonitorConfig, StateChangeCallback,
};
pub use checksum::{
    ChecksumCalculator, ChecksumConfig, ChecksumResult, ChecksumType, ChecksumVerifier, ChunkInfo,
    ChunkIntegrityChecker, ChunkUtils, ChunkVerificationResult, ChunkVerificationStats,
    StreamingHasher,
};
pub use config::ZuupConfig;
pub use download::{
    DownloadManager, DownloadSource, DownloadTask, FailoverConfig, MultiSourceCoordinator,
    SourceHealthMonitor, SourceState, SourceStatistics, SourceStatus, TaskCommand, TaskScheduler,
};
pub use engine::ZuupEngine;
pub use error::{Result, ZuupError};
pub use event::{Event, EventBus, EventSubscriber, EventType};
pub use file::{
    ConflictResolution, DiskSpaceInfo, FileManager, FileNameTemplate, FileOrganizationConfig,
    FilePermissions, FileSystemManager, TemplateVariables,
};
pub use logging::{LogFormat, LoggingConfig};
pub use media::{MediaDownloadOptions, MediaExtractor, MediaFormat, MediaInfo, YtDlpManager};
pub use metalink::{
    Metalink, MetalinkChecksum, MetalinkFile, MetalinkParser, MetalinkPiece, MetalinkSignature,
    MetalinkUrl, MetalinkVersion,
};
pub use metrics::{
    DownloadStats, HealthCheck, HealthStatus, MetricsCollector, MetricsConfig, SystemMetrics,
};
pub use protocol::{Download, ProtocolHandler};

// Conditional protocol handler re-exports
#[cfg(feature = "http")]
pub use protocols::HttpProtocolHandler;

#[cfg(feature = "ftp")]
pub use protocols::FtpProtocolHandler;

#[cfg(feature = "sftp")]
pub use protocols::SftpProtocolHandler;

#[cfg(feature = "torrent")]
pub use protocols::BitTorrentProtocolHandler;
pub use session::{
    MaintenanceReport, PartialFileInfo, SessionManager, SessionRecoveryInfo, SessionStats,
};
pub use types::{
    DownloadId, DownloadOptions, DownloadPriority as DownloadPriorityProps,
    DownloadProgress as DownloadProgressProps, DownloadRequest as DownloadRequestProps,
    DownloadState as DownloadStateProps,
};
