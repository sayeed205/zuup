//! BitTorrent protocol handler using rqbit with DHT, PEX, and seeding support

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::HashMap;

use async_trait::async_trait;
use url::Url;
use tokio::sync::{RwLock, Mutex};
use chrono::{DateTime, Utc};

use crate::{
    protocol::{ProtocolHandler, Download, ProtocolCapabilities, DownloadMetadata, DownloadOperation},
    download::{DownloadRequest, DownloadState},
    types::DownloadProgress,
    error::{RusoError, Result},
};

/// BitTorrent protocol handler with DHT and PEX support
pub struct BitTorrentProtocolHandler {
    session: Arc<RwLock<Option<Arc<librqbit::Session>>>>,
    config: BitTorrentConfig,
    peer_stats: Arc<Mutex<HashMap<String, PeerStats>>>,
    seeding_manager: Arc<Mutex<SeedingManager>>,
}

/// Configuration for BitTorrent protocol
#[derive(Debug, Clone)]
pub struct BitTorrentConfig {
    /// Enable DHT for peer discovery
    pub enable_dht: bool,
    /// Enable PEX (Peer Exchange) protocol
    pub enable_pex: bool,
    /// Enable seeding after download completion
    pub enable_seeding: bool,
    /// Target seeding ratio (upload/download ratio)
    pub seeding_ratio: f64,
    /// Maximum seeding time in seconds (0 = unlimited)
    pub max_seeding_time: u64,
    /// Maximum number of peers per torrent
    pub max_peers: u32,
    /// DHT bootstrap nodes
    pub dht_bootstrap_nodes: Vec<String>,
    /// Enable encryption for peer connections
    pub enable_encryption: bool,
    /// Port range for incoming connections
    pub port_range: (u16, u16),
    /// Enable selective file downloading
    pub enable_selective_download: bool,
    /// Default file selection strategy
    pub file_selection_strategy: FileSelectionStrategy,
    /// Tracker communication settings
    pub tracker_config: TrackerConfig,
    /// Encryption settings for peer connections
    pub encryption_config: EncryptionConfig,
}

impl Default for BitTorrentConfig {
    fn default() -> Self {
        Self {
            enable_dht: true,
            enable_pex: true,
            enable_seeding: true,
            seeding_ratio: 1.0, // Seed until 1:1 ratio
            max_seeding_time: 0, // Unlimited
            max_peers: 50,
            dht_bootstrap_nodes: vec![
                "router.bittorrent.com:6881".to_string(),
                "dht.transmissionbt.com:6881".to_string(),
                "router.utorrent.com:6881".to_string(),
            ],
            enable_encryption: true,
            port_range: (6881, 6889),
            enable_selective_download: true,
            file_selection_strategy: FileSelectionStrategy::All,
            tracker_config: TrackerConfig::default(),
            encryption_config: EncryptionConfig::default(),
        }
    }
}

/// Statistics for peer connections
#[derive(Debug, Clone)]
pub struct PeerStats {
    pub connected_peers: u32,
    pub downloading_peers: u32,
    pub seeding_peers: u32,
    pub total_downloaded: u64,
    pub total_uploaded: u64,
    pub last_updated: DateTime<Utc>,
}

/// Manager for seeding operations and ratio control
#[derive(Debug)]
pub struct SeedingManager {
    /// Active seeding torrents with their start times and ratios
    pub active_seeds: HashMap<String, SeedingInfo>,
    /// Completed torrents that have reached their seeding goals
    pub completed_seeds: HashMap<String, SeedingInfo>,
}

/// Information about a seeding torrent
#[derive(Debug, Clone)]
pub struct SeedingInfo {
    pub info_hash: String,
    pub started_at: DateTime<Utc>,
    pub downloaded_bytes: u64,
    pub uploaded_bytes: u64,
    pub target_ratio: f64,
    pub max_seeding_time: Option<Duration>,
    pub is_complete: bool,
}

impl SeedingManager {
    pub fn new() -> Self {
        Self {
            active_seeds: HashMap::new(),
            completed_seeds: HashMap::new(),
        }
    }

    /// Add a torrent to seeding management
    pub fn add_torrent(&mut self, info_hash: String, downloaded_bytes: u64, target_ratio: f64, max_time: Option<Duration>) {
        let seeding_info = SeedingInfo {
            info_hash: info_hash.clone(),
            started_at: Utc::now(),
            downloaded_bytes,
            uploaded_bytes: 0,
            target_ratio,
            max_seeding_time: max_time,
            is_complete: false,
        };

        self.active_seeds.insert(info_hash, seeding_info);
    }

    /// Update upload statistics for a torrent
    pub fn update_upload_stats(&mut self, info_hash: &str, uploaded_bytes: u64) {
        if let Some(info) = self.active_seeds.get_mut(info_hash) {
            info.uploaded_bytes = uploaded_bytes;

            // Check if seeding goals are met
            let current_ratio = if info.downloaded_bytes > 0 {
                info.uploaded_bytes as f64 / info.downloaded_bytes as f64
            } else {
                0.0
            };

            let time_limit_reached = if let Some(max_time) = info.max_seeding_time {
                Utc::now().signed_duration_since(info.started_at).to_std().unwrap_or(Duration::ZERO) >= max_time
            } else {
                false
            };

            if current_ratio >= info.target_ratio || time_limit_reached {
                info.is_complete = true;
                let completed_info = info.clone();
                self.completed_seeds.insert(info_hash.to_string(), completed_info);
                tracing::info!(
                    info_hash = %info_hash,
                    ratio = %current_ratio,
                    target_ratio = %info.target_ratio,
                    time_limit_reached = %time_limit_reached,
                    "Seeding goal reached for torrent"
                );
            }
        }
    }

    /// Check if a torrent should stop seeding
    pub fn should_stop_seeding(&self, info_hash: &str) -> bool {
        self.active_seeds.get(info_hash)
            .map(|info| info.is_complete)
            .unwrap_or(false)
    }

    /// Get seeding statistics
    pub fn get_stats(&self) -> SeedingStats {
        let active_count = self.active_seeds.len();
        let completed_count = self.completed_seeds.len();

        // Only count from active_seeds to avoid double-counting
        // (completed torrents remain in active_seeds)
        let total_uploaded: u64 = self.active_seeds.values()
            .map(|info| info.uploaded_bytes)
            .sum();
        let total_downloaded: u64 = self.active_seeds.values()
            .map(|info| info.downloaded_bytes)
            .sum();

        let overall_ratio = if total_downloaded > 0 {
            total_uploaded as f64 / total_downloaded as f64
        } else {
            0.0
        };

        SeedingStats {
            active_torrents: active_count,
            completed_torrents: completed_count,
            total_uploaded,
            total_downloaded,
            overall_ratio,
        }
    }
}

/// Statistics for seeding operations
#[derive(Debug, Clone)]
pub struct SeedingStats {
    pub active_torrents: usize,
    pub completed_torrents: usize,
    pub total_uploaded: u64,
    pub total_downloaded: u64,
    pub overall_ratio: f64,
}

#[allow(dead_code)]
impl BitTorrentProtocolHandler {
    /// Create a new BitTorrent protocol handler with default configuration
    pub fn new() -> Self {
        Self::with_config(BitTorrentConfig::default())
    }

    /// Create a new BitTorrent protocol handler with custom configuration
    pub fn with_config(config: BitTorrentConfig) -> Self {
        Self {
            session: Arc::new(RwLock::new(None)),
            config,
            peer_stats: Arc::new(Mutex::new(HashMap::new())),
            seeding_manager: Arc::new(Mutex::new(SeedingManager::new())),
        }
    }

    /// Initialize the BitTorrent session with DHT and PEX support
    async fn ensure_session(&self) -> Result<()> {
        let mut session_guard = self.session.write().await;
        if session_guard.is_none() {
            // Create session options with available configuration
            let mut session_opts = librqbit::SessionOptions::default();

            // Configure DHT if disabled (librqbit enables DHT by default)
            if !self.config.enable_dht {
                session_opts.disable_dht = true;
                tracing::info!("DHT disabled");
            } else {
                // DHT is enabled by default, we can configure it if needed
                // Note: librqbit uses PersistentDhtConfig, not DhtConfig directly
                tracing::info!("DHT enabled (default behavior)");
            }

            // Note: librqbit doesn't expose direct PEX configuration in SessionOptions
            // PEX is typically enabled by default in modern BitTorrent clients
            if self.config.enable_pex {
                tracing::info!("PEX (Peer Exchange) enabled (default behavior)");
            }

            // Note: librqbit doesn't expose direct encryption configuration in SessionOptions
            // Encryption support is typically built-in
            if self.config.enable_encryption {
                tracing::info!("Peer connection encryption enabled (default behavior)");
            }

            // Note: librqbit doesn't expose max_peers_per_torrent in SessionOptions
            // This would be configured per-torrent when adding torrents

            // Configure port range (convert tuple to Range)
            session_opts.listen_port_range = Some(self.config.port_range.0..self.config.port_range.1);

            // Create the session with configured options
            let session = librqbit::Session::new_with_opts(
                "/tmp/ruso-torrents".into(), // Default download directory
                session_opts
            ).await
                .map_err(|e| RusoError::Protocol(
                    ruso_core::error::ProtocolError::InitializationFailed(
                        format!("Failed to create BitTorrent session: {}", e)
                    )
                ))?;

            *session_guard = Some(session);

            tracing::info!(
                dht_enabled = %self.config.enable_dht,
                pex_enabled = %self.config.enable_pex,
                encryption_enabled = %self.config.enable_encryption,
                max_peers = %self.config.max_peers,
                port_range = ?self.config.port_range,
                "BitTorrent session initialized with available options"
            );
        }
        Ok(())
    }

    /// Get current peer statistics
    pub async fn get_peer_stats(&self) -> HashMap<String, PeerStats> {
        let stats = self.peer_stats.lock().await;
        stats.clone()
    }

    /// Get seeding statistics
    pub async fn get_seeding_stats(&self) -> SeedingStats {
        let manager = self.seeding_manager.lock().await;
        manager.get_stats()
    }

    /// Update peer statistics for a torrent
    async fn update_peer_stats(&self, info_hash: &str, stats: PeerStats) {
        let mut peer_stats = self.peer_stats.lock().await;
        peer_stats.insert(info_hash.to_string(), stats);
    }

    /// Start seeding management for a completed torrent
    async fn start_seeding(&self, info_hash: String, downloaded_bytes: u64) -> Result<()> {
        if !self.config.enable_seeding {
            return Ok(());
        }

        let max_time = if self.config.max_seeding_time > 0 {
            Some(Duration::from_secs(self.config.max_seeding_time))
        } else {
            None
        };

        let mut manager = self.seeding_manager.lock().await;
        manager.add_torrent(info_hash.clone(), downloaded_bytes, self.config.seeding_ratio, max_time);

        tracing::info!(
            info_hash = %info_hash,
            target_ratio = %self.config.seeding_ratio,
            max_time = ?max_time,
            "Started seeding management for torrent"
        );

        Ok(())
    }

    /// Check and stop seeding for torrents that have reached their goals
    async fn check_seeding_goals(&self) -> Result<Vec<String>> {
        let mut stopped_torrents = Vec::new();

        if !self.config.enable_seeding {
            return Ok(stopped_torrents);
        }

        let _session_guard = self.session.read().await;
        if let Some(_session) = _session_guard.as_ref() {
            let manager = self.seeding_manager.lock().await;

            for (info_hash, _) in &manager.active_seeds {
                if manager.should_stop_seeding(info_hash) {
                    // Stop seeding this torrent
                    if let Ok(torrent_id) = hex::decode(info_hash) {
                        if torrent_id.len() == 20 {
                            // Note: This is a placeholder for the actual librqbit API call
                            // The exact method to stop seeding would depend on the librqbit version
                            // In a real implementation, we would use session.delete() or similar
                            tracing::info!(info_hash = %info_hash, "Would stop seeding torrent (API call needed)");
                            stopped_torrents.push(info_hash.clone());
                        }
                    }
                }
            }
        }

        Ok(stopped_torrents)
    }

    /// Apply file selection strategy to determine which files to download
    pub fn apply_file_selection(&self, files: &mut [TorrentFileInfo], strategy: &FileSelectionStrategy) -> Result<()> {
        match strategy {
            FileSelectionStrategy::All => {
                for file in files.iter_mut() {
                    file.selected = true;
                    file.priority = FilePriority::Normal;
                }
            }
            FileSelectionStrategy::None => {
                for file in files.iter_mut() {
                    file.selected = false;
                    file.priority = FilePriority::Skip;
                }
            }
            FileSelectionStrategy::Pattern(patterns) => {
                for file in files.iter_mut() {
                    file.selected = patterns.iter().any(|pattern| {
                        // Simple glob-like matching (in production, use a proper glob library)
                        let file_path = file.path.to_string_lossy();
                        if pattern.contains('*') {
                            let pattern_parts: Vec<&str> = pattern.split('*').collect();
                            if pattern_parts.len() == 2 {
                                file_path.starts_with(pattern_parts[0]) && file_path.ends_with(pattern_parts[1])
                            } else {
                                file_path.contains(pattern)
                            }
                        } else {
                            file_path.contains(pattern)
                        }
                    });
                    file.priority = if file.selected { FilePriority::Normal } else { FilePriority::Skip };
                }
            }
            FileSelectionStrategy::Indices(indices) => {
                for (i, file) in files.iter_mut().enumerate() {
                    file.selected = indices.contains(&i);
                    file.priority = if file.selected { FilePriority::Normal } else { FilePriority::Skip };
                }
            }
            FileSelectionStrategy::LargestFirst(count) => {
                // Sort files by size (largest first) and select top N
                let mut indexed_files: Vec<(usize, u64)> = files.iter().enumerate()
                    .map(|(i, f)| (i, f.size))
                    .collect();
                indexed_files.sort_by(|a, b| b.1.cmp(&a.1));

                for file in files.iter_mut() {
                    file.selected = false;
                    file.priority = FilePriority::Skip;
                }

                for (i, _) in indexed_files.iter().take(*count) {
                    files[*i].selected = true;
                    files[*i].priority = FilePriority::High;
                }
            }
            FileSelectionStrategy::SmallestFirst(count) => {
                // Sort files by size (smallest first) and select top N
                let mut indexed_files: Vec<(usize, u64)> = files.iter().enumerate()
                    .map(|(i, f)| (i, f.size))
                    .collect();
                indexed_files.sort_by(|a, b| a.1.cmp(&b.1));

                for file in files.iter_mut() {
                    file.selected = false;
                    file.priority = FilePriority::Skip;
                }

                for (i, _) in indexed_files.iter().take(*count) {
                    files[*i].selected = true;
                    files[*i].priority = FilePriority::High;
                }
            }
        }

        tracing::info!(
            strategy = ?strategy,
            selected_count = files.iter().filter(|f| f.selected).count(),
            total_count = files.len(),
            "Applied file selection strategy"
        );

        Ok(())
    }

    /// Apply file selection criteria for advanced filtering
    pub fn apply_file_selection_criteria(&self, files: &mut [TorrentFileInfo], criteria: &FileSelectionCriteria) -> Result<()> {
        for file in files.iter_mut() {
            let file_path = file.path.to_string_lossy();
            let file_name = file.path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");

            // Check size constraints
            if let Some(min_size) = criteria.min_size {
                if file.size < min_size {
                    file.selected = false;
                    file.priority = FilePriority::Skip;
                    continue;
                }
            }

            if let Some(max_size) = criteria.max_size {
                if file.size > max_size {
                    file.selected = false;
                    file.priority = FilePriority::Skip;
                    continue;
                }
            }

            // Check extension constraints
            if !criteria.include_extensions.is_empty() {
                let has_included_ext = criteria.include_extensions.iter().any(|ext| {
                    file_name.ends_with(&format!(".{}", ext))
                });
                if !has_included_ext {
                    file.selected = false;
                    file.priority = FilePriority::Skip;
                    continue;
                }
            }

            if !criteria.exclude_extensions.is_empty() {
                let has_excluded_ext = criteria.exclude_extensions.iter().any(|ext| {
                    file_name.ends_with(&format!(".{}", ext))
                });
                if has_excluded_ext {
                    file.selected = false;
                    file.priority = FilePriority::Skip;
                    continue;
                }
            }

            // Check include patterns
            if !criteria.include_patterns.is_empty() {
                let matches_include = criteria.include_patterns.iter().any(|pattern| {
                    // Simple pattern matching
                    if pattern.contains('*') {
                        let pattern_parts: Vec<&str> = pattern.split('*').collect();
                        if pattern_parts.len() == 2 {
                            file_path.starts_with(pattern_parts[0]) && file_path.ends_with(pattern_parts[1])
                        } else {
                            file_path.contains(pattern)
                        }
                    } else {
                        file_path.contains(pattern)
                    }
                });
                if !matches_include {
                    file.selected = false;
                    file.priority = FilePriority::Skip;
                    continue;
                }
            }

            // Check exclude patterns
            if !criteria.exclude_patterns.is_empty() {
                let matches_exclude = criteria.exclude_patterns.iter().any(|pattern| {
                    if pattern.contains('*') {
                        let pattern_parts: Vec<&str> = pattern.split('*').collect();
                        if pattern_parts.len() == 2 {
                            file_path.starts_with(pattern_parts[0]) && file_path.ends_with(pattern_parts[1])
                        } else {
                            file_path.contains(pattern)
                        }
                    } else {
                        file_path.contains(pattern)
                    }
                });
                if matches_exclude {
                    file.selected = false;
                    file.priority = FilePriority::Skip;
                    continue;
                }
            }

            // If we reach here, the file passes all criteria
            file.selected = true;
            file.priority = FilePriority::Normal;
        }

        let selected_count = files.iter().filter(|f| f.selected).count();
        let total_size: u64 = files.iter().filter(|f| f.selected).map(|f| f.size).sum();

        tracing::info!(
            selected_count = selected_count,
            total_count = files.len(),
            selected_size = total_size,
            "Applied file selection criteria"
        );

        Ok(())
    }
}

#[async_trait]
impl ProtocolHandler for BitTorrentProtocolHandler {
    fn protocol(&self) -> &'static str {
        "bittorrent"
    }

    fn can_handle(&self, url: &Url) -> bool {
        matches!(url.scheme(), "magnet") ||
        url.path().ends_with(".torrent") ||
        url.to_string().starts_with("magnet:")
    }

    async fn create_download(&self, request: &DownloadRequest) -> Result<Box<dyn Download>> {
        self.ensure_session().await?;

        // Use the first URL for now
        let url = request.urls.first()
            .ok_or_else(|| RusoError::Config("No URLs provided".to_string()))?;

        let download = BitTorrentDownload::new(
            url.clone(),
            self.session.clone(),
            request.options.clone(),
            self.config.clone(),
            self.peer_stats.clone(),
            self.seeding_manager.clone(),
            request.output_path.clone(),
        ).await?;

        Ok(Box::new(download))
    }

    async fn resume_download(&self, request: &DownloadRequest, _state: &DownloadState) -> Result<Box<dyn Download>> {
        // For now, just create a new download (resume logic would be enhanced later)
        // In a full implementation, we would restore the torrent state and continue from where it left off
        self.create_download(request).await
    }

    fn capabilities(&self) -> ProtocolCapabilities {
        ProtocolCapabilities {
            supports_segments: true,  // BitTorrent naturally supports parallel downloading
            supports_resume: true,
            supports_ranges: false,   // Not applicable to BitTorrent
            supports_auth: false,     // BitTorrent doesn't use traditional auth
            supports_proxy: true,     // Can support proxy for tracker communication
            max_connections: None,    // BitTorrent manages its own connections
            supports_checksums: true, // BitTorrent has built-in piece verification
            supports_metadata: true,
        }
    }
}

/// BitTorrent download implementation with DHT, PEX, and seeding support
#[allow(dead_code)]
pub struct BitTorrentDownload {
    url: Url,
    session: Arc<RwLock<Option<Arc<librqbit::Session>>>>,
    torrent_handle: Option<Arc<librqbit::ManagedTorrent>>,
    // Shared state/progress so background task can update while DownloadTask polls
    shared_state: Arc<Mutex<DownloadState>>,
    shared_progress: Arc<Mutex<DownloadProgress>>,
    state: DownloadState,
    progress: DownloadProgress,
    torrent_info: Option<TorrentInfo>,
    options: ruso_core::types::DownloadOptions,
    config: BitTorrentConfig,
    peer_stats: Arc<Mutex<HashMap<String, PeerStats>>>,
    seeding_manager: Arc<Mutex<SeedingManager>>,
    last_stats_update: Arc<Mutex<Instant>>,
    dht_stats: Arc<Mutex<DhtStats>>,
    pex_stats: Arc<Mutex<PexStats>>,
    tracker_stats: Arc<Mutex<Vec<TrackerStats>>>,
    file_selection_criteria: Option<FileSelectionCriteria>,
    output_path: Option<PathBuf>,
}

/// DHT (Distributed Hash Table) statistics
#[derive(Debug, Clone, Default)]
pub struct DhtStats {
    pub nodes_count: u32,
    pub good_nodes: u32,
    pub queries_sent: u64,
    pub queries_received: u64,
    pub peers_found: u64,
    pub last_updated: Option<DateTime<Utc>>,
}

/// PEX (Peer Exchange) statistics
#[derive(Debug, Clone, Default)]
pub struct PexStats {
    pub peers_received: u64,
    pub peers_sent: u64,
    pub successful_exchanges: u64,
    pub failed_exchanges: u64,
    pub last_updated: Option<DateTime<Utc>>,
}

/// Information about a torrent
#[derive(Debug, Clone)]
pub struct TorrentInfo {
    pub name: String,
    pub total_size: u64,
    pub piece_count: u32,
    pub piece_size: u32,
    pub files: Vec<TorrentFileInfo>,
    pub info_hash: String,
}

/// Information about a file in a torrent
#[derive(Debug, Clone)]
pub struct TorrentFileInfo {
    pub path: PathBuf,
    pub size: u64,
    pub selected: bool,
    pub priority: FilePriority,
}

/// File selection strategy for multi-file torrents
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileSelectionStrategy {
    /// Download all files
    All,
    /// Download no files (manual selection required)
    None,
    /// Download files matching patterns
    Pattern(Vec<String>),
    /// Download files by index
    Indices(Vec<usize>),
    /// Download largest files first
    LargestFirst(usize), // Number of files to select
    /// Download smallest files first
    SmallestFirst(usize), // Number of files to select
}

/// File download priority
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilePriority {
    /// Don't download this file
    Skip = 0,
    /// Low priority
    Low = 1,
    /// Normal priority
    Normal = 4,
    /// High priority
    High = 7,
}

impl Default for FilePriority {
    fn default() -> Self {
        FilePriority::Normal
    }
}

/// Tracker communication configuration
#[derive(Debug, Clone)]
pub struct TrackerConfig {
    /// Enable tracker communication
    pub enabled: bool,
    /// Tracker announce interval in seconds
    pub announce_interval: u64,
    /// Minimum announce interval in seconds
    pub min_announce_interval: u64,
    /// Maximum number of tracker failures before giving up
    pub max_failures: u32,
    /// Timeout for tracker requests in seconds
    pub request_timeout: u64,
    /// Enable tracker scraping
    pub enable_scraping: bool,
    /// Custom user agent for tracker requests
    pub user_agent: Option<String>,
    /// Enable IPv6 tracker communication
    pub enable_ipv6: bool,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            announce_interval: 1800, // 30 minutes
            min_announce_interval: 300, // 5 minutes
            max_failures: 5,
            request_timeout: 30,
            enable_scraping: true,
            user_agent: Some("Ruso/0.1.0".to_string()),
            enable_ipv6: true,
        }
    }
}

/// Encryption configuration for peer connections
#[derive(Debug, Clone)]
pub struct EncryptionConfig {
    /// Enable message stream encryption (MSE)
    pub enable_mse: bool,
    /// Require encryption for all connections
    pub require_encryption: bool,
    /// Prefer encrypted connections
    pub prefer_encryption: bool,
    /// Allowed encryption methods
    pub allowed_methods: Vec<EncryptionMethod>,
}

impl Default for EncryptionConfig {
    fn default() -> Self {
        Self {
            enable_mse: true,
            require_encryption: false,
            prefer_encryption: true,
            allowed_methods: vec![
                EncryptionMethod::PlainText,
                EncryptionMethod::RC4,
            ],
        }
    }
}

/// Supported encryption methods
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncryptionMethod {
    /// No encryption (plain text)
    PlainText,
    /// RC4 encryption
    RC4,
}

/// Tracker statistics and information
#[derive(Debug, Clone)]
pub struct TrackerStats {
    /// Tracker URL
    pub url: String,
    /// Last announce time
    pub last_announce: Option<DateTime<Utc>>,
    /// Next announce time
    pub next_announce: Option<DateTime<Utc>>,
    /// Number of successful announces
    pub successful_announces: u64,
    /// Number of failed announces
    pub failed_announces: u64,
    /// Last error message
    pub last_error: Option<String>,
    /// Number of seeders reported by tracker
    pub seeders: Option<u32>,
    /// Number of leechers reported by tracker
    pub leechers: Option<u32>,
    /// Number of completed downloads reported by tracker
    pub completed: Option<u32>,
    /// Tracker tier (0 is highest priority)
    pub tier: u32,
}

/// File selection criteria for selective downloading
#[derive(Debug, Clone)]
pub struct FileSelectionCriteria {
    /// File patterns to include (glob patterns)
    pub include_patterns: Vec<String>,
    /// File patterns to exclude (glob patterns)
    pub exclude_patterns: Vec<String>,
    /// Minimum file size in bytes
    pub min_size: Option<u64>,
    /// Maximum file size in bytes
    pub max_size: Option<u64>,
    /// File extensions to include
    pub include_extensions: Vec<String>,
    /// File extensions to exclude
    pub exclude_extensions: Vec<String>,
}

/// Encryption status for peer connections
#[derive(Debug, Clone)]
pub struct EncryptionStatus {
    /// Whether MSE (Message Stream Encryption) is enabled
    pub mse_enabled: bool,
    /// Number of encrypted peer connections
    pub encrypted_peers: u32,
    /// Number of unencrypted peer connections
    pub unencrypted_peers: u32,
    /// Preferred encryption method
    pub preferred_method: EncryptionMethod,
    /// Whether encryption is required
    pub encryption_required: bool,
}

impl Default for EncryptionStatus {
    fn default() -> Self {
        Self {
            mse_enabled: true,
            encrypted_peers: 0,
            unencrypted_peers: 0,
            preferred_method: EncryptionMethod::RC4,
            encryption_required: false,
        }
    }
}

impl BitTorrentDownload {
    /// Create a new BitTorrent download with DHT and PEX support
    pub async fn new(
        url: Url,
        session: Arc<RwLock<Option<Arc<librqbit::Session>>>>,
        options: ruso_core::types::DownloadOptions,
        config: BitTorrentConfig,
        peer_stats: Arc<Mutex<HashMap<String, PeerStats>>>,
        seeding_manager: Arc<Mutex<SeedingManager>>,
        output_path: Option<PathBuf>,
    ) -> Result<Self> {
        let mut download = Self {
                    shared_state: Arc::new(Mutex::new(DownloadState::Pending)),
                    shared_progress: Arc::new(Mutex::new(DownloadProgress::new())),
            url,
            session,
            torrent_handle: None,
            state: DownloadState::Pending,
            progress: DownloadProgress::new(),
            torrent_info: None,
            options,
            config,
            peer_stats,
            seeding_manager,
            last_stats_update: Arc::new(Mutex::new(Instant::now())),
            dht_stats: Arc::new(Mutex::new(DhtStats::default())),
            pex_stats: Arc::new(Mutex::new(PexStats::default())),
            tracker_stats: Arc::new(Mutex::new(Vec::new())),
            file_selection_criteria: None,
            output_path,
        };

        // Parse torrent information
        download.parse_torrent().await?;

        Ok(download)
    }

    /// Get DHT statistics
    pub async fn get_dht_stats(&self) -> DhtStats {
        let stats = self.dht_stats.lock().await;
        stats.clone()
    }

    /// Get PEX statistics
    pub async fn get_pex_stats(&self) -> PexStats {
        let stats = self.pex_stats.lock().await;
        stats.clone()
    }

    /// Get tracker statistics
    pub async fn get_tracker_stats(&self) -> Vec<TrackerStats> {
        let stats = self.tracker_stats.lock().await;
        stats.clone()
    }

    /// Set file selection criteria for selective downloading
    pub fn set_file_selection_criteria(&mut self, criteria: FileSelectionCriteria) {
        self.file_selection_criteria = Some(criteria);
        tracing::info!("Set file selection criteria for selective downloading");
    }

    /// Select specific files for downloading by index
    pub async fn select_files(&mut self, file_indices: Vec<usize>) -> Result<()> {
        if let Some(ref mut info) = self.torrent_info {
            for (i, file) in info.files.iter_mut().enumerate() {
                file.selected = file_indices.contains(&i);
                file.priority = if file.selected { FilePriority::Normal } else { FilePriority::Skip };
            }

            let selected_count = info.files.iter().filter(|f| f.selected).count();
            tracing::info!(
                selected_count = selected_count,
                total_count = info.files.len(),
                "Updated file selection"
            );
        }
        Ok(())
    }

    /// Set file priorities for selective downloading
    pub async fn set_file_priorities(&mut self, priorities: Vec<(usize, FilePriority)>) -> Result<()> {
        if let Some(ref mut info) = self.torrent_info {
            let priorities_len = priorities.len();
            for (index, priority) in priorities {
                if let Some(file) = info.files.get_mut(index) {
                    file.priority = priority;
                    file.selected = priority != FilePriority::Skip;
                }
            }

            tracing::info!(
                updated_files = priorities_len,
                "Updated file priorities"
            );
        }
        Ok(())
    }

    /// Get list of files in the torrent with their selection status
    pub async fn get_file_list(&self) -> Vec<TorrentFileInfo> {
        if let Some(ref info) = self.torrent_info {
            info.files.clone()
        } else {
            Vec::new()
        }
    }

    /// Update DHT statistics
    async fn update_dht_stats(&self) {
        if !self.config.enable_dht {
            return;
        }

        // This would be updated from the actual DHT implementation
        // For now, we'll simulate some basic stats
        let mut stats = self.dht_stats.lock().await;
        stats.last_updated = Some(Utc::now());

        // In a real implementation, these would come from the DHT subsystem
        // stats.nodes_count = dht.node_count();
        // stats.good_nodes = dht.good_node_count();
        // etc.

        tracing::debug!(
            nodes = %stats.nodes_count,
            good_nodes = %stats.good_nodes,
            "Updated DHT statistics"
        );
    }

    /// Update PEX statistics
    async fn update_pex_stats(&self) {
        if !self.config.enable_pex {
            return;
        }

        // This would be updated from the actual PEX implementation
        let mut stats = self.pex_stats.lock().await;
        stats.last_updated = Some(Utc::now());

        tracing::debug!(
            peers_received = %stats.peers_received,
            peers_sent = %stats.peers_sent,
            "Updated PEX statistics"
        );
    }

    /// Update tracker statistics
    async fn update_tracker_stats(&self) {
        if !self.config.tracker_config.enabled {
            return;
        }

        // This would be updated from the actual tracker communication
        // For now, we'll maintain placeholder stats
        let mut stats = self.tracker_stats.lock().await;

        // In a real implementation, this would come from the torrent handle
        // and track actual tracker communication
        for _tracker_stat in stats.iter_mut() {
            // Update last announce time, etc.
            // This is placeholder logic
        }

        tracing::debug!(
            tracker_count = stats.len(),
            "Updated tracker statistics"
        );
    }

    /// Initialize tracker statistics from torrent metadata
    async fn initialize_tracker_stats(&self, tracker_urls: Vec<String>) {
        let mut stats = self.tracker_stats.lock().await;
        stats.clear();

        for (tier, url) in tracker_urls.iter().enumerate() {
            let tracker_stat = TrackerStats {
                url: url.clone(),
                last_announce: None,
                next_announce: None,
                successful_announces: 0,
                failed_announces: 0,
                last_error: None,
                seeders: None,
                leechers: None,
                completed: None,
                tier: tier as u32,
            };
            stats.push(tracker_stat);
        }

        tracing::info!(
            tracker_count = tracker_urls.len(),
            "Initialized tracker statistics"
        );
    }

    /// Force announce to all trackers
    pub async fn force_announce(&self) -> Result<()> {
        if !self.config.tracker_config.enabled {
            return Ok(());
        }

        // This would trigger an immediate announce to all trackers
        // In a real implementation, this would use the torrent handle's announce method
        tracing::info!("Force announce requested (would trigger tracker communication)");

        Ok(())
    }

    /// Get encryption status for peer connections
    pub async fn get_encryption_status(&self) -> EncryptionStatus {
        // This would query the actual peer connections for encryption status
        // For now, return a placeholder based on configuration
        EncryptionStatus {
            mse_enabled: self.config.encryption_config.enable_mse,
            encrypted_peers: 0, // Would get from actual peer manager
            unencrypted_peers: 0,
            preferred_method: self.config.encryption_config.allowed_methods.first()
                .cloned()
                .unwrap_or(EncryptionMethod::PlainText),
            encryption_required: self.config.encryption_config.require_encryption,
        }
    }

    /// Check if download is complete and should start seeding
    async fn check_seeding_transition(&mut self) -> Result<()> {
        if !self.config.enable_seeding {
            return Ok(());
        }

        if let Some(handle) = &self.torrent_handle {
            let stats = handle.stats();

            // Check if download is complete
            if stats.progress_bytes >= stats.total_bytes && stats.total_bytes > 0 {
                if let Some(info) = &self.torrent_info {
                    // Start seeding management
                    let mut manager = self.seeding_manager.lock().await;
                    if !manager.active_seeds.contains_key(&info.info_hash) {
                        manager.add_torrent(
                            info.info_hash.clone(),
                            stats.total_bytes,
                            self.config.seeding_ratio,
                            if self.config.max_seeding_time > 0 {
                                Some(Duration::from_secs(self.config.max_seeding_time))
                            } else {
                                None
                            }
                        );

                        tracing::info!(
                            info_hash = %info.info_hash,
                            downloaded_bytes = %stats.total_bytes,
                            target_ratio = %self.config.seeding_ratio,
                            "Started seeding for completed torrent"
                        );
                    }

                    // Update upload statistics
                    manager.update_upload_stats(&info.info_hash, stats.uploaded_bytes);
                }
            }
        }

        Ok(())
    }

    /// Parse torrent file or magnet link to extract information
    async fn parse_torrent(&mut self) -> Result<()> {
        // Check if session is initialized
        {
            let session_guard = self.session.read().await;
            if session_guard.is_none() {
                return Err(RusoError::Protocol(
                    ruso_core::error::ProtocolError::NotInitialized("BitTorrent session not initialized".to_string())
                ));
            }
        }

        if self.url.scheme() == "magnet" {
            // Parse magnet link
            self.parse_magnet_link().await?;
        } else if self.url.path().ends_with(".torrent") {
            // Parse torrent file
            self.parse_torrent_file().await?;
        } else {
            return Err(RusoError::Protocol(
                ruso_core::error::ProtocolError::UnsupportedUrl(
                    format!("Unsupported BitTorrent URL: {}", self.url)
                )
            ));
        }

        Ok(())
    }

    /// Parse magnet link
    async fn parse_magnet_link(&mut self) -> Result<()> {
        let magnet_str = self.url.to_string();

        // For now, we'll do basic magnet link validation and extract display name
        // The actual parsing will be handled by rqbit when we add the torrent
        if !magnet_str.starts_with("magnet:") {
            return Err(RusoError::Protocol(
                ruso_core::error::ProtocolError::ParseError(
                    "Invalid magnet link format".to_string()
                )
            ));
        }

        // Try to extract display name from magnet link
        let name = if let Some(dn_start) = magnet_str.find("dn=") {
            let dn_part = &magnet_str[dn_start + 3..];
            let dn_end = dn_part.find('&').unwrap_or(dn_part.len());
            urlencoding::decode(&dn_part[..dn_end])
                .map(|s| s.to_string())
                .unwrap_or_else(|_| "unknown_torrent".to_string())
        } else {
            "unknown_torrent".to_string()
        };

        // Extract info hash if present
        let info_hash = if let Some(xt_start) = magnet_str.find("xt=urn:btih:") {
            let hash_part = &magnet_str[xt_start + 13..];
            let hash_end = hash_part.find('&').unwrap_or(hash_part.len());
            hash_part[..hash_end].to_string()
        } else {
            "unknown".to_string()
        };

        self.torrent_info = Some(TorrentInfo {
            name,
            total_size: 0, // Will be updated when metadata is received
            piece_count: 0,
            piece_size: 0,
            files: Vec::new(),
            info_hash,
        });

        tracing::info!(
            magnet = %magnet_str,
            info_hash = %self.torrent_info.as_ref().unwrap().info_hash,
            "Parsed magnet link"
        );

        Ok(())
    }

    /// Parse torrent file
    async fn parse_torrent_file(&mut self) -> Result<()> {
        // For now, we'll implement basic torrent file parsing
        // In a full implementation, we would download and parse the .torrent file

        // This is a placeholder - would need to actually fetch and parse the torrent file
        let filename = self.url.path_segments()
            .and_then(|segments| segments.last())
            .unwrap_or("unknown.torrent");

        // Create some example files for demonstration
        let mut example_files = vec![
            TorrentFileInfo {
                path: PathBuf::from("video.mp4"),
                size: 1024 * 1024 * 100, // 100MB
                selected: true,
                priority: FilePriority::Normal,
            },
            TorrentFileInfo {
                path: PathBuf::from("subtitle.srt"),
                size: 1024 * 50, // 50KB
                selected: true,
                priority: FilePriority::Normal,
            },
            TorrentFileInfo {
                path: PathBuf::from("readme.txt"),
                size: 1024 * 2, // 2KB
                selected: true,
                priority: FilePriority::Low,
            },
        ];

        // Apply file selection strategy if configured
        if self.config.enable_selective_download {
            // Apply selection strategy
            if let Err(e) = self.apply_file_selection_strategy(&mut example_files) {
                tracing::warn!(error = %e, "Failed to apply file selection strategy");
            }

            // Apply selection criteria if set
            if let Some(ref criteria) = self.file_selection_criteria {
                if let Err(e) = self.apply_file_selection_criteria_to_files(&mut example_files, criteria) {
                    tracing::warn!(error = %e, "Failed to apply file selection criteria");
                }
            }
        }

        let total_size: u64 = example_files.iter().map(|f| f.size).sum();

        self.torrent_info = Some(TorrentInfo {
            name: filename.trim_end_matches(".torrent").to_string(),
            total_size,
            piece_count: (total_size / (256 * 1024)) as u32 + 1, // Assume 256KB pieces
            piece_size: 256 * 1024, // 256KB
            files: example_files,
            info_hash: "placeholder".to_string(), // Would be calculated from torrent file
        });

        // Initialize tracker stats (placeholder)
        let example_trackers = vec![
            "http://tracker.example.com:8080/announce".to_string(),
            "udp://tracker.example.org:80/announce".to_string(),
        ];
        self.initialize_tracker_stats(example_trackers).await;

        tracing::info!(
            torrent_file = %self.url,
            file_count = self.torrent_info.as_ref().unwrap().files.len(),
            total_size = total_size,
            "Parsed torrent file (placeholder implementation)"
        );

        Ok(())
    }

    /// Apply file selection strategy to files
    fn apply_file_selection_strategy(&self, files: &mut [TorrentFileInfo]) -> Result<()> {
        match &self.config.file_selection_strategy {
            FileSelectionStrategy::All => {
                for file in files.iter_mut() {
                    file.selected = true;
                    file.priority = FilePriority::Normal;
                }
            }
            FileSelectionStrategy::None => {
                for file in files.iter_mut() {
                    file.selected = false;
                    file.priority = FilePriority::Skip;
                }
            }
            FileSelectionStrategy::Pattern(patterns) => {
                for file in files.iter_mut() {
                    file.selected = patterns.iter().any(|pattern| {
                        let file_path = file.path.to_string_lossy();
                        if pattern.contains('*') {
                            let pattern_parts: Vec<&str> = pattern.split('*').collect();
                            if pattern_parts.len() == 2 {
                                file_path.starts_with(pattern_parts[0]) && file_path.ends_with(pattern_parts[1])
                            } else {
                                file_path.contains(pattern)
                            }
                        } else {
                            file_path.contains(pattern)
                        }
                    });
                    file.priority = if file.selected { FilePriority::Normal } else { FilePriority::Skip };
                }
            }
            FileSelectionStrategy::Indices(indices) => {
                for (i, file) in files.iter_mut().enumerate() {
                    file.selected = indices.contains(&i);
                    file.priority = if file.selected { FilePriority::Normal } else { FilePriority::Skip };
                }
            }
            FileSelectionStrategy::LargestFirst(count) => {
                let mut indexed_files: Vec<(usize, u64)> = files.iter().enumerate()
                    .map(|(i, f)| (i, f.size))
                    .collect();
                indexed_files.sort_by(|a, b| b.1.cmp(&a.1));

                for file in files.iter_mut() {
                    file.selected = false;
                    file.priority = FilePriority::Skip;
                }

                for (i, _) in indexed_files.iter().take(*count) {
                    files[*i].selected = true;
                    files[*i].priority = FilePriority::High;
                }
            }
            FileSelectionStrategy::SmallestFirst(count) => {
                let mut indexed_files: Vec<(usize, u64)> = files.iter().enumerate()
                    .map(|(i, f)| (i, f.size))
                    .collect();
                indexed_files.sort_by(|a, b| a.1.cmp(&b.1));

                for file in files.iter_mut() {
                    file.selected = false;
                    file.priority = FilePriority::Skip;
                }

                for (i, _) in indexed_files.iter().take(*count) {
                    files[*i].selected = true;
                    files[*i].priority = FilePriority::High;
                }
            }
        }

        Ok(())
    }

    /// Apply file selection criteria to files
    fn apply_file_selection_criteria_to_files(&self, files: &mut [TorrentFileInfo], criteria: &FileSelectionCriteria) -> Result<()> {
        for file in files.iter_mut() {
            let _file_path = file.path.to_string_lossy();
            let file_name = file.path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");

            // Check size constraints
            if let Some(min_size) = criteria.min_size {
                if file.size < min_size {
                    file.selected = false;
                    file.priority = FilePriority::Skip;
                    continue;
                }
            }

            if let Some(max_size) = criteria.max_size {
                if file.size > max_size {
                    file.selected = false;
                    file.priority = FilePriority::Skip;
                    continue;
                }
            }

            // Check extension constraints
            if !criteria.include_extensions.is_empty() {
                let has_included_ext = criteria.include_extensions.iter().any(|ext| {
                    file_name.ends_with(&format!(".{}", ext))
                });
                if !has_included_ext {
                    file.selected = false;
                    file.priority = FilePriority::Skip;
                    continue;
                }
            }

            if !criteria.exclude_extensions.is_empty() {
                let has_excluded_ext = criteria.exclude_extensions.iter().any(|ext| {
                    file_name.ends_with(&format!(".{}", ext))
                });
                if has_excluded_ext {
                    file.selected = false;
                    file.priority = FilePriority::Skip;
                    continue;
                }
            }

            // If we reach here, the file passes all criteria
            file.selected = true;
            file.priority = FilePriority::Normal;
        }

        Ok(())
    }

    /// Start background progress monitoring task
    async fn start_progress_monitoring(&self, handle: Arc<librqbit::ManagedTorrent>) {
        let shared_state = self.shared_state.clone();
        let shared_progress = self.shared_progress.clone();
        let _peer_stats = self.peer_stats.clone();
        let _seeding_manager = self.seeding_manager.clone();
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(1));

            loop {
                interval.tick().await;

                // Check if download is cancelled or completed
                {
                    let state = shared_state.lock().await;
                    if matches!(*state, DownloadState::Cancelled | DownloadState::Failed(_)) {
                        break;
                    }
                }

                // Update progress from torrent handle
                let stats = handle.stats();

                {
                    let mut progress = shared_progress.lock().await;
                    progress.total_size = Some(stats.total_bytes);
                    progress.downloaded_size = stats.progress_bytes;
                    progress.upload_size = Some(stats.uploaded_bytes);

                    // Calculate completion percentage
                    if stats.total_bytes > 0 {
                        progress.percentage = ((stats.progress_bytes as f64 / stats.total_bytes as f64) * 100.0) as u8;
                    }

                    // Update connections count (placeholder - would need actual peer count from librqbit)
                    progress.connections = 0; // Would get from handle.peer_count() or similar
                }

                // Check if download is complete
                if stats.progress_bytes >= stats.total_bytes && stats.total_bytes > 0 {
                    let mut state = shared_state.lock().await;
                    if !matches!(*state, DownloadState::Completed) {
                        *state = DownloadState::Completed;
                        tracing::info!("BitTorrent download completed");

                        // Start seeding if enabled
                        if config.enable_seeding {
                            // Note: In librqbit, torrents automatically continue seeding after completion
                            // We just need to track the seeding statistics
                            tracing::info!("Starting seeding phase");
                        }
                    }
                }

                // Small delay to prevent excessive CPU usage
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
        });
    }

    /// Update progress from torrent handle with peer and seeding information
    async fn update_progress(&mut self) {
        if let Some(handle) = &self.torrent_handle {
            let stats = handle.stats();
            self.progress.total_size = Some(stats.total_bytes);
            self.progress.downloaded_size = stats.progress_bytes;
            self.progress.upload_size = Some(stats.uploaded_bytes);

            // Calculate speeds based on time elapsed since last update
            {
                let mut last_update = self.last_stats_update.lock().await;
                let now = Instant::now();
                let time_elapsed = now.duration_since(*last_update);

                if time_elapsed >= Duration::from_secs(1) {
                    // Calculate download/upload speeds (simplified calculation)
                    // In a real implementation, we'd track previous values and calculate deltas
                    self.progress.speed = 0; // Would calculate: (current_downloaded - previous_downloaded) / time_elapsed
                    self.progress.upload_speed = Some(0); // Would calculate: (current_uploaded - previous_uploaded) / time_elapsed
                    *last_update = now;
                }
            }

            // Get peer connection count (placeholder - would need actual peer count from librqbit)
            self.progress.connections = 0; // Would get from handle.peer_count() or similar

            // Calculate completion percentage
            if stats.total_bytes > 0 {
                self.progress.percentage = ((stats.progress_bytes as f64 / stats.total_bytes as f64) * 100.0) as u8;
            }

            // Update peer statistics
            if let Some(info) = &self.torrent_info {
                let peer_stats = PeerStats {
                    connected_peers: 0, // Would get from actual peer manager
                    downloading_peers: 0,
                    seeding_peers: 0,
                    total_downloaded: stats.progress_bytes,
                    total_uploaded: stats.uploaded_bytes,
                    last_updated: Utc::now(),
                };

                let mut stats_map = self.peer_stats.lock().await;
                stats_map.insert(info.info_hash.clone(), peer_stats);
            }

            // Update DHT and PEX statistics
            self.update_dht_stats().await;
            self.update_pex_stats().await;

            // Check for seeding transition
            let _ = self.check_seeding_transition().await;
        }
    }
}

#[allow(dead_code)]
#[async_trait]
impl Download for BitTorrentDownload {
    async fn start(&mut self) -> Result<()> {
        if self.state != DownloadState::Pending {
            return Err(RusoError::InvalidStateTransition {
                from: self.state.clone(),
                to: DownloadState::Active,
            });
        }

        let session_guard = self.session.read().await;
        let _session = session_guard.as_ref()
            .ok_or_else(|| RusoError::Protocol(
                ruso_core::error::ProtocolError::NotInitialized("BitTorrent session not initialized".to_string())
            ))?.clone();
        drop(session_guard);

        // Prepare file selection for selective downloading
        let only_files = if self.config.enable_selective_download {
            if let Some(ref info) = self.torrent_info {
                let selected_indices: Vec<usize> = info.files.iter()
                    .enumerate()
                    .filter_map(|(i, file)| if file.selected { Some(i) } else { None })
                    .collect();
                if selected_indices.len() < info.files.len() { Some(selected_indices) } else { None }
            } else {
                None
            }
        } else { None };

        // Add torrent to session with selective download options
        let _add_torrent_options = librqbit::AddTorrentOptions {
            paused: false,
            only_files,
            output_folder: self.output_path.as_ref().map(|p| p.to_string_lossy().to_string()),
            overwrite: self.options.overwrite,
            ..Default::default()
        };

        if self.url.scheme() == "magnet" {
            // Add magnet link to session
            tracing::info!("BitTorrent magnet requested: {}", self.url);

            let session_guard = self.session.read().await;
            let session = session_guard.as_ref()
                .ok_or_else(|| RusoError::Protocol(
                    ruso_core::error::ProtocolError::NotInitialized("BitTorrent session not initialized".to_string())
                ))?;

            // Add magnet link to session
            let magnet_str = self.url.to_string();
            let add_torrent = librqbit::AddTorrent::from_url(&magnet_str);

            match session.add_torrent(add_torrent, Some(_add_torrent_options)).await {
                Ok(response) => {
                    // Extract the torrent ID from the response
                    let torrent_id = match response {
                        librqbit::AddTorrentResponse::AlreadyManaged(id, _) => id,
                        librqbit::AddTorrentResponse::Added(id, _) => id,
                        librqbit::AddTorrentResponse::ListOnly(_) => {
                            let error_msg = "Magnet link was added in list-only mode, cannot download".to_string();
                            tracing::error!("{}", error_msg);
                            self.state = DownloadState::Failed(error_msg.clone());
                            {
                                let mut st = self.shared_state.lock().await;
                                *st = self.state.clone();
                            }
                            return Err(RusoError::Protocol(
                                ruso_core::error::ProtocolError::InitializationFailed(error_msg)
                            ));
                        }
                    };

                    // Get the managed torrent from the session
                    let handle = session.get(librqbit::api::TorrentIdOrHash::Id(torrent_id))
                        .ok_or_else(|| RusoError::Protocol(
                            ruso_core::error::ProtocolError::InitializationFailed(
                                "Failed to get torrent handle from session".to_string()
                            )
                        ))?;

                    self.torrent_handle = Some(handle.clone());
                    self.state = DownloadState::Active;
                    {
                        let mut st = self.shared_state.lock().await;
                        *st = DownloadState::Active;
                    }

                    // Start background progress monitoring task
                    self.start_progress_monitoring(handle.clone()).await;

                    tracing::info!("Successfully added magnet link to BitTorrent session");
                    Ok(())
                }
                Err(e) => {
                    let error_msg = format!("Failed to add magnet link to session: {}", e);
                    tracing::error!("{}", error_msg);
                    self.state = DownloadState::Failed(error_msg.clone());
                    {
                        let mut st = self.shared_state.lock().await;
                        *st = self.state.clone();
                    }
                    Err(RusoError::Protocol(
                        ruso_core::error::ProtocolError::InitializationFailed(error_msg)
                    ))
                }
            }
        } else if self.url.path().ends_with(".torrent") {
            // Handle torrent file URL
            tracing::info!("BitTorrent file requested: {}", self.url);

            let session_guard = self.session.read().await;
            let session = session_guard.as_ref()
                .ok_or_else(|| RusoError::Protocol(
                    ruso_core::error::ProtocolError::NotInitialized("BitTorrent session not initialized".to_string())
                ))?;

            // Add torrent file URL to session
            let torrent_url = self.url.to_string();
            let add_torrent = librqbit::AddTorrent::from_url(&torrent_url);

            match session.add_torrent(add_torrent, Some(_add_torrent_options)).await {
                Ok(response) => {
                    // Extract the torrent ID from the response
                    let torrent_id = match response {
                        librqbit::AddTorrentResponse::AlreadyManaged(id, _) => id,
                        librqbit::AddTorrentResponse::Added(id, _) => id,
                        librqbit::AddTorrentResponse::ListOnly(_) => {
                            let error_msg = "Torrent file was added in list-only mode, cannot download".to_string();
                            tracing::error!("{}", error_msg);
                            self.state = DownloadState::Failed(error_msg.clone());
                            {
                                let mut st = self.shared_state.lock().await;
                                *st = self.state.clone();
                            }
                            return Err(RusoError::Protocol(
                                ruso_core::error::ProtocolError::InitializationFailed(error_msg)
                            ));
                        }
                    };

                    // Get the managed torrent from the session
                    let handle = session.get(librqbit::api::TorrentIdOrHash::Id(torrent_id))
                        .ok_or_else(|| RusoError::Protocol(
                            ruso_core::error::ProtocolError::InitializationFailed(
                                "Failed to get torrent handle from session".to_string()
                            )
                        ))?;

                    self.torrent_handle = Some(handle.clone());
                    self.state = DownloadState::Active;
                    {
                        let mut st = self.shared_state.lock().await;
                        *st = DownloadState::Active;
                    }

                    // Start background progress monitoring task
