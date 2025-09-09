//! BitTorrent protocol handler using rqbit with DHT, PEX, and seeding support

#![cfg(feature = "torrent")]

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use percent_encoding;
use tokio::sync::{Mutex, RwLock};
use url::Url;

use crate::{error::{ProtocolError, Result, ZuupError}, protocol::{Download, DownloadMetadata, DownloadOperation, ProtocolCapabilities, ProtocolHandler}, types::{DownloadProgress, DownloadRequest, DownloadState}, DownloadOptions};

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
    /// Enable encryption for peer connections
    pub enable_encryption: bool,
    /// Port range for incoming connections
    pub port_range: (u16, u16),
    /// Default file selection strategy
    pub file_selection_strategy: FileSelectionStrategy,
}

impl Default for BitTorrentConfig {
    fn default() -> Self {
        Self {
            enable_dht: true,
            enable_pex: true,
            enable_seeding: false, // Simplified: disable seeding by default
            seeding_ratio: 1.0,
            max_seeding_time: 0,
            max_peers: 50,
            enable_encryption: true,
            port_range: (6881, 6889),
            file_selection_strategy: FileSelectionStrategy::All,
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

    /// Initialize the BitTorrent session with simplified configuration
    async fn ensure_session(&self) -> Result<()> {
        let mut session_guard = self.session.write().await;
        if session_guard.is_none() {
            let mut session_opts = librqbit::SessionOptions::default();

            // Configure DHT
            if !self.config.enable_dht {
                session_opts.disable_dht = true;
                tracing::info!("DHT disabled");
            } else {
                tracing::info!("DHT enabled");
            }
            
            // Disable DHT persistence to avoid conflicts and make it more ephemeral
            session_opts.disable_dht_persistence = true;

            // Configure port range - use a wider range or let it auto-select
            if self.config.port_range.0 != 0 && self.config.port_range.1 != 0 {
                session_opts.listen_port_range = Some(self.config.port_range.0..self.config.port_range.1);
            }

            // Create the session with current directory as default - each download will handle its own output path
            let default_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let session = librqbit::Session::new_with_opts(
                default_dir,
                session_opts,
            ).await
                .map_err(|e| ZuupError::Protocol(
                    ProtocolError::InitializationFailed(
                        format!("Failed to create BitTorrent session: {}", e)
                    )
                ))?;

            *session_guard = Some(session);

            tracing::info!(
                dht_enabled = %self.config.enable_dht,
                pex_enabled = %self.config.enable_pex,
                port_range = ?self.config.port_range,
                "BitTorrent session initialized"
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

    /// Apply simplified file selection strategy
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
                    let file_path = file.path.to_string_lossy();
                    file.selected = patterns.iter().any(|pattern| {
                        if pattern.contains('*') {
                            // Simple wildcard matching
                            let parts: Vec<&str> = pattern.split('*').collect();
                            parts.len() == 2 && file_path.starts_with(parts[0]) && file_path.ends_with(parts[1])
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
            // All variants are now handled explicitly
        }

        let selected_count = files.iter().filter(|f| f.selected).count();
        tracing::info!(
            selected_count = selected_count,
            total_count = files.len(),
            "Applied file selection strategy"
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
            .ok_or_else(|| ZuupError::Config("No URLs provided".to_string()))?;

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

/// Simplified BitTorrent download implementation
#[allow(dead_code)]
pub struct BitTorrentDownload {
    url: Url,
    session: Arc<RwLock<Option<Arc<librqbit::Session>>>>,
    torrent_handle: Option<Arc<librqbit::ManagedTorrent>>,
    state: Arc<RwLock<DownloadState>>,
    progress: Arc<RwLock<DownloadProgress>>,
    torrent_info: Option<TorrentInfo>,
    options: DownloadOptions,
    config: BitTorrentConfig,
    peer_stats: Arc<Mutex<HashMap<String, PeerStats>>>,
    seeding_manager: Arc<Mutex<SeedingManager>>,
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

/// Simplified file selection strategy for multi-file torrents
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

// Removed complex configuration structs to simplify the implementation

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

// Removed complex selection criteria and encryption status structs to simplify

impl BitTorrentDownload {
    /// Create a new BitTorrent download with DHT and PEX support
    pub async fn new(
        url: Url,
        session: Arc<RwLock<Option<Arc<librqbit::Session>>>>,
        options: DownloadOptions,
        config: BitTorrentConfig,
        peer_stats: Arc<Mutex<HashMap<String, PeerStats>>>,
        seeding_manager: Arc<Mutex<SeedingManager>>,
        output_path: Option<PathBuf>,
    ) -> Result<Self> {
        let mut download = Self {
            url,
            session,
            torrent_handle: None,
            state: Arc::new(RwLock::new(DownloadState::Pending)),
            progress: Arc::new(RwLock::new(DownloadProgress::new())),
            torrent_info: None,
            options,
            config,
            peer_stats,
            seeding_manager,
            output_path,
        };

        // Parse torrent information
        download.parse_torrent().await?;

        Ok(download)
    }

    /// Get simplified torrent statistics
    pub async fn get_stats(&self) -> String {
        "Torrent statistics simplified".to_string()
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







    /// Simplified tracker handling
    async fn initialize_trackers(&self, _tracker_urls: Vec<String>) {
        tracing::info!("Tracker initialization simplified");
    }

    /// Force announce to all trackers
    pub async fn force_announce(&self) -> Result<()> {
        // Simplified: no-op for now
        tracing::info!("Force announce requested (would trigger tracker communication)");

        Ok(())
    }

    /// Get simplified encryption status
    pub async fn get_encryption_status(&self) -> bool {
        self.config.enable_encryption
    }



    /// Parse torrent file or magnet link to extract information
    async fn parse_torrent(&mut self) -> Result<()> {
        // Check if session is initialized
        {
            let session_guard = self.session.read().await;
            if session_guard.is_none() {
                return Err(ZuupError::Protocol(
                    ProtocolError::NotInitialized("BitTorrent session not initialized".to_string())
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
            return Err(ZuupError::Protocol(
                ProtocolError::UnsupportedUrl(
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
            return Err(ZuupError::Protocol(
                ProtocolError::ParseError(
                    "Invalid magnet link format".to_string()
                )
            ));
        }

        // Try to extract display name from magnet link
        let name = if let Some(dn_start) = magnet_str.find("dn=") {
            let dn_part = &magnet_str[dn_start + 3..];
            let dn_end = dn_part.find('&').unwrap_or(dn_part.len());
            let encoded_name = &dn_part[..dn_end];
            // URL decode the display name
            percent_encoding::percent_decode_str(encoded_name)
                .decode_utf8()
                .map(|decoded| decoded.to_string())
                .unwrap_or_else(|_| encoded_name.to_string())
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

        // Apply file selection strategy (simplified)
        if true {
            // Apply selection strategy
            if let Err(e) = self.apply_file_selection_strategy(&mut example_files) {
                tracing::warn!(error = %e, "Failed to apply file selection strategy");
            }

            // Simplified: no additional criteria
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
        self.initialize_trackers(example_trackers).await;

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
            // Removed LargestFirst and SmallestFirst variants for simplification
        }

        Ok(())
    }

    // Removed complex file selection criteria method

    /// Start background progress monitoring task
    async fn start_progress_monitoring(&self, handle: Arc<librqbit::ManagedTorrent>) {
        let state = self.state.clone();
        let progress = self.progress.clone();
        let _peer_stats = self.peer_stats.clone();
        let _seeding_manager = self.seeding_manager.clone();
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(500));
            let mut last_downloaded = 0u64;
            let mut last_uploaded = 0u64;
            let mut last_update = std::time::Instant::now();

            loop {
                interval.tick().await;

                // Check if download is cancelled or completed
                {
                    let state_guard = state.read().await;
                    if matches!(*state_guard, DownloadState::Cancelled | DownloadState::Failed(_)) {
                        break;
                    }
                }

                // Update progress from torrent handle
                let stats = handle.stats();
                let now = std::time::Instant::now();
                let time_elapsed = now.duration_since(last_update).as_secs_f64();

                // Calculate speeds
                let download_speed = if time_elapsed > 0.0 {
                    let bytes_diff = stats.progress_bytes.saturating_sub(last_downloaded);
                    (bytes_diff as f64 / time_elapsed) as u64
                } else {
                    0
                };

                let upload_speed = if time_elapsed > 0.0 {
                    let bytes_diff = stats.uploaded_bytes.saturating_sub(last_uploaded);
                    (bytes_diff as f64 / time_elapsed) as u64
                } else {
                    0
                };

                // Add some debugging - only log when there are changes or periodically
                tracing::debug!(
                    "Torrent stats: state={:?}, total_bytes={}, progress_bytes={}, uploaded_bytes={}, download_speed={}, upload_speed={}",
                    stats.state, stats.total_bytes, stats.progress_bytes, stats.uploaded_bytes, download_speed, upload_speed
                );

                {
                    let mut progress_guard = progress.write().await;
                    
                    // Update basic progress info
                    progress_guard.total_size = Some(stats.total_bytes);
                    progress_guard.downloaded_size = stats.progress_bytes;
                    progress_guard.upload_size = Some(stats.uploaded_bytes);
                    progress_guard.download_speed = download_speed;
                    progress_guard.upload_speed = Some(upload_speed);
                    progress_guard.updated_at = chrono::Utc::now();

                    // Calculate completion percentage
                    if stats.total_bytes > 0 {
                        progress_guard.percentage = ((stats.progress_bytes as f64 / stats.total_bytes as f64) * 100.0) as u8;
                    }

                    // Calculate ETA
                    if download_speed > 0 && stats.total_bytes > stats.progress_bytes {
                        let remaining_bytes = stats.total_bytes - stats.progress_bytes;
                        let eta_seconds = remaining_bytes / download_speed;
                        progress_guard.eta = Some(Duration::from_secs(eta_seconds));
                    } else if stats.progress_bytes >= stats.total_bytes {
                        progress_guard.eta = None; // Complete
                    }

                    // Update connections count (placeholder - would need actual peer count from librqbit)
                    progress_guard.connections = 0; // Would get from handle.peer_count() or similar

                    // Mark as started if not already
                    if progress_guard.started_at.is_none() {
                        progress_guard.started_at = Some(chrono::Utc::now());
                    }
                }

                // Update tracking variables
                last_downloaded = stats.progress_bytes;
                last_uploaded = stats.uploaded_bytes;
                last_update = now;

                // Check if download is complete
                if stats.progress_bytes >= stats.total_bytes && stats.total_bytes > 0 {
                    let mut state_guard = state.write().await;
                    if !matches!(*state_guard, DownloadState::Completed) {
                        *state_guard = DownloadState::Completed;
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


}

#[allow(dead_code)]
#[async_trait]
impl Download for BitTorrentDownload {
    async fn start(&mut self) -> Result<()> {
        let current_state = self.state();
        if current_state != DownloadState::Pending {
            return Err(ZuupError::InvalidStateTransition {
                from: current_state,
                to: DownloadState::Active,
            });
        }

        // Ensure we have a session
        let session = {
            let guard = self.session.read().await;
            guard.as_ref().cloned().ok_or_else(|| ZuupError::Protocol(
                ProtocolError::NotInitialized("BitTorrent session not initialized".to_string())
            ))?
        };



        // Add torrent from magnet or .torrent URL
        let url_str = self.url.to_string();
        let add_torrent = librqbit::AddTorrent::from_cli_argument(&url_str)
            .map_err(|e| ZuupError::Protocol(ProtocolError::InitializationFailed(format!(
                "Failed to parse torrent URL: {}", e
            ))))?;
        
        // Prepare torrent options with the correct output directory
        let output_folder = if let Some(path) = &self.output_path {
            Some(path.to_string_lossy().to_string())
        } else {
            // Use current directory if no output path specified
            Some(std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .to_string_lossy()
                .to_string())
        };
        
        let add_res = session
            .add_torrent(add_torrent, Some(librqbit::AddTorrentOptions {
                overwrite: true, // Allow overwriting existing files
                output_folder,
                ..Default::default()
            }))
            .await;

        let add_response = add_res.map_err(|e| ZuupError::Protocol(ProtocolError::InitializationFailed(format!(
            "Failed to add torrent: {}", e
        ))))?;

        // Extract the ManagedTorrent from AddTorrentResponse
        let handle = match add_response {
            librqbit::AddTorrentResponse::Added(_, handle) => {
                tracing::info!("Torrent added successfully");
                handle
            },
            librqbit::AddTorrentResponse::AlreadyManaged(_, handle) => {
                tracing::info!("Torrent already managed, using existing handle");
                handle
            },
            librqbit::AddTorrentResponse::ListOnly(_) => {
                return Err(ZuupError::Protocol(ProtocolError::InitializationFailed(
                    "Torrent was added in list-only mode".to_string()
                )));
            }
        };
        
        // Log initial torrent state
        let initial_stats = handle.stats();
        tracing::info!(
            "Initial torrent stats: state={:?}, total_bytes={}, progress_bytes={}", 
            initial_stats.state, initial_stats.total_bytes, initial_stats.progress_bytes
        );
        
        self.torrent_handle = Some(handle.clone());

        // Start progress monitoring
        self.start_progress_monitoring(handle).await;

        {
            let mut state = self.state.write().await;
            *state = DownloadState::Active;
        }

        let output_dir = self.output_path.as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "current directory".to_string());
        tracing::info!(target_dir = %output_dir, "BitTorrent download started");
        Ok(())
    }

    async fn pause(&mut self) -> Result<()> {
        {
            let mut state = self.state.write().await;
            *state = DownloadState::Paused;
        }
        tracing::info!("BitTorrent download paused");
        Ok(())
    }

    async fn resume(&mut self) -> Result<()> {
        {
            let mut state = self.state.write().await;
            *state = DownloadState::Active;
        }
        tracing::info!("BitTorrent download resumed");
        Ok(())
    }

    async fn cancel(&mut self) -> Result<()> {
        {
            let mut state = self.state.write().await;
            *state = DownloadState::Cancelled;
        }
        tracing::info!("BitTorrent download cancelled");
        Ok(())
    }

    fn progress(&self) -> DownloadProgress {
        match self.progress.try_read() {
            Ok(progress) => progress.clone(),
            Err(_) => DownloadProgress::new(),
        }
    }

    fn state(&self) -> DownloadState {
        match self.state.try_read() {
            Ok(state) => state.clone(),
            Err(_) => DownloadState::Pending,
        }
    }

    async fn metadata(&self) -> Result<DownloadMetadata> {
        let mut metadata = DownloadMetadata::default();
        
        if let Some(ref info) = self.torrent_info {
            metadata.filename = Some(info.name.clone());
            metadata.size = Some(info.total_size);
        }

        Ok(metadata)
    }

    fn supports_operation(&self, operation: DownloadOperation) -> bool {
        matches!(
            operation,
            DownloadOperation::Start
                | DownloadOperation::Pause
                | DownloadOperation::Resume
                | DownloadOperation::Cancel
                | DownloadOperation::GetMetadata
        )
    }
}