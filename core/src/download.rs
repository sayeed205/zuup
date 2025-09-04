use std::path::PathBuf;
use std::sync::Arc;
use std::{collections::HashMap, time::Duration};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use url::Url;

use crate::types::{DownloadId, DownloadOptions, SegmentProgress};

/// Download request containing URLs and configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadRequest {
    /// List of URLs to download from (for multi-source downloads)
    pub urls: Vec<Url>,

    /// Output directory path
    pub output_path: Option<PathBuf>,

    /// Custom filename (if not specified, derived from URL)
    pub filename: Option<String>,

    /// Download options and configuration
    pub options: DownloadOptions,
}

impl DownloadRequest {
    /// Create a new download request with a single URL
    pub fn new(url: Url) -> Self {
        Self {
            urls: vec![url],
            output_path: None,
            filename: None,
            options: DownloadOptions::default(),
        }
    }

    /// Create a new download request with multiple URLs
    pub fn with_urls(urls: Vec<Url>) -> Self {
        Self {
            urls,
            output_path: None,
            filename: None,
            options: DownloadOptions::default(),
        }
    }

    /// Set the output path
    pub fn output_path(mut self, path: PathBuf) -> Self {
        self.output_path = Some(path);
        self
    }

    /// Set the filename
    pub fn filename(mut self, name: String) -> Self {
        self.filename = Some(name);
        self
    }

    /// Set the download options
    pub fn options(mut self, options: DownloadOptions) -> Self {
        self.options = options;
        self
    }
}

/// Current state of a download
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DownloadState {
    /// Download is queued but not started
    Pending,

    /// Download is actively running
    Active,

    /// Download is paused
    Paused,

    /// Download completed successfully
    Completed,

    /// Download failed with an error
    Failed(String), // Store error as string for serialization

    /// Download was cancelled by user
    Cancelled,

    /// Download is waiting for resources
    Waiting,

    /// Download is being prepared (analyzing, creating segments, etc.)
    Preparing,
}

/// Priority level for downloads
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum DownloadPriority {
    /// Lowest priority
    Low = 1,
    /// Normal priority (default)
    Normal = 2,
    /// High priority
    High = 3,
    /// Highest priority
    Critical = 4,
}

impl Default for DownloadPriority {
    fn default() -> Self {
        DownloadPriority::Normal
    }
}

/// Task control commands
#[derive(Debug, Clone)]
pub enum TaskCommand {
    Start,
    Pause,
    Resume,
    Cancel,
    UpdatePriority(DownloadPriority),
}

impl DownloadState {
    /// Check if the download is in a terminal state
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            DownloadState::Completed | DownloadState::Failed(_) | DownloadState::Cancelled
        )
    }

    /// Check if the download is active
    pub fn is_active(&self) -> bool {
        matches!(self, DownloadState::Active)
    }

    /// Check if the download can be resumed
    pub fn can_resume(&self) -> bool {
        matches!(self, DownloadState::Paused | DownloadState::Failed(_))
    }

    /// Check if the download can be paused
    pub fn can_pause(&self) -> bool {
        matches!(self, DownloadState::Active | DownloadState::Preparing)
    }

    /// Check if the download can be started
    pub fn can_start(&self) -> bool {
        matches!(self, DownloadState::Pending | DownloadState::Waiting)
    }

    /// Check if the download is waiting for resources
    pub fn is_waiting(&self) -> bool {
        matches!(self, DownloadState::Waiting)
    }

    /// Check if the download is preparing
    pub fn is_preparing(&self) -> bool {
        matches!(self, DownloadState::Preparing)
    }

    /// Validate state transition
    pub fn can_transition_to(&self, new_state: &DownloadState) -> bool {
        use DownloadState::*;

        match (self, new_state) {
            // From Pending
            (Pending, Waiting) => true,
            (Pending, Preparing) => true,
            (Pending, Cancelled) => true,

            // From Waiting
            (Waiting, Preparing) => true,
            (Waiting, Cancelled) => true,

            // From Preparing
            (Preparing, Active) => true,
            (Preparing, Paused) => true,
            (Preparing, Failed(_)) => true,
            (Preparing, Cancelled) => true,

            // From Active
            (Active, Paused) => true,
            (Active, Completed) => true,
            (Active, Failed(_)) => true,
            (Active, Cancelled) => true,

            // From Paused
            (Paused, Active) => true,
            (Paused, Cancelled) => true,

            // From Failed
            (Failed(_), Preparing) => true,
            (Failed(_), Cancelled) => true,

            // Terminal states cannot transition
            (Completed, _) | (Cancelled, _) => false,

            // Same state is always valid
            (a, b) if a == b => true,

            // All other transitions are invalid
            _ => false,
        }
    }
}

/// Progress information for a download
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadProgress {
    /// Total size of the download in bytes (if known)
    pub total_size: Option<u64>,

    /// Number of bytes downloaded so far
    pub downloaded_size: u64,

    /// Current download speed in bytes per second
    pub download_speed: u64,

    /// Current upload speed in bytes per second (if available)
    pub upload_speed: Option<u64>,

    /// Estimated time to completion
    pub eta: Option<Duration>,

    /// Number of active connections
    pub connections: u32,

    /// Progress of individual segments
    pub segments: Vec<SegmentProgress>,
}

impl DownloadProgress {
    /// Create a new progress instance
    pub fn new() -> Self {
        Self {
            total_size: None,
            downloaded_size: 0,
            download_speed: 0,
            upload_speed: None,
            eta: None,
            connections: 0,
            segments: Vec::new(),
        }
    }

    /// Calculate completion percentage (0.0 to 1.0)
    pub fn percentage(&self) -> f64 {
        self.total_size.map_or(0.0, |total| {
            if total == 0 {
                1.0
            } else {
                self.downloaded_size as f64 / total as f64
            }
        })
    }

    /// Check if the download is complete
    pub fn is_complete(&self) -> bool {
        if let Some(total) = self.total_size {
            self.downloaded_size >= total
        } else {
            false
        }
    }
}

impl Default for DownloadProgress {
    fn default() -> Self {
        Self::new()
    }
}

/// Complete information about a download
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadInfo {
    /// Unique download identifier
    pub id: DownloadId,

    /// List of URLs being downloaded from
    pub urls: Vec<Url>,

    /// Final filename
    pub filename: String,

    /// Output path where file will be saved
    pub output_path: PathBuf,

    /// Current download state
    pub state: DownloadState,

    /// Current progress information
    pub progress: DownloadProgress,

    /// Download priority
    pub priority: DownloadPriority,

    /// When the download was created
    pub created_at: DateTime<Utc>,

    /// When the download was started (if ever)
    pub started_at: Option<DateTime<Utc>>,

    /// When the download was completed (if completed)
    pub completed_at: Option<DateTime<Utc>>,

    /// Download options used
    pub options: DownloadOptions,
}

#[derive(Clone)]
pub struct DownloadTask {
    /// Unique identifier
    id: DownloadId,
    // todo)) add more info about download task
}

impl DownloadTask {
    /// Create a new download task.
    pub fn new(id: DownloadId) -> Self {
        Self {
            id,
            // todo)) add more info about download task
        }
    }
}

/// Download manager responsible for coordinating downloads.
pub struct DownloadManager {
    /// All downloads(active, pending, completed, failed, stopped)
    downloads: Arc<RwLock<HashMap<DownloadId, Arc<DownloadTask>>>>,
}
