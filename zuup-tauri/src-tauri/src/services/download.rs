use std::sync::Arc;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;
use chrono::{DateTime, Utc};

#[derive(Error, Debug)]
pub enum DownloadError {
    #[error("Download not found: {0}")]
    NotFound(String),
    #[error("Invalid URL: {0}")]
    InvalidUrl(String),
    #[error("Core engine error: {0}")]
    CoreEngine(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadInfo {
    pub id: String,
    pub url: String,
    pub filename: String,
    pub file_size: Option<u64>,
    pub downloaded_size: u64,
    pub progress: ProgressInfo,
    pub state: DownloadState,
    pub created_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressInfo {
    pub percentage: f64,
    pub download_speed: u64,
    pub upload_speed: u64,
    pub eta: Option<u64>,
    pub connections: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DownloadState {
    Preparing,
    Active,
    Paused,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadStats {
    pub total_downloads: u32,
    pub active_downloads: u32,
    pub completed_downloads: u32,
    pub failed_downloads: u32,
    pub paused_downloads: u32,
    pub total_download_speed: u64,
    pub total_upload_speed: u64,
    pub overall_progress: f64,
    pub eta: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddDownloadRequest {
    pub url: String,
    pub download_path: Option<String>,
    pub filename: Option<String>,
}

impl Default for ProgressInfo {
    fn default() -> Self {
        Self {
            percentage: 0.0,
            download_speed: 0,
            upload_speed: 0,
            eta: None,
            connections: 0,
        }
    }
}

pub struct DownloadService {
    downloads: Arc<Mutex<Vec<DownloadInfo>>>,
    // TODO: Add ZuupEngine integration in next task
}

impl DownloadService {
    pub fn new() -> Self {
        Self {
            downloads: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn add_download(&self, request: AddDownloadRequest) -> Result<String, DownloadError> {
        // Validate URL
        if !self.is_valid_url(&request.url) {
            return Err(DownloadError::InvalidUrl(request.url));
        }

        let id = Uuid::new_v4().to_string();
        let filename = request.filename.unwrap_or_else(|| {
            self.extract_filename_from_url(&request.url)
        });

        let download = DownloadInfo {
            id: id.clone(),
            url: request.url,
            filename,
            file_size: None,
            downloaded_size: 0,
            progress: ProgressInfo::default(),
            state: DownloadState::Preparing,
            created_at: Utc::now(),
            completed_at: None,
            error: None,
        };

        self.downloads.lock().push(download);
        
        // TODO: Integrate with ZuupEngine in next task
        
        Ok(id)
    }

    pub fn pause_download(&self, id: &str) -> Result<(), DownloadError> {
        let mut downloads = self.downloads.lock();
        let download = downloads.iter_mut()
            .find(|d| d.id == id)
            .ok_or_else(|| DownloadError::NotFound(id.to_string()))?;

        if matches!(download.state, DownloadState::Active) {
            download.state = DownloadState::Paused;
            // TODO: Integrate with ZuupEngine in next task
        }

        Ok(())
    }

    pub fn resume_download(&self, id: &str) -> Result<(), DownloadError> {
        let mut downloads = self.downloads.lock();
        let download = downloads.iter_mut()
            .find(|d| d.id == id)
            .ok_or_else(|| DownloadError::NotFound(id.to_string()))?;

        if matches!(download.state, DownloadState::Paused) {
            download.state = DownloadState::Active;
            // TODO: Integrate with ZuupEngine in next task
        }

        Ok(())
    }

    pub fn cancel_download(&self, id: &str) -> Result<(), DownloadError> {
        let mut downloads = self.downloads.lock();
        let download = downloads.iter_mut()
            .find(|d| d.id == id)
            .ok_or_else(|| DownloadError::NotFound(id.to_string()))?;

        download.state = DownloadState::Cancelled;
        // TODO: Integrate with ZuupEngine in next task

        Ok(())
    }

    pub fn remove_download(&self, id: &str) -> Result<(), DownloadError> {
        let mut downloads = self.downloads.lock();
        let index = downloads.iter().position(|d| d.id == id)
            .ok_or_else(|| DownloadError::NotFound(id.to_string()))?;

        downloads.remove(index);
        Ok(())
    }

    pub fn get_downloads(&self) -> Vec<DownloadInfo> {
        self.downloads.lock().clone()
    }

    pub fn get_download_stats(&self) -> DownloadStats {
        let downloads = self.downloads.lock();
        
        let total_downloads = downloads.len() as u32;
        let active_downloads = downloads.iter().filter(|d| matches!(d.state, DownloadState::Active)).count() as u32;
        let completed_downloads = downloads.iter().filter(|d| matches!(d.state, DownloadState::Completed)).count() as u32;
        let failed_downloads = downloads.iter().filter(|d| matches!(d.state, DownloadState::Failed)).count() as u32;
        let paused_downloads = downloads.iter().filter(|d| matches!(d.state, DownloadState::Paused)).count() as u32;

        let total_download_speed = downloads.iter()
            .filter(|d| matches!(d.state, DownloadState::Active))
            .map(|d| d.progress.download_speed)
            .sum();

        let total_upload_speed = downloads.iter()
            .filter(|d| matches!(d.state, DownloadState::Active))
            .map(|d| d.progress.upload_speed)
            .sum();

        let overall_progress = if total_downloads > 0 {
            downloads.iter().map(|d| d.progress.percentage).sum::<f64>() / total_downloads as f64
        } else {
            0.0
        };

        DownloadStats {
            total_downloads,
            active_downloads,
            completed_downloads,
            failed_downloads,
            paused_downloads,
            total_download_speed,
            total_upload_speed,
            overall_progress,
            eta: None, // TODO: Calculate ETA
        }
    }

    fn is_valid_url(&self, url: &str) -> bool {
        url::Url::parse(url).is_ok()
    }

    fn extract_filename_from_url(&self, url: &str) -> String {
        if let Ok(parsed_url) = url::Url::parse(url) {
            if let Some(segments) = parsed_url.path_segments() {
                if let Some(filename) = segments.last() {
                    if !filename.is_empty() {
                        return filename.to_string();
                    }
                }
            }
        }
        "download".to_string()
    }
}