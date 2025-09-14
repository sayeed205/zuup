use super::event::EventService;
use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::thread;
use thiserror::Error;
use tokio::sync::mpsc;

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

#[derive(Clone)]
pub struct DownloadService {
    downloads: Arc<Mutex<Vec<DownloadInfo>>>,
    event_service: Arc<Mutex<Option<EventService>>>,
    download_tx: Arc<Mutex<Option<mpsc::UnboundedSender<DownloadCommand>>>>,
}

#[derive(Debug)]
enum DownloadCommand {
    AddDownload {
        id: String,
        url: String,
        filename: Option<String>,
    },
    PauseDownload(String),
    ResumeDownload(String),
    CancelDownload(String),
    RemoveDownload(String),
}

struct DownloadWorker {
    downloads: Arc<Mutex<Vec<DownloadInfo>>>,
    event_service: Arc<Mutex<Option<EventService>>>,
    running: Arc<AtomicBool>,
}

impl DownloadService {
    pub fn new() -> Self {
        Self {
            downloads: Arc::new(Mutex::new(Vec::new())),
            event_service: Arc::new(Mutex::new(None)),
            download_tx: Arc::new(Mutex::new(None)),
        }
    }

    pub fn set_event_service(&self, event_service: EventService) {
        *self.event_service.lock() = Some(event_service);

        // Start the download worker
        self.start_download_worker();
    }

    fn start_download_worker(&self) {
        let (tx, rx) = mpsc::unbounded_channel();
        *self.download_tx.lock() = Some(tx);

        let downloads = self.downloads.clone();
        let event_service = self.event_service.clone();
        let running = Arc::new(AtomicBool::new(true));

        thread::spawn(move || {
            let worker = DownloadWorker {
                downloads,
                event_service,
                running,
            };
            worker.run(rx);
        });
    }

    pub async fn add_download(&self, request: AddDownloadRequest) -> Result<String, DownloadError> {
        // Validate URL
        if !self.is_valid_url(&request.url) {
            return Err(DownloadError::InvalidUrl(request.url));
        }

        let id = uuid::Uuid::new_v4().to_string();
        let filename = request
            .filename
            .unwrap_or_else(|| self.extract_filename_from_url(&request.url));

        let download = DownloadInfo {
            id: id.clone(),
            url: request.url.clone(),
            filename: filename.clone(),
            file_size: None,
            downloaded_size: 0,
            progress: ProgressInfo::default(),
            state: DownloadState::Preparing,
            created_at: Utc::now(),
            completed_at: None,
            error: None,
        };

        self.downloads.lock().push(download.clone());

        // Emit event
        if let Some(event_service) = self.event_service.lock().as_ref() {
            event_service.emit_download_added(download);
        }

        // Send command to worker
        if let Some(tx) = self.download_tx.lock().as_ref() {
            let _ = tx.send(DownloadCommand::AddDownload {
                id: id.clone(),
                url: request.url,
                filename: Some(filename),
            });
        }

        Ok(id)
    }

    pub async fn pause_download(&self, id: &str) -> Result<(), DownloadError> {
        if let Some(tx) = self.download_tx.lock().as_ref() {
            let _ = tx.send(DownloadCommand::PauseDownload(id.to_string()));
        }
        Ok(())
    }

    pub async fn resume_download(&self, id: &str) -> Result<(), DownloadError> {
        if let Some(tx) = self.download_tx.lock().as_ref() {
            let _ = tx.send(DownloadCommand::ResumeDownload(id.to_string()));
        }
        Ok(())
    }

    pub async fn cancel_download(&self, id: &str) -> Result<(), DownloadError> {
        if let Some(tx) = self.download_tx.lock().as_ref() {
            let _ = tx.send(DownloadCommand::CancelDownload(id.to_string()));
        }
        Ok(())
    }

    pub async fn remove_download(&self, id: &str) -> Result<(), DownloadError> {
        if let Some(tx) = self.download_tx.lock().as_ref() {
            let _ = tx.send(DownloadCommand::RemoveDownload(id.to_string()));
        }
        Ok(())
    }

    pub async fn get_downloads(&self) -> Result<Vec<DownloadInfo>, DownloadError> {
        Ok(self.downloads.lock().clone())
    }

    pub async fn get_download_stats(&self) -> Result<DownloadStats, DownloadError> {
        let downloads = self.downloads.lock();

        let total_downloads = downloads.len() as u32;
        let active_downloads = downloads
            .iter()
            .filter(|d| matches!(d.state, DownloadState::Active))
            .count() as u32;
        let completed_downloads = downloads
            .iter()
            .filter(|d| matches!(d.state, DownloadState::Completed))
            .count() as u32;
        let failed_downloads = downloads
            .iter()
            .filter(|d| matches!(d.state, DownloadState::Failed))
            .count() as u32;
        let paused_downloads = downloads
            .iter()
            .filter(|d| matches!(d.state, DownloadState::Paused))
            .count() as u32;

        let total_download_speed = downloads
            .iter()
            .filter(|d| matches!(d.state, DownloadState::Active))
            .map(|d| d.progress.download_speed)
            .sum();

        let total_upload_speed = downloads
            .iter()
            .filter(|d| matches!(d.state, DownloadState::Active))
            .map(|d| d.progress.upload_speed)
            .sum();

        let overall_progress = if total_downloads > 0 {
            downloads.iter().map(|d| d.progress.percentage).sum::<f64>() / total_downloads as f64
        } else {
            0.0
        };

        Ok(DownloadStats {
            total_downloads,
            active_downloads,
            completed_downloads,
            failed_downloads,
            paused_downloads,
            total_download_speed,
            total_upload_speed,
            overall_progress,
            eta: None, // TODO: Calculate ETA
        })
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

impl DownloadWorker {
    fn run(self, mut rx: mpsc::UnboundedReceiver<DownloadCommand>) {
        while let Some(cmd) = rx.blocking_recv() {
            match cmd {
                DownloadCommand::AddDownload { id, url, filename } => {
                    self.start_download(id, url, filename);
                }
                DownloadCommand::PauseDownload(id) => {
                    self.pause_download(&id);
                }
                DownloadCommand::ResumeDownload(id) => {
                    self.resume_download(&id);
                }
                DownloadCommand::CancelDownload(id) => {
                    self.cancel_download(&id);
                }
                DownloadCommand::RemoveDownload(id) => {
                    self.remove_download(&id);
                }
            }
        }
    }

    fn start_download(&self, id: String, url: String, filename: Option<String>) {
        // Update state to active
        {
            let mut downloads = self.downloads.lock();
            if let Some(download) = downloads.iter_mut().find(|d| d.id == id) {
                download.state = DownloadState::Active;
            }
        }

        // Emit state change event
        if let Some(event_service) = self.event_service.lock().as_ref() {
            event_service.emit_download_state_change(id.clone(), DownloadState::Active);
        }

        // Start actual download in a separate thread
        let downloads = self.downloads.clone();
        let event_service = self.event_service.clone();
        let id_clone = id.clone();

        thread::spawn(move || {
            Self::perform_download(id_clone, url, filename, downloads, event_service);
        });
    }

    fn perform_download(
        id: String,
        url: String,
        filename: Option<String>,
        downloads: Arc<Mutex<Vec<DownloadInfo>>>,
        event_service: Arc<Mutex<Option<EventService>>>,
    ) {
        use std::fs::File;
        use std::io::Write;

        let filename =
            filename.unwrap_or_else(|| url.split('/').last().unwrap_or("download").to_string());

        // Create downloads directory
        let download_dir =
            dirs::download_dir().unwrap_or_else(|| dirs::home_dir().unwrap_or_default());
        let file_path = download_dir.join(&filename);

        // Perform HTTP download
        match reqwest::blocking::get(&url) {
            Ok(response) => {
                if let Some(total_size) = response.content_length() {
                    // Update file size
                    {
                        let mut downloads = downloads.lock();
                        if let Some(download) = downloads.iter_mut().find(|d| d.id == id) {
                            download.file_size = Some(total_size);
                        }
                    }

                    // Download with progress tracking
                    let mut file = File::create(&file_path).unwrap();
                    let mut downloaded = 0u64;
                    let mut stream = response.bytes().unwrap();

                    for chunk in stream.chunks(8192) {
                        file.write_all(&chunk).unwrap();
                        downloaded += chunk.len() as u64;

                        // Update progress
                        let percentage = (downloaded as f64 / total_size as f64) * 100.0;
                        let speed = 1024 * 1024; // Mock speed for now
                        let eta = if speed > 0 {
                            Some((total_size - downloaded) / speed)
                        } else {
                            None
                        };

                        {
                            let mut downloads = downloads.lock();
                            if let Some(download) = downloads.iter_mut().find(|d| d.id == id) {
                                download.downloaded_size = downloaded;
                                download.progress = ProgressInfo {
                                    percentage,
                                    download_speed: speed,
                                    upload_speed: 0,
                                    eta,
                                    connections: 1,
                                };
                            }
                        }

                        // Emit progress event
                        if let Some(event_service) = event_service.lock().as_ref() {
                            let progress = ProgressInfo {
                                percentage,
                                download_speed: speed,
                                upload_speed: 0,
                                eta,
                                connections: 1,
                            };
                            event_service.emit_download_progress(id.clone(), progress);
                        }

                        // Small delay to make progress visible
                        std::thread::sleep(std::time::Duration::from_millis(50));
                    }

                    // Mark as completed
                    {
                        let mut downloads = downloads.lock();
                        if let Some(download) = downloads.iter_mut().find(|d| d.id == id) {
                            download.state = DownloadState::Completed;
                            download.completed_at = Some(Utc::now());
                        }
                    }

                    // Emit completion event
                    if let Some(event_service) = event_service.lock().as_ref() {
                        event_service.emit_download_state_change(id, DownloadState::Completed);
                    }
                } else {
                    // No content length, download without progress
                    let mut file = File::create(&file_path).unwrap();
                    let mut stream = response.bytes().unwrap();
                    let mut downloaded = 0u64;

                    for chunk in stream.chunks(8192) {
                        file.write_all(&chunk).unwrap();
                        downloaded += chunk.len() as u64;

                        // Update progress
                        {
                            let mut downloads = downloads.lock();
                            if let Some(download) = downloads.iter_mut().find(|d| d.id == id) {
                                download.downloaded_size = downloaded;
                                download.progress.percentage = 50.0; // Unknown total
                                download.progress.download_speed = 1024 * 1024;
                            }
                        }
                    }

                    // Mark as completed
                    {
                        let mut downloads = downloads.lock();
                        if let Some(download) = downloads.iter_mut().find(|d| d.id == id) {
                            download.state = DownloadState::Completed;
                            download.completed_at = Some(Utc::now());
                            download.file_size = Some(downloaded);
                            download.progress.percentage = 100.0;
                        }
                    }

                    // Emit completion event
                    if let Some(event_service) = event_service.lock().as_ref() {
                        event_service.emit_download_state_change(id, DownloadState::Completed);
                    }
                }
            }
            Err(e) => {
                // Mark as failed
                {
                    let mut downloads = downloads.lock();
                    if let Some(download) = downloads.iter_mut().find(|d| d.id == id) {
                        download.state = DownloadState::Failed;
                        download.error = Some(e.to_string());
                    }
                }

                // Emit failure event
                if let Some(event_service) = event_service.lock().as_ref() {
                    event_service.emit_download_state_change(id, DownloadState::Failed);
                }
            }
        }
    }

    fn pause_download(&self, id: &str) {
        let mut downloads = self.downloads.lock();
        if let Some(download) = downloads.iter_mut().find(|d| d.id == id) {
            download.state = DownloadState::Paused;
        }

        if let Some(event_service) = self.event_service.lock().as_ref() {
            event_service.emit_download_state_change(id.to_string(), DownloadState::Paused);
        }
    }

    fn resume_download(&self, id: &str) {
        let mut downloads = self.downloads.lock();
        if let Some(download) = downloads.iter_mut().find(|d| d.id == id) {
            download.state = DownloadState::Active;
        }

        if let Some(event_service) = self.event_service.lock().as_ref() {
            event_service.emit_download_state_change(id.to_string(), DownloadState::Active);
        }
    }

    fn cancel_download(&self, id: &str) {
        let mut downloads = self.downloads.lock();
        if let Some(download) = downloads.iter_mut().find(|d| d.id == id) {
            download.state = DownloadState::Cancelled;
        }

        if let Some(event_service) = self.event_service.lock().as_ref() {
            event_service.emit_download_state_change(id.to_string(), DownloadState::Cancelled);
        }
    }

    fn remove_download(&self, id: &str) {
        let mut downloads = self.downloads.lock();
        downloads.retain(|d| d.id != id);

        if let Some(event_service) = self.event_service.lock().as_ref() {
            event_service.emit_download_removed(id.to_string());
        }
    }
}
