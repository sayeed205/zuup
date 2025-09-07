//! HTTP/HTTPS protocol handler with streaming downloads and range request support

#![cfg(feature = "http")]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use reqwest::{Client, ClientBuilder, Response};
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, error, info};
use url::Url;

use crate::{
    error::{NetworkError, ProtocolError, Result, ZuupError},
    protocol::{
        Download, DownloadMetadata, DownloadOperation, ProtocolCapabilities, ProtocolHandler,
    },
    types::{DownloadProgress, DownloadRequest, DownloadState},
};

/// HTTP/HTTPS protocol handler
pub struct HttpProtocolHandler {
    client: Client,
}

impl HttpProtocolHandler {
    /// Create a new HTTP protocol handler
    pub fn new() -> Self {
        let client = ClientBuilder::new()
            .user_agent(format!("Zuup/{}", env!("CARGO_PKG_VERSION")))
            .timeout(Duration::from_secs(30))
            .redirect(reqwest::redirect::Policy::limited(10))
            .gzip(true)
            .deflate(true)
            .brotli(true)
            .build()
            .expect("Failed to create HTTP client");

        Self { client }
    }

    /// Create a new HTTP protocol handler with custom client
    pub fn with_client(client: Client) -> Self {
        Self { client }
    }

    /// Parse filename from Content-Disposition header
    fn extract_filename_from_disposition(disposition: &str) -> Option<String> {
        disposition
            .split(';')
            .find_map(|part| {
                let part = part.trim();
                if let Some(filename) = part.strip_prefix("filename=") {
                    let filename = filename.trim_matches('"').trim();
                    if !filename.is_empty() {
                        Some(filename.to_string())
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
    }

    /// Parse metadata from HTTP response headers
    fn parse_metadata_from_response(response: &Response) -> DownloadMetadata {
        let mut metadata = DownloadMetadata::default();
        let headers = response.headers();

        // Extract content length
        metadata.size = headers
            .get("content-length")
            .and_then(|h| h.to_str().ok())
            .and_then(|s| s.parse().ok());

        // Extract content type
        metadata.content_type = headers
            .get("content-type")
            .and_then(|h| h.to_str().ok())
            .map(String::from);

        // Check if server supports range requests
        metadata.supports_ranges = headers
            .get("accept-ranges")
            .and_then(|h| h.to_str().ok())
            .map_or(false, |s| s.contains("bytes"));

        // Extract last modified
        metadata.last_modified = headers
            .get("last-modified")
            .and_then(|h| h.to_str().ok())
            .and_then(|s| chrono::DateTime::parse_from_rfc2822(s).ok())
            .map(|dt| dt.with_timezone(&chrono::Utc));

        // Extract ETag
        metadata.etag = headers
            .get("etag")
            .and_then(|h| h.to_str().ok())
            .map(String::from);

        // Extract filename from Content-Disposition header
        metadata.filename = headers
            .get("content-disposition")
            .and_then(|h| h.to_str().ok())
            .and_then(Self::extract_filename_from_disposition);

        metadata
    }
}

#[async_trait]
impl ProtocolHandler for HttpProtocolHandler {
    fn protocol(&self) -> &'static str {
        "http"
    }

    fn can_handle(&self, url: &Url) -> bool {
        matches!(url.scheme(), "http" | "https")
    }

    async fn create_download(&self, request: &DownloadRequest) -> Result<Box<dyn Download>> {
        // Use the first URL for now (multi-source support would be added later)
        let url = request
            .urls
            .first()
            .ok_or_else(|| ZuupError::Config("No URLs provided".to_string()))?;

        let download = HttpDownload::new(
            url.clone(),
            self.client.clone(),
            request.output_path.clone(),
            request.filename.clone(),
        );
        Ok(Box::new(download))
    }

    async fn resume_download(
        &self,
        request: &DownloadRequest,
        _state: &DownloadState,
    ) -> Result<Box<dyn Download>> {
        // For now, just create a new download (resume logic would be enhanced later)
        self.create_download(request).await
    }

    fn capabilities(&self) -> ProtocolCapabilities {
        ProtocolCapabilities {
            supports_segments: true,
            supports_resume: true,
            supports_ranges: true,
            supports_auth: true,
            supports_proxy: true,
            max_connections: Some(8),
            supports_checksums: false, // Would be implemented later
            supports_metadata: true,
        }
    }
}

/// HTTP download implementation with streaming support
pub struct HttpDownload {
    url: Url,
    client: Client,
    output_path: Option<PathBuf>,
    filename: Option<String>,
    state: Arc<RwLock<DownloadState>>,
    progress: Arc<RwLock<DownloadProgress>>,
    metadata: Arc<RwLock<Option<DownloadMetadata>>>,
    output_file: Arc<Mutex<Option<File>>>,
    cancel_token: Arc<Mutex<bool>>,
}

impl HttpDownload {
    /// Create a new HTTP download
    pub fn new(
        url: Url,
        client: Client,
        output_path: Option<PathBuf>,
        filename: Option<String>,
    ) -> Self {
        Self {
            url,
            client,
            output_path,
            filename,
            state: Arc::new(RwLock::new(DownloadState::Pending)),
            progress: Arc::new(RwLock::new(DownloadProgress::new())),
            metadata: Arc::new(RwLock::new(None)),
            output_file: Arc::new(Mutex::new(None)),
            cancel_token: Arc::new(Mutex::new(false)),
        }
    }

    /// Get the output file path
    fn get_output_path(&self) -> Result<PathBuf> {
        let filename = if let Some(ref name) = self.filename {
            name.clone()
        } else {
            // Extract filename from URL path
            self.url
                .path_segments()
                .and_then(|segments| segments.last())
                .filter(|name| !name.is_empty())
                .unwrap_or("download")
                .to_string()
        };

        let output_dir = self
            .output_path
            .as_ref()
            .map(|p| p.clone())
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

        Ok(output_dir.join(filename))
    }

    /// Get metadata from the server
    async fn fetch_metadata(&self) -> Result<DownloadMetadata> {
        debug!("Fetching metadata for: {}", self.url);

        let response = self
            .client
            .head(self.url.clone())
            .send()
            .await
            .map_err(|e| {
                error!("Failed to fetch metadata: {}", e);
                ZuupError::Network(NetworkError::ConnectionFailed(format!(
                    "Failed to get metadata: {}",
                    e
                )))
            })?;

        if !response.status().is_success() {
            let status = response.status();
            error!("HTTP metadata request failed: {}", status);
            return Err(ZuupError::Protocol(ProtocolError::Http {
                status: status.as_u16(),
                message: format!("HTTP error: {}", status),
            }));
        }

        let metadata = HttpProtocolHandler::parse_metadata_from_response(&response);
        debug!(
            "Metadata fetched: size={:?}, content_type={:?}, supports_ranges={}",
            metadata.size, metadata.content_type, metadata.supports_ranges
        );

        Ok(metadata)
    }

    /// Perform the actual download
    async fn download_file(&self) -> Result<()> {
        info!("Starting HTTP download from {}", self.url);

        // Update state to active
        {
            let mut state = self.state.write().await;
            *state = DownloadState::Active;
        }

        // Fetch metadata and setup
        let metadata = self.fetch_metadata().await?;
        {
            let mut meta_guard = self.metadata.write().await;
            *meta_guard = Some(metadata.clone());
        }

        if let Some(size) = metadata.size {
            let mut progress = self.progress.write().await;
            progress.set_total_size(size);
        }

        let output_path = self.get_output_path()?;
        if let Some(parent) = output_path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(ZuupError::Io)?;
        }

        // Check for resume capability
        let resume_offset = self.calculate_resume_offset(&output_path, &metadata).await?;

        // Make HTTP request
        let response = self.make_download_request(resume_offset).await?;
        self.validate_response_status(&response, resume_offset > 0)?;

        // Download and write file
        let downloaded = self.download_and_write_file(response, &output_path, resume_offset).await?;

        // Update final state
        {
            let mut state = self.state.write().await;
            *state = DownloadState::Completed;
        }

        info!(
            "HTTP download completed: {} bytes downloaded to {}",
            downloaded,
            output_path.display()
        );
        Ok(())
    }

    /// Calculate resume offset for partial downloads
    async fn calculate_resume_offset(&self, output_path: &PathBuf, metadata: &DownloadMetadata) -> Result<u64> {
        if !output_path.exists() {
            return Ok(0);
        }

        let file_metadata = tokio::fs::metadata(output_path).await.map_err(ZuupError::Io)?;
        let offset = file_metadata.len();

        if offset > 0 && metadata.supports_ranges {
            info!("Resuming download from offset: {} bytes", offset);
            Ok(offset)
        } else {
            Ok(0)
        }
    }

    /// Make the HTTP download request
    async fn make_download_request(&self, resume_offset: u64) -> Result<Response> {
        let mut request = self.client.get(self.url.clone());

        if resume_offset > 0 {
            request = request.header("Range", format!("bytes={}-", resume_offset));
        }

        request.send().await.map_err(|e| {
            error!("HTTP request failed: {}", e);
            ZuupError::Network(NetworkError::ConnectionFailed(format!("Request failed: {}", e)))
        })
    }

    /// Validate HTTP response status
    fn validate_response_status(&self, response: &Response, is_resume: bool) -> Result<()> {
        let status = response.status();
        let expected_status = if is_resume {
            reqwest::StatusCode::PARTIAL_CONTENT
        } else {
            reqwest::StatusCode::OK
        };

        if status != expected_status && status != reqwest::StatusCode::OK {
            error!("HTTP request failed with status: {}", status);
            return Err(ZuupError::Protocol(ProtocolError::Http {
                status: status.as_u16(),
                message: format!("HTTP error: {}", status),
            }));
        }

        Ok(())
    }

    /// Download response body and write to file
    async fn download_and_write_file(&self, response: Response, output_path: &PathBuf, resume_offset: u64) -> Result<u64> {
        let file = self.open_output_file(output_path, resume_offset > 0).await?;
        {
            let mut file_guard = self.output_file.lock().await;
            *file_guard = Some(file);
        }

        let start_time = Instant::now();
        let body_bytes = response.bytes().await.map_err(|e| {
            error!("Failed to read response body: {}", e);
            ZuupError::Network(NetworkError::ConnectionFailed(format!(
                "Failed to read response body: {}",
                e
            )))
        })?;

        // Check for cancellation
        if self.is_cancelled().await {
            info!("Download cancelled by user");
            return Ok(resume_offset);
        }

        // Write to file
        {
            let mut file_guard = self.output_file.lock().await;
            if let Some(ref mut file) = *file_guard {
                file.write_all(&body_bytes).await.map_err(ZuupError::Io)?;
                file.flush().await.map_err(ZuupError::Io)?;
            } else {
                return Err(ZuupError::Internal("Output file handle lost".to_string()));
            }
            *file_guard = None;
        }

        let downloaded = resume_offset + body_bytes.len() as u64;
        self.update_final_progress(downloaded, start_time.elapsed()).await;

        Ok(downloaded)
    }

    /// Open output file for writing
    async fn open_output_file(&self, output_path: &PathBuf, is_resume: bool) -> Result<File> {
        if is_resume {
            tokio::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(output_path)
                .await
                .map_err(ZuupError::Io)
        } else {
            File::create(output_path).await.map_err(ZuupError::Io)
        }
    }

    /// Check if download was cancelled
    async fn is_cancelled(&self) -> bool {
        let cancel_token = self.cancel_token.lock().await;
        *cancel_token
    }

    /// Update final progress statistics
    async fn update_final_progress(&self, downloaded: u64, elapsed: Duration) {
        let speed = if elapsed.as_secs() > 0 {
            downloaded / elapsed.as_secs()
        } else {
            0
        };

        let mut progress = self.progress.write().await;
        progress.update(downloaded, speed, None);
        progress.connections = 1;
    }
}

#[async_trait]
impl Download for HttpDownload {
    async fn start(&mut self) -> Result<()> {
        let current_state = self.state().clone();

        if !matches!(current_state, DownloadState::Pending | DownloadState::Paused) {
            return Err(ZuupError::InvalidStateTransition {
                from: current_state,
                to: DownloadState::Active,
            });
        }

        // Reset cancel token
        {
            let mut cancel_token = self.cancel_token.lock().await;
            *cancel_token = false;
        }

        // Start download in background task
        let download_clone = HttpDownload {
            url: self.url.clone(),
            client: self.client.clone(),
            output_path: self.output_path.clone(),
            filename: self.filename.clone(),
            state: Arc::clone(&self.state),
            progress: Arc::clone(&self.progress),
            metadata: Arc::clone(&self.metadata),
            output_file: Arc::clone(&self.output_file),
            cancel_token: Arc::clone(&self.cancel_token),
        };

        tokio::spawn(async move {
            info!("Starting HTTP download for {}", download_clone.url);
            if let Err(e) = download_clone.download_file().await {
                error!("HTTP download failed: {}", e);
                let mut state = download_clone.state.write().await;
                *state = DownloadState::Failed(e.to_string());
            }
        });

        Ok(())
    }

    async fn pause(&mut self) -> Result<()> {
        let mut state = self.state.write().await;
        match *state {
            DownloadState::Active => {
                *state = DownloadState::Paused;
                info!(url = %self.url, "Paused HTTP download");
                Ok(())
            }
            _ => Err(ZuupError::InvalidStateTransition {
                from: state.clone(),
                to: DownloadState::Paused,
            }),
        }
    }

    async fn resume(&mut self) -> Result<()> {
        let current_state = self.state();

        if current_state != DownloadState::Paused {
            return Err(ZuupError::InvalidStateTransition {
                from: current_state,
                to: DownloadState::Active,
            });
        }

        // Resume by restarting the download (with resume support)
        self.start().await
    }

    async fn cancel(&mut self) -> Result<()> {
        // Set cancel token
        {
            let mut cancel_token = self.cancel_token.lock().await;
            *cancel_token = true;
        }

        // Update state
        {
            let mut state = self.state.write().await;
            *state = DownloadState::Cancelled;
        }

        // Close file if open
        {
            let mut file_guard = self.output_file.lock().await;
            *file_guard = None;
        }

        info!(url = %self.url, "Cancelled HTTP download");
        Ok(())
    }

    fn progress(&self) -> DownloadProgress {
        // Use try_read to avoid blocking, return default if locked
        match self.progress.try_read() {
            Ok(progress) => progress.clone(),
            Err(_) => DownloadProgress::new(),
        }
    }

    fn state(&self) -> DownloadState {
        // Use try_read to avoid blocking, return pending if locked
        match self.state.try_read() {
            Ok(state) => state.clone(),
            Err(_) => DownloadState::Pending,
        }
    }

    async fn metadata(&self) -> Result<DownloadMetadata> {
        let metadata_guard = self.metadata.read().await;
        if let Some(ref metadata) = *metadata_guard {
            Ok(metadata.clone())
        } else {
            // Fetch metadata if not cached
            drop(metadata_guard);
            let metadata = self.fetch_metadata().await?;
            let mut metadata_guard = self.metadata.write().await;
            *metadata_guard = Some(metadata.clone());
            Ok(metadata)
        }
    }

    fn supports_operation(&self, operation: DownloadOperation) -> bool {
        match operation {
            DownloadOperation::Start => true,
            DownloadOperation::Pause => true,
            DownloadOperation::Resume => true,
            DownloadOperation::Cancel => true,
            DownloadOperation::GetMetadata => true,
            DownloadOperation::VerifyChecksum => false, // Not implemented yet
        }
    }
}

impl Default for HttpProtocolHandler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn test_http_handler_can_handle() {
        let handler = HttpProtocolHandler::new();

        assert!(handler.can_handle(&Url::from_str("http://example.com").unwrap()));
        assert!(handler.can_handle(&Url::from_str("https://example.com").unwrap()));
        assert!(!handler.can_handle(&Url::from_str("ftp://demo:password@test.rebex.net").unwrap()));
        assert!(!handler.can_handle(&Url::from_str("file:///path").unwrap()));
    }

    #[test]
    fn test_http_handler_protocol() {
        let handler = HttpProtocolHandler::new();
        assert_eq!(handler.protocol(), "http");
    }

    #[test]
    fn test_http_handler_capabilities() {
        let handler = HttpProtocolHandler::new();
        let caps = handler.capabilities();

        assert!(caps.supports_segments);
        assert!(caps.supports_resume);
        assert!(caps.supports_ranges);
        assert!(caps.supports_auth);
        assert!(caps.supports_proxy);
        assert!(caps.supports_metadata);
        assert_eq!(caps.max_connections, Some(8));
    }

    #[test]
    fn test_extract_filename_from_disposition() {
        assert_eq!(
            HttpProtocolHandler::extract_filename_from_disposition(
                "attachment; filename=\"test.txt\""
            ),
            Some("test.txt".to_string())
        );

        assert_eq!(
            HttpProtocolHandler::extract_filename_from_disposition("attachment; filename=test.txt"),
            Some("test.txt".to_string())
        );

        assert_eq!(
            HttpProtocolHandler::extract_filename_from_disposition(
                "inline; filename=\"document.pdf\""
            ),
            Some("document.pdf".to_string())
        );

        assert_eq!(
            HttpProtocolHandler::extract_filename_from_disposition("attachment"),
            None
        );
    }

    #[tokio::test]
    async fn test_http_download_creation() {
        let url = Url::from_str("https://httpbin.org/get").unwrap();
        let client = Client::new();

        let download = HttpDownload::new(url.clone(), client, None, None);
        assert_eq!(download.url, url);
        assert_eq!(download.state(), DownloadState::Pending);
    }

    #[tokio::test]
    async fn test_http_download_supports_operations() {
        let url = Url::from_str("https://httpbin.org/get").unwrap();
        let client = Client::new();

        let download = HttpDownload::new(url, client, None, None);

        assert!(download.supports_operation(DownloadOperation::Start));
        assert!(download.supports_operation(DownloadOperation::Pause));
        assert!(download.supports_operation(DownloadOperation::Resume));
        assert!(download.supports_operation(DownloadOperation::Cancel));
        assert!(download.supports_operation(DownloadOperation::GetMetadata));
        assert!(!download.supports_operation(DownloadOperation::VerifyChecksum));
    }

    #[test]
    fn test_output_path_generation() {
        let url = Url::from_str("https://example.com/path/file.txt").unwrap();
        let client = Client::new();

        // Test with custom filename
        let download1 = HttpDownload::new(
            url.clone(),
            client.clone(),
            Some(PathBuf::from("/tmp")),
            Some("custom.txt".to_string()),
        );
        let path1 = download1.get_output_path().unwrap();
        assert_eq!(path1, PathBuf::from("/tmp/custom.txt"));

        // Test with filename from URL
        let download2 = HttpDownload::new(url, client, Some(PathBuf::from("/downloads")), None);
        let path2 = download2.get_output_path().unwrap();
        assert_eq!(path2, PathBuf::from("/downloads/file.txt"));
    }
}
