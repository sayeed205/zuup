//! HTTP/HTTPS protocol handler with streaming downloads and range request support

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use crete::{
    download::{DownloadRequest, DownloadState},
    error::{NetworkError, ProtocolError, Result, ZuupError},
    protocol::{
        Download, DownloadMetadata, DownloadOperation, ProtocolCapabilities, ProtocolHandler,
    },
    types::DownloadProgress,
};
use futures::StreamExt;
use reqwest::{Client, ClientBuilder, Response};
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, error, info};
use url::Url;

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
        // Simple implementation - would need more robust parsing for production
        for part in disposition.split(';') {
            let part = part.trim();
            if part.starts_with("filename=") {
                let filename = part.strip_prefix("filename=")?;
                let filename = filename.trim_matches('"').trim();
                if !filename.is_empty() {
                    return Some(filename.to_string());
                }
            }
        }
        None
    }

    /// Parse metadata from HTTP response headers
    fn parse_metadata_from_response(response: &Response) -> DownloadMetadata {
        let mut metadata = DownloadMetadata::default();

        // Extract content length
        if let Some(content_length) = response.headers().get("content-length") {
            if let Ok(length_str) = content_length.to_str() {
                if let Ok(size) = length_str.parse::<u64>() {
                    metadata.size = Some(size);
                }
            }
        }

        // Extract content type
        if let Some(content_type) = response.headers().get("content-type") {
            if let Ok(type_str) = content_type.to_str() {
                metadata.content_type = Some(type_str.to_string());
            }
        }

        // Check if server supports range requests
        if let Some(accept_ranges) = response.headers().get("accept-ranges") {
            if let Ok(ranges_str) = accept_ranges.to_str() {
                metadata.supports_ranges = ranges_str.contains("bytes");
            }
        }

        // Extract last modified
        if let Some(last_modified) = response.headers().get("last-modified") {
            if let Ok(modified_str) = last_modified.to_str() {
                if let Ok(datetime) = chrono::DateTime::parse_from_rfc2822(modified_str) {
                    metadata.last_modified = Some(datetime.with_timezone(&chrono::Utc));
                }
            }
        }

        // Extract ETag
        if let Some(etag) = response.headers().get("etag") {
            if let Ok(etag_str) = etag.to_str() {
                metadata.etag = Some(etag_str.to_string());
            }
        }

        // Try to extract filename from Content-Disposition header
        if let Some(disposition) = response.headers().get("content-disposition") {
            if let Ok(disposition_str) = disposition.to_str() {
                if let Some(filename) = Self::extract_filename_from_disposition(disposition_str) {
                    metadata.filename = Some(filename);
                }
            }
        }

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
                ZuupError::Network(NetworkError::ConnectionFailed(format!(
                    "Failed to get metadata: {}",
                    e
                )))
            })?;

        if !response.status().is_success() {
            return Err(ZuupError::Protocol(ProtocolError::Http {
                status: response.status().as_u16(),
                message: format!("HTTP error: {}", response.status()),
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

        // Fetch metadata first
        info!("Fetching metadata for {}", self.url);
        let metadata = self.fetch_metadata().await?;
        info!(
            "Metadata fetch completed: size={:?}, supports_ranges={}",
            metadata.size, metadata.supports_ranges
        );
        {
            let mut meta_guard = self.metadata.write().await;
            *meta_guard = Some(metadata.clone());
        }

        // Update progress with total size
        if let Some(size) = metadata.size {
            let mut progress = self.progress.write().await;
            progress.set_total_size(size);
        }

        // Create output directory if needed
        let output_path = self.get_output_path()?;
        if let Some(parent) = output_path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| ZuupError::Io(e))?;
        }

        // Check for resume capability
        let resume_offset = if output_path.exists() {
            let file_metadata = tokio::fs::metadata(&output_path)
                .await
                .map_err(|e| ZuupError::Io(e))?;
            let offset = file_metadata.len();

            if offset > 0 && metadata.supports_ranges {
                info!("Resuming download from offset: {} bytes", offset);
                offset
            } else {
                0
            }
        } else {
            0
        };

        // Make HTTP request with range header for resume
        let mut request = self.client.get(self.url.clone());

        if resume_offset > 0 {
            request = request.header("Range", format!("bytes={}-", resume_offset));
        }

        info!("Making HTTP request to {}", self.url);
        let response = request.send().await.map_err(|e| {
            error!("HTTP request failed: {}", e);
            ZuupError::Network(NetworkError::ConnectionFailed(format!(
                "Request failed: {}",
                e
            )))
        })?;
        info!("HTTP response received: status={}", response.status());

        // Check response status
        let expected_status = if resume_offset > 0 {
            reqwest::StatusCode::PARTIAL_CONTENT
        } else {
            reqwest::StatusCode::OK
        };

        if response.status() != expected_status && response.status() != reqwest::StatusCode::OK {
            return Err(ZuupError::Protocol(ProtocolError::Http {
                status: response.status().as_u16(),
                message: format!("HTTP error: {}", response.status()),
            }));
        }

        // Open file for writing
        let file = if resume_offset > 0 {
            tokio::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&output_path)
                .await
                .map_err(|e| ZuupError::Io(e))?
        } else {
            File::create(&output_path)
                .await
                .map_err(|e| ZuupError::Io(e))?
        };

        {
            let mut file_guard = self.output_file.lock().await;
            *file_guard = Some(file);
        }

        // Stream the response body
        let mut stream = response.bytes_stream();
        let mut downloaded = resume_offset;
        let start_time = Instant::now();
        let mut last_update = start_time;

        info!(
            "Starting download stream for {} (resume_offset={})",
            output_path.display(),
            resume_offset
        );

        let mut chunk_count = 0;

        while let Some(chunk_result) = stream.next().await {
            chunk_count += 1;
            if chunk_count == 1 {
                info!("Received first chunk from stream");
            }

            // Check if download was cancelled
            {
                let cancel_token = self.cancel_token.lock().await;
                if *cancel_token {
                    info!("Download cancelled by user");
                    return Ok(());
                }
            }

            let chunk = chunk_result.map_err(|e| {
                error!("Stream error: {}", e);
                ZuupError::Network(NetworkError::ConnectionFailed(format!(
                    "Stream error: {}",
                    e
                )))
            })?;

            // Write chunk to file
            {
                let mut file_guard = self.output_file.lock().await;
                if let Some(ref mut file) = *file_guard {
                    file.write_all(&chunk).await.map_err(|e| ZuupError::Io(e))?;
                } else {
                    return Err(ZuupError::Internal(
                        "Output file handle lost during download".to_string(),
                    ));
                }
            }

            downloaded += chunk.len() as u64;

            // Update progress periodically (every 200ms)
            let now = Instant::now();
            if now.duration_since(last_update) >= Duration::from_millis(200) {
                let elapsed = now.duration_since(start_time);
                let speed = if elapsed.as_secs() > 0 {
                    downloaded / elapsed.as_secs()
                } else {
                    0
                };

                {
                    let mut progress = self.progress.write().await;
                    progress.update(downloaded, speed);
                    progress.connections = 1;
                }

                last_update = now;
                info!("Download progress: {} bytes, {} bytes/s", downloaded, speed);
            }

            // Check if download was paused
            {
                let state = self.state.read().await;
                if let DownloadState::Paused = *state {
                    info!("Download paused");
                    return Ok(());
                }
            }
        }

        info!(
            "Download stream completed, received {} chunks, total {} bytes",
            chunk_count, downloaded
        );

        // Flush and close file
        {
            let mut file_guard = self.output_file.lock().await;
            if let Some(ref mut file) = *file_guard {
                file.flush().await.map_err(|e| ZuupError::Io(e))?;
            }
            *file_guard = None;
        }

        // Update state to completed
        {
            let mut state = self.state.write().await;
            *state = DownloadState::Completed;
        }

        // Final progress update
        {
            let elapsed = start_time.elapsed();
            let avg_speed = if elapsed.as_secs() > 0 {
                downloaded / elapsed.as_secs()
            } else {
                0
            };

            let mut progress = self.progress.write().await;
            progress.update(downloaded, avg_speed);
        }

        info!(
            "HTTP download completed: {} bytes downloaded to {}",
            downloaded,
            output_path.display()
        );
        Ok(())
    }
}

#[async_trait]
impl Download for HttpDownload {
    async fn start(&mut self) -> Result<()> {
        let current_state = {
            let state = self.state.read().await;
            state.clone()
        };

        match current_state {
            DownloadState::Pending | DownloadState::Paused => {
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
                    info!(
                        "Background download task started for {}",
                        download_clone.url
                    );
                    if let Err(e) = download_clone.download_file().await {
                        error!("Download failed: {}", e);
                        let mut state = download_clone.state.write().await;
                        *state = DownloadState::Failed(e.to_string());
                    } else {
                        info!(
                            "Background download task completed successfully for {}",
                            download_clone.url
                        );
                    }
                });

                Ok(())
            }
            _ => Err(ZuupError::InvalidStateTransition {
                from: current_state,
                to: DownloadState::Active,
            }),
        }
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
        let current_state = {
            let state = self.state.read().await;
            state.clone()
        };

        match current_state {
            DownloadState::Paused => {
                // Resume by restarting the download (with resume support)
                self.start().await
            }
            _ => Err(ZuupError::InvalidStateTransition {
                from: current_state,
                to: DownloadState::Active,
            }),
        }
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
        assert!(!handler.can_handle(&Url::from_str("ftp://example.com").unwrap()));
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
