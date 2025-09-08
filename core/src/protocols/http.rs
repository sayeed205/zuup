//! HTTP/HTTPS protocol handler with streaming downloads and range request support
//! 
//! This implementation is based on the trauma library approach with improvements
//! for resume functionality using .zuup files and better error handling.

#![cfg(feature = "http")]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures_util::StreamExt;
use percent_encoding;
use reqwest::{
    header::{ACCEPT_RANGES, CONTENT_LENGTH, RANGE},
    Client, ClientBuilder, Response, StatusCode,
};
use tokio::fs::{File, OpenOptions};
use tokio::io::AsyncWriteExt;
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, error, info, warn};
use url::Url;

use crate::{
    error::{NetworkError, ProtocolError, Result, ZuupError},
    protocol::{
        Download, DownloadMetadata, DownloadOperation, ProtocolCapabilities, ProtocolHandler,
    },
    types::{DownloadProgress, DownloadRequest, DownloadState},
};

/// HTTP/HTTPS protocol handler with resume support
pub struct HttpProtocolHandler {
    client: Client,
    /// Number of retries per download
    retries: u32,
    /// Enable resume functionality
    resumable: bool,
}

impl HttpProtocolHandler {
    /// Create a new HTTP protocol handler with optimized settings
    pub fn new() -> Self {
        let client = ClientBuilder::new()
            .user_agent(format!("Zuup/{}", env!("CARGO_PKG_VERSION")))
            .timeout(Duration::from_secs(60))
            .connect_timeout(Duration::from_secs(10))
            .redirect(reqwest::redirect::Policy::limited(10))
            .gzip(true)
            .deflate(true)
            .brotli(true)
            .tcp_keepalive(Duration::from_secs(60))
            .tcp_nodelay(true) // Critical for performance - disable Nagle's algorithm
            .pool_idle_timeout(Duration::from_secs(90))
            .pool_max_idle_per_host(10)
            .build()
            .expect("Failed to create HTTP client");

        Self { 
            client,
            retries: 3,
            resumable: true,
        }
    }

    /// Create a new HTTP protocol handler with custom settings
    pub fn with_config(retries: u32, resumable: bool) -> Self {
        let mut handler = Self::new();
        handler.retries = retries;
        handler.resumable = resumable;
        handler
    }

    /// Create a new HTTP protocol handler with custom client
    pub fn with_client(client: Client) -> Self {
        Self { 
            client,
            retries: 3,
            resumable: true,
        }
    }

    /// Check whether the download is resumable
    async fn is_resumable(&self, url: &Url) -> Result<bool> {
        let response = self.client.head(url.clone()).send().await.map_err(|e| {
            ZuupError::Network(NetworkError::ConnectionFailed(format!(
                "Failed to check resumability: {}",
                e
            )))
        })?;

        let headers = response.headers();
        match headers.get(ACCEPT_RANGES) {
            None => Ok(false),
            Some(x) if x == "none" => Ok(false),
            Some(_) => Ok(true),
        }
    }

    /// Retrieve the content_length of the download
    async fn content_length(&self, url: &Url) -> Result<Option<u64>> {
        let response = self.client.head(url.clone()).send().await.map_err(|e| {
            ZuupError::Network(NetworkError::ConnectionFailed(format!(
                "Failed to get content length: {}",
                e
            )))
        })?;

        let headers = response.headers();
        match headers.get(CONTENT_LENGTH) {
            None => Ok(None),
            Some(header_value) => match header_value.to_str() {
                Ok(v) => match v.parse::<u64>() {
                    Ok(v) => Ok(Some(v)),
                    Err(_) => Ok(None),
                },
                Err(_) => Ok(None),
            },
        }
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
            .get(CONTENT_LENGTH)
            .and_then(|h| h.to_str().ok())
            .and_then(|s| s.parse().ok());

        // Extract content type
        metadata.content_type = headers
            .get("content-type")
            .and_then(|h| h.to_str().ok())
            .map(String::from);

        // Check if server supports range requests
        metadata.supports_ranges = headers
            .get(ACCEPT_RANGES)
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

        let download = HttpDownload::with_config(
            url.clone(),
            self.client.clone(),
            request.output_path.clone(),
            request.filename.clone(),
            self.retries,
            self.resumable,
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

/// HTTP download implementation with resume support
pub struct HttpDownload {
    url: Url,
    client: Client,
    output_path: Option<PathBuf>,
    filename: Option<String>,
    state: Arc<RwLock<DownloadState>>,
    progress: Arc<RwLock<DownloadProgress>>,
    metadata: Arc<RwLock<Option<DownloadMetadata>>>,
    cancel_token: Arc<Mutex<bool>>,
    retries: u32,
    resumable: bool,
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
            cancel_token: Arc::new(Mutex::new(false)),
            retries: 3,
            resumable: true,
        }
    }

    /// Create a new HTTP download with custom settings
    pub fn with_config(
        url: Url,
        client: Client,
        output_path: Option<PathBuf>,
        filename: Option<String>,
        retries: u32,
        resumable: bool,
    ) -> Self {
        Self {
            url,
            client,
            output_path,
            filename,
            state: Arc::new(RwLock::new(DownloadState::Pending)),
            progress: Arc::new(RwLock::new(DownloadProgress::new())),
            metadata: Arc::new(RwLock::new(None)),
            cancel_token: Arc::new(Mutex::new(false)),
            retries,
            resumable,
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
                .map(|s| {
                    // URL decode the filename
                    percent_encoding::percent_decode_str(s)
                        .decode_utf8()
                        .unwrap_or_default()
                        .to_string()
                })
                .unwrap_or_else(|| "download".to_string())
        };

        let output_dir = self
            .output_path
            .as_ref()
            .map(|p| p.clone())
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

        let full_path = output_dir.join(filename);
        debug!("Output path resolved to: {}", full_path.display());
        Ok(full_path)
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

    /// Load resume information from .zuup file (check both formats)
    async fn load_resume_info(&self) -> Result<Option<(u64, Option<u64>)>> {
        if !self.resumable {
            return Ok(None);
        }

        let output_path = self.get_output_path()?;
        debug!("Looking for resume info for output path: {}", output_path.display());
        
        // Check for download manager's .zuup file first (filename.zuup)
        let dm_zuup_path = PathBuf::from(format!("{}.zuup", output_path.display()));
        debug!("Checking download manager .zuup file: {}", dm_zuup_path.display());
        
        if tokio::fs::try_exists(&dm_zuup_path).await.unwrap_or(false) {
            debug!("Found download manager .zuup file: {}", dm_zuup_path.display());
            match tokio::fs::read_to_string(&dm_zuup_path).await {
                Ok(content) => {
                    match serde_json::from_str::<serde_json::Value>(&content) {
                        Ok(data) => {
                            let downloaded = data["progress"]["downloaded_size"]
                                .as_u64()
                                .unwrap_or(0);
                            let total = data["progress"]["total_size"]
                                .as_u64();
                            
                            info!("Loaded resume info from download manager .zuup: {} bytes downloaded", downloaded);
                            return Ok(Some((downloaded, total)));
                        }
                        Err(e) => {
                            debug!("Failed to parse download manager .zuup file: {}", e);
                        }
                    }
                }
                Err(e) => {
                    debug!("Failed to read download manager .zuup file: {}", e);
                }
            }
        } else {
            debug!("Download manager .zuup file does not exist: {}", dm_zuup_path.display());
        }



        debug!("No resume info found");
        Ok(None)
    }



    /// Save resume information to .zuup file (let download manager handle this)
    async fn save_resume_info(&self, _downloaded: u64, _total: Option<u64>) -> Result<()> {
        // The download manager already creates and maintains .zuup files
        // We rely on the actual file size on disk for resume functionality
        // This avoids duplicate .zuup files and ensures consistency
        Ok(())
    }

    /// Remove .zuup files on completion (handled by download manager)
    async fn cleanup_zuup_file(&self) {
        // The download manager handles .zuup file cleanup
        // We don't need to do anything here since we're not creating our own files
    }

    /// Perform the actual download with resume support
    async fn download_file(&self) -> Result<()> {
        info!("Starting HTTP download from {}", self.url);

        // Update state to active
        {
            let mut state = self.state.write().await;
            *state = DownloadState::Active;
        }

        let output_path = self.get_output_path()?;
        if let Some(parent) = output_path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(ZuupError::Io)?;
        }

        // Check for existing resume information
        let mut size_on_disk = 0u64;
        let mut can_resume = false;
        let mut content_length: Option<u64> = None;

        // First check if the file exists and get its actual size
        if output_path.exists() {
            if let Ok(file_metadata) = tokio::fs::metadata(&output_path).await {
                size_on_disk = file_metadata.len();
                info!("Found existing file with {} bytes", size_on_disk);
            }
        }

        // Load resume info from .zuup file if available (for total size info)
        match self.load_resume_info().await {
            Ok(Some((downloaded, total))) => {
                content_length = total;
                info!("Found resume info: {} bytes downloaded (from .zuup), {} bytes on disk", downloaded, size_on_disk);
                
                // Use the actual file size on disk, not the .zuup file data
                // The .zuup file might be stale if the download continued after it was last saved
                if size_on_disk > 0 {
                    info!("Using actual file size {} bytes for resume", size_on_disk);
                } else if downloaded > 0 {
                    // If no file exists but .zuup says we downloaded something, start fresh
                    warn!("Resume info found but file doesn't exist, starting fresh");
                    size_on_disk = 0;
                }
            }
            Ok(None) => {
                if size_on_disk > 0 {
                    info!("Found existing file but no resume info, will attempt resume from {} bytes", size_on_disk);
                } else {
                    debug!("No resume info found, starting fresh download");
                }
            }
            Err(e) => {
                warn!("Failed to load resume info: {}, using file size {} for resume", e, size_on_disk);
            }
        }

        // Check if server supports resuming
        if self.resumable && size_on_disk == 0 {
            match HttpProtocolHandler::new().is_resumable(&self.url).await {
                Ok(resumable) => can_resume = resumable,
                Err(e) => {
                    warn!("Failed to check resumability: {}", e);
                    can_resume = false;
                }
            }
        } else if self.resumable && size_on_disk > 0 {
            can_resume = true; // We have resume data, assume it's supported
        }

        // Get content length if we don't have it
        if content_length.is_none() {
            content_length = HttpProtocolHandler::new().content_length(&self.url).await.unwrap_or(None);
        }

        // Update progress with total size
        if let Some(total) = content_length {
            let mut progress = self.progress.write().await;
            progress.set_total_size(total);
            progress.downloaded_size = size_on_disk;
        }

        // Check if already complete
        if let Some(total) = content_length {
            if total == size_on_disk && size_on_disk > 0 {
                info!("File already fully downloaded");
                let mut state = self.state.write().await;
                *state = DownloadState::Completed;
                self.cleanup_zuup_file().await;
                return Ok(());
            }
        }

        // Perform download with retries
        let mut retry_count = 0;
        let downloaded = loop {
            let initial_size = size_on_disk;
            
            match self.fetch_with_resume(size_on_disk, can_resume, &output_path).await {
                Ok(downloaded) => break downloaded,
                Err(e) => {
                    // Check if this is a pause (not a real error)
                    if e.to_string().contains("Download was paused") {
                        // Download was paused, ensure state is set correctly
                        info!("Download paused, maintaining current state");
                        // The state should already be set to Paused by the pause() method
                        return Ok(());
                    }
                    
                    // Check if we made progress (downloaded more data)
                    let mut made_progress = false;
                    if output_path.exists() {
                        if let Ok(file_metadata) = tokio::fs::metadata(&output_path).await {
                            let new_size = file_metadata.len();
                            if new_size > initial_size {
                                made_progress = true;
                                size_on_disk = new_size;
                                info!("Made progress: {} -> {} bytes, resetting retry count", initial_size, new_size);
                            }
                        }
                    }
                    
                    // Reset retry count if we made progress
                    if made_progress {
                        retry_count = 0;
                    }
                    
                    if retry_count < self.retries {
                        retry_count += 1;
                        error!("Download failed (attempt {}): {}", retry_count, e);
                        
                        tokio::time::sleep(Duration::from_secs(2_u64.pow(retry_count))).await;
                        continue;
                    } else {
                        return Err(e);
                    }
                }
            }
        };

        // Update final state
        {
            let mut state = self.state.write().await;
            *state = DownloadState::Completed;
        }

        // Clean up .zuup file on successful completion
        self.cleanup_zuup_file().await;

        info!(
            "HTTP download completed: {} bytes downloaded to {}",
            downloaded,
            output_path.display()
        );
        Ok(())
    }

    /// Fetch file with resume support
    async fn fetch_with_resume(&self, size_on_disk: u64, can_resume: bool, output_path: &PathBuf) -> Result<u64> {
        debug!("Fetching {} with resume from {} bytes", self.url, size_on_disk);
        
        // Prepare request
        let mut req = self.client.get(self.url.clone());
        if self.resumable && can_resume && size_on_disk > 0 {
            req = req.header(RANGE, format!("bytes={}-", size_on_disk));
        }

        // Send request
        let response = req.send().await.map_err(|e| {
            ZuupError::Network(NetworkError::ConnectionFailed(format!("Request failed: {}", e)))
        })?;

        // Check status
        let status = response.status();
        if !status.is_success() && status != StatusCode::PARTIAL_CONTENT {
            return Err(ZuupError::Protocol(ProtocolError::Http {
                status: status.as_u16(),
                message: format!("HTTP error: {}", status),
            }));
        }

        // Get content length from response
        let content_length = response.headers()
            .get(CONTENT_LENGTH)
            .and_then(|h| h.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok());

        let total_size = if size_on_disk > 0 && can_resume {
            content_length.map(|len| len + size_on_disk).or(Some(size_on_disk))
        } else {
            content_length
        };

        // Open file for writing
        let mut file = if can_resume && size_on_disk > 0 {
            OpenOptions::new()
                .create(true)
                .write(true)
                .append(true)
                .open(output_path)
                .await
                .map_err(ZuupError::Io)?
        } else {
            File::create(output_path).await.map_err(ZuupError::Io)?
        };

        let mut final_size = size_on_disk;
        let mut stream = response.bytes_stream();
        let _last_save = Instant::now();
        let start_time = Instant::now();
        let initial_size = size_on_disk;
        
        // Download chunks
        let mut was_cancelled = false;
        while let Some(chunk_result) = stream.next().await {
            // Check for cancellation
            if *self.cancel_token.lock().await {
                info!("Download cancelled by user");
                was_cancelled = true;
                // Progress is automatically saved by the download manager
                break;
            }

            let chunk = chunk_result.map_err(|e| {
                ZuupError::Network(NetworkError::ConnectionFailed(format!(
                    "Failed to read chunk: {}",
                    e
                )))
            })?;

            let chunk_size = chunk.len() as u64;
            final_size += chunk_size;

            // Write chunk to file
            file.write_all(&chunk).await.map_err(ZuupError::Io)?;

            // Update progress with correct speed calculation
            {
                let mut progress = self.progress.write().await;
                progress.downloaded_size = final_size;
                if let Some(total) = total_size {
                    progress.total_size = Some(total);
                }
                
                // Calculate speed based on new data downloaded, not total
                let elapsed = start_time.elapsed();
                if elapsed.as_secs() > 0 {
                    let new_bytes = final_size - initial_size;
                    progress.download_speed = new_bytes / elapsed.as_secs();
                } else {
                    progress.download_speed = 0;
                }
                progress.connections = 1;
            }

            // Progress is automatically saved by the download manager
            // We don't need to save our own resume info since we rely on file size
        }

        // Final flush
        file.flush().await.map_err(ZuupError::Io)?;

        // Progress is automatically saved by the download manager
        
        // If cancelled, return an error to indicate the download was not completed
        if was_cancelled {
            return Err(ZuupError::Internal("Download was paused".to_string()));
        }
        
        Ok(final_size)
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
            cancel_token: Arc::clone(&self.cancel_token),
            retries: self.retries,
            resumable: self.resumable,
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
                // Set cancel token to stop the download loop
                {
                    let mut cancel_token = self.cancel_token.lock().await;
                    *cancel_token = true;
                }
                
                *state = DownloadState::Paused;
                info!(url = %self.url, "Paused HTTP download");
                
                // Progress is automatically saved by the download manager
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
        let url = Url::from_str("https://example.com/test-file").unwrap();
        let client = Client::new();

        let download = HttpDownload::new(url.clone(), client, None, None);
        assert_eq!(download.url, url);
        assert_eq!(download.state(), DownloadState::Pending);
    }

    #[tokio::test]
    async fn test_http_download_supports_operations() {
        let url = Url::from_str("https://example.com/test-file").unwrap();
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
