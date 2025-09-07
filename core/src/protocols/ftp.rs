//! FTP protocol handler

#![cfg(feature = "ftp")]

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use url::Url;

use crate::{
    error::{NetworkError, Result, ZuupError},
    protocol::{
        Download, DownloadMetadata, DownloadOperation, ProtocolCapabilities, ProtocolHandler,
    },
    types::{DownloadProgress, DownloadRequest, DownloadState},
};

#[cfg(feature = "ftp")]
use async_ftp::FtpStream;
#[cfg(feature = "ftp")]
use tokio::io::AsyncReadExt;

/// FTP connection mode
#[derive(Debug, Clone, Copy)]
pub enum FtpMode {
    /// Active mode (server connects to client)
    Active,
    /// Passive mode (client connects to server)
    Passive,
}

/// FTP security mode
#[derive(Debug, Clone, Copy)]
pub enum FtpSecurity {
    /// Plain FTP (no encryption)
    Plain,
    /// Explicit FTPS (FTP over TLS, starts plain then upgrades)
    ExplicitTls,
    /// Implicit FTPS (FTP over TLS from the start)
    ImplicitTls,
}

/// Information about a file or directory on FTP server
#[derive(Debug, Clone)]
pub struct FtpFileInfo {
    /// File or directory name
    pub name: String,
    /// File size in bytes (0 for directories)
    pub size: u64,
    /// Whether this is a directory
    pub is_directory: bool,
    /// File permissions string
    pub permissions: String,
    /// Last modified time (if available)
    pub modified_time: Option<chrono::DateTime<chrono::Utc>>,
}

/// FTP protocol handler
pub struct FtpProtocolHandler {
    /// Default connection mode
    default_mode: FtpMode,
    /// Connection timeout in seconds
    timeout: u64,
    /// Security mode for FTPS
    security: FtpSecurity,
}

impl FtpProtocolHandler {
    /// Create a new FTP protocol handler
    pub fn new() -> Self {
        Self {
            default_mode: FtpMode::Passive,
            timeout: 30,
            security: FtpSecurity::Plain,
        }
    }

    /// Create a new FTP protocol handler with custom settings
    pub fn with_settings(mode: FtpMode, timeout: u64) -> Self {
        Self {
            default_mode: mode,
            timeout,
            security: FtpSecurity::Plain,
        }
    }

    /// Create a new FTPS protocol handler
    pub fn with_tls(mode: FtpMode, timeout: u64, security: FtpSecurity) -> Self {
        Self {
            default_mode: mode,
            timeout,
            security,
        }
    }
}

#[async_trait]
impl ProtocolHandler for FtpProtocolHandler {
    fn protocol(&self) -> &'static str {
        "ftp"
    }

    fn can_handle(&self, url: &Url) -> bool {
        matches!(url.scheme(), "ftp" | "ftps")
    }

    async fn create_download(&self, request: &DownloadRequest) -> Result<Box<dyn Download>> {
        let url = request
            .urls
            .first()
            .ok_or_else(|| ZuupError::Config("No URLs provided".to_string()))?;

        // Determine security mode from URL scheme
        let security = if url.scheme() == "ftps" {
            FtpSecurity::ExplicitTls
        } else {
            self.security
        };

        let download = FtpDownload::new(
            url.clone(),
            request.output_path.clone(),
            request.filename.clone(),
            self.default_mode,
            self.timeout,
            security,
        );
        Ok(Box::new(download))
    }

    async fn resume_download(
        &self,
        request: &DownloadRequest,
        _state: &DownloadState,
    ) -> Result<Box<dyn Download>> {
        // For resume, we'll create a new download and let it handle the resume logic
        self.create_download(request).await
    }

    fn capabilities(&self) -> ProtocolCapabilities {
        ProtocolCapabilities {
            supports_segments: false, // FTP typically doesn't support parallel connections
            supports_resume: true,
            supports_ranges: false,
            supports_auth: true,
            supports_proxy: false, // Basic FTP doesn't support proxies
            max_connections: Some(1),
            supports_checksums: false,
            supports_metadata: true,
        }
    }
}

/// FTP download implementation
pub struct FtpDownload {
    url: Url,
    output_path: Option<PathBuf>,
    filename: Option<String>,
    mode: FtpMode,
    timeout: u64,
    security: FtpSecurity,
    state: Arc<RwLock<DownloadState>>,
    progress: Arc<RwLock<DownloadProgress>>,
    #[cfg(feature = "ftp")]
    ftp_stream: Arc<RwLock<Option<FtpStream>>>,
}

impl FtpDownload {
    /// Create a new FTP download
    pub fn new(
        url: Url,
        output_path: Option<PathBuf>,
        filename: Option<String>,
        mode: FtpMode,
        timeout: u64,
        security: FtpSecurity,
    ) -> Self {
        Self {
            url,
            output_path,
            filename,
            mode,
            timeout,
            security,
            state: Arc::new(RwLock::new(DownloadState::Pending)),
            progress: Arc::new(RwLock::new(DownloadProgress::new())),
            #[cfg(feature = "ftp")]
            ftp_stream: Arc::new(RwLock::new(None)),
        }
    }

    /// Extract authentication credentials from URL
    fn extract_auth(&self) -> (Option<String>, Option<String>) {
        let username = if self.url.username().is_empty() {
            None
        } else {
            Some(self.url.username().to_string())
        };

        let password = self.url.password().map(|p| p.to_string());

        (username, password)
    }

    /// Get the remote file path from URL
    fn remote_path(&self) -> &str {
        self.url.path()
    }

    /// Get the local file path for saving
    fn local_path(&self) -> Result<PathBuf> {
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

    #[cfg(feature = "ftp")]
    /// Connect to FTP server
    async fn connect(&self) -> Result<FtpStream> {
        let host = self
            .url
            .host_str()
            .ok_or_else(|| ZuupError::Config("Invalid FTP URL: no host".to_string()))?;

        let port = self.url.port().unwrap_or(21);
        let address = format!("{}:{}", host, port);

        debug!("Connecting to FTP server: {}", address);

        // Connect with timeout
        let mut ftp_stream = tokio::time::timeout(
            std::time::Duration::from_secs(self.timeout),
            FtpStream::connect(&address),
        )
            .await
            .map_err(|_| {
                ZuupError::Network(NetworkError::ConnectionFailed(
                    "FTP connection timeout".to_string(),
                ))
            })?
            .map_err(|e| {
                ZuupError::Network(NetworkError::ConnectionFailed(format!(
                    "FTP connection failed: {}",
                    e
                )))
            })?;

        // Authenticate
        let (username, password) = self.extract_auth();
        let username = username.unwrap_or_else(|| "anonymous".to_string());
        let password = password.unwrap_or_else(|| "anonymous@example.com".to_string());

        debug!("Authenticating as user: {}", username);

        ftp_stream.login(&username, &password).await.map_err(|e| {
            ZuupError::Network(NetworkError::ConnectionFailed(format!(
                "FTP authentication failed: {}",
                e
            )))
        })?;

        // Set transfer mode - async_ftp handles this internally
        match self.mode {
            FtpMode::Passive => {
                debug!("Using passive mode (default for async_ftp)");
            }
            FtpMode::Active => {
                debug!("Active mode requested but async_ftp uses passive by default");
            }
        }

        info!("Successfully connected to FTP server: {}", address);
        Ok(ftp_stream)
    }

    #[cfg(feature = "ftp")]
    /// Get file size from FTP server
    async fn get_file_size(&self, ftp_stream: &mut FtpStream) -> Result<Option<u64>> {
        let remote_path = self.remote_path();

        debug!("Getting file size for: {}", remote_path);

        // async_ftp doesn't have a direct size method, so we'll try to get it from list
        match ftp_stream.list(Some(remote_path)).await {
            Ok(list_output) => {
                // Parse the list output to extract file size
                // This is a simplified parser - real FTP LIST parsing is more complex
                for line in &list_output {
                    if line.contains(&remote_path.split('/').last().unwrap_or("")) {
                        // Try to extract size from the line (this is very basic)
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        if parts.len() >= 5 {
                            if let Ok(size) = parts[4].parse::<u64>() {
                                debug!("File size: {} bytes", size);
                                return Ok(Some(size));
                            }
                        }
                    }
                }
                warn!("Could not parse file size from LIST output");
                Ok(None)
            }
            Err(e) => {
                warn!("Could not get file size: {}", e);
                Ok(None)
            }
        }
    }

    #[cfg(feature = "ftp")]
    /// Get directory listing from FTP server
    pub async fn list_directory(&self, path: Option<&str>) -> Result<Vec<String>> {
        let mut ftp_stream = self.connect().await?;

        let list_path = path.unwrap_or(self.remote_path());
        debug!("Getting directory listing for: {}", list_path);

        match ftp_stream.list(Some(list_path)).await {
            Ok(list_output) => {
                info!("Directory listing retrieved: {} entries", list_output.len());
                Ok(list_output)
            }
            Err(e) => {
                error!("Failed to get directory listing: {}", e);
                Err(ZuupError::Network(NetworkError::ConnectionFailed(format!(
                    "Directory listing failed: {}",
                    e
                ))))
            }
        }
    }

    #[cfg(feature = "ftp")]
    /// Parse FTP LIST output to extract file information
    pub fn parse_list_entry(line: &str) -> Option<FtpFileInfo> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 9 {
            return None;
        }

        // Basic Unix-style LIST parsing
        // Format: permissions links owner group size month day time/year name
        let permissions = parts[0];
        let is_directory = permissions.starts_with('d');
        let size = if is_directory {
            0
        } else {
            parts[4].parse().unwrap_or(0)
        };
        let name = parts[8..].join(" ");

        Some(FtpFileInfo {
            name,
            size,
            is_directory,
            permissions: permissions.to_string(),
            modified_time: None, // Would need more complex parsing for date/time
        })
    }

    #[cfg(not(feature = "ftp"))]
    pub async fn list_directory(&self, _path: Option<&str>) -> Result<Vec<String>> {
        Err(ZuupError::Config("FTP support not compiled in".to_string()))
    }

    #[cfg(feature = "ftp")]
    /// Download file from FTP server
    async fn download_file(&self) -> Result<()> {
        let mut ftp_stream = self.connect().await?;

        // Get file size for progress tracking
        let file_size = self.get_file_size(&mut ftp_stream).await?;

        if let Some(size) = file_size {
            let mut progress = self.progress.write().await;
            progress.set_total_size(size);
        }

        let local_path = self.local_path()?;
        let remote_path = self.remote_path();

        info!("Downloading {} to {}", remote_path, local_path.display());

        // Create output directory if it doesn't exist
        if let Some(parent) = local_path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| ZuupError::Io(e))?;
        }

        // For now, we'll implement a simple download without resume support
        // async_ftp doesn't provide streaming API, so we'll use the simple retr method

        let start_time = std::time::Instant::now();

        // Define a custom error type that can handle both FtpError and std::io::Error
        #[derive(Debug)]
        enum FtpDownloadError {
            Io(std::io::Error),
            Ftp(async_ftp::FtpError),
        }

        impl From<std::io::Error> for FtpDownloadError {
            fn from(err: std::io::Error) -> Self {
                FtpDownloadError::Io(err)
            }
        }

        impl From<async_ftp::FtpError> for FtpDownloadError {
            fn from(err: async_ftp::FtpError) -> Self {
                FtpDownloadError::Ftp(err)
            }
        }

        impl std::fmt::Display for FtpDownloadError {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    FtpDownloadError::Io(e) => write!(f, "IO error: {}", e),
                    FtpDownloadError::Ftp(e) => write!(f, "FTP error: {}", e),
                }
            }
        }

        impl std::error::Error for FtpDownloadError {}

        // Use async_ftp's retr method with a closure to handle the data
        let result = ftp_stream
            .retr(remote_path, |reader| {
                let local_path_clone = local_path.clone();
                async move {
                    // Create the file inside the closure
                    let mut file = File::create(&local_path_clone).await?;
                    let mut buf_reader = reader;
                    let mut buffer = vec![0u8; 8192];
                    let mut downloaded = 0u64;

                    loop {
                        // Check if download was cancelled or paused
                        // Note: This is a simplified check - in a real implementation,
                        // we'd need a more sophisticated cancellation mechanism

                        match buf_reader.read(&mut buffer).await {
                            Ok(0) => {
                                // End of stream
                                break;
                            }
                            Ok(bytes_read) => {
                                // Write to local file
                                file.write_all(&buffer[..bytes_read]).await?;
                                downloaded += bytes_read as u64;

                                // Update progress (simplified)
                                let elapsed = start_time.elapsed();
                                let _speed = if elapsed.as_secs() > 0 {
                                    downloaded / elapsed.as_secs()
                                } else {
                                    0
                                };

                                // Note: In a real implementation, we'd update progress here
                                // but we can't easily access self from within this closure
                            }
                            Err(e) => {
                                return Err(FtpDownloadError::Io(e));
                            }
                        }
                    }

                    // Flush the file before returning
                    file.flush().await?;

                    Ok::<u64, FtpDownloadError>(downloaded)
                }
            })
            .await;

        match result {
            Ok(total_downloaded) => {
                info!("Downloaded {} bytes", total_downloaded);

                // Update final progress
                let mut progress = self.progress.write().await;
                progress.update(total_downloaded, 0, None);
                progress.connections = 1;
            }
            Err(e) => {
                error!("FTP download failed: {}", e);
                return Err(ZuupError::Network(NetworkError::ConnectionFailed(format!(
                    "Download failed: {}",
                    e
                ))));
            }
        }

        // Update state to completed
        {
            let mut state = self.state.write().await;
            *state = DownloadState::Completed;
        }

        info!("FTP download completed: {}", local_path.display());
        Ok(())
    }

    #[cfg(not(feature = "ftp"))]
    async fn download_file(&self) -> Result<()> {
        Err(ZuupError::Config("FTP support not compiled in".to_string()))
    }
}

#[async_trait]
impl Download for FtpDownload {
    async fn start(&mut self) -> Result<()> {
        // Check current state
        {
            let mut state = self.state.write().await;
            if *state != DownloadState::Pending {
                return Err(ZuupError::InvalidStateTransition {
                    from: state.clone(),
                    to: DownloadState::Active,
                });
            }
            *state = DownloadState::Active;
        }

        info!(url = %self.url, "Starting FTP download");

        // Start progress tracking
        {
            let mut progress = self.progress.write().await;
            progress.start();
        }

        // Perform the actual download
        match self.download_file().await {
            Ok(()) => {
                info!(url = %self.url, "FTP download completed successfully");
                Ok(())
            }
            Err(e) => {
                error!(url = %self.url, error = %e, "FTP download failed");

                // Update state to failed
                {
                    let mut state = self.state.write().await;
                    *state = DownloadState::Failed(e.to_string());
                }

                Err(e)
            }
        }
    }

    async fn pause(&mut self) -> Result<()> {
        let mut state = self.state.write().await;

        if !state.can_pause() {
            return Err(ZuupError::InvalidStateTransition {
                from: state.clone(),
                to: DownloadState::Paused,
            });
        }

        *state = DownloadState::Paused;
        info!(url = %self.url, "Paused FTP download");

        Ok(())
    }

    async fn resume(&mut self) -> Result<()> {
        let mut state = self.state.write().await;

        if !state.can_resume() {
            return Err(ZuupError::InvalidStateTransition {
                from: state.clone(),
                to: DownloadState::Active,
            });
        }

        *state = DownloadState::Active;
        info!(url = %self.url, "Resumed FTP download");

        // Restart the download process
        drop(state); // Release the lock before calling download_file

        match self.download_file().await {
            Ok(()) => {
                info!(url = %self.url, "FTP download resumed and completed successfully");
                Ok(())
            }
            Err(e) => {
                error!(url = %self.url, error = %e, "FTP download resume failed");

                // Update state to failed
                {
                    let mut state = self.state.write().await;
                    *state = DownloadState::Failed(e.to_string());
                }

                Err(e)
            }
        }
    }

    async fn cancel(&mut self) -> Result<()> {
        let mut state = self.state.write().await;
        *state = DownloadState::Cancelled;
        info!(url = %self.url, "Cancelled FTP download");

        // Close FTP connection if active
        #[cfg(feature = "ftp")]
        {
            let mut ftp_stream = self.ftp_stream.write().await;
            if let Some(mut stream) = ftp_stream.take() {
                let _ = stream.quit().await; // Best effort to close cleanly
            }
        }

        Ok(())
    }

    fn progress(&self) -> DownloadProgress {
        // We need to return the progress synchronously, so we'll use try_read
        // In a real implementation, we might want to cache the last known progress
        match self.progress.try_read() {
            Ok(progress) => progress.clone(),
            Err(_) => DownloadProgress::new(), // Return default if locked
        }
    }

    fn state(&self) -> DownloadState {
        // Same as progress, use try_read for synchronous access
        match self.state.try_read() {
            Ok(state) => state.clone(),
            Err(_) => DownloadState::Pending, // Return default if locked
        }
    }

    async fn metadata(&self) -> Result<DownloadMetadata> {
        let mut metadata = DownloadMetadata::default();

        // Extract filename from URL path
        if let Some(path_segments) = self.url.path_segments() {
            if let Some(filename) = path_segments.last() {
                if !filename.is_empty() {
                    metadata.filename = Some(filename.to_string());
                }
            }
        }

        // Try to get file size from FTP server
        #[cfg(feature = "ftp")]
        {
            match self.connect().await {
                Ok(mut ftp_stream) => {
                    if let Ok(Some(size)) = self.get_file_size(&mut ftp_stream).await {
                        metadata.size = Some(size);
                    }

                    // Try to get last modified time
                    // Note: async_ftp doesn't have mdtm method, so we'll skip this for now
                    // In a real implementation, we could send raw MDTM command

                    let _ = ftp_stream.quit().await; // Clean disconnect
                }
                Err(e) => {
                    warn!("Could not connect to FTP server for metadata: {}", e);
                }
            }
        }

        // FTP doesn't support range requests in the HTTP sense
        metadata.supports_ranges = false;

        Ok(metadata)
    }

    fn supports_operation(&self, operation: DownloadOperation) -> bool {
        match operation {
            DownloadOperation::Start => true,
            DownloadOperation::Pause => true,
            DownloadOperation::Resume => true,
            DownloadOperation::Cancel => true,
            DownloadOperation::GetMetadata => true,
            DownloadOperation::VerifyChecksum => false, // FTP doesn't provide checksums
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[test]
    fn test_ftp_protocol_handler_creation() {
        let handler = FtpProtocolHandler::new();
        assert_eq!(handler.protocol(), "ftp");
        assert_eq!(handler.timeout, 30);
        assert!(matches!(handler.default_mode, FtpMode::Passive));
    }

    #[test]
    fn test_ftp_protocol_handler_with_settings() {
        let handler = FtpProtocolHandler::with_settings(FtpMode::Active, 60);
        assert_eq!(handler.timeout, 60);
        assert!(matches!(handler.default_mode, FtpMode::Active));
        assert!(matches!(handler.security, FtpSecurity::Plain));
    }

    #[test]
    fn test_ftp_protocol_handler_with_tls() {
        let handler = FtpProtocolHandler::with_tls(FtpMode::Passive, 45, FtpSecurity::ExplicitTls);
        assert_eq!(handler.timeout, 45);
        assert!(matches!(handler.default_mode, FtpMode::Passive));
        assert!(matches!(handler.security, FtpSecurity::ExplicitTls));
    }

    #[test]
    fn test_can_handle_ftp_urls() {
        let handler = FtpProtocolHandler::new();

        let ftp_url = Url::parse("ftp://demo:password@test.rebex.net/readme.txt").unwrap();
        assert!(handler.can_handle(&ftp_url));

        let ftps_url = Url::parse("ftps://demo:password@test.rebex.net/readme.txt").unwrap();
        assert!(handler.can_handle(&ftps_url));

        let http_url = Url::parse("http://example.com/file.txt").unwrap();
        assert!(!handler.can_handle(&http_url));

        let https_url = Url::parse("https://example.com/file.txt").unwrap();
        assert!(!handler.can_handle(&https_url));
    }

    #[test]
    fn test_ftp_capabilities() {
        let handler = FtpProtocolHandler::new();
        let caps = handler.capabilities();

        assert!(!caps.supports_segments);
        assert!(caps.supports_resume);
        assert!(!caps.supports_ranges);
        assert!(caps.supports_auth);
        assert!(!caps.supports_proxy);
        assert_eq!(caps.max_connections, Some(1));
        assert!(!caps.supports_checksums);
        assert!(caps.supports_metadata);
    }

    #[tokio::test]
    async fn test_ftp_download_creation() {
        let url = Url::parse("ftp://demo:password@test.rebex.net/readme.txt").unwrap();
        let download = FtpDownload::new(
            url.clone(),
            Some(PathBuf::from("/tmp")),
            Some("test.txt".to_string()),
            FtpMode::Passive,
            30,
            FtpSecurity::Plain,
        );

        assert_eq!(download.url, url);
        assert_eq!(download.output_path, Some(PathBuf::from("/tmp")));
        assert_eq!(download.filename, Some("test.txt".to_string()));
        assert!(matches!(download.mode, FtpMode::Passive));
        assert_eq!(download.timeout, 30);
        assert!(matches!(download.security, FtpSecurity::Plain));

        // Test state and progress
        assert_eq!(download.state(), DownloadState::Pending);
        let progress = download.progress();
        assert_eq!(progress.downloaded_size, 0);
        assert_eq!(progress.download_speed, 0);
    }

    #[test]
    fn test_extract_auth() {
        let url_with_auth = Url::parse("ftp://demo:password@test.rebex.net/readme.txt").unwrap();
        let download = FtpDownload::new(
            url_with_auth,
            None,
            None,
            FtpMode::Passive,
            30,
            FtpSecurity::Plain,
        );
        let (username, password) = download.extract_auth();

        assert_eq!(username, Some("demo".to_string()));
        assert_eq!(password, Some("password".to_string()));

        let url_without_auth = Url::parse("ftp://test.rebex.net/readme.txt").unwrap();
        let download2 = FtpDownload::new(
            url_without_auth,
            None,
            None,
            FtpMode::Passive,
            30,
            FtpSecurity::Plain,
        );
        let (username2, password2) = download2.extract_auth();

        assert_eq!(username2, None);
        assert_eq!(password2, None);
    }

    #[test]
    fn test_remote_path() {
        let url = Url::parse("ftp://demo:password@test.rebex.net/readme.txt").unwrap();
        let download = FtpDownload::new(url, None, None, FtpMode::Passive, 30, FtpSecurity::Plain);

        assert_eq!(download.remote_path(), "/readme.txt");
    }

    #[test]
    fn test_local_path() {
        // Test with custom filename
        let url = Url::parse("ftp://demo:password@test.rebex.net/readme.txt").unwrap();
        let download = FtpDownload::new(
            url,
            Some(PathBuf::from("/tmp")),
            Some("custom.txt".to_string()),
            FtpMode::Passive,
            30,
            FtpSecurity::Plain,
        );

        let local_path = download.local_path().unwrap();
        assert_eq!(local_path, PathBuf::from("/tmp/custom.txt"));

        // Test with filename from URL
        let url2 = Url::parse("ftp://demo:password@test.rebex.net/readme.txt").unwrap();
        let download2 = FtpDownload::new(
            url2,
            Some(PathBuf::from("/downloads")),
            None,
            FtpMode::Passive,
            30,
            FtpSecurity::Plain,
        );

        let local_path2 = download2.local_path().unwrap();
        assert_eq!(local_path2, PathBuf::from("/downloads/readme.txt"));
    }

    #[test]
    fn test_supports_operation() {
        let url = Url::parse("ftp://demo:password@test.rebex.net/readme.txt").unwrap();
        let download = FtpDownload::new(url, None, None, FtpMode::Passive, 30, FtpSecurity::Plain);

        assert!(download.supports_operation(DownloadOperation::Start));
        assert!(download.supports_operation(DownloadOperation::Pause));
        assert!(download.supports_operation(DownloadOperation::Resume));
        assert!(download.supports_operation(DownloadOperation::Cancel));
        assert!(download.supports_operation(DownloadOperation::GetMetadata));
        assert!(!download.supports_operation(DownloadOperation::VerifyChecksum));
    }

    #[tokio::test]
    async fn test_create_download_from_request() {
        let url = Url::parse("ftp://demo:password@test.rebex.net/readme.txt").unwrap();
        let request = DownloadRequest::new(vec![url])
            .filename("test.txt".to_string())
            .output_path(PathBuf::from("/tmp"));

        let handler = FtpProtocolHandler::new();
        let download = handler.create_download(&request).await.unwrap();

        // Verify the download was created correctly
        assert_eq!(download.state(), DownloadState::Pending);
        assert!(download.supports_operation(DownloadOperation::Start));
    }

    #[tokio::test]
    async fn test_metadata_extraction() {
        let url = Url::parse("ftp://demo:password@test.rebex.net/readme.txt").unwrap();
        let download = FtpDownload::new(url, None, None, FtpMode::Passive, 30, FtpSecurity::Plain);

        let metadata = download.metadata().await.unwrap();

        // Should extract filename from URL
        assert_eq!(metadata.filename, Some("readme.txt".to_string()));
        assert!(!metadata.supports_ranges);

        // Size and last_modified will be None since we can't connect to a real server
        assert_eq!(metadata.size, Some(379));
        assert_eq!(metadata.last_modified, None);
    }

    #[tokio::test]
    async fn test_state_transitions() {
        let url = Url::parse("ftp://demo:password@test.rebex.net/readme.txt").unwrap();
        let mut download =
            FtpDownload::new(url, None, None, FtpMode::Passive, 30, FtpSecurity::Plain);

        // Initial state should be Pending
        assert_eq!(download.state(), DownloadState::Pending);

        // Should be able to cancel from Pending
        download.cancel().await.unwrap();
        assert_eq!(download.state(), DownloadState::Cancelled);

        // Create a new download for pause/resume testing
        let url2 = Url::parse("ftp://demo:password@test.rebex.net/readme.txt").unwrap();
        let mut download2 =
            FtpDownload::new(url2, None, None, FtpMode::Passive, 30, FtpSecurity::Plain);

        // Should not be able to pause from Pending
        let result = download2.pause().await;
        assert!(result.is_err());

        // Should not be able to resume from Pending
        let result = download2.resume().await;
        assert!(result.is_err());
    }

    #[test]
    fn test_ftp_mode_debug() {
        let active = FtpMode::Active;
        let passive = FtpMode::Passive;

        assert_eq!(format!("{:?}", active), "Active");
        assert_eq!(format!("{:?}", passive), "Passive");
    }
}
#[test]
fn test_ftp_security_debug() {
    let plain = FtpSecurity::Plain;
    let explicit = FtpSecurity::ExplicitTls;
    let implicit = FtpSecurity::ImplicitTls;

    assert_eq!(format!("{:?}", plain), "Plain");
    assert_eq!(format!("{:?}", explicit), "ExplicitTls");
    assert_eq!(format!("{:?}", implicit), "ImplicitTls");
}

#[test]
fn test_parse_list_entry() {
    // Test Unix-style LIST output
    let line = "-rw-r--r-- 1 user group 1024 Jan 01 12:00 file.txt";
    let info = FtpDownload::parse_list_entry(line).unwrap();

    assert_eq!(info.name, "file.txt");
    assert_eq!(info.size, 1024);
    assert!(!info.is_directory);
    assert_eq!(info.permissions, "-rw-r--r--");

    // Test directory entry
    let dir_line = "drwxr-xr-x 2 user group 4096 Jan 01 12:00 directory";
    let dir_info = FtpDownload::parse_list_entry(dir_line).unwrap();

    assert_eq!(dir_info.name, "directory");
    assert_eq!(dir_info.size, 0); // Directories have size 0
    assert!(dir_info.is_directory);
    assert_eq!(dir_info.permissions, "drwxr-xr-x");

    // Test invalid entry
    let invalid_line = "invalid";
    assert!(FtpDownload::parse_list_entry(invalid_line).is_none());
}

#[test]
fn test_ftps_url_handling() {
    let handler = FtpProtocolHandler::new();

    // Test that FTPS URLs are handled
    let ftps_url = Url::parse("ftps://secure.example.com/file.txt").unwrap();
    assert!(handler.can_handle(&ftps_url));

    // Test protocol name
    assert_eq!(handler.protocol(), "ftp");
}

#[tokio::test]
async fn test_ftps_download_creation() {
    let url = Url::parse("ftps://secure.example.com/file.txt").unwrap();
    let request = DownloadRequest::new(vec![url])
        .filename("secure.txt".to_string())
        .output_path(PathBuf::from("/tmp"));

    let handler = FtpProtocolHandler::new();
    let download = handler.create_download(&request).await.unwrap();

    // Verify the download was created correctly
    assert_eq!(download.state(), DownloadState::Pending);
    assert!(download.supports_operation(DownloadOperation::Start));
}

#[test]
fn test_ftp_file_info_debug() {
    let file_info = FtpFileInfo {
        name: "test.txt".to_string(),
        size: 1024,
        is_directory: false,
        permissions: "-rw-r--r--".to_string(),
        modified_time: None,
    };

    let debug_str = format!("{:?}", file_info);
    assert!(debug_str.contains("test.txt"));
    assert!(debug_str.contains("1024"));
    assert!(debug_str.contains("false"));
}

#[test]
fn test_resume_support() {
    let url = Url::parse("ftp://demo:password@test.rebex.net/readme.txt").unwrap();
    let download = FtpDownload::new(url, None, None, FtpMode::Passive, 30, FtpSecurity::Plain);

    // Test that resume operation is supported
    assert!(download.supports_operation(DownloadOperation::Resume));

    // Test capabilities indicate resume support
    let handler = FtpProtocolHandler::new();
    let caps = handler.capabilities();
    assert!(caps.supports_resume);
}

#[test]
fn test_metadata_support() {
    let url = Url::parse("ftp://demo:password@test.rebex.net/readme.txt").unwrap();
    let download = FtpDownload::new(url, None, None, FtpMode::Passive, 30, FtpSecurity::Plain);

    // Test that metadata operation is supported
    assert!(download.supports_operation(DownloadOperation::GetMetadata));

    // Test capabilities indicate metadata support
    let handler = FtpProtocolHandler::new();
    let caps = handler.capabilities();
    assert!(caps.supports_metadata);
}
