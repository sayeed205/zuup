//! SFTP protocol handler

#![cfg(feature = "sftp")]

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

#[cfg(feature = "sftp")]
use ssh2::{Session, Sftp};
#[cfg(feature = "sftp")]
use std::io::{Read, Seek};
#[cfg(feature = "sftp")]
use std::net::TcpStream;

/// SSH authentication method
#[derive(Debug, Clone)]
pub enum SshAuth {
    /// Password authentication
    Password { username: String, password: String },
    /// Public key authentication
    PublicKey {
        username: String,
        public_key_path: PathBuf,
        private_key_path: PathBuf,
        passphrase: Option<String>,
    },
    /// SSH agent authentication
    Agent { username: String },
}

/// SFTP protocol handler
pub struct SftpProtocolHandler {
    /// Connection timeout in seconds
    timeout: u64,
}

impl SftpProtocolHandler {
    /// Create a new SFTP protocol handler
    pub fn new() -> Self {
        Self { timeout: 30 }
    }

    /// Create a new SFTP protocol handler with custom timeout
    pub fn with_timeout(timeout: u64) -> Self {
        Self { timeout }
    }
}

#[async_trait]
impl ProtocolHandler for SftpProtocolHandler {
    fn protocol(&self) -> &'static str {
        "sftp"
    }

    fn can_handle(&self, url: &Url) -> bool {
        url.scheme() == "sftp"
    }

    async fn create_download(&self, request: &DownloadRequest) -> Result<Box<dyn Download>> {
        let url = request
            .urls
            .first()
            .ok_or_else(|| ZuupError::Config("No URLs provided".to_string()))?;

        let download = SftpDownload::new(
            url.clone(),
            request.output_path.clone(),
            request.filename.clone(),
            self.timeout,
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
            supports_segments: false, // SFTP typically doesn't support parallel connections
            supports_resume: true,
            supports_ranges: false,
            supports_auth: true,
            supports_proxy: false,
            max_connections: Some(1),
            supports_checksums: false,
            supports_metadata: true,
        }
    }
}

/// SFTP download implementation
pub struct SftpDownload {
    url: Url,
    output_path: Option<PathBuf>,
    filename: Option<String>,
    timeout: u64,
    state: Arc<RwLock<DownloadState>>,
    progress: Arc<RwLock<DownloadProgress>>,
}

impl SftpDownload {
    /// Create a new SFTP download
    pub fn new(
        url: Url,
        output_path: Option<PathBuf>,
        filename: Option<String>,
        timeout: u64,
    ) -> Self {
        Self {
            url,
            output_path,
            filename,
            timeout,
            state: Arc::new(RwLock::new(DownloadState::Pending)),
            progress: Arc::new(RwLock::new(DownloadProgress::new())),
        }
    }

    /// Extract authentication information from URL
    fn extract_auth(&self) -> SshAuth {
        let username = if self.url.username().is_empty() {
            "root".to_string()
        } else {
            self.url.username().to_string()
        };

        if let Some(password) = self.url.password() {
            SshAuth::Password {
                username,
                password: password.to_string(),
            }
        } else {
            // Default to SSH agent authentication
            SshAuth::Agent { username }
        }
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

    #[cfg(feature = "sftp")]
    /// Connect to SFTP server
    async fn connect(&self) -> Result<(Session, Sftp)> {
        let host = self
            .url
            .host_str()
            .ok_or_else(|| ZuupError::Config("Invalid SFTP URL: no host".to_string()))?;

        let port = self.url.port().unwrap_or(22);
        let address = format!("{}:{}", host, port);

        debug!("Connecting to SFTP server: {}", address);

        // Connect to SSH server
        let tcp = TcpStream::connect(&address).map_err(|e| {
            ZuupError::Network(NetworkError::ConnectionFailed(format!(
                "Failed to connect to {}: {}",
                address, e
            )))
        })?;

        let mut session = Session::new().map_err(|e| {
            ZuupError::Network(NetworkError::ConnectionFailed(format!(
                "Failed to create SSH session: {}",
                e
            )))
        })?;

        session.set_tcp_stream(tcp);
        session.handshake().map_err(|e| {
            ZuupError::Network(NetworkError::ConnectionFailed(format!(
                "SSH handshake failed: {}",
                e
            )))
        })?;

        // Authenticate
        let auth = self.extract_auth();
        match auth {
            SshAuth::Password { username, password } => {
                debug!("Authenticating with password for user: {}", username);
                session
                    .userauth_password(&username, &password)
                    .map_err(|e| {
                        ZuupError::Network(NetworkError::ConnectionFailed(format!(
                            "Password authentication failed: {}",
                            e
                        )))
                    })?;
            }
            SshAuth::PublicKey {
                username,
                public_key_path,
                private_key_path,
                passphrase,
            } => {
                debug!("Authenticating with public key for user: {}", username);
                session
                    .userauth_pubkey_file(
                        &username,
                        Some(&public_key_path),
                        &private_key_path,
                        passphrase.as_deref(),
                    )
                    .map_err(|e| {
                        ZuupError::Network(NetworkError::ConnectionFailed(format!(
                            "Public key authentication failed: {}",
                            e
                        )))
                    })?;
            }
            SshAuth::Agent { username } => {
                debug!("Authenticating with SSH agent for user: {}", username);
                session.userauth_agent(&username).map_err(|e| {
                    ZuupError::Network(NetworkError::ConnectionFailed(format!(
                        "SSH agent authentication failed: {}",
                        e
                    )))
                })?;
            }
        }

        if !session.authenticated() {
            return Err(ZuupError::Network(NetworkError::ConnectionFailed(
                "Authentication failed".to_string(),
            )));
        }

        // Create SFTP channel
        let sftp = session.sftp().map_err(|e| {
            ZuupError::Network(NetworkError::ConnectionFailed(format!(
                "Failed to create SFTP channel: {}",
                e
            )))
        })?;

        info!("Successfully connected to SFTP server: {}", address);
        Ok((session, sftp))
    }

    #[cfg(feature = "sftp")]
    /// Get file metadata from SFTP server
    async fn get_file_metadata(&self, sftp: &Sftp) -> Result<Option<ssh2::FileStat>> {
        let remote_path = self.remote_path();

        debug!("Getting file metadata for: {}", remote_path);

        match sftp.stat(std::path::Path::new(remote_path)) {
            Ok(stat) => {
                debug!("File size: {} bytes", stat.size.unwrap_or(0));
                Ok(Some(stat))
            }
            Err(e) => {
                warn!("Could not get file metadata: {}", e);
                Ok(None)
            }
        }
    }

    #[cfg(feature = "sftp")]
    /// Download file from SFTP server
    async fn download_file(&self) -> Result<()> {
        let (_session, sftp) = self.connect().await?;

        // Get file metadata for progress tracking
        let file_stat = self.get_file_metadata(&sftp).await?;

        if let Some(stat) = &file_stat {
            if let Some(size) = stat.size {
                let mut progress = self.progress.write().await;
                progress.set_total_size(size);
            }
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

        // Check if file exists for resume support
        let resume_offset = if local_path.exists() {
            let metadata = tokio::fs::metadata(&local_path)
                .await
                .map_err(|e| ZuupError::Io(e))?;
            let offset = metadata.len();

            if offset > 0 {
                info!("Resuming download from offset: {} bytes", offset);

                // Update progress
                let mut progress = self.progress.write().await;
                progress.downloaded_size = offset;

                offset
            } else {
                0
            }
        } else {
            0
        };

        // Open remote file
        let mut remote_file = sftp.open(std::path::Path::new(remote_path)).map_err(|e| {
            ZuupError::Network(NetworkError::ConnectionFailed(format!(
                "Failed to open remote file: {}",
                e
            )))
        })?;

        // Seek to resume position if needed
        if resume_offset > 0 {
            remote_file
                .seek(std::io::SeekFrom::Start(resume_offset))
                .map_err(|e| {
                    ZuupError::Network(NetworkError::ConnectionFailed(format!(
                        "Failed to seek to resume position: {}",
                        e
                    )))
                })?;
        }

        // Open local file for writing (append mode for resume)
        let mut local_file = if resume_offset > 0 {
            tokio::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&local_path)
                .await
                .map_err(|e| ZuupError::Io(e))?
        } else {
            File::create(&local_path)
                .await
                .map_err(|e| ZuupError::Io(e))?
        };

        let mut downloaded = resume_offset;
        let mut buffer = vec![0u8; 8192]; // 8KB buffer
        let start_time = std::time::Instant::now();

        loop {
            // Check if download was cancelled or paused
            {
                let state = self.state.read().await;
                match *state {
                    DownloadState::Cancelled => {
                        info!("Download cancelled");
                        return Ok(());
                    }
                    DownloadState::Paused => {
                        info!("Download paused");
                        return Ok(());
                    }
                    _ => {}
                }
            }

            // Read data from remote file
            match remote_file.read(&mut buffer) {
                Ok(0) => {
                    // End of file
                    info!("Download completed successfully");
                    break;
                }
                Ok(bytes_read) => {
                    // Write to local file
                    local_file
                        .write_all(&buffer[..bytes_read])
                        .await
                        .map_err(|e| ZuupError::Io(e))?;

                    downloaded += bytes_read as u64;

                    // Update progress
                    let elapsed = start_time.elapsed();
                    let speed = if elapsed.as_secs() > 0 {
                        downloaded / elapsed.as_secs()
                    } else {
                        0
                    };

                    let mut progress = self.progress.write().await;
                    progress.update(downloaded, speed, None);
                    progress.connections = 1;
                }
                Err(e) => {
                    error!("Error reading from SFTP: {}", e);
                    return Err(ZuupError::Network(NetworkError::ConnectionFailed(format!(
                        "Download failed: {}",
                        e
                    ))));
                }
            }
        }

        local_file.flush().await.map_err(|e| ZuupError::Io(e))?;

        // Update state to completed
        {
            let mut state = self.state.write().await;
            *state = DownloadState::Completed;
        }

        info!("SFTP download completed: {}", local_path.display());
        Ok(())
    }

    #[cfg(not(feature = "sftp"))]
    async fn download_file(&self) -> Result<()> {
        Err(ZuupError::Config(
            "SFTP support not compiled in".to_string(),
        ))
    }
}

#[async_trait]
impl Download for SftpDownload {
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

        info!(url = %self.url, "Starting SFTP download");

        // Start progress tracking
        {
            let mut progress = self.progress.write().await;
            progress.start();
        }

        // Perform the actual download
        match self.download_file().await {
            Ok(()) => {
                info!(url = %self.url, "SFTP download completed successfully");
                Ok(())
            }
            Err(e) => {
                error!(url = %self.url, error = %e, "SFTP download failed");

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
        info!(url = %self.url, "Paused SFTP download");

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
        info!(url = %self.url, "Resumed SFTP download");

        // Restart the download process
        drop(state); // Release the lock before calling download_file

        match self.download_file().await {
            Ok(()) => {
                info!(url = %self.url, "SFTP download resumed and completed successfully");
                Ok(())
            }
            Err(e) => {
                error!(url = %self.url, error = %e, "SFTP download resume failed");

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
        info!(url = %self.url, "Cancelled SFTP download");

        Ok(())
    }

    fn progress(&self) -> DownloadProgress {
        // We need to return the progress synchronously, so we'll use try_read
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

        // Try to get file metadata from SFTP server
        #[cfg(feature = "sftp")]
        {
            match self.connect().await {
                Ok((_session, sftp)) => {
                    if let Ok(Some(stat)) = self.get_file_metadata(&sftp).await {
                        if let Some(size) = stat.size {
                            metadata.size = Some(size);
                        }

                        if let Some(mtime) = stat.mtime {
                            // Convert Unix timestamp to DateTime
                            if let Some(datetime) =
                                chrono::DateTime::from_timestamp(mtime as i64, 0)
                            {
                                metadata.last_modified = Some(datetime);
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!("Could not connect to SFTP server for metadata: {}", e);
                }
            }
        }

        // SFTP doesn't support range requests in the HTTP sense
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
            DownloadOperation::VerifyChecksum => false, // SFTP doesn't provide checksums
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[test]
    fn test_sftp_protocol_handler_creation() {
        let handler = SftpProtocolHandler::new();
        assert_eq!(handler.protocol(), "sftp");
        assert_eq!(handler.timeout, 30);
    }

    #[test]
    fn test_sftp_protocol_handler_with_timeout() {
        let handler = SftpProtocolHandler::with_timeout(60);
        assert_eq!(handler.timeout, 60);
    }

    #[test]
    fn test_can_handle_sftp_urls() {
        let handler = SftpProtocolHandler::new();

        let sftp_url = Url::parse("sftp://example.com/file.txt").unwrap();
        assert!(handler.can_handle(&sftp_url));

        let ftp_url = Url::parse("ftp://example.com/file.txt").unwrap();
        assert!(!handler.can_handle(&ftp_url));

        let http_url = Url::parse("http://example.com/file.txt").unwrap();
        assert!(!handler.can_handle(&http_url));
    }

    #[test]
    fn test_sftp_capabilities() {
        let handler = SftpProtocolHandler::new();
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
    async fn test_sftp_download_creation() {
        let url = Url::parse("sftp://user:pass@example.com/path/file.txt").unwrap();
        let download = SftpDownload::new(
            url.clone(),
            Some(PathBuf::from("/tmp")),
            Some("test.txt".to_string()),
            30,
        );

        assert_eq!(download.url, url);
        assert_eq!(download.output_path, Some(PathBuf::from("/tmp")));
        assert_eq!(download.filename, Some("test.txt".to_string()));
        assert_eq!(download.timeout, 30);

        // Test state and progress
        assert_eq!(download.state(), DownloadState::Pending);
        let progress = download.progress();
        assert_eq!(progress.downloaded_size, 0);
        assert_eq!(progress.download_speed, 0);
    }

    #[test]
    fn test_extract_auth_password() {
        let url_with_auth = Url::parse("sftp://user:pass@example.com/file.txt").unwrap();
        let download = SftpDownload::new(url_with_auth, None, None, 30);
        let auth = download.extract_auth();

        match auth {
            SshAuth::Password { username, password } => {
                assert_eq!(username, "user");
                assert_eq!(password, "pass");
            }
            _ => panic!("Expected password authentication"),
        }
    }

    #[test]
    fn test_extract_auth_agent() {
        let url_without_pass = Url::parse("sftp://user@example.com/file.txt").unwrap();
        let download = SftpDownload::new(url_without_pass, None, None, 30);
        let auth = download.extract_auth();

        match auth {
            SshAuth::Agent { username } => {
                assert_eq!(username, "user");
            }
            _ => panic!("Expected agent authentication"),
        }

        let url_no_user = Url::parse("sftp://example.com/file.txt").unwrap();
        let download2 = SftpDownload::new(url_no_user, None, None, 30);
        let auth2 = download2.extract_auth();

        match auth2 {
            SshAuth::Agent { username } => {
                assert_eq!(username, "root");
            }
            _ => panic!("Expected agent authentication with root user"),
        }
    }

    #[test]
    fn test_remote_path() {
        let url = Url::parse("sftp://example.com/path/to/file.txt").unwrap();
        let download = SftpDownload::new(url, None, None, 30);

        assert_eq!(download.remote_path(), "/path/to/file.txt");
    }

    #[test]
    fn test_local_path() {
        // Test with custom filename
        let url = Url::parse("sftp://example.com/path/file.txt").unwrap();
        let download = SftpDownload::new(
            url,
            Some(PathBuf::from("/tmp")),
            Some("custom.txt".to_string()),
            30,
        );

        let local_path = download.local_path().unwrap();
        assert_eq!(local_path, PathBuf::from("/tmp/custom.txt"));

        // Test with filename from URL
        let url2 = Url::parse("sftp://example.com/path/file.txt").unwrap();
        let download2 = SftpDownload::new(url2, Some(PathBuf::from("/downloads")), None, 30);

        let local_path2 = download2.local_path().unwrap();
        assert_eq!(local_path2, PathBuf::from("/downloads/file.txt"));
    }

    #[test]
    fn test_supports_operation() {
        let url = Url::parse("sftp://example.com/file.txt").unwrap();
        let download = SftpDownload::new(url, None, None, 30);

        assert!(download.supports_operation(DownloadOperation::Start));
        assert!(download.supports_operation(DownloadOperation::Pause));
        assert!(download.supports_operation(DownloadOperation::Resume));
        assert!(download.supports_operation(DownloadOperation::Cancel));
        assert!(download.supports_operation(DownloadOperation::GetMetadata));
        assert!(!download.supports_operation(DownloadOperation::VerifyChecksum));
    }

    #[tokio::test]
    async fn test_create_download_from_request() {
        let url = Url::parse("sftp://example.com/file.txt").unwrap();
        let request = DownloadRequest::new(vec![url])
            .filename("test.txt".to_string())
            .output_path(PathBuf::from("/tmp"));

        let handler = SftpProtocolHandler::new();
        let download = handler.create_download(&request).await.unwrap();

        // Verify the download was created correctly
        assert_eq!(download.state(), DownloadState::Pending);
        assert!(download.supports_operation(DownloadOperation::Start));
    }

    #[tokio::test]
    async fn test_metadata_extraction() {
        let url = Url::parse("sftp://example.com/path/to/file.txt").unwrap();
        let download = SftpDownload::new(url, None, None, 30);

        let metadata = download.metadata().await.unwrap();

        // Should extract filename from URL
        assert_eq!(metadata.filename, Some("file.txt".to_string()));
        assert!(!metadata.supports_ranges);

        // Size and last_modified will be None since we can't connect to a real server
        assert_eq!(metadata.size, None);
        assert_eq!(metadata.last_modified, None);
    }

    #[tokio::test]
    async fn test_state_transitions() {
        let url = Url::parse("sftp://example.com/file.txt").unwrap();
        let mut download = SftpDownload::new(url, None, None, 30);

        // Initial state should be Pending
        assert_eq!(download.state(), DownloadState::Pending);

        // Should be able to cancel from Pending
        download.cancel().await.unwrap();
        assert_eq!(download.state(), DownloadState::Cancelled);

        // Create a new download for pause/resume testing
        let url2 = Url::parse("sftp://example.com/file2.txt").unwrap();
        let mut download2 = SftpDownload::new(url2, None, None, 30);

        // Should not be able to pause from Pending
        let result = download2.pause().await;
        assert!(result.is_err());

        // Should not be able to resume from Pending
        let result = download2.resume().await;
        assert!(result.is_err());
    }

    #[test]
    fn test_ssh_auth_debug() {
        let password_auth = SshAuth::Password {
            username: "user".to_string(),
            password: "pass".to_string(),
        };

        let agent_auth = SshAuth::Agent {
            username: "user".to_string(),
        };

        let pubkey_auth = SshAuth::PublicKey {
            username: "user".to_string(),
            public_key_path: PathBuf::from("/home/user/.ssh/id_rsa.pub"),
            private_key_path: PathBuf::from("/home/user/.ssh/id_rsa"),
            passphrase: None,
        };

        // Just test that debug formatting works
        assert!(format!("{:?}", password_auth).contains("Password"));
        assert!(format!("{:?}", agent_auth).contains("Agent"));
        assert!(format!("{:?}", pubkey_auth).contains("PublicKey"));
    }
}
