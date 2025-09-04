//! Media download integration with yt-dlp

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::process::Command;
use url::Url;

use crate::error::{MediaError, Result, ZuupError};

/// Media download options for yt-dlp integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaDownloadOptions {
    /// Specific format ID to download
    pub format_id: Option<String>,

    /// Quality preference (best, worst, or specific)
    pub quality: Option<String>,

    /// Extract audio only
    pub extract_audio: bool,

    /// Audio format for extraction
    pub audio_format: Option<String>,

    /// Output filename template
    pub output_template: Option<String>,

    /// Download subtitles
    pub download_subtitles: bool,

    /// Subtitle languages to download
    pub subtitle_languages: Vec<String>,

    /// Additional yt-dlp arguments
    pub extra_args: Vec<String>,
}

impl Default for MediaDownloadOptions {
    fn default() -> Self {
        Self {
            format_id: None,
            quality: Some("best".to_string()),
            extract_audio: false,
            audio_format: Some("mp3".to_string()),
            output_template: None,
            download_subtitles: false,
            subtitle_languages: vec!["en".to_string()],
            extra_args: Vec::new(),
        }
    }
}

/// Information about a media resource
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MediaInfo {
    /// Media title
    pub title: String,

    /// Description
    pub description: Option<String>,

    /// Duration in seconds
    pub duration: Option<u64>,

    /// Uploader/channel name
    pub uploader: Option<String>,

    /// Upload date
    pub upload_date: Option<String>,

    /// Thumbnail URL
    pub thumbnail: Option<String>,

    /// Available formats
    pub formats: Vec<MediaFormat>,

    /// Whether this is a playlist
    pub is_playlist: bool,

    /// Playlist entries (if applicable)
    pub playlist_entries: Option<Vec<MediaInfo>>,

    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Format selection preferences
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FormatPreferences {
    /// Type of format to prefer
    pub format_type: FormatType,

    /// Quality preference
    pub quality_preference: QualityPreference,

    /// Preferred file extension
    pub preferred_extension: Option<String>,

    /// Preferred video codec
    pub preferred_video_codec: Option<String>,

    /// Preferred audio codec
    pub preferred_audio_codec: Option<String>,

    /// Maximum file size in bytes
    pub max_file_size: Option<u64>,

    /// Minimum file size in bytes
    pub min_file_size: Option<u64>,
}

impl Default for FormatPreferences {
    fn default() -> Self {
        Self {
            format_type: FormatType::VideoWithAudio,
            quality_preference: QualityPreference::Best,
            preferred_extension: None,
            preferred_video_codec: None,
            preferred_audio_codec: None,
            max_file_size: None,
            min_file_size: None,
        }
    }
}

/// Type of media format
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FormatType {
    /// Video only (no audio)
    VideoOnly,
    /// Audio only (no video)
    AudioOnly,
    /// Video with audio
    VideoWithAudio,
    /// Any format type
    Any,
}

/// Quality preference for format selection
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QualityPreference {
    /// Best available quality
    Best,
    /// Worst available quality
    Worst,
    /// Specific quality target
    Specific(QualityTarget),
}

/// Specific quality target
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityTarget {
    /// Target resolution (e.g., "1920x1080")
    pub resolution: Option<String>,
    /// Target bitrate in kbps
    pub bitrate: Option<f64>,
    /// Target frame rate
    pub fps: Option<f64>,
}

/// Playlist download options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaylistOptions {
    /// Start index for playlist downloads (0-based)
    pub start_index: Option<usize>,

    /// End index for playlist downloads (0-based, inclusive)
    pub end_index: Option<usize>,

    /// Specific indices to download
    pub selected_indices: Vec<usize>,

    /// Minimum duration filter (in seconds)
    pub min_duration: Option<u64>,

    /// Maximum duration filter (in seconds)
    pub max_duration: Option<u64>,

    /// Title filter (entries must contain one of these strings)
    pub title_filter: Vec<String>,

    /// Output path for playlist downloads
    pub output_path: Option<PathBuf>,

    /// Filename template for playlist entries
    pub filename_template: Option<String>,

    /// Whether to create subdirectory for playlist
    pub create_subdirectory: bool,

    /// Subdirectory name template
    pub subdirectory_template: Option<String>,
}

impl Default for PlaylistOptions {
    fn default() -> Self {
        Self {
            start_index: None,
            end_index: None,
            selected_indices: Vec::new(),
            min_duration: None,
            max_duration: None,
            title_filter: Vec::new(),
            output_path: None,
            filename_template: None,
            create_subdirectory: false,
            subdirectory_template: None,
        }
    }
}

/// Format conversion settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormatConversionSettings {
    /// Extract audio only
    pub extract_audio: bool,

    /// Target audio format
    pub audio_format: Option<String>,

    /// Target video format
    pub video_format: Option<String>,

    /// Quality preference for conversion
    pub quality_preference: Option<String>,

    /// Custom conversion arguments
    pub custom_args: Vec<String>,
}

impl Default for FormatConversionSettings {
    fn default() -> Self {
        Self {
            extract_audio: false,
            audio_format: Some("mp3".to_string()),
            video_format: None,
            quality_preference: Some("best".to_string()),
            custom_args: Vec::new(),
        }
    }
}

/// Media download coordination manager
#[derive(Debug)]
pub struct MediaDownloadCoordinator {
    /// yt-dlp manager for media extraction
    ytdlp_manager: Arc<YtDlpManager>,

    /// Default media download options
    default_options: MediaDownloadOptions,

    /// Default playlist options
    default_playlist_options: PlaylistOptions,
}

impl MediaDownloadCoordinator {
    /// Create a new media download coordinator
    pub async fn new() -> Result<Self> {
        let ytdlp_manager = Arc::new(YtDlpManager::new().await?);

        Ok(Self {
            ytdlp_manager,
            default_options: MediaDownloadOptions::default(),
            default_playlist_options: PlaylistOptions::default(),
        })
    }

    /// Create coordinator with custom yt-dlp path
    pub async fn with_ytdlp_path(ytdlp_path: PathBuf) -> Result<Self> {
        let ytdlp_manager = Arc::new(YtDlpManager::with_path(ytdlp_path).await?);

        Ok(Self {
            ytdlp_manager,
            default_options: MediaDownloadOptions::default(),
            default_playlist_options: PlaylistOptions::default(),
        })
    }

    /// Add a media download to the download manager
    pub async fn add_media_download(
        &self,
        url: &str,
        options: Option<MediaDownloadOptions>,
        output_path: Option<PathBuf>,
    ) -> Result<crate::types::DownloadRequest> {
        let options = options.unwrap_or_else(|| self.default_options.clone());
        let request = self
            .ytdlp_manager
            .create_media_download_request(url, options, output_path)
            .await?;

        // Set download type to Media
        // Note: This would be set in the actual download manager when creating DownloadInfo

        Ok(request)
    }

    /// Add a playlist download with selective options
    pub async fn add_playlist_download(
        &self,
        url: &str,
        media_options: Option<MediaDownloadOptions>,
        playlist_options: Option<PlaylistOptions>,
    ) -> Result<Vec<crate::types::DownloadRequest>> {
        let media_options = media_options.unwrap_or_else(|| self.default_options.clone());
        let playlist_options =
            playlist_options.unwrap_or_else(|| self.default_playlist_options.clone());

        self.ytdlp_manager
            .handle_playlist_download(url, media_options, playlist_options)
            .await
    }

    /// Check if URL is supported for media download
    pub async fn is_supported_url(&self, url: &str) -> bool {
        self.ytdlp_manager.is_supported_url(url).await
    }

    /// Get available formats for a URL
    pub async fn get_available_formats(&self, url: &str) -> Result<Vec<MediaFormat>> {
        self.ytdlp_manager.get_formats(url).await
    }

    /// Get best format based on preferences
    pub async fn get_best_format(
        &self,
        url: &str,
        preferences: &FormatPreferences,
    ) -> Result<MediaFormat> {
        self.ytdlp_manager.get_best_format(url, preferences).await
    }

    /// Extract metadata for file organization
    pub async fn extract_metadata(&self, url: &str) -> Result<MediaMetadata> {
        self.ytdlp_manager.extract_metadata(url).await
    }

    /// Set default media download options
    pub fn set_default_options(&mut self, options: MediaDownloadOptions) {
        self.default_options = options;
    }

    /// Set default playlist options
    pub fn set_default_playlist_options(&mut self, options: PlaylistOptions) {
        self.default_playlist_options = options;
    }

    /// Get yt-dlp version
    pub async fn get_ytdlp_version(&self) -> Result<String> {
        self.ytdlp_manager.check_installation().await
    }

    /// Update yt-dlp to latest version
    pub async fn update_ytdlp(&self) -> Result<()> {
        self.ytdlp_manager.update_ytdlp().await
    }

    /// Clean up temporary files
    pub async fn cleanup(&self) -> Result<()> {
        self.ytdlp_manager.cleanup().await
    }

    /// Process media download with format conversion
    pub async fn process_media_download(
        &self,
        url: &str,
        media_options: MediaDownloadOptions,
        conversion_settings: Option<FormatConversionSettings>,
    ) -> Result<crate::types::DownloadRequest> {
        // First, extract media info to determine if it's a playlist
        let media_info = self.ytdlp_manager.extract_info(url).await?;

        if media_info.is_playlist {
            return Err(ZuupError::MediaDownload(MediaError::InvalidMediaFormat(
                "Use add_playlist_download for playlist URLs".to_string(),
            )));
        }

        // Create the download request
        let request = self
            .ytdlp_manager
            .create_media_download_request(url, media_options.clone(), None)
            .await?;

        // Apply format conversion if specified
        if let Some(_conversion) = conversion_settings {
            let _conversion_settings = self
                .ytdlp_manager
                .create_format_conversion_settings(&media_options);
            // Store conversion settings in the request metadata for later use
            // This would be handled by the download manager during actual download
        }

        Ok(request)
    }

    /// Validate media URL and extract basic info
    pub async fn validate_media_url(&self, url: &str) -> Result<MediaValidationResult> {
        match self.ytdlp_manager.extract_info(url).await {
            Ok(info) => Ok(MediaValidationResult {
                is_valid: true,
                is_playlist: info.is_playlist,
                title: Some(info.title),
                duration: info.duration,
                format_count: info.formats.len(),
                error: None,
            }),
            Err(e) => Ok(MediaValidationResult {
                is_valid: false,
                is_playlist: false,
                title: None,
                duration: None,
                format_count: 0,
                error: Some(e.to_string()),
            }),
        }
    }

    /// Get playlist information without downloading
    pub async fn get_playlist_info(&self, url: &str) -> Result<PlaylistInfo> {
        let media_info = self.ytdlp_manager.extract_info(url).await?;

        if !media_info.is_playlist {
            return Err(ZuupError::MediaDownload(MediaError::InvalidMediaFormat(
                "URL is not a playlist".to_string(),
            )));
        }

        let entries = media_info.playlist_entries.unwrap_or_default();
        let total_duration = entries.iter().filter_map(|e| e.duration).sum();

        Ok(PlaylistInfo {
            title: media_info.title,
            entry_count: entries.len(),
            total_duration: Some(total_duration),
            entries: entries
                .into_iter()
                .map(|entry| PlaylistEntry {
                    title: entry.title,
                    duration: entry.duration,
                    uploader: entry.uploader,
                    upload_date: entry.upload_date,
                    thumbnail: entry.thumbnail,
                })
                .collect(),
        })
    }

    /// Create download requests for selected playlist entries
    pub async fn create_playlist_downloads(
        &self,
        url: &str,
        selection: PlaylistSelection,
        media_options: MediaDownloadOptions,
    ) -> Result<Vec<crate::types::DownloadRequest>> {
        let playlist_options = PlaylistOptions {
            start_index: selection.start_index,
            end_index: selection.end_index,
            selected_indices: selection.selected_indices,
            min_duration: selection.min_duration,
            max_duration: selection.max_duration,
            title_filter: selection.title_filter,
            output_path: selection.output_path,
            filename_template: selection.filename_template,
            create_subdirectory: selection.create_subdirectory,
            subdirectory_template: selection.subdirectory_template,
        };

        self.ytdlp_manager
            .handle_playlist_download(url, media_options, playlist_options)
            .await
    }
}

/// Result of media URL validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaValidationResult {
    /// Whether the URL is valid for media extraction
    pub is_valid: bool,

    /// Whether the URL points to a playlist
    pub is_playlist: bool,

    /// Title of the media (if available)
    pub title: Option<String>,

    /// Duration in seconds (if available)
    pub duration: Option<u64>,

    /// Number of available formats
    pub format_count: usize,

    /// Error message if validation failed
    pub error: Option<String>,
}

/// Information about a playlist
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaylistInfo {
    /// Playlist title
    pub title: String,

    /// Number of entries in the playlist
    pub entry_count: usize,

    /// Total duration of all entries (if available)
    pub total_duration: Option<u64>,

    /// List of playlist entries
    pub entries: Vec<PlaylistEntry>,
}

/// Information about a single playlist entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaylistEntry {
    /// Entry title
    pub title: String,

    /// Duration in seconds (if available)
    pub duration: Option<u64>,

    /// Uploader/channel name
    pub uploader: Option<String>,

    /// Upload date
    pub upload_date: Option<String>,

    /// Thumbnail URL
    pub thumbnail: Option<String>,
}

/// Selection criteria for playlist downloads
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaylistSelection {
    /// Start index for playlist downloads (0-based)
    pub start_index: Option<usize>,

    /// End index for playlist downloads (0-based, inclusive)
    pub end_index: Option<usize>,

    /// Specific indices to download
    pub selected_indices: Vec<usize>,

    /// Minimum duration filter (in seconds)
    pub min_duration: Option<u64>,

    /// Maximum duration filter (in seconds)
    pub max_duration: Option<u64>,

    /// Title filter (entries must contain one of these strings)
    pub title_filter: Vec<String>,

    /// Output path for playlist downloads
    pub output_path: Option<PathBuf>,

    /// Filename template for playlist entries
    pub filename_template: Option<String>,

    /// Whether to create subdirectory for playlist
    pub create_subdirectory: bool,

    /// Subdirectory name template
    pub subdirectory_template: Option<String>,
}

impl Default for PlaylistSelection {
    fn default() -> Self {
        Self {
            start_index: None,
            end_index: None,
            selected_indices: Vec::new(),
            min_duration: None,
            max_duration: None,
            title_filter: Vec::new(),
            output_path: None,
            filename_template: None,
            create_subdirectory: false,
            subdirectory_template: None,
        }
    }
}

/// Extended metadata for media files
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MediaMetadata {
    /// Media title
    pub title: String,

    /// Description
    pub description: Option<String>,

    /// Duration in seconds
    pub duration: Option<u64>,

    /// Uploader/channel name
    pub uploader: Option<String>,

    /// Upload date
    pub upload_date: Option<String>,

    /// Thumbnail URL
    pub thumbnail: Option<String>,

    /// Tags associated with the media
    pub tags: Vec<String>,

    /// Categories
    pub categories: Vec<String>,

    /// Language code
    pub language: Option<String>,

    /// View count
    pub view_count: Option<u64>,

    /// Like count
    pub like_count: Option<u64>,

    /// Comment count
    pub comment_count: Option<u64>,

    /// Channel ID
    pub channel_id: Option<String>,

    /// Channel URL
    pub channel_url: Option<String>,

    /// Original webpage URL
    pub webpage_url: String,

    /// Original URL used for extraction
    pub original_url: String,
}

/// Information about a specific media format
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MediaFormat {
    /// Format ID
    pub format_id: String,

    /// File extension
    pub ext: String,

    /// Quality description
    pub quality: Option<String>,

    /// File size in bytes
    pub filesize: Option<u64>,

    /// Video codec
    pub vcodec: Option<String>,

    /// Audio codec
    pub acodec: Option<String>,

    /// Video resolution
    pub resolution: Option<String>,

    /// Frame rate
    pub fps: Option<f64>,

    /// Audio bitrate
    pub abr: Option<f64>,

    /// Video bitrate
    pub vbr: Option<f64>,

    /// Direct download URL
    pub url: String,

    /// Format note/description
    pub format_note: Option<String>,
}

/// Manager for yt-dlp integration
#[derive(Debug)]
pub struct YtDlpManager {
    /// Path to yt-dlp executable
    ytdlp_path: PathBuf,

    /// Temporary directory for operations
    temp_dir: PathBuf,

    /// Default options
    default_options: MediaDownloadOptions,
}

impl YtDlpManager {
    /// Create a new yt-dlp manager
    pub async fn new() -> Result<Self> {
        let ytdlp_path = Self::find_ytdlp_executable().await?;
        let temp_dir = std::env::temp_dir().join("zuup-ytdlp");
        tokio::fs::create_dir_all(&temp_dir).await?;

        Ok(Self {
            ytdlp_path,
            temp_dir,
            default_options: MediaDownloadOptions::default(),
        })
    }

    /// Create a new yt-dlp manager with custom path
    pub async fn with_path(ytdlp_path: PathBuf) -> Result<Self> {
        if !ytdlp_path.exists() {
            return Err(ZuupError::MediaDownload(MediaError::YtDlpNotFound));
        }

        let temp_dir = std::env::temp_dir().join("zuup-ytdlp");
        tokio::fs::create_dir_all(&temp_dir).await?;

        Ok(Self {
            ytdlp_path,
            temp_dir,
            default_options: MediaDownloadOptions::default(),
        })
    }

    /// Find yt-dlp executable in PATH
    async fn find_ytdlp_executable() -> Result<PathBuf> {
        // Try common names for yt-dlp
        let names = ["yt-dlp", "yt-dlp.exe"];

        for name in &names {
            if let Ok(path) = which::which(name) {
                return Ok(path);
            }
        }

        Err(ZuupError::MediaDownload(MediaError::YtDlpNotFound))
    }

    /// Check if yt-dlp is available and get version
    pub async fn check_installation(&self) -> Result<String> {
        let output = Command::new(&self.ytdlp_path)
            .arg("--version")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await?;

        if !output.status.success() {
            let error = String::from_utf8_lossy(&output.stderr);
            return Err(ZuupError::MediaDownload(MediaError::YtDlpExecutionFailed(
                error.to_string(),
            )));
        }

        let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
        Ok(version)
    }

    /// Extract media information from URL
    pub async fn extract_info(&self, url: &str) -> Result<MediaInfo> {
        let output = Command::new(&self.ytdlp_path)
            .args(["--dump-json", "--no-download", "--flat-playlist", url])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await?;

        if !output.status.success() {
            let error = String::from_utf8_lossy(&output.stderr);
            return Err(ZuupError::MediaDownload(MediaError::ExtractionFailed(
                error.to_string(),
            )));
        }

        let json_output = String::from_utf8_lossy(&output.stdout);
        self.parse_media_info(&json_output)
    }

    /// Get available formats for a URL
    pub async fn get_formats(&self, url: &str) -> Result<Vec<MediaFormat>> {
        let info = self.extract_info(url).await?;
        Ok(info.formats)
    }

    /// Get best format based on quality preferences
    pub async fn get_best_format(
        &self,
        url: &str,
        preferences: &FormatPreferences,
    ) -> Result<MediaFormat> {
        let formats = self.get_formats(url).await?;
        self.select_best_format(&formats, preferences)
    }

    /// Select best format from available formats based on preferences
    pub fn select_best_format(
        &self,
        formats: &[MediaFormat],
        preferences: &FormatPreferences,
    ) -> Result<MediaFormat> {
        if formats.is_empty() {
            return Err(ZuupError::MediaDownload(MediaError::FormatNotAvailable(
                "No formats available".to_string(),
            )));
        }

        let mut candidates: Vec<&MediaFormat> = formats.iter().collect();

        // Filter by format type preference
        match preferences.format_type {
            FormatType::VideoOnly => {
                candidates.retain(|f| {
                    f.vcodec.is_some() && f.vcodec.as_ref() != Some(&"none".to_string())
                });
            }
            FormatType::AudioOnly => {
                candidates.retain(|f| {
                    f.acodec.is_some()
                        && f.acodec.as_ref() != Some(&"none".to_string())
                        && (f.vcodec.is_none() || f.vcodec.as_ref() == Some(&"none".to_string()))
                });
            }
            FormatType::VideoWithAudio => {
                candidates.retain(|f| {
                    f.vcodec.is_some()
                        && f.vcodec.as_ref() != Some(&"none".to_string())
                        && f.acodec.is_some()
                        && f.acodec.as_ref() != Some(&"none".to_string())
                });
            }
            FormatType::Any => {} // No filtering
        }

        if candidates.is_empty() {
            return Err(ZuupError::MediaDownload(MediaError::FormatNotAvailable(
                format!(
                    "No formats matching type preference: {:?}",
                    preferences.format_type
                ),
            )));
        }

        // Filter by file extension if specified
        if let Some(ref ext) = preferences.preferred_extension {
            let ext_candidates: Vec<&MediaFormat> = candidates
                .iter()
                .filter(|f| f.ext.eq_ignore_ascii_case(ext))
                .copied()
                .collect();
            if !ext_candidates.is_empty() {
                candidates = ext_candidates;
            }
        }

        // Filter by codec preferences
        if let Some(ref vcodec) = preferences.preferred_video_codec {
            let codec_candidates: Vec<&MediaFormat> = candidates
                .iter()
                .filter(|f| f.vcodec.as_ref().map_or(false, |c| c.contains(vcodec)))
                .copied()
                .collect();
            if !codec_candidates.is_empty() {
                candidates = codec_candidates;
            }
        }

        if let Some(ref acodec) = preferences.preferred_audio_codec {
            let codec_candidates: Vec<&MediaFormat> = candidates
                .iter()
                .filter(|f| f.acodec.as_ref().map_or(false, |c| c.contains(acodec)))
                .copied()
                .collect();
            if !codec_candidates.is_empty() {
                candidates = codec_candidates;
            }
        }

        // Sort by quality preference
        match preferences.quality_preference {
            QualityPreference::Best => {
                candidates.sort_by(|a, b| {
                    // Sort by resolution first (if available), then by bitrate
                    let a_score = self.calculate_quality_score(a);
                    let b_score = self.calculate_quality_score(b);
                    b_score
                        .partial_cmp(&a_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            QualityPreference::Worst => {
                candidates.sort_by(|a, b| {
                    let a_score = self.calculate_quality_score(a);
                    let b_score = self.calculate_quality_score(b);
                    a_score
                        .partial_cmp(&b_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            QualityPreference::Specific(ref target) => {
                candidates.sort_by(|a, b| {
                    let a_diff = self.calculate_quality_difference(a, target);
                    let b_diff = self.calculate_quality_difference(b, target);
                    a_diff
                        .partial_cmp(&b_diff)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }

        // Filter by file size limits
        if let Some(max_size) = preferences.max_file_size {
            candidates.retain(|f| f.filesize.map_or(true, |size| size <= max_size));
        }

        if let Some(min_size) = preferences.min_file_size {
            candidates.retain(|f| f.filesize.map_or(true, |size| size >= min_size));
        }

        candidates.first().map(|f| (*f).clone()).ok_or_else(|| {
            ZuupError::MediaDownload(MediaError::FormatNotAvailable(
                "No formats match the specified preferences".to_string(),
            ))
        })
    }

    /// Calculate quality score for format comparison
    fn calculate_quality_score(&self, format: &MediaFormat) -> f64 {
        let mut score = 0.0;

        // Resolution score (width * height)
        if let Some(ref resolution) = format.resolution {
            if let Some((width, height)) = self.parse_resolution(resolution) {
                score += (width * height) as f64;
            }
        }

        // Bitrate score
        if let Some(vbr) = format.vbr {
            score += vbr * 1000.0; // Weight video bitrate more
        }

        if let Some(abr) = format.abr {
            score += abr * 100.0; // Weight audio bitrate less
        }

        // FPS score
        if let Some(fps) = format.fps {
            score += fps * 10.0;
        }

        // File size as a tiebreaker (larger is generally better quality)
        if let Some(filesize) = format.filesize {
            score += (filesize as f64) / 1_000_000.0; // Convert to MB
        }

        score
    }

    /// Calculate quality difference from target
    fn calculate_quality_difference(&self, format: &MediaFormat, target: &QualityTarget) -> f64 {
        let mut diff = 0.0;

        if let Some(target_resolution) = &target.resolution {
            if let Some(ref format_resolution) = format.resolution {
                if let (Some((target_w, target_h)), Some((format_w, format_h))) = (
                    self.parse_resolution(target_resolution),
                    self.parse_resolution(format_resolution),
                ) {
                    let target_pixels = (target_w * target_h) as f64;
                    let format_pixels = (format_w * format_h) as f64;
                    diff += (target_pixels - format_pixels).abs() / target_pixels;
                }
            }
        }

        if let Some(target_bitrate) = target.bitrate {
            if let Some(format_bitrate) = format.vbr.or(format.abr) {
                diff += (target_bitrate - format_bitrate).abs() / target_bitrate;
            }
        }

        if let Some(target_fps) = target.fps {
            if let Some(format_fps) = format.fps {
                diff += (target_fps - format_fps).abs() / target_fps;
            }
        }

        diff
    }

    /// Parse resolution string (e.g., "1920x1080") into (width, height)
    fn parse_resolution(&self, resolution: &str) -> Option<(u32, u32)> {
        let parts: Vec<&str> = resolution.split('x').collect();
        if parts.len() == 2 {
            if let (Ok(width), Ok(height)) = (parts[0].parse::<u32>(), parts[1].parse::<u32>()) {
                return Some((width, height));
            }
        }
        None
    }

    /// Extract detailed metadata for file naming and organization
    pub async fn extract_metadata(&self, url: &str) -> Result<MediaMetadata> {
        let info = self.extract_info(url).await?;

        Ok(MediaMetadata {
            title: info.title.clone(),
            description: info.description.clone(),
            duration: info.duration,
            uploader: info.uploader.clone(),
            upload_date: info.upload_date.clone(),
            thumbnail: info.thumbnail.clone(),
            tags: self.extract_tags(&info.metadata),
            categories: self.extract_categories(&info.metadata),
            language: self.extract_language(&info.metadata),
            view_count: self.extract_view_count(&info.metadata),
            like_count: self.extract_like_count(&info.metadata),
            comment_count: self.extract_comment_count(&info.metadata),
            channel_id: self.extract_channel_id(&info.metadata),
            channel_url: self.extract_channel_url(&info.metadata),
            webpage_url: url.to_string(),
            original_url: url.to_string(),
        })
    }

    /// Extract tags from metadata
    fn extract_tags(
        &self,
        metadata: &std::collections::HashMap<String, serde_json::Value>,
    ) -> Vec<String> {
        metadata
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Extract categories from metadata
    fn extract_categories(
        &self,
        metadata: &std::collections::HashMap<String, serde_json::Value>,
    ) -> Vec<String> {
        metadata
            .get("categories")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Extract language from metadata
    fn extract_language(
        &self,
        metadata: &std::collections::HashMap<String, serde_json::Value>,
    ) -> Option<String> {
        metadata
            .get("language")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }

    /// Extract view count from metadata
    fn extract_view_count(
        &self,
        metadata: &std::collections::HashMap<String, serde_json::Value>,
    ) -> Option<u64> {
        metadata.get("view_count").and_then(|v| v.as_u64())
    }

    /// Extract like count from metadata
    fn extract_like_count(
        &self,
        metadata: &std::collections::HashMap<String, serde_json::Value>,
    ) -> Option<u64> {
        metadata.get("like_count").and_then(|v| v.as_u64())
    }

    /// Extract comment count from metadata
    fn extract_comment_count(
        &self,
        metadata: &std::collections::HashMap<String, serde_json::Value>,
    ) -> Option<u64> {
        metadata.get("comment_count").and_then(|v| v.as_u64())
    }

    /// Extract channel ID from metadata
    fn extract_channel_id(
        &self,
        metadata: &std::collections::HashMap<String, serde_json::Value>,
    ) -> Option<String> {
        metadata
            .get("channel_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }

    /// Extract channel URL from metadata
    fn extract_channel_url(
        &self,
        metadata: &std::collections::HashMap<String, serde_json::Value>,
    ) -> Option<String> {
        metadata
            .get("channel_url")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }

    /// Generate filename from metadata and template
    pub fn generate_filename(
        &self,
        metadata: &MediaMetadata,
        template: &str,
        format: &MediaFormat,
    ) -> String {
        let mut filename = template.to_string();

        // Replace template variables
        filename = filename.replace("%(title)s", &self.sanitize_filename(&metadata.title));
        filename = filename.replace(
            "%(uploader)s",
            &metadata
                .uploader
                .as_ref()
                .map_or("Unknown".to_string(), |u| self.sanitize_filename(u)),
        );
        filename = filename.replace(
            "%(upload_date)s",
            &metadata
                .upload_date
                .as_ref()
                .unwrap_or(&"Unknown".to_string()),
        );
        filename = filename.replace("%(ext)s", &format.ext);
        filename = filename.replace("%(format_id)s", &format.format_id);

        if let Some(duration) = metadata.duration {
            filename = filename.replace("%(duration)s", &duration.to_string());
        }

        if let Some(view_count) = metadata.view_count {
            filename = filename.replace("%(view_count)s", &view_count.to_string());
        }

        // Add resolution if available
        if let Some(ref resolution) = format.resolution {
            filename = filename.replace("%(resolution)s", resolution);
        }

        // Ensure filename is valid
        self.sanitize_filename(&filename)
    }

    /// Sanitize filename for filesystem compatibility
    fn sanitize_filename(&self, filename: &str) -> String {
        filename
            .chars()
            .map(|c| match c {
                '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
                c if c.is_control() => '_',
                c => c,
            })
            .collect::<String>()
            .trim()
            .to_string()
    }

    /// Download media with specified options
    pub async fn download(&self, url: &str, options: MediaDownloadOptions) -> Result<PathBuf> {
        let mut args = vec![
            "--no-playlist".to_string(),
            "--output".to_string(),
            self.temp_dir
                .join("%(title)s.%(ext)s")
                .to_string_lossy()
                .to_string(),
        ];

        // Add format selection
        if let Some(format_id) = &options.format_id {
            args.extend(["--format".to_string(), format_id.clone()]);
        } else if let Some(quality) = &options.quality {
            args.extend(["--format".to_string(), quality.clone()]);
        }

        // Add audio extraction options
        if options.extract_audio {
            args.push("--extract-audio".to_string());
            if let Some(audio_format) = &options.audio_format {
                args.extend(["--audio-format".to_string(), audio_format.clone()]);
            }
        }

        // Add subtitle options
        if options.download_subtitles {
            args.push("--write-subs".to_string());
            if !options.subtitle_languages.is_empty() {
                args.extend([
                    "--sub-langs".to_string(),
                    options.subtitle_languages.join(","),
                ]);
            }
        }

        // Add custom output template
        if let Some(template) = &options.output_template {
            args.extend(["--output".to_string(), template.clone()]);
        }

        // Add extra arguments
        args.extend(options.extra_args.clone());

        // Add URL
        args.push(url.to_string());

        let output = Command::new(&self.ytdlp_path)
            .args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await?;

        if !output.status.success() {
            let error = String::from_utf8_lossy(&output.stderr);
            return Err(ZuupError::MediaDownload(MediaError::YtDlpExecutionFailed(
                error.to_string(),
            )));
        }

        // Find the downloaded file
        self.find_downloaded_file().await
    }

    /// Parse media info from yt-dlp JSON output
    fn parse_media_info(&self, json_output: &str) -> Result<MediaInfo> {
        // Handle multiple JSON objects (playlist case)
        let lines: Vec<&str> = json_output.lines().collect();

        if lines.len() == 1 {
            // Single video
            let info: serde_json::Value = serde_json::from_str(lines[0])?;
            self.parse_single_media_info(&info)
        } else {
            // Playlist
            let mut entries = Vec::new();
            for line in lines {
                if !line.trim().is_empty() {
                    let info: serde_json::Value = serde_json::from_str(line)?;
                    entries.push(self.parse_single_media_info(&info)?);
                }
            }

            if entries.is_empty() {
                return Err(ZuupError::MediaDownload(MediaError::ExtractionFailed(
                    "No entries found".to_string(),
                )));
            }

            // Use first entry as base, mark as playlist
            let mut base_info = entries[0].clone();
            base_info.is_playlist = true;
            base_info.playlist_entries = Some(entries);

            Ok(base_info)
        }
    }

    /// Parse single media info from JSON
    fn parse_single_media_info(&self, info: &serde_json::Value) -> Result<MediaInfo> {
        let title = info
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown")
            .to_string();

        let description = info
            .get("description")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let duration = info.get("duration").and_then(|v| v.as_u64());

        let uploader = info
            .get("uploader")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let upload_date = info
            .get("upload_date")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let thumbnail = info
            .get("thumbnail")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        // Parse formats
        let formats = if let Some(formats_array) = info.get("formats").and_then(|v| v.as_array()) {
            formats_array
                .iter()
                .filter_map(|format_info| self.parse_media_format(format_info).ok())
                .collect()
        } else {
            Vec::new()
        };

        // Extract additional metadata
        let mut metadata = HashMap::new();
        for (key, value) in info.as_object().unwrap_or(&serde_json::Map::new()) {
            if ![
                "title",
                "description",
                "duration",
                "uploader",
                "upload_date",
                "thumbnail",
                "formats",
            ]
            .contains(&key.as_str())
            {
                metadata.insert(key.clone(), value.clone());
            }
        }

        Ok(MediaInfo {
            title,
            description,
            duration,
            uploader,
            upload_date,
            thumbnail,
            formats,
            is_playlist: false,
            playlist_entries: None,
            metadata,
        })
    }

    /// Parse media format from JSON
    fn parse_media_format(&self, format_info: &serde_json::Value) -> Result<MediaFormat> {
        let format_id = format_info
            .get("format_id")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        let ext = format_info
            .get("ext")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        let quality = format_info
            .get("quality")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let filesize = format_info.get("filesize").and_then(|v| v.as_u64());

        let vcodec = format_info
            .get("vcodec")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let acodec = format_info
            .get("acodec")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let resolution = format_info
            .get("resolution")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let fps = format_info.get("fps").and_then(|v| v.as_f64());

        let abr = format_info.get("abr").and_then(|v| v.as_f64());

        let vbr = format_info.get("vbr").and_then(|v| v.as_f64());

        let url = format_info
            .get("url")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let format_note = format_info
            .get("format_note")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        Ok(MediaFormat {
            format_id,
            ext,
            quality,
            filesize,
            vcodec,
            acodec,
            resolution,
            fps,
            abr,
            vbr,
            url,
            format_note,
        })
    }

    /// Find the downloaded file in temp directory
    async fn find_downloaded_file(&self) -> Result<PathBuf> {
        let mut entries = tokio::fs::read_dir(&self.temp_dir).await?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.is_file() {
                return Ok(path);
            }
        }

        Err(ZuupError::MediaDownload(MediaError::ExtractionFailed(
            "Downloaded file not found".to_string(),
        )))
    }

    /// Set default options
    pub fn set_default_options(&mut self, options: MediaDownloadOptions) {
        self.default_options = options;
    }

    /// Get default options
    pub fn default_options(&self) -> &MediaDownloadOptions {
        &self.default_options
    }

    /// Check if URL is supported by yt-dlp
    pub async fn is_supported_url(&self, url: &str) -> bool {
        // Try to extract info without downloading
        self.extract_info(url).await.is_ok()
    }

    /// Update yt-dlp to latest version
    pub async fn update_ytdlp(&self) -> Result<()> {
        let output = Command::new(&self.ytdlp_path)
            .args(["-U", "--no-check-certificate"])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await?;

        if !output.status.success() {
            let error = String::from_utf8_lossy(&output.stderr);
            return Err(ZuupError::MediaDownload(MediaError::YtDlpExecutionFailed(
                format!("Update failed: {}", error),
            )));
        }

        Ok(())
    }

    /// Clean up temporary files
    pub async fn cleanup(&self) -> Result<()> {
        if self.temp_dir.exists() {
            tokio::fs::remove_dir_all(&self.temp_dir).await?;
        }
        Ok(())
    }

    /// Create a download request for media content
    pub async fn create_media_download_request(
        &self,
        url: &str,
        options: MediaDownloadOptions,
        output_path: Option<PathBuf>,
    ) -> Result<crate::types::DownloadRequest> {
        // Extract media info first
        let media_info = self.extract_info(url).await?;

        // Select best format based on options
        let format = if let Some(format_id) = &options.format_id {
            media_info
                .formats
                .iter()
                .find(|f| f.format_id == *format_id)
                .cloned()
                .ok_or_else(|| {
                    ZuupError::MediaDownload(MediaError::FormatNotAvailable(format_id.clone()))
                })?
        } else {
            let preferences = FormatPreferences {
                format_type: if options.extract_audio {
                    FormatType::AudioOnly
                } else {
                    FormatType::VideoWithAudio
                },
                quality_preference: match options.quality.as_deref() {
                    Some("best") => QualityPreference::Best,
                    Some("worst") => QualityPreference::Worst,
                    _ => QualityPreference::Best,
                },
                preferred_extension: if options.extract_audio {
                    options.audio_format.clone()
                } else {
                    None
                },
                ..Default::default()
            };
            self.select_best_format(&media_info.formats, &preferences)?
        };

        // Generate filename from template or use default
        let metadata = MediaMetadata {
            title: media_info.title.clone(),
            description: media_info.description.clone(),
            duration: media_info.duration,
            uploader: media_info.uploader.clone(),
            upload_date: media_info.upload_date.clone(),
            thumbnail: media_info.thumbnail.clone(),
            tags: vec![],
            categories: vec![],
            language: None,
            view_count: None,
            like_count: None,
            comment_count: None,
            channel_id: None,
            channel_url: None,
            webpage_url: url.to_string(),
            original_url: url.to_string(),
        };

        let template = options
            .output_template
            .as_deref()
            .unwrap_or("%(title)s.%(ext)s");
        let filename = self.generate_filename(&metadata, template, &format);

        // Create download request
        let download_url = url::Url::parse(&format.url)?;
        let mut request = crate::types::DownloadRequest::new(download_url).with_filename(filename);

        if let Some(path) = output_path {
            request = request.with_output_path(path);
        }

        Ok(request)
    }

    /// Handle playlist downloads with selective downloading
    pub async fn handle_playlist_download(
        &self,
        url: &str,
        options: MediaDownloadOptions,
        playlist_options: PlaylistOptions,
    ) -> Result<Vec<crate::types::DownloadRequest>> {
        let media_info = self.extract_info(url).await?;

        if !media_info.is_playlist {
            // Single video, return as single-item list
            let request = self
                .create_media_download_request(url, options, playlist_options.output_path)
                .await?;
            return Ok(vec![request]);
        }

        let playlist_entries = media_info.playlist_entries.ok_or_else(|| {
            ZuupError::MediaDownload(MediaError::PlaylistFailed(
                "No playlist entries found".to_string(),
            ))
        })?;

        let mut requests = Vec::new();

        for (index, entry) in playlist_entries.iter().enumerate() {
            // Check if this entry should be downloaded based on selection criteria
            if !self.should_download_playlist_entry(index, entry, &playlist_options) {
                continue;
            }

            // Create individual download request for this entry
            if let Some(best_format) = entry.formats.first() {
                let entry_url = url::Url::parse(&best_format.url)?;

                // Generate filename for playlist entry
                let entry_metadata = MediaMetadata {
                    title: entry.title.clone(),
                    description: entry.description.clone(),
                    duration: entry.duration,
                    uploader: entry.uploader.clone(),
                    upload_date: entry.upload_date.clone(),
                    thumbnail: entry.thumbnail.clone(),
                    tags: vec![],
                    categories: vec![],
                    language: None,
                    view_count: None,
                    like_count: None,
                    comment_count: None,
                    channel_id: None,
                    channel_url: None,
                    webpage_url: format!("{}#{}", url, index),
                    original_url: url.to_string(),
                };

                let template = playlist_options
                    .filename_template
                    .as_deref()
                    .unwrap_or("%(playlist_index)s - %(title)s.%(ext)s");
                let mut filename = self.generate_filename(&entry_metadata, template, best_format);

                // Add playlist index to filename
                filename = filename.replace("%(playlist_index)s", &format!("{:03}", index + 1));

                let mut request =
                    crate::types::DownloadRequest::new(entry_url).with_filename(filename);

                if let Some(ref path) = playlist_options.output_path {
                    request = request.with_output_path(path.clone());
                }

                requests.push(request);
            }
        }

        if requests.is_empty() {
            return Err(ZuupError::MediaDownload(MediaError::PlaylistFailed(
                "No entries selected for download".to_string(),
            )));
        }

        Ok(requests)
    }

    /// Check if a playlist entry should be downloaded based on selection criteria
    fn should_download_playlist_entry(
        &self,
        index: usize,
        entry: &MediaInfo,
        options: &PlaylistOptions,
    ) -> bool {
        // Check index range
        if let Some(start) = options.start_index {
            if index < start {
                return false;
            }
        }

        if let Some(end) = options.end_index {
            if index > end {
                return false;
            }
        }

        // Check specific indices
        if !options.selected_indices.is_empty() {
            if !options.selected_indices.contains(&index) {
                return false;
            }
        }

        // Check duration filter
        if let Some(min_duration) = options.min_duration {
            if let Some(duration) = entry.duration {
                if duration < min_duration {
                    return false;
                }
            }
        }

        if let Some(max_duration) = options.max_duration {
            if let Some(duration) = entry.duration {
                if duration > max_duration {
                    return false;
                }
            }
        }

        // Check title filter
        if !options.title_filter.is_empty() {
            let title_lower = entry.title.to_lowercase();
            let matches = options
                .title_filter
                .iter()
                .any(|filter| title_lower.contains(&filter.to_lowercase()));
            if !matches {
                return false;
            }
        }

        true
    }

    /// Convert media download options to format conversion settings
    pub fn create_format_conversion_settings(
        &self,
        options: &MediaDownloadOptions,
    ) -> FormatConversionSettings {
        FormatConversionSettings {
            extract_audio: options.extract_audio,
            audio_format: options.audio_format.clone(),
            video_format: None, // Will be determined by format selection
            quality_preference: options.quality.clone(),
            custom_args: options.extra_args.clone(),
        }
    }

    /// Apply format conversion during download
    pub async fn apply_format_conversion(
        &self,
        input_path: &PathBuf,
        settings: &FormatConversionSettings,
    ) -> Result<PathBuf> {
        if !settings.extract_audio && settings.video_format.is_none() {
            // No conversion needed
            return Ok(input_path.clone());
        }

        let output_path = if settings.extract_audio {
            let audio_ext = settings.audio_format.as_deref().unwrap_or("mp3");
            input_path.with_extension(audio_ext)
        } else if let Some(ref video_format) = settings.video_format {
            input_path.with_extension(video_format)
        } else {
            input_path.clone()
        };

        let mut args = vec![
            "--no-playlist".to_string(),
            "--output".to_string(),
            output_path.to_string_lossy().to_string(),
        ];

        if settings.extract_audio {
            args.push("--extract-audio".to_string());
            if let Some(ref audio_format) = settings.audio_format {
                args.extend(["--audio-format".to_string(), audio_format.clone()]);
            }
        }

        if let Some(ref quality) = settings.quality_preference {
            args.extend(["--format".to_string(), quality.clone()]);
        }

        args.extend(settings.custom_args.clone());
        args.push(input_path.to_string_lossy().to_string());

        let output = tokio::process::Command::new(&self.ytdlp_path)
            .args(&args)
            .output()
            .await?;

        if !output.status.success() {
            let error = String::from_utf8_lossy(&output.stderr);
            return Err(ZuupError::MediaDownload(MediaError::YtDlpExecutionFailed(
                error.to_string(),
            )));
        }

        Ok(output_path)
    }
}

/// Trait for media extraction services
#[async_trait]
pub trait MediaExtractor: Send + Sync {
    /// Extract media information from URL
    async fn extract_info(&self, url: &Url) -> Result<MediaInfo>;

    /// Get available formats for URL
    async fn get_formats(&self, url: &Url) -> Result<Vec<MediaFormat>>;

    /// Get best format based on preferences
    async fn get_best_format(
        &self,
        url: &Url,
        preferences: &FormatPreferences,
    ) -> Result<MediaFormat>;

    /// Extract detailed metadata for file naming and organization
    async fn extract_metadata(&self, url: &Url) -> Result<MediaMetadata>;

    /// Check if URL is supported
    async fn supports_url(&self, url: &Url) -> bool;

    /// Download media with options
    async fn download(&self, url: &Url, options: MediaDownloadOptions) -> Result<PathBuf>;

    /// Generate filename from metadata and template
    fn generate_filename(
        &self,
        metadata: &MediaMetadata,
        template: &str,
        format: &MediaFormat,
    ) -> String;
}

#[async_trait]
impl MediaExtractor for YtDlpManager {
    async fn extract_info(&self, url: &Url) -> Result<MediaInfo> {
        self.extract_info(url.as_str()).await
    }

    async fn get_formats(&self, url: &Url) -> Result<Vec<MediaFormat>> {
        self.get_formats(url.as_str()).await
    }

    async fn get_best_format(
        &self,
        url: &Url,
        preferences: &FormatPreferences,
    ) -> Result<MediaFormat> {
        self.get_best_format(url.as_str(), preferences).await
    }

    async fn extract_metadata(&self, url: &Url) -> Result<MediaMetadata> {
        self.extract_metadata(url.as_str()).await
    }

    async fn supports_url(&self, url: &Url) -> bool {
        self.is_supported_url(url.as_str()).await
    }

    async fn download(&self, url: &Url, options: MediaDownloadOptions) -> Result<PathBuf> {
        self.download(url.as_str(), options).await
    }

    fn generate_filename(
        &self,
        metadata: &MediaMetadata,
        template: &str,
        format: &MediaFormat,
    ) -> String {
        self.generate_filename(metadata, template, format)
    }
}
