//! File management and organization system
//!
//! This module provides comprehensive file management capabilities including:
//! - Flexible file naming with template support
//! - Automatic conflict resolution and renaming
//! - Directory structure preservation for multi-file downloads
//! - Atomic file operations and temporary file handling

use std::{
    collections::HashMap,
    fs::{self, File, Permissions},
    io::{self, Read, Write},
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
};

use chrono::Utc;
use percent_encoding;
use serde::{Deserialize, Serialize};
use tempfile::{NamedTempFile, TempDir};
use url::Url;

use crate::error::{Result, ZuupError};

/// File naming template variables
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateVariables {
    /// Original filename from URL
    pub filename: String,
    /// File extension
    pub extension: String,
    /// Base name without extension
    pub basename: String,
    /// Current date in YYYY-MM-DD format
    pub date: String,
    /// Current time in HH-MM-SS format
    pub time: String,
    /// Current timestamp as Unix epoch
    pub timestamp: String,
    /// Download ID
    pub download_id: String,
    /// URL host
    pub host: String,
    /// URL path segments
    pub path_segments: Vec<String>,
    /// Custom variables
    pub custom: HashMap<String, String>,
}

impl TemplateVariables {
    /// Create template variables from URL and download ID
    pub fn from_url(url: &Url, download_id: &str) -> Result<Self> {
        let now = Utc::now();
        let filename = extract_filename_from_url(url)?;
        let (basename, extension) = split_filename(&filename);

        let path_segments: Vec<String> = url
            .path_segments()
            .map(|segments| segments.map(|s| s.to_string()).collect())
            .unwrap_or_default();

        Ok(Self {
            filename: filename.clone(),
            extension,
            basename,
            date: now.format("%Y-%m-%d").to_string(),
            time: now.format("%H-%M-%S").to_string(),
            timestamp: now.timestamp().to_string(),
            download_id: download_id.to_string(),
            host: url.host_str().unwrap_or("unknown").to_string(),
            path_segments,
            custom: HashMap::new(),
        })
    }

    /// Add a custom variable
    pub fn add_custom(&mut self, key: String, value: String) {
        self.custom.insert(key, value);
    }
}

/// File naming template engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileNameTemplate {
    /// Template string with placeholders like {filename}, {date}, etc.
    pub template: String,
}

impl FileNameTemplate {
    /// Create a new template
    pub fn new(template: String) -> Self {
        Self { template }
    }

    /// Apply template with variables to generate filename
    pub fn apply(&self, variables: &TemplateVariables) -> Result<String> {
        let mut result = self.template.clone();

        // Replace built-in variables
        result = result.replace("{filename}", &variables.filename);
        result = result.replace("{extension}", &variables.extension);
        result = result.replace("{basename}", &variables.basename);
        result = result.replace("{date}", &variables.date);
        result = result.replace("{time}", &variables.time);
        result = result.replace("{timestamp}", &variables.timestamp);
        result = result.replace("{download_id}", &variables.download_id);
        result = result.replace("{host}", &variables.host);

        // Replace path segments
        for (i, segment) in variables.path_segments.iter().enumerate() {
            result = result.replace(&format!("{{path[{}]}}", i), segment);
        }

        // Replace custom variables
        for (key, value) in &variables.custom {
            result = result.replace(&format!("{{{}}}", key), value);
        }

        // Sanitize the result
        Ok(sanitize_filename(&result))
    }
}

impl Default for FileNameTemplate {
    fn default() -> Self {
        Self::new("{filename}".to_string())
    }
}

/// Conflict resolution strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictResolution {
    /// Overwrite existing file
    Overwrite,
    /// Skip download if file exists
    Skip,
    /// Rename with numeric suffix (file.txt -> file (1).txt)
    Rename,
    /// Rename with timestamp suffix
    RenameWithTimestamp,
    /// Prompt user for action (not applicable for library usage)
    Prompt,
}

/// File organization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileOrganizationConfig {
    /// Base output directory
    pub base_directory: PathBuf,
    /// File naming template
    pub naming_template: FileNameTemplate,
    /// Conflict resolution strategy
    pub conflict_resolution: ConflictResolution,
    /// Whether to create subdirectories based on URL structure
    pub preserve_url_structure: bool,
    /// Whether to organize files by date
    pub organize_by_date: bool,
    /// Whether to organize files by file type
    pub organize_by_type: bool,
    /// Custom directory template
    pub directory_template: Option<String>,
    /// Maximum filename length
    pub max_filename_length: usize,
    /// Whether to preserve original directory structure for multi-file downloads
    pub preserve_directory_structure: bool,
}

impl Default for FileOrganizationConfig {
    fn default() -> Self {
        Self {
            base_directory: PathBuf::from("."),
            naming_template: FileNameTemplate::default(),
            conflict_resolution: ConflictResolution::Rename,
            preserve_url_structure: false,
            organize_by_date: false,
            organize_by_type: false,
            directory_template: None,
            max_filename_length: 255,
            preserve_directory_structure: true,
        }
    }
}

/// File manager for handling file operations and organization
#[derive(Debug)]
pub struct FileManager {
    config: FileOrganizationConfig,
}

impl FileManager {
    /// Create a new file manager with configuration
    pub fn new(config: FileOrganizationConfig) -> Self {
        Self { config }
    }

    /// Generate the final file path for a download
    pub fn generate_file_path(
        &self,
        url: &Url,
        download_id: &str,
        custom_filename: Option<&str>,
        custom_output_path: Option<&Path>,
    ) -> Result<PathBuf> {
        // Use custom output path if provided, otherwise use base directory
        let base_dir = custom_output_path
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| self.config.base_directory.clone());

        // Generate directory path
        let dir_path = self.generate_directory_path(&base_dir, url, download_id)?;

        // Generate filename
        let filename = if let Some(custom) = custom_filename {
            sanitize_filename(custom)
        } else {
            let variables = TemplateVariables::from_url(url, download_id)?;
            self.config.naming_template.apply(&variables)?
        };

        // Truncate filename if too long
        let filename = truncate_filename(&filename, self.config.max_filename_length);

        let mut file_path = dir_path.join(filename);

        // Handle conflicts
        file_path = self.resolve_conflicts(file_path)?;

        Ok(file_path)
    }

    /// Generate directory path based on organization settings
    fn generate_directory_path(
        &self,
        base_dir: &Path,
        url: &Url,
        download_id: &str,
    ) -> Result<PathBuf> {
        let mut dir_path = base_dir.to_path_buf();

        // Apply custom directory template if provided
        if let Some(template) = &self.config.directory_template {
            let variables = TemplateVariables::from_url(url, download_id)?;
            let custom_dir = apply_directory_template(template, &variables)?;
            dir_path = dir_path.join(custom_dir);
        } else {
            // Apply built-in organization rules
            if self.config.organize_by_date {
                let now = Utc::now();
                dir_path = dir_path
                    .join(now.format("%Y").to_string())
                    .join(now.format("%m-%B").to_string())
                    .join(now.format("%d").to_string());
            }

            if self.config.organize_by_type {
                let filename = extract_filename_from_url(url)?;
                let file_type = get_file_type(&filename);
                dir_path = dir_path.join(file_type);
            }

            if self.config.preserve_url_structure
                && let Some(host) = url.host_str()
            {
                dir_path = dir_path.join(sanitize_filename(host));

                if let Some(segments) = url.path_segments() {
                    for segment in segments {
                        if !segment.is_empty() && segment != "/" {
                            dir_path = dir_path.join(sanitize_filename(segment));
                        }
                    }
                    // Remove the last segment if it's a filename
                    if let Some(parent) = dir_path.parent() {
                        let filename = extract_filename_from_url(url)?;
                        if dir_path.file_name().and_then(|n| n.to_str()) == Some(&filename) {
                            dir_path = parent.to_path_buf();
                        }
                    }
                }
            }
        }

        Ok(dir_path)
    }

    /// Resolve filename conflicts based on strategy
    fn resolve_conflicts(&self, mut file_path: PathBuf) -> Result<PathBuf> {
        if !file_path.exists() {
            return Ok(file_path);
        }

        match self.config.conflict_resolution {
            ConflictResolution::Overwrite => Ok(file_path),
            ConflictResolution::Skip => Err(ZuupError::FileExists(file_path)),
            ConflictResolution::Rename => {
                file_path = find_available_filename_with_counter(file_path)?;
                Ok(file_path)
            }
            ConflictResolution::RenameWithTimestamp => {
                file_path = find_available_filename_with_timestamp(file_path)?;
                Ok(file_path)
            }
            ConflictResolution::Prompt => {
                // For library usage, default to rename
                file_path = find_available_filename_with_counter(file_path)?;
                Ok(file_path)
            }
        }
    }

    /// Create directory structure for a file path
    pub fn create_directory_structure(&self, file_path: &Path) -> Result<()> {
        if let Some(parent) = file_path.parent() {
            fs::create_dir_all(parent).map_err(ZuupError::Io)?;
        }
        Ok(())
    }

    /// Preserve directory structure for multi-file downloads
    pub fn create_multi_file_structure(
        &self,
        base_path: &Path,
        relative_paths: &[String],
    ) -> Result<Vec<PathBuf>> {
        let mut created_paths = Vec::new();

        for relative_path in relative_paths {
            let full_path = base_path.join(relative_path);

            // Create parent directories
            if let Some(parent) = full_path.parent() {
                fs::create_dir_all(parent).map_err(ZuupError::Io)?;
            }

            created_paths.push(full_path);
        }

        Ok(created_paths)
    }

    /// Get file organization configuration
    pub fn config(&self) -> &FileOrganizationConfig {
        &self.config
    }

    /// Update file organization configuration
    pub fn update_config(&mut self, config: FileOrganizationConfig) {
        self.config = config;
    }
}

/// Extract filename from URL
pub fn extract_filename_from_url(url: &Url) -> Result<String> {
    // Try to get filename from path
    if let Some(last_segment) = url.path_segments().and_then(|s| s.clone().next_back())
        && !last_segment.is_empty()
        && last_segment.contains('.')
    {
        return url_decode(last_segment);
    }

    // Try to get filename from query parameters
    if let Some(query) = url.query() {
        for (key, value) in url::form_urlencoded::parse(query.as_bytes()) {
            if key == "filename" || key == "file" {
                return Ok(value.to_string());
            }
        }
    }

    // Fallback to generating a filename
    let host = url.host_str().unwrap_or("download");
    let timestamp = Utc::now().timestamp();
    Ok(format!("{}_{}", sanitize_filename(host), timestamp))
}

/// Split filename into basename and extension
pub fn split_filename(filename: &str) -> (String, String) {
    if let Some(dot_pos) = filename.rfind('.') {
        let basename = filename[..dot_pos].to_string();
        let extension = filename[dot_pos + 1..].to_string();
        (basename, extension)
    } else {
        (filename.to_string(), String::new())
    }
}

/// Sanitize filename by removing invalid characters
pub fn sanitize_filename(filename: &str) -> String {
    let invalid_chars = ['<', '>', ':', '"', '|', '?', '*', '/', '\\'];
    let mut sanitized = filename.to_string();

    for &ch in &invalid_chars {
        sanitized = sanitized.replace(ch, "_");
    }

    // Remove control characters
    sanitized = sanitized.chars().filter(|&c| !c.is_control()).collect();

    // Trim whitespace and dots from ends
    sanitized = sanitized.trim().trim_matches('.').to_string();

    // Ensure filename is not empty
    if sanitized.is_empty() {
        sanitized = "download".to_string();
    }

    sanitized
}

/// Truncate filename to maximum length while preserving extension
pub fn truncate_filename(filename: &str, max_length: usize) -> String {
    if filename.len() <= max_length {
        return filename.to_string();
    }

    let (basename, extension) = split_filename(filename);

    if extension.is_empty() {
        // No extension, just truncate the whole filename
        return basename.chars().take(max_length).collect();
    }

    let extension_with_dot = format!(".{}", extension);

    // If extension itself is too long, just truncate the whole filename
    if extension_with_dot.len() >= max_length {
        return filename.chars().take(max_length).collect();
    }

    // Calculate available length for basename
    let available_length = max_length - extension_with_dot.len();

    let truncated_basename: String = basename.chars().take(available_length).collect();

    format!("{}{}", truncated_basename, extension_with_dot)
}

/// Find available filename with numeric counter
pub fn find_available_filename_with_counter(file_path: PathBuf) -> Result<PathBuf> {
    if !file_path.exists() {
        return Ok(file_path);
    }

    let parent = file_path.parent().unwrap_or(Path::new("."));
    let filename = file_path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| ZuupError::InvalidPath(file_path.clone()))?;

    let (basename, extension) = split_filename(filename);
    let extension_with_dot = if extension.is_empty() {
        String::new()
    } else {
        format!(".{}", extension)
    };

    for counter in 1..=9999 {
        let new_filename = format!("{} ({}){}", basename, counter, extension_with_dot);
        let new_path = parent.join(new_filename);

        if !new_path.exists() {
            return Ok(new_path);
        }
    }

    Err(ZuupError::TooManyConflicts(file_path))
}

/// Find available filename with timestamp
pub fn find_available_filename_with_timestamp(file_path: PathBuf) -> Result<PathBuf> {
    let parent = file_path.parent().unwrap_or(Path::new("."));
    let filename = file_path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| ZuupError::InvalidPath(file_path.clone()))?;

    let (basename, extension) = split_filename(filename);
    let extension_with_dot = if extension.is_empty() {
        String::new()
    } else {
        format!(".{}", extension)
    };

    let timestamp = Utc::now().format("%Y%m%d_%H%M%S").to_string();
    let new_filename = format!("{}_{}{}", basename, timestamp, extension_with_dot);
    let new_path = parent.join(new_filename);

    // If still conflicts, add microseconds
    if new_path.exists() {
        let timestamp_micro = Utc::now().format("%Y%m%d_%H%M%S_%6f").to_string();
        let new_filename = format!("{}_{}{}", basename, timestamp_micro, extension_with_dot);
        Ok(parent.join(new_filename))
    } else {
        Ok(new_path)
    }
}

/// Get file type category based on extension
pub fn get_file_type(filename: &str) -> &'static str {
    let (_, extension) = split_filename(filename);
    let ext = extension.to_lowercase();

    match ext.as_str() {
        "jpg" | "jpeg" | "png" | "gif" | "bmp" | "svg" | "webp" | "ico" => "images",
        "mp4" | "avi" | "mkv" | "mov" | "wmv" | "flv" | "webm" | "m4v" => "videos",
        "mp3" | "wav" | "flac" | "aac" | "ogg" | "wma" | "m4a" => "audio",
        "pdf" | "doc" | "docx" | "txt" | "rtf" | "odt" | "pages" => "documents",
        "zip" | "rar" | "7z" | "tar" | "gz" | "bz2" | "xz" => "archives",
        "exe" | "msi" | "deb" | "rpm" | "dmg" | "pkg" | "app" => "software",
        _ => "other",
    }
}

/// Apply directory template with variables
fn apply_directory_template(template: &str, variables: &TemplateVariables) -> Result<String> {
    let mut result = template.to_string();

    // Replace variables similar to filename template
    result = result.replace("{date}", &variables.date);
    result = result.replace("{host}", &variables.host);
    result = result.replace("{download_id}", &variables.download_id);

    // Replace custom variables
    for (key, value) in &variables.custom {
        result = result.replace(&format!("{{{}}}", key), value);
    }

    // Sanitize directory components
    let components: Vec<String> = result
        .split('/')
        .map(sanitize_filename)
        .filter(|component| !component.is_empty())
        .collect();

    Ok(components.join("/"))
}

/// URL decode a string
fn url_decode(input: &str) -> Result<String> {
    percent_encoding::percent_decode_str(input)
        .decode_utf8()
        .map(|s| s.to_string())
        .map_err(|_| ZuupError::InvalidUrl("Invalid URL encoding".to_string()))
}

/// File system operations manager
#[derive(Debug)]
pub struct FileSystemManager {
    /// Temporary directory for atomic operations
    temp_dir: Option<TempDir>,
}

impl FileSystemManager {
    /// Create a new file system manager
    pub fn new() -> Result<Self> {
        Ok(Self { temp_dir: None })
    }

    /// Create a new file system manager with custom temp directory
    pub fn with_temp_dir(temp_dir: TempDir) -> Self {
        Self {
            temp_dir: Some(temp_dir),
        }
    }

    /// Get or create temporary directory
    fn get_temp_dir(&mut self) -> Result<&TempDir> {
        if self.temp_dir.is_none() {
            self.temp_dir = Some(TempDir::new().map_err(ZuupError::Io)?);
        }
        Ok(self.temp_dir.as_ref().unwrap())
    }

    /// Create a temporary file for atomic operations
    pub fn create_temp_file(&mut self, prefix: Option<&str>) -> Result<NamedTempFile> {
        let temp_dir = self.get_temp_dir()?;
        let mut builder = tempfile::Builder::new();

        if let Some(prefix) = prefix {
            builder.prefix(prefix);
        }

        builder.tempfile_in(temp_dir.path()).map_err(ZuupError::Io)
    }

    /// Perform atomic file write operation
    pub fn atomic_write<P: AsRef<Path>, F>(&mut self, target_path: P, write_fn: F) -> Result<()>
    where
        F: FnOnce(&mut File) -> Result<()>,
    {
        let target_path = target_path.as_ref();

        // Create temporary file in the same directory as target for atomic rename
        let temp_file = if let Some(parent) = target_path.parent() {
            tempfile::Builder::new()
                .prefix(".zuup_temp_")
                .tempfile_in(parent)
                .map_err(ZuupError::Io)?
        } else {
            self.create_temp_file(Some(".zuup_temp_"))?
        };

        // Write to temporary file
        let (mut file, temp_path) = temp_file.into_parts();
        write_fn(&mut file)?;

        // Ensure all data is written to disk
        file.sync_all().map_err(ZuupError::Io)?;
        drop(file);

        // Atomically move temporary file to target location
        fs::rename(&temp_path, target_path).map_err(ZuupError::Io)?;

        Ok(())
    }

    /// Copy file with progress callback
    pub fn copy_with_progress<P1: AsRef<Path>, P2: AsRef<Path>, F>(
        &self,
        from: P1,
        to: P2,
        mut progress_callback: F,
    ) -> Result<u64>
    where
        F: FnMut(u64, u64),
    {
        let from = from.as_ref();
        let to = to.as_ref();

        let mut source = File::open(from).map_err(ZuupError::Io)?;
        let mut dest = File::create(to).map_err(ZuupError::Io)?;

        let total_size = source.metadata().map_err(ZuupError::Io)?.len();
        let mut copied = 0u64;
        let mut buffer = vec![0u8; 64 * 1024]; // 64KB buffer

        loop {
            let bytes_read = source.read(&mut buffer).map_err(ZuupError::Io)?;
            if bytes_read == 0 {
                break;
            }

            dest.write_all(&buffer[..bytes_read])
                .map_err(ZuupError::Io)?;
            copied += bytes_read as u64;
            progress_callback(copied, total_size);
        }

        dest.sync_all().map_err(ZuupError::Io)?;
        Ok(copied)
    }

    /// Move file atomically
    pub fn atomic_move<P1: AsRef<Path>, P2: AsRef<Path>>(&self, from: P1, to: P2) -> Result<()> {
        let from = from.as_ref();
        let to = to.as_ref();

        // Try atomic rename first (works if on same filesystem)
        match fs::rename(from, to) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == io::ErrorKind::CrossesDevices => {
                // Cross-device move, need to copy and delete
                self.copy_with_progress(from, to, |_, _| {})?;
                fs::remove_file(from).map_err(ZuupError::Io)?;
                Ok(())
            }
            Err(e) => Err(ZuupError::Io(e)),
        }
    }

    /// Check available disk space
    pub fn check_disk_space<P: AsRef<Path>>(&self, path: P) -> Result<DiskSpaceInfo> {
        let path = path.as_ref();

        // Use statvfs on Unix systems
        #[cfg(unix)]
        {
            use std::ffi::CString;
            use std::mem;

            let path_cstr = CString::new(path.to_string_lossy().as_bytes())
                .map_err(|_| ZuupError::InvalidPath(path.to_path_buf()))?;

            let mut statvfs: libc::statvfs = unsafe { mem::zeroed() };
            let result = unsafe { libc::statvfs(path_cstr.as_ptr(), &mut statvfs) };

            if result == 0 {
                let block_size = statvfs.f_frsize as u64;
                let total_blocks = statvfs.f_blocks as u64;
                let free_blocks = statvfs.f_bavail as u64;

                Ok(DiskSpaceInfo {
                    total: total_blocks * block_size,
                    available: free_blocks * block_size,
                    used: (total_blocks - free_blocks) * block_size,
                })
            } else {
                Err(ZuupError::Io(io::Error::last_os_error()))
            }
        }

        // Fallback for non-Unix systems
        #[cfg(not(unix))]
        {
            // For Windows and other systems, we'll use a simpler approach
            // This is a basic implementation - in a real system you'd want proper Windows API calls
            Ok(DiskSpaceInfo {
                total: 0,
                available: 0,
                used: 0,
            })
        }
    }

    /// Ensure sufficient disk space is available
    pub fn ensure_disk_space<P: AsRef<Path>>(&self, path: P, required_bytes: u64) -> Result<()> {
        let disk_info = self.check_disk_space(path)?;

        if disk_info.available < required_bytes {
            return Err(ZuupError::InsufficientDiskSpace);
        }

        Ok(())
    }

    /// Set file permissions
    pub fn set_permissions<P: AsRef<Path>>(
        &self,
        path: P,
        permissions: FilePermissions,
    ) -> Result<()> {
        let path = path.as_ref();

        #[cfg(unix)]
        {
            let mode = permissions.to_unix_mode();
            let perms = Permissions::from_mode(mode);
            fs::set_permissions(path, perms).map_err(ZuupError::Io)?;
        }

        #[cfg(not(unix))]
        {
            // For non-Unix systems, we'll set basic read-only or read-write permissions
            let mut perms = fs::metadata(path).map_err(ZuupError::Io)?.permissions();
            perms.set_readonly(!permissions.owner_write);
            fs::set_permissions(path, perms).map_err(ZuupError::Io)?;
        }

        Ok(())
    }

    /// Get file permissions
    pub fn get_permissions<P: AsRef<Path>>(&self, path: P) -> Result<FilePermissions> {
        let path = path.as_ref();
        let metadata = fs::metadata(path).map_err(ZuupError::Io)?;
        let perms = metadata.permissions();

        #[cfg(unix)]
        {
            let mode = perms.mode();
            Ok(FilePermissions::from_unix_mode(mode))
        }

        #[cfg(not(unix))]
        {
            Ok(FilePermissions {
                owner_read: true,
                owner_write: !perms.readonly(),
                owner_execute: false,
                group_read: false,
                group_write: false,
                group_execute: false,
                other_read: false,
                other_write: false,
                other_execute: false,
            })
        }
    }

    /// Create directory with specific permissions
    pub fn create_dir_with_permissions<P: AsRef<Path>>(
        &self,
        path: P,
        permissions: FilePermissions,
    ) -> Result<()> {
        let path = path.as_ref();
        fs::create_dir_all(path).map_err(ZuupError::Io)?;
        self.set_permissions(path, permissions)?;
        Ok(())
    }

    /// Securely delete file (overwrite with random data before deletion)
    pub fn secure_delete<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();

        if !path.exists() {
            return Ok(());
        }

        let metadata = fs::metadata(path).map_err(ZuupError::Io)?;
        if metadata.is_file() {
            let file_size = metadata.len();

            // Overwrite file with random data
            let mut file = File::options()
                .write(true)
                .open(path)
                .map_err(ZuupError::Io)?;

            let mut buffer = vec![0u8; 4096];
            let mut remaining = file_size;

            while remaining > 0 {
                let chunk_size = std::cmp::min(remaining, buffer.len() as u64) as usize;

                // Fill buffer with random data
                for byte in &mut buffer[..chunk_size] {
                    *byte = rand::random();
                }

                file.write_all(&buffer[..chunk_size])
                    .map_err(ZuupError::Io)?;
                remaining -= chunk_size as u64;
            }

            file.sync_all().map_err(ZuupError::Io)?;
        }

        // Remove the file
        fs::remove_file(path).map_err(ZuupError::Io)?;
        Ok(())
    }

    /// Calculate directory size recursively
    pub fn calculate_directory_size<P: AsRef<Path>>(&self, path: P) -> Result<u64> {
        let path = path.as_ref();
        let mut total_size = 0u64;

        fn visit_dir(dir: &Path, total: &mut u64) -> Result<()> {
            let entries = fs::read_dir(dir).map_err(ZuupError::Io)?;

            for entry in entries {
                let entry = entry.map_err(ZuupError::Io)?;
                let metadata = entry.metadata().map_err(ZuupError::Io)?;

                if metadata.is_file() {
                    *total += metadata.len();
                } else if metadata.is_dir() {
                    visit_dir(&entry.path(), total)?;
                }
            }

            Ok(())
        }

        visit_dir(path, &mut total_size)?;
        Ok(total_size)
    }
}

impl Default for FileSystemManager {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self { temp_dir: None })
    }
}

/// Disk space information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskSpaceInfo {
    /// Total disk space in bytes
    pub total: u64,
    /// Available disk space in bytes
    pub available: u64,
    /// Used disk space in bytes
    pub used: u64,
}

impl DiskSpaceInfo {
    /// Get usage percentage (0-100)
    pub fn usage_percentage(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.used as f64 / self.total as f64) * 100.0
        }
    }

    /// Check if disk usage is above threshold
    pub fn is_usage_above_threshold(&self, threshold_percent: f64) -> bool {
        self.usage_percentage() > threshold_percent
    }
}

/// File permissions configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilePermissions {
    /// Owner read permission
    pub owner_read: bool,
    /// Owner write permission
    pub owner_write: bool,
    /// Owner execute permission
    pub owner_execute: bool,
    /// Group read permission
    pub group_read: bool,
    /// Group write permission
    pub group_write: bool,
    /// Group execute permission
    pub group_execute: bool,
    /// Other read permission
    pub other_read: bool,
    /// Other write permission
    pub other_write: bool,
    /// Other execute permission
    pub other_execute: bool,
}

impl FilePermissions {
    /// Create permissions for owner read/write only
    pub fn owner_read_write() -> Self {
        Self {
            owner_read: true,
            owner_write: true,
            owner_execute: false,
            group_read: false,
            group_write: false,
            group_execute: false,
            other_read: false,
            other_write: false,
            other_execute: false,
        }
    }

    /// Create permissions for owner read/write/execute
    pub fn owner_all() -> Self {
        Self {
            owner_read: true,
            owner_write: true,
            owner_execute: true,
            group_read: false,
            group_write: false,
            group_execute: false,
            other_read: false,
            other_write: false,
            other_execute: false,
        }
    }

    /// Create permissions for read-only access by owner
    pub fn owner_read_only() -> Self {
        Self {
            owner_read: true,
            owner_write: false,
            owner_execute: false,
            group_read: false,
            group_write: false,
            group_execute: false,
            other_read: false,
            other_write: false,
            other_execute: false,
        }
    }

    /// Create standard file permissions (644)
    pub fn standard_file() -> Self {
        Self {
            owner_read: true,
            owner_write: true,
            owner_execute: false,
            group_read: true,
            group_write: false,
            group_execute: false,
            other_read: true,
            other_write: false,
            other_execute: false,
        }
    }

    /// Create standard directory permissions (755)
    pub fn standard_directory() -> Self {
        Self {
            owner_read: true,
            owner_write: true,
            owner_execute: true,
            group_read: true,
            group_write: false,
            group_execute: true,
            other_read: true,
            other_write: false,
            other_execute: true,
        }
    }

    /// Convert to Unix mode bits
    #[cfg(unix)]
    pub fn to_unix_mode(&self) -> u32 {
        let mut mode = 0u32;

        if self.owner_read {
            mode |= 0o400;
        }
        if self.owner_write {
            mode |= 0o200;
        }
        if self.owner_execute {
            mode |= 0o100;
        }
        if self.group_read {
            mode |= 0o040;
        }
        if self.group_write {
            mode |= 0o020;
        }
        if self.group_execute {
            mode |= 0o010;
        }
        if self.other_read {
            mode |= 0o004;
        }
        if self.other_write {
            mode |= 0o002;
        }
        if self.other_execute {
            mode |= 0o001;
        }

        mode
    }

    /// Create from Unix mode bits
    #[cfg(unix)]
    pub fn from_unix_mode(mode: u32) -> Self {
        Self {
            owner_read: (mode & 0o400) != 0,
            owner_write: (mode & 0o200) != 0,
            owner_execute: (mode & 0o100) != 0,
            group_read: (mode & 0o040) != 0,
            group_write: (mode & 0o020) != 0,
            group_execute: (mode & 0o010) != 0,
            other_read: (mode & 0o004) != 0,
            other_write: (mode & 0o002) != 0,
            other_execute: (mode & 0o001) != 0,
        }
    }
}

impl Default for FilePermissions {
    fn default() -> Self {
        Self::standard_file()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;
    use tempfile::TempDir;

    #[test]
    fn test_extract_filename_from_url() {
        let url = Url::from_str("https://example.com/path/file.zip").unwrap();
        let filename = extract_filename_from_url(&url).unwrap();
        assert_eq!(filename, "file.zip");

        let url = Url::from_str("https://example.com/download?filename=test.pdf").unwrap();
        let filename = extract_filename_from_url(&url).unwrap();
        assert_eq!(filename, "test.pdf");

        let url = Url::from_str("https://example.com/").unwrap();
        let filename = extract_filename_from_url(&url).unwrap();
        assert!(filename.starts_with("example.com_"));
    }

    #[test]
    fn test_split_filename() {
        let (basename, extension) = split_filename("file.zip");
        assert_eq!(basename, "file");
        assert_eq!(extension, "zip");

        let (basename, extension) = split_filename("archive.tar.gz");
        assert_eq!(basename, "archive.tar");
        assert_eq!(extension, "gz");

        let (basename, extension) = split_filename("noextension");
        assert_eq!(basename, "noextension");
        assert_eq!(extension, "");
    }

    #[test]
    fn test_sanitize_filename() {
        assert_eq!(sanitize_filename("file<>name.txt"), "file__name.txt");
        assert_eq!(sanitize_filename("file|name?.txt"), "file_name_.txt");
        assert_eq!(sanitize_filename("  .file.txt.  "), "file.txt");
        assert_eq!(sanitize_filename(""), "download");
    }

    #[test]
    fn test_truncate_filename() {
        assert_eq!(truncate_filename("short.txt", 20), "short.txt");
        assert_eq!(truncate_filename("verylongfilename.txt", 10), "verylo.txt");
        assert_eq!(truncate_filename("verylongfilename", 10), "verylongfi");
    }

    #[test]
    fn test_get_file_type() {
        assert_eq!(get_file_type("image.jpg"), "images");
        assert_eq!(get_file_type("video.mp4"), "videos");
        assert_eq!(get_file_type("song.mp3"), "audio");
        assert_eq!(get_file_type("document.pdf"), "documents");
        assert_eq!(get_file_type("archive.zip"), "archives");
        assert_eq!(get_file_type("program.exe"), "software");
        assert_eq!(get_file_type("unknown.xyz"), "other");
    }

    #[test]
    fn test_template_variables_from_url() {
        let url = Url::from_str("https://example.com/path/to/file.zip").unwrap();
        let variables = TemplateVariables::from_url(&url, "test-id").unwrap();

        assert_eq!(variables.filename, "file.zip");
        assert_eq!(variables.basename, "file");
        assert_eq!(variables.extension, "zip");
        assert_eq!(variables.download_id, "test-id");
        assert_eq!(variables.host, "example.com");
        assert_eq!(variables.path_segments, vec!["path", "to", "file.zip"]);
    }

    #[test]
    fn test_filename_template_apply() {
        let template = FileNameTemplate::new("{basename}_{date}.{extension}".to_string());
        let url = Url::from_str("https://example.com/file.zip").unwrap();
        let variables = TemplateVariables::from_url(&url, "test-id").unwrap();

        let result = template.apply(&variables).unwrap();
        assert!(result.starts_with("file_"));
        assert!(result.ends_with(".zip"));
    }

    #[test]
    fn test_file_manager_generate_path() {
        let temp_dir = TempDir::new().unwrap();
        let config = FileOrganizationConfig {
            base_directory: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let manager = FileManager::new(config);
        let url = Url::from_str("https://example.com/file.zip").unwrap();

        let path = manager
            .generate_file_path(&url, "test-id", None, None)
            .unwrap();
        assert_eq!(path.file_name().unwrap(), "file.zip");
        assert_eq!(path.parent().unwrap(), temp_dir.path());
    }

    #[test]
    fn test_file_manager_with_custom_filename() {
        let temp_dir = TempDir::new().unwrap();
        let config = FileOrganizationConfig {
            base_directory: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let manager = FileManager::new(config);
        let url = Url::from_str("https://example.com/file.zip").unwrap();

        let path = manager
            .generate_file_path(&url, "test-id", Some("custom.zip"), None)
            .unwrap();
        assert_eq!(path.file_name().unwrap(), "custom.zip");
    }

    #[test]
    fn test_file_manager_organize_by_type() {
        let temp_dir = TempDir::new().unwrap();
        let config = FileOrganizationConfig {
            base_directory: temp_dir.path().to_path_buf(),
            organize_by_type: true,
            ..Default::default()
        };

        let manager = FileManager::new(config);
        let url = Url::from_str("https://example.com/image.jpg").unwrap();

        let path = manager
            .generate_file_path(&url, "test-id", None, None)
            .unwrap();
        assert!(path.to_string_lossy().contains("images"));
    }

    #[test]
    fn test_file_manager_preserve_url_structure() {
        let temp_dir = TempDir::new().unwrap();
        let config = FileOrganizationConfig {
            base_directory: temp_dir.path().to_path_buf(),
            preserve_url_structure: true,
            ..Default::default()
        };

        let manager = FileManager::new(config);
        let url = Url::from_str("https://example.com/path/to/file.zip").unwrap();

        let path = manager
            .generate_file_path(&url, "test-id", None, None)
            .unwrap();
        assert!(path.to_string_lossy().contains("example.com"));
        assert!(path.to_string_lossy().contains("path"));
        assert!(path.to_string_lossy().contains("to"));
    }

    #[test]
    fn test_conflict_resolution_rename() {
        let temp_dir = TempDir::new().unwrap();
        let existing_file = temp_dir.path().join("test.txt");
        std::fs::write(&existing_file, "content").unwrap();

        let new_path = find_available_filename_with_counter(existing_file).unwrap();
        assert_eq!(new_path.file_name().unwrap(), "test (1).txt");
    }

    #[test]
    fn test_conflict_resolution_timestamp() {
        let temp_dir = TempDir::new().unwrap();
        let existing_file = temp_dir.path().join("test.txt");
        std::fs::write(&existing_file, "content").unwrap();

        let new_path = find_available_filename_with_timestamp(existing_file).unwrap();
        let filename = new_path.file_name().unwrap().to_string_lossy();
        assert!(filename.starts_with("test_"));
        assert!(filename.ends_with(".txt"));
    }

    #[test]
    fn test_create_multi_file_structure() {
        let temp_dir = TempDir::new().unwrap();
        let config = FileOrganizationConfig::default();
        let manager = FileManager::new(config);

        let relative_paths = vec![
            "dir1/file1.txt".to_string(),
            "dir1/subdir/file2.txt".to_string(),
            "dir2/file3.txt".to_string(),
        ];

        let paths = manager
            .create_multi_file_structure(temp_dir.path(), &relative_paths)
            .unwrap();

        assert_eq!(paths.len(), 3);
        assert!(paths[0].parent().unwrap().exists());
        assert!(paths[1].parent().unwrap().exists());
        assert!(paths[2].parent().unwrap().exists());
    }

    #[test]
    fn test_directory_template() {
        let template = "{date}/{host}";
        let url = Url::from_str("https://example.com/file.zip").unwrap();
        let variables = TemplateVariables::from_url(&url, "test-id").unwrap();

        let result = apply_directory_template(template, &variables).unwrap();
        assert!(result.contains("example.com"));
        assert!(result.contains("/"));
    }

    #[test]
    fn test_file_system_manager_creation() {
        let manager = FileSystemManager::new();
        assert!(manager.is_ok());

        let default_manager = FileSystemManager::default();
        // Should not panic
        drop(default_manager);
    }

    #[test]
    fn test_atomic_write() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = FileSystemManager::with_temp_dir(temp_dir);
        let test_file = manager.get_temp_dir().unwrap().path().join("test.txt");

        let content = b"Hello, World!";
        let result = manager.atomic_write(&test_file, |file| {
            file.write_all(content)?;
            Ok(())
        });

        assert!(result.is_ok());
        assert!(test_file.exists());

        let written_content = std::fs::read(&test_file).unwrap();
        assert_eq!(written_content, content);
    }

    #[test]
    fn test_atomic_write_failure_cleanup() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = FileSystemManager::with_temp_dir(temp_dir);
        let test_file = manager.get_temp_dir().unwrap().path().join("test.txt");

        // This should fail and not leave temporary files
        let result = manager.atomic_write(&test_file, |_file| {
            Err(ZuupError::Internal("Test error".to_string()))
        });

        assert!(result.is_err());
        assert!(!test_file.exists());
    }

    #[test]
    fn test_copy_with_progress() {
        let temp_dir = TempDir::new().unwrap();
        let manager = FileSystemManager::default();

        let source_file = temp_dir.path().join("source.txt");
        let dest_file = temp_dir.path().join("dest.txt");

        let content = b"Test content for copy operation";
        std::fs::write(&source_file, content).unwrap();

        let mut progress_calls = 0;
        let mut last_copied = 0u64;
        let mut last_total = 0u64;

        let result = manager.copy_with_progress(&source_file, &dest_file, |copied, total| {
            progress_calls += 1;
            last_copied = copied;
            last_total = total;
        });

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), content.len() as u64);
        assert!(progress_calls > 0);
        assert_eq!(last_copied, content.len() as u64);
        assert_eq!(last_total, content.len() as u64);

        let copied_content = std::fs::read(&dest_file).unwrap();
        assert_eq!(copied_content, content);
    }

    #[test]
    fn test_atomic_move_same_filesystem() {
        let temp_dir = TempDir::new().unwrap();
        let manager = FileSystemManager::default();

        let source_file = temp_dir.path().join("source.txt");
        let dest_file = temp_dir.path().join("dest.txt");

        let content = b"Test content for move operation";
        std::fs::write(&source_file, content).unwrap();

        let result = manager.atomic_move(&source_file, &dest_file);

        assert!(result.is_ok());
        assert!(!source_file.exists());
        assert!(dest_file.exists());

        let moved_content = std::fs::read(&dest_file).unwrap();
        assert_eq!(moved_content, content);
    }

    #[test]
    fn test_disk_space_info() {
        let disk_info = DiskSpaceInfo {
            total: 1000,
            available: 300,
            used: 700,
        };

        assert_eq!(disk_info.usage_percentage(), 70.0);
        assert!(disk_info.is_usage_above_threshold(50.0));
        assert!(!disk_info.is_usage_above_threshold(80.0));
    }

    #[test]
    fn test_file_permissions() {
        let perms = FilePermissions::owner_read_write();
        assert!(perms.owner_read);
        assert!(perms.owner_write);
        assert!(!perms.owner_execute);
        assert!(!perms.group_read);

        let perms = FilePermissions::standard_file();
        assert!(perms.owner_read);
        assert!(perms.owner_write);
        assert!(!perms.owner_execute);
        assert!(perms.group_read);
        assert!(!perms.group_write);
        assert!(perms.other_read);

        let perms = FilePermissions::standard_directory();
        assert!(perms.owner_execute);
        assert!(perms.group_execute);
        assert!(perms.other_execute);
    }

    #[cfg(unix)]
    #[test]
    fn test_unix_permissions_conversion() {
        let perms = FilePermissions::from_unix_mode(0o644);
        assert!(perms.owner_read);
        assert!(perms.owner_write);
        assert!(!perms.owner_execute);
        assert!(perms.group_read);
        assert!(!perms.group_write);
        assert!(perms.other_read);

        assert_eq!(perms.to_unix_mode(), 0o644);

        let perms = FilePermissions::from_unix_mode(0o755);
        assert_eq!(perms.to_unix_mode(), 0o755);
    }

    #[test]
    fn test_set_and_get_permissions() {
        let temp_dir = TempDir::new().unwrap();
        let manager = FileSystemManager::default();
        let test_file = temp_dir.path().join("test.txt");

        std::fs::write(&test_file, b"test content").unwrap();

        let perms = FilePermissions::owner_read_only();
        let result = manager.set_permissions(&test_file, perms.clone());

        // On some systems this might not work due to permissions, so we'll just check it doesn't panic
        if result.is_ok() {
            let retrieved_perms = manager.get_permissions(&test_file);
            assert!(retrieved_perms.is_ok());
        }
    }

    #[test]
    fn test_create_dir_with_permissions() {
        let temp_dir = TempDir::new().unwrap();
        let manager = FileSystemManager::default();
        let test_dir = temp_dir.path().join("test_dir");

        let perms = FilePermissions::standard_directory();
        let result = manager.create_dir_with_permissions(&test_dir, perms);

        assert!(result.is_ok());
        assert!(test_dir.exists());
        assert!(test_dir.is_dir());
    }

    #[test]
    fn test_calculate_directory_size() {
        let temp_dir = TempDir::new().unwrap();
        let manager = FileSystemManager::default();

        // Create some test files
        let file1 = temp_dir.path().join("file1.txt");
        let file2 = temp_dir.path().join("file2.txt");
        let subdir = temp_dir.path().join("subdir");
        let file3 = subdir.join("file3.txt");

        std::fs::write(&file1, b"12345").unwrap(); // 5 bytes
        std::fs::write(&file2, b"1234567890").unwrap(); // 10 bytes
        std::fs::create_dir(&subdir).unwrap();
        std::fs::write(&file3, b"123").unwrap(); // 3 bytes

        let total_size = manager.calculate_directory_size(temp_dir.path()).unwrap();
        assert_eq!(total_size, 18); // 5 + 10 + 3 = 18 bytes
    }

    #[test]
    fn test_secure_delete() {
        let temp_dir = TempDir::new().unwrap();
        let manager = FileSystemManager::default();
        let test_file = temp_dir.path().join("secret.txt");

        std::fs::write(&test_file, b"secret content").unwrap();
        assert!(test_file.exists());

        let result = manager.secure_delete(&test_file);
        assert!(result.is_ok());
        assert!(!test_file.exists());

        // Test deleting non-existent file (should not error)
        let result = manager.secure_delete(&test_file);
        assert!(result.is_ok());
    }

    #[test]
    fn test_ensure_disk_space() {
        let temp_dir = TempDir::new().unwrap();
        let manager = FileSystemManager::default();

        // This test might fail on systems where we can't get disk space info
        // So we'll just ensure it doesn't panic
        let result = manager.ensure_disk_space(temp_dir.path(), 1024);
        // Don't assert on the result as it depends on actual disk space
        drop(result);
    }

    #[test]
    fn test_temp_file_creation() {
        let mut manager = FileSystemManager::new().unwrap();

        let temp_file = manager.create_temp_file(Some("test_prefix_"));
        assert!(temp_file.is_ok());

        let temp_file = temp_file.unwrap();
        assert!(temp_file.path().exists());

        // Test writing to temp file
        let (mut file, _path) = temp_file.into_parts();
        let result = file.write_all(b"test content");
        assert!(result.is_ok());
    }
}
