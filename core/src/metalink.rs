//! Metalink support for multi-source downloads
//!
//! This module provides parsing and handling for Metalink 3.0 and 4.0 formats,
//! enabling multi-source downloads with checksum verification and source prioritization.

use std::{fmt, str::FromStr};

use chrono::{DateTime, Utc};
use quick_xml::{Reader, events::Event};
use serde::{Deserialize, Serialize};
use url::Url;

use crate::{
    error::{NetworkError, Result, ZuupError},
    types::{ChecksumAlgorithm, ChecksumConfig, DownloadOptions},
};

/// Metalink document containing file information and sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metalink {
    /// Metalink version (3.0 or 4.0)
    pub version: MetalinkVersion,
    /// Generator information
    pub generator: Option<String>,
    /// Publication date
    pub published: Option<DateTime<Utc>>,
    /// Files described in this metalink
    pub files: Vec<MetalinkFile>,
}

/// Metalink version
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetalinkVersion {
    /// Metalink 3.0 format
    V3,
    /// Metalink 4.0 format (RFC 5854)
    V4,
}

/// A file described in a Metalink document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalinkFile {
    /// File name
    pub name: String,
    /// File size in bytes
    pub size: Option<u64>,
    /// File description
    pub description: Option<String>,
    /// MIME type
    pub mime_type: Option<String>,
    /// Operating system compatibility
    pub os: Vec<String>,
    /// Language codes
    pub language: Vec<String>,
    /// Download sources (URLs)
    pub urls: Vec<MetalinkUrl>,
    /// Checksums for integrity verification
    pub checksums: Vec<MetalinkChecksum>,
    /// Digital signatures
    pub signatures: Vec<MetalinkSignature>,
    /// Piece hashes for chunk verification
    pub pieces: Vec<MetalinkPiece>,
}

/// A download source URL
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalinkUrl {
    /// The URL
    pub url: Url,
    /// Priority (higher number = higher priority)
    pub priority: Option<u32>,
    /// Location (country code)
    pub location: Option<String>,
    /// Maximum connections allowed
    pub max_connections: Option<u32>,
}

/// Checksum information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalinkChecksum {
    /// Checksum type (MD5, SHA-1, SHA-256, etc.)
    pub hash_type: ChecksumAlgorithm,
    /// Hexadecimal hash value
    pub hash: String,
}

/// Digital signature information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalinkSignature {
    /// Signature type
    pub signature_type: String,
    /// Signature data
    pub signature: String,
}

/// Piece hash for chunk verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalinkPiece {
    /// Piece length in bytes
    pub length: u64,
    /// Piece type (usually "sha1")
    pub piece_type: String,
    /// Hash values for each piece
    pub hashes: Vec<String>,
}

/// Metalink parser for both v3 and v4 formats
pub struct MetalinkParser;

impl MetalinkParser {
    /// Parse a Metalink document from XML string
    pub fn parse(xml_content: &str) -> Result<Metalink> {
        // First, detect the version by looking at the root element
        let version = Self::detect_version(xml_content)?;

        match version {
            MetalinkVersion::V3 => Self::parse_v3(xml_content),
            MetalinkVersion::V4 => Self::parse_v4(xml_content),
        }
    }

    /// Parse a Metalink document from a file
    pub async fn parse_file(file_path: &std::path::Path) -> Result<Metalink> {
        let content = tokio::fs::read_to_string(file_path)
            .await
            .map_err(ZuupError::Io)?;
        Self::parse(&content)
    }

    /// Parse a Metalink document from a URL
    #[cfg(feature = "http")]
    pub async fn parse_url(url: &Url) -> Result<Metalink> {
        let client = reqwest::Client::new();
        let response = client
            .get(url.as_str())
            .send()
            .await
            .map_err(|e| ZuupError::Network(NetworkError::ConnectionFailed(e.to_string())))?;

        let content = response
            .text()
            .await
            .map_err(|e| ZuupError::Network(NetworkError::InvalidResponse(e.to_string())))?;

        Self::parse(&content)
    }

    /// Detect Metalink version from XML content
    fn detect_version(xml_content: &str) -> Result<MetalinkVersion> {
        let mut reader = Reader::from_str(xml_content);
        reader.config_mut().trim_text(true);

        let mut buf = Vec::new();

        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(ref e)) => {
                    let name = e.name();
                    if name.as_ref() == b"metalink" {
                        // Check for version attribute or xmlns
                        for attr in e.attributes() {
                            let attr = attr.map_err(|e| {
                                ZuupError::Config(format!("Invalid XML attribute: {}", e))
                            })?;
                            match attr.key.as_ref() {
                                b"version" => {
                                    let version_str =
                                        std::str::from_utf8(&attr.value).map_err(|e| {
                                            ZuupError::Config(format!(
                                                "Invalid UTF-8 in version: {}",
                                                e
                                            ))
                                        })?;
                                    if version_str.starts_with("4.") {
                                        return Ok(MetalinkVersion::V4);
                                    } else if version_str.starts_with("3.") {
                                        return Ok(MetalinkVersion::V3);
                                    }
                                }
                                b"xmlns" => {
                                    let xmlns = std::str::from_utf8(&attr.value).map_err(|e| {
                                        ZuupError::Config(format!("Invalid UTF-8 in xmlns: {}", e))
                                    })?;
                                    if xmlns.contains("metalink/4.0") || xmlns.contains("rfc5854") {
                                        return Ok(MetalinkVersion::V4);
                                    } else if xmlns.contains("metalink/3.0") {
                                        return Ok(MetalinkVersion::V3);
                                    }
                                }
                                _ => {}
                            }
                        }
                        // Default to v4 if no version specified
                        return Ok(MetalinkVersion::V4);
                    }
                }
                Ok(Event::Eof) => break,
                Err(e) => return Err(ZuupError::Config(format!("XML parsing error: {}", e))),
                _ => {}
            }
            buf.clear();
        }

        Err(ZuupError::Config(
            "No metalink root element found".to_string(),
        ))
    }

    /// Parse Metalink 3.0 format
    fn parse_v3(xml_content: &str) -> Result<Metalink> {
        let mut reader = Reader::from_str(xml_content);
        reader.config_mut().trim_text(true);

        let mut buf = Vec::new();
        let mut files = Vec::new();
        let generator = None;
        let published = None;

        let mut current_file: Option<MetalinkFile> = None;
        let mut current_urls = Vec::new();
        let mut current_checksums = Vec::new();
        let mut current_signatures = Vec::new();
        let mut current_pieces = Vec::new();

        let mut _in_file = false;
        let mut in_resources = false;
        let mut in_verification = false;

        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(ref e)) => {
                    match e.name().as_ref() {
                        b"file" => {
                            _in_file = true;
                            let mut name = String::new();
                            let size = None;

                            for attr in e.attributes() {
                                let attr = attr.map_err(|e| {
                                    ZuupError::Config(format!("Invalid XML attribute: {}", e))
                                })?;

                                if attr.key.as_ref() == b"name" {
                                    let name_str =
                                        std::str::from_utf8(&attr.value).map_err(|e| {
                                            ZuupError::Config(format!(
                                                "Invalid UTF-8 in name: {}",
                                                e
                                            ))
                                        })?;
                                    name = name_str.to_string();
                                }
                            }

                            current_file = Some(MetalinkFile {
                                name,
                                size,
                                description: None,
                                mime_type: None,
                                os: Vec::new(),
                                language: Vec::new(),
                                urls: Vec::new(),
                                checksums: Vec::new(),
                                signatures: Vec::new(),
                                pieces: Vec::new(),
                            });
                        }
                        b"resources" => {
                            in_resources = true;
                        }
                        b"url" if in_resources => {
                            let mut priority = None;
                            let mut location = None;
                            let mut max_connections = None;

                            for attr in e.attributes() {
                                let attr = attr.map_err(|e| {
                                    ZuupError::Config(format!("Invalid XML attribute: {}", e))
                                })?;
                                match attr.key.as_ref() {
                                    b"type" => {
                                        // Skip non-HTTP URLs for now
                                    }
                                    b"preference" => {
                                        let pref_str =
                                            std::str::from_utf8(&attr.value).map_err(|e| {
                                                ZuupError::Config(format!(
                                                    "Invalid UTF-8 in preference: {}",
                                                    e
                                                ))
                                            })?;
                                        priority = pref_str.parse().ok();
                                    }
                                    b"location" => {
                                        location = Some(
                                            std::str::from_utf8(&attr.value)
                                                .map_err(|e| {
                                                    ZuupError::Config(format!(
                                                        "Invalid UTF-8 in location: {}",
                                                        e
                                                    ))
                                                })?
                                                .to_string(),
                                        );
                                    }
                                    b"maxconnections" => {
                                        let conn_str =
                                            std::str::from_utf8(&attr.value).map_err(|e| {
                                                ZuupError::Config(format!(
                                                    "Invalid UTF-8 in maxconnections: {}",
                                                    e
                                                ))
                                            })?;
                                        max_connections = conn_str.parse().ok();
                                    }
                                    _ => {}
                                }
                            }

                            // Read URL content
                            let mut url_content = String::new();
                            loop {
                                match reader.read_event_into(&mut buf) {
                                    Ok(Event::Text(e)) => {
                                        let text =
                                            std::str::from_utf8(e.as_ref()).map_err(|e| {
                                                ZuupError::Config(format!(
                                                    "Invalid UTF-8 in XML: {}",
                                                    e
                                                ))
                                            })?;
                                        url_content.push_str(text);
                                    }
                                    Ok(Event::End(ref e)) if e.name().as_ref() == b"url" => break,
                                    Ok(Event::Eof) => {
                                        return Err(ZuupError::Config(
                                            "Unexpected EOF in URL".to_string(),
                                        ));
                                    }
                                    Err(e) => {
                                        return Err(ZuupError::Config(format!(
                                            "XML parsing error: {}",
                                            e
                                        )));
                                    }
                                    _ => {}
                                }
                            }

                            if let Ok(url) = Url::parse(url_content.trim()) {
                                current_urls.push(MetalinkUrl {
                                    url,
                                    priority,
                                    location,
                                    max_connections,
                                });
                            }
                        }
                        b"verification" => {
                            in_verification = true;
                        }
                        b"hash" if in_verification => {
                            let mut hash_type = None;

                            for attr in e.attributes() {
                                let attr = attr.map_err(|e| {
                                    ZuupError::Config(format!("Invalid XML attribute: {}", e))
                                })?;
                                if attr.key.as_ref() == b"type" {
                                    let type_str =
                                        std::str::from_utf8(&attr.value).map_err(|e| {
                                            ZuupError::Config(format!(
                                                "Invalid UTF-8 in hash type: {}",
                                                e
                                            ))
                                        })?;
                                    hash_type = Self::parse_checksum_algorithm(type_str).ok();
                                }
                            }

                            if let Some(hash_type) = hash_type {
                                let mut hash_content = String::new();
                                loop {
                                    match reader.read_event_into(&mut buf) {
                                        Ok(Event::Text(e)) => {
                                            let text =
                                                std::str::from_utf8(e.as_ref()).map_err(|e| {
                                                    ZuupError::Config(format!(
                                                        "Invalid UTF-8 in XML: {}",
                                                        e
                                                    ))
                                                })?;
                                            hash_content.push_str(text);
                                        }
                                        Ok(Event::End(ref e)) if e.name().as_ref() == b"hash" => {
                                            break;
                                        }
                                        Ok(Event::Eof) => {
                                            return Err(ZuupError::Config(
                                                "Unexpected EOF in hash".to_string(),
                                            ));
                                        }
                                        Err(e) => {
                                            return Err(ZuupError::Config(format!(
                                                "XML parsing error: {}",
                                                e
                                            )));
                                        }
                                        _ => {}
                                    }
                                }

                                current_checksums.push(MetalinkChecksum {
                                    hash_type,
                                    hash: hash_content.trim().to_string(),
                                });
                            }
                        }
                        _ => {}
                    }
                }
                Ok(Event::End(ref e)) => match e.name().as_ref() {
                    b"file" => {
                        if let Some(mut file) = current_file.take() {
                            file.urls = std::mem::take(&mut current_urls);
                            file.checksums = std::mem::take(&mut current_checksums);
                            file.signatures = std::mem::take(&mut current_signatures);
                            file.pieces = std::mem::take(&mut current_pieces);
                            files.push(file);
                        }
                        _in_file = false;
                    }
                    b"resources" => {
                        in_resources = false;
                    }
                    b"verification" => {
                        in_verification = false;
                    }
                    _ => {}
                },
                Ok(Event::Eof) => break,
                Err(e) => return Err(ZuupError::Config(format!("XML parsing error: {}", e))),
                _ => {}
            }
            buf.clear();
        }

        Ok(Metalink {
            version: MetalinkVersion::V3,
            generator,
            published,
            files,
        })
    }

    /// Parse Metalink 4.0 format (RFC 5854)
    fn parse_v4(xml_content: &str) -> Result<Metalink> {
        // todo)) Similar to v3 but with different XML structure
        // For now, implement a basic parser - can be enhanced later
        Self::parse_v3(xml_content).map(|mut metalink| {
            metalink.version = MetalinkVersion::V4;
            metalink
        })
    }

    /// Parse checksum algorithm from string
    fn parse_checksum_algorithm(s: &str) -> Result<ChecksumAlgorithm> {
        match s.to_lowercase().as_str() {
            "md5" => Ok(ChecksumAlgorithm::Md5),
            "sha1" | "sha-1" => Ok(ChecksumAlgorithm::Sha1),
            "sha256" | "sha-256" => Ok(ChecksumAlgorithm::Sha256),
            "sha512" | "sha-512" => Ok(ChecksumAlgorithm::Sha512),
            _ => Err(ZuupError::Config(format!(
                "Unsupported checksum algorithm: {}",
                s
            ))),
        }
    }
}

impl fmt::Display for MetalinkVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetalinkVersion::V3 => write!(f, "3.0"),
            MetalinkVersion::V4 => write!(f, "4.0"),
        }
    }
}

impl FromStr for MetalinkVersion {
    type Err = ZuupError;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "3.0" | "3" => Ok(MetalinkVersion::V3),
            "4.0" | "4" => Ok(MetalinkVersion::V4),
            _ => Err(ZuupError::Config(format!(
                "Unknown Metalink version: {}",
                s
            ))),
        }
    }
}

impl MetalinkFile {
    /// Get the best checksum for verification (prefer stronger algorithms)
    pub fn best_checksum(&self) -> Option<&MetalinkChecksum> {
        // Prefer SHA-512 > SHA-256 > SHA-1 > MD5
        self.checksums
            .iter()
            .max_by_key(|checksum| match checksum.hash_type {
                ChecksumAlgorithm::Sha512 => 4,
                ChecksumAlgorithm::Sha256 => 3,
                ChecksumAlgorithm::Sha1 => 2,
                ChecksumAlgorithm::Md5 => 1,
            })
    }

    /// Get URLs sorted by priority (highest first)
    pub fn sorted_urls(&self) -> Vec<&MetalinkUrl> {
        let mut urls: Vec<&MetalinkUrl> = self.urls.iter().collect();
        urls.sort_by(|a, b| {
            let priority_a = a.priority.unwrap_or(0);
            let priority_b = b.priority.unwrap_or(0);
            priority_b.cmp(&priority_a) // Higher priority first
        });
        urls
    }

    /// Convert to download options with multi-source support
    pub fn to_download_options(&self) -> DownloadOptions {
        let mut options = DownloadOptions::default();

        // Set checksum if available
        if let Some(checksum) = self.best_checksum() {
            options.checksum = Some(ChecksumConfig {
                expected: checksum.hash.clone(),
                algorithm: checksum.hash_type.clone(),
            });
        }

        // Set max connections from the first URL that specifies it
        options.max_connections = self.urls.first().and_then(|url| url.max_connections);

        options
    }
}

impl MetalinkUrl {
    /// Check if this URL is preferred over another based on priority and location
    pub fn is_preferred_over(&self, other: &MetalinkUrl, preferred_location: Option<&str>) -> bool {
        // First compare by priority
        let self_priority = self.priority.unwrap_or(0);
        let other_priority = other.priority.unwrap_or(0);

        if self_priority != other_priority {
            return self_priority > other_priority;
        }

        // If priorities are equal, prefer by location
        if let Some(pref_loc) = preferred_location {
            match (&self.location, &other.location) {
                (Some(self_loc), Some(other_loc)) => {
                    if self_loc == pref_loc && other_loc != pref_loc {
                        return true;
                    }
                    if other_loc == pref_loc && self_loc != pref_loc {
                        return false;
                    }
                }
                (Some(self_loc), None) => {
                    if self_loc == pref_loc {
                        return true;
                    }
                }
                (None, Some(other_loc)) => {
                    if other_loc == pref_loc {
                        return false;
                    }
                }
                _ => {}
            }
        }

        // Default to maintaining current order
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_detection() {
        let v3_xml = r#"<?xml version="1.0" encoding="UTF-8"?>
            <metalink version="3.0" xmlns="http://www.metalinker.org/">
            </metalink>"#;

        let v4_xml = r#"<?xml version="1.0" encoding="UTF-8"?>
            <metalink xmlns="urn:ietf:rfc:5854">
            </metalink>"#;

        assert_eq!(
            MetalinkParser::detect_version(v3_xml).unwrap(),
            MetalinkVersion::V3
        );
        assert_eq!(
            MetalinkParser::detect_version(v4_xml).unwrap(),
            MetalinkVersion::V4
        );
    }

    #[test]
    fn test_parse_simple_metalink() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
            <metalink version="3.0" xmlns="http://www.metalinker.org/">
                <file name="example.zip">
                    <resources>
                        <url type="http" preference="100">http://example.com/example.zip</url>
                        <url type="http" preference="90">http://mirror.com/example.zip</url>
                    </resources>
                    <verification>
                        <hash type="sha256">abcdef1234567890</hash>
                    </verification>
                </file>
            </metalink>"#;

        let metalink = MetalinkParser::parse(xml).unwrap();
        assert_eq!(metalink.version, MetalinkVersion::V3);
        assert_eq!(metalink.files.len(), 1);

        let file = &metalink.files[0];
        assert_eq!(file.name, "example.zip");
        assert_eq!(file.urls.len(), 2);
        assert_eq!(file.checksums.len(), 1);

        let sorted_urls = file.sorted_urls();
        assert_eq!(sorted_urls[0].priority, Some(100));
        assert_eq!(sorted_urls[1].priority, Some(90));

        let checksum = file.best_checksum().unwrap();
        assert_eq!(checksum.hash_type, ChecksumAlgorithm::Sha256);
        assert_eq!(checksum.hash, "abcdef1234567890");
    }

    #[tokio::test]
    async fn test_parse_from_file() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
            <metalink version="3.0" xmlns="http://www.metalinker.org/">
                <file name="test.txt">
                    <resources>
                        <url type="http">http://example.com/test.txt</url>
                    </resources>
                </file>
            </metalink>"#;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(xml.as_bytes()).unwrap();

        let metalink = MetalinkParser::parse_file(temp_file.path()).await.unwrap();
        assert_eq!(metalink.files.len(), 1);
        assert_eq!(metalink.files[0].name, "test.txt");
    }
}
    /// Parse a Metalink document from a URL (stub when HTTP feature is disabled)
    #[cfg(not(feature = "http"))]
    pub async fn parse_url(_url: &Url) -> Result<Metalink> {
        Err(ZuupError::Config(
            "HTTP feature is required to parse Metalink from URL. Enable the 'http' feature.".to_string()
        ))
    }