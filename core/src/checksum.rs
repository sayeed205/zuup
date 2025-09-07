//! Checksum calculation and verification functionality
//!
//! This module provides comprehensive checksum support for download integrity verification,
//! including streaming calculation for large files and multiple hash algorithms.

use std::{collections::HashMap, fmt, path::Path, sync::Arc};

use bytes::Bytes;
use ring::digest::{Context, SHA1_FOR_LEGACY_USE_ONLY, SHA256, SHA512};
use serde::{Deserialize, Serialize};
use tokio::{
    fs::File,
    io::{AsyncRead, AsyncReadExt},
    sync::RwLock,
};
use tracing::{debug, error, info, warn};

use crate::error::{Result, ZuupError};

/// Supported checksum algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChecksumType {
    /// MD5 hash (deprecated, provided for compatibility)
    #[serde(rename = "md5")]
    Md5,
    /// SHA-1 hash (legacy use only)
    #[serde(rename = "sha1")]
    Sha1,
    /// SHA-256 hash (recommended)
    #[serde(rename = "sha256")]
    Sha256,
    /// SHA-512 hash
    #[serde(rename = "sha512")]
    Sha512,
}

impl fmt::Display for ChecksumType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChecksumType::Md5 => write!(f, "MD5"),
            ChecksumType::Sha1 => write!(f, "SHA-1"),
            ChecksumType::Sha256 => write!(f, "SHA-256"),
            ChecksumType::Sha512 => write!(f, "SHA-512"),
        }
    }
}

impl std::str::FromStr for ChecksumType {
    type Err = ZuupError;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "md5" => Ok(ChecksumType::Md5),
            "sha1" | "sha-1" => Ok(ChecksumType::Sha1),
            "sha256" | "sha-256" => Ok(ChecksumType::Sha256),
            "sha512" | "sha-512" => Ok(ChecksumType::Sha512),
            _ => Err(ZuupError::Config(format!(
                "Unsupported checksum type: {}",
                s
            ))),
        }
    }
}

/// Configuration for checksum verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChecksumConfig {
    /// Expected checksum value (hex-encoded)
    pub expected: String,
    /// Checksum algorithm to use
    pub algorithm: ChecksumType,
    /// Whether to verify during download (streaming)
    pub verify_during_download: bool,
    /// Whether to verify after download completion
    pub verify_after_download: bool,
}

impl ChecksumConfig {
    /// Create a new checksum configuration
    pub fn new(expected: String, algorithm: ChecksumType) -> Self {
        Self {
            expected,
            algorithm,
            verify_during_download: true,
            verify_after_download: true,
        }
    }

    /// Create MD5 checksum configuration
    pub fn md5(expected: String) -> Self {
        Self::new(expected, ChecksumType::Md5)
    }

    /// Create SHA-1 checksum configuration
    pub fn sha1(expected: String) -> Self {
        Self::new(expected, ChecksumType::Sha1)
    }

    /// Create SHA-256 checksum configuration
    pub fn sha256(expected: String) -> Self {
        Self::new(expected, ChecksumType::Sha256)
    }

    /// Create SHA-512 checksum configuration
    pub fn sha512(expected: String) -> Self {
        Self::new(expected, ChecksumType::Sha512)
    }
}

/// Result of checksum calculation
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChecksumResult {
    /// The calculated checksum (hex-encoded)
    pub calculated: String,
    /// The checksum algorithm used
    pub algorithm: ChecksumType,
    /// Whether the checksum matches the expected value (if provided)
    pub verified: Option<bool>,
    /// Expected checksum value (if provided)
    pub expected: Option<String>,
}

impl ChecksumResult {
    /// Create a new checksum result
    pub fn new(calculated: String, algorithm: ChecksumType) -> Self {
        Self {
            calculated,
            algorithm,
            verified: None,
            expected: None,
        }
    }

    /// Create a verified checksum result
    pub fn verified(calculated: String, algorithm: ChecksumType, expected: String) -> Self {
        let verified = calculated.eq_ignore_ascii_case(&expected);
        Self {
            calculated,
            algorithm,
            verified: Some(verified),
            expected: Some(expected),
        }
    }

    /// Check if the checksum is verified and matches
    pub fn is_valid(&self) -> bool {
        self.verified.unwrap_or(true)
    }
}

impl fmt::Display for ChecksumResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.algorithm, self.calculated)?;
        if let Some(expected) = &self.expected {
            write!(f, " (expected: {})", expected)?;
        }
        if let Some(verified) = self.verified {
            write!(f, " [{}]", if verified { "VALID" } else { "INVALID" })?;
        }
        Ok(())
    }
}

/// Streaming hasher for incremental checksum calculation
pub struct StreamingHasher {
    md5_context: Option<md5::Context>,
    sha1_context: Option<Context>,
    sha256_context: Option<Context>,
    sha512_context: Option<Context>,
    algorithms: Vec<ChecksumType>,
}

impl StreamingHasher {
    /// Create a new streaming hasher for the specified algorithms
    pub fn new(algorithms: &[ChecksumType]) -> Self {
        let mut hasher = Self {
            md5_context: None,
            sha1_context: None,
            sha256_context: None,
            sha512_context: None,
            algorithms: algorithms.to_vec(),
        };

        for &algorithm in algorithms {
            match algorithm {
                ChecksumType::Md5 => {
                    hasher.md5_context = Some(md5::Context::new());
                }
                ChecksumType::Sha1 => {
                    hasher.sha1_context = Some(Context::new(&SHA1_FOR_LEGACY_USE_ONLY));
                }
                ChecksumType::Sha256 => {
                    hasher.sha256_context = Some(Context::new(&SHA256));
                }
                ChecksumType::Sha512 => {
                    hasher.sha512_context = Some(Context::new(&SHA512));
                }
            }
        }

        hasher
    }

    /// Create a streaming hasher for a single algorithm
    pub fn single(algorithm: ChecksumType) -> Self {
        Self::new(&[algorithm])
    }

    /// Update the hasher with new data
    pub fn update(&mut self, data: &[u8]) {
        if let Some(ref mut ctx) = self.md5_context {
            ctx.consume(data);
        }
        if let Some(ref mut ctx) = self.sha1_context {
            ctx.update(data);
        }
        if let Some(ref mut ctx) = self.sha256_context {
            ctx.update(data);
        }
        if let Some(ref mut ctx) = self.sha512_context {
            ctx.update(data);
        }
    }

    /// Finalize the hash calculation and return results
    pub fn finalize(self) -> HashMap<ChecksumType, ChecksumResult> {
        let mut results = HashMap::new();

        if let Some(ctx) = self.md5_context {
            let digest = ctx.finalize();
            let hex = hex::encode(digest.0);
            results.insert(
                ChecksumType::Md5,
                ChecksumResult::new(hex, ChecksumType::Md5),
            );
        }

        if let Some(ctx) = self.sha1_context {
            let digest = ctx.finish();
            let hex = hex::encode(digest.as_ref());
            results.insert(
                ChecksumType::Sha1,
                ChecksumResult::new(hex, ChecksumType::Sha1),
            );
        }

        if let Some(ctx) = self.sha256_context {
            let digest = ctx.finish();
            let hex = hex::encode(digest.as_ref());
            results.insert(
                ChecksumType::Sha256,
                ChecksumResult::new(hex, ChecksumType::Sha256),
            );
        }

        if let Some(ctx) = self.sha512_context {
            let digest = ctx.finish();
            let hex = hex::encode(digest.as_ref());
            results.insert(
                ChecksumType::Sha512,
                ChecksumResult::new(hex, ChecksumType::Sha512),
            );
        }

        results
    }

    /// Get the algorithms being calculated
    pub fn algorithms(&self) -> &[ChecksumType] {
        &self.algorithms
    }
}

/// Checksum calculator for files and data
pub struct ChecksumCalculator;

impl ChecksumCalculator {
    /// Calculate checksum for a byte slice
    pub fn calculate_bytes(data: &[u8], algorithm: ChecksumType) -> Result<ChecksumResult> {
        let mut hasher = StreamingHasher::single(algorithm);
        hasher.update(data);
        let results = hasher.finalize();

        results
            .get(&algorithm)
            .cloned()
            .ok_or_else(|| ZuupError::Config("Failed to calculate checksum".to_string()))
    }

    /// Calculate checksum for a Bytes object
    pub fn calculate_bytes_obj(data: &Bytes, algorithm: ChecksumType) -> Result<ChecksumResult> {
        Self::calculate_bytes(data, algorithm)
    }

    /// Calculate checksum for a file asynchronously
    pub async fn calculate_file<P: AsRef<Path>>(
        path: P,
        algorithm: ChecksumType,
    ) -> Result<ChecksumResult> {
        let mut file = File::open(path).await.map_err(ZuupError::Io)?;

        Self::calculate_async_reader(&mut file, algorithm).await
    }

    /// Calculate checksum for an async reader
    pub async fn calculate_async_reader<R: AsyncRead + Unpin>(
        reader: &mut R,
        algorithm: ChecksumType,
    ) -> Result<ChecksumResult> {
        let mut hasher = StreamingHasher::single(algorithm);
        let mut buffer = vec![0u8; 8192]; // 8KB buffer

        loop {
            let bytes_read = reader.read(&mut buffer).await.map_err(ZuupError::Io)?;

            if bytes_read == 0 {
                break;
            }

            hasher.update(&buffer[..bytes_read]);
        }

        let results = hasher.finalize();
        results
            .get(&algorithm)
            .cloned()
            .ok_or_else(|| ZuupError::Config("Failed to calculate checksum".to_string()))
    }

    /// Calculate multiple checksums for a file
    pub async fn calculate_file_multiple<P: AsRef<Path>>(
        path: P,
        algorithms: &[ChecksumType],
    ) -> Result<HashMap<ChecksumType, ChecksumResult>> {
        let mut file = File::open(path).await.map_err(ZuupError::Io)?;

        Self::calculate_async_reader_multiple(&mut file, algorithms).await
    }

    /// Calculate multiple checksums for an async reader
    pub async fn calculate_async_reader_multiple<R: AsyncRead + Unpin>(
        reader: &mut R,
        algorithms: &[ChecksumType],
    ) -> Result<HashMap<ChecksumType, ChecksumResult>> {
        let mut hasher = StreamingHasher::new(algorithms);
        let mut buffer = vec![0u8; 8192]; // 8KB buffer

        loop {
            let bytes_read = reader.read(&mut buffer).await.map_err(ZuupError::Io)?;

            if bytes_read == 0 {
                break;
            }

            hasher.update(&buffer[..bytes_read]);
        }

        Ok(hasher.finalize())
    }
}

/// Checksum verifier for integrity checking
pub struct ChecksumVerifier;

impl ChecksumVerifier {
    /// Verify checksum for a byte slice
    pub fn verify_bytes(data: &[u8], config: &ChecksumConfig) -> Result<ChecksumResult> {
        let calculated = ChecksumCalculator::calculate_bytes(data, config.algorithm)?;
        Ok(ChecksumResult::verified(
            calculated.calculated,
            config.algorithm,
            config.expected.clone(),
        ))
    }

    /// Verify checksum for a Bytes object
    pub fn verify_bytes_obj(data: &Bytes, config: &ChecksumConfig) -> Result<ChecksumResult> {
        Self::verify_bytes(data, config)
    }

    /// Verify checksum for a file
    pub async fn verify_file<P: AsRef<Path>>(
        path: P,
        config: &ChecksumConfig,
    ) -> Result<ChecksumResult> {
        let calculated = ChecksumCalculator::calculate_file(path, config.algorithm).await?;
        Ok(ChecksumResult::verified(
            calculated.calculated,
            config.algorithm,
            config.expected.clone(),
        ))
    }

    /// Verify multiple checksums for a file
    pub async fn verify_file_multiple<P: AsRef<Path>>(
        path: P,
        configs: &[ChecksumConfig],
    ) -> Result<HashMap<ChecksumType, ChecksumResult>> {
        let algorithms: Vec<ChecksumType> = configs.iter().map(|c| c.algorithm).collect();
        let calculated = ChecksumCalculator::calculate_file_multiple(path, &algorithms).await?;

        let mut results = HashMap::new();
        for config in configs {
            if let Some(calc_result) = calculated.get(&config.algorithm) {
                let verified_result = ChecksumResult::verified(
                    calc_result.calculated.clone(),
                    config.algorithm,
                    config.expected.clone(),
                );
                results.insert(config.algorithm, verified_result);
            }
        }

        Ok(results)
    }

    /// Check if verification passed
    pub fn is_verification_successful(results: &HashMap<ChecksumType, ChecksumResult>) -> bool {
        results.values().all(|result| result.is_valid())
    }

    /// Log verification results
    pub fn log_verification_results(results: &HashMap<ChecksumType, ChecksumResult>) {
        for result in results.values() {
            match result.verified {
                Some(true) => {
                    info!("Checksum verification passed: {}", result);
                }
                Some(false) => {
                    warn!("Checksum verification failed: {}", result);
                }
                None => {
                    debug!("Checksum calculated: {}", result);
                }
            }
        }
    }
}

/// Information about a chunk for integrity checking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkInfo {
    /// Chunk index/identifier
    pub index: u64,
    /// Start offset in the file
    pub start_offset: u64,
    /// End offset in the file
    pub end_offset: u64,
    /// Size of the chunk
    pub size: u64,
    /// Expected checksum for this chunk
    pub expected_checksum: Option<ChecksumConfig>,
}

impl ChunkInfo {
    /// Create a new chunk info
    pub fn new(index: u64, start_offset: u64, end_offset: u64) -> Self {
        Self {
            index,
            start_offset,
            end_offset,
            size: end_offset - start_offset,
            expected_checksum: None,
        }
    }

    /// Create a chunk info with expected checksum
    pub fn with_checksum(
        index: u64,
        start_offset: u64,
        end_offset: u64,
        expected_checksum: ChecksumConfig,
    ) -> Self {
        Self {
            index,
            start_offset,
            end_offset,
            size: end_offset - start_offset,
            expected_checksum: Some(expected_checksum),
        }
    }

    /// Check if this chunk has an expected checksum
    pub fn has_checksum(&self) -> bool {
        self.expected_checksum.is_some()
    }
}

/// Result of chunk integrity verification
#[derive(Debug, Clone)]
pub struct ChunkVerificationResult {
    /// Chunk information
    pub chunk: ChunkInfo,
    /// Checksum verification result
    pub checksum_result: Option<ChecksumResult>,
    /// Whether the chunk is valid
    pub is_valid: bool,
    /// Error if verification failed
    pub error: Option<String>,
}

impl ChunkVerificationResult {
    /// Create a successful verification result
    pub fn success(chunk: ChunkInfo, checksum_result: ChecksumResult) -> Self {
        let is_valid = checksum_result.is_valid();
        Self {
            chunk,
            checksum_result: Some(checksum_result),
            is_valid,
            error: None,
        }
    }

    /// Create a failed verification result
    pub fn failure(chunk: ChunkInfo, error: String) -> Self {
        Self {
            chunk,
            checksum_result: None,
            is_valid: false,
            error: Some(error),
        }
    }

    /// Create a result for chunk without checksum
    pub fn no_checksum(chunk: ChunkInfo) -> Self {
        Self {
            chunk,
            checksum_result: None,
            is_valid: true, // Assume valid if no checksum to verify
            error: None,
        }
    }
}

/// Statistics for chunk verification
#[derive(Debug, Clone, Default)]
pub struct ChunkVerificationStats {
    /// Total number of chunks
    pub total_chunks: u64,
    /// Number of chunks verified successfully
    pub verified_chunks: u64,
    /// Number of chunks that failed verification
    pub failed_chunks: u64,
    /// Number of chunks without checksums
    pub no_checksum_chunks: u64,
    /// Number of chunks that need re-download
    pub corrupted_chunks: u64,
}

impl ChunkVerificationStats {
    /// Update stats with a verification result
    pub fn update(&mut self, result: &ChunkVerificationResult) {
        self.total_chunks += 1;

        if result.checksum_result.is_none() {
            self.no_checksum_chunks += 1;
        } else if result.is_valid {
            self.verified_chunks += 1;
        } else {
            self.failed_chunks += 1;
            self.corrupted_chunks += 1;
        }
    }

    /// Get verification success rate (0.0 to 1.0)
    pub fn success_rate(&self) -> f64 {
        if self.total_chunks == 0 {
            return 1.0;
        }
        (self.verified_chunks + self.no_checksum_chunks) as f64 / self.total_chunks as f64
    }

    /// Check if all chunks are valid
    pub fn all_valid(&self) -> bool {
        self.failed_chunks == 0
    }
}

/// Chunk integrity checker for download verification
pub struct ChunkIntegrityChecker {
    /// Verification results by chunk index
    results: Arc<RwLock<HashMap<u64, ChunkVerificationResult>>>,
    /// Verification statistics
    stats: Arc<RwLock<ChunkVerificationStats>>,
}

impl ChunkIntegrityChecker {
    /// Create a new chunk integrity checker
    pub fn new() -> Self {
        Self {
            results: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(ChunkVerificationStats::default())),
        }
    }

    /// Verify a chunk's integrity
    pub async fn verify_chunk(
        &self,
        chunk: ChunkInfo,
        data: &[u8],
    ) -> Result<ChunkVerificationResult> {
        let result = if let Some(ref checksum_config) = chunk.expected_checksum {
            // Verify chunk with expected checksum
            match ChecksumVerifier::verify_bytes(data, checksum_config) {
                Ok(checksum_result) => {
                    if checksum_result.is_valid() {
                        debug!(
                            "Chunk {} verification passed: {}",
                            chunk.index, checksum_result
                        );
                        ChunkVerificationResult::success(chunk, checksum_result)
                    } else {
                        warn!(
                            "Chunk {} verification failed: expected {}, got {}",
                            chunk.index, checksum_config.expected, checksum_result.calculated
                        );
                        ChunkVerificationResult::success(chunk, checksum_result)
                    }
                }
                Err(e) => {
                    error!("Failed to verify chunk {}: {}", chunk.index, e);
                    ChunkVerificationResult::failure(chunk, e.to_string())
                }
            }
        } else {
            // No checksum to verify
            debug!("Chunk {} has no checksum to verify", chunk.index);
            ChunkVerificationResult::no_checksum(chunk)
        };

        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.update(&result);
        }

        // Store result
        {
            let mut results = self.results.write().await;
            results.insert(result.chunk.index, result.clone());
        }

        Ok(result)
    }

    /// Verify multiple chunks
    pub async fn verify_chunks(
        &self,
        chunks_data: Vec<(ChunkInfo, Vec<u8>)>,
    ) -> Result<Vec<ChunkVerificationResult>> {
        let mut results = Vec::new();

        for (chunk, data) in chunks_data {
            let result = self.verify_chunk(chunk, &data).await?;
            results.push(result);
        }

        Ok(results)
    }

    /// Get verification result for a specific chunk
    pub async fn get_chunk_result(&self, chunk_index: u64) -> Option<ChunkVerificationResult> {
        let results = self.results.read().await;
        results.get(&chunk_index).cloned()
    }

    /// Get all verification results
    pub async fn get_all_results(&self) -> HashMap<u64, ChunkVerificationResult> {
        let results = self.results.read().await;
        results.clone()
    }

    /// Get verification statistics
    pub async fn get_stats(&self) -> ChunkVerificationStats {
        let stats = self.stats.read().await;
        stats.clone()
    }

    /// Get list of corrupted chunks that need re-download
    pub async fn get_corrupted_chunks(&self) -> Vec<ChunkInfo> {
        let results = self.results.read().await;
        results
            .values()
            .filter(|result| !result.is_valid)
            .map(|result| result.chunk.clone())
            .collect()
    }

    /// Check if all chunks are valid
    pub async fn all_chunks_valid(&self) -> bool {
        let stats = self.stats.read().await;
        stats.all_valid()
    }

    /// Reset verification state
    pub async fn reset(&self) {
        {
            let mut results = self.results.write().await;
            results.clear();
        }
        {
            let mut stats = self.stats.write().await;
            *stats = ChunkVerificationStats::default();
        }
    }

    /// Log verification summary
    pub async fn log_verification_summary(&self) {
        let stats = self.stats.read().await;

        info!(
            "Chunk verification summary: {} total, {} verified, {} failed, {} no checksum, success rate: {:.2}%",
            stats.total_chunks,
            stats.verified_chunks,
            stats.failed_chunks,
            stats.no_checksum_chunks,
            stats.success_rate() * 100.0
        );

        if stats.corrupted_chunks > 0 {
            warn!(
                "{} chunks need re-download due to corruption",
                stats.corrupted_chunks
            );
        }
    }
}

impl Default for ChunkIntegrityChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for chunk management
pub struct ChunkUtils;

impl ChunkUtils {
    /// Split a file size into chunks of specified size
    pub fn create_chunks(file_size: u64, chunk_size: u64) -> Vec<ChunkInfo> {
        let mut chunks = Vec::new();
        let mut offset = 0;
        let mut index = 0;

        while offset < file_size {
            let end_offset = std::cmp::min(offset + chunk_size, file_size);
            chunks.push(ChunkInfo::new(index, offset, end_offset));
            offset = end_offset;
            index += 1;
        }

        chunks
    }

    /// Create chunks with checksums from a list of expected checksums
    pub fn create_chunks_with_checksums(
        file_size: u64,
        chunk_size: u64,
        checksums: Vec<ChecksumConfig>,
    ) -> Result<Vec<ChunkInfo>> {
        let mut chunks = Self::create_chunks(file_size, chunk_size);

        if chunks.len() != checksums.len() {
            return Err(ZuupError::Config(format!(
                "Chunk count mismatch: {} chunks but {} checksums",
                chunks.len(),
                checksums.len()
            )));
        }

        for (chunk, checksum) in chunks.iter_mut().zip(checksums.into_iter()) {
            chunk.expected_checksum = Some(checksum);
        }

        Ok(chunks)
    }

    /// Verify chunk data matches expected size
    pub fn verify_chunk_size(chunk: &ChunkInfo, data: &[u8]) -> Result<()> {
        if data.len() as u64 != chunk.size {
            return Err(ZuupError::Config(format!(
                "Chunk {} size mismatch: expected {} bytes, got {} bytes",
                chunk.index,
                chunk.size,
                data.len()
            )));
        }
        Ok(())
    }

    /// Calculate optimal chunk size based on file size and connection count
    pub fn calculate_optimal_chunk_size(file_size: u64, connection_count: u32) -> u64 {
        const MIN_CHUNK_SIZE: u64 = 1024 * 1024; // 1MB minimum
        const MAX_CHUNK_SIZE: u64 = 10 * 1024 * 1024; // 10MB maximum

        if file_size == 0 || connection_count == 0 {
            return MIN_CHUNK_SIZE;
        }

        let calculated_size = file_size / connection_count as u64;
        calculated_size.clamp(MIN_CHUNK_SIZE, MAX_CHUNK_SIZE)
    }
}
