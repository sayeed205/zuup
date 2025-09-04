//! Bandwidth management and throttling

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};
use tokio::time::{Interval, interval, sleep_until};

use crate::error::Result;
use crate::types::DownloadId;

/// Bandwidth limit configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BandwidthLimit {
    /// Maximum bytes per second (0 means unlimited)
    pub bytes_per_second: u64,
}

impl BandwidthLimit {
    /// Create unlimited bandwidth
    pub fn unlimited() -> Self {
        Self {
            bytes_per_second: 0,
        }
    }

    /// Create bandwidth limit in bytes per second
    pub fn bytes_per_second(bytes: u64) -> Self {
        Self {
            bytes_per_second: bytes,
        }
    }

    /// Create bandwidth limit in kilobytes per second
    pub fn kilobytes_per_second(kb: u64) -> Self {
        Self {
            bytes_per_second: kb * 1024,
        }
    }

    /// Create bandwidth limit in megabytes per second
    pub fn megabytes_per_second(mb: u64) -> Self {
        Self {
            bytes_per_second: mb * 1024 * 1024,
        }
    }

    /// Check if this limit is unlimited
    pub fn is_unlimited(&self) -> bool {
        self.bytes_per_second == 0
    }

    /// Get the limit in bytes per second
    pub fn as_bytes_per_second(&self) -> Option<u64> {
        if self.is_unlimited() {
            None
        } else {
            Some(self.bytes_per_second)
        }
    }
}

impl Default for BandwidthLimit {
    fn default() -> Self {
        Self::unlimited()
    }
}

/// Bandwidth usage statistics
#[derive(Debug, Clone)]
pub struct BandwidthStats {
    /// Total bytes transferred
    pub total_bytes: u64,

    /// Current transfer rate in bytes per second
    pub current_rate: u64,

    /// Average transfer rate in bytes per second
    pub average_rate: u64,

    /// Peak transfer rate in bytes per second
    pub peak_rate: u64,

    /// Number of active transfers
    pub active_transfers: u32,

    /// Time when statistics started
    pub start_time: Instant,

    /// Last update time
    pub last_update: Instant,
}

impl BandwidthStats {
    /// Create new bandwidth statistics
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            total_bytes: 0,
            current_rate: 0,
            average_rate: 0,
            peak_rate: 0,
            active_transfers: 0,
            start_time: now,
            last_update: now,
        }
    }

    /// Update statistics with new transfer data
    pub fn update(&mut self, bytes_transferred: u64, active_count: u32) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_update);

        self.total_bytes += bytes_transferred;
        self.active_transfers = active_count;

        if elapsed.as_secs_f64() > 0.0 {
            self.current_rate = (bytes_transferred as f64 / elapsed.as_secs_f64()) as u64;

            if self.current_rate > self.peak_rate {
                self.peak_rate = self.current_rate;
            }

            let total_elapsed = now.duration_since(self.start_time);
            if total_elapsed.as_secs_f64() > 0.0 {
                self.average_rate = (self.total_bytes as f64 / total_elapsed.as_secs_f64()) as u64;
            }
        }

        self.last_update = now;
    }

    /// Get duration since statistics started
    pub fn duration(&self) -> Duration {
        self.last_update.duration_since(self.start_time)
    }
}

impl Default for BandwidthStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Token bucket for rate limiting
#[derive(Debug)]
pub struct TokenBucket {
    /// Maximum number of tokens (burst capacity)
    capacity: u64,

    /// Current number of tokens
    tokens: f64,

    /// Rate at which tokens are added (tokens per second)
    refill_rate: f64,

    /// Last refill time
    last_refill: Instant,
}

impl TokenBucket {
    /// Create a new token bucket
    pub fn new(capacity: u64, refill_rate: u64) -> Self {
        Self {
            capacity,
            tokens: capacity as f64,
            refill_rate: refill_rate as f64,
            last_refill: Instant::now(),
        }
    }

    /// Try to consume tokens from the bucket
    pub fn try_consume(&mut self, tokens: u64) -> bool {
        self.refill();

        if self.tokens >= tokens as f64 {
            self.tokens -= tokens as f64;
            true
        } else {
            false
        }
    }

    /// Get the number of available tokens
    pub fn available_tokens(&mut self) -> u64 {
        self.refill();
        self.tokens as u64
    }

    /// Calculate time until enough tokens are available
    pub fn time_until_available(&mut self, tokens: u64) -> Duration {
        self.refill();

        if self.tokens >= tokens as f64 {
            Duration::ZERO
        } else {
            let needed_tokens = tokens as f64 - self.tokens;
            let time_needed = needed_tokens / self.refill_rate;
            Duration::from_secs_f64(time_needed)
        }
    }

    /// Refill tokens based on elapsed time
    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill);

        let new_tokens = elapsed.as_secs_f64() * self.refill_rate;
        self.tokens = (self.tokens + new_tokens).min(self.capacity as f64);
        self.last_refill = now;
    }
}

/// Per-download bandwidth tracker
#[derive(Debug)]
pub struct DownloadBandwidthTracker {
    /// Download ID
    pub download_id: DownloadId,

    /// Per-download bandwidth limit
    pub limit: BandwidthLimit,

    /// Token bucket for rate limiting
    pub bucket: Option<TokenBucket>,

    /// Bandwidth statistics
    pub stats: BandwidthStats,

    /// Last activity time
    pub last_activity: Instant,
}

impl DownloadBandwidthTracker {
    /// Create a new download bandwidth tracker
    pub fn new(download_id: DownloadId, limit: BandwidthLimit) -> Self {
        let bucket = if limit.is_unlimited() {
            None
        } else {
            // Use burst capacity of 2x the per-second limit
            let capacity = limit.bytes_per_second * 2;
            Some(TokenBucket::new(capacity, limit.bytes_per_second))
        };

        Self {
            download_id,
            limit,
            bucket,
            stats: BandwidthStats::new(),
            last_activity: Instant::now(),
        }
    }

    /// Try to allocate bandwidth for a transfer
    pub async fn try_allocate(&mut self, bytes: u64) -> Result<Duration> {
        self.last_activity = Instant::now();

        if let Some(bucket) = &mut self.bucket {
            if bucket.try_consume(bytes) {
                Ok(Duration::ZERO)
            } else {
                Ok(bucket.time_until_available(bytes))
            }
        } else {
            Ok(Duration::ZERO)
        }
    }

    /// Record bytes transferred
    pub fn record_transfer(&mut self, bytes: u64) {
        self.stats.update(bytes, 1);
        self.last_activity = Instant::now();
    }

    /// Check if this tracker is inactive
    pub fn is_inactive(&self, timeout: Duration) -> bool {
        self.last_activity.elapsed() > timeout
    }

    /// Update bandwidth limit
    pub fn update_limit(&mut self, limit: BandwidthLimit) {
        self.limit = limit;

        if limit.is_unlimited() {
            self.bucket = None;
        } else {
            let capacity = limit.bytes_per_second * 2;
            self.bucket = Some(TokenBucket::new(capacity, limit.bytes_per_second));
        }
    }
}

/// Global bandwidth manager
#[derive(Debug)]
pub struct BandwidthManager {
    /// Global download bandwidth limit
    global_download_limit: Arc<RwLock<BandwidthLimit>>,

    /// Global upload bandwidth limit
    global_upload_limit: Arc<RwLock<BandwidthLimit>>,

    /// Global download token bucket
    global_download_bucket: Arc<Mutex<Option<TokenBucket>>>,

    /// Global upload token bucket
    global_upload_bucket: Arc<Mutex<Option<TokenBucket>>>,

    /// Per-download bandwidth trackers
    download_trackers: Arc<RwLock<HashMap<DownloadId, DownloadBandwidthTracker>>>,

    /// Global bandwidth statistics
    global_stats: Arc<RwLock<BandwidthStats>>,

    /// Statistics update interval
    stats_interval: Arc<Mutex<Interval>>,

    /// Cleanup interval for inactive trackers
    cleanup_interval: Arc<Mutex<Interval>>,
}

impl BandwidthManager {
    /// Create a new bandwidth manager
    pub fn new() -> Self {
        let stats_interval = interval(Duration::from_secs(1));
        let cleanup_interval = interval(Duration::from_secs(60));

        Self {
            global_download_limit: Arc::new(RwLock::new(BandwidthLimit::unlimited())),
            global_upload_limit: Arc::new(RwLock::new(BandwidthLimit::unlimited())),
            global_download_bucket: Arc::new(Mutex::new(None)),
            global_upload_bucket: Arc::new(Mutex::new(None)),
            download_trackers: Arc::new(RwLock::new(HashMap::new())),
            global_stats: Arc::new(RwLock::new(BandwidthStats::new())),
            stats_interval: Arc::new(Mutex::new(stats_interval)),
            cleanup_interval: Arc::new(Mutex::new(cleanup_interval)),
        }
    }

    /// Set global download bandwidth limit
    pub async fn set_global_download_limit(&self, limit: BandwidthLimit) {
        *self.global_download_limit.write().await = limit;

        let mut bucket = self.global_download_bucket.lock().await;
        if limit.is_unlimited() {
            *bucket = None;
        } else {
            let capacity = limit.bytes_per_second * 2;
            *bucket = Some(TokenBucket::new(capacity, limit.bytes_per_second));
        }
    }

    /// Set global upload bandwidth limit
    pub async fn set_global_upload_limit(&self, limit: BandwidthLimit) {
        *self.global_upload_limit.write().await = limit;

        let mut bucket = self.global_upload_bucket.lock().await;
        if limit.is_unlimited() {
            *bucket = None;
        } else {
            let capacity = limit.bytes_per_second * 2;
            *bucket = Some(TokenBucket::new(capacity, limit.bytes_per_second));
        }
    }

    /// Get global download bandwidth limit
    pub async fn global_download_limit(&self) -> BandwidthLimit {
        *self.global_download_limit.read().await
    }

    /// Get global upload bandwidth limit
    pub async fn global_upload_limit(&self) -> BandwidthLimit {
        *self.global_upload_limit.read().await
    }

    /// Set per-download bandwidth limit
    pub async fn set_download_limit(&self, download_id: DownloadId, limit: BandwidthLimit) {
        let mut trackers = self.download_trackers.write().await;

        if let Some(tracker) = trackers.get_mut(&download_id) {
            tracker.update_limit(limit);
        } else {
            let tracker = DownloadBandwidthTracker::new(download_id.clone(), limit);
            trackers.insert(download_id, tracker);
        }
    }

    /// Remove download bandwidth tracker
    pub async fn remove_download(&self, download_id: &DownloadId) {
        let mut trackers = self.download_trackers.write().await;
        trackers.remove(download_id);
    }

    /// Try to allocate bandwidth for a download
    pub async fn try_allocate_download(
        &self,
        download_id: &DownloadId,
        bytes: u64,
    ) -> Result<Duration> {
        // Check global limit first
        let global_delay = {
            let mut bucket = self.global_download_bucket.lock().await;
            if let Some(bucket) = bucket.as_mut() {
                if bucket.try_consume(bytes) {
                    Duration::ZERO
                } else {
                    bucket.time_until_available(bytes)
                }
            } else {
                Duration::ZERO
            }
        };

        // Check per-download limit
        let download_delay = {
            let mut trackers = self.download_trackers.write().await;
            if let Some(tracker) = trackers.get_mut(download_id) {
                tracker.try_allocate(bytes).await?
            } else {
                // Create tracker with unlimited limit if not exists
                let tracker =
                    DownloadBandwidthTracker::new(download_id.clone(), BandwidthLimit::unlimited());
                trackers.insert(download_id.clone(), tracker);
                Duration::ZERO
            }
        };

        // Return the maximum delay
        Ok(global_delay.max(download_delay))
    }

    /// Record bytes transferred for a download
    pub async fn record_download_transfer(&self, download_id: &DownloadId, bytes: u64) {
        let mut trackers = self.download_trackers.write().await;
        if let Some(tracker) = trackers.get_mut(download_id) {
            tracker.record_transfer(bytes);
        }

        // Update global stats
        let mut global_stats = self.global_stats.write().await;
        let active_count = trackers.len() as u32;
        global_stats.update(bytes, active_count);
    }

    /// Get download bandwidth statistics
    pub async fn download_stats(&self, download_id: &DownloadId) -> Option<BandwidthStats> {
        let trackers = self.download_trackers.read().await;
        trackers
            .get(download_id)
            .map(|tracker| tracker.stats.clone())
    }

    /// Get global bandwidth statistics
    pub async fn global_stats(&self) -> BandwidthStats {
        self.global_stats.read().await.clone()
    }

    /// Get all download statistics
    pub async fn all_download_stats(&self) -> HashMap<DownloadId, BandwidthStats> {
        let trackers = self.download_trackers.read().await;
        trackers
            .iter()
            .map(|(id, tracker)| (id.clone(), tracker.stats.clone()))
            .collect()
    }

    /// Start background tasks for statistics and cleanup
    pub async fn start_background_tasks(&self) {
        let manager = Arc::new(self.clone());

        // Statistics update task
        let stats_manager = manager.clone();
        tokio::spawn(async move {
            let mut interval = stats_manager.stats_interval.lock().await;
            loop {
                interval.tick().await;
                // Statistics are updated on each transfer, so nothing to do here
                // This could be used for periodic calculations if needed
            }
        });

        // Cleanup task for inactive trackers
        let cleanup_manager = manager.clone();
        tokio::spawn(async move {
            let mut interval = cleanup_manager.cleanup_interval.lock().await;
            loop {
                interval.tick().await;
                cleanup_manager.cleanup_inactive_trackers().await;
            }
        });
    }

    /// Clean up inactive download trackers
    async fn cleanup_inactive_trackers(&self) {
        let timeout = Duration::from_secs(300); // 5 minutes
        let mut trackers = self.download_trackers.write().await;

        trackers.retain(|_, tracker| !tracker.is_inactive(timeout));
    }

    /// Wait for bandwidth allocation
    pub async fn wait_for_allocation(&self, download_id: &DownloadId, bytes: u64) -> Result<()> {
        let delay = self.try_allocate_download(download_id, bytes).await?;

        if delay > Duration::ZERO {
            sleep_until(tokio::time::Instant::now() + delay).await;
        }

        Ok(())
    }
}

impl Clone for BandwidthManager {
    fn clone(&self) -> Self {
        Self {
            global_download_limit: self.global_download_limit.clone(),
            global_upload_limit: self.global_upload_limit.clone(),
            global_download_bucket: self.global_download_bucket.clone(),
            global_upload_bucket: self.global_upload_bucket.clone(),
            download_trackers: self.download_trackers.clone(),
            global_stats: self.global_stats.clone(),
            stats_interval: self.stats_interval.clone(),
            cleanup_interval: self.cleanup_interval.clone(),
        }
    }
}

impl Default for BandwidthManager {
    fn default() -> Self {
        Self::new()
    }
}
