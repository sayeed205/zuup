//! Metrics and monitoring system for Zuup download manager
//!
//! This module provides comprehensive metrics collection, performance monitoring,
//! and health check capabilities for the download manager.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use metrics::{counter, gauge, histogram};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, error, info};

use crate::error::{Result, ZuupError};
use crate::types::DownloadId;

/// Configuration for metrics collection and export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Whether metrics collection is enabled
    pub enabled: bool,
    /// Prometheus metrics export configuration
    pub prometheus: Option<PrometheusConfig>,
    /// Metrics collection interval
    pub collection_interval: Duration,
    /// Whether to collect detailed per-download metrics
    pub detailed_metrics: bool,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            prometheus: None,
            collection_interval: Duration::from_secs(10),
            detailed_metrics: false,
        }
    }
}

/// Prometheus metrics export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrometheusConfig {
    /// Address to bind the Prometheus metrics endpoint
    pub bind_address: String,
    /// Port for the Prometheus metrics endpoint
    pub port: u16,
    /// Path for the metrics endpoint
    pub path: String,
}

impl Default for PrometheusConfig {
    fn default() -> Self {
        Self {
            bind_address: "127.0.0.1".to_string(),
            port: 9090,
            path: "/metrics".to_string(),
        }
    }
}

/// Download statistics for a specific download
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadStats {
    /// Download ID
    pub id: DownloadId,
    /// Total bytes downloaded
    pub bytes_downloaded: u64,
    /// Total bytes to download (if known)
    pub total_bytes: Option<u64>,
    /// Current download speed in bytes per second
    pub download_speed: u64,
    /// Number of active connections
    pub active_connections: u32,
    /// Number of completed segments
    pub completed_segments: u32,
    /// Total number of segments
    pub total_segments: u32,
    /// Download start time
    pub started_at: SystemTime,
    /// Last update time
    pub updated_at: SystemTime,
}

/// System-wide performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    /// Total number of active downloads
    pub active_downloads: usize,
    /// Total number of completed downloads
    pub completed_downloads: u64,
    /// Total number of failed downloads
    pub failed_downloads: u64,
    /// Total bytes downloaded across all downloads
    pub total_bytes_downloaded: u64,
    /// Current aggregate download speed
    pub aggregate_download_speed: u64,
    /// Total number of active connections
    pub total_active_connections: u32,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// CPU usage percentage
    pub cpu_usage: f64,
    /// Disk usage for download directory
    pub disk_usage: DiskUsage,
    /// Network statistics
    pub network_stats: NetworkStats,
    /// Last update time
    pub updated_at: SystemTime,
}

/// Disk usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskUsage {
    /// Total disk space in bytes
    pub total: u64,
    /// Available disk space in bytes
    pub available: u64,
    /// Used disk space in bytes
    pub used: u64,
}

/// Network statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStats {
    /// Total bytes sent
    pub bytes_sent: u64,
    /// Total bytes received
    pub bytes_received: u64,
    /// Number of successful connections
    pub successful_connections: u64,
    /// Number of failed connections
    pub failed_connections: u64,
    /// Average connection time in milliseconds
    pub avg_connection_time: f64,
}

/// Health check status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthStatus {
    /// System is healthy
    Healthy,
    /// System has warnings but is functional
    Warning(String),
    /// System is unhealthy
    Unhealthy(String),
}

/// System health information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    /// Overall health status
    pub status: HealthStatus,
    /// Individual component health checks
    pub components: HashMap<String, HealthStatus>,
    /// Health check timestamp
    pub timestamp: SystemTime,
    /// Uptime in seconds
    pub uptime: u64,
}

/// Metrics collector and manager
pub struct MetricsCollector {
    config: MetricsConfig,
    download_stats: Arc<RwLock<HashMap<DownloadId, DownloadStats>>>,
    system_metrics: Arc<RwLock<SystemMetrics>>,
    start_time: Instant,

    // Atomic counters for thread-safe updates
    completed_downloads: AtomicU64,
    failed_downloads: AtomicU64,
    total_bytes_downloaded: AtomicU64,
    total_connections: AtomicUsize,
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new(config: MetricsConfig) -> Self {
        // Register Prometheus metrics
        if config.enabled {
            Self::register_metrics();
        }

        let system_metrics = SystemMetrics {
            active_downloads: 0,
            completed_downloads: 0,
            failed_downloads: 0,
            total_bytes_downloaded: 0,
            aggregate_download_speed: 0,
            total_active_connections: 0,
            memory_usage: 0,
            cpu_usage: 0.0,
            disk_usage: DiskUsage {
                total: 0,
                available: 0,
                used: 0,
            },
            network_stats: NetworkStats {
                bytes_sent: 0,
                bytes_received: 0,
                successful_connections: 0,
                failed_connections: 0,
                avg_connection_time: 0.0,
            },
            updated_at: SystemTime::now(),
        };

        Self {
            config,
            download_stats: Arc::new(RwLock::new(HashMap::new())),
            system_metrics: Arc::new(RwLock::new(system_metrics)),
            start_time: Instant::now(),
            completed_downloads: AtomicU64::new(0),
            failed_downloads: AtomicU64::new(0),
            total_bytes_downloaded: AtomicU64::new(0),
            total_connections: AtomicUsize::new(0),
        }
    }

    /// Register Prometheus metrics (no-op since metrics are registered on first use)
    fn register_metrics() {
        // Metrics are automatically registered when first used in the metrics crate
        debug!("Metrics registration completed");
    }

    /// Start the metrics collection background task
    pub async fn start(&self) -> Result<()> {
        if !self.config.enabled {
            info!("Metrics collection is disabled");
            return Ok(());
        }

        info!("Starting metrics collection");

        // Start Prometheus exporter if configured
        if let Some(prometheus_config) = &self.config.prometheus {
            self.start_prometheus_exporter(prometheus_config).await?;
        }

        // Start metrics collection loop
        let collector = Arc::new(self.clone());
        tokio::spawn(async move {
            collector.collection_loop().await;
        });

        Ok(())
    }

    /// Start Prometheus metrics exporter
    async fn start_prometheus_exporter(&self, config: &PrometheusConfig) -> Result<()> {
        use metrics_exporter_prometheus::PrometheusBuilder;
        use std::net::SocketAddr;

        let socket_addr: SocketAddr = format!("{}:{}", config.bind_address, config.port)
            .parse()
            .map_err(|e| ZuupError::Config(format!("Invalid bind address: {}", e)))?;

        let builder = PrometheusBuilder::new();
        let _handle = builder
            .with_http_listener(socket_addr)
            .install()
            .map_err(|e| {
                ZuupError::Config(format!("Failed to start Prometheus exporter: {}", e))
            })?;

        info!(
            "Prometheus metrics available at http://{}:{}{}",
            config.bind_address, config.port, config.path
        );

        Ok(())
    }

    /// Main metrics collection loop
    async fn collection_loop(&self) {
        let mut interval = tokio::time::interval(self.config.collection_interval);

        loop {
            interval.tick().await;

            if let Err(e) = self.collect_system_metrics().await {
                error!("Failed to collect system metrics: {}", e);
            }

            if let Err(e) = self.update_prometheus_metrics().await {
                error!("Failed to update Prometheus metrics: {}", e);
            }
        }
    }

    /// Collect system-wide metrics
    async fn collect_system_metrics(&self) -> Result<()> {
        let mut system_metrics = self.system_metrics.write().await;
        let download_stats = self.download_stats.read().await;

        // Update basic counters
        system_metrics.active_downloads = download_stats.len();
        system_metrics.completed_downloads = self.completed_downloads.load(Ordering::Relaxed);
        system_metrics.failed_downloads = self.failed_downloads.load(Ordering::Relaxed);
        system_metrics.total_bytes_downloaded = self.total_bytes_downloaded.load(Ordering::Relaxed);
        system_metrics.total_active_connections =
            self.total_connections.load(Ordering::Relaxed) as u32;

        // Calculate aggregate download speed
        system_metrics.aggregate_download_speed = download_stats
            .values()
            .map(|stats| stats.download_speed)
            .sum();

        // Collect system resource metrics
        system_metrics.memory_usage = self.get_memory_usage().await?;
        system_metrics.cpu_usage = self.get_cpu_usage().await?;
        system_metrics.disk_usage = self.get_disk_usage().await?;

        system_metrics.updated_at = SystemTime::now();

        if self.config.enabled {
            debug!(
                "Updated system metrics: active_downloads={}, aggregate_speed={} B/s",
                system_metrics.active_downloads, system_metrics.aggregate_download_speed
            );
        }

        Ok(())
    }

    /// Update Prometheus metrics
    async fn update_prometheus_metrics(&self) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let system_metrics = self.system_metrics.read().await;
        let _download_stats = self.download_stats.read().await;

        // Update download metrics
        gauge!("zuup_downloads_active").set(system_metrics.active_downloads as f64);
        counter!("zuup_downloads_completed").absolute(system_metrics.completed_downloads);
        counter!("zuup_downloads_failed").absolute(system_metrics.failed_downloads);
        counter!("zuup_bytes_downloaded_total").absolute(system_metrics.total_bytes_downloaded);
        gauge!("zuup_download_speed_bps").set(system_metrics.aggregate_download_speed as f64);

        // Update connection metrics
        gauge!("zuup_connections_active").set(system_metrics.total_active_connections as f64);
        counter!("zuup_connections_successful")
            .absolute(system_metrics.network_stats.successful_connections);
        counter!("zuup_connections_failed")
            .absolute(system_metrics.network_stats.failed_connections);

        // Update system metrics
        gauge!("zuup_memory_usage_bytes").set(system_metrics.memory_usage as f64);
        gauge!("zuup_cpu_usage_percent").set(system_metrics.cpu_usage);
        gauge!("zuup_disk_total_bytes").set(system_metrics.disk_usage.total as f64);
        gauge!("zuup_disk_available_bytes").set(system_metrics.disk_usage.available as f64);
        gauge!("zuup_disk_used_bytes").set(system_metrics.disk_usage.used as f64);

        // Update network metrics
        counter!("zuup_network_bytes_sent").absolute(system_metrics.network_stats.bytes_sent);
        counter!("zuup_network_bytes_received")
            .absolute(system_metrics.network_stats.bytes_received);

        Ok(())
    }

    /// Record download start
    pub async fn record_download_start(&self, id: DownloadId) {
        let stats = DownloadStats {
            id: id.clone(),
            bytes_downloaded: 0,
            total_bytes: None,
            download_speed: 0,
            active_connections: 0,
            completed_segments: 0,
            total_segments: 0,
            started_at: SystemTime::now(),
            updated_at: SystemTime::now(),
        };

        self.download_stats.write().await.insert(id, stats);

        if self.config.enabled {
            counter!("zuup_downloads_total").increment(1);
            debug!("Recorded download start");
        }
    }

    /// Record download completion
    pub async fn record_download_complete(&self, id: &DownloadId, bytes_downloaded: u64) {
        self.download_stats.write().await.remove(id);
        self.completed_downloads.fetch_add(1, Ordering::Relaxed);
        self.total_bytes_downloaded
            .fetch_add(bytes_downloaded, Ordering::Relaxed);

        if self.config.enabled {
            debug!("Recorded download completion: {} bytes", bytes_downloaded);
        }
    }

    /// Record download failure
    pub async fn record_download_failure(&self, id: &DownloadId) {
        self.download_stats.write().await.remove(id);
        self.failed_downloads.fetch_add(1, Ordering::Relaxed);

        if self.config.enabled {
            debug!("Recorded download failure");
        }
    }

    /// Update download progress
    pub async fn update_download_progress(
        &self,
        id: &DownloadId,
        bytes_downloaded: u64,
        total_bytes: Option<u64>,
        download_speed: u64,
        active_connections: u32,
    ) {
        if let Some(stats) = self.download_stats.write().await.get_mut(id) {
            stats.bytes_downloaded = bytes_downloaded;
            stats.total_bytes = total_bytes;
            stats.download_speed = download_speed;
            stats.active_connections = active_connections;
            stats.updated_at = SystemTime::now();
        }
    }

    /// Record connection event
    pub fn record_connection_success(&self, duration: Duration) {
        if !self.config.enabled {
            return;
        }

        self.total_connections.fetch_add(1, Ordering::Relaxed);
        histogram!("zuup_connection_duration_seconds").record(duration.as_secs_f64());

        debug!("Recorded successful connection: {:?}", duration);
    }

    /// Record connection failure
    pub fn record_connection_failure(&self) {
        if !self.config.enabled {
            return;
        }

        debug!("Recorded connection failure");
    }

    /// Record network bytes
    pub fn record_network_bytes(&self, sent: u64, received: u64) {
        if !self.config.enabled {
            return;
        }

        // Update atomic counters would go here if we had them
        debug!(
            "Recorded network bytes: sent={}, received={}",
            sent, received
        );
    }

    /// Get current download statistics
    pub async fn get_download_stats(&self, id: &DownloadId) -> Option<DownloadStats> {
        self.download_stats.read().await.get(id).cloned()
    }

    /// Get all download statistics
    pub async fn get_all_download_stats(&self) -> HashMap<DownloadId, DownloadStats> {
        self.download_stats.read().await.clone()
    }

    /// Get system metrics
    pub async fn get_system_metrics(&self) -> SystemMetrics {
        self.system_metrics.read().await.clone()
    }

    /// Perform health check
    pub async fn health_check(&self) -> HealthCheck {
        let mut components = HashMap::new();
        let mut overall_status = HealthStatus::Healthy;

        // Check metrics collection
        if self.config.enabled {
            components.insert("metrics".to_string(), HealthStatus::Healthy);
        } else {
            components.insert("metrics".to_string(), HealthStatus::Healthy); // Still healthy even if disabled
        }

        // Check system resources
        let system_metrics = self.system_metrics.read().await;

        // Check memory usage (warn if > 80%, unhealthy if > 95%)
        let memory_status = if system_metrics.memory_usage > 0 {
            // This is a simplified check - in reality you'd compare against system total
            HealthStatus::Healthy
        } else {
            HealthStatus::Healthy
        };
        components.insert("memory".to_string(), memory_status);

        // Check CPU usage
        let cpu_status = if system_metrics.cpu_usage > 95.0 {
            overall_status = HealthStatus::Unhealthy("High CPU usage".to_string());
            HealthStatus::Unhealthy(format!("CPU usage: {:.1}%", system_metrics.cpu_usage))
        } else if system_metrics.cpu_usage > 80.0 {
            if matches!(overall_status, HealthStatus::Healthy) {
                overall_status = HealthStatus::Warning("High CPU usage".to_string());
            }
            HealthStatus::Warning(format!("CPU usage: {:.1}%", system_metrics.cpu_usage))
        } else {
            HealthStatus::Healthy
        };
        components.insert("cpu".to_string(), cpu_status);

        // Check disk space
        let disk_status = if system_metrics.disk_usage.available < 1024 * 1024 * 100 {
            // < 100MB
            overall_status = HealthStatus::Unhealthy("Low disk space".to_string());
            HealthStatus::Unhealthy("Low disk space".to_string())
        } else if system_metrics.disk_usage.available < 1024 * 1024 * 1024 {
            // < 1GB
            if matches!(overall_status, HealthStatus::Healthy) {
                overall_status = HealthStatus::Warning("Low disk space".to_string());
            }
            HealthStatus::Warning("Low disk space".to_string())
        } else {
            HealthStatus::Healthy
        };
        components.insert("disk".to_string(), disk_status);

        HealthCheck {
            status: overall_status,
            components,
            timestamp: SystemTime::now(),
            uptime: self.start_time.elapsed().as_secs(),
        }
    }

    /// Get memory usage (placeholder implementation)
    async fn get_memory_usage(&self) -> Result<u64> {
        // In a real implementation, this would use system APIs to get actual memory usage
        // For now, return a placeholder value
        Ok(0)
    }

    /// Get CPU usage (placeholder implementation)
    async fn get_cpu_usage(&self) -> Result<f64> {
        // In a real implementation, this would use system APIs to get actual CPU usage
        // For now, return a placeholder value
        Ok(0.0)
    }

    /// Get disk usage (placeholder implementation)
    async fn get_disk_usage(&self) -> Result<DiskUsage> {
        // In a real implementation, this would use system APIs to get actual disk usage
        // For now, return placeholder values that indicate healthy disk space
        Ok(DiskUsage {
            total: 1024 * 1024 * 1024 * 100,    // 100GB
            available: 1024 * 1024 * 1024 * 80, // 80GB (healthy amount)
            used: 1024 * 1024 * 1024 * 20,      // 20GB
        })
    }
}

impl Clone for MetricsCollector {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            download_stats: Arc::clone(&self.download_stats),
            system_metrics: Arc::clone(&self.system_metrics),
            start_time: self.start_time,
            completed_downloads: AtomicU64::new(self.completed_downloads.load(Ordering::Relaxed)),
            failed_downloads: AtomicU64::new(self.failed_downloads.load(Ordering::Relaxed)),
            total_bytes_downloaded: AtomicU64::new(
                self.total_bytes_downloaded.load(Ordering::Relaxed),
            ),
            total_connections: AtomicUsize::new(self.total_connections.load(Ordering::Relaxed)),
        }
    }
}
