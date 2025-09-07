//! Proxy support infrastructure with HTTP/HTTPS/SOCKS proxy support

use std::{collections::HashMap, net::IpAddr, time::Duration};

use async_trait::async_trait;
use reqwest::Proxy;
use serde::{Deserialize, Serialize};
use tokio::time::timeout;
use url::Url;

use crate::{
    error::{NetworkError, Result},
    types::{ProxyAuth, ProxyConfig},
};

/// Proxy type enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProxyType {
    /// HTTP proxy
    Http,
    /// HTTPS proxy (HTTP proxy over TLS)
    Https,
    /// SOCKS4 proxy
    Socks4,
    /// SOCKS5 proxy
    Socks5,
}

impl ProxyType {
    /// Parse proxy type from URL scheme
    pub fn from_scheme(scheme: &str) -> Option<Self> {
        match scheme.to_lowercase().as_str() {
            "http" => Some(ProxyType::Http),
            "https" => Some(ProxyType::Https),
            "socks4" => Some(ProxyType::Socks4),
            "socks5" => Some(ProxyType::Socks5),
            _ => None,
        }
    }

    /// Get the default port for this proxy type
    pub fn default_port(&self) -> u16 {
        match self {
            ProxyType::Http => 8080,
            ProxyType::Https => 8443,
            ProxyType::Socks4 => 1080,
            ProxyType::Socks5 => 1080,
        }
    }
}

/// Enhanced proxy configuration with additional features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedProxyConfig {
    /// Proxy URL
    pub url: Url,
    /// Proxy type (auto-detected from URL if not specified)
    pub proxy_type: Option<ProxyType>,
    /// Authentication credentials
    pub auth: Option<ProxyAuth>,
    /// Connection timeout
    pub timeout: Duration,
    /// Whether to use this proxy for HTTPS connections
    pub https_enabled: bool,
    /// Domains/IPs to bypass proxy for
    pub bypass_list: Vec<String>,
    /// Maximum number of connections through this proxy
    pub max_connections: Option<u32>,
    /// Health check URL for proxy validation
    pub health_check_url: Option<Url>,
    /// Custom headers to send through proxy
    pub headers: HashMap<String, String>,
}

impl Default for EnhancedProxyConfig {
    fn default() -> Self {
        Self {
            url: Url::parse("http://localhost:8080").unwrap(),
            proxy_type: None,
            auth: None,
            timeout: Duration::from_secs(30),
            https_enabled: true,
            bypass_list: Vec::new(),
            max_connections: None,
            health_check_url: None,
            headers: HashMap::new(),
        }
    }
}

impl From<ProxyConfig> for EnhancedProxyConfig {
    fn from(config: ProxyConfig) -> Self {
        let proxy_type = ProxyType::from_scheme(config.url.scheme());

        Self {
            url: config.url,
            proxy_type,
            auth: config.auth,
            timeout: Duration::from_secs(30),
            https_enabled: true,
            bypass_list: Vec::new(),
            max_connections: None,
            health_check_url: None,
            headers: HashMap::new(),
        }
    }
}

/// Proxy chain configuration for failover and load balancing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyChainConfig {
    /// List of proxy configurations in order of preference
    pub proxies: Vec<EnhancedProxyConfig>,
    /// Failover strategy
    pub failover_strategy: FailoverStrategy,
    /// Health check interval
    pub health_check_interval: Duration,
    /// Maximum failures before marking proxy as unhealthy
    pub max_failures: u32,
    /// Recovery time before retrying failed proxy
    pub recovery_time: Duration,
}

/// Failover strategy for proxy chains
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FailoverStrategy {
    /// Use proxies in order, failover to next on failure
    Sequential,
    /// Round-robin between healthy proxies
    RoundRobin,
    /// Random selection from healthy proxies
    Random,
    /// Weighted selection based on proxy performance
    Weighted,
}

impl Default for ProxyChainConfig {
    fn default() -> Self {
        Self {
            proxies: Vec::new(),
            failover_strategy: FailoverStrategy::Sequential,
            health_check_interval: Duration::from_secs(60),
            max_failures: 3,
            recovery_time: Duration::from_secs(300),
        }
    }
}

/// Proxy health status
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub enum ProxyHealth {
    /// Proxy is healthy and available
    Healthy,
    /// Proxy is unhealthy but being monitored
    Unhealthy,
    /// Proxy is temporarily disabled
    Disabled,
    /// Proxy health is unknown (not tested yet)
    #[default]
    Unknown,
}

/// Proxy statistics
#[derive(Debug, Clone, Default)]
pub struct ProxyStats {
    /// Total requests through this proxy
    pub requests: u64,
    /// Successful requests
    pub successes: u64,
    /// Failed requests
    pub failures: u64,
    /// Average response time
    pub avg_response_time: Duration,
    /// Current health status
    pub health: ProxyHealth,
    /// Last health check time
    pub last_health_check: Option<std::time::Instant>,
    /// Consecutive failures
    pub consecutive_failures: u32,
}

/// Proxy manager for handling multiple proxies with failover
pub struct ProxyManager {
    /// Proxy chain configuration
    config: ProxyChainConfig,
    /// Proxy statistics
    stats: HashMap<String, ProxyStats>,
    /// Current proxy index for round-robin
    current_index: std::sync::atomic::AtomicUsize,
    /// Random number generator for weighted selection
    rng: tokio::sync::Mutex<rand::rngs::ThreadRng>,
}

impl ProxyManager {
    /// Create a new proxy manager
    pub fn new(config: ProxyChainConfig) -> Self {
        Self {
            config,
            stats: HashMap::new(),
            current_index: std::sync::atomic::AtomicUsize::new(0),
            rng: tokio::sync::Mutex::new(rand::rng()),
        }
    }

    /// Get the best available proxy based on the failover strategy
    pub async fn get_proxy(&mut self, target_url: &Url) -> Result<Option<EnhancedProxyConfig>> {
        // Check if target should bypass proxy
        if self.should_bypass(target_url) {
            return Ok(None);
        }

        let healthy_proxies = self.get_healthy_proxies().await;
        if healthy_proxies.is_empty() {
            return Err(NetworkError::Proxy("No healthy proxies available".to_string()).into());
        }

        let selected_proxy = match self.config.failover_strategy {
            FailoverStrategy::Sequential => healthy_proxies.first().cloned(),
            FailoverStrategy::RoundRobin => self.select_round_robin(&healthy_proxies),
            FailoverStrategy::Random => self.select_random(&healthy_proxies).await,
            FailoverStrategy::Weighted => self.select_weighted(&healthy_proxies).await,
        };

        Ok(selected_proxy)
    }

    /// Check if a URL should bypass proxy
    fn should_bypass(&self, url: &Url) -> bool {
        let host = url.host_str().unwrap_or("");

        for bypass_pattern in &self
            .config
            .proxies
            .iter()
            .flat_map(|p| &p.bypass_list)
            .collect::<Vec<_>>()
        {
            if self.matches_bypass_pattern(host, bypass_pattern) {
                return true;
            }
        }

        false
    }

    /// Check if host matches bypass pattern
    fn matches_bypass_pattern(&self, host: &str, pattern: &str) -> bool {
        // Support wildcards and exact matches
        if pattern == "*" {
            return true;
        }

        if let Some(domain) = pattern.strip_prefix("*.") {
            return host.ends_with(domain) || host == domain;
        }

        if pattern.contains('/') {
            // CIDR notation for IP ranges
            if let Ok(host_ip) = host.parse::<IpAddr>() {
                return self.ip_in_cidr(host_ip, pattern);
            }
        }

        host == pattern
    }

    /// Check if IP is in CIDR range
    fn ip_in_cidr(&self, ip: IpAddr, cidr: &str) -> bool {
        // Simplified CIDR matching - in production, use a proper CIDR library
        if let Some((network, prefix_len)) = cidr.split_once('/') {
            if let (Ok(network_ip), Ok(prefix)) =
                (network.parse::<IpAddr>(), prefix_len.parse::<u8>())
            {
                match (ip, network_ip) {
                    (IpAddr::V4(ip4), IpAddr::V4(net4)) => {
                        let mask = !((1u32 << (32 - prefix)) - 1);
                        (u32::from(ip4) & mask) == (u32::from(net4) & mask)
                    }
                    (IpAddr::V6(ip6), IpAddr::V6(net6)) => {
                        let ip_bytes = ip6.octets();
                        let net_bytes = net6.octets();
                        let full_bytes = prefix / 8;
                        let remaining_bits = prefix % 8;

                        // Check full bytes
                        if ip_bytes[..full_bytes as usize] != net_bytes[..full_bytes as usize] {
                            return false;
                        }

                        // Check remaining bits
                        if remaining_bits > 0 && full_bytes < 16 {
                            let mask = 0xFF << (8 - remaining_bits);
                            let ip_byte = ip_bytes[full_bytes as usize];
                            let net_byte = net_bytes[full_bytes as usize];
                            return (ip_byte & mask) == (net_byte & mask);
                        }

                        true
                    }
                    _ => false,
                }
            } else {
                false
            }
        } else {
            false
        }
    }

    /// Get list of healthy proxies
    async fn get_healthy_proxies(&self) -> Vec<EnhancedProxyConfig> {
        let mut healthy = Vec::new();

        for proxy in &self.config.proxies {
            let proxy_key = self.get_proxy_key(proxy);
            if let Some(stats) = self.stats.get(&proxy_key) {
                if stats.health == ProxyHealth::Healthy {
                    healthy.push(proxy.clone());
                }
            } else {
                // Unknown health, assume healthy for first try
                healthy.push(proxy.clone());
            }
        }

        healthy
    }

    /// Select proxy using round-robin strategy
    fn select_round_robin(&self, proxies: &[EnhancedProxyConfig]) -> Option<EnhancedProxyConfig> {
        if proxies.is_empty() {
            return None;
        }

        let index = self
            .current_index
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            % proxies.len();
        Some(proxies[index].clone())
    }

    /// Select proxy using random strategy
    async fn select_random(&self, proxies: &[EnhancedProxyConfig]) -> Option<EnhancedProxyConfig> {
        if proxies.is_empty() {
            return None;
        }

        use rand::Rng;
        let mut rng = self.rng.lock().await;
        let index = rng.random_range(0..proxies.len());
        Some(proxies[index].clone())
    }

    /// Select proxy using weighted strategy based on performance
    async fn select_weighted(
        &self,
        proxies: &[EnhancedProxyConfig],
    ) -> Option<EnhancedProxyConfig> {
        if proxies.is_empty() {
            return None;
        }

        // Calculate weights based on success rate and response time
        let mut weights = Vec::new();
        let mut total_weight = 0.0;

        for proxy in proxies {
            let proxy_key = self.get_proxy_key(proxy);
            let weight = if let Some(stats) = self.stats.get(&proxy_key) {
                if stats.requests > 0 {
                    let success_rate = stats.successes as f64 / stats.requests as f64;
                    let response_time_factor =
                        1.0 / (stats.avg_response_time.as_millis() as f64 + 1.0);
                    success_rate * response_time_factor
                } else {
                    1.0 // Default weight for untested proxies
                }
            } else {
                1.0 // Default weight for unknown proxies
            };

            weights.push(weight);
            total_weight += weight;
        }

        if total_weight == 0.0 {
            return self.select_random(proxies).await;
        }

        // Select based on weighted random
        use rand::Rng;
        let mut rng = self.rng.lock().await;
        let mut random_value = rng.random::<f64>() * total_weight;

        for (i, weight) in weights.iter().enumerate() {
            random_value -= weight;
            if random_value <= 0.0 {
                return Some(proxies[i].clone());
            }
        }

        // Fallback to last proxy
        proxies.last().cloned()
    }

    /// Record proxy usage statistics
    pub fn record_request(
        &mut self,
        proxy: &EnhancedProxyConfig,
        success: bool,
        response_time: Duration,
    ) {
        let proxy_key = self.get_proxy_key(proxy);
        let stats = self.stats.entry(proxy_key).or_default();

        stats.requests += 1;
        if success {
            stats.successes += 1;
            stats.consecutive_failures = 0;
            stats.health = ProxyHealth::Healthy;
        } else {
            stats.failures += 1;
            stats.consecutive_failures += 1;

            if stats.consecutive_failures >= self.config.max_failures {
                stats.health = ProxyHealth::Unhealthy;
            }
        }

        // Update average response time using exponential moving average
        if stats.requests == 1 {
            stats.avg_response_time = response_time;
        } else {
            let alpha = 0.1; // Smoothing factor
            let current_ms = stats.avg_response_time.as_millis() as f64;
            let new_ms = response_time.as_millis() as f64;
            let updated_ms = (alpha * new_ms + (1.0 - alpha) * current_ms) as u64;
            stats.avg_response_time = Duration::from_millis(updated_ms);
        }
    }

    /// Perform health check on all proxies
    pub async fn health_check(&mut self) {
        for proxy in &self.config.proxies {
            let health = self.check_proxy_health(proxy).await;
            let proxy_key = self.get_proxy_key(proxy);
            let stats = self.stats.entry(proxy_key).or_default();

            let is_healthy = health == ProxyHealth::Healthy;
            stats.health = health;
            stats.last_health_check = Some(std::time::Instant::now());

            // Reset consecutive failures if proxy is healthy again
            if is_healthy {
                stats.consecutive_failures = 0;
            }
        }
    }

    /// Check health of a specific proxy
    async fn check_proxy_health(&self, proxy: &EnhancedProxyConfig) -> ProxyHealth {
        let default_url = Url::parse("http://httpbin.org/ip").unwrap();
        let health_check_url = proxy.health_check_url.as_ref().unwrap_or(&default_url);

        match self.test_proxy_connection(proxy, health_check_url).await {
            Ok(_) => ProxyHealth::Healthy,
            Err(_) => ProxyHealth::Unhealthy,
        }
    }

    /// Test proxy connection
    async fn test_proxy_connection(
        &self,
        proxy: &EnhancedProxyConfig,
        test_url: &Url,
    ) -> Result<()> {
        let client = self.build_proxy_client(proxy)?;

        let response = timeout(proxy.timeout, client.get(test_url.as_str()).send())
            .await
            .map_err(|_| NetworkError::Timeout)?
            .map_err(|e| NetworkError::Proxy(format!("Proxy test failed: {}", e)))?;

        if response.status().is_success() {
            Ok(())
        } else {
            Err(
                NetworkError::Proxy(format!("Proxy test returned status: {}", response.status()))
                    .into(),
            )
        }
    }

    /// Build reqwest client with proxy configuration
    fn build_proxy_client(&self, proxy_config: &EnhancedProxyConfig) -> Result<reqwest::Client> {
        let mut proxy = Proxy::all(proxy_config.url.to_string())
            .map_err(|e| NetworkError::Proxy(format!("Failed to create proxy: {}", e)))?;

        // Add authentication if provided
        if let Some(auth) = &proxy_config.auth {
            proxy = proxy.basic_auth(&auth.username, &auth.password);
        }

        let client = reqwest::Client::builder()
            .proxy(proxy)
            .timeout(proxy_config.timeout)
            .build()
            .map_err(|e| NetworkError::Proxy(format!("Failed to build proxy client: {}", e)))?;

        Ok(client)
    }

    /// Get unique key for proxy identification
    fn get_proxy_key(&self, proxy: &EnhancedProxyConfig) -> String {
        format!(
            "{}://{}",
            proxy.url.scheme(),
            proxy.url.host_str().unwrap_or("unknown")
        )
    }

    /// Get proxy statistics
    pub fn get_stats(&self) -> &HashMap<String, ProxyStats> {
        &self.stats
    }

    /// Start background health check task
    pub fn start_health_check_task(&self) -> tokio::task::JoinHandle<()> {
        let config = self.config.clone();
        let interval = config.health_check_interval;

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            loop {
                interval_timer.tick().await;

                // In a real implementation, we'd need to share the ProxyManager
                // across tasks using Arc<Mutex<ProxyManager>>
                // For now, this is just the structure
                tracing::debug!("Performing proxy health checks");
            }
        })
    }
}

/// Trait for proxy-aware network operations
#[async_trait]
pub trait ProxyAware {
    /// Execute request through proxy if configured
    async fn execute_with_proxy(
        &self,
        request: reqwest::Request,
        proxy: Option<&EnhancedProxyConfig>,
    ) -> Result<reqwest::Response>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn test_proxy_type_from_scheme() {
        assert_eq!(ProxyType::from_scheme("http"), Some(ProxyType::Http));
        assert_eq!(ProxyType::from_scheme("https"), Some(ProxyType::Https));
        assert_eq!(ProxyType::from_scheme("socks4"), Some(ProxyType::Socks4));
        assert_eq!(ProxyType::from_scheme("socks5"), Some(ProxyType::Socks5));
        assert_eq!(ProxyType::from_scheme("invalid"), None);
    }

    #[test]
    fn test_proxy_type_default_ports() {
        assert_eq!(ProxyType::Http.default_port(), 8080);
        assert_eq!(ProxyType::Https.default_port(), 8443);
        assert_eq!(ProxyType::Socks4.default_port(), 1080);
        assert_eq!(ProxyType::Socks5.default_port(), 1080);
    }

    #[test]
    fn test_bypass_pattern_matching() {
        let config = ProxyChainConfig::default();
        let manager = ProxyManager::new(config);

        // Test wildcard
        assert!(manager.matches_bypass_pattern("example.com", "*"));

        // Test exact match
        assert!(manager.matches_bypass_pattern("example.com", "example.com"));
        assert!(!manager.matches_bypass_pattern("example.com", "other.com"));

        // Test subdomain wildcard
        assert!(manager.matches_bypass_pattern("sub.example.com", "*.example.com"));
        assert!(manager.matches_bypass_pattern("example.com", "*.example.com"));
        assert!(!manager.matches_bypass_pattern("other.com", "*.example.com"));
    }

    #[test]
    fn test_enhanced_proxy_config_from_basic() {
        let basic_config = ProxyConfig {
            url: Url::from_str("http://proxy.example.com:8080").unwrap(),
            auth: Some(ProxyAuth {
                username: "user".to_string(),
                password: "pass".to_string(),
            }),
        };

        let enhanced: EnhancedProxyConfig = basic_config.into();
        assert_eq!(enhanced.url.host_str(), Some("proxy.example.com"));
        assert_eq!(enhanced.url.port(), Some(8080));
        assert!(enhanced.auth.is_some());
        assert_eq!(enhanced.proxy_type, Some(ProxyType::Http));
    }

    #[tokio::test]
    async fn test_proxy_manager_creation() {
        let config = ProxyChainConfig::default();
        let manager = ProxyManager::new(config);

        // Test that manager is created successfully
        assert_eq!(manager.stats.len(), 0);
    }

    #[test]
    fn test_proxy_stats_default() {
        let stats = ProxyStats::default();
        assert_eq!(stats.requests, 0);
        assert_eq!(stats.successes, 0);
        assert_eq!(stats.failures, 0);
        assert_eq!(stats.health, ProxyHealth::Unknown);
        assert_eq!(stats.consecutive_failures, 0);
    }

    #[test]
    fn test_failover_strategy_serialization() {
        let strategies = vec![
            FailoverStrategy::Sequential,
            FailoverStrategy::RoundRobin,
            FailoverStrategy::Random,
            FailoverStrategy::Weighted,
        ];

        for strategy in strategies {
            let json = serde_json::to_string(&strategy).unwrap();
            let deserialized: FailoverStrategy = serde_json::from_str(&json).unwrap();
            assert_eq!(strategy, deserialized);
        }
    }
}
