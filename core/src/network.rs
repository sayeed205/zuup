//! Network layer with connection pooling, proxy support, and TLS handling

pub mod proxy;
pub mod tls;

use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

use bytes::Bytes;
use reqwest::Proxy;
use tokio::sync::RwLock;
use trust_dns_resolver::TokioAsyncResolver;
use url::Url;

use self::{
    proxy::{EnhancedProxyConfig, ProxyChainConfig, ProxyManager},
    tls::{EnhancedTlsConfig, TlsContextManager},
};
use crate::{
    error::{NetworkError, Result},
    types::{ProxyConfig, TlsConfig},
};

/// Connection pool entry with metadata
#[derive(Debug)]
struct PooledConnection {
    /// The actual connection
    connection: reqwest::Client,
    /// When this connection was created
    created_at: Instant,
    /// When this connection was last used
    last_used: Instant,
    /// Number of requests made with this connection
    request_count: u64,
    /// Whether this connection is currently in use
    in_use: bool,
}

impl PooledConnection {
    fn new(connection: reqwest::Client) -> Self {
        let now = Instant::now();
        Self {
            connection,
            created_at: now,
            last_used: now,
            request_count: 0,
            in_use: false,
        }
    }

    fn is_expired(&self, max_age: Duration, max_idle: Duration) -> bool {
        let now = Instant::now();
        now.duration_since(self.created_at) > max_age
            || now.duration_since(self.last_used) > max_idle
    }

    fn mark_used(&mut self) {
        self.last_used = Instant::now();
        self.request_count += 1;
        self.in_use = true;
    }

    fn mark_available(&mut self) {
        self.in_use = false;
    }
}

/// Connection pool configuration
#[derive(Debug, Clone)]
pub struct ConnectionPoolConfig {
    /// Maximum connections per host
    pub max_connections_per_host: usize,
    /// Maximum total connections
    pub max_total_connections: usize,
    /// Maximum age of a connection before it's discarded
    pub max_connection_age: Duration,
    /// Maximum idle time before a connection is discarded
    pub max_idle_time: Duration,
    /// Connection timeout
    pub connection_timeout: Duration,
    /// Keep-alive timeout
    pub keep_alive_timeout: Duration,
    /// Health check interval
    pub health_check_interval: Duration,
}

impl Default for ConnectionPoolConfig {
    fn default() -> Self {
        Self {
            max_connections_per_host: 8,
            max_total_connections: 100,
            max_connection_age: Duration::from_secs(300), // 5 minutes
            max_idle_time: Duration::from_secs(90),       // 90 seconds
            connection_timeout: Duration::from_secs(30),
            keep_alive_timeout: Duration::from_secs(90),
            health_check_interval: Duration::from_secs(60),
        }
    }
}

/// Connection pool for managing reusable HTTP clients
pub struct ConnectionPool {
    /// Pool of connections organized by host
    connections: Arc<RwLock<HashMap<String, Vec<PooledConnection>>>>,
    /// Pool configuration
    config: ConnectionPoolConfig,
    /// DNS resolver
    resolver: Arc<TokioAsyncResolver>,
    /// Connection statistics
    stats: Arc<RwLock<ConnectionPoolStats>>,
}

/// Connection pool statistics
#[derive(Debug, Default, Clone)]
pub struct ConnectionPoolStats {
    pub total_connections: usize,
    pub active_connections: usize,
    pub idle_connections: usize,
    pub connections_created: u64,
    pub connections_reused: u64,
    pub connections_expired: u64,
    pub connection_errors: u64,
    pub requests_served: u64,
}

impl ConnectionPool {
    /// Create a new connection pool
    pub async fn new(config: ConnectionPoolConfig) -> Result<Self> {
        let resolver = TokioAsyncResolver::tokio_from_system_conf()
            .map_err(|e| NetworkError::DnsResolutionFailed(e.to_string()))?;

        let pool = Self {
            connections: Arc::new(RwLock::new(HashMap::new())),
            config,
            resolver: Arc::new(resolver),
            stats: Arc::new(RwLock::new(ConnectionPoolStats::default())),
        };

        // Start background cleanup task
        pool.start_cleanup_task();

        Ok(pool)
    }

    /// Get a connection from the pool or create a new one
    pub async fn get_connection(
        &self,
        url: &Url,
        tls_config: &TlsConfig,
        proxy: Option<&ProxyConfig>,
    ) -> Result<reqwest::Client> {
        let host_key = self.get_host_key(url);

        // Try to get an existing connection
        {
            let mut connections = self.connections.write().await;
            if let Some(host_connections) = connections.get_mut(&host_key) {
                // Find an available, non-expired connection
                for conn in host_connections.iter_mut() {
                    if !conn.in_use
                        && !conn
                            .is_expired(self.config.max_connection_age, self.config.max_idle_time)
                    {
                        conn.mark_used();

                        // Update stats
                        let mut stats = self.stats.write().await;
                        stats.connections_reused += 1;
                        stats.active_connections += 1;
                        stats.idle_connections = stats.idle_connections.saturating_sub(1);

                        return Ok(conn.connection.clone());
                    }
                }
            }
        }

        // Create a new connection
        let client = self.create_connection(url, tls_config, proxy).await?;

        // Add to pool
        {
            let mut connections = self.connections.write().await;
            let host_connections = connections.entry(host_key).or_insert_with(Vec::new);

            // Check if we can add more connections for this host
            if host_connections.len() < self.config.max_connections_per_host {
                let mut pooled = PooledConnection::new(client.clone());
                pooled.mark_used();
                host_connections.push(pooled);

                // Update stats
                let mut stats = self.stats.write().await;
                stats.connections_created += 1;
                stats.total_connections += 1;
                stats.active_connections += 1;
            }
        }

        Ok(client)
    }

    /// Return a connection to the pool
    pub async fn return_connection(&self, url: &Url, _client: reqwest::Client) {
        let host_key = self.get_host_key(url);

        let mut connections = self.connections.write().await;
        if let Some(host_connections) = connections.get_mut(&host_key) {
            // Find the connection and mark it as available
            for conn in host_connections.iter_mut() {
                if conn.in_use {
                    conn.mark_available();

                    // Update stats
                    let mut stats = self.stats.write().await;
                    stats.active_connections = stats.active_connections.saturating_sub(1);
                    stats.idle_connections += 1;
                    break;
                }
            }
        }
    }

    /// Create a new HTTP client with the specified configuration
    async fn create_connection(
        &self,
        url: &Url,
        tls_config: &TlsConfig,
        proxy: Option<&ProxyConfig>,
    ) -> Result<reqwest::Client> {
        let mut client_builder = reqwest::Client::builder()
            .timeout(self.config.connection_timeout)
            .pool_idle_timeout(Some(self.config.keep_alive_timeout))
            .pool_max_idle_per_host(self.config.max_connections_per_host)
            .tcp_keepalive(Some(Duration::from_secs(60)));

        // Configure TLS
        if url.scheme() == "https" {
            let tls_connector = self.build_tls_connector(tls_config).await?;
            client_builder = client_builder.use_preconfigured_tls(tls_connector);
        }

        // Configure proxy
        if let Some(proxy_config) = proxy {
            let proxy = self.build_proxy(proxy_config)?;
            client_builder = client_builder.proxy(proxy);
        }

        client_builder
            .build()
            .map_err(|e| NetworkError::ConnectionFailed(e.to_string()).into())
    }

    /// Build TLS connector with custom configuration
    async fn build_tls_connector(&self, tls_config: &TlsConfig) -> Result<reqwest::Client> {
        // For now, return a basic client with simple TLS configuration
        reqwest::Client::builder()
            .danger_accept_invalid_certs(!tls_config.verify_certificates)
            .build()
            .map_err(|e| NetworkError::ConnectionFailed(e.to_string()).into())
    }

    // Client certificate loading removed for simplicity

    /// Build proxy configuration
    fn build_proxy(&self, proxy_config: &ProxyConfig) -> Result<Proxy> {
        let mut proxy = Proxy::all(&proxy_config.url.to_string())
            .map_err(|e| NetworkError::Proxy(format!("Failed to create proxy: {}", e)))?;

        // Add authentication if provided
        if let Some(auth) = &proxy_config.auth {
            proxy = proxy.basic_auth(&auth.username, &auth.password);
        }

        Ok(proxy)
    }

    /// Get host key for connection pooling
    fn get_host_key(&self, url: &Url) -> String {
        format!(
            "{}://{}:{}",
            url.scheme(),
            url.host_str().unwrap_or("localhost"),
            url.port_or_known_default().unwrap_or(80)
        )
    }

    /// Clean up expired connections
    pub async fn cleanup_expired(&self) {
        let mut connections = self.connections.write().await;
        let mut expired_count = 0;

        for host_connections in connections.values_mut() {
            let initial_len = host_connections.len();
            host_connections.retain(|conn| {
                !conn.is_expired(self.config.max_connection_age, self.config.max_idle_time)
            });
            expired_count += initial_len - host_connections.len();
        }

        // Remove empty host entries
        connections.retain(|_, host_connections| !host_connections.is_empty());

        // Update stats
        if expired_count > 0 {
            let mut stats = self.stats.write().await;
            stats.connections_expired += expired_count as u64;
            stats.total_connections = stats.total_connections.saturating_sub(expired_count);
        }
    }

    /// Start background cleanup task
    fn start_cleanup_task(&self) {
        let connections = Arc::clone(&self.connections);
        let config = self.config.clone();
        let stats = Arc::clone(&self.stats);

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.health_check_interval);

            loop {
                interval.tick().await;

                // Cleanup expired connections
                let mut connections_guard = connections.write().await;
                let mut expired_count = 0;

                for host_connections in connections_guard.values_mut() {
                    let initial_len = host_connections.len();
                    host_connections.retain(|conn| {
                        !conn.is_expired(config.max_connection_age, config.max_idle_time)
                    });
                    expired_count += initial_len - host_connections.len();
                }

                connections_guard.retain(|_, host_connections| !host_connections.is_empty());
                drop(connections_guard);

                // Update stats
                if expired_count > 0 {
                    let mut stats_guard = stats.write().await;
                    stats_guard.connections_expired += expired_count as u64;
                    stats_guard.total_connections =
                        stats_guard.total_connections.saturating_sub(expired_count);
                }
            }
        });
    }

    /// Get connection pool statistics
    pub async fn stats(&self) -> ConnectionPoolStats {
        self.stats.read().await.clone()
    }

    /// Get detailed connection information
    pub async fn connection_info(&self) -> HashMap<String, Vec<ConnectionInfo>> {
        let connections = self.connections.read().await;
        let mut info = HashMap::new();

        for (host, host_connections) in connections.iter() {
            let host_info: Vec<ConnectionInfo> = host_connections
                .iter()
                .map(|conn| ConnectionInfo {
                    created_at: conn.created_at,
                    last_used: conn.last_used,
                    request_count: conn.request_count,
                    in_use: conn.in_use,
                    age: conn.created_at.elapsed(),
                    idle_time: conn.last_used.elapsed(),
                })
                .collect();
            info.insert(host.clone(), host_info);
        }

        info
    }
}

/// Information about a specific connection
#[derive(Debug, Clone)]
pub struct ConnectionInfo {
    pub created_at: Instant,
    pub last_used: Instant,
    pub request_count: u64,
    pub in_use: bool,
    pub age: Duration,
    pub idle_time: Duration,
}

/// Network client configuration
#[derive(Debug, Clone)]
pub struct NetworkClientConfig {
    /// User agent string
    pub user_agent: String,
    /// Request timeout
    pub timeout: Duration,
    /// Maximum redirects to follow
    pub max_redirects: u32,
    /// Proxy configuration
    pub proxy: Option<ProxyConfig>,
    /// TLS configuration
    pub tls: TlsConfig,
    /// Enhanced TLS configuration
    pub enhanced_tls: Option<EnhancedTlsConfig>,
    /// Whether to enable HTTP/2
    pub enable_http2: bool,
    /// Whether to enable compression
    pub enable_compression: bool,
    /// Custom headers to include in all requests
    pub default_headers: HashMap<String, String>,
    /// Connection pool configuration
    pub pool_config: ConnectionPoolConfig,
}

impl Default for NetworkClientConfig {
    fn default() -> Self {
        Self {
            user_agent: format!("Ruso/{}", env!("CARGO_PKG_VERSION")),
            timeout: Duration::from_secs(30),
            max_redirects: 10,
            proxy: None,
            tls: TlsConfig::default(),
            enhanced_tls: None,
            enable_http2: true,
            enable_compression: true,
            default_headers: HashMap::new(),
            pool_config: ConnectionPoolConfig::default(),
        }
    }
}

/// Network client for making HTTP requests with connection pooling
pub struct NetworkClient {
    /// Client configuration
    config: NetworkClientConfig,
    /// Connection pool
    pool: Arc<ConnectionPool>,
    /// Proxy manager for handling proxy chains and failover
    proxy_manager: Option<Arc<tokio::sync::Mutex<ProxyManager>>>,
    /// TLS context manager for advanced TLS features
    tls_manager: Option<Arc<tokio::sync::Mutex<TlsContextManager>>>,
}

impl NetworkClient {
    /// Create a new network client
    pub async fn new(config: NetworkClientConfig) -> Result<Self> {
        let pool = Arc::new(ConnectionPool::new(config.pool_config.clone()).await?);

        // Initialize proxy manager if proxy is configured
        let proxy_manager = if config.proxy.is_some() {
            let proxy_config = config.proxy.as_ref().unwrap();
            let enhanced_proxy: EnhancedProxyConfig = proxy_config.clone().into();
            let chain_config = ProxyChainConfig {
                proxies: vec![enhanced_proxy],
                ..Default::default()
            };
            Some(Arc::new(tokio::sync::Mutex::new(ProxyManager::new(
                chain_config,
            ))))
        } else {
            None
        };

        // Initialize TLS manager if enhanced TLS is configured
        let tls_manager = if let Some(enhanced_tls) = &config.enhanced_tls {
            let tls_ctx = TlsContextManager::new(enhanced_tls.clone()).await?;
            Some(Arc::new(tokio::sync::Mutex::new(tls_ctx)))
        } else {
            None
        };

        Ok(Self {
            config,
            pool,
            proxy_manager,
            tls_manager,
        })
    }

    /// Create a new network client with proxy chain
    pub async fn new_with_proxy_chain(
        config: NetworkClientConfig,
        proxy_chain: ProxyChainConfig,
    ) -> Result<Self> {
        let pool = Arc::new(ConnectionPool::new(config.pool_config.clone()).await?);
        let proxy_manager = Some(Arc::new(tokio::sync::Mutex::new(ProxyManager::new(
            proxy_chain,
        ))));

        // Initialize TLS manager if enhanced TLS is configured
        let tls_manager = if let Some(enhanced_tls) = &config.enhanced_tls {
            let tls_ctx = TlsContextManager::new(enhanced_tls.clone()).await?;
            Some(Arc::new(tokio::sync::Mutex::new(tls_ctx)))
        } else {
            None
        };

        Ok(Self {
            config,
            pool,
            proxy_manager,
            tls_manager,
        })
    }

    /// Create a new network client with enhanced TLS configuration
    pub async fn new_with_enhanced_tls(
        mut config: NetworkClientConfig,
        tls_config: EnhancedTlsConfig,
    ) -> Result<Self> {
        config.enhanced_tls = Some(tls_config);
        Self::new(config).await
    }

    /// Make an HTTP GET request
    pub async fn get(&self, url: &Url) -> Result<HttpResponse> {
        self.request(HttpMethod::Get, url, None, None).await
    }

    /// Make an HTTP POST request
    pub async fn post(&self, url: &Url, body: Option<Bytes>) -> Result<HttpResponse> {
        self.request(HttpMethod::Post, url, body, None).await
    }

    /// Make an HTTP HEAD request
    pub async fn head(&self, url: &Url) -> Result<HttpResponse> {
        self.request(HttpMethod::Head, url, None, None).await
    }

    /// Make a range request
    pub async fn get_range(&self, url: &Url, start: u64, end: Option<u64>) -> Result<HttpResponse> {
        let range_header = if let Some(end) = end {
            format!("bytes={}-{}", start, end)
        } else {
            format!("bytes={}-", start)
        };

        let mut headers = HashMap::new();
        headers.insert("Range".to_string(), range_header);

        self.request(HttpMethod::Get, url, None, Some(headers))
            .await
    }

    /// Make a generic HTTP request
    pub async fn request(
        &self,
        method: HttpMethod,
        url: &Url,
        body: Option<Bytes>,
        additional_headers: Option<HashMap<String, String>>,
    ) -> Result<HttpResponse> {
        let client = self
            .pool
            .get_connection(url, &self.config.tls, self.config.proxy.as_ref())
            .await?;

        let mut request_builder = match method {
            HttpMethod::Get => client.get(url.as_str()),
            HttpMethod::Post => client.post(url.as_str()),
            HttpMethod::Put => client.put(url.as_str()),
            HttpMethod::Delete => client.delete(url.as_str()),
            HttpMethod::Head => client.head(url.as_str()),
            HttpMethod::Options => client.request(reqwest::Method::OPTIONS, url.as_str()),
            HttpMethod::Patch => client.patch(url.as_str()),
        };

        // Add default headers
        for (key, value) in &self.config.default_headers {
            request_builder = request_builder.header(key, value);
        }

        // Add additional headers
        if let Some(headers) = additional_headers {
            for (key, value) in headers {
                request_builder = request_builder.header(key, value);
            }
        }

        // Add user agent
        request_builder = request_builder.header("User-Agent", &self.config.user_agent);

        // Add body if provided
        if let Some(body_data) = body {
            request_builder = request_builder.body(body_data);
        }

        // Set timeout
        request_builder = request_builder.timeout(self.config.timeout);

        // Execute request
        let response = request_builder
            .send()
            .await
            .map_err(|e| NetworkError::ConnectionFailed(e.to_string()))?;

        // Convert response
        let status = response.status().as_u16();
        let headers: HashMap<String, String> = response
            .headers()
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
            .collect();

        let body = response
            .bytes()
            .await
            .map_err(|e| NetworkError::InvalidResponse(e.to_string()))?
            .to_vec();

        // Return connection to pool
        self.pool.return_connection(url, client).await;

        Ok(HttpResponse {
            status,
            headers,
            body,
        })
    }

    /// Check if a URL supports range requests
    pub async fn supports_ranges(&self, url: &Url) -> Result<bool> {
        let response = self.head(url).await?;
        Ok(response
            .headers
            .get("accept-ranges")
            .map(|v| v.to_lowercase() == "bytes")
            .unwrap_or(false))
    }

    /// Get content length for a URL
    pub async fn get_content_length(&self, url: &Url) -> Result<Option<u64>> {
        let response = self.head(url).await?;

        if let Some(length_str) = response.headers.get("content-length") {
            length_str.parse().map(Some).map_err(|_| {
                NetworkError::InvalidResponse("Invalid content-length header".to_string()).into()
            })
        } else {
            Ok(None)
        }
    }

    /// Get connection pool statistics
    pub async fn pool_stats(&self) -> ConnectionPoolStats {
        self.pool.stats().await
    }

    /// Get detailed connection information
    pub async fn connection_info(&self) -> HashMap<String, Vec<ConnectionInfo>> {
        self.pool.connection_info().await
    }

    /// Perform connection health check
    pub async fn health_check(&self, url: &Url) -> Result<Duration> {
        let start = Instant::now();
        let _response = self.head(url).await?;
        Ok(start.elapsed())
    }
}

/// HTTP methods
#[derive(Debug, Clone, Copy)]
pub enum HttpMethod {
    Get,
    Post,
    Put,
    Delete,
    Head,
    Options,
    Patch,
}

impl std::fmt::Display for HttpMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HttpMethod::Get => write!(f, "GET"),
            HttpMethod::Post => write!(f, "POST"),
            HttpMethod::Put => write!(f, "PUT"),
            HttpMethod::Delete => write!(f, "DELETE"),
            HttpMethod::Head => write!(f, "HEAD"),
            HttpMethod::Options => write!(f, "OPTIONS"),
            HttpMethod::Patch => write!(f, "PATCH"),
        }
    }
}

/// HTTP response
#[derive(Debug, Clone)]
pub struct HttpResponse {
    pub status: u16,
    pub headers: HashMap<String, String>,
    pub body: Vec<u8>,
}

impl HttpResponse {
    /// Check if the response indicates success
    pub fn is_success(&self) -> bool {
        self.status >= 200 && self.status < 300
    }

    /// Check if the response is a redirect
    pub fn is_redirect(&self) -> bool {
        self.status >= 300 && self.status < 400
    }

    /// Get the redirect location
    pub fn redirect_location(&self) -> Option<&String> {
        self.headers.get("location")
    }

    /// Get content type
    pub fn content_type(&self) -> Option<&String> {
        self.headers.get("content-type")
    }

    /// Get content length
    pub fn content_length(&self) -> Option<u64> {
        self.headers
            .get("content-length")
            .and_then(|s| s.parse().ok())
    }

    /// Check if the response supports range requests
    pub fn supports_ranges(&self) -> bool {
        self.headers
            .get("accept-ranges")
            .map(|v| v.to_lowercase() == "bytes")
            .unwrap_or(false)
    }

    /// Get the content range from a partial response
    pub fn content_range(&self) -> Option<ContentRange> {
        self.headers
            .get("content-range")
            .and_then(|range_str| ContentRange::parse(range_str))
    }

    /// Get ETag if present
    pub fn etag(&self) -> Option<&String> {
        self.headers.get("etag")
    }

    /// Get Last-Modified if present
    pub fn last_modified(&self) -> Option<&String> {
        self.headers.get("last-modified")
    }
}

/// Content range information from HTTP responses
#[derive(Debug, Clone, PartialEq)]
pub struct ContentRange {
    pub start: u64,
    pub end: u64,
    pub total: Option<u64>,
}

impl ContentRange {
    /// Parse a Content-Range header value
    pub fn parse(range_str: &str) -> Option<Self> {
        // Format: "bytes start-end/total" or "bytes start-end/*"
        if !range_str.starts_with("bytes ") {
            return None;
        }

        let range_part = &range_str[6..]; // Skip "bytes "
        let parts: Vec<&str> = range_part.split('/').collect();
        if parts.len() != 2 {
            return None;
        }

        let range_spec = parts[0];
        let total_spec = parts[1];

        // Parse range (start-end)
        let range_parts: Vec<&str> = range_spec.split('-').collect();
        if range_parts.len() != 2 {
            return None;
        }

        let start = range_parts[0].parse().ok()?;
        let end = range_parts[1].parse().ok()?;

        // Parse total
        let total = if total_spec == "*" {
            None
        } else {
            total_spec.parse().ok()
        };

        Some(ContentRange { start, end, total })
    }
}
