//! Network layer with connection pooling, proxy support, and TLS handling

#[cfg(feature = "http")]
pub mod proxy;
#[cfg(feature = "http")]
pub mod tls;

#[cfg(feature = "http")]
mod http_impl {
    use std::{
        collections::HashMap,
        sync::Arc,
        time::{Duration, Instant},
    };

    use reqwest::Proxy;
    use tokio::sync::RwLock;
    use trust_dns_resolver::TokioAsyncResolver;
    use url::Url;

    use super::{proxy::{ProxyChainConfig, ProxyManager}, tls::{EnhancedTlsConfig, TlsContextManager}};
    use crate::{
        error::{NetworkError, Result},
        types::{ProxyConfig, TlsConfig},
    };

    /// Pooled connection wrapper
    struct PooledConnection {
        /// The actual connection
        connection: reqwest::Client,
        /// When this connection was created
        created_at: Instant,
        /// When this connection was last used
        last_used: Instant,
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
                in_use: false,
            }
        }

        fn mark_used(&mut self) {
            self.last_used = Instant::now();
            self.in_use = true;
        }

        fn mark_unused(&mut self) {
            self.in_use = false;
        }
    }

    /// Connection pool configuration
    #[derive(Debug, Clone)]
    pub struct ConnectionPoolConfig {
        /// Maximum number of connections per host
        pub max_connections_per_host: u32,
        /// Maximum idle time before connection is closed
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
                max_connections_per_host: 10,
                max_idle_time: Duration::from_secs(300),
                connection_timeout: Duration::from_secs(30),
                keep_alive_timeout: Duration::from_secs(60),
                health_check_interval: Duration::from_secs(30),
            }
        }
    }

    /// Connection pool for managing reusable HTTP clients
    pub struct ConnectionPool {
        /// Pool configuration
        config: ConnectionPoolConfig,
        /// Connections grouped by host
        connections: Arc<RwLock<HashMap<String, Vec<PooledConnection>>>>,
        /// DNS resolver
        resolver: Arc<TokioAsyncResolver>,
        /// Connection statistics
        stats: Arc<RwLock<ConnectionPoolStats>>,
    }

    /// Connection pool statistics
    #[derive(Debug, Clone, Default)]
    pub struct ConnectionPoolStats {
        pub total_connections: u64,
        pub active_connections: u64,
        pub idle_connections: u64,
        pub connection_errors: u64,
        pub requests_served: u64,
    }

    impl ConnectionPool {
        pub async fn new(config: ConnectionPoolConfig) -> Result<Self> {
            let resolver = TokioAsyncResolver::tokio_from_system_conf()
                .map_err(|e| NetworkError::DnsResolutionFailed(e.to_string()))?;

            Ok(Self {
                config,
                connections: Arc::new(RwLock::new(HashMap::new())),
                resolver: Arc::new(resolver),
                stats: Arc::new(RwLock::new(ConnectionPoolStats::default())),
            })
        }

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
                    for conn in host_connections.iter_mut() {
                        if !conn.in_use && conn.last_used.elapsed() < self.config.max_idle_time {
                            conn.mark_used();
                            return Ok(conn.connection.clone());
                        }
                    }
                }
            }

            // Create new connection
            self.create_connection(url, tls_config, proxy).await
        }

        /// Return a connection to the pool
        pub async fn return_connection(&self, url: &Url, _client: reqwest::Client) {
            let host_key = self.get_host_key(url);

            let mut connections = self.connections.write().await;
            if let Some(host_connections) = connections.get_mut(&host_key) {
                for conn in host_connections.iter_mut() {
                    if conn.in_use {
                        conn.mark_unused();
                        break;
                    }
                }
            }
        }

        async fn create_connection(
            &self,
            url: &Url,
            tls_config: &TlsConfig,
            proxy: Option<&ProxyConfig>,
        ) -> Result<reqwest::Client> {
            let mut client_builder = reqwest::Client::builder()
                .user_agent(format!("Zuup/{}", env!("CARGO_PKG_VERSION")))
                .timeout(self.config.connection_timeout)
                .pool_idle_timeout(Some(self.config.keep_alive_timeout))
                .tcp_keepalive(Some(Duration::from_secs(60)));

            // Configure proxy if provided
            if let Some(proxy_config) = proxy {
                let proxy = self.build_proxy(proxy_config)?;
                client_builder = client_builder.proxy(proxy);
            }

            // Configure TLS
            if url.scheme() == "https" {
                client_builder = client_builder
                    .danger_accept_invalid_certs(!tls_config.verify_certificates);
            }

            let client = client_builder
                .build()
                .map_err(|e| NetworkError::ConnectionFailed(e.to_string()))?;

            // Add to pool
            let host_key = self.get_host_key(url);
            let mut connections = self.connections.write().await;
            let host_connections = connections.entry(host_key).or_insert_with(Vec::new);
            host_connections.push(PooledConnection::new(client.clone()));

            Ok(client)
        }

        /// Build TLS connector with custom configuration
        async fn build_tls_connector(&self, tls_config: &TlsConfig) -> Result<reqwest::Client> {
            // For now, return a basic client with simple TLS configuration
            Ok(reqwest::Client::builder()
                .danger_accept_invalid_certs(!tls_config.verify_certificates)
                .build()
                .map_err(|e| NetworkError::ConnectionFailed(e.to_string()))?)
        }

        fn build_proxy(&self, proxy_config: &ProxyConfig) -> Result<Proxy> {
            let mut proxy = Proxy::all(proxy_config.url.to_string())
                .map_err(|e| NetworkError::Proxy(format!("Failed to create proxy: {}", e)))?;

            if let Some(auth) = &proxy_config.auth {
                proxy = proxy.basic_auth(&auth.username, &auth.password);
            }

            Ok(proxy)
        }

        fn get_host_key(&self, url: &Url) -> String {
            format!("{}://{}", url.scheme(), url.host_str().unwrap_or(""))
        }

        pub async fn cleanup_idle_connections(&self) {
            let mut connections = self.connections.write().await;
            for host_connections in connections.values_mut() {
                host_connections.retain(|conn| {
                    !conn.in_use && conn.last_used.elapsed() < self.config.max_idle_time
                });
            }
        }

        pub async fn connection_info(&self) -> HashMap<String, Vec<ConnectionInfo>> {
            let connections = self.connections.read().await;
            let mut info = HashMap::new();

            for (host, host_connections) in connections.iter() {
                let conn_info: Vec<ConnectionInfo> = host_connections
                    .iter()
                    .map(|conn| ConnectionInfo {
                        created_at: conn.created_at,
                        last_used: conn.last_used,
                        in_use: conn.in_use,
                        age: conn.created_at.elapsed(),
                        idle_time: conn.last_used.elapsed(),
                    })
                    .collect();
                info.insert(host.clone(), conn_info);
            }

            info
        }
    }

    /// Information about a specific connection
    #[derive(Debug, Clone)]
    pub struct ConnectionInfo {
        pub created_at: Instant,
        pub last_used: Instant,
        pub in_use: bool,
        pub age: Duration,
        pub idle_time: Duration,
    }

    /// Network client configuration
    #[derive(Debug, Clone)]
    pub struct NetworkClientConfig {
        pub user_agent: String,
        pub timeout: Duration,
        pub max_redirects: u32,
        pub proxy: Option<ProxyConfig>,
        pub tls: TlsConfig,
        pub enhanced_tls: Option<EnhancedTlsConfig>,
        pub enable_http2: bool,
        pub enable_compression: bool,
        pub default_headers: HashMap<String, String>,
        pub pool_config: ConnectionPoolConfig,
        pub connection_timeout: Duration,
        pub keep_alive_timeout: Duration,
    }

    impl Default for NetworkClientConfig {
        fn default() -> Self {
            Self {
                user_agent: format!("Zuup/{}", env!("CARGO_PKG_VERSION")),
                timeout: Duration::from_secs(30),
                max_redirects: 10,
                proxy: None,
                tls: TlsConfig::default(),
                enhanced_tls: None,
                enable_http2: true,
                enable_compression: true,
                default_headers: HashMap::new(),
                pool_config: ConnectionPoolConfig::default(),
                connection_timeout: Duration::from_secs(30),
                keep_alive_timeout: Duration::from_secs(60),
            }
        }
    }

    /// Network client for making HTTP requests with connection pooling
    pub struct NetworkClient {
        config: NetworkClientConfig,
        pool: Arc<ConnectionPool>,
        /// Proxy manager for advanced proxy features
        proxy_manager: Option<Arc<tokio::sync::Mutex<ProxyManager>>>,
        /// TLS context manager for advanced TLS features
        tls_manager: Option<Arc<tokio::sync::Mutex<TlsContextManager>>>,
    }

    impl NetworkClient {
        pub async fn new(config: NetworkClientConfig) -> Result<Self> {
            let pool = Arc::new(ConnectionPool::new(config.pool_config.clone()).await?);

            let proxy_manager = if config.proxy.is_some() {
                let proxy_config = config.proxy.as_ref().unwrap();
                let chain_config = ProxyChainConfig {
                    proxies: vec![],
                    ..Default::default()
                };
                Some(Arc::new(tokio::sync::Mutex::new(ProxyManager::new(ProxyChainConfig::default()))))
            } else {
                None
            };

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

        pub async fn with_proxy_chain(
            mut config: NetworkClientConfig,
            proxy_chain: ProxyChainConfig,
            enhanced_tls: Option<EnhancedTlsConfig>,
        ) -> Result<Self> {
            let pool = Arc::new(ConnectionPool::new(config.pool_config.clone()).await?);
            let proxy_manager = Some(Arc::new(tokio::sync::Mutex::new(ProxyManager::new(proxy_chain))));

            let tls_manager = if let Some(enhanced_tls) = enhanced_tls {
                let tls_ctx = TlsContextManager::new(enhanced_tls).await?;
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

        pub async fn with_enhanced_tls(
            mut config: NetworkClientConfig,
            tls_config: EnhancedTlsConfig,
        ) -> Result<Self> {
            config.enhanced_tls = Some(tls_config);
            Self::new(config).await
        }

        pub async fn request(
            &self,
            method: HttpMethod,
            url: &Url,
            headers: Option<HashMap<String, String>>,
            body: Option<Vec<u8>>,
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
            let mut headers = HashMap::new();
            for (key, value) in &self.config.default_headers {
                headers.insert(key.clone(), value.clone());
            }

            // Add additional headers
            if let Some(additional) = additional_headers {
                headers.extend(additional);
            }

            // Add headers to request
            for (key, value) in headers {
                request_builder = request_builder.header(&key, &value);
            }

            // Set user agent
            request_builder = request_builder.header("User-Agent", &self.config.user_agent);

            // Add body if provided
            if let Some(body_data) = body {
                request_builder = request_builder.body(body_data);
            }

            // Set timeout
            request_builder = request_builder.timeout(self.config.timeout);

            let response = request_builder
                .send()
                .await
                .map_err(|e| NetworkError::InvalidResponse(e.to_string()))?;

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

            self.pool.return_connection(url, client).await;

            Ok(HttpResponse {
                status,
                headers,
                body,
            })
        }

        pub async fn connection_info(&self) -> HashMap<String, Vec<ConnectionInfo>> {
            self.pool.connection_info().await
        }

        pub async fn health_check(&self, url: &Url) -> Result<Duration> {
            let start = Instant::now();
            let _response = self.request(HttpMethod::Head, url, None, None, None).await?;
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
        pub fn is_success(&self) -> bool {
            (200..300).contains(&self.status)
        }

        pub fn is_redirect(&self) -> bool {
            (300..400).contains(&self.status)
        }

        pub fn is_client_error(&self) -> bool {
            (400..500).contains(&self.status)
        }

        pub fn is_server_error(&self) -> bool {
            (500..600).contains(&self.status)
        }

        pub fn content_length(&self) -> Option<u64> {
            self.headers
                .get("content-length")
                .and_then(|v| v.parse().ok())
        }

        pub fn content_type(&self) -> Option<&str> {
            self.headers.get("content-type").map(|s| s.as_str())
        }

        pub fn etag(&self) -> Option<&str> {
            self.headers.get("etag").map(|s| s.as_str())
        }

        pub fn last_modified(&self) -> Option<&str> {
            self.headers.get("last-modified").map(|s| s.as_str())
        }
    }

    /// Content range information from HTTP responses
    #[derive(Debug, Clone)]
    pub struct ContentRange {
        pub start: u64,
        pub end: u64,
        pub total: Option<u64>,
    }

    impl ContentRange {
        pub fn parse(content_range: &str) -> Option<Self> {
            // Parse "bytes start-end/total" format
            if !content_range.starts_with("bytes ") {
                return None;
            }

            let range_part = &content_range[6..]; // Skip "bytes "
            let parts: Vec<&str> = range_part.split('/').collect();
            if parts.len() != 2 {
                return None;
            }

            let range_spec = parts[0];
            let total_spec = parts[1];

            let range_parts: Vec<&str> = range_spec.split('-').collect();
            if range_parts.len() != 2 {
                return None;
            }

            let start = range_parts[0].parse().ok()?;
            let end = range_parts[1].parse().ok()?;
            let total = if total_spec == "*" {
                None
            } else {
                total_spec.parse().ok()
            };

            Some(ContentRange { start, end, total })
        }
    }
}

// Re-export HTTP implementations when feature is enabled
#[cfg(feature = "http")]
pub use http_impl::*;

// Stub implementations when HTTP feature is not available
#[cfg(not(feature = "http"))]
pub mod proxy {
    //! Stub proxy module when HTTP feature is disabled
    use serde::{Deserialize, Serialize};
    
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub enum ProxyType {
        Http,
        Https,
        Socks5,
    }
    
    #[derive(Debug, Clone, Serialize, Deserialize, Default)]
    pub struct EnhancedProxyConfig {
        pub proxy_type: ProxyType,
    }
    
    impl Default for ProxyType {
        fn default() -> Self {
            ProxyType::Http
        }
    }
    
    #[derive(Debug, Clone, Serialize, Deserialize, Default)]
    pub struct ProxyChainConfig {
        pub proxies: Vec<EnhancedProxyConfig>,
    }
    
    pub struct ProxyManager;
    
    impl ProxyManager {
        pub fn new() -> Self {
            Self
        }
    }
}

#[cfg(not(feature = "http"))]
pub mod tls {
    //! Stub TLS module when HTTP feature is disabled
    use serde::{Deserialize, Serialize};
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct EnhancedTlsConfig {
        pub verify_certificates: bool,
    }
    
    pub struct TlsContextManager;
    
    impl TlsContextManager {
        pub fn new() -> Self {
            Self
        }
    }
}

#[cfg(not(feature = "http"))]
pub struct NetworkClient;

#[cfg(not(feature = "http"))]
impl NetworkClient {
    pub fn new(_config: NetworkClientConfig) -> Self {
        Self
    }
}

#[cfg(not(feature = "http"))]
pub struct NetworkClientConfig {
    pub connection_timeout: std::time::Duration,
    pub keep_alive_timeout: std::time::Duration,
}

#[cfg(not(feature = "http"))]
impl Default for NetworkClientConfig {
    fn default() -> Self {
        Self {
            connection_timeout: std::time::Duration::from_secs(30),
            keep_alive_timeout: std::time::Duration::from_secs(60),
        }
    }
}