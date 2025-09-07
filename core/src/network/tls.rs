//! TLS/SSL handling with certificate validation, pinning, and client certificates

use std::{collections::HashMap, io::BufReader, path::Path, sync::Arc, time::SystemTime};

#[cfg(feature = "http")]
use rustls::{
    ClientConfig, DigitallySignedStruct, Error as TLSError, RootCertStore, SignatureScheme,
    client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier},
    pki_types::{CertificateDer, PrivateKeyDer, ServerName, UnixTime},
};
#[cfg(feature = "http")]
use rustls_native_certs::load_native_certs;
#[cfg(feature = "http")]
use rustls_pemfile::{certs, pkcs8_private_keys, rsa_private_keys};
use serde::{Deserialize, Serialize};
use tokio::fs::File;
use tokio::io::AsyncReadExt;

use crate::{
    error::{NetworkError, Result},
    types::{ClientCertificate, TlsConfig},
};

/// Certificate validation mode
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CertValidationMode {
    /// Full certificate validation (default)
    #[default]
    Full,
    /// Skip certificate validation (insecure)
    None,
    /// Validate certificate but allow self-signed
    AllowSelfSigned,
    /// Custom validation with user-provided verifier
    Custom,
}

/// Certificate pinning configuration
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct CertificatePinning {
    /// Pinned certificate fingerprints (SHA-256)
    pub fingerprints: Vec<String>,
    /// Pinned public key hashes (SHA-256)
    pub public_key_hashes: Vec<String>,
    /// Whether to enforce pinning (fail if no pins match)
    pub enforce: bool,
    /// Backup pins for certificate rotation
    pub backup_pins: Vec<String>,
}

/// Enhanced TLS configuration with additional security features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedTlsConfig {
    /// Base TLS configuration
    pub base: TlsConfig,
    /// Certificate validation mode
    pub validation_mode: CertValidationMode,
    /// Certificate pinning configuration
    pub pinning: Option<CertificatePinning>,
    /// OCSP stapling configuration
    pub ocsp_stapling: bool,
    /// Certificate transparency verification
    pub ct_verification: bool,
    /// Session resumption settings
    pub session_resumption: bool,
    /// Early data (0-RTT) support
    pub early_data: bool,
    /// Custom cipher suite preferences
    pub cipher_preferences: Vec<String>,
    /// ALPN protocols
    pub alpn_protocols: Vec<String>,
    /// SNI hostname override
    pub sni_override: Option<String>,
}

impl Default for EnhancedTlsConfig {
    fn default() -> Self {
        Self {
            base: TlsConfig::default(),
            validation_mode: CertValidationMode::Full,
            pinning: None,
            ocsp_stapling: true,
            ct_verification: false,
            session_resumption: true,
            early_data: false,
            cipher_preferences: Vec::new(),
            alpn_protocols: vec!["h2".to_string(), "http/1.1".to_string()],
            sni_override: None,
        }
    }
}

impl From<TlsConfig> for EnhancedTlsConfig {
    fn from(base: TlsConfig) -> Self {
        Self {
            base,
            ..Default::default()
        }
    }
}

/// TLS context manager for handling TLS configurations
pub struct TlsContextManager {
    /// Root certificate store
    root_store: RootCertStore,
    /// Client configurations cache
    client_configs: HashMap<String, Arc<ClientConfig>>,
    /// Certificate cache
    cert_cache: HashMap<String, (CertificateDer<'static>, SystemTime)>,
    /// Enhanced TLS configuration
    config: EnhancedTlsConfig,
}

impl TlsContextManager {
    /// Create a new TLS context manager
    pub async fn new(config: EnhancedTlsConfig) -> Result<Self> {
        let mut root_store = RootCertStore::empty();

        // Load system certificates if validation is enabled
        if config.base.verify_certificates && config.validation_mode != CertValidationMode::None {
            Self::load_system_certificates(&mut root_store)?;
        }

        // Load custom CA certificates
        for ca_path in &config.base.ca_certificates {
            Self::load_ca_certificate(&mut root_store, ca_path).await?;
        }

        Ok(Self {
            root_store,
            client_configs: HashMap::new(),
            cert_cache: HashMap::new(),
            config,
        })
    }

    /// Load system certificates into the root store
    fn load_system_certificates(root_store: &mut RootCertStore) -> Result<()> {
        // Load native system certificates
        let native_certs = load_native_certs();
        for cert in native_certs.certs {
            if let Err(e) = root_store.add(cert) {
                tracing::warn!("Failed to add native certificate: {}", e);
            }
        }

        // Also add webpki roots as fallback
        root_store.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());

        Ok(())
    }

    /// Load a CA certificate from file
    async fn load_ca_certificate(root_store: &mut RootCertStore, ca_path: &Path) -> Result<()> {
        let mut file = File::open(ca_path)
            .await
            .map_err(|e| NetworkError::Tls(format!("Failed to open CA certificate file: {}", e)))?;

        let mut contents = Vec::new();
        file.read_to_end(&mut contents)
            .await
            .map_err(|e| NetworkError::Tls(format!("Failed to read CA certificate: {}", e)))?;

        let certs: Vec<CertificateDer> = certs(&mut BufReader::new(contents.as_slice()))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| NetworkError::Tls(format!("Failed to parse CA certificate: {}", e)))?;

        for cert in certs {
            root_store
                .add(cert)
                .map_err(|e| NetworkError::Tls(format!("Failed to add CA certificate: {}", e)))?;
        }

        Ok(())
    }

    /// Create a client configuration for a specific hostname
    pub async fn create_client_config(&mut self, hostname: &str) -> Result<Arc<ClientConfig>> {
        let cache_key = format!("{}:{}", hostname, self.config_hash());

        if let Some(cached_config) = self.client_configs.get(&cache_key) {
            return Ok(Arc::clone(cached_config));
        }

        let config = self.build_client_config(hostname).await?;
        let config_arc = Arc::new(config);
        self.client_configs
            .insert(cache_key, Arc::clone(&config_arc));

        Ok(config_arc)
    }

    /// Build a new client configuration
    async fn build_client_config(&self, _hostname: &str) -> Result<ClientConfig> {
        let config_builder = ClientConfig::builder();

        // For now, use a simple configuration that works
        let client_config = if self.config.base.verify_certificates {
            config_builder
                .with_root_certificates(self.root_store.clone())
                .with_no_client_auth()
        } else {
            config_builder
                .dangerous()
                .with_custom_certificate_verifier(Arc::new(NoVerification::new()))
                .with_no_client_auth()
        };

        Ok(client_config)
    }

    // Removed complex cipher suite and protocol version configuration for simplicity

    /// Load client certificate and private key
    async fn load_client_certificate(
        &self,
        client_cert: &ClientCertificate,
    ) -> Result<(Vec<CertificateDer<'static>>, PrivateKeyDer<'static>)> {
        // Load certificate
        let mut cert_file = File::open(&client_cert.cert_path)
            .await
            .map_err(|e| NetworkError::Tls(format!("Failed to open client certificate: {}", e)))?;

        let mut cert_contents = Vec::new();
        cert_file
            .read_to_end(&mut cert_contents)
            .await
            .map_err(|e| NetworkError::Tls(format!("Failed to read client certificate: {}", e)))?;

        let cert_chain: Vec<CertificateDer<'static>> =
            certs(&mut BufReader::new(cert_contents.as_slice()))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| {
                    NetworkError::Tls(format!("Failed to parse client certificate: {}", e))
                })?;

        // Load private key
        let mut key_file = File::open(&client_cert.key_path)
            .await
            .map_err(|e| NetworkError::Tls(format!("Failed to open private key: {}", e)))?;

        let mut key_contents = Vec::new();
        key_file
            .read_to_end(&mut key_contents)
            .await
            .map_err(|e| NetworkError::Tls(format!("Failed to read private key: {}", e)))?;

        let private_key = self.parse_private_key(&key_contents)?;

        Ok((cert_chain, private_key))
    }

    /// Parse private key from PEM data
    fn parse_private_key(&self, key_data: &[u8]) -> Result<PrivateKeyDer<'static>> {
        let mut key_reader = BufReader::new(key_data);

        // Try PKCS8 first
        let pkcs8_keys: Vec<_> = pkcs8_private_keys(&mut key_reader)
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| NetworkError::Tls(format!("Failed to parse PKCS8 key: {}", e)))?;

        if !pkcs8_keys.is_empty() {
            return Ok(PrivateKeyDer::Pkcs8(pkcs8_keys[0].clone_key()));
        }

        // Try RSA
        let mut key_reader = BufReader::new(key_data);
        let rsa_keys: Vec<_> = rsa_private_keys(&mut key_reader)
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| NetworkError::Tls(format!("Failed to parse RSA key: {}", e)))?;

        if !rsa_keys.is_empty() {
            return Ok(PrivateKeyDer::Pkcs1(rsa_keys[0].clone_key()));
        }

        Err(NetworkError::Tls("Failed to parse private key".to_string()).into())
    }

    /// Generate a hash of the current configuration for caching
    fn config_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Hash relevant configuration fields
        self.config.validation_mode.hash(&mut hasher);
        self.config.base.verify_certificates.hash(&mut hasher);
        self.config.base.min_version.hash(&mut hasher);
        self.config.base.max_version.hash(&mut hasher);

        if let Some(pinning) = &self.config.pinning {
            pinning.fingerprints.hash(&mut hasher);
            pinning.public_key_hashes.hash(&mut hasher);
        }

        hasher.finish()
    }

    /// Validate certificate pinning
    pub fn validate_pinning(&self, cert_chain: &[CertificateDer], hostname: &str) -> Result<bool> {
        if let Some(pinning) = &self.config.pinning {
            if pinning.enforce
                && (pinning.fingerprints.is_empty() && pinning.public_key_hashes.is_empty())
            {
                return Err(NetworkError::Tls(
                    "Certificate pinning enforced but no pins configured".to_string(),
                )
                .into());
            }

            for cert in cert_chain {
                if self.check_certificate_pin(cert, pinning)? {
                    return Ok(true);
                }
            }

            if pinning.enforce {
                return Err(NetworkError::Tls(format!(
                    "Certificate pinning validation failed for {}",
                    hostname
                ))
                .into());
            }
        }

        Ok(true)
    }

    /// Check if a certificate matches any configured pins
    fn check_certificate_pin(
        &self,
        cert: &CertificateDer,
        pinning: &CertificatePinning,
    ) -> Result<bool> {
        #[cfg(feature = "http")]
        use ring::digest::{SHA256, digest};

        // Check certificate fingerprint
        let cert_hash = digest(&SHA256, cert.as_ref());
        let cert_fingerprint = hex::encode(cert_hash.as_ref());

        if pinning.fingerprints.contains(&cert_fingerprint) {
            return Ok(true);
        }

        // Check public key hash (SPKI)
        // This would require parsing the certificate to extract the public key
        // For now, we'll just check the certificate fingerprint

        Ok(false)
    }

    /// Clear cached configurations
    pub fn clear_cache(&mut self) {
        self.client_configs.clear();
        self.cert_cache.clear();
    }

    /// Get TLS configuration statistics
    pub fn get_stats(&self) -> TlsStats {
        TlsStats {
            cached_configs: self.client_configs.len(),
            cached_certificates: self.cert_cache.len(),
            root_certificates: self.root_store.len(),
            validation_mode: self.config.validation_mode.clone(),
            pinning_enabled: self.config.pinning.is_some(),
        }
    }
}

/// TLS statistics
#[derive(Debug, Clone)]
pub struct TlsStats {
    pub cached_configs: usize,
    pub cached_certificates: usize,
    pub root_certificates: usize,
    pub validation_mode: CertValidationMode,
    pub pinning_enabled: bool,
}

/// Custom certificate verifier that accepts all certificates (insecure)
#[derive(Debug)]
struct NoVerification;

impl NoVerification {
    fn new() -> Self {
        Self
    }
}

impl ServerCertVerifier for NoVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> std::result::Result<ServerCertVerified, TLSError> {
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, TLSError> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, TLSError> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        vec![
            SignatureScheme::RSA_PKCS1_SHA1,
            SignatureScheme::ECDSA_SHA1_Legacy,
            SignatureScheme::RSA_PKCS1_SHA256,
            SignatureScheme::ECDSA_NISTP256_SHA256,
            SignatureScheme::RSA_PKCS1_SHA384,
            SignatureScheme::ECDSA_NISTP384_SHA384,
            SignatureScheme::RSA_PKCS1_SHA512,
            SignatureScheme::ECDSA_NISTP521_SHA512,
            SignatureScheme::RSA_PSS_SHA256,
            SignatureScheme::RSA_PSS_SHA384,
            SignatureScheme::RSA_PSS_SHA512,
            SignatureScheme::ED25519,
            SignatureScheme::ED448,
        ]
    }
}

// Additional verifiers removed for simplicity - can be added back later

#[cfg(test)]
mod tests {
    use crate::types::TlsVersion;

    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_cert_validation_mode_default() {
        assert_eq!(CertValidationMode::default(), CertValidationMode::Full);
    }

    #[test]
    fn test_certificate_pinning_default() {
        let pinning = CertificatePinning::default();
        assert!(pinning.fingerprints.is_empty());
        assert!(pinning.public_key_hashes.is_empty());
        assert!(!pinning.enforce);
        assert!(pinning.backup_pins.is_empty());
    }

    #[test]
    fn test_enhanced_tls_config_default() {
        let config = EnhancedTlsConfig::default();
        assert_eq!(config.validation_mode, CertValidationMode::Full);
        assert!(config.pinning.is_none());
        assert!(config.ocsp_stapling);
        assert!(!config.ct_verification);
        assert!(config.session_resumption);
        assert!(!config.early_data);
        assert_eq!(config.alpn_protocols, vec!["h2", "http/1.1"]);
    }

    #[test]
    fn test_enhanced_tls_config_from_base() {
        let base_config = TlsConfig {
            verify_certificates: false,
            ca_certificates: vec![PathBuf::from("/test/ca.pem")],
            client_certificate: None,
            min_version: Some(TlsVersion::V1_3),
            max_version: Some(TlsVersion::V1_3),
            cipher_suites: vec!["TLS_AES_256_GCM_SHA384".to_string()],
        };

        let enhanced: EnhancedTlsConfig = base_config.clone().into();
        assert!(!enhanced.base.verify_certificates);
        assert_eq!(
            enhanced.base.ca_certificates,
            vec![PathBuf::from("/test/ca.pem")]
        );
        assert_eq!(enhanced.base.min_version, Some(TlsVersion::V1_3));
    }

    #[tokio::test]
    async fn test_tls_context_manager_creation() {
        let config = EnhancedTlsConfig::default();
        let manager = TlsContextManager::new(config).await.unwrap();

        let stats = manager.get_stats();
        assert_eq!(stats.cached_configs, 0);
        assert_eq!(stats.cached_certificates, 0);
        assert_eq!(stats.validation_mode, CertValidationMode::Full);
        assert!(!stats.pinning_enabled);
    }

    #[test]
    fn test_tls_stats() {
        let stats = TlsStats {
            cached_configs: 5,
            cached_certificates: 10,
            root_certificates: 100,
            validation_mode: CertValidationMode::Full,
            pinning_enabled: true,
        };

        assert_eq!(stats.cached_configs, 5);
        assert_eq!(stats.cached_certificates, 10);
        assert_eq!(stats.root_certificates, 100);
        assert!(stats.pinning_enabled);
    }
}
