"""SSL/TLS security utilities and certificate validation for pycurl engine."""

from __future__ import annotations

import hashlib
import logging
import ssl
from pathlib import Path
from typing import Any

import pycurl
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization

from .pycurl_models import HttpFtpConfig, SSLSecurityProfile

logger = logging.getLogger(__name__)


class SSLCertificateValidator:
    """SSL certificate validation and security utilities."""

    def __init__(self, config: HttpFtpConfig) -> None:
        """
        Initialize SSL certificate validator.

        Args:
            config: HTTP/FTP configuration with SSL settings
        """
        self.config = config

    def validate_certificate_chain(self, cert_path: Path) -> bool:
        """
        Validate a certificate chain file.

        Args:
            cert_path: Path to certificate file

        Returns:
            True if certificate chain is valid, False otherwise
        """
        try:
            if not cert_path.exists():
                logger.error(f"Certificate file does not exist: {cert_path}")
                return False

            with open(cert_path, "rb") as cert_file:
                cert_data = cert_file.read()

            # Try to parse as PEM first
            try:
                cert = x509.load_pem_x509_certificate(cert_data)
            except ValueError:
                # Try DER format
                try:
                    cert = x509.load_der_x509_certificate(cert_data)
                except ValueError as e:
                    logger.error(f"Failed to parse certificate {cert_path}: {e}")
                    return False

            # Basic certificate validation
            current_time = x509.datetime.datetime.now()
            
            if cert.not_valid_before > current_time:
                logger.error(f"Certificate {cert_path} is not yet valid")
                return False
            
            if cert.not_valid_after < current_time:
                logger.error(f"Certificate {cert_path} has expired")
                return False

            logger.info(f"Certificate {cert_path} is valid")
            return True

        except Exception as e:
            logger.error(f"Error validating certificate {cert_path}: {e}")
            return False

    def extract_public_key_hash(self, cert_path: Path) -> str | None:
        """
        Extract SHA256 hash of public key from certificate for pinning.

        Args:
            cert_path: Path to certificate file

        Returns:
            Base64-encoded SHA256 hash of public key, or None if extraction fails
        """
        try:
            if not cert_path.exists():
                logger.error(f"Certificate file does not exist: {cert_path}")
                return None

            with open(cert_path, "rb") as cert_file:
                cert_data = cert_file.read()

            # Try to parse certificate
            try:
                cert = x509.load_pem_x509_certificate(cert_data)
            except ValueError:
                try:
                    cert = x509.load_der_x509_certificate(cert_data)
                except ValueError as e:
                    logger.error(f"Failed to parse certificate {cert_path}: {e}")
                    return None

            # Extract public key
            public_key = cert.public_key()
            
            # Serialize public key in DER format
            public_key_der = public_key.public_key_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )

            # Calculate SHA256 hash
            sha256_hash = hashlib.sha256(public_key_der).digest()
            
            # Encode as base64
            import base64
            public_key_hash = base64.b64encode(sha256_hash).decode('ascii')

            logger.info(f"Extracted public key hash from {cert_path}: sha256//{public_key_hash}")
            return public_key_hash

        except Exception as e:
            logger.error(f"Error extracting public key hash from {cert_path}: {e}")
            return None

    def validate_private_key(self, key_path: Path, password: str | None = None) -> bool:
        """
        Validate a private key file.

        Args:
            key_path: Path to private key file
            password: Optional password for encrypted keys

        Returns:
            True if private key is valid, False otherwise
        """
        try:
            if not key_path.exists():
                logger.error(f"Private key file does not exist: {key_path}")
                return False

            with open(key_path, "rb") as key_file:
                key_data = key_file.read()

            # Try to load private key
            try:
                # Try PEM format first
                private_key = serialization.load_pem_private_key(
                    key_data, 
                    password=password.encode('utf-8') if password else None
                )
            except ValueError:
                try:
                    # Try DER format
                    private_key = serialization.load_der_private_key(
                        key_data,
                        password=password.encode('utf-8') if password else None
                    )
                except ValueError as e:
                    logger.error(f"Failed to parse private key {key_path}: {e}")
                    return False

            # Validate key by checking if we can get key size
            key_size = private_key.key_size
            logger.info(f"Private key {key_path} is valid (key size: {key_size} bits)")
            return True

        except Exception as e:
            logger.error(f"Error validating private key {key_path}: {e}")
            return False

    def validate_certificate_key_pair(self, cert_path: Path, key_path: Path, 
                                    key_password: str | None = None) -> bool:
        """
        Validate that a certificate and private key match.

        Args:
            cert_path: Path to certificate file
            key_path: Path to private key file
            key_password: Optional password for encrypted keys

        Returns:
            True if certificate and key match, False otherwise
        """
        try:
            # Load certificate
            with open(cert_path, "rb") as cert_file:
                cert_data = cert_file.read()

            try:
                cert = x509.load_pem_x509_certificate(cert_data)
            except ValueError:
                cert = x509.load_der_x509_certificate(cert_data)

            # Load private key
            with open(key_path, "rb") as key_file:
                key_data = key_file.read()

            try:
                private_key = serialization.load_pem_private_key(
                    key_data,
                    password=key_password.encode('utf-8') if key_password else None
                )
            except ValueError:
                private_key = serialization.load_der_private_key(
                    key_data,
                    password=key_password.encode('utf-8') if key_password else None
                )

            # Compare public keys
            cert_public_key = cert.public_key()
            private_public_key = private_key.public_key()

            # Serialize both public keys for comparison
            cert_public_der = cert_public_key.public_key_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            private_public_der = private_public_key.public_key_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )

            if cert_public_der == private_public_der:
                logger.info(f"Certificate {cert_path} and private key {key_path} match")
                return True
            else:
                logger.error(f"Certificate {cert_path} and private key {key_path} do not match")
                return False

        except Exception as e:
            logger.error(f"Error validating certificate/key pair: {e}")
            return False


class SSLSecurityManager:
    """Manages SSL security profiles and configurations."""

    def __init__(self) -> None:
        """Initialize SSL security manager."""
        self.validator = None

    def create_security_profile_from_config(self, config: HttpFtpConfig) -> SSLSecurityProfile:
        """
        Create an SSL security profile from configuration.

        Args:
            config: HTTP/FTP configuration

        Returns:
            SSL security profile based on configuration
        """
        # Determine security level based on configuration
        if config.ssl_development_mode:
            return SSLSecurityProfile.create_development_profile()
        elif config.ssl_verify_status and config.ssl_pinned_public_key:
            return SSLSecurityProfile.create_high_security_profile()
        else:
            return SSLSecurityProfile.create_balanced_profile()

    def validate_ssl_configuration(self, config: HttpFtpConfig) -> list[str]:
        """
        Validate SSL configuration and return list of issues.

        Args:
            config: HTTP/FTP configuration to validate

        Returns:
            List of validation issues (empty if no issues)
        """
        issues = []
        
        if not self.validator:
            self.validator = SSLCertificateValidator(config)

        # Check certificate files
        if config.client_cert_path:
            if not self.validator.validate_certificate_chain(config.client_cert_path):
                issues.append(f"Invalid client certificate: {config.client_cert_path}")

        # Check private key
        if config.client_key_path:
            if not self.validator.validate_private_key(
                config.client_key_path, 
                config.ssl_key_password
            ):
                issues.append(f"Invalid private key: {config.client_key_path}")

        # Check certificate/key pair match
        if config.client_cert_path and config.client_key_path:
            if not self.validator.validate_certificate_key_pair(
                config.client_cert_path,
                config.client_key_path,
                config.ssl_key_password
            ):
                issues.append("Client certificate and private key do not match")

        # Check CA certificate
        if config.ca_cert_path and config.ca_cert_path.exists():
            if not self.validator.validate_certificate_chain(config.ca_cert_path):
                issues.append(f"Invalid CA certificate: {config.ca_cert_path}")

        # Check CA certificate directory
        if config.ssl_ca_cert_dir and not config.ssl_ca_cert_dir.exists():
            issues.append(f"CA certificate directory does not exist: {config.ssl_ca_cert_dir}")

        # Check CRL file
        if config.ssl_crl_file and not config.ssl_crl_file.exists():
            issues.append(f"CRL file does not exist: {config.ssl_crl_file}")

        # Security warnings
        if not config.verify_ssl and not config.ssl_development_mode:
            issues.append("SSL verification is disabled but not in development mode - SECURITY RISK")

        if config.ssl_development_mode:
            issues.append("Development mode enabled - SSL verification disabled - NOT FOR PRODUCTION")

        # Check for weak SSL versions
        if config.ssl_version in ("SSLv2", "SSLv3", "TLSv1", "TLSv1.0"):
            issues.append(f"Weak SSL/TLS version specified: {config.ssl_version} - consider TLSv1.2 or higher")

        return issues

    def setup_curl_ssl_security(self, curl_handle: pycurl.Curl, config: HttpFtpConfig) -> None:
        """
        Setup comprehensive SSL security for a curl handle.

        Args:
            curl_handle: Curl handle to configure
            config: HTTP/FTP configuration with SSL settings
        """
        # Validate configuration first
        issues = self.validate_ssl_configuration(config)
        for issue in issues:
            if "SECURITY RISK" in issue or "NOT FOR PRODUCTION" in issue:
                logger.warning(f"SSL Security Warning: {issue}")
            else:
                logger.info(f"SSL Configuration: {issue}")

        # Apply security profile if available
        if config.ssl_security_profile:
            logger.info(f"Applying SSL security profile: {config.ssl_security_profile.name}")

        # The actual curl SSL setup is handled by the existing _setup_ssl_options methods
        # This method provides additional validation and logging

    def generate_certificate_pinning_hash(self, hostname: str, port: int = 443) -> str | None:
        """
        Generate certificate pinning hash for a remote server.

        Args:
            hostname: Server hostname
            port: Server port (default 443 for HTTPS)

        Returns:
            SHA256 hash of server's public key, or None if failed
        """
        try:
            # Get server certificate
            context = ssl.create_default_context()
            
            with ssl.create_connection((hostname, port)) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    # Get peer certificate in DER format
                    cert_der = ssock.getpeercert(binary_form=True)
                    
                    # Parse certificate
                    cert = x509.load_der_x509_certificate(cert_der)
                    
                    # Extract public key
                    public_key = cert.public_key()
                    
                    # Serialize public key in DER format
                    public_key_der = public_key.public_key_bytes(
                        encoding=serialization.Encoding.DER,
                        format=serialization.PublicFormat.SubjectPublicKeyInfo
                    )
                    
                    # Calculate SHA256 hash
                    sha256_hash = hashlib.sha256(public_key_der).digest()
                    
                    # Encode as base64
                    import base64
                    public_key_hash = base64.b64encode(sha256_hash).decode('ascii')
                    
                    logger.info(f"Generated pinning hash for {hostname}:{port}: sha256//{public_key_hash}")
                    return public_key_hash

        except Exception as e:
            logger.error(f"Failed to generate certificate pinning hash for {hostname}:{port}: {e}")
            return None


# Global SSL security manager instance
ssl_security_manager = SSLSecurityManager()