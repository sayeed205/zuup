#!/usr/bin/env python3
"""Test script for SSL/TLS security features in pycurl implementation."""

import tempfile
from pathlib import Path

from src.zuup.engines.pycurl_models import (
    AuthConfig,
    HttpFtpConfig,
    SecureCredentialStore,
    SSLSecurityProfile,
)
from src.zuup.engines.ssl_security import SSLCertificateValidator, SSLSecurityManager


def test_ssl_security_profiles():
    """Test SSL security profiles."""
    print("Testing SSL Security Profiles...")
    
    # Test high security profile
    high_security = SSLSecurityProfile.create_high_security_profile()
    print(f"High Security Profile: {high_security.name} - {high_security.description}")
    assert high_security.verify_ssl is True
    assert high_security.ssl_version == "TLSv1.2"
    assert high_security.ssl_development_mode is False
    
    # Test development profile
    dev_profile = SSLSecurityProfile.create_development_profile()
    print(f"Development Profile: {dev_profile.name} - {dev_profile.description}")
    assert dev_profile.verify_ssl is False
    assert dev_profile.ssl_development_mode is True
    
    # Test balanced profile
    balanced = SSLSecurityProfile.create_balanced_profile()
    print(f"Balanced Profile: {balanced.name} - {balanced.description}")
    assert balanced.verify_ssl is True
    assert balanced.ssl_version == "TLSv1.1"
    
    print("✓ SSL Security Profiles test passed")


def test_secure_credential_store():
    """Test secure credential storage."""
    print("\nTesting Secure Credential Store...")
    
    # Create credential store
    store = SecureCredentialStore()
    
    # Store credentials
    store.store_credential("username", "testuser")
    store.store_credential("password", "testpass")
    store.store_credential("token", "testtoken123")
    
    # Retrieve credentials
    assert store.get_credential("username") == "testuser"
    assert store.get_credential("password") == "testpass"
    assert store.get_credential("token") == "testtoken123"
    assert store.get_credential("nonexistent") is None
    
    # Test encrypted storage (mock)
    encrypted_store = SecureCredentialStore(encrypted=True)
    encrypted_store.store_credential("secret", "mysecret")
    retrieved = encrypted_store.get_credential("secret")
    assert retrieved == "mysecret"  # Mock encryption just adds/removes prefix
    
    # Clear credentials
    store.clear_credentials()
    assert store.get_credential("username") is None
    
    print("✓ Secure Credential Store test passed")


def test_auth_config_with_secure_storage():
    """Test authentication configuration with secure storage."""
    print("\nTesting AuthConfig with Secure Storage...")
    
    # Create credential store
    store = SecureCredentialStore()
    
    # Create auth config with secure storage (no method initially)
    auth = AuthConfig(
        use_secure_storage=True,
        credential_store=store
    )
    
    # Set method and store credentials securely
    auth.method = "basic"
    auth.store_credentials_securely(username="secureuser", password="securepass")
    
    # Validate credentials are available
    auth.validate_credentials_available()  # Should not raise
    
    # Retrieve credentials
    assert auth.get_username() == "secureuser"
    assert auth.get_password() == "securepass"
    
    # Test with direct credentials (should take precedence)
    auth.username = "directuser"
    assert auth.get_username() == "directuser"
    
    # Clear direct credentials
    auth.username = None
    assert auth.get_username() == "secureuser"  # Falls back to secure storage
    
    # Test bearer token
    auth.method = "bearer"
    auth.store_credentials_securely(token="securetoken123")
    assert auth.get_token() == "securetoken123"
    
    # Test validation failure
    auth.method = "basic"
    auth.clear_credentials()
    try:
        auth.validate_credentials_available()
        assert False, "Should have raised validation error"
    except ValueError as e:
        assert "requires a username" in str(e)
    
    print("✓ AuthConfig with Secure Storage test passed")


def test_http_ftp_config_ssl_features():
    """Test HttpFtpConfig SSL features."""
    print("\nTesting HttpFtpConfig SSL Features...")
    
    # Test basic SSL configuration
    config = HttpFtpConfig(
        verify_ssl=True,
        ssl_version="TLSv1.2",
        ssl_cipher_list="HIGH:!aNULL:!eNULL",
        ssl_verify_status=True,
        ssl_pinned_public_key="sha256//abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        ssl_development_mode=False
    )
    
    assert config.verify_ssl is True
    assert config.ssl_version == "TLSv1.2"
    assert config.ssl_verify_status is True
    assert config.ssl_development_mode is False
    
    # Test applying security profiles
    config.apply_ssl_security_profile(SSLSecurityProfile.create_high_security_profile())
    assert config.ssl_security_profile.name == "high_security"
    
    # Test factory methods
    high_sec_config = HttpFtpConfig.create_with_high_security()
    assert high_sec_config.ssl_security_profile.name == "high_security"
    assert high_sec_config.verify_ssl is True
    
    dev_config = HttpFtpConfig.create_with_development_profile()
    assert dev_config.ssl_security_profile.name == "development"
    assert dev_config.ssl_development_mode is True
    
    print("✓ HttpFtpConfig SSL Features test passed")


def test_ssl_validation():
    """Test SSL validation features."""
    print("\nTesting SSL Validation...")
    
    # Test SSL version validation
    try:
        config = HttpFtpConfig(ssl_version="InvalidVersion")
        assert False, "Should have raised validation error"
    except ValueError as e:
        assert "SSL version must be one of" in str(e)
    
    # Test SSL certificate type validation
    try:
        config = HttpFtpConfig(ssl_cert_type="INVALID")
        assert False, "Should have raised validation error"
    except ValueError as e:
        assert "SSL cert/key type must be one of" in str(e)
    
    # Test SSL debug level validation
    try:
        config = HttpFtpConfig(ssl_debug_level=5)
        assert False, "Should have raised validation error"
    except ValueError as e:
        assert "SSL debug level must be between 0 and 3" in str(e)
    
    # Test valid configurations
    config = HttpFtpConfig(
        ssl_version="TLSv1.3",
        ssl_cert_type="PEM",
        ssl_key_type="DER",
        ssl_debug_level=2
    )
    assert config.ssl_version == "TLSv1.3"
    assert config.ssl_cert_type == "PEM"
    assert config.ssl_key_type == "DER"
    assert config.ssl_debug_level == 2
    
    print("✓ SSL Validation test passed")


def test_ssl_security_manager():
    """Test SSL security manager."""
    print("\nTesting SSL Security Manager...")
    
    manager = SSLSecurityManager()
    
    # Test creating security profile from config
    config = HttpFtpConfig(ssl_development_mode=True)
    profile = manager.create_security_profile_from_config(config)
    assert profile.name == "development"
    
    config = HttpFtpConfig(ssl_verify_status=True, ssl_pinned_public_key="sha256//abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890")
    profile = manager.create_security_profile_from_config(config)
    assert profile.name == "high_security"
    
    # Test configuration validation
    config = HttpFtpConfig(verify_ssl=False, ssl_development_mode=False)
    issues = manager.validate_ssl_configuration(config)
    assert any("SECURITY RISK" in issue for issue in issues)
    
    config = HttpFtpConfig(ssl_development_mode=True)
    issues = manager.validate_ssl_configuration(config)
    assert any("Development mode enabled" in issue for issue in issues)
    
    print("✓ SSL Security Manager test passed")


def main():
    """Run all SSL security tests."""
    print("Running SSL/TLS Security Features Tests...")
    print("=" * 50)
    
    try:
        test_ssl_security_profiles()
        test_secure_credential_store()
        test_auth_config_with_secure_storage()
        test_http_ftp_config_ssl_features()
        test_ssl_validation()
        test_ssl_security_manager()
        
        print("\n" + "=" * 50)
        print("✅ All SSL/TLS security features tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)