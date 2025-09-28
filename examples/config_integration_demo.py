#!/usr/bin/env python3
"""
Configuration Integration Demo

This example demonstrates the comprehensive configuration integration and validation
features of the pycurl HTTP/FTP engine, including:

1. Configuration mapping from TaskConfig to HttpFtpConfig
2. Configuration profiles for different use cases
3. Configuration validation with meaningful error messages
4. Hot-reload support for development
5. SSL security profiles
"""

import json
import sys
import tempfile
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from zuup.storage.models import TaskConfig, GlobalConfig, ProxyConfig as CoreProxyConfig
from zuup.engines.config_integration import ConfigurationManager
from zuup.engines.pycurl_models import HttpFtpConfig, AuthMethod


def demo_basic_configuration_mapping():
    """Demonstrate basic configuration mapping."""
    print("=== Basic Configuration Mapping ===")
    
    # Create task-specific configuration
    task_config = TaskConfig(
        max_connections=6,
        timeout=45,
        retry_attempts=4,
        headers={
            "User-Agent": "MyApp/1.0",
            "Accept": "application/octet-stream",
            "Authorization": "Bearer token123"
        },
        cookies={
            "session_id": "abc123def456",
            "preferences": "theme=dark;lang=en"
        },
        download_speed_limit=1024 * 1024,  # 1MB/s
    )
    
    # Create global configuration
    global_config = GlobalConfig(
        max_connections_per_download=8,
        user_agent="GlobalApp/2.0",
        default_download_path=Path.home() / "Downloads",
    )
    
    # Create configuration manager
    config_manager = ConfigurationManager()
    
    # Map to HttpFtpConfig
    http_config = config_manager.create_engine_config(
        task_config=task_config,
        global_config=global_config,
        validate=True
    )
    
    print(f"Mapped Configuration:")
    print(f"  Max Connections: {http_config.max_connections}")
    print(f"  Timeout: {http_config.timeout}s")
    print(f"  Retry Attempts: {http_config.retry_attempts}")
    print(f"  User Agent: {http_config.user_agent}")
    print(f"  Custom Headers: {len(http_config.custom_headers)} headers")
    print(f"  Cookies: {len(http_config.cookies)} cookies")
    print(f"  Low Speed Limit: {http_config.low_speed_limit} bytes/s")
    print()


def demo_configuration_profiles():
    """Demonstrate configuration profiles."""
    print("=== Configuration Profiles ===")
    
    config_manager = ConfigurationManager()
    
    # List available profiles
    profiles = config_manager.get_profile_names()
    print(f"Available profiles: {', '.join(profiles)}")
    print()
    
    # Demonstrate each profile
    task_config = TaskConfig(max_connections=4, timeout=30)
    
    for profile_name in profiles:
        print(f"Profile: {profile_name}")
        
        # Get profile information
        profile_info = config_manager.get_profile_info(profile_name)
        print(f"  Description: {profile_info['description']}")
        print(f"  Tags: {', '.join(profile_info['tags'])}")
        
        # Create config with profile
        config = config_manager.create_engine_config(
            task_config=task_config,
            profile_name=profile_name,
            validate=True
        )
        
        print(f"  Config Summary:")
        print(f"    Max Connections: {config.max_connections}")
        print(f"    Segment Size: {config.segment_size // 1024}KB")
        print(f"    SSL Verification: {config.verify_ssl}")
        print(f"    Development Mode: {config.ssl_development_mode}")
        print()


def demo_configuration_validation():
    """Demonstrate configuration validation."""
    print("=== Configuration Validation ===")
    
    config_manager = ConfigurationManager()
    
    # Test valid configuration
    print("Testing valid configuration:")
    valid_config = TaskConfig(
        max_connections=4,
        timeout=30,
        retry_attempts=3
    )
    
    result = config_manager.validate_task_config(valid_config)
    print(f"  Valid: {result.valid}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Warnings: {len(result.warnings)}")
    print()
    
    # Test configuration with warnings
    print("Testing configuration with warnings:")
    warning_config = TaskConfig(
        max_connections=20,  # Very high - should warn
        timeout=5,  # Very low - should warn
        retry_attempts=15,  # Very high - should warn
    )
    
    result = config_manager.validate_task_config(warning_config)
    print(f"  Valid: {result.valid}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Warnings: {len(result.warnings)}")
    
    if result.warnings:
        print("  Warning messages:")
        for warning in result.warnings:
            print(f"    - {warning}")
    print()


def demo_ssl_security_profiles():
    """Demonstrate SSL security profiles."""
    print("=== SSL Security Profiles ===")
    
    config_manager = ConfigurationManager()
    task_config = TaskConfig()
    
    # High security profile
    print("High Security Profile:")
    high_sec_config = config_manager.create_engine_config(
        task_config=task_config,
        profile_name="conservative",  # Uses high security SSL
        validate=True
    )
    
    print(f"  SSL Verification: {high_sec_config.verify_ssl}")
    print(f"  SSL Version: {high_sec_config.ssl_version}")
    print(f"  SSL Cipher List: {high_sec_config.ssl_cipher_list[:50]}...")
    print(f"  OCSP Verification: {high_sec_config.ssl_verify_status}")
    print()
    
    # Development profile
    print("Development Profile:")
    dev_config = config_manager.create_engine_config(
        task_config=task_config,
        profile_name="development",
        validate=True
    )
    
    print(f"  SSL Verification: {dev_config.verify_ssl}")
    print(f"  Development Mode: {dev_config.ssl_development_mode}")
    print(f"  SSL Version: {dev_config.ssl_version}")
    print()


def demo_proxy_configuration():
    """Demonstrate proxy configuration."""
    print("=== Proxy Configuration ===")
    
    # Create task config with proxy
    proxy_config = CoreProxyConfig(
        enabled=True,
        http_proxy="http://proxyuser:proxypass@proxy.example.com:8080",
        https_proxy="https://proxyuser:proxypass@proxy.example.com:8080",
        username="proxyuser",
        password="proxypass"
    )
    
    task_config = TaskConfig(proxy=proxy_config)
    
    config_manager = ConfigurationManager()
    config = config_manager.create_engine_config(task_config, validate=True)
    
    print(f"  Proxy Enabled: {config.proxy.enabled}")
    print(f"  Proxy Type: {config.proxy.proxy_type.value}")
    print(f"  Proxy Host: {config.proxy.host}")
    print(f"  Proxy Port: {config.proxy.port}")
    print(f"  Proxy URL: {config.proxy.proxy_url}")
    print()


def demo_performance_comparison():
    """Demonstrate performance differences between profiles."""
    print("=== Performance Profile Comparison ===")
    
    config_manager = ConfigurationManager()
    task_config = TaskConfig(max_connections=8)
    
    profiles_to_compare = ["high_performance", "conservative", "mobile"]
    
    print("Profile Performance Characteristics:")
    print(f"{'Profile':<15} {'Connections':<11} {'Segment Size':<12} {'Buffer Size':<11} {'Compression':<11}")
    print("-" * 70)
    
    for profile_name in profiles_to_compare:
        config = config_manager.create_engine_config(
            task_config=task_config,
            profile_name=profile_name,
            validate=True
        )
        
        segment_size_kb = config.segment_size // 1024
        buffer_size_kb = config.buffer_size // 1024
        
        print(f"{profile_name:<15} {config.max_connections:<11} {segment_size_kb}KB{'':<7} {buffer_size_kb}KB{'':<7} {config.enable_compression}")
    
    print()


def main():
    """Run all configuration integration demos."""
    print("🔧 Configuration Integration and Validation Demo")
    print("=" * 60)
    print()
    
    demos = [
        demo_basic_configuration_mapping,
        demo_configuration_profiles,
        demo_configuration_validation,
        demo_ssl_security_profiles,
        demo_proxy_configuration,
        demo_performance_comparison,
    ]
    
    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"❌ Demo {demo.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("🎉 Configuration integration demo completed!")
    print()
    print("Key Features Demonstrated:")
    print("✓ TaskConfig to HttpFtpConfig mapping")
    print("✓ Configuration profiles for different use cases")
    print("✓ Comprehensive validation with meaningful error messages")
    print("✓ SSL security profiles")
    print("✓ Proxy configuration")
    print("✓ Performance profile comparison")


if __name__ == "__main__":
    main()