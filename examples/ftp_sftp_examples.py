#!/usr/bin/env python3
"""
Examples demonstrating FTP/SFTP functionality in the pycurl engine.

This module provides examples of how to use the enhanced FTP/SFTP features
including authentication, directory listing, and batch downloads.
"""

import asyncio
from pathlib import Path

from src.zuup.engines.http_ftp_engine import HttpFtpEngine
from src.zuup.engines.pycurl_models import (
    AuthConfig,
    AuthMethod,
    HttpFtpConfig,
    ProxyConfig,
    ProxyType,
    SshConfig,
)
from src.zuup.storage.models import DownloadTask, TaskStatus


async def example_ftp_anonymous_download():
    """Example: Anonymous FTP download."""
    print("=== Anonymous FTP Download Example ===")

    config = HttpFtpConfig(
        max_connections=4,
        ftp_use_epsv=True,
        ftp_use_eprt=True,
        auth=AuthConfig(method=AuthMethod.NONE),  # Anonymous
    )

    engine = HttpFtpEngine(config)

    # Example anonymous FTP download
    task = DownloadTask(
        id="ftp_anon_001",
        url="ftp://ftp.example.com/pub/file.zip",
        filename="file.zip",
        destination=Path("downloads/file.zip"),
        status=TaskStatus.PENDING,
    )

    print(f"Configured for anonymous FTP download: {task.url}")
    return engine, task


async def example_ftp_authenticated_download():
    """Example: Authenticated FTP download."""
    print("=== Authenticated FTP Download Example ===")

    config = HttpFtpConfig(
        max_connections=2,  # Some FTP servers limit connections
        ftp_use_epsv=True,
        ftp_use_eprt=False,  # Disable EPRT for firewall compatibility
        ftp_skip_pasv_ip=True,  # Skip PASV IP for NAT/firewall issues
        auth=AuthConfig(
            method=AuthMethod.BASIC, username="ftpuser", password="ftppass"
        ),
    )

    engine = HttpFtpEngine(config)

    task = DownloadTask(
        id="ftp_auth_001",
        url="ftp://private.example.com/files/document.pdf",
        filename="document.pdf",
        destination=Path("downloads/document.pdf"),
        status=TaskStatus.PENDING,
    )

    print(f"Configured for authenticated FTP download: {task.url}")
    return engine, task


async def example_sftp_password_download():
    """Example: SFTP download with password authentication."""
    print("=== SFTP Password Authentication Example ===")

    config = HttpFtpConfig(
        max_connections=2,  # SFTP typically uses fewer connections
        auth=AuthConfig(
            method=AuthMethod.BASIC, username="sftpuser", password="sftppass"
        ),
        ssh=SshConfig(
            known_hosts_path=Path.home() / ".ssh" / "known_hosts",
            # No key files for password auth
        ),
    )

    engine = HttpFtpEngine(config)

    task = DownloadTask(
        id="sftp_pass_001",
        url="sftp://secure.example.com/files/backup.tar.gz",
        filename="backup.tar.gz",
        destination=Path("downloads/backup.tar.gz"),
        status=TaskStatus.PENDING,
    )

    print(f"Configured for SFTP password download: {task.url}")
    return engine, task


async def example_sftp_key_download():
    """Example: SFTP download with SSH key authentication."""
    print("=== SFTP Key Authentication Example ===")

    config = HttpFtpConfig(
        max_connections=1,  # Single connection for key auth
        auth=AuthConfig(
            method=AuthMethod.NONE,  # No password needed with keys
            username="keyuser",
        ),
        ssh=SshConfig(
            private_key_path=Path.home() / ".ssh" / "id_rsa",
            public_key_path=Path.home() / ".ssh" / "id_rsa.pub",
            known_hosts_path=Path.home() / ".ssh" / "known_hosts",
            passphrase="key_passphrase",  # If key is encrypted
        ),
    )

    engine = HttpFtpEngine(config)

    task = DownloadTask(
        id="sftp_key_001",
        url="sftp://keyauth.example.com/data/dataset.zip",
        filename="dataset.zip",
        destination=Path("downloads/dataset.zip"),
        status=TaskStatus.PENDING,
    )

    print(f"Configured for SFTP key download: {task.url}")
    return engine, task


async def example_ftp_directory_listing():
    """Example: FTP directory listing and batch download."""
    print("=== FTP Directory Listing Example ===")

    config = HttpFtpConfig(
        auth=AuthConfig(method=AuthMethod.BASIC, username="ftpuser", password="ftppass")
    )

    engine = HttpFtpEngine(config)

    directory_url = "ftp://ftp.example.com/pub/files/"

    try:
        # List directory contents
        print(f"Listing directory: {directory_url}")
        files = await engine.list_ftp_directory_detailed(directory_url)

        print(f"Found {len(files)} items:")
        for file_info in files[:5]:  # Show first 5 items
            print(
                f"  - {file_info['name']} ({file_info['type']}) - {file_info.get('size', 'unknown')} bytes"
            )

        # Create batch download tasks for ZIP files
        print("\nCreating batch download tasks for ZIP files...")
        tasks = await engine.create_batch_download_tasks(directory_url, r".*\.zip$")

        print(f"Created {len(tasks)} download tasks:")
        for task in tasks[:3]:  # Show first 3 tasks
            print(f"  - {task.filename} -> {task.destination}")

    except Exception as e:
        print(f"Directory listing example (expected to fail without real server): {e}")


async def example_ftp_with_proxy():
    """Example: FTP download through proxy."""
    print("=== FTP with Proxy Example ===")

    config = HttpFtpConfig(
        auth=AuthConfig(method=AuthMethod.NONE),
        proxy=ProxyConfig(
            enabled=True,
            proxy_type=ProxyType.HTTP,
            host="proxy.company.com",
            port=8080,
            username="proxyuser",
            password="proxypass",
        ),
    )

    engine = HttpFtpEngine(config)

    task = DownloadTask(
        id="ftp_proxy_001",
        url="ftp://external.example.com/public/file.zip",
        filename="file.zip",
        destination=Path("downloads/file.zip"),
        status=TaskStatus.PENDING,
    )

    print(f"Configured for FTP download through proxy: {task.url}")
    return engine, task


async def example_ftps_secure_download():
    """Example: Secure FTPS download with SSL verification."""
    print("=== FTPS Secure Download Example ===")

    config = HttpFtpConfig(
        verify_ssl=True,
        ca_cert_path=Path("/etc/ssl/certs/ca-certificates.crt"),
        auth=AuthConfig(
            method=AuthMethod.BASIC, username="secureuser", password="securepass"
        ),
        ftp_use_epsv=True,
        ftp_use_eprt=True,
    )

    engine = HttpFtpEngine(config)

    task = DownloadTask(
        id="ftps_secure_001",
        url="ftps://secure-ftp.example.com/confidential/report.pdf",
        filename="report.pdf",
        destination=Path("downloads/report.pdf"),
        status=TaskStatus.PENDING,
    )

    print(f"Configured for secure FTPS download: {task.url}")
    return engine, task


async def main():
    """Run all FTP/SFTP examples."""
    print("FTP/SFTP Engine Examples")
    print("=" * 50)

    examples = [
        example_ftp_anonymous_download,
        example_ftp_authenticated_download,
        example_sftp_password_download,
        example_sftp_key_download,
        example_ftp_directory_listing,
        example_ftp_with_proxy,
        example_ftps_secure_download,
    ]

    for example_func in examples:
        try:
            result = await example_func()
            if result:
                engine, task = result
                print("✓ Example configured successfully")
            print()
        except Exception as e:
            print(f"✗ Example error: {e}")
            print()

    print("All examples completed!")
    print("\nNote: These examples show configuration only.")
    print("Actual downloads would require real FTP/SFTP servers.")


if __name__ == "__main__":
    asyncio.run(main())
