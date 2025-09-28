"""Cookie management for media downloads."""

import json
import logging
import sqlite3
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
import asyncio
from datetime import datetime, timezone
import base64
import os
import platform

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CookieInfo(BaseModel):
    """Information about a browser cookie."""
    
    name: str
    value: str
    domain: str
    path: str = "/"
    secure: bool = False
    http_only: bool = False
    expires: Optional[float] = None


class CookieManager:
    """
    Enhanced cookie manager for media downloads with browser integration.
    
    This class provides comprehensive functionality to extract cookies from browsers,
    manage cookie files, handle session persistence, and support secure cookie storage.
    """
    
    def __init__(self, cookies_file: Optional[Path] = None) -> None:
        """
        Initialize cookie manager.
        
        Args:
            cookies_file: Path to cookies file (Netscape format)
        """
        self.cookies_file = cookies_file
        self.logger = logging.getLogger(__name__)
        
        # Browser cookie database paths
        self._browser_paths = self._get_browser_cookie_paths()
        
        # Session management
        self._session_cookies: Dict[str, Dict[str, str]] = {}
        self._cookie_cache: Dict[str, Dict[str, CookieInfo]] = {}
        self._cache_expiry: Dict[str, datetime] = {}
        
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()
    
    async def load_cookies(self, domain: Optional[str] = None) -> Dict[str, str]:
        """
        Load cookies from file with optional domain filtering.
        
        Args:
            domain: Optional domain to filter cookies for
            
        Returns:
            Dictionary of cookie name -> value pairs
        """
        async with self._lock:
            if not self.cookies_file or not self.cookies_file.exists():
                return {}
            
            # Check cache first
            cache_key = domain or "all"
            if cache_key in self._cookie_cache:
                cache_time = self._cache_expiry.get(cache_key)
                if cache_time and datetime.now(timezone.utc) < cache_time:
                    return {name: cookie.value for name, cookie in self._cookie_cache[cache_key].items()}
            
            cookies = {}
            cookie_objects = {}
            
            try:
                with open(self.cookies_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        
                        # Skip comments and empty lines
                        if not line or line.startswith('#'):
                            continue
                        
                        # Parse Netscape cookie format
                        parts = line.split('\t')
                        if len(parts) >= 7:
                            cookie_domain, flag, path, secure, expires, name, value = parts[:7]
                            
                            # Apply domain filter if specified
                            if domain and not self._domain_matches(cookie_domain, domain):
                                continue
                            
                            cookies[name] = value
                            
                            # Create cookie object for cache
                            cookie_info = CookieInfo(
                                name=name,
                                value=value,
                                domain=cookie_domain,
                                path=path,
                                secure=secure.lower() == 'true',
                                expires=float(expires) if expires != '0' else None
                            )
                            cookie_objects[name] = cookie_info
                
                # Update cache
                self._cookie_cache[cache_key] = cookie_objects
                from datetime import timedelta
                self._cache_expiry[cache_key] = datetime.now(timezone.utc) + timedelta(minutes=5)
                        
            except Exception as e:
                self.logger.error(f"Failed to load cookies from {self.cookies_file}: {e}")
            
            return cookies
    
    async def save_cookies(
        self, 
        cookies: Dict[str, str], 
        domain: str = ".example.com",
        secure: bool = False,
        expires: Optional[float] = None
    ) -> None:
        """
        Save cookies to file in Netscape format with enhanced metadata.
        
        Args:
            cookies: Dictionary of cookie name -> value pairs
            domain: Domain for the cookies
            secure: Whether cookies should be marked as secure
            expires: Expiration timestamp (0 for session cookies)
        """
        async with self._lock:
            if not self.cookies_file:
                return
            
            try:
                # Ensure parent directory exists
                self.cookies_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Read existing cookies to preserve them
                existing_cookies = []
                if self.cookies_file.exists():
                    with open(self.cookies_file, 'r', encoding='utf-8') as f:
                        existing_cookies = f.readlines()
                
                # Write updated cookies
                with open(self.cookies_file, 'w', encoding='utf-8') as f:
                    f.write("# Netscape HTTP Cookie File\n")
                    f.write("# This is a generated file! Do not edit.\n")
                    f.write(f"# Generated on {datetime.now(timezone.utc).isoformat()}\n\n")
                    
                    # Write existing cookies (skip duplicates)
                    existing_names = set()
                    for line in existing_cookies:
                        if line.startswith('#') or not line.strip():
                            continue
                        parts = line.strip().split('\t')
                        if len(parts) >= 7:
                            existing_name = parts[5]
                            if existing_name not in cookies:
                                f.write(line)
                                existing_names.add(existing_name)
                    
                    # Write new/updated cookies
                    expires_str = str(int(expires)) if expires else "0"
                    secure_str = "TRUE" if secure else "FALSE"
                    
                    for name, value in cookies.items():
                        # Format: domain, flag, path, secure, expires, name, value
                        f.write(f"{domain}\tTRUE\t/\t{secure_str}\t{expires_str}\t{name}\t{value}\n")
                
                # Clear cache to force reload
                self._cookie_cache.clear()
                self._cache_expiry.clear()
                        
            except Exception as e:
                self.logger.error(f"Failed to save cookies to {self.cookies_file}: {e}")
    
    async def extract_browser_cookies(self, browser: str, domain: Optional[str] = None) -> Dict[str, str]:
        """
        Extract cookies from browser database with enhanced error handling.
        
        Args:
            browser: Browser name ('chrome', 'firefox', 'edge', 'safari')
            domain: Optional domain filter
            
        Returns:
            Dictionary of cookie name -> value pairs
        """
        async with self._lock:
            browser_lower = browser.lower()
            
            try:
                if browser_lower in ['chrome', 'chromium']:
                    return await self._extract_chrome_cookies(domain)
                elif browser_lower == 'firefox':
                    return await self._extract_firefox_cookies(domain)
                elif browser_lower == 'edge':
                    return await self._extract_edge_cookies(domain)
                elif browser_lower == 'safari':
                    return await self._extract_safari_cookies(domain)
                else:
                    self.logger.warning(f"Unsupported browser: {browser}")
                    return {}
            except Exception as e:
                self.logger.error(f"Failed to extract cookies from {browser}: {e}")
                return {}
    
    async def get_cookies_for_domain(self, domain: str) -> Dict[str, str]:
        """
        Get cookies for a specific domain from all available sources.
        
        Args:
            domain: Domain to get cookies for
            
        Returns:
            Dictionary of cookie name -> value pairs
        """
        all_cookies = {}
        
        # Try to load from file first
        if self.cookies_file and self.cookies_file.exists():
            try:
                file_cookies = await self.load_cookies(domain)
                all_cookies.update(file_cookies)
            except Exception as e:
                self.logger.error(f"Failed to load cookies for {domain}: {e}")
        
        # Check session cookies
        if domain in self._session_cookies:
            all_cookies.update(self._session_cookies[domain])
        
        # Try to extract from browsers
        browsers = ['chrome', 'firefox', 'edge']
        for browser in browsers:
            try:
                browser_cookies = await self.extract_browser_cookies(browser, domain)
                all_cookies.update(browser_cookies)
            except Exception as e:
                self.logger.debug(f"Failed to extract {browser} cookies for {domain}: {e}")
        
        return all_cookies
    
    async def set_session_cookies(self, domain: str, cookies: Dict[str, str]) -> None:
        """
        Set session cookies for a domain.
        
        Args:
            domain: Domain to set cookies for
            cookies: Dictionary of cookie name -> value pairs
        """
        async with self._lock:
            if domain not in self._session_cookies:
                self._session_cookies[domain] = {}
            self._session_cookies[domain].update(cookies)
            self.logger.debug(f"Set {len(cookies)} session cookies for {domain}")
    
    async def clear_session_cookies(self, domain: Optional[str] = None) -> None:
        """
        Clear session cookies for a domain or all domains.
        
        Args:
            domain: Domain to clear cookies for, or None to clear all
        """
        async with self._lock:
            if domain:
                self._session_cookies.pop(domain, None)
                self.logger.debug(f"Cleared session cookies for {domain}")
            else:
                self._session_cookies.clear()
                self.logger.debug("Cleared all session cookies")
    
    async def refresh_cookies(self, domain: str, browser: Optional[str] = None) -> Dict[str, str]:
        """
        Refresh cookies for a domain by re-extracting from browser.
        
        Args:
            domain: Domain to refresh cookies for
            browser: Specific browser to extract from, or None for all
            
        Returns:
            Dictionary of refreshed cookies
        """
        # Clear cache for this domain
        cache_key = domain
        self._cookie_cache.pop(cache_key, None)
        self._cache_expiry.pop(cache_key, None)
        
        if browser:
            return await self.extract_browser_cookies(browser, domain)
        else:
            return await self.get_cookies_for_domain(domain)
    
    def _domain_matches(self, cookie_domain: str, target_domain: str) -> bool:
        """
        Check if a cookie domain matches a target domain.
        
        Args:
            cookie_domain: Domain from cookie
            target_domain: Target domain to match
            
        Returns:
            True if domains match
        """
        # Remove leading dot from cookie domain
        if cookie_domain.startswith('.'):
            cookie_domain = cookie_domain[1:]
        
        # Exact match
        if cookie_domain == target_domain:
            return True
        
        # Subdomain match
        if target_domain.endswith('.' + cookie_domain):
            return True
        
        # Parent domain match
        if cookie_domain.endswith('.' + target_domain):
            return True
        
        return False
    
    def _get_browser_cookie_paths(self) -> Dict[str, List[Path]]:
        """Get paths to browser cookie databases."""
        import platform
        
        system = platform.system()
        home = Path.home()
        
        paths = {}
        
        if system == "Windows":
            # Windows paths
            paths['chrome'] = [
                home / "AppData/Local/Google/Chrome/User Data/Default/Cookies",
                home / "AppData/Local/Google/Chrome/User Data/Default/Network/Cookies",
            ]
            paths['edge'] = [
                home / "AppData/Local/Microsoft/Edge/User Data/Default/Cookies",
            ]
            paths['firefox'] = [
                home / "AppData/Roaming/Mozilla/Firefox/Profiles/*/cookies.sqlite",
            ]
            
        elif system == "Darwin":  # macOS
            paths['chrome'] = [
                home / "Library/Application Support/Google/Chrome/Default/Cookies",
            ]
            paths['edge'] = [
                home / "Library/Application Support/Microsoft Edge/Default/Cookies",
            ]
            paths['firefox'] = [
                home / "Library/Application Support/Firefox/Profiles/*/cookies.sqlite",
            ]
            paths['safari'] = [
                home / "Library/Cookies/Cookies.binarycookies",
            ]
            
        else:  # Linux
            paths['chrome'] = [
                home / ".config/google-chrome/Default/Cookies",
                home / ".config/chromium/Default/Cookies",
            ]
            paths['firefox'] = [
                home / ".mozilla/firefox/*/cookies.sqlite",
            ]
        
        return paths
    
    async def _extract_chrome_cookies(self, domain: Optional[str] = None) -> Dict[str, str]:
        """Extract cookies from Chrome/Chromium with enhanced security handling."""
        cookies = {}
        
        for cookie_path in self._browser_paths.get('chrome', []):
            if cookie_path.exists():
                try:
                    cookies.update(await self._read_chrome_cookies(cookie_path, domain))
                    break  # Use first available database
                except Exception as e:
                    self.logger.debug(f"Failed to read Chrome cookies from {cookie_path}: {e}")
        
        return cookies
    
    async def _extract_firefox_cookies(self, domain: Optional[str] = None) -> Dict[str, str]:
        """Extract cookies from Firefox with profile detection."""
        cookies = {}
        
        # Firefox uses glob patterns, need to find actual profile directories
        firefox_base = Path.home() / ".mozilla/firefox" if Path.home().exists() else None
        
        if not firefox_base or not firefox_base.exists():
            return cookies
        
        # Find profile directories
        for profile_dir in firefox_base.glob("*.default*"):
            cookie_file = profile_dir / "cookies.sqlite"
            if cookie_file.exists():
                try:
                    cookies.update(await self._read_firefox_cookies(cookie_file, domain))
                    break  # Use first available database
                except Exception as e:
                    self.logger.debug(f"Failed to read Firefox cookies from {cookie_file}: {e}")
        
        return cookies
    
    async def _extract_edge_cookies(self, domain: Optional[str] = None) -> Dict[str, str]:
        """Extract cookies from Microsoft Edge."""
        cookies = {}
        
        for cookie_path in self._browser_paths.get('edge', []):
            if cookie_path.exists():
                try:
                    # Edge uses same format as Chrome
                    cookies.update(await self._read_chrome_cookies(cookie_path, domain))
                    break
                except Exception as e:
                    self.logger.debug(f"Failed to read Edge cookies from {cookie_path}: {e}")
        
        return cookies
    
    async def _extract_safari_cookies(self, domain: Optional[str] = None) -> Dict[str, str]:
        """Extract cookies from Safari (macOS only)."""
        # Safari uses a binary format that's more complex to parse
        # For now, return empty dict - would need specialized library
        self.logger.warning("Safari cookie extraction not implemented")
        return {}
    
    async def _read_chrome_cookies(self, cookie_path: Path, domain: Optional[str] = None) -> Dict[str, str]:
        """Read cookies from Chrome/Chromium database with secure handling."""
        cookies = {}
        temp_db = None
        
        try:
            # Create a temporary copy to avoid locking issues
            with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as temp_file:
                temp_db = Path(temp_file.name)
                shutil.copy2(cookie_path, temp_db)
            
            # Chrome cookies are stored in SQLite database
            conn = sqlite3.connect(str(temp_db))
            cursor = conn.cursor()
            
            # Query cookies with additional fields for better filtering
            query = """
                SELECT name, value, host_key, path, expires_utc, is_secure, is_httponly 
                FROM cookies
            """
            params = []
            
            if domain:
                query += " WHERE host_key LIKE ? OR host_key LIKE ?"
                params.extend([f"%{domain}%", f"%.{domain}"])
            
            cursor.execute(query, params)
            
            for name, value, host_key, path, expires_utc, is_secure, is_httponly in cursor.fetchall():
                # Check if cookie is expired
                if expires_utc and expires_utc > 0:
                    # Chrome stores time in microseconds since Windows epoch
                    # Convert to Unix timestamp
                    unix_timestamp = (expires_utc - 11644473600000000) / 1000000
                    if unix_timestamp < datetime.now(timezone.utc).timestamp():
                        continue  # Skip expired cookies
                
                # Apply domain filtering
                if domain and not self._domain_matches(host_key, domain):
                    continue
                
                # Decrypt value if needed (Chrome encrypts cookies on some platforms)
                decrypted_value = await self._decrypt_chrome_cookie(value)
                cookies[name] = decrypted_value or value
            
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to read Chrome cookies: {e}")
        finally:
            # Clean up temporary file
            if temp_db and temp_db.exists():
                try:
                    temp_db.unlink()
                except Exception:
                    pass
        
        return cookies
    
    async def _read_firefox_cookies(self, cookie_path: Path, domain: Optional[str] = None) -> Dict[str, str]:
        """Read cookies from Firefox database with enhanced filtering."""
        cookies = {}
        temp_db = None
        
        try:
            # Create a temporary copy to avoid locking issues
            with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as temp_file:
                temp_db = Path(temp_file.name)
                shutil.copy2(cookie_path, temp_db)
            
            conn = sqlite3.connect(str(temp_db))
            cursor = conn.cursor()
            
            # Query with additional fields for better filtering
            query = """
                SELECT name, value, host, path, expiry, isSecure, isHttpOnly 
                FROM moz_cookies
            """
            params = []
            
            if domain:
                query += " WHERE host LIKE ? OR host LIKE ?"
                params.extend([f"%{domain}%", f"%.{domain}"])
            
            cursor.execute(query, params)
            
            for name, value, host, path, expiry, is_secure, is_httponly in cursor.fetchall():
                # Check if cookie is expired
                if expiry and expiry > 0:
                    if expiry < datetime.now(timezone.utc).timestamp():
                        continue  # Skip expired cookies
                
                # Apply domain filtering
                if domain and not self._domain_matches(host, domain):
                    continue
                
                cookies[name] = value
            
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to read Firefox cookies: {e}")
        finally:
            # Clean up temporary file
            if temp_db and temp_db.exists():
                try:
                    temp_db.unlink()
                except Exception:
                    pass
        
        return cookies
    
    async def _decrypt_chrome_cookie(self, encrypted_value: bytes) -> Optional[str]:
        """
        Decrypt Chrome cookie value if encrypted.
        
        Args:
            encrypted_value: Encrypted cookie value
            
        Returns:
            Decrypted value or None if decryption fails
        """
        try:
            # Chrome cookie decryption is platform-specific and complex
            # This is a simplified implementation that handles basic cases
            
            if not isinstance(encrypted_value, bytes):
                return str(encrypted_value)  # Already decrypted
            
            # Check if value starts with encryption prefix
            if encrypted_value.startswith(b'v10') or encrypted_value.startswith(b'v11'):
                # This would require platform-specific decryption libraries
                # For now, return None to indicate decryption is needed but not available
                self.logger.debug("Chrome cookie decryption not implemented for this platform")
                return None
            
            # Try to decode as UTF-8 (unencrypted cookies)
            return encrypted_value.decode('utf-8', errors='ignore')
            
        except Exception as e:
            self.logger.debug(f"Failed to decrypt Chrome cookie: {e}")
            return None


class AuthenticationError(Exception):
    """Authentication-related error."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, auth_method: Optional[str] = None):
        super().__init__(message)
        self.error_code = error_code
        self.auth_method = auth_method


class CredentialManager:
    """
    Manages secure credential storage and retrieval.
    """
    
    def __init__(self, storage_path: Optional[Path] = None) -> None:
        """
        Initialize credential manager.
        
        Args:
            storage_path: Path to secure credential storage
        """
        self.storage_path = storage_path or Path.home() / ".zuup" / "credentials"
        self.logger = logging.getLogger(__name__)
        self._ensure_storage_directory()
    
    def _ensure_storage_directory(self) -> None:
        """Ensure credential storage directory exists with proper permissions."""
        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            # Set restrictive permissions on Unix-like systems
            if platform.system() != "Windows":
                os.chmod(self.storage_path, 0o700)
        except Exception as e:
            self.logger.error(f"Failed to create credential storage directory: {e}")
    
    async def store_credentials(self, service: str, username: str, password: str) -> None:
        """
        Store credentials securely.
        
        Args:
            service: Service identifier
            username: Username
            password: Password
        """
        try:
            # Simple base64 encoding for now - in production, use proper encryption
            encoded_password = base64.b64encode(password.encode()).decode()
            
            credential_file = self.storage_path / f"{service}.json"
            credentials = {
                "username": username,
                "password": encoded_password,
                "stored_at": datetime.now(timezone.utc).isoformat()
            }
            
            with open(credential_file, 'w') as f:
                json.dump(credentials, f)
            
            # Set restrictive permissions
            if platform.system() != "Windows":
                os.chmod(credential_file, 0o600)
                
        except Exception as e:
            self.logger.error(f"Failed to store credentials for {service}: {e}")
            raise AuthenticationError(f"Failed to store credentials: {e}")
    
    async def retrieve_credentials(self, service: str) -> Optional[Dict[str, str]]:
        """
        Retrieve stored credentials.
        
        Args:
            service: Service identifier
            
        Returns:
            Dictionary with username and password, or None if not found
        """
        try:
            credential_file = self.storage_path / f"{service}.json"
            if not credential_file.exists():
                return None
            
            with open(credential_file, 'r') as f:
                credentials = json.load(f)
            
            # Decode password
            password = base64.b64decode(credentials["password"]).decode()
            
            return {
                "username": credentials["username"],
                "password": password
            }
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve credentials for {service}: {e}")
            return None
    
    async def delete_credentials(self, service: str) -> None:
        """
        Delete stored credentials.
        
        Args:
            service: Service identifier
        """
        try:
            credential_file = self.storage_path / f"{service}.json"
            if credential_file.exists():
                credential_file.unlink()
        except Exception as e:
            self.logger.error(f"Failed to delete credentials for {service}: {e}")


class AuthenticationManager:
    """
    Enhanced authentication manager for media downloads.
    
    This class handles various authentication methods including
    username/password, cookies, OAuth tokens, and credential refresh.
    """
    
    def __init__(self, cookies_file: Optional[Path] = None) -> None:
        """
        Initialize authentication manager.
        
        Args:
            cookies_file: Path to cookies file
        """
        self.logger = logging.getLogger(__name__)
        self.cookie_manager = CookieManager(cookies_file)
        self.credential_manager = CredentialManager()
        self._auth_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_expiry: Dict[str, datetime] = {}
    
    async def setup_authentication(self, auth_config: Dict[str, Any], url: str) -> Dict[str, Any]:
        """
        Setup authentication based on configuration with enhanced error handling.
        
        Args:
            auth_config: Authentication configuration
            url: URL being accessed (for domain-specific auth)
            
        Returns:
            yt-dlp compatible authentication options
            
        Raises:
            AuthenticationError: If authentication setup fails
        """
        options = {}
        auth_method = auth_config.get('method', 'none')
        
        try:
            if auth_method == 'username_password':
                username = auth_config.get('username')
                password = auth_config.get('password')
                
                if not username or not password:
                    # Try to retrieve from credential manager
                    from urllib.parse import urlparse
                    domain = urlparse(url).netloc
                    stored_creds = await self.credential_manager.retrieve_credentials(domain)
                    if stored_creds:
                        username = stored_creds['username']
                        password = stored_creds['password']
                
                if username and password:
                    options['username'] = username
                    options['password'] = password
                else:
                    raise AuthenticationError("Username and password required", auth_method=auth_method)
            
            elif auth_method == 'cookies':
                cookies_file = auth_config.get('cookies_file')
                if cookies_file:
                    options['cookiefile'] = str(cookies_file)
                    
                    # Also extract browser cookies if available
                    from urllib.parse import urlparse
                    domain = urlparse(url).netloc
                    browser_cookies = await self.cookie_manager.get_cookies_for_domain(domain)
                    if browser_cookies:
                        # Save browser cookies to file for yt-dlp
                        await self.cookie_manager.save_cookies(browser_cookies, domain)
                else:
                    raise AuthenticationError("Cookies file required", auth_method=auth_method)
            
            elif auth_method == 'netrc':
                netrc_file = auth_config.get('netrc_file')
                if netrc_file:
                    if not Path(netrc_file).exists():
                        raise AuthenticationError(f"Netrc file not found: {netrc_file}", auth_method=auth_method)
                    options['usenetrc'] = True
                    options['netrc_location'] = str(netrc_file)
                else:
                    # Use default netrc location
                    default_netrc = Path.home() / ".netrc"
                    if default_netrc.exists():
                        options['usenetrc'] = True
                    else:
                        raise AuthenticationError("No netrc file found", auth_method=auth_method)
            
            elif auth_method == 'oauth':
                oauth_token = auth_config.get('oauth_token')
                if oauth_token:
                    options['oauth_token'] = oauth_token
                else:
                    raise AuthenticationError("OAuth token required", auth_method=auth_method)
            
            # Cache successful authentication setup
            cache_key = f"{auth_method}:{url}"
            self._auth_cache[cache_key] = options.copy()
            from datetime import timedelta
            self._cache_expiry[cache_key] = datetime.now(timezone.utc) + timedelta(minutes=30)
            
        except AuthenticationError:
            raise
        except Exception as e:
            raise AuthenticationError(f"Authentication setup failed: {e}", auth_method=auth_method)
        
        return options
    
    async def extract_cookies_for_url(self, url: str, browser: Optional[str] = None) -> Dict[str, str]:
        """
        Extract cookies for a specific URL with caching.
        
        Args:
            url: URL to extract cookies for
            browser: Specific browser to extract from
            
        Returns:
            Dictionary of cookies for the URL
        """
        from urllib.parse import urlparse
        
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            
            # Check cache first
            cache_key = f"cookies:{domain}:{browser or 'all'}"
            if cache_key in self._auth_cache:
                cache_time = self._cache_expiry.get(cache_key)
                if cache_time and datetime.now(timezone.utc) < cache_time:
                    return self._auth_cache[cache_key]
            
            if browser:
                cookies = await self.cookie_manager.extract_browser_cookies(browser, domain)
            else:
                cookies = await self.cookie_manager.get_cookies_for_domain(domain)
            
            # Cache the result
            self._auth_cache[cache_key] = cookies
            from datetime import timedelta
            self._cache_expiry[cache_key] = datetime.now(timezone.utc) + timedelta(minutes=10)
            
            return cookies
                
        except Exception as e:
            self.logger.error(f"Failed to extract cookies for {url}: {e}")
            return {}
    
    async def validate_authentication(self, auth_config: Dict[str, Any], url: Optional[str] = None) -> bool:
        """
        Validate authentication configuration with enhanced checks.
        
        Args:
            auth_config: Authentication configuration to validate
            url: Optional URL for domain-specific validation
            
        Returns:
            True if configuration is valid
        """
        auth_method = auth_config.get('method', 'none')
        
        try:
            if auth_method == 'none':
                return True
            
            elif auth_method == 'username_password':
                username = auth_config.get('username')
                password = auth_config.get('password')
                
                if not (username and password) and url:
                    # Check if credentials are stored
                    from urllib.parse import urlparse
                    domain = urlparse(url).netloc
                    stored_creds = await self.credential_manager.retrieve_credentials(domain)
                    return stored_creds is not None
                
                return bool(username and password)
            
            elif auth_method == 'cookies':
                cookies_file = auth_config.get('cookies_file')
                if cookies_file and Path(cookies_file).exists():
                    return True
                
                # Check if browser cookies are available
                if url:
                    from urllib.parse import urlparse
                    domain = urlparse(url).netloc
                    browser_cookies = await self.cookie_manager.get_cookies_for_domain(domain)
                    return len(browser_cookies) > 0
                
                return False
            
            elif auth_method == 'netrc':
                netrc_file = auth_config.get('netrc_file')
                if netrc_file:
                    return Path(netrc_file).exists()
                else:
                    # Check default netrc location
                    return (Path.home() / '.netrc').exists()
            
            elif auth_method == 'oauth':
                return bool(auth_config.get('oauth_token'))
            
            return False
            
        except Exception as e:
            self.logger.error(f"Authentication validation failed: {e}")
            return False
    
    async def refresh_authentication(self, auth_config: Dict[str, Any], url: str) -> Dict[str, Any]:
        """
        Refresh authentication credentials.
        
        Args:
            auth_config: Current authentication configuration
            url: URL being accessed
            
        Returns:
            Refreshed authentication options
            
        Raises:
            AuthenticationError: If refresh fails
        """
        auth_method = auth_config.get('method', 'none')
        
        try:
            # Clear cache for this URL
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            cache_keys_to_remove = [
                key for key in self._auth_cache.keys() 
                if domain in key or url in key
            ]
            for key in cache_keys_to_remove:
                self._auth_cache.pop(key, None)
                self._cache_expiry.pop(key, None)
            
            if auth_method == 'cookies':
                # Refresh cookies from browser
                refreshed_cookies = await self.cookie_manager.refresh_cookies(domain)
                if refreshed_cookies:
                    # Update cookies file
                    await self.cookie_manager.save_cookies(refreshed_cookies, domain)
                    self.logger.info(f"Refreshed {len(refreshed_cookies)} cookies for {domain}")
                else:
                    raise AuthenticationError("No cookies available for refresh", auth_method=auth_method)
            
            elif auth_method == 'oauth':
                # OAuth token refresh would be platform-specific
                self.logger.warning("OAuth token refresh not implemented")
                raise AuthenticationError("OAuth refresh not implemented", auth_method=auth_method)
            
            # Re-setup authentication with refreshed credentials
            return await self.setup_authentication(auth_config, url)
            
        except AuthenticationError:
            raise
        except Exception as e:
            raise AuthenticationError(f"Authentication refresh failed: {e}", auth_method=auth_method)
    
    async def handle_authentication_error(
        self, 
        error: Exception, 
        auth_config: Dict[str, Any], 
        url: str
    ) -> Dict[str, Any]:
        """
        Handle authentication errors with automatic retry strategies.
        
        Args:
            error: Authentication error that occurred
            auth_config: Current authentication configuration
            url: URL that failed authentication
            
        Returns:
            Updated authentication options for retry
            
        Raises:
            AuthenticationError: If error cannot be handled
        """
        auth_method = auth_config.get('method', 'none')
        error_message = str(error).lower()
        
        try:
            # Analyze error type and determine recovery strategy
            if any(keyword in error_message for keyword in ['cookie', 'session', 'expired']):
                if auth_method == 'cookies':
                    self.logger.info("Attempting to refresh expired cookies")
                    return await self.refresh_authentication(auth_config, url)
            
            elif any(keyword in error_message for keyword in ['login', 'password', 'unauthorized']):
                if auth_method == 'username_password':
                    # Check if we have alternative credentials stored
                    from urllib.parse import urlparse
                    domain = urlparse(url).netloc
                    stored_creds = await self.credential_manager.retrieve_credentials(domain)
                    if stored_creds:
                        self.logger.info("Trying stored credentials")
                        updated_config = auth_config.copy()
                        updated_config.update(stored_creds)
                        return await self.setup_authentication(updated_config, url)
            
            elif any(keyword in error_message for keyword in ['token', 'oauth']):
                if auth_method == 'oauth':
                    self.logger.warning("OAuth token error - manual refresh required")
                    raise AuthenticationError("OAuth token refresh required", auth_method=auth_method)
            
            # If no specific handling, try to refresh anyway
            self.logger.info("Attempting generic authentication refresh")
            return await self.refresh_authentication(auth_config, url)
            
        except AuthenticationError:
            raise
        except Exception as e:
            raise AuthenticationError(f"Authentication error handling failed: {e}", auth_method=auth_method)
    
    async def cleanup(self) -> None:
        """Clean up authentication manager resources."""
        self._auth_cache.clear()
        self._cache_expiry.clear()
        await self.cookie_manager.clear_session_cookies()
        self.logger.info("Authentication manager cleaned up")