"""Cookie management for media downloads."""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    Manages browser cookies for media downloads.
    
    This class provides functionality to extract cookies from browsers,
    manage cookie files, and handle authentication for media downloads.
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
    
    async def load_cookies(self) -> Dict[str, str]:
        """
        Load cookies from file.
        
        Returns:
            Dictionary of cookie name -> value pairs
        """
        if not self.cookies_file or not self.cookies_file.exists():
            return {}
        
        cookies = {}
        
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
                        domain, _, path, secure, expires, name, value = parts[:7]
                        cookies[name] = value
                        
        except Exception as e:
            self.logger.error(f"Failed to load cookies from {self.cookies_file}: {e}")
        
        return cookies
    
    async def save_cookies(self, cookies: Dict[str, str]) -> None:
        """
        Save cookies to file in Netscape format.
        
        Args:
            cookies: Dictionary of cookie name -> value pairs
        """
        if not self.cookies_file:
            return
        
        try:
            # Ensure parent directory exists
            self.cookies_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.cookies_file, 'w', encoding='utf-8') as f:
                f.write("# Netscape HTTP Cookie File\n")
                f.write("# This is a generated file! Do not edit.\n\n")
                
                for name, value in cookies.items():
                    # Write in Netscape format: domain, flag, path, secure, expires, name, value
                    f.write(f".example.com\tTRUE\t/\tFALSE\t0\t{name}\t{value}\n")
                    
        except Exception as e:
            self.logger.error(f"Failed to save cookies to {self.cookies_file}: {e}")
    
    def extract_browser_cookies(self, browser: str, domain: Optional[str] = None) -> Dict[str, str]:
        """
        Extract cookies from browser database.
        
        Args:
            browser: Browser name ('chrome', 'firefox', 'edge', 'safari')
            domain: Optional domain filter
            
        Returns:
            Dictionary of cookie name -> value pairs
        """
        browser_lower = browser.lower()
        
        if browser_lower in ['chrome', 'chromium']:
            return self._extract_chrome_cookies(domain)
        elif browser_lower == 'firefox':
            return self._extract_firefox_cookies(domain)
        elif browser_lower == 'edge':
            return self._extract_edge_cookies(domain)
        elif browser_lower == 'safari':
            return self._extract_safari_cookies(domain)
        else:
            self.logger.warning(f"Unsupported browser: {browser}")
            return {}
    
    def get_cookies_for_domain(self, domain: str) -> Dict[str, str]:
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
                file_cookies = self._load_cookies_for_domain(domain)
                all_cookies.update(file_cookies)
            except Exception as e:
                self.logger.error(f"Failed to load cookies for {domain}: {e}")
        
        # Try to extract from browsers
        browsers = ['chrome', 'firefox', 'edge']
        for browser in browsers:
            try:
                browser_cookies = self.extract_browser_cookies(browser, domain)
                all_cookies.update(browser_cookies)
            except Exception as e:
                self.logger.debug(f"Failed to extract {browser} cookies for {domain}: {e}")
        
        return all_cookies
    
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
    
    def _extract_chrome_cookies(self, domain: Optional[str] = None) -> Dict[str, str]:
        """Extract cookies from Chrome/Chromium."""
        cookies = {}
        
        for cookie_path in self._browser_paths.get('chrome', []):
            if cookie_path.exists():
                try:
                    cookies.update(self._read_chrome_cookies(cookie_path, domain))
                    break  # Use first available database
                except Exception as e:
                    self.logger.debug(f"Failed to read Chrome cookies from {cookie_path}: {e}")
        
        return cookies
    
    def _extract_firefox_cookies(self, domain: Optional[str] = None) -> Dict[str, str]:
        """Extract cookies from Firefox."""
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
                    cookies.update(self._read_firefox_cookies(cookie_file, domain))
                    break  # Use first available database
                except Exception as e:
                    self.logger.debug(f"Failed to read Firefox cookies from {cookie_file}: {e}")
        
        return cookies
    
    def _extract_edge_cookies(self, domain: Optional[str] = None) -> Dict[str, str]:
        """Extract cookies from Microsoft Edge."""
        cookies = {}
        
        for cookie_path in self._browser_paths.get('edge', []):
            if cookie_path.exists():
                try:
                    # Edge uses same format as Chrome
                    cookies.update(self._read_chrome_cookies(cookie_path, domain))
                    break
                except Exception as e:
                    self.logger.debug(f"Failed to read Edge cookies from {cookie_path}: {e}")
        
        return cookies
    
    def _extract_safari_cookies(self, domain: Optional[str] = None) -> Dict[str, str]:
        """Extract cookies from Safari (macOS only)."""
        # Safari uses a binary format that's more complex to parse
        # For now, return empty dict - would need specialized library
        self.logger.warning("Safari cookie extraction not implemented")
        return {}
    
    def _read_chrome_cookies(self, cookie_path: Path, domain: Optional[str] = None) -> Dict[str, str]:
        """Read cookies from Chrome/Chromium database."""
        cookies = {}
        
        try:
            # Chrome cookies are stored in SQLite database
            conn = sqlite3.connect(str(cookie_path))
            cursor = conn.cursor()
            
            # Query cookies
            query = "SELECT name, value, host_key FROM cookies"
            params = []
            
            if domain:
                query += " WHERE host_key LIKE ?"
                params.append(f"%{domain}%")
            
            cursor.execute(query, params)
            
            for name, value, host_key in cursor.fetchall():
                # Decrypt value if needed (Chrome encrypts cookies)
                # For now, use raw value - decryption would require platform-specific code
                cookies[name] = value
            
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to read Chrome cookies: {e}")
        
        return cookies
    
    def _read_firefox_cookies(self, cookie_path: Path, domain: Optional[str] = None) -> Dict[str, str]:
        """Read cookies from Firefox database."""
        cookies = {}
        
        try:
            conn = sqlite3.connect(str(cookie_path))
            cursor = conn.cursor()
            
            query = "SELECT name, value, host FROM moz_cookies"
            params = []
            
            if domain:
                query += " WHERE host LIKE ?"
                params.append(f"%{domain}%")
            
            cursor.execute(query, params)
            
            for name, value, host in cursor.fetchall():
                cookies[name] = value
            
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to read Firefox cookies: {e}")
        
        return cookies
    
    def _load_cookies_for_domain(self, domain: str) -> Dict[str, str]:
        """Load cookies for specific domain from file."""
        if not self.cookies_file or not self.cookies_file.exists():
            return {}
        
        cookies = {}
        
        try:
            with open(self.cookies_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split('\t')
                    if len(parts) >= 7:
                        cookie_domain, _, path, secure, expires, name, value = parts[:7]
                        
                        # Check if cookie domain matches requested domain
                        if domain in cookie_domain or cookie_domain in domain:
                            cookies[name] = value
                            
        except Exception as e:
            self.logger.error(f"Failed to load cookies for domain {domain}: {e}")
        
        return cookies


class AuthenticationManager:
    """
    Manages authentication for media downloads.
    
    This class handles various authentication methods including
    username/password, cookies, and OAuth tokens.
    """
    
    def __init__(self) -> None:
        """Initialize authentication manager."""
        self.logger = logging.getLogger(__name__)
        self.cookie_manager = CookieManager()
    
    def setup_authentication(self, auth_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Setup authentication based on configuration.
        
        Args:
            auth_config: Authentication configuration
            
        Returns:
            yt-dlp compatible authentication options
        """
        options = {}
        
        auth_method = auth_config.get('method', 'none')
        
        if auth_method == 'username_password':
            username = auth_config.get('username')
            password = auth_config.get('password')
            
            if username and password:
                options['username'] = username
                options['password'] = password
        
        elif auth_method == 'cookies':
            cookies_file = auth_config.get('cookies_file')
            if cookies_file:
                options['cookiefile'] = str(cookies_file)
        
        elif auth_method == 'netrc':
            netrc_file = auth_config.get('netrc_file')
            if netrc_file:
                options['usenetrc'] = True
                options['netrc_location'] = str(netrc_file)
            else:
                options['usenetrc'] = True
        
        elif auth_method == 'oauth':
            oauth_token = auth_config.get('oauth_token')
            if oauth_token:
                # OAuth handling is platform-specific
                # This would need to be implemented per platform
                options['oauth_token'] = oauth_token
        
        return options
    
    def extract_cookies_for_url(self, url: str, browser: Optional[str] = None) -> Dict[str, str]:
        """
        Extract cookies for a specific URL.
        
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
            
            if browser:
                return self.cookie_manager.extract_browser_cookies(browser, domain)
            else:
                return self.cookie_manager.get_cookies_for_domain(domain)
                
        except Exception as e:
            self.logger.error(f"Failed to extract cookies for {url}: {e}")
            return {}
    
    def validate_authentication(self, auth_config: Dict[str, Any]) -> bool:
        """
        Validate authentication configuration.
        
        Args:
            auth_config: Authentication configuration to validate
            
        Returns:
            True if configuration is valid
        """
        auth_method = auth_config.get('method', 'none')
        
        if auth_method == 'none':
            return True
        
        elif auth_method == 'username_password':
            return bool(auth_config.get('username') and auth_config.get('password'))
        
        elif auth_method == 'cookies':
            cookies_file = auth_config.get('cookies_file')
            return bool(cookies_file and Path(cookies_file).exists())
        
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