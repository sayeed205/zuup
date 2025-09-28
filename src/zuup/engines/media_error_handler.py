"""Comprehensive error handling and recovery system for media downloads using yt-dlp."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from .media_models import MediaFormat, MediaInfo


class MediaErrorCategory(Enum):
    """Categories of media download errors."""
    
    EXTRACTION = "extraction"  # URL extraction, info gathering failures
    NETWORK = "network"  # Network connectivity issues
    AUTHENTICATION = "authentication"  # Login, cookies, access issues
    FORMAT = "format"  # Format unavailable, codec issues
    PROCESSING = "processing"  # Post-processing, conversion failures
    PLATFORM = "platform"  # Platform-specific restrictions
    GEO_BLOCKING = "geo_blocking"  # Geographic restrictions
    RATE_LIMITING = "rate_limiting"  # Too many requests
    TEMPORARY = "temporary"  # Temporary server issues
    PERMANENT = "permanent"  # Permanent failures


class MediaErrorAction(Enum):
    """Actions to take in response to media errors."""
    
    RETRY = "retry"  # Retry with same parameters
    RETRY_WITH_DELAY = "retry_with_delay"  # Retry after delay
    RETRY_WITH_FALLBACK = "retry_with_fallback"  # Try fallback extractor
    REDUCE_QUALITY = "reduce_quality"  # Try lower quality format
    USE_ALTERNATIVE_FORMAT = "use_alternative_format"  # Try different format
    SKIP_ITEM = "skip_item"  # Skip this item in batch
    FAIL_DOWNLOAD = "fail_download"  # Fail entire download
    REQUEST_AUTH = "request_auth"  # Request authentication
    USE_PROXY = "use_proxy"  # Suggest proxy usage


class MediaError(BaseModel):
    """Structured media download error information."""
    
    category: MediaErrorCategory
    action: MediaErrorAction
    message: str
    original_error: str
    error_code: Optional[str] = None
    extractor: Optional[str] = None
    url: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)
    retry_count: int = 0
    context: Dict[str, Any] = Field(default_factory=dict)


class ExtractionError(Exception):
    """Exception for media extraction failures."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, extractor: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.extractor = extractor


class DownloadError(Exception):
    """Exception for media download failures."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, format_id: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.format_id = format_id


class ProcessingError(Exception):
    """Exception for post-processing failures."""
    
    def __init__(self, message: str, step: Optional[str] = None, file_path: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.step = step
        self.file_path = file_path


class RetryStrategy(BaseModel):
    """Configuration for retry strategies."""
    
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter_factor: float = 0.1
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt with exponential backoff and jitter."""
        delay = self.base_delay * (self.backoff_factor ** attempt)
        delay = min(delay, self.max_delay)
        
        # Add jitter to prevent thundering herd
        jitter = delay * self.jitter_factor * random.random()
        return delay + jitter


class MediaErrorHandler:
    """
    Comprehensive error handling and recovery system for media downloads.
    
    This class provides categorized error processing, retry strategies with
    exponential backoff, fallback extractor support, and format alternative
    suggestions for yt-dlp-based media downloads.
    """
    
    def __init__(
        self,
        max_retry_attempts: int = 3,
        base_retry_delay: float = 1.0,
        max_retry_delay: float = 60.0,
        backoff_factor: float = 2.0,
        enable_fallback_extractors: bool = True,
        enable_format_alternatives: bool = True,
    ) -> None:
        """
        Initialize the media error handler.
        
        Args:
            max_retry_attempts: Maximum number of retry attempts per error
            base_retry_delay: Base delay in seconds for retry backoff
            max_retry_delay: Maximum delay in seconds for retry backoff
            backoff_factor: Exponential backoff multiplier
            enable_fallback_extractors: Whether to use fallback extractors
            enable_format_alternatives: Whether to suggest format alternatives
        """
        self.retry_strategy = RetryStrategy(
            max_attempts=max_retry_attempts,
            base_delay=base_retry_delay,
            max_delay=max_retry_delay,
            backoff_factor=backoff_factor,
        )
        self.enable_fallback_extractors = enable_fallback_extractors
        self.enable_format_alternatives = enable_format_alternatives
        
        # Error tracking for intelligent retry decisions
        self._error_history: Dict[str, List[MediaError]] = {}  # url -> errors
        self._extractor_failures: Dict[str, int] = {}  # extractor -> failure_count
        self._format_failures: Dict[str, int] = {}  # format_id -> failure_count
        
        # Fallback extractor mappings
        self._fallback_extractors: Dict[str, Optional[str]] = self._initialize_fallback_extractors()
        
        self.logger = logging.getLogger(__name__)
    
    async def handle_extraction_error(
        self, 
        error: ExtractionError, 
        url: str,
        attempt: int = 0
    ) -> MediaErrorAction:
        """
        Handle extraction errors and determine appropriate action.
        
        Args:
            error: The extraction error that occurred
            url: URL that failed extraction
            attempt: Current attempt number
            
        Returns:
            Action to take in response to the error
        """
        media_error = self._categorize_extraction_error(error, url, attempt)
        self._record_error(url, media_error)
        
        self.logger.error(
            f"Extraction error for {url}: {media_error.message}",
            extra={
                "category": media_error.category.value,
                "action": media_error.action.value,
                "attempt": attempt,
                "extractor": media_error.extractor,
            }
        )
        
        return media_error.action
    
    async def handle_download_error(
        self, 
        error: DownloadError, 
        info: MediaInfo,
        attempt: int = 0
    ) -> MediaErrorAction:
        """
        Handle download errors and determine appropriate action.
        
        Args:
            error: The download error that occurred
            info: Media information for the failed download
            attempt: Current attempt number
            
        Returns:
            Action to take in response to the error
        """
        media_error = self._categorize_download_error(error, info, attempt)
        self._record_error(info.webpage_url, media_error)
        
        self.logger.error(
            f"Download error for {info.title}: {media_error.message}",
            extra={
                "category": media_error.category.value,
                "action": media_error.action.value,
                "attempt": attempt,
                "format_id": error.format_id,
            }
        )
        
        return media_error.action
    
    async def handle_processing_error(
        self, 
        error: ProcessingError, 
        file_path: str,
        attempt: int = 0
    ) -> MediaErrorAction:
        """
        Handle post-processing errors and determine appropriate action.
        
        Args:
            error: The processing error that occurred
            file_path: Path to file that failed processing
            attempt: Current attempt number
            
        Returns:
            Action to take in response to the error
        """
        media_error = self._categorize_processing_error(error, file_path, attempt)
        self._record_error(file_path, media_error)
        
        self.logger.error(
            f"Processing error for {file_path}: {media_error.message}",
            extra={
                "category": media_error.category.value,
                "action": media_error.action.value,
                "attempt": attempt,
                "step": error.step,
            }
        )
        
        return media_error.action
    
    def should_retry_extraction(self, error: ExtractionError, attempt: int) -> bool:
        """
        Determine if extraction should be retried.
        
        Args:
            error: The extraction error that occurred
            attempt: Current attempt number
            
        Returns:
            True if extraction should be retried
        """
        if attempt >= self.retry_strategy.max_attempts:
            return False
        
        # Check for non-retryable error patterns
        non_retryable_patterns = [
            "unsupported url",
            "no video formats found",
            "private video",
            "video unavailable",
            "copyright",
            "removed",
            "deleted",
        ]
        
        error_lower = error.message.lower()
        if any(pattern in error_lower for pattern in non_retryable_patterns):
            return False
        
        return True
    
    def get_fallback_extractor(self, url: str) -> Optional[str]:
        """
        Get fallback extractor for a URL.
        
        Args:
            url: URL to get fallback extractor for
            
        Returns:
            Fallback extractor name or None if no fallback available
        """
        if not self.enable_fallback_extractors:
            return None
        
        # Extract domain from URL
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]
            
            return self._fallback_extractors.get(domain)
        except Exception:
            return None
    
    def suggest_format_alternatives(
        self, 
        failed_format: MediaFormat, 
        available_formats: List[MediaFormat]
    ) -> List[MediaFormat]:
        """
        Suggest alternative formats when a format fails.
        
        Args:
            failed_format: Format that failed to download
            available_formats: List of available formats
            
        Returns:
            List of suggested alternative formats, ordered by preference
        """
        if not self.enable_format_alternatives:
            return []
        
        alternatives: List[MediaFormat] = []
        
        # Filter out the failed format and already failed formats
        candidates = [
            fmt for fmt in available_formats 
            if fmt.format_id != failed_format.format_id
            and self._format_failures.get(fmt.format_id, 0) < 2
        ]
        
        # Sort by preference based on similarity to failed format
        candidates.sort(key=lambda fmt: self._calculate_format_similarity(failed_format, fmt), reverse=True)
        
        # Return top 3 alternatives
        return candidates[:3]
    
    async def calculate_retry_delay(self, attempt: int, error_category: MediaErrorCategory) -> float:
        """
        Calculate delay before retrying based on error category and attempt.
        
        Args:
            attempt: Current attempt number
            error_category: Category of error
            
        Returns:
            Delay in seconds before retrying
        """
        base_delay = self.retry_strategy.calculate_delay(attempt)
        
        # Adjust delay based on error category
        category_multipliers = {
            MediaErrorCategory.NETWORK: 1.0,
            MediaErrorCategory.RATE_LIMITING: 3.0,
            MediaErrorCategory.TEMPORARY: 2.0,
            MediaErrorCategory.GEO_BLOCKING: 1.5,
            MediaErrorCategory.PLATFORM: 2.0,
            MediaErrorCategory.EXTRACTION: 1.0,
            MediaErrorCategory.FORMAT: 0.5,
            MediaErrorCategory.PROCESSING: 0.5,
        }
        
        multiplier = category_multipliers.get(error_category, 1.0)
        return base_delay * multiplier
    
    def get_user_friendly_message(self, error: MediaError) -> str:
        """
        Generate user-friendly error message.
        
        Args:
            error: Media error information
            
        Returns:
            User-friendly error message
        """
        category_messages = {
            MediaErrorCategory.EXTRACTION: "Failed to extract video information",
            MediaErrorCategory.NETWORK: "Network connection failed",
            MediaErrorCategory.AUTHENTICATION: "Authentication required or failed",
            MediaErrorCategory.FORMAT: "Requested format not available",
            MediaErrorCategory.PROCESSING: "Post-processing failed",
            MediaErrorCategory.PLATFORM: "Platform-specific error occurred",
            MediaErrorCategory.GEO_BLOCKING: "Content blocked in your region",
            MediaErrorCategory.RATE_LIMITING: "Too many requests - rate limited",
            MediaErrorCategory.TEMPORARY: "Temporary server issue",
            MediaErrorCategory.PERMANENT: "Permanent failure - content unavailable",
        }
        
        base_message = category_messages.get(error.category, "Unknown error occurred")
        
        # Add specific context
        if error.url:
            base_message += f" for {error.url}"
        
        if error.retry_count > 0:
            base_message += f" (attempt {error.retry_count + 1})"
        
        # Add action suggestion
        action_suggestions = {
            MediaErrorAction.RETRY_WITH_DELAY: "Will retry after delay",
            MediaErrorAction.RETRY_WITH_FALLBACK: "Trying alternative method",
            MediaErrorAction.REDUCE_QUALITY: "Trying lower quality",
            MediaErrorAction.USE_ALTERNATIVE_FORMAT: "Trying different format",
            MediaErrorAction.REQUEST_AUTH: "Authentication may be required",
            MediaErrorAction.USE_PROXY: "Consider using a proxy",
        }
        
        suggestion = action_suggestions.get(error.action)
        if suggestion:
            base_message += f" - {suggestion}"
        
        return base_message
    
    def reset_error_tracking(self, url: str) -> None:
        """
        Reset error tracking for a URL.
        
        Args:
            url: URL to reset tracking for
        """
        self._error_history.pop(url, None)
    
    def get_error_statistics(self, url: str) -> Dict[str, Any]:
        """
        Get error statistics for a URL.
        
        Args:
            url: URL to get statistics for
            
        Returns:
            Dictionary containing error statistics
        """
        errors = self._error_history.get(url, [])
        
        if not errors:
            return {"total_errors": 0}
        
        categories: Dict[str, int] = {}
        for error in errors:
            category = error.category.value
            categories[category] = categories.get(category, 0) + 1
        
        return {
            "total_errors": len(errors),
            "error_categories": categories,
            "last_error_time": errors[-1].timestamp if errors else None,
            "most_recent_error": errors[-1].message if errors else None,
        }
    
    # Private methods
    
    def _categorize_extraction_error(
        self, 
        error: ExtractionError, 
        url: str, 
        attempt: int
    ) -> MediaError:
        """Categorize extraction error and determine action."""
        message = error.message.lower()
        
        # Network-related extraction errors
        if any(pattern in message for pattern in [
            "network", "connection", "timeout", "unreachable", "dns"
        ]):
            return MediaError(
                category=MediaErrorCategory.NETWORK,
                action=MediaErrorAction.RETRY_WITH_DELAY,
                message=error.message,
                original_error=str(error),
                error_code=error.error_code,
                extractor=error.extractor,
                url=url,
                retry_count=attempt,
            )
        
        # Authentication errors
        if any(pattern in message for pattern in [
            "login", "authentication", "credentials", "unauthorized", "forbidden"
        ]):
            return MediaError(
                category=MediaErrorCategory.AUTHENTICATION,
                action=MediaErrorAction.REQUEST_AUTH,
                message=error.message,
                original_error=str(error),
                error_code=error.error_code,
                extractor=error.extractor,
                url=url,
                retry_count=attempt,
            )
        
        # Geo-blocking errors
        if any(pattern in message for pattern in [
            "geo", "region", "country", "location", "blocked", "restricted"
        ]):
            return MediaError(
                category=MediaErrorCategory.GEO_BLOCKING,
                action=MediaErrorAction.USE_PROXY,
                message=error.message,
                original_error=str(error),
                error_code=error.error_code,
                extractor=error.extractor,
                url=url,
                retry_count=attempt,
            )
        
        # Rate limiting errors
        if any(pattern in message for pattern in [
            "rate", "limit", "too many", "quota", "throttle"
        ]):
            return MediaError(
                category=MediaErrorCategory.RATE_LIMITING,
                action=MediaErrorAction.RETRY_WITH_DELAY,
                message=error.message,
                original_error=str(error),
                error_code=error.error_code,
                extractor=error.extractor,
                url=url,
                retry_count=attempt,
            )
        
        # Permanent failures
        if any(pattern in message for pattern in [
            "unavailable", "removed", "deleted", "private", "copyright", "not found"
        ]):
            return MediaError(
                category=MediaErrorCategory.PERMANENT,
                action=MediaErrorAction.FAIL_DOWNLOAD,
                message=error.message,
                original_error=str(error),
                error_code=error.error_code,
                extractor=error.extractor,
                url=url,
                retry_count=attempt,
            )
        
        # Unsupported URL or extractor issues
        if any(pattern in message for pattern in [
            "unsupported", "no suitable", "extractor"
        ]):
            action = (MediaErrorAction.RETRY_WITH_FALLBACK 
                     if self.get_fallback_extractor(url) 
                     else MediaErrorAction.FAIL_DOWNLOAD)
            
            return MediaError(
                category=MediaErrorCategory.EXTRACTION,
                action=action,
                message=error.message,
                original_error=str(error),
                error_code=error.error_code,
                extractor=error.extractor,
                url=url,
                retry_count=attempt,
            )
        
        # Default: temporary extraction error
        return MediaError(
            category=MediaErrorCategory.TEMPORARY,
            action=MediaErrorAction.RETRY_WITH_DELAY,
            message=error.message,
            original_error=str(error),
            error_code=error.error_code,
            extractor=error.extractor,
            url=url,
            retry_count=attempt,
        )
    
    def _categorize_download_error(
        self, 
        error: DownloadError, 
        info: MediaInfo, 
        attempt: int
    ) -> MediaError:
        """Categorize download error and determine action."""
        message = error.message.lower()
        
        # Network download errors
        if any(pattern in message for pattern in [
            "network", "connection", "timeout", "reset", "broken pipe"
        ]):
            return MediaError(
                category=MediaErrorCategory.NETWORK,
                action=MediaErrorAction.RETRY_WITH_DELAY,
                message=error.message,
                original_error=str(error),
                error_code=error.error_code,
                url=info.webpage_url,
                retry_count=attempt,
                context={"format_id": error.format_id},
            )
        
        # Format-specific errors
        if any(pattern in message for pattern in [
            "format", "codec", "not available", "expired", "manifest"
        ]):
            return MediaError(
                category=MediaErrorCategory.FORMAT,
                action=MediaErrorAction.USE_ALTERNATIVE_FORMAT,
                message=error.message,
                original_error=str(error),
                error_code=error.error_code,
                url=info.webpage_url,
                retry_count=attempt,
                context={"format_id": error.format_id},
            )
        
        # HTTP errors that might indicate quality issues
        if any(pattern in message for pattern in [
            "403", "404", "410", "http"
        ]):
            return MediaError(
                category=MediaErrorCategory.FORMAT,
                action=MediaErrorAction.REDUCE_QUALITY,
                message=error.message,
                original_error=str(error),
                error_code=error.error_code,
                url=info.webpage_url,
                retry_count=attempt,
                context={"format_id": error.format_id},
            )
        
        # Default: temporary download error
        return MediaError(
            category=MediaErrorCategory.TEMPORARY,
            action=MediaErrorAction.RETRY_WITH_DELAY,
            message=error.message,
            original_error=str(error),
            error_code=error.error_code,
            url=info.webpage_url,
            retry_count=attempt,
            context={"format_id": error.format_id},
        )
    
    def _categorize_processing_error(
        self, 
        error: ProcessingError, 
        file_path: str, 
        attempt: int
    ) -> MediaError:
        """Categorize processing error and determine action."""
        message = error.message.lower()
        
        # FFmpeg/conversion errors
        if any(pattern in message for pattern in [
            "ffmpeg", "conversion", "codec", "format", "encoding"
        ]):
            return MediaError(
                category=MediaErrorCategory.PROCESSING,
                action=MediaErrorAction.SKIP_ITEM if attempt > 1 else MediaErrorAction.RETRY,
                message=error.message,
                original_error=str(error),
                url=file_path,
                retry_count=attempt,
                context={"step": error.step, "file_path": error.file_path},
            )
        
        # File system errors
        if any(pattern in message for pattern in [
            "permission", "disk", "space", "write", "read", "file"
        ]):
            return MediaError(
                category=MediaErrorCategory.PROCESSING,
                action=MediaErrorAction.FAIL_DOWNLOAD,
                message=error.message,
                original_error=str(error),
                url=file_path,
                retry_count=attempt,
                context={"step": error.step, "file_path": error.file_path},
            )
        
        # Default: processing error
        return MediaError(
            category=MediaErrorCategory.PROCESSING,
            action=MediaErrorAction.RETRY if attempt < 2 else MediaErrorAction.SKIP_ITEM,
            message=error.message,
            original_error=str(error),
            url=file_path,
            retry_count=attempt,
            context={"step": error.step, "file_path": error.file_path},
        )
    
    def _record_error(self, identifier: str, error: MediaError) -> None:
        """Record error in history for tracking."""
        if identifier not in self._error_history:
            self._error_history[identifier] = []
        
        self._error_history[identifier].append(error)
        
        # Track extractor failures
        if error.extractor:
            self._extractor_failures[error.extractor] = (
                self._extractor_failures.get(error.extractor, 0) + 1
            )
        
        # Track format failures
        format_id = error.context.get("format_id")
        if format_id:
            self._format_failures[format_id] = (
                self._format_failures.get(format_id, 0) + 1
            )
        
        # Limit history size to prevent memory issues
        if len(self._error_history[identifier]) > 10:
            self._error_history[identifier] = self._error_history[identifier][-10:]
    
    def _calculate_format_similarity(
        self, 
        failed_format: MediaFormat, 
        candidate: MediaFormat
    ) -> float:
        """Calculate similarity score between formats for alternative suggestions."""
        score = 0.0
        
        # Prefer same container format
        if failed_format.ext == candidate.ext:
            score += 0.3
        
        # Prefer similar quality
        if failed_format.quality and candidate.quality:
            quality_diff = abs(failed_format.quality - candidate.quality)
            score += max(0, 0.2 - quality_diff * 0.1)
        
        # Prefer similar resolution
        if failed_format.resolution and candidate.resolution:
            if failed_format.resolution == candidate.resolution:
                score += 0.2
        
        # Prefer same codecs
        if failed_format.vcodec == candidate.vcodec:
            score += 0.15
        if failed_format.acodec == candidate.acodec:
            score += 0.15
        
        return score
    
    def _initialize_fallback_extractors(self) -> Dict[str, Optional[str]]:
        """Initialize fallback extractor mappings."""
        return {
            # YouTube fallbacks
            "youtube.com": "generic",
            "youtu.be": "generic",
            
            # Social media fallbacks
            "twitter.com": "generic",
            "x.com": "generic",
            "instagram.com": "generic",
            "facebook.com": "generic",
            
            # Video platforms
            "vimeo.com": "generic",
            "dailymotion.com": "generic",
            
            # Generic fallback for unknown domains
            "generic": None,  # No fallback for generic
        }