"""Metadata extraction and file organization manager."""

import asyncio
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field

from .media_models import MediaInfo, MediaMetadata

logger = logging.getLogger(__name__)


class MetadataTemplate(BaseModel):
    """Template configuration for metadata-based file organization."""
    
    # Filename template with variable substitution
    filename_template: str = "%(title)s.%(ext)s"
    
    # Directory structure template
    directory_template: str = "%(uploader)s"
    
    # Available variables for substitution
    available_variables: List[str] = Field(default_factory=lambda: [
        "title", "uploader", "upload_date", "id", "extractor", "duration",
        "view_count", "like_count", "description", "ext", "format_id",
        "resolution", "fps", "vcodec", "acodec", "filesize", "playlist",
        "playlist_index", "playlist_title", "channel", "channel_id",
        "series", "season", "episode", "year", "month", "day"
    ])
    
    # Character replacement rules for filesystem safety
    char_replacements: Dict[str, str] = Field(default_factory=lambda: {
        '<': '＜', '>': '＞', ':': '：', '"': '＂', '/': '／',
        '\\': '＼', '|': '｜', '?': '？', '*': '＊',
        '\n': ' ', '\r': ' ', '\t': ' '
    })
    
    # Maximum filename length
    max_filename_length: int = 200
    
    # Whether to create subdirectories
    create_subdirectories: bool = True


class ThumbnailConfig(BaseModel):
    """Configuration for thumbnail handling."""
    
    # Whether to download thumbnails
    download_thumbnails: bool = True
    
    # Whether to embed thumbnails in media files
    embed_thumbnails: bool = True
    
    # Whether to save thumbnails as separate files
    save_separate_thumbnails: bool = False
    
    # Thumbnail format preference
    preferred_formats: List[str] = Field(default_factory=lambda: ["jpg", "png", "webp"])
    
    # Maximum thumbnail size (width x height)
    max_size: Optional[tuple[int, int]] = None
    
    # Thumbnail quality (1-100 for JPEG)
    quality: int = 85


class MetadataExtractor:
    """Enhanced metadata extraction from yt-dlp information."""
    
    def __init__(self) -> None:
        """Initialize metadata extractor."""
        self.logger = logging.getLogger(f"{__name__}.MetadataExtractor")
    
    def extract_comprehensive_metadata(self, yt_dlp_info: Dict[str, Any]) -> MediaMetadata:
        """
        Extract comprehensive metadata from yt-dlp info dictionary.
        
        Args:
            yt_dlp_info: Raw yt-dlp information dictionary
            
        Returns:
            MediaMetadata with comprehensive information
        """
        self.logger.debug("Extracting comprehensive metadata")
        
        # Parse upload date
        upload_date = None
        if yt_dlp_info.get("upload_date"):
            try:
                upload_date = datetime.strptime(yt_dlp_info["upload_date"], "%Y%m%d")
            except (ValueError, TypeError):
                self.logger.warning(f"Could not parse upload_date: {yt_dlp_info.get('upload_date')}")
        
        # Extract tags from various sources
        tags = []
        if yt_dlp_info.get("tags"):
            tags.extend(yt_dlp_info["tags"])
        if yt_dlp_info.get("categories"):
            tags.extend(yt_dlp_info["categories"])
        
        # Create comprehensive metadata
        metadata = MediaMetadata(
            title=yt_dlp_info.get("title"),
            artist=yt_dlp_info.get("uploader") or yt_dlp_info.get("channel"),
            album=yt_dlp_info.get("playlist_title") or yt_dlp_info.get("series"),
            description=yt_dlp_info.get("description"),
            upload_date=upload_date,
            duration=yt_dlp_info.get("duration"),
            thumbnail_url=self._get_best_thumbnail_url(yt_dlp_info),
            uploader=yt_dlp_info.get("uploader") or yt_dlp_info.get("channel"),
            view_count=yt_dlp_info.get("view_count"),
            like_count=yt_dlp_info.get("like_count"),
            tags=tags
        )
        
        self.logger.debug(f"Extracted metadata for: {metadata.title}")
        return metadata
    
    def _get_best_thumbnail_url(self, yt_dlp_info: Dict[str, Any]) -> Optional[str]:
        """
        Get the best available thumbnail URL.
        
        Args:
            yt_dlp_info: yt-dlp information dictionary
            
        Returns:
            Best thumbnail URL or None
        """
        # Try direct thumbnail URL first
        if yt_dlp_info.get("thumbnail"):
            return yt_dlp_info["thumbnail"]
        
        # Try thumbnails list
        thumbnails = yt_dlp_info.get("thumbnails", [])
        if not thumbnails:
            return None
        
        # Sort by preference (highest resolution first)
        sorted_thumbnails = sorted(
            thumbnails,
            key=lambda t: (
                t.get("preference", 0),
                t.get("width", 0) * t.get("height", 0)
            ),
            reverse=True
        )
        
        return sorted_thumbnails[0].get("url") if sorted_thumbnails else None
    
    def create_template_variables(self, info: MediaInfo, metadata: MediaMetadata) -> Dict[str, str]:
        """
        Create template variables for filename and directory substitution.
        
        Args:
            info: Media information
            metadata: Extracted metadata
            
        Returns:
            Dictionary of template variables
        """
        variables = {}
        
        # Basic information
        variables["id"] = info.id
        variables["title"] = self._sanitize_for_template(metadata.title or "Unknown")
        variables["uploader"] = self._sanitize_for_template(metadata.uploader or "Unknown")
        variables["extractor"] = info.extractor_key
        variables["webpage_url"] = info.webpage_url
        
        # Date information
        if metadata.upload_date:
            variables["upload_date"] = metadata.upload_date.strftime("%Y-%m-%d")
            variables["year"] = str(metadata.upload_date.year)
            variables["month"] = f"{metadata.upload_date.month:02d}"
            variables["day"] = f"{metadata.upload_date.day:02d}"
        else:
            variables["upload_date"] = "Unknown"
            variables["year"] = "Unknown"
            variables["month"] = "Unknown"
            variables["day"] = "Unknown"
        
        # Media properties
        variables["duration"] = str(int(metadata.duration)) if metadata.duration else "Unknown"
        variables["view_count"] = str(metadata.view_count) if metadata.view_count else "0"
        variables["like_count"] = str(metadata.like_count) if metadata.like_count else "0"
        
        # Format information (from first available format)
        if info.formats:
            first_format = info.formats[0]
            variables["ext"] = first_format.ext
            variables["format_id"] = first_format.format_id
            variables["resolution"] = first_format.resolution or "Unknown"
            variables["fps"] = str(first_format.fps) if first_format.fps else "Unknown"
            variables["vcodec"] = first_format.vcodec or "Unknown"
            variables["acodec"] = first_format.acodec or "Unknown"
            variables["filesize"] = str(first_format.filesize) if first_format.filesize else "Unknown"
        else:
            variables.update({
                "ext": "unknown",
                "format_id": "unknown",
                "resolution": "Unknown",
                "fps": "Unknown",
                "vcodec": "Unknown",
                "acodec": "Unknown",
                "filesize": "Unknown"
            })
        
        # Playlist information (if available)
        variables["playlist"] = "Unknown"
        variables["playlist_index"] = "Unknown"
        variables["playlist_title"] = "Unknown"
        
        # Channel information
        variables["channel"] = variables["uploader"]
        variables["channel_id"] = "Unknown"
        
        # Series/episode information
        variables["series"] = metadata.album or "Unknown"
        variables["season"] = "Unknown"
        variables["episode"] = "Unknown"
        
        # Description (truncated for filename safety)
        description = metadata.description or ""
        variables["description"] = self._sanitize_for_template(description[:50])
        
        return variables
    
    def _sanitize_for_template(self, text: Optional[str]) -> str:
        """
        Sanitize text for use in templates.
        
        Args:
            text: Text to sanitize
            
        Returns:
            Sanitized text safe for filenames
        """
        if not text:
            return "Unknown"
        
        # Remove control characters and normalize whitespace
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text


class FilenameGenerator:
    """Generate filenames and directory paths from templates."""
    
    def __init__(self, template_config: MetadataTemplate) -> None:
        """
        Initialize filename generator.
        
        Args:
            template_config: Template configuration
        """
        self.config = template_config
        self.logger = logging.getLogger(f"{__name__}.FilenameGenerator")
    
    def generate_filename(self, variables: Dict[str, str], original_ext: str) -> str:
        """
        Generate filename from template and variables.
        
        Args:
            variables: Template variables
            original_ext: Original file extension
            
        Returns:
            Generated filename
        """
        # Ensure ext variable is set
        if "ext" not in variables:
            variables["ext"] = original_ext.lstrip(".")
        
        # Apply template substitution
        filename = self._substitute_template(self.config.filename_template, variables)
        
        # Sanitize for filesystem
        filename = self._sanitize_filename(filename)
        
        # Ensure proper extension
        if not filename.endswith(f".{variables['ext']}"):
            filename = f"{filename}.{variables['ext']}"
        
        # Limit length
        if len(filename) > self.config.max_filename_length:
            # Truncate while preserving extension
            ext = f".{variables['ext']}"
            max_stem_length = self.config.max_filename_length - len(ext)
            filename = filename[:max_stem_length] + ext
        
        self.logger.debug(f"Generated filename: {filename}")
        return filename
    
    def generate_directory_path(self, variables: Dict[str, str], base_path: Path) -> Path:
        """
        Generate directory path from template and variables.
        
        Args:
            variables: Template variables
            base_path: Base output directory
            
        Returns:
            Generated directory path
        """
        if not self.config.create_subdirectories:
            return base_path
        
        # Apply template substitution
        dir_structure = self._substitute_template(self.config.directory_template, variables)
        
        # Split into path components and sanitize each
        components = []
        for component in dir_structure.split("/"):
            if component.strip():
                sanitized = self._sanitize_filename(component.strip())
                if sanitized:
                    components.append(sanitized)
        
        # Build final path
        if components:
            result_path = base_path
            for component in components:
                result_path = result_path / component
            self.logger.debug(f"Generated directory path: {result_path}")
            return result_path
        else:
            return base_path
    
    def _substitute_template(self, template: str, variables: Dict[str, str]) -> str:
        """
        Substitute template variables in string.
        
        Args:
            template: Template string with %(variable)s placeholders
            variables: Variable values
            
        Returns:
            String with variables substituted
        """
        try:
            # Use Python string formatting
            return template % variables
        except (KeyError, ValueError) as e:
            self.logger.warning(f"Template substitution failed: {e}")
            # Fallback to basic substitution
            result = template
            for key, value in variables.items():
                result = result.replace(f"%({key})s", value)
            return result
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename for filesystem compatibility.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Apply character replacements
        for old_char, new_char in self.config.char_replacements.items():
            filename = filename.replace(old_char, new_char)
        
        # Remove leading/trailing dots and spaces
        filename = filename.strip(". ")
        
        # Ensure not empty
        if not filename:
            filename = "untitled"
        
        return filename


class ThumbnailManager:
    """Manage thumbnail downloading and processing."""
    
    def __init__(self, config: ThumbnailConfig) -> None:
        """
        Initialize thumbnail manager.
        
        Args:
            config: Thumbnail configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ThumbnailManager")
    
    async def download_thumbnail(self, url: str, output_path: Path) -> Optional[Path]:
        """
        Download thumbnail from URL.
        
        Args:
            url: Thumbnail URL
            output_path: Path to save thumbnail
            
        Returns:
            Path to downloaded thumbnail or None if failed
        """
        if not self.config.download_thumbnails or not url:
            return None
        
        self.logger.info(f"Downloading thumbnail from: {url}")
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                
                # Ensure output directory exists
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write thumbnail data
                with open(output_path, "wb") as f:
                    f.write(response.content)
                
                self.logger.debug(f"Thumbnail downloaded to: {output_path}")
                return output_path
                
        except Exception as e:
            self.logger.error(f"Failed to download thumbnail: {e}")
            return None
    
    def get_thumbnail_path(self, media_path: Path, thumbnail_url: str) -> Path:
        """
        Generate thumbnail file path based on media file path.
        
        Args:
            media_path: Path to media file
            thumbnail_url: Thumbnail URL for extension detection
            
        Returns:
            Path for thumbnail file
        """
        # Determine extension from URL or use default
        parsed_url = urlparse(thumbnail_url)
        url_path = parsed_url.path.lower()
        
        if url_path.endswith('.png'):
            ext = '.png'
        elif url_path.endswith('.webp'):
            ext = '.webp'
        else:
            ext = '.jpg'  # Default
        
        # Create thumbnail path alongside media file
        return media_path.with_suffix(f".thumb{ext}")


class MetadataManager:
    """Comprehensive metadata extraction and file organization manager."""
    
    def __init__(
        self,
        template_config: Optional[MetadataTemplate] = None,
        thumbnail_config: Optional[ThumbnailConfig] = None
    ) -> None:
        """
        Initialize metadata manager.
        
        Args:
            template_config: Template configuration for file organization
            thumbnail_config: Thumbnail handling configuration
        """
        self.template_config = template_config or MetadataTemplate()
        self.thumbnail_config = thumbnail_config or ThumbnailConfig()
        
        self.extractor = MetadataExtractor()
        self.filename_generator = FilenameGenerator(self.template_config)
        self.thumbnail_manager = ThumbnailManager(self.thumbnail_config)
        
        self.logger = logging.getLogger(f"{__name__}.MetadataManager")
        self.logger.info("MetadataManager initialized")
    
    async def process_media_metadata(
        self,
        info: MediaInfo,
        yt_dlp_info: Dict[str, Any],
        base_output_path: Path,
        original_filename: str
    ) -> tuple[Path, MediaMetadata, Optional[Path]]:
        """
        Process media metadata and organize files.
        
        Args:
            info: Media information from format extractor
            yt_dlp_info: Raw yt-dlp information dictionary
            base_output_path: Base output directory
            original_filename: Original filename
            
        Returns:
            Tuple of (organized_file_path, metadata, thumbnail_path)
        """
        self.logger.info(f"Processing metadata for: {info.title}")
        
        # Extract comprehensive metadata
        metadata = self.extractor.extract_comprehensive_metadata(yt_dlp_info)
        
        # Create template variables
        variables = self.extractor.create_template_variables(info, metadata)
        
        # Generate organized paths
        directory_path = self.filename_generator.generate_directory_path(
            variables, base_output_path
        )
        
        # Get original extension
        original_ext = Path(original_filename).suffix
        filename = self.filename_generator.generate_filename(variables, original_ext)
        
        # Create full organized path
        organized_path = directory_path / filename
        
        # Ensure directory exists
        directory_path.mkdir(parents=True, exist_ok=True)
        
        # Handle thumbnail if available
        thumbnail_path = None
        if metadata.thumbnail_url and self.thumbnail_config.save_separate_thumbnails:
            thumbnail_path = self.thumbnail_manager.get_thumbnail_path(
                organized_path, metadata.thumbnail_url
            )
            thumbnail_path = await self.thumbnail_manager.download_thumbnail(
                metadata.thumbnail_url, thumbnail_path
            )
        
        self.logger.info(f"Organized path: {organized_path}")
        return organized_path, metadata, thumbnail_path
    
    def create_metadata_from_info(self, info: MediaInfo) -> MediaMetadata:
        """
        Create basic metadata from MediaInfo (fallback method).
        
        Args:
            info: Media information
            
        Returns:
            Basic metadata
        """
        return MediaMetadata(
            title=info.title,
            artist=info.uploader,
            description=info.description,
            uploader=info.uploader,
            view_count=info.view_count,
            like_count=info.like_count,
            duration=info.duration,
            thumbnail_url=info.thumbnail,
        )