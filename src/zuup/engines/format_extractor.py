"""Format extraction and selection for media downloads using yt-dlp."""

import asyncio
import logging
from typing import Any

import yt_dlp  # type: ignore[import-untyped]

from .format_selector import FormatSelector, QualityTier
from .media_models import (
    ChapterInfo,
    FormatPreferences,
    MediaConfig,
    MediaFormat,
    MediaInfo,
    SubtitleInfo,
)
from .network_manager import NetworkManager

logger = logging.getLogger(__name__)


class FormatExtractor:
    """Extracts available formats and metadata from URLs using yt-dlp."""

    def __init__(self, config: MediaConfig) -> None:
        """
        Initialize FormatExtractor with configuration.

        Args:
            config: Media configuration for extraction
        """
        self.config = config
        self.format_selector = FormatSelector()

        # Initialize network manager for advanced proxy and geo-bypass support
        self.network_manager = NetworkManager(
            network_config=config.network_config,
            proxy_config=config.proxy_config,
            geo_bypass_config=config.geo_bypass_config,
        )

        logger.info(
            "FormatExtractor initialized with advanced format selection and network management"
        )

    def _create_yt_dlp_options(self, url: str) -> dict[str, Any]:
        """
        Create yt-dlp options from configuration with advanced network support.

        Args:
            url: URL being processed (for platform-specific settings)

        Returns:
            Dictionary of yt-dlp options
        """
        opts = {
            # Don't download, just extract info
            "quiet": True,
            "no_warnings": False,
            "extract_flat": False,
            "writethumbnail": False,
            "writesubtitles": False,
            "writeinfojson": False,
            "writedescription": False,
            # Extractor arguments
            "extractor_args": self.config.extractor_args,
        }

        # Add advanced network configuration
        network_opts = self.network_manager.create_yt_dlp_options(url)
        opts.update(network_opts)

        # Add platform-specific optimizations
        platform_opts = self.network_manager.get_platform_specific_options(url)
        opts.update(platform_opts)

        # Add authentication if configured
        if self.config.auth_config.username:
            opts["username"] = self.config.auth_config.username
        if self.config.auth_config.password:
            opts["password"] = self.config.auth_config.password
        if self.config.auth_config.cookies_file:
            opts["cookiefile"] = str(self.config.auth_config.cookies_file)
        if self.config.auth_config.netrc_file:
            opts["usenetrc"] = True
            opts["netrc_location"] = str(self.config.auth_config.netrc_file)

        return opts

    async def extract_info(self, url: str) -> MediaInfo:
        """
        Extract comprehensive media information from URL.

        Args:
            url: URL to extract information from

        Returns:
            MediaInfo object with extracted information

        Raises:
            ValueError: If URL is invalid or extraction fails
            RuntimeError: If yt-dlp extraction fails
        """
        if not url or not url.strip():
            raise ValueError("URL cannot be empty")

        logger.info(f"Extracting info for URL: {url}")

        try:
            # Run yt-dlp extraction in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            info_dict = await loop.run_in_executor(None, self._extract_info_sync, url)

            # Convert yt-dlp info dict to our MediaInfo model
            media_info = self._parse_info_dict(info_dict)

            logger.info(f"Successfully extracted info for: {media_info.title}")
            return media_info

        except yt_dlp.DownloadError as e:
            logger.error(f"yt-dlp extraction failed for {url}: {e}")
            raise RuntimeError(f"Failed to extract media info: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error extracting info for {url}: {e}")
            raise RuntimeError(f"Unexpected extraction error: {e}") from e

    async def get_raw_info(self, url: str) -> dict[str, Any]:
        """
        Get raw yt-dlp information dictionary for comprehensive metadata processing.

        Args:
            url: URL to extract information from

        Returns:
            Raw yt-dlp info dictionary

        Raises:
            ValueError: If URL is invalid
            RuntimeError: If yt-dlp extraction fails
        """
        if not url or not url.strip():
            raise ValueError("URL cannot be empty")

        logger.debug(f"Extracting raw yt-dlp info for URL: {url}")

        try:
            # Run yt-dlp extraction in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            info_dict = await loop.run_in_executor(None, self._extract_info_sync, url)

            logger.debug(
                f"Successfully extracted raw info for: {info_dict.get('title', 'Unknown')}"
            )
            return info_dict

        except yt_dlp.DownloadError as e:
            logger.error(f"yt-dlp raw info extraction failed for {url}: {e}")
            raise RuntimeError(f"Failed to extract raw media info: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error extracting raw info for {url}: {e}")
            raise RuntimeError(f"Unexpected raw extraction error: {e}") from e

    def _extract_info_sync(self, url: str) -> dict[str, Any]:
        """
        Synchronous yt-dlp info extraction with advanced network support.

        Args:
            url: URL to extract information from

        Returns:
            Raw yt-dlp info dictionary
        """
        # Create URL-specific options
        yt_dlp_opts = self._create_yt_dlp_options(url)

        with yt_dlp.YoutubeDL(yt_dlp_opts) as ydl:
            result: dict[str, Any] = ydl.extract_info(url, download=False)
            return result

    def _parse_info_dict(self, info_dict: dict[str, Any]) -> MediaInfo:
        """
        Parse yt-dlp info dictionary into MediaInfo model.

        Args:
            info_dict: Raw yt-dlp info dictionary

        Returns:
            Parsed MediaInfo object
        """
        # Extract basic information
        media_id = info_dict.get("id", "")
        title = info_dict.get("title", "Unknown Title")
        description = info_dict.get("description")
        uploader = info_dict.get("uploader")
        upload_date = info_dict.get("upload_date")
        duration = info_dict.get("duration")
        view_count = info_dict.get("view_count")
        like_count = info_dict.get("like_count")
        thumbnail = info_dict.get("thumbnail")
        webpage_url = info_dict.get("webpage_url", "")
        extractor = info_dict.get("extractor", "")
        extractor_key = info_dict.get("extractor_key", "")

        # Parse formats
        formats = self._parse_formats(info_dict.get("formats", []))

        # Parse subtitles
        subtitles = self._parse_subtitles(info_dict.get("subtitles", {}))

        # Parse chapters
        chapters = self._parse_chapters(info_dict.get("chapters", []))

        # Determine if this is a playlist
        is_playlist = "_type" in info_dict and info_dict["_type"] == "playlist"

        return MediaInfo(
            id=media_id,
            title=title,
            description=description,
            uploader=uploader,
            upload_date=upload_date,
            duration=duration,
            view_count=view_count,
            like_count=like_count,
            thumbnail=thumbnail,
            formats=formats,
            subtitles=subtitles,
            chapters=chapters,
            webpage_url=webpage_url,
            extractor=extractor,
            extractor_key=extractor_key,
            is_playlist=is_playlist,
        )

    def _parse_formats(self, formats_list: list[dict[str, Any]]) -> list[MediaFormat]:
        """
        Parse yt-dlp formats list into MediaFormat objects.

        Args:
            formats_list: List of format dictionaries from yt-dlp

        Returns:
            List of MediaFormat objects
        """
        formats = []

        for fmt in formats_list:
            try:
                media_format = MediaFormat(
                    format_id=fmt.get("format_id", ""),
                    ext=fmt.get("ext", ""),
                    resolution=fmt.get("resolution"),
                    fps=fmt.get("fps"),
                    vcodec=fmt.get("vcodec"),
                    acodec=fmt.get("acodec"),
                    abr=fmt.get("abr"),
                    vbr=fmt.get("vbr"),
                    filesize=fmt.get("filesize"),
                    filesize_approx=fmt.get("filesize_approx"),
                    quality=fmt.get("quality"),
                    format_note=fmt.get("format_note"),
                    language=fmt.get("language"),
                    preference=fmt.get("preference"),
                )
                formats.append(media_format)
            except Exception as e:
                logger.warning(
                    f"Failed to parse format {fmt.get('format_id', 'unknown')}: {e}"
                )
                continue

        return formats

    def _parse_subtitles(
        self, subtitles_dict: dict[str, list[dict[str, Any]]]
    ) -> dict[str, list[SubtitleInfo]]:
        """
        Parse yt-dlp subtitles dictionary into SubtitleInfo objects.

        Args:
            subtitles_dict: Subtitles dictionary from yt-dlp

        Returns:
            Dictionary mapping language codes to lists of SubtitleInfo objects
        """
        parsed_subtitles = {}

        for lang, sub_list in subtitles_dict.items():
            parsed_list = []
            for sub in sub_list:
                try:
                    subtitle_info = SubtitleInfo(
                        url=sub.get("url", ""),
                        ext=sub.get("ext", ""),
                        language=lang,
                        name=sub.get("name"),
                    )
                    parsed_list.append(subtitle_info)
                except Exception as e:
                    logger.warning(f"Failed to parse subtitle for language {lang}: {e}")
                    continue

            if parsed_list:
                parsed_subtitles[lang] = parsed_list

        return parsed_subtitles

    def _parse_chapters(self, chapters_list: list[dict[str, Any]]) -> list[ChapterInfo]:
        """
        Parse yt-dlp chapters list into ChapterInfo objects.

        Args:
            chapters_list: List of chapter dictionaries from yt-dlp

        Returns:
            List of ChapterInfo objects
        """
        chapters = []

        for chapter in chapters_list:
            try:
                chapter_info = ChapterInfo(
                    title=chapter.get("title", ""),
                    start_time=chapter.get("start_time", 0.0),
                    end_time=chapter.get("end_time", 0.0),
                )
                chapters.append(chapter_info)
            except Exception as e:
                logger.warning(
                    f"Failed to parse chapter {chapter.get('title', 'unknown')}: {e}"
                )
                continue

        return chapters

    async def get_formats(self, url: str) -> list[MediaFormat]:
        """
        Get available formats for a URL.

        Args:
            url: URL to get formats for

        Returns:
            List of available MediaFormat objects
        """
        media_info = await self.extract_info(url)
        return media_info.formats

    def select_format(
        self, formats: list[MediaFormat], preferences: FormatPreferences | None = None
    ) -> MediaFormat:
        """
        Select the best format using advanced quality control.

        Args:
            formats: List of available formats
            preferences: Format selection preferences (uses config if None)

        Returns:
            Selected MediaFormat

        Raises:
            ValueError: If no formats available or no suitable format found
        """
        if not formats:
            raise ValueError("No formats available for selection")

        # Use provided preferences or fall back to config preferences
        prefs = preferences or self.config.format_preferences

        logger.info(
            f"Selecting format from {len(formats)} available formats using advanced selection"
        )

        # Use advanced format selector
        try:
            # Convert target quality string to enum if specified
            target_quality = None
            if prefs.target_quality:
                target_quality = QualityTier(prefs.target_quality)

            selected_format = self.format_selector.select_optimal_format(
                formats=formats,
                preferences=prefs,
                target_quality=target_quality,
                adaptive=prefs.adaptive_quality,
            )

            logger.info(
                f"Selected format: {selected_format.format_id} ({selected_format.ext})"
            )
            return selected_format

        except Exception as e:
            logger.warning(
                f"Advanced format selection failed: {e}, falling back to basic selection"
            )
            return self._fallback_format_selection(formats, prefs)

    def _filter_formats(
        self, formats: list[MediaFormat], preferences: FormatPreferences
    ) -> list[MediaFormat]:
        """
        Filter formats based on preferences.

        Args:
            formats: List of formats to filter
            preferences: Filtering preferences

        Returns:
            Filtered list of formats
        """
        filtered = formats.copy()

        # Filter by audio/video only preferences
        if preferences.audio_only:
            filtered = [f for f in filtered if f.vcodec in (None, "none")]
        elif preferences.video_only:
            filtered = [f for f in filtered if f.acodec in (None, "none")]

        # Filter by maximum dimensions
        if preferences.max_height:
            filtered = [
                f
                for f in filtered
                if not f.resolution
                or (
                    (height := self._get_height_from_resolution(f.resolution))
                    is not None
                    and height <= preferences.max_height
                )
            ]

        if preferences.max_width:
            filtered = [
                f
                for f in filtered
                if not f.resolution
                or (
                    (width := self._get_width_from_resolution(f.resolution)) is not None
                    and width <= preferences.max_width
                )
            ]

        # Filter by preferred codecs
        if preferences.preferred_codecs:
            codec_filtered = []
            for fmt in filtered:
                if (
                    fmt.vcodec
                    and any(
                        codec in fmt.vcodec for codec in preferences.preferred_codecs
                    )
                ) or (
                    fmt.acodec
                    and any(
                        codec in fmt.acodec for codec in preferences.preferred_codecs
                    )
                ):
                    codec_filtered.append(fmt)
            if codec_filtered:
                filtered = codec_filtered

        # Filter by preferred containers
        if preferences.preferred_containers:
            container_filtered = [
                f for f in filtered if f.ext in preferences.preferred_containers
            ]
            if container_filtered:
                filtered = container_filtered

        # Filter by free formats preference
        if preferences.prefer_free_formats:
            free_formats = [
                f
                for f in filtered
                if f.ext in ("webm", "ogg", "opus", "flac")
                or (
                    f.vcodec
                    and any(codec in f.vcodec for codec in ("vp8", "vp9", "av01"))
                )
                or (
                    f.acodec
                    and any(codec in f.acodec for codec in ("vorbis", "opus", "flac"))
                )
            ]
            if free_formats:
                filtered = free_formats

        return filtered

    def _select_best_format(
        self, formats: list[MediaFormat], preferences: FormatPreferences
    ) -> MediaFormat:
        """
        Select the best format from a filtered list.

        Args:
            formats: Filtered list of formats
            preferences: Selection preferences

        Returns:
            Best format based on quality and preferences
        """

        # Sort formats by quality metrics
        def format_score(fmt: MediaFormat) -> tuple[int, float, int, float, float]:
            """Calculate format score for sorting."""
            # Primary: explicit preference value (higher is better)
            preference_score = fmt.preference or 0

            # Secondary: quality value (higher is better)
            quality_score = fmt.quality or 0

            # Tertiary: resolution (higher is better)
            resolution_score = 0
            if fmt.resolution:
                height = self._get_height_from_resolution(fmt.resolution)
                if height:
                    resolution_score = height

            # Quaternary: bitrate (higher is better)
            bitrate_score = 0.0
            if fmt.vbr:
                bitrate_score += fmt.vbr
            if fmt.abr:
                bitrate_score += fmt.abr

            # Quinary: filesize (smaller is better for tie-breaking)
            filesize_score = 0.0
            if fmt.filesize:
                filesize_score = -float(fmt.filesize)  # Negative for reverse sorting
            elif fmt.filesize_approx:
                filesize_score = -float(fmt.filesize_approx)

            return (
                preference_score,
                quality_score,
                resolution_score,
                bitrate_score,
                filesize_score,
            )

        # Sort by score (descending)
        sorted_formats = sorted(formats, key=format_score, reverse=True)

        return sorted_formats[0]

    def select_audio_only_format(
        self,
        formats: list[MediaFormat],
        preferences: FormatPreferences | None = None,
        target_bitrate: int | None = None,
        allow_conversion: bool = True,
    ) -> MediaFormat:
        """
        Select optimal audio-only format with advanced quality control and conversion support.

        Args:
            formats: List of available formats
            preferences: Format selection preferences (uses config if None)
            target_bitrate: Target audio bitrate in kbps
            allow_conversion: Allow selection of video formats for audio extraction

        Returns:
            Selected audio MediaFormat

        Raises:
            ValueError: If no suitable audio format found
        """
        if not formats:
            raise ValueError("No formats available for audio selection")

        prefs = preferences or self.config.format_preferences

        # Use target bitrate from preferences if not specified
        if target_bitrate is None:
            target_bitrate = prefs.target_audio_bitrate

        logger.info(
            f"Selecting audio-only format from {len(formats)} available formats"
        )

        try:
            selected_format = self.format_selector.select_audio_only_format(
                formats=formats,
                preferences=prefs,
                target_bitrate=target_bitrate,
                allow_conversion=allow_conversion,
            )

            logger.info(
                f"Selected audio format: {selected_format.format_id} "
                f"({selected_format.ext}, {selected_format.abr or 'unknown'} kbps)"
            )
            return selected_format

        except Exception as e:
            logger.warning(
                f"Advanced audio selection failed: {e}, falling back to basic selection"
            )
            return self._fallback_audio_selection(formats, prefs)

    def get_quality_alternatives(
        self,
        formats: list[MediaFormat],
        current_format: MediaFormat,
        direction: str = "lower",
    ) -> list[MediaFormat]:
        """
        Get quality alternatives for adaptive selection.

        Args:
            formats: List of available formats
            current_format: Current format to find alternatives for
            direction: "lower" or "higher" quality

        Returns:
            List of alternative formats ordered by preference
        """
        logger.info(
            f"Finding {direction} quality alternatives for {current_format.format_id}"
        )

        try:
            alternatives = self.format_selector.get_quality_alternatives(
                formats=formats, current_format=current_format, direction=direction
            )

            logger.info(f"Found {len(alternatives)} {direction} quality alternatives")
            return alternatives

        except Exception as e:
            logger.error(f"Failed to get quality alternatives: {e}")
            return []

    def analyze_available_formats(self, formats: list[MediaFormat]) -> dict[str, Any]:
        """
        Analyze format distribution for debugging and optimization.

        Args:
            formats: List of formats to analyze

        Returns:
            Analysis dictionary with format distribution information
        """
        return self.format_selector.analyze_format_distribution(formats)

    def select_format_with_fallback(
        self,
        formats: list[MediaFormat],
        preferences: FormatPreferences | None = None,
        max_attempts: int = 3,
    ) -> MediaFormat:
        """
        Select format with automatic quality fallback on failure.

        Args:
            formats: List of available formats
            preferences: Format selection preferences
            max_attempts: Maximum fallback attempts

        Returns:
            Selected format with fallback applied if needed

        Raises:
            ValueError: If no suitable format found after all attempts
        """
        prefs = preferences or self.config.format_preferences

        if not prefs.quality_fallback:
            return self.select_format(formats, preferences)

        logger.info("Selecting format with quality fallback enabled")

        # Try original selection first
        try:
            return self.select_format(formats, preferences)
        except Exception as e:
            logger.warning(f"Primary format selection failed: {e}")

        # Try with progressively lower quality targets
        quality_fallback_order = [
            QualityTier.HIGH,
            QualityTier.MEDIUM,
            QualityTier.LOW,
            QualityTier.VERY_LOW,
        ]

        for attempt, fallback_quality in enumerate(quality_fallback_order):
            if attempt >= max_attempts - 1:
                break

            try:
                logger.info(
                    f"Attempting fallback with {fallback_quality.value} quality"
                )

                # Create modified preferences with lower quality target
                fallback_prefs = prefs.model_copy()
                fallback_prefs.target_quality = fallback_quality.value

                return self.format_selector.select_optimal_format(
                    formats=formats,
                    preferences=fallback_prefs,
                    target_quality=fallback_quality,
                    adaptive=True,
                )

            except Exception as e:
                logger.warning(f"Fallback attempt {attempt + 1} failed: {e}")
                continue

        # Final fallback to basic selection
        logger.warning("All quality fallbacks failed, using basic selection")
        return self._fallback_format_selection(formats, prefs)

    def _fallback_format_selection(
        self, formats: list[MediaFormat], preferences: FormatPreferences
    ) -> MediaFormat:
        """Fallback to basic format selection when advanced selection fails."""
        logger.info("Using fallback format selection")

        # Filter formats based on basic preferences
        filtered_formats = self._filter_formats(formats, preferences)

        if not filtered_formats:
            logger.warning("No formats match preferences, using all formats")
            filtered_formats = formats

        # Select best format from filtered list using basic scoring
        selected_format = self._select_best_format(filtered_formats, preferences)

        return selected_format

    def _fallback_audio_selection(
        self, formats: list[MediaFormat], preferences: FormatPreferences
    ) -> MediaFormat:
        """Fallback to basic audio format selection."""
        logger.info("Using fallback audio format selection")

        # Filter to audio formats
        audio_formats = [
            fmt for fmt in formats if fmt.acodec is not None and fmt.acodec != "none"
        ]

        if not audio_formats:
            raise ValueError("No audio formats available")

        # Prefer audio-only formats
        audio_only = [fmt for fmt in audio_formats if fmt.vcodec in (None, "none")]
        if audio_only:
            audio_formats = audio_only

        # Sort by audio bitrate (higher is better)
        audio_formats.sort(key=lambda x: x.abr or 0, reverse=True)

        return audio_formats[0]

    def _get_height_from_resolution(self, resolution: str) -> int | None:
        """
        Extract height from resolution string.

        Args:
            resolution: Resolution string (e.g., "1920x1080", "1080p")

        Returns:
            Height in pixels, or None if cannot parse
        """
        try:
            if "x" in resolution:
                # Format: "1920x1080"
                return int(resolution.split("x")[1])
            elif "p" in resolution:
                # Format: "1080p"
                return int(resolution.replace("p", ""))
            else:
                # Try to parse as integer
                return int(resolution)
        except (ValueError, IndexError):
            return None

    def _get_width_from_resolution(self, resolution: str) -> int | None:
        """
        Extract width from resolution string.

        Args:
            resolution: Resolution string (e.g., "1920x1080")

        Returns:
            Width in pixels, or None if cannot parse
        """
        try:
            if "x" in resolution:
                # Format: "1920x1080"
                return int(resolution.split("x")[0])
            else:
                # For formats like "1080p", estimate width based on 16:9 aspect ratio
                height = self._get_height_from_resolution(resolution)
                if height:
                    return int(height * 16 / 9)
                return None
        except (ValueError, IndexError):
            return None

    def check_format_compatibility(self, format_obj: MediaFormat) -> bool:
        """
        Check if a format is compatible with current system/configuration.

        Args:
            format_obj: Format to check compatibility for

        Returns:
            True if format is compatible, False otherwise
        """
        # Check if format has required codecs
        if not format_obj.vcodec and not format_obj.acodec:
            logger.warning(f"Format {format_obj.format_id} has no codecs")
            return False

        # Check for known problematic codecs
        problematic_codecs = ["none", "unknown"]
        if (
            format_obj.vcodec in problematic_codecs
            and format_obj.acodec in problematic_codecs
        ):
            logger.warning(f"Format {format_obj.format_id} has problematic codecs")
            return False

        # Check file extension
        if not format_obj.ext or format_obj.ext == "unknown":
            logger.warning(f"Format {format_obj.format_id} has unknown extension")
            return False

        # All checks passed
        return True

    def get_fallback_formats(
        self, formats: list[MediaFormat], failed_format: MediaFormat
    ) -> list[MediaFormat]:
        """
        Get fallback formats when primary format fails.

        Args:
            formats: All available formats
            failed_format: Format that failed

        Returns:
            List of fallback formats, ordered by preference
        """
        logger.info(
            f"Finding fallback formats for failed format: {failed_format.format_id}"
        )

        # Remove the failed format from consideration
        available_formats = [
            f for f in formats if f.format_id != failed_format.format_id
        ]

        if not available_formats:
            return []

        # Prefer formats with similar characteristics
        fallback_formats = []

        # First, try formats with same container
        same_container = [f for f in available_formats if f.ext == failed_format.ext]
        fallback_formats.extend(same_container)

        # Then, try formats with similar codecs
        if failed_format.vcodec:
            similar_vcodec = [
                f
                for f in available_formats
                if f.vcodec
                and f.vcodec == failed_format.vcodec
                and f not in fallback_formats
            ]
            fallback_formats.extend(similar_vcodec)

        if failed_format.acodec:
            similar_acodec = [
                f
                for f in available_formats
                if f.acodec
                and f.acodec == failed_format.acodec
                and f not in fallback_formats
            ]
            fallback_formats.extend(similar_acodec)

        # Finally, add remaining formats sorted by quality
        remaining_formats = [f for f in available_formats if f not in fallback_formats]
        remaining_formats.sort(key=lambda x: x.quality or 0, reverse=True)
        fallback_formats.extend(remaining_formats)

        logger.info(f"Found {len(fallback_formats)} fallback formats")
        return fallback_formats

    def supports_url(self, url: str) -> bool:
        """
        Check if URL is supported by yt-dlp extractors.

        Args:
            url: URL to check

        Returns:
            True if URL is supported, False otherwise
        """
        try:
            # Use yt-dlp's built-in extractor matching
            with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
                # This will return the extractor class if supported
                extractor = ydl._get_info_extractor_class(url)
                return extractor is not None
        except Exception:
            # If any error occurs, assume not supported
            return False

    def select_adaptive_quality_format(
        self,
        formats: list[MediaFormat],
        preferences: FormatPreferences | None = None,
        available_bandwidth: float | None = None,
        device_capabilities: dict[str, Any] | None = None,
    ) -> MediaFormat:
        """
        Select format using adaptive quality selection based on available formats and conditions.

        Args:
            formats: List of available formats
            preferences: Format selection preferences (uses config if None)
            available_bandwidth: Available bandwidth in bytes/second
            device_capabilities: Device capability constraints

        Returns:
            Adaptively selected optimal format

        Raises:
            ValueError: If no suitable format found
        """
        if not formats:
            raise ValueError("No formats available for adaptive selection")

        prefs = preferences or self.config.format_preferences

        logger.info("Performing adaptive quality format selection")

        try:
            return self.format_selector.select_adaptive_quality_format(
                formats=formats,
                preferences=prefs,
                available_bandwidth=available_bandwidth,
                device_capabilities=device_capabilities,
            )
        except Exception as e:
            logger.warning(
                f"Adaptive quality selection failed: {e}, falling back to standard selection"
            )
            return self.select_format(formats, preferences)

    def get_format_conversion_requirements(
        self,
        selected_format: MediaFormat,
        target_preferences: FormatPreferences | None = None,
    ) -> dict[str, Any]:
        """
        Determine what format conversion is needed for the selected format.

        Args:
            selected_format: The selected format
            target_preferences: Target format preferences (uses config if None)

        Returns:
            Dictionary describing conversion requirements
        """
        prefs = target_preferences or self.config.format_preferences

        return self.format_selector.get_format_conversion_requirements(
            selected_format, prefs
        )

    def select_format_for_bandwidth(
        self,
        formats: list[MediaFormat],
        available_bandwidth: float,
        preferences: FormatPreferences | None = None,
    ) -> MediaFormat:
        """
        Select optimal format based on available bandwidth.

        Args:
            formats: List of available formats
            available_bandwidth: Available bandwidth in bytes/second
            preferences: Format selection preferences (uses config if None)

        Returns:
            Bandwidth-optimized format selection

        Raises:
            ValueError: If no suitable format found
        """
        if not formats:
            raise ValueError("No formats available for bandwidth-based selection")

        prefs = preferences or self.config.format_preferences

        logger.info(f"Selecting format for bandwidth: {available_bandwidth:.0f} B/s")

        # Use the enhanced format selector with connection speed
        try:
            return self.format_selector.select_optimal_format(
                formats=formats,
                preferences=prefs,
                target_quality=None,  # Let it auto-determine
                adaptive=True,
                connection_speed=available_bandwidth,
            )
        except Exception as e:
            logger.warning(
                f"Bandwidth-based selection failed: {e}, falling back to standard selection"
            )
            return self.select_format(formats, preferences)

    def get_quality_alternatives_for_format(
        self,
        formats: list[MediaFormat],
        current_format: MediaFormat,
        direction: str = "lower",
    ) -> list[MediaFormat]:
        """
        Get quality alternatives for adaptive selection during downloads.

        Args:
            formats: List of available formats
            current_format: Current format to find alternatives for
            direction: "lower" or "higher" quality

        Returns:
            List of alternative formats ordered by preference
        """
        logger.info(
            f"Finding {direction} quality alternatives for format {current_format.format_id}"
        )

        try:
            alternatives = self.format_selector.get_quality_alternatives(
                formats=formats, current_format=current_format, direction=direction
            )

            logger.info(f"Found {len(alternatives)} {direction} quality alternatives")
            return alternatives

        except Exception as e:
            logger.error(f"Failed to get quality alternatives: {e}")
            return []

    def get_supported_sites(self) -> list[str]:
        """
        Get list of sites supported by yt-dlp.

        Returns:
            List of supported site names
        """
        try:
            # Get all extractor classes
            extractors = yt_dlp.extractor.gen_extractors()

            # Extract site names (IE_NAME attribute)
            sites = []
            for extractor in extractors:
                if hasattr(extractor, "IE_NAME") and extractor.IE_NAME:
                    sites.append(extractor.IE_NAME)

            return sorted(set(sites))
        except Exception as e:
            logger.error(f"Failed to get supported sites: {e}")
            return []
