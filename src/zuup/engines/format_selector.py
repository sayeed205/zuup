"""Advanced format selection and quality control for media downloads."""

from enum import Enum
import logging

from .media_models import FormatPreferences, MediaFormat

logger = logging.getLogger(__name__)


class QualityTier(Enum):
    """Quality tier enumeration for adaptive selection."""

    ULTRA_HIGH = "ultra_high"  # 4K+, >2160p
    HIGH = "high"  # 1080p-2160p
    MEDIUM = "medium"  # 720p-1080p
    LOW = "low"  # 480p-720p
    VERY_LOW = "very_low"  # <480p


class CodecPreference(Enum):
    """Codec preference levels."""

    PREFERRED = "preferred"  # User's preferred codecs
    COMPATIBLE = "compatible"  # Widely compatible codecs
    FALLBACK = "fallback"  # Last resort codecs


class FormatSelector:
    """Advanced format selection with intelligent quality control."""

    def __init__(self) -> None:
        """Initialize format selector."""
        # Define codec compatibility tiers
        self.video_codec_tiers = {
            CodecPreference.PREFERRED: ["av01", "vp9", "h264", "hevc"],
            CodecPreference.COMPATIBLE: ["h264", "vp8", "vp9"],
            CodecPreference.FALLBACK: ["h263", "theora", "wmv"],
        }

        self.audio_codec_tiers = {
            CodecPreference.PREFERRED: ["opus", "aac", "mp3", "flac"],
            CodecPreference.COMPATIBLE: ["aac", "mp3", "vorbis"],
            CodecPreference.FALLBACK: ["wma", "ac3"],
        }

        # Define container preferences
        self.container_preferences = {
            "video": ["mp4", "mkv", "webm", "avi", "mov"],
            "audio": ["m4a", "mp3", "opus", "flac", "ogg"],
        }

        logger.info("FormatSelector initialized with codec and container preferences")

    def select_optimal_format(
        self,
        formats: list[MediaFormat],
        preferences: FormatPreferences,
        target_quality: QualityTier | None = None,
        adaptive: bool = True,
        connection_speed: float | None = None,
    ) -> MediaFormat:
        """
        Select optimal format using intelligent quality control.

        Args:
            formats: Available formats
            preferences: User format preferences
            target_quality: Target quality tier (auto-detected if None)
            adaptive: Enable adaptive quality selection
            connection_speed: Connection speed in bytes/second for adaptive selection

        Returns:
            Selected optimal format

        Raises:
            ValueError: If no suitable format found
        """
        if not formats:
            raise ValueError("No formats available for selection")

        logger.info(f"Selecting optimal format from {len(formats)} available formats")

        # Filter formats based on preferences
        filtered_formats = self._apply_preference_filters(formats, preferences)

        if not filtered_formats:
            logger.warning("No formats match preferences, using all formats")
            filtered_formats = formats

        # Determine target quality if not specified
        if target_quality is None and adaptive:
            target_quality = self._determine_optimal_quality(
                filtered_formats, connection_speed
            )

        # Apply intelligent quality-based filtering
        quality_filtered = self._apply_quality_based_filtering(
            filtered_formats, preferences, target_quality
        )

        if not quality_filtered:
            logger.warning(
                "Quality-based filtering removed all formats, using filtered formats"
            )
            quality_filtered = filtered_formats

        # Score and rank formats
        scored_formats = self._score_formats(
            quality_filtered, preferences, target_quality, connection_speed
        )

        # Select best format
        best_format = max(scored_formats, key=lambda x: x[1])[0]

        logger.info(
            f"Selected format: {best_format.format_id} "
            f"({best_format.ext}, {best_format.resolution or 'unknown'})"
        )

        return best_format

    def select_audio_only_format(
        self,
        formats: list[MediaFormat],
        preferences: FormatPreferences,
        target_bitrate: int | None = None,
        allow_conversion: bool = True,
    ) -> MediaFormat:
        """
        Select optimal audio-only format with quality control and conversion support.

        Args:
            formats: Available formats
            preferences: User format preferences
            target_bitrate: Target audio bitrate in kbps
            allow_conversion: Allow selection of video formats for audio extraction

        Returns:
            Selected audio format

        Raises:
            ValueError: If no suitable audio format found
        """
        logger.info("Selecting audio-only format")

        # Filter to audio formats with conversion support
        audio_formats = []

        # First priority: Pure audio-only formats
        pure_audio = [
            fmt
            for fmt in formats
            if fmt.vcodec in (None, "none") and fmt.acodec is not None
        ]
        audio_formats.extend(pure_audio)

        # Second priority: Formats with audio streams (if conversion allowed)
        if allow_conversion and not audio_formats:
            audio_with_video = [
                fmt
                for fmt in formats
                if fmt.acodec is not None and fmt.acodec != "none"
            ]
            audio_formats.extend(audio_with_video)

        if not audio_formats:
            raise ValueError("No audio formats available for extraction")

        # Apply audio-specific filtering
        filtered_formats = self._filter_audio_formats(
            audio_formats, preferences, target_bitrate
        )

        if not filtered_formats:
            filtered_formats = audio_formats

        # Score audio formats with conversion consideration
        scored_formats = self._score_audio_formats(
            filtered_formats, preferences, target_bitrate, allow_conversion
        )

        best_format = max(scored_formats, key=lambda x: x[1])[0]

        logger.info(
            f"Selected audio format: {best_format.format_id} "
            f"({best_format.ext}, {best_format.abr or 'unknown'} kbps)"
        )

        return best_format

    def get_quality_alternatives(
        self,
        formats: list[MediaFormat],
        current_format: MediaFormat,
        direction: str = "lower",
    ) -> list[MediaFormat]:
        """
        Get quality alternatives for adaptive selection.

        Args:
            formats: Available formats
            current_format: Current format to find alternatives for
            direction: "lower" or "higher" quality

        Returns:
            List of alternative formats ordered by preference
        """
        logger.info(
            f"Finding {direction} quality alternatives for {current_format.format_id}"
        )

        current_height = self._get_format_height(current_format)
        if current_height is None:
            return []

        alternatives = []

        for fmt in formats:
            if fmt.format_id == current_format.format_id:
                continue

            fmt_height = self._get_format_height(fmt)
            if fmt_height is None:
                continue

            if (direction == "lower" and fmt_height < current_height) or (
                direction == "higher" and fmt_height > current_height
            ):
                alternatives.append(fmt)

        # Sort by quality (ascending for lower, descending for higher)
        reverse_sort = direction == "higher"
        alternatives.sort(
            key=lambda x: self._get_format_height(x) or 0, reverse=reverse_sort
        )

        logger.info(f"Found {len(alternatives)} {direction} quality alternatives")
        return alternatives

    def _apply_preference_filters(
        self, formats: list[MediaFormat], preferences: FormatPreferences
    ) -> list[MediaFormat]:
        """Apply user preference filters to format list."""
        filtered = formats.copy()

        # Filter by audio/video only preferences
        if preferences.audio_only:
            filtered = [f for f in filtered if f.vcodec in (None, "none")]
        elif preferences.video_only:
            filtered = [f for f in filtered if f.acodec in (None, "none")]

        # Filter by resolution limits
        if preferences.max_height:
            filtered = [
                f
                for f in filtered
                if self._get_format_height(f) is None
                or self._get_format_height(f) <= preferences.max_height
            ]

        if preferences.max_width:
            filtered = [
                f
                for f in filtered
                if self._get_format_width(f) is None
                or self._get_format_width(f) <= preferences.max_width
            ]

        # Filter by preferred codecs
        if preferences.preferred_codecs:
            codec_filtered = []
            for fmt in filtered:
                if self._format_has_preferred_codec(fmt, preferences.preferred_codecs):
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

        # Filter by free format preference
        if preferences.prefer_free_formats:
            free_formats = [f for f in filtered if self._is_free_format(f)]
            if free_formats:
                filtered = free_formats

        return filtered

    def _filter_audio_formats(
        self,
        formats: list[MediaFormat],
        preferences: FormatPreferences,
        target_bitrate: int | None,
    ) -> list[MediaFormat]:
        """Filter formats for audio-only selection."""
        filtered = formats.copy()

        # Prefer audio-only formats
        audio_only = [f for f in filtered if f.vcodec in (None, "none")]
        if audio_only:
            filtered = audio_only

        # Filter by bitrate if specified
        if target_bitrate:
            bitrate_filtered = [
                f
                for f in filtered
                if f.abr is not None
                and abs(f.abr - target_bitrate) <= target_bitrate * 0.2
            ]
            if bitrate_filtered:
                filtered = bitrate_filtered

        # Apply codec preferences
        if preferences.preferred_codecs:
            codec_filtered = [
                f
                for f in filtered
                if f.acodec
                and any(codec in f.acodec for codec in preferences.preferred_codecs)
            ]
            if codec_filtered:
                filtered = codec_filtered

        return filtered

    def _score_formats(
        self,
        formats: list[MediaFormat],
        preferences: FormatPreferences,
        target_quality: QualityTier | None,
        connection_speed: float | None = None,
    ) -> list[tuple[MediaFormat, float]]:
        """Score formats based on quality and preferences."""
        scored_formats = []

        for fmt in formats:
            score = 0.0

            # Base quality score
            score += self._calculate_quality_score(fmt, target_quality)

            # Codec preference score
            score += self._calculate_codec_score(fmt, preferences)

            # Container preference score
            score += self._calculate_container_score(fmt, preferences)

            # Filesize efficiency score
            score += self._calculate_efficiency_score(fmt)

            # Preference value from yt-dlp
            if fmt.preference:
                score += fmt.preference * 0.1

            # Connection speed optimization score
            if connection_speed:
                score += self._calculate_connection_speed_score(fmt, connection_speed)

            # Advanced preference scores
            score += self._calculate_advanced_preference_score(fmt, preferences)

            scored_formats.append((fmt, score))

        return scored_formats

    def _score_audio_formats(
        self,
        formats: list[MediaFormat],
        preferences: FormatPreferences,
        target_bitrate: int | None,
        allow_conversion: bool = True,
    ) -> list[tuple[MediaFormat, float]]:
        """Score audio formats specifically."""
        scored_formats = []

        for fmt in formats:
            score = 0.0

            # Audio quality score
            if fmt.abr:
                if target_bitrate:
                    # Score based on proximity to target bitrate
                    bitrate_diff = abs(fmt.abr - target_bitrate)
                    score += max(0, 100 - bitrate_diff)
                else:
                    # Higher bitrate is better (up to a point)
                    score += min(fmt.abr, 320) * 0.3

            # Audio codec preference
            if fmt.acodec:
                for tier, codecs in self.audio_codec_tiers.items():
                    if any(codec in fmt.acodec for codec in codecs):
                        if tier == CodecPreference.PREFERRED:
                            score += 50
                        elif tier == CodecPreference.COMPATIBLE:
                            score += 30
                        else:
                            score += 10
                        break

            # Container preference for audio
            if fmt.ext in self.container_preferences["audio"]:
                score += 20

            # Prefer audio-only formats
            if fmt.vcodec in (None, "none"):
                score += 30
            elif allow_conversion and fmt.vcodec:
                # Slight penalty for needing conversion, but still viable
                score -= 5

            # Format conversion preference scoring
            if preferences.allow_format_conversion and fmt.vcodec:
                # Bonus for formats that are good for audio extraction
                if fmt.ext in ["mp4", "mkv", "webm"]:
                    score += 10
            elif not preferences.allow_format_conversion and fmt.vcodec:
                # Heavy penalty if conversion not allowed but format needs it
                score -= 50

            # Audio quality indicators
            if fmt.abr:
                # Prefer lossless or high-quality audio
                if fmt.abr >= 320:
                    score += 20
                elif fmt.abr >= 256:
                    score += 15
                elif fmt.abr >= 192:
                    score += 10
                elif fmt.abr >= 128:
                    score += 5

            scored_formats.append((fmt, score))

        return scored_formats

    def _apply_quality_based_filtering(
        self,
        formats: list[MediaFormat],
        preferences: FormatPreferences,
        target_quality: QualityTier | None,
    ) -> list[MediaFormat]:
        """Apply advanced quality-based filtering with resolution, bitrate, and codec preferences."""
        filtered = formats.copy()

        # Filter by bitrate preferences
        if preferences.max_video_bitrate:
            filtered = [
                f
                for f in filtered
                if f.vbr is None or f.vbr <= preferences.max_video_bitrate
            ]

        if preferences.min_video_bitrate:
            filtered = [
                f
                for f in filtered
                if f.vbr is None or f.vbr >= preferences.min_video_bitrate
            ]

        if preferences.max_audio_bitrate:
            filtered = [
                f
                for f in filtered
                if f.abr is None or f.abr <= preferences.max_audio_bitrate
            ]

        if preferences.min_audio_bitrate:
            filtered = [
                f
                for f in filtered
                if f.abr is None or f.abr >= preferences.min_audio_bitrate
            ]

        # Filter by FPS preferences
        if preferences.max_fps:
            filtered = [
                f for f in filtered if f.fps is None or f.fps <= preferences.max_fps
            ]

        if preferences.min_fps:
            filtered = [
                f for f in filtered if f.fps is None or f.fps >= preferences.min_fps
            ]

        # Prefer 60fps if requested
        if preferences.prefer_60fps:
            fps_60_formats = [f for f in filtered if f.fps and f.fps >= 60]
            if fps_60_formats:
                filtered = fps_60_formats

        # Filter by hardware decodable preference
        if preferences.prefer_hardware_decodable:
            hw_decodable_codecs = ["h264", "hevc", "h265", "vp9"]
            hw_formats = [
                f
                for f in filtered
                if f.vcodec
                and any(codec in f.vcodec.lower() for codec in hw_decodable_codecs)
            ]
            if hw_formats:
                # Prefer hardware decodable but don't exclude others entirely
                hw_formats.extend([f for f in filtered if f not in hw_formats])
                filtered = hw_formats

        # Avoid experimental codecs if requested
        if preferences.avoid_experimental_codecs:
            experimental_codecs = ["av01", "vvc", "h266"]
            stable_formats = [
                f
                for f in filtered
                if not f.vcodec
                or not any(codec in f.vcodec.lower() for codec in experimental_codecs)
            ]
            if stable_formats:
                filtered = stable_formats

        # Apply target quality filtering if specified
        if target_quality:
            quality_filtered = self._filter_by_quality_tier(filtered, target_quality)
            if quality_filtered:
                filtered = quality_filtered

        return filtered

    def _filter_by_quality_tier(
        self, formats: list[MediaFormat], target_quality: QualityTier
    ) -> list[MediaFormat]:
        """Filter formats by quality tier with tolerance."""
        quality_ranges = {
            QualityTier.ULTRA_HIGH: (2160, float("inf")),
            QualityTier.HIGH: (1080, 2160),
            QualityTier.MEDIUM: (720, 1080),
            QualityTier.LOW: (480, 720),
            QualityTier.VERY_LOW: (0, 480),
        }

        target_min, target_max = quality_ranges[target_quality]

        # Primary filter: exact quality tier match
        exact_matches = []
        for fmt in formats:
            height = self._get_format_height(fmt)
            if height and target_min <= height < target_max:
                exact_matches.append(fmt)

        if exact_matches:
            return exact_matches

        # Secondary filter: allow one tier up or down
        tolerance_formats = []
        for fmt in formats:
            height = self._get_format_height(fmt)
            if height:
                # Allow formats within reasonable range of target
                if (
                    (target_quality == QualityTier.ULTRA_HIGH and height >= 1440)
                    or (target_quality == QualityTier.HIGH and 720 <= height <= 2880)
                    or (target_quality == QualityTier.MEDIUM and 480 <= height <= 1440)
                    or (target_quality == QualityTier.LOW and 360 <= height <= 900)
                    or (target_quality == QualityTier.VERY_LOW and height <= 720)
                ):
                    tolerance_formats.append(fmt)

        return tolerance_formats

    def _determine_optimal_quality(
        self, formats: list[MediaFormat], connection_speed: float | None = None
    ) -> QualityTier:
        """Determine optimal quality tier based on available formats and connection speed."""
        heights = [
            self._get_format_height(f) for f in formats if self._get_format_height(f)
        ]

        if not heights:
            return QualityTier.MEDIUM

        max_height = max(heights)

        # Adjust quality based on connection speed if provided
        if connection_speed:
            # Connection speed thresholds (bytes per second)
            if connection_speed >= 10 * 1024 * 1024:  # >= 10 MB/s
                target_by_speed = (
                    QualityTier.ULTRA_HIGH if max_height >= 2160 else QualityTier.HIGH
                )
            elif connection_speed >= 5 * 1024 * 1024:  # >= 5 MB/s
                target_by_speed = QualityTier.HIGH
            elif connection_speed >= 2 * 1024 * 1024:  # >= 2 MB/s
                target_by_speed = QualityTier.MEDIUM
            elif connection_speed >= 500 * 1024:  # >= 500 KB/s
                target_by_speed = QualityTier.LOW
            else:
                target_by_speed = QualityTier.VERY_LOW

            # Choose the more conservative option between available quality and speed-based quality
            available_quality = self._get_quality_tier_for_height(max_height)
            return min(
                available_quality,
                target_by_speed,
                key=lambda x: list(QualityTier).index(x),
            )

        # Default quality selection based on available formats
        return self._get_quality_tier_for_height(max_height)

    def _get_quality_tier_for_height(self, height: int) -> QualityTier:
        """Get quality tier for a given height."""
        if height >= 2160:
            return QualityTier.HIGH  # Conservative default, not ultra-high
        elif height >= 1080:
            return QualityTier.HIGH
        elif height >= 720:
            return QualityTier.MEDIUM
        elif height >= 480:
            return QualityTier.LOW
        else:
            return QualityTier.VERY_LOW

    def _calculate_quality_score(
        self, fmt: MediaFormat, target_quality: QualityTier | None
    ) -> float:
        """Calculate quality score for a format."""
        if not target_quality:
            return 0.0

        height = self._get_format_height(fmt)
        if height is None:
            return 0.0

        # Define quality ranges
        quality_ranges = {
            QualityTier.ULTRA_HIGH: (2160, 4320),
            QualityTier.HIGH: (1080, 2160),
            QualityTier.MEDIUM: (720, 1080),
            QualityTier.LOW: (480, 720),
            QualityTier.VERY_LOW: (0, 480),
        }

        target_min, target_max = quality_ranges[target_quality]

        if target_min <= height <= target_max:
            # Perfect match for target quality
            return 100.0
        elif height < target_min:
            # Lower than target - penalize based on difference
            return max(0, 50 - (target_min - height) * 0.1)
        else:
            # Higher than target - slight penalty for bandwidth
            return max(0, 80 - (height - target_max) * 0.05)

    def _calculate_codec_score(
        self, fmt: MediaFormat, preferences: FormatPreferences
    ) -> float:
        """Calculate codec preference score."""
        score = 0.0

        # Video codec scoring
        if fmt.vcodec and fmt.vcodec != "none":
            for tier, codecs in self.video_codec_tiers.items():
                if any(codec in fmt.vcodec for codec in codecs):
                    if tier == CodecPreference.PREFERRED:
                        score += 30
                    elif tier == CodecPreference.COMPATIBLE:
                        score += 20
                    else:
                        score += 5
                    break

        # Audio codec scoring
        if fmt.acodec and fmt.acodec != "none":
            for tier, codecs in self.audio_codec_tiers.items():
                if any(codec in fmt.acodec for codec in codecs):
                    if tier == CodecPreference.PREFERRED:
                        score += 20
                    elif tier == CodecPreference.COMPATIBLE:
                        score += 15
                    else:
                        score += 5
                    break

        return score

    def _calculate_container_score(
        self, fmt: MediaFormat, preferences: FormatPreferences
    ) -> float:
        """Calculate container preference score."""
        if fmt.ext in self.container_preferences["video"]:
            return 15.0
        elif fmt.ext in self.container_preferences["audio"]:
            return 10.0
        return 0.0

    def _calculate_connection_speed_score(
        self, fmt: MediaFormat, connection_speed: float
    ) -> float:
        """Calculate score based on connection speed optimization."""
        if not fmt.filesize and not fmt.filesize_approx:
            return 0.0

        filesize = fmt.filesize or fmt.filesize_approx

        # Estimate download time
        estimated_time = (
            filesize / connection_speed if connection_speed > 0 else float("inf")
        )

        # Score based on reasonable download time (prefer formats that download in reasonable time)
        if estimated_time <= 300:  # <= 5 minutes
            return 20.0
        elif estimated_time <= 900:  # <= 15 minutes
            return 10.0
        elif estimated_time <= 1800:  # <= 30 minutes
            return 5.0
        else:
            return -5.0  # Penalize very long downloads

    def _calculate_advanced_preference_score(
        self, fmt: MediaFormat, preferences: FormatPreferences
    ) -> float:
        """Calculate score based on advanced preferences."""
        score = 0.0

        # FPS preference scoring
        if fmt.fps:
            if preferences.prefer_60fps and fmt.fps >= 60:
                score += 15.0
            elif (preferences.max_fps and fmt.fps <= preferences.max_fps) or (
                preferences.min_fps and fmt.fps >= preferences.min_fps
            ):
                score += 5.0

        # Hardware decodable preference
        if preferences.prefer_hardware_decodable and fmt.vcodec:
            hw_codecs = ["h264", "hevc", "h265", "vp9"]
            if any(codec in fmt.vcodec.lower() for codec in hw_codecs):
                score += 10.0

        # Native format preference
        if preferences.prefer_native_formats:
            # Prefer formats that don't need conversion
            native_containers = ["mp4", "mkv", "webm", "m4a", "mp3"]
            if fmt.ext in native_containers:
                score += 8.0

        # Bitrate preference scoring
        if fmt.vbr and preferences.max_video_bitrate:
            # Score based on how close to max bitrate (but not over)
            if fmt.vbr <= preferences.max_video_bitrate:
                ratio = fmt.vbr / preferences.max_video_bitrate
                score += ratio * 10.0  # Higher bitrate within limit is better

        if fmt.abr and preferences.target_audio_bitrate:
            # Score based on proximity to target audio bitrate
            diff = abs(fmt.abr - preferences.target_audio_bitrate)
            if diff <= 32:  # Within 32 kbps
                score += 15.0 - (diff * 0.3)

        return score

    def _calculate_efficiency_score(self, fmt: MediaFormat) -> float:
        """Calculate filesize efficiency score."""
        if not fmt.filesize and not fmt.filesize_approx:
            return 0.0

        filesize = fmt.filesize or fmt.filesize_approx
        height = self._get_format_height(fmt)

        if not height or not filesize:
            return 0.0

        # Calculate bytes per pixel (lower is more efficient)
        pixels = height * (height * 16 // 9)  # Assume 16:9 aspect ratio
        bytes_per_pixel = filesize / pixels

        # Score based on efficiency (arbitrary scale)
        if bytes_per_pixel < 0.1:
            return 10.0
        elif bytes_per_pixel < 0.2:
            return 5.0
        else:
            return 0.0

    def _get_format_height(self, fmt: MediaFormat) -> int | None:
        """Extract height from format resolution."""
        if not fmt.resolution:
            return None

        try:
            if "x" in fmt.resolution:
                return int(fmt.resolution.split("x")[1])
            elif "p" in fmt.resolution:
                return int(fmt.resolution.replace("p", ""))
            else:
                return int(fmt.resolution)
        except (ValueError, IndexError):
            return None

    def _get_format_width(self, fmt: MediaFormat) -> int | None:
        """Extract width from format resolution."""
        if not fmt.resolution:
            return None

        try:
            if "x" in fmt.resolution:
                return int(fmt.resolution.split("x")[0])
            else:
                # Estimate width from height assuming 16:9
                height = self._get_format_height(fmt)
                return int(height * 16 / 9) if height else None
        except (ValueError, IndexError):
            return None

    def _format_has_preferred_codec(
        self, fmt: MediaFormat, preferred_codecs: list[str]
    ) -> bool:
        """Check if format has any of the preferred codecs."""
        if fmt.vcodec and any(codec in fmt.vcodec for codec in preferred_codecs):
            return True
        if fmt.acodec and any(codec in fmt.acodec for codec in preferred_codecs):
            return True
        return False

    def _is_free_format(self, fmt: MediaFormat) -> bool:
        """Check if format uses free/open codecs and containers."""
        free_containers = {"webm", "ogg", "opus", "flac"}
        free_video_codecs = {"vp8", "vp9", "av01", "theora"}
        free_audio_codecs = {"vorbis", "opus", "flac"}

        if fmt.ext in free_containers:
            return True

        if fmt.vcodec and any(codec in fmt.vcodec for codec in free_video_codecs):
            return True

        if fmt.acodec and any(codec in fmt.acodec for codec in free_audio_codecs):
            return True

        return False

    def select_adaptive_quality_format(
        self,
        formats: list[MediaFormat],
        preferences: FormatPreferences,
        available_bandwidth: float | None = None,
        device_capabilities: dict[str, any] | None = None,
    ) -> MediaFormat:
        """
        Select format using adaptive quality selection based on available formats and conditions.

        Args:
            formats: Available formats
            preferences: User format preferences
            available_bandwidth: Available bandwidth in bytes/second
            device_capabilities: Device capability constraints (screen resolution, codecs, etc.)

        Returns:
            Adaptively selected optimal format

        Raises:
            ValueError: If no suitable format found
        """
        if not formats:
            raise ValueError("No formats available for adaptive selection")

        logger.info(
            f"Performing adaptive quality selection from {len(formats)} formats"
        )

        # Analyze available format distribution
        format_analysis = self.analyze_format_distribution(formats)

        # Create adaptive preferences based on conditions
        adaptive_prefs = self._create_adaptive_preferences(
            preferences, available_bandwidth, device_capabilities, format_analysis
        )

        # Determine adaptive quality target
        adaptive_quality = self._determine_adaptive_quality_target(
            formats, available_bandwidth, device_capabilities, format_analysis
        )

        logger.info(
            f"Adaptive quality target: {adaptive_quality.value if adaptive_quality else 'auto'}"
        )

        try:
            # Use adaptive preferences for selection
            return self.select_optimal_format(
                formats=formats,
                preferences=adaptive_prefs,
                target_quality=adaptive_quality,
                adaptive=True,
                connection_speed=available_bandwidth,
            )
        except Exception as e:
            logger.warning(
                f"Adaptive selection failed: {e}, falling back to standard selection"
            )
            return self.select_optimal_format(formats, preferences, adaptive=True)

    def _create_adaptive_preferences(
        self,
        base_preferences: FormatPreferences,
        available_bandwidth: float | None,
        device_capabilities: dict[str, any] | None,
        format_analysis: dict[str, any],
    ) -> FormatPreferences:
        """Create adaptive preferences based on current conditions."""
        adaptive_prefs = base_preferences.model_copy()

        # Adapt based on bandwidth
        if available_bandwidth:
            if available_bandwidth < 1 * 1024 * 1024:  # < 1 MB/s
                adaptive_prefs.max_height = min(adaptive_prefs.max_height or 720, 480)
                adaptive_prefs.max_video_bitrate = min(
                    adaptive_prefs.max_video_bitrate or 2000, 1000
                )
                adaptive_prefs.target_quality = "low"
            elif available_bandwidth < 3 * 1024 * 1024:  # < 3 MB/s
                adaptive_prefs.max_height = min(adaptive_prefs.max_height or 1080, 720)
                adaptive_prefs.max_video_bitrate = min(
                    adaptive_prefs.max_video_bitrate or 5000, 3000
                )
                adaptive_prefs.target_quality = "medium"
            elif available_bandwidth < 8 * 1024 * 1024:  # < 8 MB/s
                adaptive_prefs.max_height = min(adaptive_prefs.max_height or 1440, 1080)
                adaptive_prefs.target_quality = "high"

        # Adapt based on device capabilities
        if device_capabilities:
            screen_height = device_capabilities.get("screen_height")
            if screen_height:
                # Don't download higher resolution than screen can display
                adaptive_prefs.max_height = min(
                    adaptive_prefs.max_height or screen_height, screen_height
                )

            supported_codecs = device_capabilities.get("supported_codecs", [])
            if supported_codecs:
                # Filter to supported codecs
                adaptive_prefs.preferred_codecs = [
                    codec
                    for codec in (adaptive_prefs.preferred_codecs or [])
                    if codec in supported_codecs
                ]
                if not adaptive_prefs.preferred_codecs:
                    adaptive_prefs.preferred_codecs = supported_codecs

        # Adapt based on available format distribution
        if not format_analysis.get("has_audio_only") and adaptive_prefs.audio_only:
            # Enable conversion if no pure audio formats available
            adaptive_prefs.allow_format_conversion = True

        # Prefer free formats if limited format variety
        if format_analysis.get("total_formats", 0) < 5:
            adaptive_prefs.prefer_free_formats = False  # Be less restrictive

        return adaptive_prefs

    def _determine_adaptive_quality_target(
        self,
        formats: list[MediaFormat],
        available_bandwidth: float | None,
        device_capabilities: dict[str, any] | None,
        format_analysis: dict[str, any],
    ) -> QualityTier | None:
        """Determine adaptive quality target based on conditions."""
        # Start with format-based quality
        base_quality = self._determine_optimal_quality(formats, available_bandwidth)

        # Adjust based on format availability
        quality_dist = format_analysis.get("quality_distribution", {})

        # If target quality has very few options, consider alternatives
        target_formats_count = 0
        if base_quality == QualityTier.ULTRA_HIGH:
            target_formats_count = quality_dist.get("4K+", 0)
        elif base_quality == QualityTier.HIGH:
            target_formats_count = quality_dist.get("1080p", 0)
        elif base_quality == QualityTier.MEDIUM:
            target_formats_count = quality_dist.get("720p", 0)
        elif base_quality == QualityTier.LOW:
            target_formats_count = quality_dist.get("480p", 0)

        # If very few formats in target quality, be more flexible
        if target_formats_count <= 1:
            logger.info(
                f"Limited formats in {base_quality.value} quality, enabling adaptive fallback"
            )
            # Return None to enable more flexible selection
            return None

        return base_quality

    def get_format_conversion_requirements(
        self, selected_format: MediaFormat, target_preferences: FormatPreferences
    ) -> dict[str, any]:
        """
        Determine what format conversion is needed for the selected format.

        Args:
            selected_format: The selected format
            target_preferences: Target format preferences

        Returns:
            Dictionary describing conversion requirements
        """
        conversion_info = {
            "needs_conversion": False,
            "conversion_type": None,
            "target_container": None,
            "target_audio_codec": None,
            "target_video_codec": None,
            "extract_audio_only": False,
            "estimated_quality_loss": "none",
        }

        # Check if audio extraction is needed
        if (
            target_preferences.audio_only
            and selected_format.vcodec
            and selected_format.vcodec != "none"
        ):
            conversion_info["needs_conversion"] = True
            conversion_info["conversion_type"] = "audio_extraction"
            conversion_info["extract_audio_only"] = True
            conversion_info["estimated_quality_loss"] = "minimal"

            # Determine target audio format
            if target_preferences.preferred_containers:
                audio_containers = [
                    c
                    for c in target_preferences.preferred_containers
                    if c in ["mp3", "m4a", "ogg", "flac"]
                ]
                if audio_containers:
                    conversion_info["target_container"] = audio_containers[0]

            if not conversion_info["target_container"]:
                # Default audio container based on codec
                if selected_format.acodec:
                    if "aac" in selected_format.acodec:
                        conversion_info["target_container"] = "m4a"
                    elif "mp3" in selected_format.acodec:
                        conversion_info["target_container"] = "mp3"
                    elif "opus" in selected_format.acodec:
                        conversion_info["target_container"] = "ogg"
                    else:
                        conversion_info["target_container"] = "mp3"  # Safe default

        # Check if container conversion is needed
        if (
            target_preferences.preferred_containers
            and selected_format.ext not in target_preferences.preferred_containers
        ):
            conversion_info["needs_conversion"] = True
            if conversion_info["conversion_type"] is None:
                conversion_info["conversion_type"] = "container_conversion"
            conversion_info["target_container"] = (
                target_preferences.preferred_containers[0]
            )
            conversion_info["estimated_quality_loss"] = "minimal"

        # Check if codec conversion is needed
        needs_video_conversion = False
        needs_audio_conversion = False

        if (
            target_preferences.preferred_codecs
            and selected_format.vcodec
            and not any(
                codec in selected_format.vcodec
                for codec in target_preferences.preferred_codecs
            )
        ):
            needs_video_conversion = True
            conversion_info["target_video_codec"] = target_preferences.preferred_codecs[
                0
            ]

        if (
            target_preferences.preferred_codecs
            and selected_format.acodec
            and not any(
                codec in selected_format.acodec
                for codec in target_preferences.preferred_codecs
            )
        ):
            needs_audio_conversion = True
            # Find audio codec in preferred list
            audio_codecs = ["aac", "mp3", "opus", "flac"]
            for codec in target_preferences.preferred_codecs:
                if codec in audio_codecs:
                    conversion_info["target_audio_codec"] = codec
                    break

        if needs_video_conversion or needs_audio_conversion:
            conversion_info["needs_conversion"] = True
            if conversion_info["conversion_type"] is None:
                conversion_info["conversion_type"] = "codec_conversion"
            conversion_info["estimated_quality_loss"] = (
                "moderate" if needs_video_conversion else "minimal"
            )

        return conversion_info

    def analyze_format_distribution(self, formats: list[MediaFormat]) -> dict[str, any]:
        """Analyze the distribution of available formats for debugging."""
        analysis = {
            "total_formats": len(formats),
            "quality_distribution": {},
            "codec_distribution": {"video": {}, "audio": {}},
            "container_distribution": {},
            "has_audio_only": False,
            "has_video_only": False,
        }

        for fmt in formats:
            # Quality distribution
            height = self._get_format_height(fmt)
            if height:
                if height >= 2160:
                    tier = "4K+"
                elif height >= 1080:
                    tier = "1080p"
                elif height >= 720:
                    tier = "720p"
                elif height >= 480:
                    tier = "480p"
                else:
                    tier = "<480p"

                analysis["quality_distribution"][tier] = (
                    analysis["quality_distribution"].get(tier, 0) + 1
                )

            # Codec distribution
            if fmt.vcodec and fmt.vcodec != "none":
                analysis["codec_distribution"]["video"][fmt.vcodec] = (
                    analysis["codec_distribution"]["video"].get(fmt.vcodec, 0) + 1
                )

            if fmt.acodec and fmt.acodec != "none":
                analysis["codec_distribution"]["audio"][fmt.acodec] = (
                    analysis["codec_distribution"]["audio"].get(fmt.acodec, 0) + 1
                )

            # Container distribution
            analysis["container_distribution"][fmt.ext] = (
                analysis["container_distribution"].get(fmt.ext, 0) + 1
            )

            # Audio/video only detection
            if fmt.vcodec in (None, "none") and fmt.acodec:
                analysis["has_audio_only"] = True
            if fmt.acodec in (None, "none") and fmt.vcodec:
                analysis["has_video_only"] = True

        return analysis
