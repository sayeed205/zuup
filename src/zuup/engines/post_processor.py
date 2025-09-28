"""Post-processor for format conversion and metadata handling."""

import asyncio
import logging
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from .media_models import (
    ChapterInfo,
    MediaConfig,
    MediaInfo,
    MediaMetadata,
    ProcessingResult,
    ProcessingStep,
    SubtitleInfo,
)

logger = logging.getLogger(__name__)


class PostProcessor:
    """Handles post-processing tasks like format conversion and metadata embedding."""

    def __init__(self, config: MediaConfig) -> None:
        """
        Initialize PostProcessor with configuration.

        Args:
            config: Media configuration for post-processing
        """
        self.config = config
        self._check_ffmpeg_availability()
        logger.info("PostProcessor initialized")

    def _check_ffmpeg_availability(self) -> None:
        """Check if FFmpeg is available for post-processing."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                logger.info("FFmpeg is available for post-processing")
            else:
                logger.warning("FFmpeg not found - format conversion will be limited")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("FFmpeg not found - format conversion will be limited")

    async def process_media(self, file_path: Path, info: MediaInfo) -> ProcessingResult:
        """
        Execute post-processing pipeline.

        Args:
            file_path: Path to the downloaded media file
            info: Media information from extraction

        Returns:
            ProcessingResult with success status and details
        """
        logger.info(f"Starting post-processing for {file_path}")

        result = ProcessingResult(
            success=True, output_path=file_path, processing_time=0.0, steps_completed=[]
        )

        start_time = time.time()

        try:
            # Convert format if audio extraction is enabled
            if self.config.extract_audio and not self._is_audio_file(file_path):
                logger.info("Converting to audio format")
                file_path = await self.convert_to_audio(file_path, info)
                result.steps_completed.append(ProcessingStep.CONVERT_FORMAT)
                result.output_path = file_path

            # Embed metadata if enabled
            if self.config.embed_metadata:
                logger.info("Embedding metadata")
                await self.embed_metadata(
                    file_path, self._create_metadata_from_info(info)
                )
                result.steps_completed.append(ProcessingStep.EMBED_METADATA)

            # Embed thumbnail if enabled and available
            if self.config.embed_thumbnail and info.thumbnail:
                logger.info("Embedding thumbnail")
                await self.embed_thumbnail(file_path, info.thumbnail)
                result.steps_completed.append(ProcessingStep.EMBED_THUMBNAIL)

            # Embed subtitles if enabled and available
            if self.config.embed_subtitles and info.subtitles:
                logger.info("Embedding subtitles")
                await self.embed_subtitles(file_path, info.subtitles)
                result.steps_completed.append(ProcessingStep.EMBED_SUBTITLES)

            # Organize files based on metadata templates
            if self.config.create_subdirectories:
                logger.info("Organizing files")
                organized_path = await self.organize_files(file_path, info)
                if organized_path != file_path:
                    result.output_path = organized_path
                result.steps_completed.append(ProcessingStep.ORGANIZE_FILES)

        except Exception as e:
            logger.error(f"Post-processing failed: {e}")
            result.success = False
            result.errors.append(str(e))

        result.processing_time = time.time() - start_time
        logger.info(f"Post-processing completed in {result.processing_time:.2f}s")

        return result

    async def convert_format(self, input_path: Path, output_format: str) -> Path:
        """
        Convert media file to specified format.

        Args:
            input_path: Path to input file
            output_format: Target format (e.g., 'mp3', 'mp4')

        Returns:
            Path to converted file

        Raises:
            RuntimeError: If conversion fails
        """
        logger.info(f"Converting {input_path} to {output_format}")

        # Create output path with new extension
        output_path = input_path.with_suffix(f".{output_format}")

        # Build FFmpeg command
        cmd = [
            "ffmpeg",
            "-i",
            str(input_path),
            "-y",  # Overwrite output file
        ]

        # Add format-specific options
        if output_format.lower() in ["mp3", "m4a", "aac", "ogg"]:
            # Audio conversion
            cmd.extend(
                [
                    "-vn",  # No video
                    "-acodec",
                    self._get_audio_codec(output_format),
                    "-ab",
                    f"{self.config.audio_quality}k",
                ]
            )
        elif output_format.lower() in ["mp4", "mkv", "avi"]:
            # Video conversion
            cmd.extend(
                [
                    "-c:v",
                    "libx264",
                    "-c:a",
                    "aac",
                    "-preset",
                    "medium",
                ]
            )

        cmd.append(str(output_path))

        try:
            # Run conversion in thread pool
            await self._run_ffmpeg_command(cmd)

            # Remove original file if conversion successful
            if output_path.exists() and output_path != input_path:
                input_path.unlink()
                logger.info(f"Conversion successful: {output_path}")
                return output_path
            else:
                raise RuntimeError("Conversion failed - output file not created")

        except Exception as e:
            logger.error(f"Format conversion failed: {e}")
            raise RuntimeError(f"Format conversion failed: {e}") from e

    async def convert_to_audio(self, input_path: Path, info: MediaInfo) -> Path:
        """
        Convert video file to audio format.

        Args:
            input_path: Path to input video file
            info: Media information

        Returns:
            Path to converted audio file
        """
        target_format = self.config.audio_format or "mp3"
        return await self.convert_format(input_path, target_format)

    async def embed_metadata(self, file_path: Path, metadata: MediaMetadata) -> None:
        """
        Embed metadata into media file.

        Args:
            file_path: Path to media file
            metadata: Metadata to embed

        Raises:
            RuntimeError: If metadata embedding fails
        """
        logger.info(f"Embedding metadata into {file_path}")

        # Create temporary file for processing
        temp_path = file_path.with_suffix(f".temp{file_path.suffix}")

        # Build FFmpeg command for metadata embedding
        cmd = [
            "ffmpeg",
            "-i",
            str(file_path),
            "-c",
            "copy",  # Copy streams without re-encoding
            "-y",  # Overwrite output file
        ]

        # Add metadata options
        if metadata.title:
            cmd.extend(["-metadata", f"title={metadata.title}"])
        if metadata.artist:
            cmd.extend(["-metadata", f"artist={metadata.artist}"])
        if metadata.album:
            cmd.extend(["-metadata", f"album={metadata.album}"])
        if metadata.description:
            cmd.extend(["-metadata", f"comment={metadata.description}"])
        if metadata.upload_date:
            cmd.extend(
                ["-metadata", f"date={metadata.upload_date.strftime('%Y-%m-%d')}"]
            )
        if metadata.uploader:
            cmd.extend(["-metadata", f"album_artist={metadata.uploader}"])

        cmd.append(str(temp_path))

        try:
            await self._run_ffmpeg_command(cmd)

            # Replace original file with processed file
            if temp_path.exists():
                file_path.unlink()
                temp_path.rename(file_path)
                logger.info("Metadata embedding successful")
            else:
                raise RuntimeError("Metadata embedding failed - temp file not created")

        except Exception as e:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            logger.error(f"Metadata embedding failed: {e}")
            raise RuntimeError(f"Metadata embedding failed: {e}") from e

    async def embed_thumbnail(self, file_path: Path, thumbnail_url: str) -> None:
        """
        Download and embed thumbnail into media file.

        Args:
            file_path: Path to media file
            thumbnail_url: URL of thumbnail image

        Raises:
            RuntimeError: If thumbnail embedding fails
        """
        logger.info(f"Embedding thumbnail from {thumbnail_url}")

        # Download thumbnail to temporary file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_thumb:
            temp_thumb_path = Path(temp_thumb.name)

        try:
            # Download thumbnail
            await self._download_thumbnail(thumbnail_url, temp_thumb_path)

            # Create temporary output file
            temp_output = file_path.with_suffix(f".temp{file_path.suffix}")

            # Build FFmpeg command for thumbnail embedding
            cmd = [
                "ffmpeg",
                "-i",
                str(file_path),
                "-i",
                str(temp_thumb_path),
                "-c",
                "copy",
                "-map",
                "0",
                "-map",
                "1",
                "-disposition:v:1",
                "attached_pic",
                "-y",
                str(temp_output),
            ]

            await self._run_ffmpeg_command(cmd)

            # Replace original file
            if temp_output.exists():
                file_path.unlink()
                temp_output.rename(file_path)
                logger.info("Thumbnail embedding successful")
            else:
                raise RuntimeError("Thumbnail embedding failed")

        except Exception as e:
            logger.error(f"Thumbnail embedding failed: {e}")
            raise RuntimeError(f"Thumbnail embedding failed: {e}") from e
        finally:
            # Clean up temporary files
            if temp_thumb_path.exists():
                temp_thumb_path.unlink()
            temp_output_path = file_path.with_suffix(f".temp{file_path.suffix}")
            if temp_output_path.exists():
                temp_output_path.unlink()

    async def embed_subtitles(
        self, file_path: Path, subtitles: dict[str, list[SubtitleInfo]]
    ) -> None:
        """
        Download and embed subtitles into media file.

        Args:
            file_path: Path to media file
            subtitles: Dictionary of subtitle tracks by language

        Raises:
            RuntimeError: If subtitle embedding fails
        """
        logger.info(f"Embedding subtitles into {file_path}")

        if not subtitles:
            logger.info("No subtitles to embed")
            return

        # Download subtitle files
        subtitle_files: list[Path] = []
        try:
            for lang, sub_list in subtitles.items():
                if sub_list:  # Take first subtitle for each language
                    sub_info = sub_list[0]
                    sub_path = await self._download_subtitle(sub_info, lang)
                    subtitle_files.append(sub_path)

            if not subtitle_files:
                logger.info("No subtitle files downloaded")
                return

            # Create temporary output file
            temp_output = file_path.with_suffix(f".temp{file_path.suffix}")

            # Build FFmpeg command for subtitle embedding
            cmd = [
                "ffmpeg",
                "-i",
                str(file_path),
            ]

            # Add subtitle inputs
            for sub_file in subtitle_files:
                cmd.extend(["-i", str(sub_file)])

            # Copy video and audio streams
            cmd.extend(["-c:v", "copy", "-c:a", "copy"])

            # Add subtitle streams
            for i, sub_file in enumerate(subtitle_files):
                cmd.extend(
                    [f"-c:s:{i}", "mov_text"]
                )  # Use mov_text for MP4 compatibility

            cmd.extend(["-y", str(temp_output)])

            await self._run_ffmpeg_command(cmd)

            # Replace original file
            if temp_output.exists():
                file_path.unlink()
                temp_output.rename(file_path)
                logger.info("Subtitle embedding successful")
            else:
                raise RuntimeError("Subtitle embedding failed")

        except Exception as e:
            logger.error(f"Subtitle embedding failed: {e}")
            raise RuntimeError(f"Subtitle embedding failed: {e}") from e
        finally:
            # Clean up subtitle files
            for sub_file in subtitle_files:
                if sub_file.exists():
                    sub_file.unlink()
            temp_output_path = file_path.with_suffix(f".temp{file_path.suffix}")
            if temp_output_path.exists():
                temp_output_path.unlink()

    async def embed_chapters(
        self, file_path: Path, chapters: list[ChapterInfo]
    ) -> None:
        """
        Embed chapter information into media file.

        Args:
            file_path: Path to media file
            chapters: List of chapter information

        Raises:
            RuntimeError: If chapter embedding fails
        """
        logger.info(f"Embedding {len(chapters)} chapters into {file_path}")

        if not chapters:
            logger.info("No chapters to embed")
            return

        # Create chapter metadata file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as chapter_file:
            chapter_path = Path(chapter_file.name)

            # Write chapter metadata in FFmpeg format
            chapter_file.write(";FFMETADATA1\n")
            for chapter in chapters:
                start_ms = int(chapter.start_time * 1000)
                end_ms = int(chapter.end_time * 1000)
                chapter_file.write("[CHAPTER]\n")
                chapter_file.write("TIMEBASE=1/1000\n")
                chapter_file.write(f"START={start_ms}\n")
                chapter_file.write(f"END={end_ms}\n")
                chapter_file.write(f"title={chapter.title}\n")

        try:
            # Create temporary output file
            temp_output = file_path.with_suffix(f".temp{file_path.suffix}")

            # Build FFmpeg command for chapter embedding
            cmd = [
                "ffmpeg",
                "-i",
                str(file_path),
                "-i",
                str(chapter_path),
                "-map_metadata",
                "1",
                "-c",
                "copy",
                "-y",
                str(temp_output),
            ]

            await self._run_ffmpeg_command(cmd)

            # Replace original file
            if temp_output.exists():
                file_path.unlink()
                temp_output.rename(file_path)
                logger.info("Chapter embedding successful")
            else:
                raise RuntimeError("Chapter embedding failed")

        except Exception as e:
            logger.error(f"Chapter embedding failed: {e}")
            raise RuntimeError(f"Chapter embedding failed: {e}") from e
        finally:
            # Clean up chapter file
            if chapter_path.exists():
                chapter_path.unlink()
            temp_output_path = file_path.with_suffix(f".temp{file_path.suffix}")
            if temp_output_path.exists():
                temp_output_path.unlink()

    async def organize_files(self, file_path: Path, info: MediaInfo) -> Path:
        """
        Organize files based on metadata templates.

        Args:
            file_path: Current file path
            info: Media information for organization

        Returns:
            New file path after organization

        Raises:
            RuntimeError: If file organization fails
        """
        logger.info(f"Organizing file {file_path}")

        try:
            # Create organized directory structure
            organized_dir = self._create_organized_directory(info)

            # Generate organized filename
            organized_filename = self._generate_organized_filename(file_path, info)

            # Create full organized path
            organized_path = organized_dir / organized_filename

            # Ensure target directory exists
            organized_path.parent.mkdir(parents=True, exist_ok=True)

            # Move file if path is different
            if organized_path != file_path:
                if organized_path.exists():
                    # Handle file conflicts
                    counter = 1
                    base_path = organized_path
                    while organized_path.exists():
                        stem = base_path.stem
                        suffix = base_path.suffix
                        organized_path = base_path.parent / f"{stem}_{counter}{suffix}"
                        counter += 1

                # Move the file
                shutil.move(str(file_path), str(organized_path))
                logger.info(f"File organized to: {organized_path}")
                return organized_path
            else:
                logger.info("File already in correct location")
                return file_path

        except Exception as e:
            logger.error(f"File organization failed: {e}")
            raise RuntimeError(f"File organization failed: {e}") from e

    def _create_organized_directory(self, info: MediaInfo) -> Path:
        """
        Create organized directory structure based on metadata.

        Args:
            info: Media information

        Returns:
            Path to organized directory
        """
        base_dir = self.config.output_directory

        # Create subdirectory based on uploader/channel
        if info.uploader:
            # Sanitize uploader name for filesystem
            uploader_safe = self._sanitize_filename(info.uploader)
            return base_dir / uploader_safe
        else:
            # Use extractor name as fallback
            extractor_safe = self._sanitize_filename(info.extractor_key)
            return base_dir / extractor_safe

    def _generate_organized_filename(self, file_path: Path, info: MediaInfo) -> str:
        """
        Generate organized filename based on metadata template.

        Args:
            file_path: Original file path
            info: Media information

        Returns:
            Organized filename
        """
        # Use title if available, otherwise use original filename
        if info.title:
            title_safe = self._sanitize_filename(info.title)
            return f"{title_safe}{file_path.suffix}"
        else:
            return file_path.name

    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename for filesystem compatibility.

        Args:
            filename: Original filename

        Returns:
            Sanitized filename
        """
        # Remove or replace problematic characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, "_")

        # Limit length and strip whitespace
        filename = filename.strip()[:200]  # Limit to 200 characters

        return filename

    def _create_metadata_from_info(self, info: MediaInfo) -> MediaMetadata:
        """
        Create MediaMetadata from MediaInfo.

        Args:
            info: Media information

        Returns:
            MediaMetadata object
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

    def _is_audio_file(self, file_path: Path) -> bool:
        """
        Check if file is an audio file.

        Args:
            file_path: Path to check

        Returns:
            True if file is audio format
        """
        audio_extensions = {".mp3", ".aac", ".ogg", ".flac", ".m4a", ".wav"}
        return file_path.suffix.lower() in audio_extensions

    def _get_audio_codec(self, format_name: str) -> str:
        """
        Get appropriate audio codec for format.

        Args:
            format_name: Target format name

        Returns:
            FFmpeg codec name
        """
        codec_map = {
            "mp3": "libmp3lame",
            "aac": "aac",
            "m4a": "aac",
            "ogg": "libvorbis",
            "flac": "flac",
            "wav": "pcm_s16le",
        }
        return codec_map.get(format_name.lower(), "libmp3lame")

    async def _run_ffmpeg_command(self, cmd: list[str]) -> None:
        """
        Run FFmpeg command in thread pool.

        Args:
            cmd: FFmpeg command as list of strings

        Raises:
            RuntimeError: If command fails
        """
        logger.debug(f"Running FFmpeg command: {' '.join(cmd)}")

        def run_command() -> None:
            """Run command synchronously."""
            try:
                result = subprocess.run(
                    cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout
                )
                if result.returncode != 0:
                    raise RuntimeError(f"FFmpeg failed: {result.stderr}")
            except subprocess.TimeoutExpired as e:
                raise RuntimeError(f"FFmpeg command timed out: {e}") from e

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, run_command)

    async def _download_thumbnail(self, url: str, output_path: Path) -> None:
        """
        Download thumbnail image.

        Args:
            url: Thumbnail URL
            output_path: Path to save thumbnail

        Raises:
            RuntimeError: If download fails
        """
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=30)
                response.raise_for_status()

                with open(output_path, "wb") as f:
                    f.write(response.content)

                logger.debug(f"Downloaded thumbnail to {output_path}")

        except Exception as e:
            logger.error(f"Thumbnail download failed: {e}")
            raise RuntimeError(f"Thumbnail download failed: {e}") from e

    async def _download_subtitle(self, sub_info: SubtitleInfo, language: str) -> Path:
        """
        Download subtitle file.

        Args:
            sub_info: Subtitle information
            language: Language code

        Returns:
            Path to downloaded subtitle file

        Raises:
            RuntimeError: If download fails
        """
        try:
            import httpx

            # Create temporary subtitle file
            with tempfile.NamedTemporaryFile(
                suffix=f".{language}.{sub_info.ext}", delete=False
            ) as temp_sub:
                temp_path = Path(temp_sub.name)

            async with httpx.AsyncClient() as client:
                response = await client.get(sub_info.url, timeout=30)
                response.raise_for_status()

                with open(temp_path, "wb") as f:
                    f.write(response.content)

                logger.debug(f"Downloaded subtitle to {temp_path}")
                return temp_path

        except Exception as e:
            logger.error(f"Subtitle download failed: {e}")
            raise RuntimeError(f"Subtitle download failed: {e}") from e

    async def cleanup(self) -> None:
        """Clean up post-processor resources."""
        logger.info("Cleaning up PostProcessor")
        # No persistent resources to clean up
        logger.info("PostProcessor cleanup completed")
