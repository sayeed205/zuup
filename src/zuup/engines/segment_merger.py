"""SegmentMerger for file assembly and integrity verification with memory-mapped I/O support."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

from .pycurl_models import (
    CompletedSegment,
    DownloadSegment,
    FinalizeResult,
    MergeResult,
    SegmentMergeInfo,
    SegmentStatus,
)
from .memory_mapped_io import (
    MemoryMappedSegmentWriter,
    should_use_memory_mapping,
    get_optimal_mmap_threshold,
)

logger = logging.getLogger(__name__)


class SegmentMerger:
    """Handles merging completed segments into the final file with memory-mapped I/O optimization."""

    def __init__(
        self, 
        target_path: Path, 
        temp_dir: Path,
        total_file_size: int = 0,
        use_memory_mapping: bool | None = None,
    ) -> None:
        """
        Initialize SegmentMerger with memory-mapped I/O support.

        Args:
            target_path: Final destination path for the merged file
            temp_dir: Directory for temporary files and resume data
            total_file_size: Total expected file size for optimization decisions
            use_memory_mapping: Force enable/disable memory mapping (None for auto-detect)
        """
        self.target_path = target_path
        self.temp_dir = temp_dir
        self.total_file_size = total_file_size
        self.resume_data_path = temp_dir / "resume_data.json"

        # Determine if we should use memory mapping
        if use_memory_mapping is None:
            self.use_memory_mapping = should_use_memory_mapping(
                total_file_size, get_optimal_mmap_threshold()
            )
        else:
            self.use_memory_mapping = use_memory_mapping

        # Memory-mapped writer for large files
        self._mmap_writer: MemoryMappedSegmentWriter | None = None

        # Ensure directories exist
        self.target_path.parent.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Merge tracking
        self._merge_info = SegmentMergeInfo(
            total_segments=0, merged_segments=0, bytes_merged=0, total_bytes=total_file_size
        )

        logger.info(
            f"Initialized SegmentMerger for {target_path} "
            f"(size={total_file_size}, mmap={'enabled' if self.use_memory_mapping else 'disabled'})"
        )

        logger.debug(f"Initialized SegmentMerger for {target_path}")

    async def merge_segment(self, segment: CompletedSegment) -> MergeResult:
        """
        Merge a completed segment into the final file.

        Args:
            segment: Completed segment to merge

        Returns:
            Dictionary containing merge result information

        Raises:
            FileNotFoundError: If segment temp file doesn't exist
            IOError: If merge operation fails
        """
        logger.info(f"Merging segment {segment.segment.id}")

        start_time = time.time()

        try:
            # Validate segment file exists and has expected size
            if not segment.temp_file_path.exists():
                raise FileNotFoundError(
                    f"Segment temp file not found: {segment.temp_file_path}"
                )

            actual_size = segment.temp_file_path.stat().st_size
            expected_size = segment.segment.segment_size

            if actual_size != expected_size:
                logger.warning(
                    f"Segment {segment.segment.id} size mismatch: "
                    f"expected {expected_size}, got {actual_size}"
                )

            # Verify segment integrity if checksum is provided
            if segment.checksum:
                calculated_checksum = await self._calculate_file_checksum(
                    segment.temp_file_path
                )
                if calculated_checksum != segment.checksum:
                    raise ValueError(
                        f"Segment {segment.segment.id} checksum mismatch: "
                        f"expected {segment.checksum}, got {calculated_checksum}"
                    )
                logger.debug(f"Segment {segment.segment.id} checksum verified")

            # Perform atomic merge operation
            bytes_merged = await self._merge_segment_data(segment)

            # Update merge tracking
            self._merge_info.merged_segments += 1
            self._merge_info.bytes_merged += bytes_merged
            self._merge_info.current_segment = segment.segment.id

            # Update segment status
            segment.segment.status = SegmentStatus.MERGED

            merge_time = time.time() - start_time

            logger.info(
                f"Successfully merged segment {segment.segment.id} "
                f"({bytes_merged} bytes in {merge_time:.2f}s)"
            )

            return {
                "success": True,
                "segment_id": segment.segment.id,
                "bytes_merged": bytes_merged,
                "merge_time": merge_time,
                "checksum_verified": segment.checksum is not None,
            }

        except Exception as e:
            logger.error(f"Failed to merge segment {segment.segment.id}: {e}")
            return {
                "success": False,
                "segment_id": segment.segment.id,
                "error": str(e),
                "merge_time": time.time() - start_time,
            }

    async def _merge_segment_data(self, segment: CompletedSegment) -> int:
        """
        Merge segment data into the target file using optimized I/O.

        Args:
            segment: Completed segment to merge

        Returns:
            Number of bytes merged

        Raises:
            IOError: If merge operation fails
        """
        if self.use_memory_mapping and self.total_file_size > 0:
            return await self._merge_segment_mmap(segment)
        else:
            return await self._merge_segment_traditional(segment)

    async def _merge_segment_mmap(self, segment: CompletedSegment) -> int:
        """
        Merge segment using memory-mapped I/O for large files.

        Args:
            segment: Completed segment to merge

        Returns:
            Number of bytes merged
        """
        # Initialize memory-mapped writer if not already done
        if self._mmap_writer is None:
            self._mmap_writer = MemoryMappedSegmentWriter(
                self.target_path, self.total_file_size
            )
            self._mmap_writer.open()

        # Read segment data
        with open(segment.temp_file_path, "rb") as segment_file:
            segment_data = segment_file.read()

        # Write segment to memory-mapped file
        bytes_merged = self._mmap_writer.write_segment(
            segment.segment.start_byte, segment_data
        )

        # Sync to disk periodically
        self._mmap_writer.sync()

        logger.debug(
            f"Memory-mapped merge: {bytes_merged} bytes for segment {segment.segment.id}"
        )

        return bytes_merged

    async def _merge_segment_traditional(self, segment: CompletedSegment) -> int:
        """
        Merge segment using traditional file I/O for smaller files.

        Args:
            segment: Completed segment to merge

        Returns:
            Number of bytes merged
        """
        # Create target file if it doesn't exist
        if not self.target_path.exists():
            if self.total_file_size > 0:
                # Create sparse file of expected size
                with open(self.target_path, "wb") as f:
                    f.seek(self.total_file_size - 1)
                    f.write(b"\0")
            else:
                self.target_path.touch()

        bytes_merged = 0

        # Use direct file I/O for smaller files or when mmap is disabled
        try:
            with open(self.target_path, "r+b") as target_file:
                # Seek to the correct position for this segment
                target_file.seek(segment.segment.start_byte)

                # Copy segment data in chunks
                with open(segment.temp_file_path, "rb") as segment_file:
                    while True:
                        chunk = segment_file.read(64 * 1024)  # 64KB chunks
                        if not chunk:
                            break
                        target_file.write(chunk)
                        bytes_merged += len(chunk)

                # Ensure data is written to disk
                target_file.flush()
                os.fsync(target_file.fileno())

            logger.debug(
                f"Traditional merge: {bytes_merged} bytes for segment {segment.segment.id}"
            )

        except Exception as e:
            raise OSError(f"Failed to merge segment data: {e}") from e

        return bytes_merged

    async def finalize_download(
        self, segments: list[CompletedSegment]
    ) -> FinalizeResult:
        """
        Finalize the download by verifying all segments are merged correctly.

        Args:
            segments: List of all completed segments

        Returns:
            Dictionary containing finalization result

        Raises:
            ValueError: If segments are incomplete or invalid
            IOError: If finalization fails
        """
        logger.info(f"Finalizing download with {len(segments)} segments")

        start_time = time.time()

        try:
            # Validate all segments are present and merged
            self._validate_segments_complete(segments)

            # Verify file integrity
            integrity_result = await self._verify_file_integrity(segments)

            # Clean up temporary files
            self.cleanup_temp_files(segments)

            # Remove resume data
            if self.resume_data_path.exists():
                self.resume_data_path.unlink()
                logger.debug("Removed resume data file")

            finalize_time = time.time() - start_time

            logger.info(
                f"Successfully finalized download: {self.target_path} "
                f"({integrity_result['total_size']} bytes in {finalize_time:.2f}s)"
            )

            return {
                "success": True,
                "target_path": str(self.target_path),
                "total_size": integrity_result["total_size"],
                "segments_count": len(segments),
                "finalize_time": finalize_time,
                "integrity_verified": integrity_result["verified"],
                "checksum": integrity_result.get("checksum"),
            }

        except Exception as e:
            logger.error(f"Failed to finalize download: {e}")
            return {
                "success": False,
                "error": str(e),
                "finalize_time": time.time() - start_time,
            }

    def _validate_segments_complete(self, segments: list[CompletedSegment]) -> None:
        """
        Validate that all segments are complete and properly ordered.

        Args:
            segments: List of completed segments

        Raises:
            ValueError: If segments are incomplete or invalid
        """
        if not segments:
            raise ValueError("No segments provided for finalization")

        # Sort segments by start byte
        sorted_segments = sorted(segments, key=lambda s: s.segment.start_byte)

        # Check for gaps or overlaps
        expected_byte = 0
        for segment in sorted_segments:
            if segment.segment.start_byte != expected_byte:
                if segment.segment.start_byte > expected_byte:
                    raise ValueError(
                        f"Gap in segments: expected byte {expected_byte}, "
                        f"got {segment.segment.start_byte}"
                    )
                else:
                    raise ValueError(
                        f"Overlapping segments: expected byte {expected_byte}, "
                        f"got {segment.segment.start_byte}"
                    )

            # Verify segment is marked as merged
            if segment.segment.status != SegmentStatus.MERGED:
                raise ValueError(
                    f"Segment {segment.segment.id} not merged (status: {segment.segment.status})"
                )

            expected_byte = segment.segment.end_byte + 1

        logger.debug(f"Validated {len(segments)} segments are complete and ordered")

    async def _verify_file_integrity(
        self, segments: list[CompletedSegment]
    ) -> dict[str, Any]:
        """
        Verify the integrity of the final merged file.

        Args:
            segments: List of completed segments

        Returns:
            Dictionary with integrity verification results
        """
        if not self.target_path.exists():
            raise FileNotFoundError(f"Target file not found: {self.target_path}")

        # Check file size
        actual_size = self.target_path.stat().st_size
        expected_size = sum(segment.segment.segment_size for segment in segments)

        if actual_size != expected_size:
            raise ValueError(
                f"File size mismatch: expected {expected_size}, got {actual_size}"
            )

        # Calculate file checksum for integrity verification
        file_checksum = await self._calculate_file_checksum(self.target_path)

        logger.debug(
            f"File integrity verified: {actual_size} bytes, checksum {file_checksum[:16]}..."
        )

        return {
            "verified": True,
            "total_size": actual_size,
            "checksum": file_checksum,
        }

    async def _calculate_file_checksum(self, file_path: Path) -> str:
        """
        Calculate SHA-256 checksum of a file.

        Args:
            file_path: Path to the file

        Returns:
            Hexadecimal checksum string
        """
        sha256_hash = hashlib.sha256()

        def _hash_file() -> str:
            with open(file_path, "rb") as f:
                # Read in 64KB chunks to handle large files efficiently
                while chunk := f.read(64 * 1024):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()

        # Run in executor to avoid blocking the event loop
        return await asyncio.get_event_loop().run_in_executor(None, _hash_file)

    def cleanup_temp_files(self, segments: list[DownloadSegment]) -> None:
        """
        Clean up temporary files for completed segments and memory-mapped resources.

        Args:
            segments: List of download segments to clean up
        """
        logger.info(f"Cleaning up temporary files for {len(segments)} segments")

        # Close memory-mapped writer if open
        if self._mmap_writer:
            try:
                self._mmap_writer.close()
                logger.debug("Closed memory-mapped writer")
            except Exception as e:
                logger.warning(f"Error closing memory-mapped writer: {e}")
            finally:
                self._mmap_writer = None

        cleaned_count = 0
        error_count = 0

        for segment in segments:
            try:
                if segment.temp_file_path.exists():
                    segment.temp_file_path.unlink()
                    cleaned_count += 1
                    logger.debug(f"Removed temp file: {segment.temp_file_path}")
            except Exception as e:
                error_count += 1
                logger.warning(
                    f"Failed to remove temp file {segment.temp_file_path}: {e}"
                )

        # Clean up empty temp directories
        try:
            if self.temp_dir.exists() and not any(self.temp_dir.iterdir()):
                self.temp_dir.rmdir()
                logger.debug(f"Removed empty temp directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to remove temp directory {self.temp_dir}: {e}")

        logger.info(
            f"Cleanup completed: {cleaned_count} files removed, {error_count} errors"
        )

    def get_performance_stats(self) -> dict[str, Any]:
        """
        Get performance statistics for the segment merger.

        Returns:
            Dictionary with performance statistics
        """
        stats = {
            "use_memory_mapping": self.use_memory_mapping,
            "total_file_size": self.total_file_size,
            "target_path": str(self.target_path),
            "temp_dir": str(self.temp_dir),
            "merge_info": {
                "total_segments": self._merge_info.total_segments,
                "merged_segments": self._merge_info.merged_segments,
                "bytes_merged": self._merge_info.bytes_merged,
                "total_bytes": self._merge_info.total_bytes,
                "current_segment": self._merge_info.current_segment,
            },
        }

        # Add memory-mapped writer stats if available
        if self._mmap_writer:
            stats["mmap_writer"] = self._mmap_writer.get_stats()

        # Add file system information
        if self.target_path.exists():
            stat = self.target_path.stat()
            stats["target_file"] = {
                "exists": True,
                "size": stat.st_size,
                "modified_time": stat.st_mtime,
            }
        else:
            stats["target_file"] = {"exists": False}

        return stats

    def __del__(self) -> None:
        """Cleanup when merger is destroyed."""
        if self._mmap_writer:
            logger.warning("SegmentMerger destroyed without proper cleanup")
            try:
                self._mmap_writer.close()
            except Exception:
                pass

    def save_resume_data(self, task_id: str, segments: list[DownloadSegment]) -> None:
        """
        Save resume data for interrupted downloads.

        Args:
            task_id: Download task identifier
            segments: List of download segments to save
        """
        logger.info(f"Saving resume data for task {task_id}")

        try:
            resume_data = {
                "task_id": task_id,
                "target_path": str(self.target_path),
                "temp_dir": str(self.temp_dir),
                "timestamp": time.time(),
                "segments": [
                    {
                        "id": segment.id,
                        "url": segment.url,
                        "start_byte": segment.start_byte,
                        "end_byte": segment.end_byte,
                        "temp_file_path": str(segment.temp_file_path),
                        "status": segment.status.value,
                        "downloaded_bytes": segment.downloaded_bytes,
                        "retry_count": segment.retry_count,
                        "error_message": segment.error_message,
                    }
                    for segment in segments
                ],
            }

            # Write resume data atomically
            temp_resume_path = self.resume_data_path.with_suffix(".tmp")

            # Ensure the directory exists
            temp_resume_path.parent.mkdir(parents=True, exist_ok=True)

            with open(temp_resume_path, "w", encoding="utf-8") as f:
                json.dump(resume_data, f, indent=2)

            temp_resume_path.replace(self.resume_data_path)

            logger.debug(f"Resume data saved to {self.resume_data_path}")

        except Exception as e:
            logger.error(f"Failed to save resume data: {e}")
            raise

    def load_resume_data(self, task_id: str) -> list[DownloadSegment] | None:
        """
        Load resume data for interrupted downloads.

        Args:
            task_id: Download task identifier

        Returns:
            List of download segments or None if no resume data found

        Raises:
            ValueError: If resume data is invalid or corrupted
        """
        if not self.resume_data_path.exists():
            logger.debug(f"No resume data found for task {task_id}")
            return None

        try:
            with open(self.resume_data_path, encoding="utf-8") as f:
                resume_data = json.load(f)

            # Validate resume data
            if resume_data.get("task_id") != task_id:
                logger.warning(
                    f"Resume data task ID mismatch: expected {task_id}, "
                    f"got {resume_data.get('task_id')}"
                )
                return None

            # Check if resume data is not too old (24 hours)
            timestamp = resume_data.get("timestamp", 0)
            if time.time() - timestamp > 24 * 3600:
                logger.warning("Resume data is too old, ignoring")
                return None

            # Reconstruct segments
            segments = []
            for segment_data in resume_data.get("segments", []):
                try:
                    segment = DownloadSegment(
                        id=segment_data["id"],
                        task_id=task_id,
                        url=segment_data["url"],
                        start_byte=segment_data["start_byte"],
                        end_byte=segment_data["end_byte"],
                        temp_file_path=Path(segment_data["temp_file_path"]),
                        status=SegmentStatus(segment_data["status"]),
                        downloaded_bytes=segment_data["downloaded_bytes"],
                        retry_count=segment_data["retry_count"],
                        error_message=segment_data.get("error_message"),
                    )

                    # Verify temp file still exists for incomplete segments
                    if (
                        segment.status
                        in (
                            SegmentStatus.PENDING,
                            SegmentStatus.DOWNLOADING,
                            SegmentStatus.PAUSED,
                        )
                        and segment.downloaded_bytes > 0
                        and not segment.temp_file_path.exists()
                    ):
                        logger.warning(
                            f"Temp file missing for segment {segment.id}, "
                            f"resetting to pending"
                        )
                        segment.status = SegmentStatus.PENDING
                        segment.downloaded_bytes = 0

                    segments.append(segment)

                except Exception as e:
                    logger.warning(f"Failed to reconstruct segment: {e}")
                    continue

            if segments:
                logger.info(f"Loaded resume data for {len(segments)} segments")
                return segments
            else:
                logger.warning("No valid segments found in resume data")
                return None

        except Exception as e:
            logger.error(f"Failed to load resume data: {e}")
            raise ValueError(f"Invalid resume data: {e}") from e

    def get_merge_info(self) -> SegmentMergeInfo:
        """
        Get current merge progress information.

        Returns:
            Current merge information
        """
        return self._merge_info.model_copy()

    def update_merge_info(self, total_segments: int, total_bytes: int) -> None:
        """
        Update merge tracking information.

        Args:
            total_segments: Total number of segments
            total_bytes: Total bytes to merge
        """
        self._merge_info.total_segments = total_segments
        self._merge_info.total_bytes = total_bytes

        logger.debug(
            f"Updated merge info: {total_segments} segments, {total_bytes} bytes"
        )

    def reset_merge_info(self) -> None:
        """Reset merge tracking information."""
        self._merge_info = SegmentMergeInfo(
            total_segments=0, merged_segments=0, bytes_merged=0, total_bytes=0
        )
        logger.debug("Reset merge tracking information")
