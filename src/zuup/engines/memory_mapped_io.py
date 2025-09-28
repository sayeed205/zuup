"""Memory-mapped file operations for efficient handling of large downloads."""

from __future__ import annotations

import logging
import mmap
import os
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)


class MemoryMappedFile:
    """Memory-mapped file wrapper for efficient large file operations."""

    def __init__(
        self,
        file_path: Path,
        size: int,
        mode: str = "r+b",
        create_if_missing: bool = True,
    ) -> None:
        """
        Initialize memory-mapped file.

        Args:
            file_path: Path to the file
            size: Expected file size
            mode: File open mode
            create_if_missing: Create file if it doesn't exist

        Raises:
            OSError: If file operations fail
        """
        self.file_path = file_path
        self.size = size
        self.mode = mode

        self._file_handle: Optional[Any] = None
        self._mmap_handle: Optional[mmap.mmap] = None
        self._is_open = False

        # Create file if needed
        if create_if_missing and not file_path.exists():
            self._create_sparse_file()

        logger.debug(f"Initialized MemoryMappedFile for {file_path} ({size} bytes)")

    def _create_sparse_file(self) -> None:
        """Create a sparse file of the specified size."""
        try:
            # Ensure parent directory exists
            self.file_path.parent.mkdir(parents=True, exist_ok=True)

            # Create sparse file
            with open(self.file_path, "wb") as f:
                f.seek(self.size - 1)
                f.write(b"\0")

            logger.debug(f"Created sparse file {self.file_path} ({self.size} bytes)")

        except Exception as e:
            logger.error(f"Failed to create sparse file {self.file_path}: {e}")
            raise

    def open(self) -> None:
        """Open the file and create memory mapping."""
        if self._is_open:
            return

        try:
            # Open file
            self._file_handle = open(self.file_path, self.mode)

            # Create memory mapping
            access = mmap.ACCESS_WRITE if "w" in self.mode or "+" in self.mode else mmap.ACCESS_READ
            self._mmap_handle = mmap.mmap(self._file_handle.fileno(), 0, access=access)

            self._is_open = True
            logger.debug(f"Opened memory-mapped file {self.file_path}")

        except Exception as e:
            logger.error(f"Failed to open memory-mapped file {self.file_path}: {e}")
            self._cleanup()
            raise

    def close(self) -> None:
        """Close memory mapping and file."""
        if not self._is_open:
            return

        self._cleanup()
        logger.debug(f"Closed memory-mapped file {self.file_path}")

    def _cleanup(self) -> None:
        """Internal cleanup method."""
        if self._mmap_handle:
            try:
                self._mmap_handle.close()
            except Exception as e:
                logger.warning(f"Error closing mmap handle: {e}")
            finally:
                self._mmap_handle = None

        if self._file_handle:
            try:
                self._file_handle.close()
            except Exception as e:
                logger.warning(f"Error closing file handle: {e}")
            finally:
                self._file_handle = None

        self._is_open = False

    def write_segment(self, offset: int, data: bytes) -> int:
        """
        Write data to a specific offset in the file.

        Args:
            offset: Byte offset to write at
            data: Data to write

        Returns:
            Number of bytes written

        Raises:
            ValueError: If file is not open or offset is invalid
            OSError: If write operation fails
        """
        if not self._is_open or not self._mmap_handle:
            raise ValueError("File is not open")

        if offset < 0 or offset >= self.size:
            raise ValueError(f"Invalid offset {offset} for file size {self.size}")

        if offset + len(data) > self.size:
            raise ValueError(f"Write would exceed file size: {offset + len(data)} > {self.size}")

        try:
            # Seek to offset and write data
            self._mmap_handle.seek(offset)
            bytes_written = self._mmap_handle.write(data)

            # Flush to ensure data is written
            self._mmap_handle.flush()

            logger.debug(f"Wrote {bytes_written} bytes at offset {offset}")
            return bytes_written

        except Exception as e:
            logger.error(f"Failed to write segment at offset {offset}: {e}")
            raise

    def read_segment(self, offset: int, length: int) -> bytes:
        """
        Read data from a specific offset in the file.

        Args:
            offset: Byte offset to read from
            length: Number of bytes to read

        Returns:
            Data read from file

        Raises:
            ValueError: If file is not open or parameters are invalid
            OSError: If read operation fails
        """
        if not self._is_open or not self._mmap_handle:
            raise ValueError("File is not open")

        if offset < 0 or offset >= self.size:
            raise ValueError(f"Invalid offset {offset} for file size {self.size}")

        if offset + length > self.size:
            length = self.size - offset

        try:
            # Seek to offset and read data
            self._mmap_handle.seek(offset)
            data = self._mmap_handle.read(length)

            logger.debug(f"Read {len(data)} bytes from offset {offset}")
            return data

        except Exception as e:
            logger.error(f"Failed to read segment at offset {offset}: {e}")
            raise

    def sync(self) -> None:
        """Synchronize memory-mapped data to disk."""
        if not self._is_open or not self._mmap_handle:
            return

        try:
            self._mmap_handle.flush()
            logger.debug(f"Synchronized {self.file_path} to disk")
        except Exception as e:
            logger.warning(f"Failed to sync {self.file_path}: {e}")

    def get_stats(self) -> dict[str, Any]:
        """
        Get file statistics.

        Returns:
            Dictionary with file statistics
        """
        stats = {
            "file_path": str(self.file_path),
            "size": self.size,
            "is_open": self._is_open,
            "exists": self.file_path.exists(),
        }

        if self.file_path.exists():
            stat = self.file_path.stat()
            stats.update({
                "actual_size": stat.st_size,
                "modified_time": stat.st_mtime,
                "is_sparse": stat.st_blocks * 512 < stat.st_size,  # Approximate sparse detection
            })

        return stats

    def __enter__(self) -> MemoryMappedFile:
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()

    def __del__(self) -> None:
        """Cleanup when object is destroyed."""
        if self._is_open:
            logger.warning(f"MemoryMappedFile {self.file_path} destroyed without proper cleanup")
            self._cleanup()


class MemoryMappedSegmentWriter:
    """Optimized writer for download segments using memory-mapped files."""

    def __init__(
        self,
        target_file: Path,
        total_size: int,
        use_mmap_threshold: int = 50 * 1024 * 1024,  # 50MB
    ) -> None:
        """
        Initialize segment writer.

        Args:
            target_file: Final target file path
            total_size: Total expected file size
            use_mmap_threshold: Minimum file size to use memory mapping
        """
        self.target_file = target_file
        self.total_size = total_size
        self.use_mmap_threshold = use_mmap_threshold

        self._use_mmap = total_size >= use_mmap_threshold
        self._mmap_file: Optional[MemoryMappedFile] = None
        self._regular_file: Optional[Any] = None

        # Track written segments for integrity checking
        self._written_segments: set[tuple[int, int]] = set()
        self._total_written = 0

        logger.info(
            f"Initialized SegmentWriter for {target_file} "
            f"({total_size} bytes, mmap={'enabled' if self._use_mmap else 'disabled'})"
        )

    def open(self) -> None:
        """Open the target file for writing."""
        try:
            if self._use_mmap:
                self._mmap_file = MemoryMappedFile(
                    self.target_file, self.total_size, mode="r+b", create_if_missing=True
                )
                self._mmap_file.open()
                logger.debug(f"Opened memory-mapped file for {self.target_file}")
            else:
                # Ensure parent directory exists
                self.target_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Create file if it doesn't exist
                if not self.target_file.exists():
                    with open(self.target_file, "wb") as f:
                        f.seek(self.total_size - 1)
                        f.write(b"\0")
                
                self._regular_file = open(self.target_file, "r+b")
                logger.debug(f"Opened regular file for {self.target_file}")

        except Exception as e:
            logger.error(f"Failed to open target file {self.target_file}: {e}")
            raise

    def write_segment(self, start_offset: int, data: bytes) -> int:
        """
        Write a segment to the target file.

        Args:
            start_offset: Starting byte offset
            data: Segment data

        Returns:
            Number of bytes written

        Raises:
            ValueError: If writer is not open or parameters are invalid
            OSError: If write operation fails
        """
        if not data:
            return 0

        end_offset = start_offset + len(data)

        if start_offset < 0 or end_offset > self.total_size:
            raise ValueError(
                f"Invalid segment range {start_offset}-{end_offset} for file size {self.total_size}"
            )

        try:
            if self._use_mmap and self._mmap_file:
                bytes_written = self._mmap_file.write_segment(start_offset, data)
            elif self._regular_file:
                self._regular_file.seek(start_offset)
                bytes_written = self._regular_file.write(data)
                self._regular_file.flush()
            else:
                raise ValueError("Writer is not open")

            # Track written segment
            self._written_segments.add((start_offset, end_offset))
            self._total_written += bytes_written

            logger.debug(
                f"Wrote segment {start_offset}-{end_offset} ({bytes_written} bytes)"
            )
            return bytes_written

        except Exception as e:
            logger.error(f"Failed to write segment {start_offset}-{end_offset}: {e}")
            raise

    def sync(self) -> None:
        """Synchronize data to disk."""
        try:
            if self._use_mmap and self._mmap_file:
                self._mmap_file.sync()
            elif self._regular_file:
                self._regular_file.flush()
                os.fsync(self._regular_file.fileno())

            logger.debug(f"Synchronized {self.target_file} to disk")

        except Exception as e:
            logger.warning(f"Failed to sync {self.target_file}: {e}")

    def verify_integrity(self) -> dict[str, Any]:
        """
        Verify file integrity by checking for gaps.

        Returns:
            Dictionary with integrity information
        """
        # Sort segments by start offset
        sorted_segments = sorted(self._written_segments)

        gaps = []
        covered_bytes = 0
        last_end = 0

        for start, end in sorted_segments:
            if start > last_end:
                # Gap found
                gaps.append((last_end, start))

            covered_bytes += end - start
            last_end = max(last_end, end)

        # Check if file is complete
        is_complete = covered_bytes == self.total_size and not gaps

        integrity_info = {
            "is_complete": is_complete,
            "total_size": self.total_size,
            "covered_bytes": covered_bytes,
            "coverage_percentage": (covered_bytes / self.total_size * 100) if self.total_size > 0 else 0,
            "segments_written": len(self._written_segments),
            "gaps": gaps,
            "gap_count": len(gaps),
            "total_gap_size": sum(end - start for start, end in gaps),
        }

        logger.debug(f"Integrity check: {integrity_info}")
        return integrity_info

    def close(self) -> None:
        """Close the target file."""
        try:
            if self._mmap_file:
                self._mmap_file.close()
                self._mmap_file = None

            if self._regular_file:
                self._regular_file.close()
                self._regular_file = None

            logger.debug(f"Closed target file {self.target_file}")

        except Exception as e:
            logger.warning(f"Error closing target file: {e}")

    def get_stats(self) -> dict[str, Any]:
        """
        Get writer statistics.

        Returns:
            Dictionary with writer statistics
        """
        stats = {
            "target_file": str(self.target_file),
            "total_size": self.total_size,
            "use_mmap": self._use_mmap,
            "total_written": self._total_written,
            "segments_count": len(self._written_segments),
        }

        if self._mmap_file:
            stats["mmap_stats"] = self._mmap_file.get_stats()

        # Add integrity information
        stats.update(self.verify_integrity())

        return stats

    def __enter__(self) -> MemoryMappedSegmentWriter:
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()

    def __del__(self) -> None:
        """Cleanup when object is destroyed."""
        if self._mmap_file or self._regular_file:
            logger.warning(f"SegmentWriter for {self.target_file} destroyed without proper cleanup")
            self.close()


def should_use_memory_mapping(file_size: int, threshold: int = 50 * 1024 * 1024) -> bool:
    """
    Determine if memory mapping should be used for a file.

    Args:
        file_size: Size of the file in bytes
        threshold: Minimum size threshold for using memory mapping

    Returns:
        True if memory mapping should be used
    """
    return file_size >= threshold


def get_optimal_mmap_threshold() -> int:
    """
    Get optimal memory mapping threshold based on system resources.

    Returns:
        Optimal threshold in bytes
    """
    try:
        # Try to get available memory
        import psutil
        available_memory = psutil.virtual_memory().available
        
        # Use 1% of available memory as threshold, with min 50MB and max 500MB
        threshold = max(50 * 1024 * 1024, min(500 * 1024 * 1024, available_memory // 100))
        
        logger.debug(f"Calculated optimal mmap threshold: {threshold} bytes")
        return threshold
        
    except ImportError:
        # psutil not available, use default
        default_threshold = 50 * 1024 * 1024
        logger.debug(f"Using default mmap threshold: {default_threshold} bytes")
        return default_threshold
    except Exception as e:
        logger.warning(f"Error calculating optimal mmap threshold: {e}")
        return 50 * 1024 * 1024