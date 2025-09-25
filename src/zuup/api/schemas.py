"""API response schemas."""


from pydantic import BaseModel

from ..storage.models import EngineType, TaskStatus


class TaskResponse(BaseModel):
    """Response schema for task information."""

    id: str
    url: str
    filename: str | None
    status: TaskStatus
    engine_type: EngineType
    progress_percentage: float | None
    download_speed: float
    eta_seconds: float | None
    error_message: str | None


class ProgressResponse(BaseModel):
    """Response schema for progress information."""

    task_id: str
    downloaded_bytes: int
    total_bytes: int | None
    progress_percentage: float | None
    download_speed: float
    upload_speed: float | None
    eta_seconds: float | None
    status: TaskStatus

    # Torrent-specific fields
    peers_connected: int | None = None
    seeds_connected: int | None = None
    ratio: float | None = None
    is_seeding: bool | None = None


class CreateTaskRequest(BaseModel):
    """Request schema for creating a new task."""

    url: str
    destination: str | None = None
    filename: str | None = None
    engine_type: EngineType | None = None  # Auto-detect if not provided


class UpdateTaskRequest(BaseModel):
    """Request schema for updating task configuration."""

    max_connections: int | None = None
    download_speed_limit: int | None = None
    upload_speed_limit: int | None = None
