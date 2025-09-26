"""FastAPI server implementation."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

if TYPE_CHECKING:
    from ..config.manager import ConfigManager
    from ..core.task_manager import TaskManager

logger = logging.getLogger(__name__)


class APIServer:
    """FastAPI server for REST API endpoints."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8080,
        task_manager: TaskManager | None = None,
        config_manager: ConfigManager | None = None,
    ) -> None:
        """
        Initialize API server with dependency injection.

        Args:
            host: Server host address
            port: Server port number
            task_manager: Task manager instance
            config_manager: Configuration manager instance
        """
        self.host = host
        self.port = port
        self.task_manager = task_manager
        self.config_manager = config_manager

        # FastAPI app
        self.app = FastAPI(
            title="Zuup Download Manager API",
            description="REST API for unified download management",
            version="1.0.0",
        )

        # Server state
        self._server: uvicorn.Server | None = None
        self._server_task: asyncio.Task[None] | None = None

        # Setup middleware and routes
        self._setup_middleware()
        self._setup_routes()

        logger.info(f"APIServer initialized on {host}:{port}")

    def _setup_middleware(self) -> None:
        """Setup FastAPI middleware."""
        # CORS middleware for browser extension support
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure based on security requirements
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self) -> None:
        """Setup API routes."""

        @self.app.get("/")
        async def root():
            """Root endpoint."""
            return {"message": "Zuup Download Manager API", "version": "1.0.0"}

        @self.app.get("/api/v1/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "components": {
                    "task_manager": self.task_manager is not None,
                    "config_manager": self.config_manager is not None,
                }
            }

        @self.app.get("/api/v1/status")
        async def get_status():
            """Get application status."""
            if not self.task_manager:
                return {"error": "Task manager not available"}

            return {
                "status": "running",
                "stats": self.task_manager.get_manager_stats(),
            }

        @self.app.get("/api/v1/tasks")
        async def list_tasks():
            """List all download tasks."""
            if not self.task_manager:
                return {"error": "Task manager not available"}

            tasks = self.task_manager.list_tasks()
            return {
                "tasks": [task.model_dump() for task in tasks],
                "total": len(tasks),
            }

        @self.app.get("/api/v1/tasks/{task_id}")
        async def get_task(task_id: str):
            """Get specific task by ID."""
            if not self.task_manager:
                return {"error": "Task manager not available"}

            try:
                task = self.task_manager.get_task(task_id)
                return task.model_dump()
            except Exception as e:
                return {"error": str(e)}

        @self.app.post("/api/v1/tasks")
        async def create_task(task_data: dict):
            """Create a new download task."""
            if not self.task_manager:
                return {"error": "Task manager not available"}

            try:
                task = await self.task_manager.create_task(
                    url=task_data["url"],
                    destination=task_data["destination"],
                    filename=task_data.get("filename"),
                    auto_start=task_data.get("auto_start", True),
                )
                return task.model_dump()
            except Exception as e:
                return {"error": str(e)}

        @self.app.post("/api/v1/tasks/{task_id}/start")
        async def start_task(task_id: str):
            """Start a download task."""
            if not self.task_manager:
                return {"error": "Task manager not available"}

            try:
                await self.task_manager.start_task(task_id)
                return {"message": f"Task {task_id} started"}
            except Exception as e:
                return {"error": str(e)}

        @self.app.post("/api/v1/tasks/{task_id}/pause")
        async def pause_task(task_id: str):
            """Pause a download task."""
            if not self.task_manager:
                return {"error": "Task manager not available"}

            try:
                await self.task_manager.pause_task(task_id)
                return {"message": f"Task {task_id} paused"}
            except Exception as e:
                return {"error": str(e)}

        @self.app.post("/api/v1/tasks/{task_id}/resume")
        async def resume_task(task_id: str):
            """Resume a download task."""
            if not self.task_manager:
                return {"error": "Task manager not available"}

            try:
                await self.task_manager.resume_task(task_id)
                return {"message": f"Task {task_id} resumed"}
            except Exception as e:
                return {"error": str(e)}

        @self.app.post("/api/v1/tasks/{task_id}/cancel")
        async def cancel_task(task_id: str):
            """Cancel a download task."""
            if not self.task_manager:
                return {"error": "Task manager not available"}

            try:
                await self.task_manager.cancel_task(task_id)
                return {"message": f"Task {task_id} cancelled"}
            except Exception as e:
                return {"error": str(e)}

        @self.app.delete("/api/v1/tasks/{task_id}")
        async def delete_task(task_id: str):
            """Delete a download task."""
            if not self.task_manager:
                return {"error": "Task manager not available"}

            try:
                deleted = await self.task_manager.delete_task(task_id)
                if deleted:
                    return {"message": f"Task {task_id} deleted"}
                else:
                    return {"error": f"Task {task_id} not found"}
            except Exception as e:
                return {"error": str(e)}

    async def start(self) -> None:
        """Start the API server."""
        logger.info(f"Starting API server on {self.host}:{self.port}")

        try:
            # Create uvicorn server
            config = uvicorn.Config(
                app=self.app,
                host=self.host,
                port=self.port,
                log_level="info",
                access_log=True,
            )
            self._server = uvicorn.Server(config)

            # Start server
            await self._server.serve()

        except Exception as e:
            logger.error(f"Failed to start API server: {e}")
            raise

    async def stop(self) -> None:
        """Stop the API server."""
        logger.info("Stopping API server")

        if self._server:
            self._server.should_exit = True

        if self._server_task and not self._server_task.done():
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass

        logger.info("API server stopped")
