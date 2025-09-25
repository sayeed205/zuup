"""FastAPI server implementation."""

import logging

logger = logging.getLogger(__name__)


class APIServer:
    """FastAPI server for REST API endpoints."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8080) -> None:
        """
        Initialize API server.

        Args:
            host: Server host address
            port: Server port number
        """
        self.host = host
        self.port = port
        logger.info(f"APIServer initialized on {host}:{port}")

    async def start(self) -> None:
        """Start the API server."""
        # Implementation will be added in task 7
        logger.info(f"Starting API server on {self.host}:{self.port}")
        raise NotImplementedError("API server will be implemented in task 7")

    async def stop(self) -> None:
        """Stop the API server."""
        # Implementation will be added in task 7
        logger.info("Stopping API server")
        raise NotImplementedError("API server will be implemented in task 7")
