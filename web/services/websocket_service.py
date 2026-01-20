"""WebSocket service for real-time updates."""

import asyncio
from typing import Dict, Set
from fastapi import WebSocket

from web.models.dto import ProgressUpdate


class WebSocketService:
    """
    Service for WebSocket connection management and broadcasting.

    Responsibilities:
    - Manage WebSocket connections per project
    - Broadcast progress updates to connected clients
    - Handle connection lifecycle (connect, disconnect, cleanup)

    Does NOT:
    - Handle HTTP/WebSocket endpoint routing (that's API layer)
    - Parse pipeline output (that's pipeline_runner)
    """

    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.progress_cache: Dict[str, dict] = {}

    async def connect(self, websocket: WebSocket, project_id: str):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        if project_id not in self.active_connections:
            self.active_connections[project_id] = set()
        self.active_connections[project_id].add(websocket)

    def disconnect(self, websocket: WebSocket, project_id: str):
        """Remove a WebSocket connection."""
        if project_id in self.active_connections:
            self.active_connections[project_id].discard(websocket)
            if not self.active_connections[project_id]:
                del self.active_connections[project_id]

    async def send_to_project(self, project_id: str, message: dict):
        """Send a message to all connections for a project."""
        if project_id in self.active_connections:
            disconnected = set()
            for connection in self.active_connections[project_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    disconnected.add(connection)

            for conn in disconnected:
                self.active_connections[project_id].discard(conn)

    async def broadcast(self, message: dict):
        """Broadcast a message to all connections."""
        for project_id in list(self.active_connections.keys()):
            await self.send_to_project(project_id, message)

    async def send_progress(self, project_id: str, data: dict):
        """Send progress update to all connected clients for a project."""
        message = {
            "type": "progress",
            "project_id": project_id,
            **data,
        }
        self.progress_cache[project_id] = message
        await self.send_to_project(project_id, message)

    async def send_stage_complete(
        self,
        project_id: str,
        stage: str,
        stage_index: int,
        total_stages: int
    ):
        """Send stage completion notification."""
        message = {
            "type": "stage_complete",
            "project_id": project_id,
            "stage": stage,
            "stage_index": stage_index,
            "total_stages": total_stages,
        }
        await self.send_to_project(project_id, message)

    async def send_pipeline_complete(
        self,
        project_id: str,
        success: bool = True,
        error: str = None
    ):
        """Send pipeline completion notification."""
        message = {
            "type": "pipeline_complete",
            "project_id": project_id,
            "success": success,
            "error": error,
        }
        await self.send_to_project(project_id, message)

    async def send_error(self, project_id: str, error: str):
        """Send error notification."""
        message = {
            "type": "error",
            "project_id": project_id,
            "error": error,
        }
        await self.send_to_project(project_id, message)

    async def send_log(self, project_id: str, line: str):
        """Send log line to connected clients."""
        message = {
            "type": "log",
            "project_id": project_id,
            "line": line,
        }
        await self.send_to_project(project_id, message)

    def get_cached_progress(self, project_id: str) -> dict:
        """Get cached progress for a project."""
        return self.progress_cache.get(project_id, {})

    def update_progress_sync(self, project_id: str, data: dict, event_loop: asyncio.AbstractEventLoop):
        """
        Thread-safe progress update (called from background threads).

        Stores progress in cache and schedules async broadcast.
        """
        message = {
            "type": "progress",
            "project_id": project_id,
            **data,
        }
        self.progress_cache[project_id] = message

        try:
            asyncio.run_coroutine_threadsafe(
                self.send_to_project(project_id, message),
                event_loop
            )
        except RuntimeError:
            pass


_websocket_service = None


def get_websocket_service() -> WebSocketService:
    """Get the global WebSocket service instance."""
    global _websocket_service
    if _websocket_service is None:
        _websocket_service = WebSocketService()
    return _websocket_service
