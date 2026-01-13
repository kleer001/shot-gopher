"""WebSocket handlers for real-time progress updates."""

import asyncio
import json
from typing import Dict, Optional, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()

# Store active WebSocket connections per project
connections: Dict[str, Set[WebSocket]] = {}

# Store for progress updates (set by pipeline_runner)
progress_updates: Dict[str, dict] = {}

# Store reference to main event loop for thread-safe updates
_main_loop: Optional[asyncio.AbstractEventLoop] = None


def set_main_loop(loop: asyncio.AbstractEventLoop):
    """Store reference to the main event loop for thread-safe updates."""
    global _main_loop
    _main_loop = loop


class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}

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

            # Clean up disconnected
            for conn in disconnected:
                self.active_connections[project_id].discard(conn)

    async def broadcast(self, message: dict):
        """Broadcast a message to all connections."""
        for project_id in list(self.active_connections.keys()):
            await self.send_to_project(project_id, message)


manager = ConnectionManager()


async def send_progress(project_id: str, data: dict):
    """Send progress update to all connected clients for a project.

    This is called from pipeline_runner to push updates.
    """
    message = {
        "type": "progress",
        "project_id": project_id,
        **data,
    }
    await manager.send_to_project(project_id, message)


async def send_stage_complete(project_id: str, stage: str, stage_index: int, total_stages: int):
    """Send stage completion notification."""
    message = {
        "type": "stage_complete",
        "project_id": project_id,
        "stage": stage,
        "stage_index": stage_index,
        "total_stages": total_stages,
    }
    await manager.send_to_project(project_id, message)


async def send_pipeline_complete(project_id: str, success: bool = True, error: str = None):
    """Send pipeline completion notification."""
    message = {
        "type": "pipeline_complete",
        "project_id": project_id,
        "success": success,
        "error": error,
    }
    await manager.send_to_project(project_id, message)


async def send_error(project_id: str, error: str):
    """Send error notification."""
    message = {
        "type": "error",
        "project_id": project_id,
        "error": error,
    }
    await manager.send_to_project(project_id, message)


async def send_log(project_id: str, line: str):
    """Send log line to connected clients."""
    message = {
        "type": "log",
        "project_id": project_id,
        "line": line,
    }
    await manager.send_to_project(project_id, message)


@router.websocket("/ws/{project_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: str):
    """WebSocket endpoint for project progress updates."""
    await manager.connect(websocket, project_id)

    try:
        # Send initial status if available
        if project_id in progress_updates:
            await websocket.send_json({
                "type": "progress",
                "project_id": project_id,
                **progress_updates[project_id],
            })

        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for messages (ping/pong or commands)
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0  # 30 second timeout for ping
                )

                # Handle incoming messages
                try:
                    message = json.loads(data)
                    if message.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                except json.JSONDecodeError:
                    pass

            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                try:
                    await websocket.send_json({"type": "ping"})
                except Exception:
                    break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket, project_id)


def update_progress(project_id: str, data: dict):
    """Update stored progress (called from pipeline_runner synchronously).

    This stores the update for new connections and triggers async broadcast.
    Thread-safe: can be called from background threads.
    """
    progress_updates[project_id] = data

    # Schedule async broadcast using thread-safe method
    if _main_loop is not None:
        try:
            # run_coroutine_threadsafe is safe to call from any thread
            asyncio.run_coroutine_threadsafe(send_progress(project_id, data), _main_loop)
        except RuntimeError:
            # Loop closed or other issue
            pass
