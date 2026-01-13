"""VFX Pipeline Web Interface.

A browser-based interface for the VFX pipeline with:
- Drag-and-drop or browse video upload
- Stage selection with presets
- Real-time progress monitoring via WebSocket
- Output file browsing

Usage:
    ./start_web.py  # From repo root
"""

from .server import app

__all__ = ['app']
