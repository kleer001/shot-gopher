"""FastAPI server for VFX Pipeline web interface."""

import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.openapi.utils import get_openapi

from .api import router as api_router
from .websocket import router as ws_router, set_main_loop
from comfyui_manager import start_comfyui, stop_comfyui


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for app startup/shutdown."""
    set_main_loop(asyncio.get_running_loop())

    comfyui_started = await asyncio.to_thread(start_comfyui)
    if not comfyui_started:
        print("Warning: ComfyUI not available - some stages will fail")

    yield

    await asyncio.to_thread(stop_comfyui)


app = FastAPI(
    title="VFX Ingest Platform API",
    description="REST API for VFX pipeline project management and execution",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)


def custom_openapi():
    """Custom OpenAPI schema with enhanced documentation."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="VFX Ingest Platform API",
        version="1.0.0",
        description="""
# VFX Ingest Platform API

REST API for managing VFX pipeline projects and executing stages.

## Features
- Create and manage projects
- Upload video files
- Execute pipeline stages (ingest, depth, roto, colmap, mocap, etc.)
- Real-time progress updates via WebSocket
- Video metadata extraction

## Architecture
- **Layered architecture** with proper separation of concerns
- **Repository pattern** for data access abstraction
- **Service layer** for business logic
- **DTOs** for API contracts with validation

## Authentication
Currently no authentication required (local deployment).

## Rate Limiting
No rate limiting (single-user deployment).
        """,
        routes=app.routes,
    )

    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

# Get paths
WEB_DIR = Path(__file__).parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Setup templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Include routers
app.include_router(api_router, prefix="/api")
app.include_router(ws_router)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main web interface."""
    return templates.TemplateResponse("index-dashboard.html", {"request": request})


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
