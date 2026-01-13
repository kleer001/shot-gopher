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

from .api import router as api_router
from .websocket import router as ws_router, set_main_loop


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for app startup/shutdown."""
    # Capture the main event loop for thread-safe WebSocket updates
    set_main_loop(asyncio.get_running_loop())
    yield


# Create FastAPI app
app = FastAPI(
    title="VFX Pipeline",
    description="Web interface for automated VFX processing",
    version="1.0.0",
    lifespan=lifespan,
)

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
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
