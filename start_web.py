#!/usr/bin/env python3
"""Launch the VFX Pipeline web interface.

Usage:
    ./start_web.py              # Start server and open browser
    ./start_web.py --no-browser # Start server only
    ./start_web.py --port 8080  # Use custom port
"""

import argparse
import os
import sys
import webbrowser
from pathlib import Path

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from log_manager import LogCapture


def _add_tools_to_path() -> None:
    """Add .vfx_pipeline/tools/*/bin directories to PATH."""
    tools_dir = Path(__file__).parent / ".vfx_pipeline" / "tools"
    if not tools_dir.exists():
        return
    extra = []
    for tool_dir in sorted(tools_dir.iterdir()):
        bin_dir = tool_dir / "bin"
        if bin_dir.is_dir():
            extra.append(str(bin_dir))
    if extra:
        os.environ["PATH"] = os.pathsep.join(extra) + os.pathsep + os.environ.get("PATH", "")


def main():
    parser = argparse.ArgumentParser(description="Launch VFX Pipeline web interface")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    parser.add_argument("--port", type=int, default=5000, help="Server port (default: 5000)")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    args = parser.parse_args()

    _add_tools_to_path()

    # Check conda environment first
    try:
        from env_config import require_conda_env
        require_conda_env()  # Exits with helpful message if wrong env
    except ImportError:
        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        if not conda_prefix or Path(conda_prefix).name != "vfx-pipeline":
            print("""
ERROR: Wrong conda environment.

Please activate the VFX Pipeline environment first:
    conda activate <vfx-pipeline prefix>

Then re-run this script.
""")
            sys.exit(1)

    # Check for required dependencies
    try:
        import uvicorn
    except ImportError:
        print("""
ERROR: Web GUI dependencies not installed.

Install them with:
    pip install fastapi uvicorn python-multipart websockets jinja2

Or run the install wizard:
    python scripts/install_wizard.py
""")
        sys.exit(1)

    url = f"http://{args.host}:{args.port}"

    print(f"""
╔════════════════════════════════════════════════════════╗
║           VFX Pipeline Web Interface                   ║
╠════════════════════════════════════════════════════════╣
║                                                        ║
║   Server running at: {url:<29} ║
║                                                        ║
║   Press Ctrl+C to stop                                 ║
║                                                        ║
╚════════════════════════════════════════════════════════╝
""")

    # Open browser (unless disabled)
    if not args.no_browser:
        webbrowser.open(url)

    # Start server
    uvicorn.run("web.server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    with LogCapture():
        main()
