#!/usr/bin/env python3
"""Launch the VFX Pipeline web interface.

Usage:
    ./start_web.py              # Start server and open browser
    ./start_web.py --no-browser # Start server only
    ./start_web.py --port 8080  # Use custom port
"""

import argparse
import sys
import webbrowser
from pathlib import Path

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

def main():
    parser = argparse.ArgumentParser(description="Launch VFX Pipeline web interface")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    parser.add_argument("--port", type=int, default=5000, help="Server port (default: 5000)")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    args = parser.parse_args()

    # Check for required dependencies
    try:
        import uvicorn
    except ImportError:
        print("""
ERROR: Web GUI dependencies not installed.

Install them with:
    pip install fastapi uvicorn python-multipart websockets

Or run the janitor to install all updates:
    python scripts/janitor.py -u
""")
        sys.exit(1)

    # Check conda environment
    try:
        from env_config import check_conda_env_or_warn
        check_conda_env_or_warn()
    except ImportError:
        pass  # Not critical, continue anyway

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
    main()
