#!/usr/bin/env python3
"""Launch the VFX Pipeline web GUI with proper startup sequence.

This script:
1. Starts the backend server as a subprocess
2. Waits for the server to be ready (polls /health endpoint)
3. Opens the browser once the server is responsive
4. Keeps running until the user presses Ctrl+C

Usage:
    python scripts/launch_web_gui.py
    python scripts/launch_web_gui.py --port 8080
"""

import argparse
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

STARTUP_TIMEOUT = 30
POLL_INTERVAL = 0.5


def wait_for_server(url: str, timeout: float = STARTUP_TIMEOUT) -> bool:
    """Wait for the server to become responsive.

    Args:
        url: Base URL to check (will append /health)
        timeout: Maximum seconds to wait

    Returns:
        True if server is ready, False if timeout
    """
    health_url = f"{url}/health"
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            with urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    return True
        except (URLError, OSError):
            pass
        time.sleep(POLL_INTERVAL)

    return False


def main():
    parser = argparse.ArgumentParser(description="Launch VFX Pipeline Web GUI")
    parser.add_argument("--port", type=int, default=5000, help="Server port (default: 5000)")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    repo_root = Path(__file__).parent.parent

    print("""
 ____  _           _      ____             _
/ ___|| |__   ___ | |_   / ___| ___  _ __ | |__   ___ _ __
\___ \| '_ \ / _ \| __| | |  _ / _ \| '_ \| '_ \ / _ \ '__|
 ___) | | | | (_) | |_  | |_| | (_) | |_) | | | |  __/ |
|____/|_| |_|\___/ \__|  \____|\___/| .__/|_| |_|\___|_|
                                    |_|
    """)
    print(f"Starting VFX Pipeline Web GUI...")
    print(f"Server URL: {url}")
    print()

    start_web_script = repo_root / "start_web.py"
    server_process = subprocess.Popen(
        [sys.executable, str(start_web_script), "--no-browser", "--port", str(args.port), "--host", args.host],
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    print("Waiting for server to start...")

    if wait_for_server(url):
        print(f"Server is ready!")
        print()

        if not args.no_browser:
            print(f"Opening browser: {url}")
            webbrowser.open(url)

        print()
        print("Press Ctrl+C to stop the server")
        print("-" * 40)
        print()

        try:
            for line in server_process.stdout:
                print(line, end="")
        except KeyboardInterrupt:
            print()
            print("Shutting down...")
    else:
        print(f"ERROR: Server failed to start within {STARTUP_TIMEOUT} seconds")
        server_process.terminate()
        sys.exit(1)

    server_process.terminate()
    try:
        server_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server_process.kill()

    print("Server stopped.")


if __name__ == "__main__":
    main()
