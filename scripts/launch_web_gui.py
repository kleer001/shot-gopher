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
import os
import select
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

STARTUP_TIMEOUT = 30
POLL_INTERVAL = 0.5


def check_server_health(url: str) -> bool:
    """Check if server health endpoint is responding."""
    health_url = f"{url}/health"
    try:
        with urlopen(health_url, timeout=2) as response:
            return response.status == 200
    except (URLError, OSError):
        return False


def stream_output_nonblocking(process: subprocess.Popen) -> None:
    """Read and print any available output from process without blocking."""
    if sys.platform == "win32":
        return

    if process.stdout is None:
        return

    fd = process.stdout.fileno()
    readable, _, _ = select.select([fd], [], [], 0)
    if readable:
        line = process.stdout.readline()
        if line:
            print(line, end="", flush=True)


def wait_for_server_with_output(
    url: str, process: subprocess.Popen, timeout: float = STARTUP_TIMEOUT
) -> bool:
    """Wait for server while streaming its output."""
    start_time = time.time()

    while time.time() - start_time < timeout:
        stream_output_nonblocking(process)

        if process.poll() is not None:
            print()
            print(f"ERROR: Server process exited with code {process.returncode}")
            remaining = process.stdout.read() if process.stdout else ""
            if remaining:
                print(remaining)
            return False

        if check_server_health(url):
            return True

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

    print(r"""
 ____  _           _      ____             _
/ ___|| |__   ___ | |_   / ___| ___  _ __ | |__   ___ _ __
\___ \| '_ \ / _ \| __| | |  _ / _ \| '_ \| '_ \ / _ \ '__|
 ___) | | | | (_) | |_  | |_| | (_) | |_) | | | |  __/ |
|____/|_| |_|\___/ \__|  \____|\___/| .__/|_| |_|\___|_|
                                    |_|
    """)
    print(f"Starting Shot Gopher...")
    print(f"Server URL: {url}")
    print()

    start_web_script = repo_root / "start_web.py"
    if not start_web_script.exists():
        print(f"ERROR: Server script not found: {start_web_script}")
        sys.exit(1)

    use_pty = sys.platform != "win32"

    if use_pty:
        os.set_blocking(sys.stdout.fileno(), True)

    server_process = subprocess.Popen(
        [sys.executable, str(start_web_script), "--no-browser", "--port", str(args.port), "--host", args.host],
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    print("Waiting for server to start...")
    print("-" * 40)

    if wait_for_server_with_output(url, server_process):
        print("-" * 40)
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
        print("-" * 40)
        print(f"ERROR: Server failed to start within {STARTUP_TIMEOUT} seconds")
        print()
        if server_process.stdout:
            remaining_output = server_process.stdout.read()
            if remaining_output:
                print("Remaining server output:")
                print(remaining_output)
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
