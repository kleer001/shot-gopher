#!/usr/bin/env python3
"""GPU VRAM monitor for profiling pipeline stages.

Polls nvidia-smi and logs VRAM usage to a timestamped file.
Run in a separate terminal alongside processing to capture real usage data.

Usage:
    python gpu_monitor.py                    # Log to gpu_usage.log
    python gpu_monitor.py -o cleanplate.log  # Custom output file
    python gpu_monitor.py -i 0.5             # Poll every 0.5 seconds
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def get_gpu_stats() -> dict | None:
    """Query nvidia-smi for current GPU stats."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.free,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None

        parts = result.stdout.strip().split(", ")
        if len(parts) >= 5:
            return {
                "name": parts[0],
                "used_mb": int(parts[1]),
                "free_mb": int(parts[2]),
                "total_mb": int(parts[3]),
                "gpu_util": int(parts[4]),
            }
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


def format_log_line(stats: dict, peak_mb: int) -> str:
    """Format a single log line."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    used_gb = stats["used_mb"] / 1024
    total_gb = stats["total_mb"] / 1024
    peak_gb = peak_mb / 1024
    pct = (stats["used_mb"] / stats["total_mb"]) * 100
    return (
        f"{timestamp}  "
        f"VRAM: {used_gb:5.2f}/{total_gb:.1f}GB ({pct:4.1f}%)  "
        f"Peak: {peak_gb:5.2f}GB  "
        f"GPU: {stats['gpu_util']:3d}%"
    )


def monitor_loop(output_path: Path, interval: float, quiet: bool) -> None:
    """Main monitoring loop."""
    stats = get_gpu_stats()
    if not stats:
        print("Error: Could not query nvidia-smi. Is NVIDIA driver installed?")
        sys.exit(1)

    peak_mb = 0
    start_time = datetime.now()

    header = f"# GPU Monitor - {stats['name']} ({stats['total_mb']/1024:.1f}GB)\n"
    header += f"# Started: {start_time.isoformat()}\n"
    header += f"# Interval: {interval}s\n"
    header += "#\n"

    with open(output_path, "w") as f:
        f.write(header)

    if not quiet:
        print(f"Monitoring GPU: {stats['name']} ({stats['total_mb']/1024:.1f}GB)")
        print(f"Logging to: {output_path}")
        print(f"Interval: {interval}s")
        print("Press Ctrl+C to stop\n")

    try:
        while True:
            stats = get_gpu_stats()
            if stats:
                peak_mb = max(peak_mb, stats["used_mb"])
                line = format_log_line(stats, peak_mb)

                with open(output_path, "a") as f:
                    f.write(line + "\n")

                if not quiet:
                    print(f"\r{line}", end="", flush=True)

            time.sleep(interval)

    except KeyboardInterrupt:
        duration = datetime.now() - start_time
        summary = f"\n# Stopped after {duration}\n"
        summary += f"# Peak VRAM: {peak_mb/1024:.2f}GB\n"

        with open(output_path, "a") as f:
            f.write(summary)

        if not quiet:
            print(f"\n\nPeak VRAM: {peak_mb/1024:.2f}GB")
            print(f"Duration: {duration}")
            print(f"Log saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Monitor GPU VRAM usage and log to file"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("gpu_usage.log"),
        help="Output log file (default: gpu_usage.log)",
    )
    parser.add_argument(
        "-i", "--interval",
        type=float,
        default=1.0,
        help="Polling interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress terminal output (log to file only)",
    )

    args = parser.parse_args()
    monitor_loop(args.output, args.interval, args.quiet)


if __name__ == "__main__":
    main()
