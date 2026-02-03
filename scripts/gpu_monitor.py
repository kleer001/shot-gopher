#!/usr/bin/env python3
"""GPU VRAM monitor for profiling pipeline stages.

Polls nvidia-smi and logs VRAM usage to a timestamped file.
Can run standalone or be integrated into the pipeline.

Standalone usage:
    python gpu_monitor.py                    # Log to gpu_usage.log
    python gpu_monitor.py -o cleanplate.log  # Custom output file
    python gpu_monitor.py -i 0.5             # Poll every 0.5 seconds

Pipeline integration:
    from gpu_monitor import GpuMonitor
    monitor = GpuMonitor(output_path, interval=0.5)
    monitor.start()
    monitor.log_stage("cleanplate")
    # ... run stage ...
    monitor.stop()
"""

import argparse
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


def get_gpu_stats() -> Optional[dict]:
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


class GpuMonitor:
    """Background GPU monitor with stage logging."""

    def __init__(
        self,
        output_path: Path,
        interval: float = 1.0,
        quiet: bool = True,
    ):
        self.output_path = Path(output_path)
        self.interval = interval
        self.quiet = quiet
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._peak_mb = 0
        self._current_stage = "init"
        self._start_time: Optional[datetime] = None
        self._lock = threading.Lock()

    def start(self) -> bool:
        """Start the background monitor thread."""
        stats = get_gpu_stats()
        if not stats:
            if not self.quiet:
                print("Warning: Could not query nvidia-smi, GPU monitoring disabled")
            return False

        self._start_time = datetime.now()
        self._peak_mb = 0

        header = f"# GPU Monitor - {stats['name']} ({stats['total_mb']/1024:.1f}GB)\n"
        header += f"# Started: {self._start_time.isoformat()}\n"
        header += f"# Interval: {self.interval}s\n"
        header += "#\n"
        header += "# Format: timestamp | stage | used_gb | peak_gb | gpu_util%\n"
        header += "#\n"

        with open(self.output_path, "w") as f:
            f.write(header)

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

        if not self.quiet:
            print(f"GPU monitoring started: {self.output_path}")

        return True

    def stop(self) -> dict:
        """Stop the monitor and return summary stats."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

        duration = datetime.now() - self._start_time if self._start_time else None

        summary = {
            "peak_vram_gb": self._peak_mb / 1024,
            "duration": str(duration) if duration else "unknown",
            "log_file": str(self.output_path),
        }

        footer = f"\n# Stopped: {datetime.now().isoformat()}\n"
        if duration:
            footer += f"# Duration: {duration}\n"
        footer += f"# Peak VRAM: {self._peak_mb/1024:.2f}GB\n"

        with open(self.output_path, "a") as f:
            f.write(footer)

        if not self.quiet:
            print(f"GPU monitoring stopped. Peak VRAM: {self._peak_mb/1024:.2f}GB")

        return summary

    def log_stage(self, stage_name: str) -> None:
        """Log a stage transition marker."""
        with self._lock:
            self._current_stage = stage_name

        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        marker = f"\n# === STAGE: {stage_name} ({timestamp}) ===\n"

        with open(self.output_path, "a") as f:
            f.write(marker)

        if not self.quiet:
            print(f"  [GPU] Stage: {stage_name}")

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            stats = get_gpu_stats()
            if stats:
                self._peak_mb = max(self._peak_mb, stats["used_mb"])

                with self._lock:
                    stage = self._current_stage

                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                used_gb = stats["used_mb"] / 1024
                peak_gb = self._peak_mb / 1024

                line = f"{timestamp} | {stage:12} | {used_gb:5.2f}GB | {peak_gb:5.2f}GB | {stats['gpu_util']:3d}%\n"

                with open(self.output_path, "a") as f:
                    f.write(line)

            time.sleep(self.interval)


_active_monitor: Optional[GpuMonitor] = None


def start_pipeline_monitor(project_dir: Path, interval: float = 0.5) -> Optional[GpuMonitor]:
    """Start a GPU monitor for a pipeline run.

    Args:
        project_dir: Project directory (log saved here)
        interval: Polling interval in seconds

    Returns:
        GpuMonitor instance or None if GPU not available
    """
    global _active_monitor

    log_path = project_dir / "gpu_profile.log"
    monitor = GpuMonitor(log_path, interval=interval, quiet=False)

    if monitor.start():
        _active_monitor = monitor
        return monitor

    return None


def stop_pipeline_monitor() -> Optional[dict]:
    """Stop the active pipeline monitor."""
    global _active_monitor

    if _active_monitor:
        summary = _active_monitor.stop()
        _active_monitor = None
        return summary

    return None


def log_stage(stage_name: str) -> None:
    """Log a stage marker to the active monitor."""
    if _active_monitor:
        _active_monitor.log_stage(stage_name)


def monitor_cli(output_path: Path, interval: float, quiet: bool) -> None:
    """CLI monitoring mode."""
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

                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                used_gb = stats["used_mb"] / 1024
                total_gb = stats["total_mb"] / 1024
                peak_gb = peak_mb / 1024
                pct = (stats["used_mb"] / stats["total_mb"]) * 100

                line = (
                    f"{timestamp}  "
                    f"VRAM: {used_gb:5.2f}/{total_gb:.1f}GB ({pct:4.1f}%)  "
                    f"Peak: {peak_gb:5.2f}GB  "
                    f"GPU: {stats['gpu_util']:3d}%"
                )

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
    monitor_cli(args.output, args.interval, args.quiet)


if __name__ == "__main__":
    main()
