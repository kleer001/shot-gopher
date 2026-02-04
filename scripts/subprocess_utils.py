"""Subprocess execution utilities.

Provides reusable components for running external processes with:
- Streaming output capture
- Progress parsing and display
- TTY-aware output throttling
- Consistent error handling
"""

import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


@dataclass
class ProgressPattern:
    """Defines a regex pattern for extracting progress information.

    Attributes:
        name: Descriptive name for this progress type
        pattern: Compiled regex with groups for current/total
        format_str: Format string for display (receives current, total, percent)
        current_group: Regex group index for current value (default: 1)
        total_group: Regex group index for total value (default: 2)
    """

    name: str
    pattern: re.Pattern
    format_str: str = "{name}: {current}/{total}"
    current_group: int = 1
    total_group: int = 2

    def match(self, line: str) -> Optional[tuple[int, int]]:
        """Try to match this pattern against a line.

        Returns:
            Tuple of (current, total) if matched, None otherwise
        """
        match = self.pattern.search(line)
        if match:
            try:
                current = int(match.group(self.current_group))
                total = int(match.group(self.total_group))
                return current, total
            except (IndexError, ValueError):
                return None
        return None

    def format(self, current: int, total: int) -> str:
        """Format progress for display."""
        percent = int(100 * current / total) if total > 0 else 0
        return self.format_str.format(
            name=self.name,
            current=current,
            total=total,
            percent=percent,
        )


@dataclass
class ProcessResult:
    """Result of a subprocess execution.

    Compatible with subprocess.CompletedProcess interface.
    """

    returncode: int
    stdout: str
    stderr: str = ""
    cmd: list[str] = field(default_factory=list)


class ProgressTracker:
    """Tracks and displays progress from subprocess output.

    Handles TTY detection, output throttling, and deduplication.
    """

    def __init__(
        self,
        patterns: list[ProgressPattern] = None,
        throttle_interval: float = 2.0,
        min_total: int = 10,
        report_interval: int = 0,
    ):
        """Initialize progress tracker.

        Args:
            patterns: List of progress patterns to match
            throttle_interval: Seconds between updates (non-TTY only)
            min_total: Minimum total value to show progress
            report_interval: Report every N items (0 = report all)
        """
        self.patterns = patterns or []
        self.throttle_interval = throttle_interval
        self.min_total = min_total
        self.report_interval = report_interval
        self.is_tty = sys.stdout.isatty()
        self._last_progress = ""
        self._last_time = 0.0
        self._last_reported = 0

    def _should_print(self, current: int = 0, total: int = 0) -> bool:
        """Determine if progress should be printed now."""
        if self.is_tty:
            return True

        now = time.time()
        if now - self._last_time < self.throttle_interval:
            return False
        self._last_time = now

        if self.report_interval > 0:
            if current - self._last_reported < self.report_interval:
                if current != total:
                    return False
            self._last_reported = current

        return True

    def process_line(self, line: str) -> Optional[str]:
        """Process a line of output and return progress string if applicable.

        Args:
            line: Line of output to process

        Returns:
            Progress string to display, or None
        """
        line_stripped = line.strip()
        if not line_stripped:
            return None

        for pattern in self.patterns:
            result = pattern.match(line_stripped)
            if result:
                current, total = result
                if total < self.min_total:
                    continue

                progress = pattern.format(current, total)
                if progress == self._last_progress:
                    continue

                if not self._should_print(current, total):
                    continue

                self._last_progress = progress
                return progress

        return None

    def print_progress(self, progress: str) -> None:
        """Print progress string with appropriate formatting."""
        if self.is_tty:
            print(f"\r    {progress}    ", end="")
        else:
            print(f"    {progress}")
        sys.stdout.flush()

    def finish(self) -> None:
        """Clean up after progress tracking (print newline if needed)."""
        if self.is_tty and self._last_progress:
            print()


class ProcessRunner:
    """Runs external processes with streaming output and progress tracking."""

    def __init__(
        self,
        progress_tracker: Optional[ProgressTracker] = None,
        capture_output: bool = True,
        print_command: bool = True,
        shell: bool = False,
    ):
        """Initialize process runner.

        Args:
            progress_tracker: Optional progress tracker for output parsing
            capture_output: Whether to capture stdout/stderr
            print_command: Whether to print the command before running
            shell: Whether to use shell execution
        """
        self.progress_tracker = progress_tracker
        self.capture_output = capture_output
        self.print_command = print_command
        self.shell = shell

    def run(
        self,
        cmd: list[str],
        description: str = "",
        cwd: Optional[Path] = None,
        env: Optional[dict] = None,
        timeout: Optional[int] = None,
        check: bool = True,
        line_callback: Optional[Callable[[str], None]] = None,
    ) -> ProcessResult:
        """Run a command with streaming output.

        Args:
            cmd: Command and arguments as list
            description: Human-readable description for logging
            cwd: Working directory
            env: Environment variables (merged with current env)
            timeout: Timeout in seconds (None = no timeout)
            check: Raise exception on non-zero exit
            line_callback: Optional callback for each output line

        Returns:
            ProcessResult with captured output

        Raises:
            subprocess.CalledProcessError: If check=True and command fails
        """
        if description:
            print(f"  → {description}")
        if self.print_command:
            print(f"    $ {' '.join(cmd)}")
        sys.stdout.flush()

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE if self.capture_output else None,
            stderr=subprocess.STDOUT if self.capture_output else None,
            text=True,
            bufsize=1,
            cwd=cwd,
            env=env,
            shell=self.shell,
        )

        stdout_lines = []
        start_time = time.time()

        if self.capture_output and process.stdout:
            for line in iter(process.stdout.readline, ''):
                stdout_lines.append(line)

                if line_callback:
                    line_callback(line)

                if self.progress_tracker:
                    progress = self.progress_tracker.process_line(line)
                    if progress:
                        self.progress_tracker.print_progress(progress)

                if timeout and (time.time() - start_time) > timeout:
                    process.kill()
                    process.wait()
                    raise subprocess.TimeoutExpired(cmd, timeout)

        if self.progress_tracker:
            self.progress_tracker.finish()

        process.wait()
        stdout = ''.join(stdout_lines)

        if check and process.returncode != 0:
            print(f"    Error output:\n{stdout}", file=sys.stderr)
            raise subprocess.CalledProcessError(
                process.returncode, cmd, stdout, ""
            )

        return ProcessResult(
            returncode=process.returncode,
            stdout=stdout,
            stderr="",
            cmd=cmd,
        )


def create_colmap_patterns() -> list[ProgressPattern]:
    """Create progress patterns for COLMAP operations."""
    return [
        ProgressPattern(
            name="Extracting features",
            pattern=re.compile(r'Processed file \[(\d+)/(\d+)\]'),
            format_str="Extracting features: {current}/{total}",
        ),
        ProgressPattern(
            name="Matching",
            pattern=re.compile(r'Matching block \[(\d+)/(\d+)'),
            format_str="Matching: block {current}/{total}",
        ),
        ProgressPattern(
            name="Registered",
            pattern=re.compile(r'Registering image #(\d+)\s*\((\d+)\)'),
            format_str="Registered {current}/{total} images",
        ),
        ProgressPattern(
            name="Registered",
            pattern=re.compile(r'Registered\s+(\d+)/(\d+)\s+images'),
            format_str="Registered {current}/{total} images",
        ),
    ]


def create_training_patterns() -> list[ProgressPattern]:
    """Create progress patterns for ML training operations (GS-IR, etc.)."""
    return [
        ProgressPattern(
            name="Training",
            pattern=re.compile(r'Iteration:\s*(\d+)/(\d+)'),
            format_str="Training: {current}/{total} ({percent}%)",
        ),
        ProgressPattern(
            name="Training",
            pattern=re.compile(r'(\d+)%\|.*\|\s*(\d+)/(\d+)'),
            format_str="Training: {current}/{total} ({percent}%)",
            current_group=2,
            total_group=3,
        ),
    ]


def create_generic_patterns() -> list[ProgressPattern]:
    """Create generic progress patterns for various operations."""
    return [
        ProgressPattern(
            name="Processing",
            pattern=re.compile(r'\[(\d+)/(\d+)\]'),
            format_str="Processing: {current}/{total}",
        ),
        ProgressPattern(
            name="Progress",
            pattern=re.compile(r'(\d+)/(\d+)'),
            format_str="Progress: {current}/{total}",
        ),
    ]
