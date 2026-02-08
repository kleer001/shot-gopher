#!/usr/bin/env python3
"""Bug reporter - View and submit error logs.

Simple utility to help users report bugs by viewing captured logs
and providing instructions for submission.

Usage:
    python bug_reporter.py                 # List recent logs
    python bug_reporter.py --log <file>    # View specific log
    python bug_reporter.py --latest        # View latest log
    python bug_reporter.py --email         # Show email instructions

Examples:
    python bug_reporter.py
    python bug_reporter.py --latest
    python bug_reporter.py --log logs/20260119_143022_123456_linux_conda.log
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from log_manager import get_recent_logs, print_log_summary


MAINTAINER_EMAIL = "kleer001code@gmail.com"


def list_logs(count: int = 10) -> None:
    """List recent log files with metadata.

    Args:
        count: Number of logs to display (default: 10)
    """
    logs = get_recent_logs(count=count)

    if not logs:
        print("No logs found.")
        print("\nLogs are automatically created when you run pipeline commands.")
        return

    print(f"\nRecent Logs (newest first):\n")
    print(f"{'#':<4} {'Filename':<45} {'Size':<10} {'Modified'}")
    print("-" * 80)

    for i, log_file in enumerate(logs, 1):
        try:
            stat_info = log_file.stat()
            size_kb = stat_info.st_size / 1024
            mtime_str = datetime.fromtimestamp(stat_info.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            print(f"{i:<4} {log_file.name:<45} {size_kb:>7.1f} KB {mtime_str}")
        except (FileNotFoundError, OSError):
            print(f"{i:<4} {log_file.name:<45} {'<deleted>':<10} {'<unavailable>'}")
            continue

    print(f"\nTo view a log: python {Path(__file__).name} --log {logs[0].name}")
    print(f"Latest log:    python {Path(__file__).name} --latest")


def view_log(log_file: Path) -> None:
    """Display log file contents.

    Args:
        log_file: Path to log file
    """
    if not log_file.exists():
        print(f"Error: Log file not found: {log_file}", file=sys.stderr)
        sys.exit(1)

    print_log_summary(log_file)

    print(f"\n{'='*80}")
    print(f"Log file: {log_file.absolute()}")
    try:
        size_kb = log_file.stat().st_size / 1024
        print(f"Size: {size_kb:.1f} KB")
    except (FileNotFoundError, OSError):
        print("Size: <unavailable>")
    print(f"{'='*80}\n")


def show_email_instructions(log_file: Path = None) -> None:
    """Display instructions for submitting bug reports via email.

    Args:
        log_file: Optional specific log to reference
    """
    if log_file is None:
        logs = get_recent_logs(count=1)
        if logs:
            log_file = logs[0]

    print("\n" + "="*80)
    print("Bug Report Submission Instructions")
    print("="*80 + "\n")

    print("Thank you for helping improve this project!\n")

    if log_file:
        print(f"Latest log file: {log_file.absolute()}\n")

    print("To submit a bug report:\n")
    print(f"1. Email: {MAINTAINER_EMAIL}")
    print("2. Subject: [Bug Report] Brief description of the issue")
    print("3. Attach the log file shown above")
    print("4. Include:")
    print("   - What you were trying to do")
    print("   - What command you ran")
    print("   - What you expected to happen")
    print("   - What actually happened\n")

    print("Example email:\n")
    print(f"  To: {MAINTAINER_EMAIL}")
    print("  Subject: [Bug Report] Pipeline fails at depth stage")
    print("  Attachment: " + (log_file.name if log_file else "latest_log.log"))
    print("""
  Body:
  Hi,

  I'm having trouble running the pipeline. Here's what happened:

  Command: python run_pipeline.py my_video.mp4 --stages depth,roto
  Expected: Pipeline completes successfully
  Actual: Pipeline crashes at depth stage with CUDA error

  Log file attached.

  Thanks!
""")

    print("="*80 + "\n")

    if log_file:
        print(f"Log file location: {log_file.absolute()}")
        print(f"\nOn macOS/Linux: open {log_file.parent}")
        print(f"On Windows: explorer {log_file.parent}\n")


def main() -> None:
    """Main entry point for bug reporter CLI."""
    parser = argparse.ArgumentParser(
        description="View and submit bug reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bug_reporter.py              # List recent logs
  python bug_reporter.py --latest     # View latest log
  python bug_reporter.py --email      # Email instructions
        """
    )

    parser.add_argument(
        "--log",
        type=str,
        metavar="FILE",
        help="View specific log file"
    )

    parser.add_argument(
        "--latest",
        action="store_true",
        help="View latest log file"
    )

    parser.add_argument(
        "--email",
        action="store_true",
        help="Show email submission instructions"
    )

    parser.add_argument(
        "--count",
        type=int,
        default=10,
        metavar="N",
        help="Number of logs to list (default: 10)"
    )

    args = parser.parse_args()

    args.count = max(1, min(args.count, 100))

    if args.email:
        logs = get_recent_logs(count=1)
        show_email_instructions(logs[0] if logs else None)

    elif args.latest:
        logs = get_recent_logs(count=1)
        if not logs:
            print("No logs found.", file=sys.stderr)
            sys.exit(1)
        view_log(logs[0])
        print("\nTo submit this log:")
        print(f"  python {Path(__file__).name} --email\n")

    elif args.log:
        log_path = Path(args.log)
        if not log_path.is_absolute():
            repo_root = Path(__file__).parent.parent
            log_dir = repo_root / "logs"
            log_path = log_dir / log_path
        view_log(log_path)
        print("\nTo submit this log:")
        print(f"  python {Path(__file__).name} --email\n")

    else:
        list_logs(count=args.count)
        print("\nTo submit a bug report:")
        print(f"  python {Path(__file__).name} --email\n")


if __name__ == "__main__":
    main()
