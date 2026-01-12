#!/usr/bin/env python3
"""VFX Pipeline Janitor - Maintenance and diagnostic tool.

Performs health checks, updates, cleanup, and repairs on your VFX pipeline installation.

Usage:
    python janitor.py --health         # Check installation health
    python janitor.py --update         # Check for and apply updates
    python janitor.py --clean          # Clean up orphaned files
    python janitor.py --repair         # Repair broken components
    python janitor.py --report         # Generate detailed status report
    python janitor.py --all            # Run all checks and operations

Examples:
    python janitor.py -h               # Health check (quick)
    python janitor.py -u -y            # Update all components (auto-confirm)
    python janitor.py -c --dry-run     # Preview cleanup without deleting
    python janitor.py -a               # Full maintenance run
"""

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

# Import utilities from install_wizard
sys.path.insert(0, str(Path(__file__).parent))
try:
    from install_wizard import (
        CondaEnvironmentManager,
        InstallationStateManager,
        CheckpointDownloader,
        InstallationValidator,
        print_success,
        print_warning,
        print_error,
        run_command
    )
except ImportError:
    print("Error: Could not import from install_wizard.py", file=sys.stderr)
    print("Ensure janitor.py is in the same directory as install_wizard.py")
    sys.exit(1)


class GitRepoChecker:
    """Check git repository status and updates."""

    def __init__(self, repo_path: Path, name: str):
        self.repo_path = repo_path
        self.name = name

    def exists(self) -> bool:
        """Check if repo exists."""
        return (self.repo_path / ".git").exists()

    def is_clean(self) -> Tuple[bool, str]:
        """Check if working directory is clean."""
        if not self.exists():
            return False, "Not a git repository"

        success, output = run_command(
            ["git", "-C", str(self.repo_path), "status", "--porcelain"],
            check=False,
            capture=True
        )

        if not success:
            return False, "Failed to check status"

        if output.strip():
            return False, f"Uncommitted changes:\n{output}"

        return True, "Clean"

    def check_updates(self) -> Tuple[bool, str]:
        """Check if updates are available."""
        if not self.exists():
            return False, "Not a git repository"

        # Fetch latest
        success, _ = run_command(
            ["git", "-C", str(self.repo_path), "fetch"],
            check=False,
            capture=True
        )

        if not success:
            return False, "Failed to fetch updates"

        # Compare local and remote
        success, local_hash = run_command(
            ["git", "-C", str(self.repo_path), "rev-parse", "HEAD"],
            check=False,
            capture=True
        )

        if not success:
            return False, "Failed to get local hash"

        success, remote_hash = run_command(
            ["git", "-C", str(self.repo_path), "rev-parse", "@{u}"],
            check=False,
            capture=True
        )

        if not success:
            return False, "No upstream branch"

        local_hash = local_hash.strip()
        remote_hash = remote_hash.strip()

        if local_hash == remote_hash:
            return True, "Up to date"

        # Get commit count
        success, count = run_command(
            ["git", "-C", str(self.repo_path), "rev-list", "--count", f"{local_hash}..{remote_hash}"],
            check=False,
            capture=True
        )

        if success:
            return True, f"{count.strip()} commit(s) behind"

        return True, "Updates available"

    def update(self) -> Tuple[bool, str]:
        """Pull latest changes."""
        if not self.exists():
            return False, "Not a git repository"

        success, output = run_command(
            ["git", "-C", str(self.repo_path), "pull"],
            check=False,
            capture=True
        )

        if success:
            return True, "Updated successfully"

        return False, f"Update failed: {output}"

    def get_current_commit(self) -> str:
        """Get current commit hash."""
        if not self.exists():
            return "N/A"

        success, output = run_command(
            ["git", "-C", str(self.repo_path), "rev-parse", "--short", "HEAD"],
            check=False,
            capture=True
        )

        return output.strip() if success else "Unknown"


class DiskUsageAnalyzer:
    """Analyze disk usage by component."""

    def __init__(self, install_dir: Path):
        self.install_dir = install_dir

    def get_dir_size(self, path: Path) -> int:
        """Get directory size in bytes."""
        if not path.exists():
            return 0

        if path.is_file():
            return path.stat().st_size

        total = 0
        try:
            for item in path.rglob("*"):
                if item.is_file():
                    try:
                        total += item.stat().st_size
                    except (OSError, PermissionError):
                        pass
        except (OSError, PermissionError):
            pass

        return total

    def format_size(self, size_bytes: int) -> str:
        """Format bytes to human-readable size."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"

    def analyze(self) -> Dict[str, Tuple[int, str]]:
        """Analyze disk usage by component.

        Returns:
            Dict mapping component name to (size_bytes, formatted_size)
        """
        components = {
            "WHAM": self.install_dir / "WHAM",
            "TAVA": self.install_dir / "tava",
            "ECON": self.install_dir / "ECON",
            "ComfyUI": self.install_dir / "ComfyUI",
            "State files": self.install_dir / "install_state.json",
            "Config files": self.install_dir / "config.json",
        }

        results = {}
        for name, path in components.items():
            size = self.get_dir_size(path)
            results[name] = (size, self.format_size(size))

        # Calculate total
        total_size = sum(size for size, _ in results.values())
        results["TOTAL"] = (total_size, self.format_size(total_size))

        return results


class Janitor:
    """Main janitor class for maintenance operations."""

    def __init__(self, install_dir: Optional[Path] = None):
        # Detect installation directory
        repo_root = Path(__file__).parent.parent.resolve()
        self.install_dir = install_dir or (repo_root / ".vfx_pipeline")

        # Initialize managers
        self.conda_manager = CondaEnvironmentManager()
        self.state_manager = InstallationStateManager(self.install_dir / "install_state.json")
        self.checkpoint_downloader = CheckpointDownloader(self.install_dir)
        self.validator = InstallationValidator(self.conda_manager, self.install_dir)
        self.disk_analyzer = DiskUsageAnalyzer(self.install_dir)

        # Git repos
        self.repos = {
            'WHAM': GitRepoChecker(self.install_dir / "WHAM", "WHAM"),
            'TAVA': GitRepoChecker(self.install_dir / "tava", "TAVA"),
            'ECON': GitRepoChecker(self.install_dir / "ECON", "ECON"),
            'ComfyUI': GitRepoChecker(self.install_dir / "ComfyUI", "ComfyUI"),
        }

    def health_check(self) -> bool:
        """Run comprehensive health check.

        Returns:
            True if all checks pass
        """
        print("\n" + "=" * 60)
        print("HEALTH CHECK")
        print("=" * 60)

        all_passed = True

        # Check installation directory
        print("\n[Installation Directory]")
        if self.install_dir.exists():
            print_success(f"Found: {self.install_dir}")
        else:
            print_error(f"Not found: {self.install_dir}")
            print("Run install_wizard.py first to set up the pipeline")
            return False

        # Check conda environment
        print("\n[Conda Environment]")
        if self.conda_manager.detect_conda():
            print_success(f"Conda detected: {self.conda_manager.conda_exe}")

            if self.conda_manager.environment_exists():
                print_success(f"Environment '{self.conda_manager.env_name}' exists")
            else:
                print_warning(f"Environment '{self.conda_manager.env_name}' not found")
                all_passed = False
        else:
            print_error("Conda not detected")
            all_passed = False

        # Check git repositories
        print("\n[Git Repositories]")
        for name, repo in self.repos.items():
            if repo.exists():
                is_clean, status = repo.is_clean()
                if is_clean:
                    commit = repo.get_current_commit()
                    print_success(f"{name}: Clean ({commit})")
                else:
                    print_warning(f"{name}: {status}")
                    all_passed = False
            else:
                print_warning(f"{name}: Not installed")

        # Check checkpoints
        print("\n[Checkpoints]")
        checkpoint_status = self.validator.validate_checkpoint_files()
        if checkpoint_status:
            for component, exists in checkpoint_status.items():
                if exists:
                    print_success(f"{component}: Found")
                else:
                    print_warning(f"{component}: Missing")
        else:
            print_warning("No checkpoints validated")

        # Check installation state
        print("\n[Installation State]")
        state = self.state_manager.state
        if state:
            installed = [k for k, v in state.get('components', {}).items() if v == 'completed']
            if installed:
                print_success(f"Installed components: {', '.join(installed)}")
            else:
                print_warning("No components marked as installed")
        else:
            print_warning("No installation state found")

        # Run validation tests
        print("\n[Validation Tests]")
        validation_passed = self.validator.validate_and_report()

        if not validation_passed:
            all_passed = False

        print("\n" + "=" * 60)
        if all_passed:
            print_success("HEALTH CHECK PASSED")
        else:
            print_warning("HEALTH CHECK: Some issues detected")
        print("=" * 60 + "\n")

        return all_passed

    def check_updates(self) -> Dict[str, Tuple[bool, str]]:
        """Check for updates to all git repositories.

        Returns:
            Dict mapping repo name to (has_updates, status_message)
        """
        print("\n" + "=" * 60)
        print("UPDATE CHECK")
        print("=" * 60 + "\n")

        results = {}

        for name, repo in self.repos.items():
            if not repo.exists():
                print(f"[{name}] Not installed - skipping")
                results[name] = (False, "Not installed")
                continue

            print(f"[{name}] Checking for updates...")
            has_update, status = repo.check_updates()
            results[name] = (has_update, status)

            if "behind" in status.lower():
                print_warning(f"  → {status}")
            elif "up to date" in status.lower():
                print_success(f"  → {status}")
            else:
                print(f"  → {status}")

        return results

    def apply_updates(self, auto_confirm: bool = False) -> bool:
        """Apply updates to repositories.

        Args:
            auto_confirm: If True, don't ask for confirmation

        Returns:
            True if all updates succeeded
        """
        # Check what needs updating
        updates_available = self.check_updates()

        needs_update = {
            name: status
            for name, (_, status) in updates_available.items()
            if "behind" in status.lower()
        }

        if not needs_update:
            print_success("\nAll components are up to date!")
            return True

        print(f"\n{len(needs_update)} component(s) have updates available:")
        for name, status in needs_update.items():
            print(f"  - {name}: {status}")

        if not auto_confirm:
            response = input("\nApply updates? [y/N]: ").strip().lower()
            if response != 'y':
                print("Update cancelled")
                return False

        print("\n" + "=" * 60)
        print("APPLYING UPDATES")
        print("=" * 60 + "\n")

        all_success = True

        for name in needs_update.keys():
            repo = self.repos[name]
            print(f"[{name}] Updating...")

            # Check if clean first
            is_clean, clean_status = repo.is_clean()
            if not is_clean:
                print_warning(f"  → Skipping: {clean_status}")
                all_success = False
                continue

            success, message = repo.update()
            if success:
                print_success(f"  → {message}")
            else:
                print_error(f"  → {message}")
                all_success = False

        return all_success

    def cleanup(self, dry_run: bool = False) -> int:
        """Clean up orphaned files and temporary data.

        Args:
            dry_run: If True, only report what would be deleted

        Returns:
            Size of cleaned data in bytes
        """
        print("\n" + "=" * 60)
        if dry_run:
            print("CLEANUP (DRY RUN)")
        else:
            print("CLEANUP")
        print("=" * 60 + "\n")

        total_cleaned = 0

        # Find temporary files
        temp_patterns = [
            "**/*.tmp",
            "**/*.pyc",
            "**/__pycache__",
            "**/temp_*",
            "**/.DS_Store",
        ]

        print("[Scanning for temporary files...]")

        files_to_remove = []
        for pattern in temp_patterns:
            for path in self.install_dir.glob(pattern):
                if path.exists():
                    files_to_remove.append(path)

        if not files_to_remove:
            print_success("No temporary files found")
            return 0

        # Calculate size
        for path in files_to_remove:
            if path.is_file():
                total_cleaned += path.stat().st_size
            elif path.is_dir():
                total_cleaned += self.disk_analyzer.get_dir_size(path)

        formatted_size = self.disk_analyzer.format_size(total_cleaned)

        print(f"\nFound {len(files_to_remove)} temporary file(s) ({formatted_size}):")
        for path in files_to_remove[:10]:  # Show first 10
            rel_path = path.relative_to(self.install_dir)
            print(f"  - {rel_path}")

        if len(files_to_remove) > 10:
            print(f"  ... and {len(files_to_remove) - 10} more")

        if dry_run:
            print_warning("\nDry run - no files deleted")
            return total_cleaned

        # Confirm deletion
        response = input("\nDelete these files? [y/N]: ").strip().lower()
        if response != 'y':
            print("Cleanup cancelled")
            return 0

        # Delete files
        deleted_count = 0
        for path in files_to_remove:
            try:
                if path.is_file():
                    path.unlink()
                    deleted_count += 1
                elif path.is_dir():
                    shutil.rmtree(path)
                    deleted_count += 1
            except (OSError, PermissionError) as e:
                print_warning(f"Could not delete {path}: {e}")

        print_success(f"\nDeleted {deleted_count} file(s), freed {formatted_size}")

        return total_cleaned

    def repair(self, auto_confirm: bool = False) -> bool:
        """Repair broken components.

        Args:
            auto_confirm: If True, don't ask for confirmation

        Returns:
            True if repairs succeeded
        """
        print("\n" + "=" * 60)
        print("REPAIR")
        print("=" * 60 + "\n")

        print("[Checking for issues...]")

        # Check what's broken
        issues = []

        # Check conda environment
        if self.conda_manager.detect_conda():
            if not self.conda_manager.environment_exists():
                issues.append(("conda_env", "Conda environment missing"))

        # Check checkpoints
        checkpoint_status = self.validator.validate_checkpoint_files()
        for component, exists in checkpoint_status.items():
            if not exists:
                issues.append(("checkpoint", f"{component} checkpoint missing"))

        if not issues:
            print_success("No issues detected!")
            return True

        print(f"\nFound {len(issues)} issue(s):")
        for issue_type, description in issues:
            print(f"  - {description}")

        if not auto_confirm:
            response = input("\nAttempt repairs? [y/N]: ").strip().lower()
            if response != 'y':
                print("Repair cancelled")
                return False

        print("\n[Applying repairs...]")

        all_success = True

        for issue_type, description in issues:
            if issue_type == "conda_env":
                print(f"\nRepairing: {description}")
                if self.conda_manager.create_environment():
                    print_success("  → Environment created")
                else:
                    print_error("  → Failed to create environment")
                    all_success = False

            elif issue_type == "checkpoint":
                print(f"\nRepairing: {description}")
                # Extract component name from description
                component = description.split()[0].lower()

                # Re-download checkpoint
                # Note: This requires the state manager to have component info
                if self.checkpoint_downloader.download_all_checkpoints([component]):
                    print_success("  → Checkpoint downloaded")
                else:
                    print_error("  → Failed to download checkpoint")
                    all_success = False

        return all_success

    def generate_report(self) -> None:
        """Generate detailed status report."""
        print("\n" + "=" * 60)
        print("VFX PIPELINE STATUS REPORT")
        print("=" * 60)
        print(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Install directory: {self.install_dir}")
        print("=" * 60)

        # Disk usage
        print("\n[Disk Usage]")
        usage = self.disk_analyzer.analyze()

        # Sort by size (excluding TOTAL)
        sorted_components = sorted(
            [(k, v) for k, v in usage.items() if k != "TOTAL"],
            key=lambda x: x[1][0],
            reverse=True
        )

        for name, (size_bytes, formatted) in sorted_components:
            if size_bytes > 0:
                print(f"  {name:20s} {formatted:>12s}")

        print("  " + "-" * 34)
        total_size, total_formatted = usage["TOTAL"]
        print(f"  {'TOTAL':20s} {total_formatted:>12s}")

        # Component status
        print("\n[Components]")
        state = self.state_manager.state
        installed = state.get('components', {})

        for component, status in installed.items():
            print(f"  {component:20s} {status}")

        # Repository status
        print("\n[Git Repositories]")
        for name, repo in self.repos.items():
            if repo.exists():
                commit = repo.get_current_commit()
                is_clean, status = repo.is_clean()
                clean_marker = "✓" if is_clean else "✗"
                print(f"  {name:20s} {commit:>10s} {clean_marker}")
            else:
                print(f"  {name:20s} {'Not installed':>10s}")

        # Conda environment
        print("\n[Conda Environment]")
        if self.conda_manager.detect_conda():
            print(f"  Conda: {self.conda_manager.conda_exe}")
            if self.conda_manager.environment_exists():
                print(f"  Environment: {self.conda_manager.env_name} (exists)")
            else:
                print(f"  Environment: {self.conda_manager.env_name} (missing)")
        else:
            print("  Conda: Not detected")

        print("\n" + "=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="VFX Pipeline Janitor - Maintenance and diagnostic tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Operations
    parser.add_argument(
        "--health", "-H",
        action="store_true",
        help="Run health check"
    )
    parser.add_argument(
        "--update", "-u",
        action="store_true",
        help="Check for and apply updates"
    )
    parser.add_argument(
        "--clean", "-c",
        action="store_true",
        help="Clean up temporary files"
    )
    parser.add_argument(
        "--repair", "-r",
        action="store_true",
        help="Repair broken components"
    )
    parser.add_argument(
        "--report", "-R",
        action="store_true",
        help="Generate detailed status report"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run all operations (health, update, clean, report)"
    )

    # Options
    parser.add_argument(
        "--install-dir", "-i",
        type=Path,
        default=None,
        help="Override installation directory"
    )
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Preview cleanup without deleting (for --clean)"
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Auto-confirm all prompts"
    )

    args = parser.parse_args()

    # If no operation specified, show help
    if not any([args.health, args.update, args.clean, args.repair, args.report, args.all]):
        parser.print_help()
        sys.exit(0)

    # Initialize janitor
    janitor = Janitor(install_dir=args.install_dir)

    exit_code = 0

    # Run operations
    if args.all or args.health:
        if not janitor.health_check():
            exit_code = 1

    if args.all or args.update:
        if not janitor.apply_updates(auto_confirm=args.yes):
            exit_code = 1

    if args.all or args.clean:
        janitor.cleanup(dry_run=args.dry_run)

    if args.repair:
        if not janitor.repair(auto_confirm=args.yes):
            exit_code = 1

    if args.all or args.report:
        janitor.generate_report()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
