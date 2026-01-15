"""Command-line interface for the installation wizard.

This module provides the main entry point and argument parsing
for the installation wizard.
"""

import argparse
import sys

from .wizard import InstallationWizard


def main():
    """Main entry point for the installation wizard."""
    parser = argparse.ArgumentParser(
        description="Interactive installation wizard for VFX pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/install_wizard.py                  # Interactive install
    python scripts/install_wizard.py --component mocap # Install specific component
    python scripts/install_wizard.py --check-only     # Check status only
    python scripts/install_wizard.py --validate       # Run validation tests
"""
    )
    parser.add_argument(
        "--component", "-C",
        type=str,
        choices=['core', 'pytorch', 'colmap', 'mocap_core', 'wham', 'econ', 'comfyui'],
        help="Install specific component"
    )
    parser.add_argument(
        "--check-only", "-c",
        action="store_true",
        help="Check installation status only (don't install)"
    )
    parser.add_argument(
        "--validate", "-v",
        action="store_true",
        help="Run validation tests on existing installation"
    )
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume previous interrupted installation"
    )
    parser.add_argument(
        "--yolo", "-y",
        action="store_true",
        help="Non-interactive full stack install (option 3 + auto-yes)"
    )

    args = parser.parse_args()

    wizard = InstallationWizard()

    if args.check_only:
        wizard.check_system_requirements()
        status = wizard.check_all_components()
        wizard.print_status(status)
        sys.exit(0)

    if args.validate:
        wizard.validator.validate_and_report()
        sys.exit(0)

    success = wizard.interactive_install(component=args.component, resume=args.resume, yolo=args.yolo)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
