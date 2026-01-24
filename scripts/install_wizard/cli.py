"""Command-line interface for the installation wizard.

This module provides the main entry point and argument parsing
for the installation wizard, supporting both conda and Docker modes.
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
    python scripts/install_wizard.py                  # Interactive conda install
    python scripts/install_wizard.py --docker         # Docker-based install
    python scripts/install_wizard.py --component mocap_core # Install specific component
    python scripts/install_wizard.py --check-only     # Check status only
    python scripts/install_wizard.py --validate       # Run validation tests
    python scripts/install_wizard.py --docker --yolo  # Non-interactive Docker install
"""
    )
    parser.add_argument(
        "--docker", "-d",
        action="store_true",
        help="Use Docker-based installation instead of conda"
    )
    parser.add_argument(
        "--component", "-C",
        type=str,
        choices=['core', 'pytorch', 'colmap', 'mocap_core', 'wham', 'econ', 'comfyui'],
        help="Install specific component (conda mode only)"
    )
    parser.add_argument(
        "--check-only", "-c",
        action="store_true",
        help="Check installation status only (don't install)"
    )
    parser.add_argument(
        "--validate", "-v",
        action="store_true",
        help="Run validation tests on existing installation (conda mode only)"
    )
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume previous interrupted installation"
    )
    parser.add_argument(
        "--yolo", "-y",
        action="store_true",
        help="Non-interactive full install with auto-yes"
    )
    parser.add_argument(
        "--test", "-t",
        action="store_true",
        help="Run test pipeline after installation (Docker mode only)"
    )

    args = parser.parse_args()

    if args.docker:
        from .docker import DockerWizard

        if args.component:
            print("Warning: --component is ignored in Docker mode")
        if args.validate:
            print("Warning: --validate is ignored in Docker mode")

        wizard = DockerWizard()
        wizard.interactive_install(
            check_only=args.check_only,
            run_test=args.test,
            yolo=args.yolo,
            resume=args.resume
        )
    else:
        if args.test:
            print("Warning: --test is only available in Docker mode (--docker)")

        wizard = InstallationWizard()

        if args.check_only:
            wizard.check_system_requirements()
            status = wizard.check_all_components()
            wizard.print_status(status)
            sys.exit(0)

        if args.validate:
            wizard.validator.validate_and_report()
            sys.exit(0)

        success = wizard.interactive_install(
            component=args.component,
            resume=args.resume,
            yolo=args.yolo
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
