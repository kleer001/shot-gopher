#!/usr/bin/env python3
"""Interactive installation wizard for Docker-based VFX pipeline.

This is a thin wrapper that imports from the install_wizard package.
The actual implementation is in scripts/install_wizard/docker.py.

For the unified entry point, use:
    python scripts/install_wizard.py --docker

This script is maintained for backward compatibility.

Automates Docker setup for Linux and Windows/WSL2:
- Platform detection and validation
- Docker installation (guided)
- NVIDIA Container Toolkit installation (automated with fallbacks)
- Model downloads (with retry and fallback mechanisms)
- Docker image build
- Test pipeline execution
- Resume capability for interrupted installations

Usage:
    python scripts/install_wizard_docker.py
    python scripts/install_wizard_docker.py --check-only
    python scripts/install_wizard_docker.py --test       # Run test pipeline after install
    python scripts/install_wizard_docker.py --yolo       # Non-interactive full install
    python scripts/install_wizard_docker.py --resume     # Resume interrupted install
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from install_wizard.docker import DockerWizard


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Docker-based VFX Pipeline Installation Wizard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/install_wizard_docker.py                  # Interactive install
    python scripts/install_wizard_docker.py --check-only     # Check prerequisites only
    python scripts/install_wizard_docker.py --test           # Run test pipeline after install
    python scripts/install_wizard_docker.py --yolo           # Non-interactive full install
    python scripts/install_wizard_docker.py --resume         # Resume interrupted install

Note: You can also use the unified wizard with --docker flag:
    python scripts/install_wizard.py --docker
"""
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check prerequisites without installing"
    )
    parser.add_argument(
        "--test", "-t",
        action="store_true",
        help="Run test pipeline after installation"
    )
    parser.add_argument(
        "--yolo", "-y",
        action="store_true",
        help="Non-interactive full install with auto-yes"
    )
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume previous interrupted installation"
    )

    args = parser.parse_args()

    wizard = DockerWizard()
    wizard.interactive_install(
        check_only=args.check_only,
        run_test=args.test,
        yolo=args.yolo,
        resume=args.resume
    )


if __name__ == "__main__":
    main()
