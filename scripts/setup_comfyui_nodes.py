#!/usr/bin/env python3
"""Setup script for ComfyUI custom nodes.

Installs the custom nodes required by the VFX pipeline to the local
.vfx_pipeline/ComfyUI/custom_nodes/ directory.

Usage:
    python scripts/setup_comfyui_nodes.py
    python scripts/setup_comfyui_nodes.py --check   # Check only, don't install
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from install_wizard.installers import ComfyUICustomNodesInstaller
from install_wizard.utils import print_error, print_header, print_info, print_success


def main():
    parser = argparse.ArgumentParser(
        description="Install ComfyUI custom nodes for the VFX pipeline"
    )
    parser.add_argument(
        "--check", "-c",
        action="store_true",
        help="Check if nodes are installed without installing"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List the custom nodes that will be installed"
    )
    args = parser.parse_args()

    installer = ComfyUICustomNodesInstaller()

    if args.list:
        print_header("ComfyUI Custom Nodes")
        print_info(f"Install directory: {installer.install_dir}")
        print()
        for node in installer.CUSTOM_NODES:
            print(f"  - {node['name']}")
            print(f"    {node['url']}")
        return 0

    if args.check:
        print_header("Checking ComfyUI Custom Nodes")
        if installer.check():
            print_success("All custom nodes are installed")
            return 0
        else:
            print_error("Some custom nodes are missing")
            print_info(f"Run without --check to install: python {__file__}")
            return 1

    print_header("Installing ComfyUI Custom Nodes")
    print_info(f"Target directory: {installer.install_dir}")
    print()

    if installer.install():
        print()
        print_success("ComfyUI custom nodes setup complete!")
        return 0
    else:
        print()
        print_error("Some nodes failed to install")
        return 1


if __name__ == "__main__":
    sys.exit(main())
