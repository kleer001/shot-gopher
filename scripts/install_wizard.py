#!/usr/bin/env python3
"""Interactive installation wizard for the VFX pipeline.

This is a thin wrapper that imports from the install_wizard package.
The actual implementation is in scripts/install_wizard/.

Guides users through installing all dependencies for:
- Core pipeline (ComfyUI workflows, COLMAP, etc.)
- Dynamic scene segmentation (SAM3)
- Human motion capture (WHAM, ECON)

Usage:
    python scripts/install_wizard.py
    python scripts/install_wizard.py --component mocap
    python scripts/install_wizard.py --check-only
"""

from install_wizard import main

if __name__ == "__main__":
    main()
