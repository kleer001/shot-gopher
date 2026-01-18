#!/usr/bin/env python3
"""Test container-aware code changes (Phase 1B)."""

import os
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from env_config import (
    is_in_container,
    check_conda_env_or_warn,
    require_conda_env,
    INSTALL_DIR,
    DEFAULT_PROJECTS_DIR,
)

def test_container_detection():
    """Test container detection logic."""
    print("\n=== Test 1: Container Detection ===")

    # Test with no indicators (should be False in local environment)
    is_container = is_in_container()
    print(f"is_in_container() = {is_container}")

    # Test with CONTAINER=true env var
    os.environ["CONTAINER"] = "true"
    is_container_with_env = is_in_container()
    print(f"is_in_container() with CONTAINER=true = {is_container_with_env}")
    del os.environ["CONTAINER"]

    assert is_container_with_env == True, "Should detect CONTAINER=true env var"
    print("✓ Container detection works correctly")

def test_path_env_overrides():
    """Test that environment variables override default paths."""
    print("\n=== Test 2: Path Environment Variable Overrides ===")

    # Save original values
    original_install = str(INSTALL_DIR)
    original_projects = str(DEFAULT_PROJECTS_DIR)

    print(f"Default INSTALL_DIR: {INSTALL_DIR}")
    print(f"Default DEFAULT_PROJECTS_DIR: {DEFAULT_PROJECTS_DIR}")

    # Test with environment variables set (simulate container)
    os.environ["VFX_INSTALL_DIR"] = "/app/.vfx_pipeline"
    os.environ["VFX_PROJECTS_DIR"] = "/workspace/projects"

    # Reimport to pick up new env vars
    import importlib
    import env_config as ec
    importlib.reload(ec)

    print(f"With env vars - INSTALL_DIR: {ec.INSTALL_DIR}")
    print(f"With env vars - DEFAULT_PROJECTS_DIR: {ec.DEFAULT_PROJECTS_DIR}")

    assert str(ec.INSTALL_DIR) == "/app/.vfx_pipeline", "Should use VFX_INSTALL_DIR env var"
    assert str(ec.DEFAULT_PROJECTS_DIR) == "/workspace/projects", "Should use VFX_PROJECTS_DIR env var"

    # Clean up
    del os.environ["VFX_INSTALL_DIR"]
    del os.environ["VFX_PROJECTS_DIR"]

    print("✓ Environment variable overrides work correctly")

def test_conda_skip_in_container():
    """Test that conda checks are skipped in containers."""
    print("\n=== Test 3: Conda Checks Skipped in Container ===")

    # Test in "container" mode
    os.environ["CONTAINER"] = "true"

    # These should return True and not fail, even without conda
    result1 = check_conda_env_or_warn()
    result2 = require_conda_env(exit_on_fail=False)

    del os.environ["CONTAINER"]

    assert result1 == True, "check_conda_env_or_warn should return True in container"
    assert result2 == True, "require_conda_env should return True in container"

    print("✓ Conda checks correctly skipped in container mode")

def main():
    print("="*60)
    print("Phase 1B: Container-Aware Code Tests")
    print("="*60)

    try:
        test_container_detection()
        test_path_env_overrides()
        test_conda_skip_in_container()

        print("\n" + "="*60)
        print("✓ All Phase 1B tests passed!")
        print("="*60 + "\n")
        return 0
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}\n", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}\n", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
