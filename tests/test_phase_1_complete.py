#!/usr/bin/env python3
"""Comprehensive tests for Roadmap 1 (Phases A, B, C, D, E)."""

import os
import sys
import subprocess
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

def test_imports():
    """Test that all modified modules import without errors."""
    print("\n=== Test: Python Module Imports ===")

    errors = []

    # Test env_config
    try:
        import env_config
        assert hasattr(env_config, 'is_in_container')
        assert hasattr(env_config, 'INSTALL_DIR')
        assert hasattr(env_config, 'DEFAULT_PROJECTS_DIR')
        print("✓ env_config imports and has required functions")
    except Exception as e:
        errors.append(f"env_config: {e}")

    # Test comfyui_manager
    try:
        import comfyui_manager
        assert hasattr(comfyui_manager, 'start_comfyui')
        print("✓ comfyui_manager imports successfully")
    except Exception as e:
        errors.append(f"comfyui_manager: {e}")

    # Test run_pipeline
    try:
        import run_pipeline
        assert hasattr(run_pipeline, 'run_pipeline')
        print("✓ run_pipeline imports successfully")
    except Exception as e:
        errors.append(f"run_pipeline: {e}")

    # Test verify_models
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        import verify_models
        assert hasattr(verify_models, 'check_model')
        assert hasattr(verify_models, 'REQUIRED_MODELS')
        print("✓ verify_models imports successfully")
    except Exception as e:
        errors.append(f"verify_models: {e}")

    if errors:
        print(f"\n✗ Import errors:")
        for err in errors:
            print(f"  - {err}")
        return False

    return True


def test_shell_scripts_syntax():
    """Test that shell scripts have valid syntax."""
    print("\n=== Test: Shell Script Syntax ===")

    scripts = [
        "scripts/download_models.sh",
        "scripts/run_docker.sh",
        "tests/fixtures/download_football.sh",
        "tests/integration/test_docker_build.sh",
        "docker/entrypoint.sh",
    ]

    errors = []
    for script in scripts:
        script_path = Path(__file__).parent.parent / script
        if not script_path.exists():
            errors.append(f"{script}: File not found")
            continue

        # Check bash syntax
        result = subprocess.run(
            ["bash", "-n", str(script_path)],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            errors.append(f"{script}: {result.stderr}")
        else:
            print(f"✓ {script} - valid syntax")

    if errors:
        print(f"\n✗ Shell script errors:")
        for err in errors:
            print(f"  - {err}")
        return False

    return True


def test_file_permissions():
    """Test that scripts have execute permissions."""
    print("\n=== Test: File Permissions ===")

    executables = [
        "scripts/download_models.sh",
        "scripts/verify_models.py",
        "scripts/run_docker.sh",
        "tests/fixtures/download_football.sh",
        "tests/integration/test_docker_build.sh",
        "docker/entrypoint.sh",
    ]

    errors = []
    for exe in executables:
        exe_path = Path(__file__).parent.parent / exe
        if not exe_path.exists():
            errors.append(f"{exe}: File not found")
            continue

        if not os.access(exe_path, os.X_OK):
            errors.append(f"{exe}: Not executable")
        else:
            print(f"✓ {exe} - executable")

    if errors:
        print(f"\n✗ Permission errors:")
        for err in errors:
            print(f"  - {err}")
        return False

    return True


def test_dockerfile_exists():
    """Test that Dockerfile and related files exist."""
    print("\n=== Test: Docker Files ===")

    files = [
        "Dockerfile",
        "docker-compose.yml",
        ".dockerignore",
        "docker/entrypoint.sh",
    ]

    errors = []
    for file in files:
        file_path = Path(__file__).parent.parent / file
        if not file_path.exists():
            errors.append(f"{file}: Not found")
        else:
            print(f"✓ {file} exists")

    if errors:
        print(f"\n✗ Missing files:")
        for err in errors:
            print(f"  - {err}")
        return False

    return True


def test_env_config_container_detection():
    """Test container detection logic."""
    print("\n=== Test: Container Detection Logic ===")

    import env_config
    import importlib

    # Test with CONTAINER env var
    original = os.environ.get("CONTAINER")

    os.environ["CONTAINER"] = "true"
    importlib.reload(env_config)

    if not env_config.is_in_container():
        print("✗ Should detect CONTAINER=true")
        if original:
            os.environ["CONTAINER"] = original
        else:
            del os.environ["CONTAINER"]
        return False

    print("✓ Detects CONTAINER=true correctly")

    # Cleanup
    if original:
        os.environ["CONTAINER"] = original
    else:
        del os.environ["CONTAINER"]

    importlib.reload(env_config)

    return True


def test_model_verification_logic():
    """Test model verification script logic."""
    print("\n=== Test: Model Verification Logic ===")

    import verify_models

    # Test REQUIRED_MODELS structure
    assert "sam3" in verify_models.REQUIRED_MODELS
    assert "videodepthanything" in verify_models.REQUIRED_MODELS
    assert "wham" in verify_models.REQUIRED_MODELS

    for name, config in verify_models.REQUIRED_MODELS.items():
        assert "path" in config
        assert "files" in config
        assert "description" in config
        assert isinstance(config["files"], list)
        assert len(config["files"]) > 0

    print("✓ REQUIRED_MODELS structure is valid")

    # Test check_model function
    fake_config = {
        "path": Path("/nonexistent"),
        "files": ["test.txt"],
    }

    ok, msg = verify_models.check_model("test", fake_config)
    assert not ok
    assert "not found" in msg.lower()

    print("✓ check_model() handles missing paths correctly")

    return True


def test_documentation_exists():
    """Test that documentation files exist."""
    print("\n=== Test: Documentation Files ===")

    docs = [
        "docs/README-DOCKER.md",
        "docs/ATLAS.md",
        "docs/ROADMAP-1-DOCKER.md",
    ]

    errors = []
    for doc in docs:
        doc_path = Path(__file__).parent.parent / doc
        if not doc_path.exists():
            errors.append(f"{doc}: Not found")
        else:
            # Check not empty
            size = doc_path.stat().st_size
            if size < 100:
                errors.append(f"{doc}: Too small ({size} bytes)")
            else:
                print(f"✓ {doc} exists ({size:,} bytes)")

    if errors:
        print(f"\n✗ Documentation errors:")
        for err in errors:
            print(f"  - {err}")
        return False

    return True


def main():
    """Run all tests."""
    print("="*60)
    print("Roadmap 1 - Comprehensive Test Suite")
    print("Testing Phases A, B, C, D, E")
    print("="*60)

    tests = [
        ("Python Imports", test_imports),
        ("Shell Script Syntax", test_shell_scripts_syntax),
        ("File Permissions", test_file_permissions),
        ("Docker Files", test_dockerfile_exists),
        ("Container Detection", test_env_config_container_detection),
        ("Model Verification", test_model_verification_logic),
        ("Documentation", test_documentation_exists),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ {name} raised exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8s} {name}")

    total = len(results)
    passed_count = sum(1 for _, p in results if p)

    print("="*60)
    print(f"Results: {passed_count}/{total} tests passed")

    if passed_count == total:
        print("\n✓ All tests passed! Roadmap 1 implementation complete.")
        return 0
    else:
        print(f"\n✗ {total - passed_count} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
