#!/usr/bin/env python3
"""Diagnostic script for VHS_LoadImagesPath issues.

Run from host or inside container to diagnose why source frames are not visible.

Usage:
    python scripts/diagnose_vhs_paths.py <project_dir>
    python scripts/diagnose_vhs_paths.py ../vfx_projects/MONK0001_pl01_ref_v001
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

CONTAINER_NAME = "vfx-ingest-interactive"
REPO_ROOT = Path(__file__).resolve().parent.parent


def print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def check_environment():
    print_section("Environment")
    print(f"Running as UID: {os.getuid()}")
    print(f"Running as GID: {os.getgid()}")
    print(f"HOME: {os.environ.get('HOME', 'NOT SET')}")
    print(f"CONTAINER: {os.environ.get('CONTAINER', 'NOT SET')}")
    print(f"CWD: {os.getcwd()}")

    in_container = os.environ.get('CONTAINER') == 'true'
    print(f"In container: {in_container}")
    return in_container


def find_project_dir():
    print_section("Finding Project Directory")

    candidates = [
        Path("/workspace/projects"),
        Path.cwd(),
        Path.cwd().parent,
    ]

    for candidate in candidates:
        if candidate.exists():
            projects = list(candidate.glob("*/source/frames"))
            if projects:
                project_dir = projects[0].parent.parent
                print(f"Found project: {project_dir}")
                return project_dir

    print("ERROR: Could not find project directory with source/frames")
    return None


def check_source_frames(project_dir: Path):
    print_section("Source Frames Check")

    frames_dir = project_dir / "source" / "frames"
    print(f"Frames directory: {frames_dir}")
    print(f"  Exists: {frames_dir.exists()}")

    if not frames_dir.exists():
        print("  ERROR: frames directory does not exist!")
        return

    print(f"  Is symlink: {frames_dir.is_symlink()}")
    if frames_dir.is_symlink():
        print(f"  Symlink target: {frames_dir.resolve()}")

    # Check permissions
    stat = frames_dir.stat()
    print(f"  Owner UID: {stat.st_uid}")
    print(f"  Owner GID: {stat.st_gid}")
    print(f"  Mode: {oct(stat.st_mode)}")

    # List files
    try:
        files = sorted(frames_dir.iterdir())
        print(f"  File count: {len(files)}")
        if files:
            print(f"  First 5 files:")
            for f in files[:5]:
                fstat = f.stat()
                print(f"    {f.name} (UID:{fstat.st_uid}, GID:{fstat.st_gid}, {oct(fstat.st_mode)})")
            if len(files) > 5:
                print(f"    ... and {len(files) - 5} more")
    except PermissionError as e:
        print(f"  ERROR: Permission denied listing files: {e}")


def check_comfyui_input(project_dir: Path):
    print_section("ComfyUI Input Directory")

    comfyui_input = Path("/app/.vfx_pipeline/ComfyUI/input")
    if not comfyui_input.exists():
        comfyui_input = Path.home() / ".vfx_pipeline" / "ComfyUI" / "input"

    print(f"ComfyUI input dir: {comfyui_input}")
    print(f"  Exists: {comfyui_input.exists()}")

    if comfyui_input.exists():
        stat = comfyui_input.stat()
        print(f"  Owner UID: {stat.st_uid}")
        print(f"  Owner GID: {stat.st_gid}")
        print(f"  Mode: {oct(stat.st_mode)}")

        # Check for project symlink
        project_link = comfyui_input / project_dir.name
        print(f"  Project symlink: {project_link}")
        print(f"    Exists: {project_link.exists()}")
        if project_link.is_symlink():
            print(f"    Target: {project_link.resolve()}")


def check_workflow(project_dir: Path):
    print_section("Workflow Analysis")

    workflow_file = project_dir / "comfyui" / "05_interactive_Segmentation.json"
    if not workflow_file.exists():
        print(f"Workflow not found: {workflow_file}")
        return

    print(f"Workflow file: {workflow_file}")

    try:
        with open(workflow_file) as f:
            workflow = json.load(f)

        # Find VHS_LoadImagesPath nodes
        for node_id, node in workflow.items():
            if not isinstance(node, dict):
                continue
            class_type = node.get("class_type", "")
            if "LoadImages" in class_type or "VHS" in class_type:
                print(f"\nNode {node_id}: {class_type}")
                inputs = node.get("inputs", {})
                for key, value in inputs.items():
                    if "path" in key.lower() or "dir" in key.lower():
                        print(f"  {key}: {value}")
                        # Check if path exists
                        if isinstance(value, str) and value:
                            p = Path(value)
                            print(f"    Path exists: {p.exists()}")
                            if p.exists():
                                files = list(p.glob("*"))
                                print(f"    Files found: {len(files)}")
    except Exception as e:
        print(f"Error reading workflow: {e}")


def check_vhs_node_expectations():
    print_section("VHS Node Path Expectations")

    # Try to find VHS node code
    vhs_paths = [
        Path("/app/.vfx_pipeline/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite"),
        Path.home() / ".vfx_pipeline" / "ComfyUI" / "custom_nodes" / "ComfyUI-VideoHelperSuite",
    ]

    for vhs_path in vhs_paths:
        if vhs_path.exists():
            print(f"VHS installed at: {vhs_path}")

            # Look for LoadImagesPath implementation
            load_images = vhs_path / "videohelpersuite" / "load_images.py"
            if load_images.exists():
                print(f"  load_images.py found")
                # Search for path handling
                with open(load_images) as f:
                    content = f.read()
                    if "directory" in content.lower():
                        print("  Contains directory handling")
                    if "input" in content.lower():
                        print("  References 'input' (likely ComfyUI input folder)")
            break
    else:
        print("VHS custom node not found")


def check_docker_mounts():
    print_section("Docker Mount Points (if in container)")

    mounts = ["/workspace", "/models", "/app"]
    for mount in mounts:
        p = Path(mount)
        print(f"\n{mount}:")
        print(f"  Exists: {p.exists()}")
        if p.exists():
            stat = p.stat()
            print(f"  Owner UID: {stat.st_uid}")
            print(f"  Owner GID: {stat.st_gid}")
            try:
                contents = list(p.iterdir())[:5]
                print(f"  Contents: {[c.name for c in contents]}")
            except PermissionError:
                print("  ERROR: Permission denied")


def test_file_access(project_dir: Path):
    print_section("File Access Test")

    frames_dir = project_dir / "source" / "frames"
    if not frames_dir.exists():
        print("Frames directory doesn't exist, skipping")
        return

    try:
        files = list(frames_dir.glob("*.png")) or list(frames_dir.glob("*.jpg")) or list(frames_dir.glob("*.exr"))
        if not files:
            print("No image files found")
            return

        test_file = files[0]
        print(f"Testing file: {test_file}")

        # Try to read first bytes
        try:
            with open(test_file, 'rb') as f:
                data = f.read(100)
            print(f"  Read test: SUCCESS (read {len(data)} bytes)")
        except PermissionError:
            print("  Read test: FAILED (Permission denied)")
        except Exception as e:
            print(f"  Read test: FAILED ({e})")

    except Exception as e:
        print(f"Error testing file access: {e}")


def is_container_running() -> bool:
    """Check if the diagnostic container is running."""
    try:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER_NAME],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip() == "true"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def start_container_for_diagnostics(project_dir: Path, models_dir: Path) -> bool:
    """Start a container for running diagnostics."""
    print(f"Starting container for diagnostics...")
    print(f"  Project: {project_dir}")
    print(f"  Models: {models_dir}")

    env = os.environ.copy()
    env["HOST_UID"] = str(os.getuid())
    env["HOST_GID"] = str(os.getgid())
    env["VFX_PROJECTS_DIR"] = str(project_dir.parent)
    env["VFX_MODELS_DIR"] = str(models_dir)

    cmd = [
        "docker", "compose",
        "-f", str(REPO_ROOT / "docker-compose.yml"),
        "run", "--rm", "-d",
        "--name", CONTAINER_NAME,
        "-v", f"{project_dir.parent}:/workspace/projects",
        "-v", f"{models_dir}:/models:ro",
        "vfx-ingest",
        "sleep", "300"  # Keep alive for 5 minutes
    ]

    try:
        result = subprocess.run(cmd, cwd=REPO_ROOT, env=env,
                                capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"Error starting container: {result.stderr}")
            return False
        print(f"  Container started: {result.stdout.strip()[:12]}")
        return True
    except subprocess.TimeoutExpired:
        print("Timeout starting container")
        return False


def stop_container():
    """Stop the diagnostic container."""
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME],
                   capture_output=True, timeout=10)


def run_in_container(project_name: str) -> str:
    """Run diagnostics inside the container."""
    container_project = f"/workspace/projects/{project_name}"

    cmd = [
        "docker", "exec", CONTAINER_NAME,
        "python3", "/app/scripts/diagnose_vhs_paths.py",
        "--in-container", container_project
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return "Timeout running diagnostics"


def find_models_dir() -> Path:
    """Find the models directory."""
    candidates = [
        REPO_ROOT / ".vfx_pipeline" / "models",
        REPO_ROOT.parent / "models",
        Path.home() / ".vfx_pipeline" / "models",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def run_from_host(project_dir: Path):
    """Run diagnostics from the host by starting a container."""
    print_section("Host-Side Diagnostics")

    project_dir = project_dir.resolve()
    print(f"Project directory: {project_dir}")
    print(f"  Exists: {project_dir.exists()}")

    frames_dir = project_dir / "source" / "frames"
    print(f"Source frames: {frames_dir}")
    print(f"  Exists: {frames_dir.exists()}")

    if frames_dir.exists():
        files = list(frames_dir.glob("*"))
        print(f"  File count: {len(files)}")
        if files:
            stat = files[0].stat()
            print(f"  First file owner: UID={stat.st_uid}, GID={stat.st_gid}")
            print(f"  Current user: UID={os.getuid()}, GID={os.getgid()}")

    # Check prepared workflow
    workflow_file = project_dir / "comfyui" / "05_interactive_segmentation.json"
    if workflow_file.exists():
        print(f"\nPrepared workflow: {workflow_file}")
        try:
            with open(workflow_file) as f:
                workflow = json.load(f)
            for node in workflow.get("nodes", []):
                if node.get("type") == "VHS_LoadImagesPath":
                    widgets = node.get("widgets_values", [])
                    if widgets:
                        print(f"  VHS_LoadImagesPath directory: {widgets[0]}")
        except Exception as e:
            print(f"  Error reading: {e}")
    else:
        print(f"\nNo prepared workflow found at {workflow_file}")

    print_section("Container Diagnostics")

    models_dir = find_models_dir()
    container_was_running = is_container_running()

    if not container_was_running:
        if not start_container_for_diagnostics(project_dir, models_dir):
            print("Failed to start container for diagnostics")
            return

    try:
        output = run_in_container(project_dir.name)
        print(output)
    finally:
        if not container_was_running:
            print("\nStopping diagnostic container...")
            stop_container()


def run_in_container_mode(project_dir: Path):
    """Run diagnostics inside the container."""
    print("VHS_LoadImagesPath Diagnostic Tool (Container Mode)")
    print("=" * 60)

    in_container = check_environment()

    if project_dir:
        check_source_frames(project_dir)
        check_comfyui_input(project_dir)
        check_workflow(project_dir)
        test_file_access(project_dir)

    check_vhs_node_expectations()

    if in_container:
        check_docker_mounts()

    print_section("Summary")
    print("""
Possible issues:
1. Path in workflow doesn't match actual file location
2. Files owned by different UID than container process
3. VHS expects files in ComfyUI/input, not direct path
4. Symlinks not resolving correctly in container
5. Directory permissions blocking listing
""")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose VHS_LoadImagesPath issues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/diagnose_vhs_paths.py ../vfx_projects/MONK0001_pl01_ref_v001
    python scripts/diagnose_vhs_paths.py /path/to/project
"""
    )
    parser.add_argument(
        "project_dir",
        type=Path,
        nargs="?",
        help="Project directory to diagnose"
    )
    parser.add_argument(
        "--in-container",
        action="store_true",
        help="Running inside container (internal use)"
    )

    args = parser.parse_args()

    if args.in_container:
        project_dir = Path(args.project_dir) if args.project_dir else find_project_dir()
        run_in_container_mode(project_dir)
    else:
        if not args.project_dir:
            print("Error: project_dir required when running from host")
            print("Usage: python scripts/diagnose_vhs_paths.py <project_dir>")
            sys.exit(1)
        run_from_host(args.project_dir)


if __name__ == "__main__":
    main()
