#!/usr/bin/env python3
"""Diagnostic script for VHS_LoadImagesPath issues.

Starts a container, runs diagnostics inside, and reports findings.

Usage:
    python scripts/diagnose_vhs_paths.py <project_dir>
    python scripts/diagnose_vhs_paths.py ../vfx_projects/MONK0001_pl01_ref_v001
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CONTAINER_NAME = "vfx-diag-temp"
REPO_ROOT = Path(__file__).resolve().parent.parent


def print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f" {title}")
    print("=" * 60)


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


def stop_container():
    """Stop and remove the diagnostic container."""
    subprocess.run(
        ["docker", "rm", "-f", CONTAINER_NAME],
        capture_output=True,
        timeout=10
    )


def start_container(project_dir: Path, models_dir: Path) -> bool:
    """Start a temporary container for diagnostics."""
    print("Starting diagnostic container...")

    stop_container()

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
        "--entrypoint", "sleep",
        "-v", f"{project_dir.parent}:/workspace/projects",
        "-v", f"{models_dir}:/models:ro",
        "vfx-ingest",
        "600"
    ]

    try:
        result = subprocess.run(
            cmd, cwd=REPO_ROOT, env=env,
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return False
        print(f"  Container ID: {result.stdout.strip()[:12]}")
        time.sleep(2)
        return True
    except subprocess.TimeoutExpired:
        print("Timeout starting container")
        return False


def docker_exec(cmd: str) -> str:
    """Run a command inside the container."""
    try:
        result = subprocess.run(
            ["docker", "exec", CONTAINER_NAME, "bash", "-c", cmd],
            capture_output=True, text=True, timeout=30
        )
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return "TIMEOUT"


def run_diagnostics(project_dir: Path):
    """Run all diagnostics."""
    project_name = project_dir.name
    container_project = f"/workspace/projects/{project_name}"
    container_frames = f"{container_project}/source/frames"

    print_section("1. Host-Side Check")
    print(f"Project: {project_dir}")
    print(f"  Exists: {project_dir.exists()}")

    frames_dir = project_dir / "source" / "frames"
    print(f"Frames dir: {frames_dir}")
    print(f"  Exists: {frames_dir.exists()}")

    if frames_dir.exists():
        files = list(frames_dir.glob("*"))
        print(f"  File count: {len(files)}")
        if files:
            stat = files[0].stat()
            print(f"  File owner: UID={stat.st_uid}, GID={stat.st_gid}")
    print(f"Current user: UID={os.getuid()}, GID={os.getgid()}")

    workflow_file = project_dir / "comfyui" / "05_interactive_segmentation.json"
    if workflow_file.exists():
        print(f"\nWorkflow: {workflow_file}")
        try:
            with open(workflow_file) as f:
                workflow = json.load(f)
            for node in workflow.get("nodes", []):
                if node.get("type") == "VHS_LoadImagesPath":
                    widgets = node.get("widgets_values", [])
                    if widgets:
                        print(f"  VHS path in workflow: {widgets[0]}")
        except Exception as e:
            print(f"  Error: {e}")
    else:
        print(f"\nNo workflow at: {workflow_file}")

    print_section("2. Container Environment")
    print(docker_exec("echo \"UID: $(id -u), GID: $(id -g)\""))
    print(docker_exec("echo \"HOME: $HOME\""))
    print(docker_exec("echo \"CONTAINER: $CONTAINER\""))

    print_section("3. Container View of Project")
    print(f"Checking: {container_project}")
    print(docker_exec(f"ls -la {container_project}/ 2>&1 | head -10"))

    print_section("4. Container View of Frames")
    print(f"Checking: {container_frames}")
    output = docker_exec(f"""
if [ -d "{container_frames}" ]; then
    echo "Directory exists: YES"
    echo "File count: $(ls -1 "{container_frames}" 2>/dev/null | wc -l)"
    echo "First 5 files:"
    ls -la "{container_frames}" 2>/dev/null | head -7
    echo ""
    echo "Testing read access..."
    FIRST_FILE=$(ls "{container_frames}"/*.png 2>/dev/null | head -1)
    if [ -n "$FIRST_FILE" ]; then
        if head -c 10 "$FIRST_FILE" >/dev/null 2>&1; then
            echo "Read test: SUCCESS"
        else
            echo "Read test: FAILED"
        fi
    else
        echo "No PNG files found"
    fi
else
    echo "Directory exists: NO"
    echo "Parent contents:"
    ls -la "{container_project}/source/" 2>&1
fi
""")
    print(output)

    print_section("5. ComfyUI Input Directory")
    print(docker_exec("""
echo "ComfyUI input dir: /app/.vfx_pipeline/ComfyUI/input"
ls -la /app/.vfx_pipeline/ComfyUI/input/ 2>&1 | head -10
"""))

    print_section("6. Workflow Path in Container")
    container_workflow = f"{container_project}/comfyui/05_interactive_segmentation.json"
    output = docker_exec(f"""
if [ -f "{container_workflow}" ]; then
    echo "Workflow exists: YES"
    echo "VHS_LoadImagesPath directory value:"
    python3 -c "
import json
with open('{container_workflow}') as f:
    w = json.load(f)
for n in w.get('nodes', []):
    if n.get('type') == 'VHS_LoadImagesPath':
        v = n.get('widgets_values', [])
        if v:
            print(f'  Path: {{v[0]}}')
            import os
            print(f'  Exists: {{os.path.isdir(v[0])}}')
            if os.path.isdir(v[0]):
                print(f'  Files: {{len(os.listdir(v[0]))}}')
"
else
    echo "Workflow exists: NO"
fi
""")
    print(output)

    print_section("7. Summary")
    print("""
If the path in the workflow points to the correct location and files exist
but VHS still cannot see them, the issue is likely:

1. The workflow was prepared with HOST paths, not CONTAINER paths
   - Workflow should use: /workspace/projects/<name>/source/frames
   - Not: /media/... or /home/...

2. Permission mismatch between container UID and file owner

3. VHS node caching old path - try refreshing the node in ComfyUI

To fix, delete the prepared workflow and let it regenerate:
  rm -rf <project>/comfyui/
  python scripts/launch_interactive_segmentation.py <project>
""")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose VHS_LoadImagesPath issues"
    )
    parser.add_argument(
        "project_dir",
        type=Path,
        help="Project directory to diagnose"
    )
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    if not project_dir.exists():
        print(f"Error: Project not found: {project_dir}")
        sys.exit(1)

    models_dir = find_models_dir()
    print(f"Models directory: {models_dir}")

    if not start_container(project_dir, models_dir):
        print("Failed to start container")
        sys.exit(1)

    try:
        run_diagnostics(project_dir)
    finally:
        print("\nCleaning up container...")
        stop_container()


if __name__ == "__main__":
    main()
