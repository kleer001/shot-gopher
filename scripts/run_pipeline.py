#!/usr/bin/env python3
"""Automated VFX pipeline runner.

Single command to process footage through the entire pipeline:
  Movie file → Frame extraction → ComfyUI workflows → Post-processing

Usage:
    python run_pipeline.py <input_movie> [options]

Example:
    python run_pipeline.py /path/to/footage.mp4 --name "My_Shot" --stages all
    python run_pipeline.py /path/to/footage.mp4 --stages depth,camera
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

# Pipeline configuration
DEFAULT_PROJECTS_DIR = Path.home() / "vfx_projects"
DEFAULT_COMFYUI_URL = "http://127.0.0.1:8188"
START_FRAME = 1001
SUPPORTED_FORMATS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mxf", ".exr", ".dpx", ".jpg", ".png"}

# Stage definitions
STAGES = {
    "ingest": "Extract frames from movie",
    "depth": "Run depth analysis (01_analysis.json)",
    "roto": "Run segmentation (02_segmentation.json)",
    "cleanplate": "Run clean plate generation (03_cleanplate.json)",
    "colmap": "Run COLMAP SfM reconstruction",
    "mocap": "Run human motion capture (WHAM + TAVA)",
    "gsir": "Run GS-IR material decomposition",
    "camera": "Export camera to Alembic",
}


def run_command(cmd: list[str], description: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command with logging."""
    print(f"  → {description}")
    print(f"    $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"    Error: {result.stderr}", file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result


def extract_frames(
    input_path: Path,
    output_dir: Path,
    start_frame: int = START_FRAME,
    fps: Optional[float] = None
) -> int:
    """Extract frames from video file using ffmpeg.

    Returns:
        Number of frames extracted
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = output_dir / "frame_%04d.png"

    cmd = ["ffmpeg", "-i", str(input_path)]

    if fps:
        cmd.extend(["-vf", f"fps={fps}"])

    cmd.extend([
        "-start_number", str(start_frame),
        "-q:v", "2",  # High quality
        str(output_pattern),
        "-y"  # Overwrite
    ])

    run_command(cmd, f"Extracting frames to {output_dir}")

    # Count extracted frames
    frames = list(output_dir.glob("frame_*.png"))
    return len(frames)


def get_video_info(input_path: Path) -> dict:
    """Get video metadata using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(input_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {}
    return json.loads(result.stdout)


def check_comfyui_running(url: str) -> bool:
    """Check if ComfyUI server is accessible."""
    try:
        urllib.request.urlopen(f"{url}/system_stats", timeout=5)
        return True
    except (urllib.error.URLError, TimeoutError):
        return False


def convert_workflow_to_api_format(workflow: dict) -> dict:
    """Convert ComfyUI workflow format to API format if needed.

    Workflow format (saved from UI): {"nodes": [...], "links": [...]}
    API format (for /prompt): {"1": {"class_type": "...", "inputs": {...}}, ...}

    Note: Full conversion from workflow format requires node type definitions.
    Workflow templates should be saved in API format directly.

    Args:
        workflow: Workflow dict (either format)

    Returns:
        Workflow in API format for /prompt endpoint
    """
    # If already in API format (no "nodes" key), return as-is
    if "nodes" not in workflow:
        return workflow

    # Workflow format detected - attempt basic conversion
    # Note: This only handles linked inputs, not widget values
    print("  Warning: Workflow in UI format, attempting conversion...", file=sys.stderr)
    print("  For reliable execution, save workflows in API format", file=sys.stderr)

    nodes = workflow.get("nodes", [])
    links = workflow.get("links", [])

    # Build link lookup: link_id -> (source_node_id, source_slot)
    link_lookup = {}
    for link in links:
        # Link format: [link_id, source_node_id, source_slot, dest_node_id, dest_slot, type]
        if len(link) >= 5:
            link_id, src_node, src_slot, dst_node, dst_slot = link[:5]
            link_lookup[link_id] = (src_node, src_slot)

    api_workflow = {}

    for node in nodes:
        node_id = str(node.get("id"))
        node_type = node.get("type")

        # Skip special nodes that don't execute
        if node_type in ("Note", "Reroute"):
            continue

        inputs = {}

        # Process linked inputs only
        node_inputs = node.get("inputs", [])
        for inp in node_inputs:
            inp_name = inp.get("name")
            link_id = inp.get("link")

            if link_id is not None and link_id in link_lookup:
                src_node, src_slot = link_lookup[link_id]
                inputs[inp_name] = [str(src_node), src_slot]

        api_workflow[node_id] = {
            "class_type": node_type,
            "inputs": inputs
        }

        if node.get("title"):
            api_workflow[node_id]["_meta"] = {"title": node.get("title")}

    return api_workflow


def queue_workflow(workflow_path: Path, comfyui_url: str) -> str:
    """Queue a workflow for execution via ComfyUI API.

    Returns:
        Prompt ID for tracking
    """
    with open(workflow_path) as f:
        workflow = json.load(f)

    # Convert workflow format to API format if needed
    api_workflow = convert_workflow_to_api_format(workflow)

    # Wrap workflow in prompt format
    prompt_data = {"prompt": api_workflow}

    data = json.dumps(prompt_data).encode('utf-8')
    req = urllib.request.Request(
        f"{comfyui_url}/prompt",
        data=data,
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode())
        return result.get("prompt_id", "")


def wait_for_completion(prompt_id: str, comfyui_url: str, timeout: int = 3600) -> bool:
    """Wait for a queued workflow to complete.

    Returns:
        True if completed successfully, False otherwise
    """
    start_time = time.time()
    check_interval = 2  # seconds

    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(f"{comfyui_url}/history/{prompt_id}") as response:
                history = json.loads(response.read().decode())

                if prompt_id in history:
                    status = history[prompt_id].get("status", {})
                    if status.get("completed", False):
                        return True
                    if status.get("status_str") == "error":
                        print(f"    Workflow error: {status}", file=sys.stderr)
                        return False

        except urllib.error.URLError:
            pass

        time.sleep(check_interval)

    print("    Timeout waiting for workflow completion", file=sys.stderr)
    return False


def run_comfyui_workflow(
    workflow_path: Path,
    comfyui_url: str,
    wait: bool = True,
    timeout: int = 3600
) -> bool:
    """Run a ComfyUI workflow via API.

    Returns:
        True if successful
    """
    if not check_comfyui_running(comfyui_url):
        print(f"    Error: ComfyUI not running at {comfyui_url}", file=sys.stderr)
        print("    Start ComfyUI first: python main.py --listen", file=sys.stderr)
        return False

    print(f"  → Queuing workflow: {workflow_path.name}")
    prompt_id = queue_workflow(workflow_path, comfyui_url)

    if not prompt_id:
        print("    Error: Failed to queue workflow", file=sys.stderr)
        return False

    print(f"    Prompt ID: {prompt_id}")

    if wait:
        print("    Waiting for completion...")
        return wait_for_completion(prompt_id, comfyui_url, timeout)

    return True


def run_export_camera(project_dir: Path, fps: float = 24.0) -> bool:
    """Run camera export script."""
    script_path = Path(__file__).parent / "export_camera.py"

    if not script_path.exists():
        print("    Error: export_camera.py not found", file=sys.stderr)
        return False

    cmd = [
        sys.executable, str(script_path),
        str(project_dir),
        "--fps", str(fps),
        "--format", "both"
    ]

    try:
        run_command(cmd, "Exporting camera data")
        return True
    except subprocess.CalledProcessError:
        return False


def run_colmap_reconstruction(
    project_dir: Path,
    quality: str = "medium",
    run_dense: bool = False,
    run_mesh: bool = False,
    use_masks: bool = True
) -> bool:
    """Run COLMAP Structure-from-Motion reconstruction.

    Args:
        project_dir: Project directory containing source/frames/
        quality: Quality preset ('low', 'medium', 'high')
        run_dense: Whether to run dense reconstruction
        run_mesh: Whether to generate mesh
        use_masks: If True, use segmentation masks from roto/ (if available)

    Returns:
        True if reconstruction succeeded
    """
    script_path = Path(__file__).parent / "run_colmap.py"

    if not script_path.exists():
        print("    Error: run_colmap.py not found", file=sys.stderr)
        return False

    cmd = [
        sys.executable, str(script_path),
        str(project_dir),
        "--quality", quality,
    ]

    if run_dense:
        cmd.append("--dense")
    if run_mesh:
        cmd.append("--mesh")
    if not use_masks:
        cmd.append("--no-masks")

    try:
        run_command(cmd, "Running COLMAP reconstruction")
        return True
    except subprocess.CalledProcessError:
        return False


def run_mocap(
    project_dir: Path,
    skip_texture: bool = False,
    keyframe_interval: int = 25,
    fps: float = 24.0
) -> bool:
    """Run human motion capture with WHAM + TAVA.

    Args:
        project_dir: Project directory with frames and camera data
        skip_texture: Skip texture projection (faster)
        keyframe_interval: ECON keyframe interval
        fps: Frame rate for export

    Returns:
        True if mocap succeeded
    """
    script_path = Path(__file__).parent / "run_mocap.py"

    if not script_path.exists():
        print("    Error: run_mocap.py not found", file=sys.stderr)
        return False

    cmd = [
        sys.executable, str(script_path),
        str(project_dir),
        "--keyframe-interval", str(keyframe_interval),
        "--fps", str(fps),
    ]

    if skip_texture:
        cmd.append("--skip-texture")

    try:
        run_command(cmd, "Running motion capture")
        return True
    except subprocess.CalledProcessError:
        return False


def run_gsir_materials(
    project_dir: Path,
    iterations_stage1: int = 30000,
    iterations_stage2: int = 35000,
    gsir_path: Optional[str] = None
) -> bool:
    """Run GS-IR material decomposition.

    Args:
        project_dir: Project directory with COLMAP output
        iterations_stage1: Training iterations for stage 1
        iterations_stage2: Total training iterations
        gsir_path: Path to GS-IR installation

    Returns:
        True if material decomposition succeeded
    """
    script_path = Path(__file__).parent / "run_gsir.py"

    if not script_path.exists():
        print("    Error: run_gsir.py not found", file=sys.stderr)
        return False

    cmd = [
        sys.executable, str(script_path),
        str(project_dir),
        "--iterations-stage1", str(iterations_stage1),
        "--iterations-stage2", str(iterations_stage2),
    ]

    if gsir_path:
        cmd.extend(["--gsir-path", gsir_path])

    try:
        run_command(cmd, "Running GS-IR material decomposition")
        return True
    except subprocess.CalledProcessError:
        return False


def setup_project(
    project_dir: Path,
    workflows_dir: Path
) -> bool:
    """Set up project structure and populate workflows."""
    script_path = Path(__file__).parent / "setup_project.py"

    cmd = [
        sys.executable, str(script_path),
        str(project_dir),
        "--workflows-dir", str(workflows_dir)
    ]

    try:
        run_command(cmd, "Setting up project structure")
        return True
    except subprocess.CalledProcessError:
        return False


def run_pipeline(
    input_path: Path,
    project_name: Optional[str] = None,
    projects_dir: Path = DEFAULT_PROJECTS_DIR,
    stages: list[str] = None,
    comfyui_url: str = DEFAULT_COMFYUI_URL,
    fps: Optional[float] = None,
    skip_existing: bool = False,
    colmap_quality: str = "medium",
    colmap_dense: bool = False,
    colmap_mesh: bool = False,
    colmap_use_masks: bool = True,
    gsir_iterations: int = 35000,
    gsir_path: Optional[str] = None,
) -> bool:
    """Run the full VFX pipeline.

    Args:
        input_path: Input movie file
        project_name: Project name (default: derived from filename)
        projects_dir: Parent directory for projects
        stages: Which stages to run (default: all)
        comfyui_url: ComfyUI API URL
        fps: Override frame rate
        skip_existing: Skip stages with existing output
        colmap_quality: COLMAP quality preset ('low', 'medium', 'high')
        colmap_dense: Run COLMAP dense reconstruction
        colmap_mesh: Generate mesh from COLMAP dense reconstruction
        colmap_use_masks: Use segmentation masks for COLMAP (if available)
        gsir_iterations: Total GS-IR training iterations
        gsir_path: Path to GS-IR installation

    Returns:
        True if all stages successful
    """
    stages = stages or list(STAGES.keys())

    # Derive project name from input if not specified
    if not project_name:
        project_name = input_path.stem.replace(" ", "_")

    project_dir = projects_dir / project_name
    source_frames = project_dir / "source" / "frames"
    workflows_dir = Path(__file__).parent.parent / "workflow_templates"

    print(f"\n{'='*60}")
    print(f"VFX Pipeline: {project_name}")
    print(f"{'='*60}")
    print(f"Input: {input_path}")
    print(f"Project: {project_dir}")
    print(f"Stages: {', '.join(stages)}")
    print()

    # Get video info for fps
    if not fps and input_path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mxf"}:
        info = get_video_info(input_path)
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video":
                fps_str = stream.get("r_frame_rate", "24/1")
                if "/" in fps_str:
                    num, den = fps_str.split("/")
                    fps = float(num) / float(den)
                else:
                    fps = float(fps_str)
                break
    fps = fps or 24.0
    print(f"Frame rate: {fps} fps")

    # Stage: Setup
    print("\n[Setup]")
    if not setup_project(project_dir, workflows_dir):
        print("Failed to set up project", file=sys.stderr)
        return False

    # Stage: Ingest
    if "ingest" in stages:
        print("\n[Ingest]")
        if skip_existing and list(source_frames.glob("frame_*.png")):
            print("  → Skipping (frames exist)")
        else:
            frame_count = extract_frames(input_path, source_frames, START_FRAME, fps)
            print(f"  → Extracted {frame_count} frames")

    # Stage: Depth
    if "depth" in stages:
        print("\n[Depth Analysis]")
        workflow_path = project_dir / "workflows" / "01_analysis.json"
        if not workflow_path.exists():
            print("  → Skipping (workflow not found)")
        elif skip_existing and list((project_dir / "depth").glob("*.png")):
            print("  → Skipping (depth maps exist)")
        else:
            if not run_comfyui_workflow(workflow_path, comfyui_url):
                print("  → Depth stage failed", file=sys.stderr)
                return False

    # Stage: Roto
    if "roto" in stages:
        print("\n[Segmentation]")
        workflow_path = project_dir / "workflows" / "02_segmentation.json"
        if not workflow_path.exists():
            print("  → Skipping (workflow not found)")
        elif skip_existing and list((project_dir / "roto").glob("*.png")):
            print("  → Skipping (masks exist)")
        else:
            if not run_comfyui_workflow(workflow_path, comfyui_url):
                print("  → Segmentation stage failed", file=sys.stderr)
                return False

    # Stage: Cleanplate
    if "cleanplate" in stages:
        print("\n[Clean Plate]")
        workflow_path = project_dir / "workflows" / "03_cleanplate.json"
        if not workflow_path.exists():
            print("  → Skipping (workflow not found)")
        elif skip_existing and list((project_dir / "cleanplate").glob("*.png")):
            print("  → Skipping (cleanplates exist)")
        else:
            if not run_comfyui_workflow(workflow_path, comfyui_url):
                print("  → Cleanplate stage failed", file=sys.stderr)
                return False

    # Stage: COLMAP reconstruction
    if "colmap" in stages:
        print("\n[COLMAP Reconstruction]")
        colmap_sparse = project_dir / "colmap" / "sparse" / "0"
        if skip_existing and colmap_sparse.exists():
            print("  → Skipping (COLMAP sparse model exists)")
        else:
            if not run_colmap_reconstruction(
                project_dir,
                quality=colmap_quality,
                run_dense=colmap_dense,
                run_mesh=colmap_mesh,
                use_masks=colmap_use_masks
            ):
                print("  → COLMAP reconstruction failed", file=sys.stderr)
                # Non-fatal for pipeline - continue to camera export
                # (may use DA3 camera data as fallback)

    # Stage: Motion capture
    if "mocap" in stages:
        print("\n[Motion Capture]")
        mocap_output = project_dir / "mocap" / "tava" / "mesh_sequence.pkl"
        camera_dir = project_dir / "camera"
        if not camera_dir.exists() or not (camera_dir / "extrinsics.json").exists():
            print("  → Skipping (camera data required - run colmap stage first)")
        elif skip_existing and mocap_output.exists():
            print("  → Skipping (mocap data exists)")
        else:
            if not run_mocap(
                project_dir,
                skip_texture=False,  # Could add as pipeline option
                keyframe_interval=25,
                fps=fps
            ):
                print("  → Motion capture failed", file=sys.stderr)
                # Non-fatal - continue to other stages

    # Stage: GS-IR material decomposition
    if "gsir" in stages:
        print("\n[GS-IR Material Decomposition]")
        colmap_sparse = project_dir / "colmap" / "sparse" / "0"
        gsir_checkpoint = project_dir / "gsir" / "model" / f"chkpnt{gsir_iterations}.pth"
        if not colmap_sparse.exists():
            print("  → Skipping (COLMAP reconstruction required first)")
        elif skip_existing and gsir_checkpoint.exists():
            print("  → Skipping (GS-IR checkpoint exists)")
        else:
            if not run_gsir_materials(
                project_dir,
                iterations_stage1=30000,
                iterations_stage2=gsir_iterations,
                gsir_path=gsir_path
            ):
                print("  → GS-IR material decomposition failed", file=sys.stderr)
                # Non-fatal - continue to camera export

    # Stage: Camera export
    if "camera" in stages:
        print("\n[Camera Export]")
        camera_dir = project_dir / "camera"
        if not (camera_dir / "extrinsics.json").exists():
            print("  → Skipping (no camera data - run depth or colmap stage first)")
        elif skip_existing and (camera_dir / "camera.abc").exists():
            print("  → Skipping (camera.abc exists)")
        else:
            if not run_export_camera(project_dir, fps):
                print("  → Camera export failed", file=sys.stderr)
                # Non-fatal - continue

    print(f"\n{'='*60}")
    print(f"Pipeline complete: {project_dir}")
    print(f"{'='*60}\n")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Automated VFX pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input movie file or image sequence"
    )
    parser.add_argument(
        "--name", "-n",
        type=str,
        default=None,
        help="Project name (default: derived from input filename)"
    )
    parser.add_argument(
        "--projects-dir", "-p",
        type=Path,
        default=DEFAULT_PROJECTS_DIR,
        help=f"Projects directory (default: {DEFAULT_PROJECTS_DIR})"
    )
    parser.add_argument(
        "--stages", "-s",
        type=str,
        default="all",
        help=f"Comma-separated stages to run: {','.join(STAGES.keys())} or 'all'"
    )
    parser.add_argument(
        "--comfyui-url", "-c",
        type=str,
        default=DEFAULT_COMFYUI_URL,
        help=f"ComfyUI API URL (default: {DEFAULT_COMFYUI_URL})"
    )
    parser.add_argument(
        "--fps", "-f",
        type=float,
        default=None,
        help="Override frame rate (default: auto-detect)"
    )
    parser.add_argument(
        "--skip-existing", "-e",
        action="store_true",
        help="Skip stages that have existing output"
    )
    parser.add_argument(
        "--list-stages", "-l",
        action="store_true",
        help="List available stages and exit"
    )

    # COLMAP options
    parser.add_argument(
        "--colmap-quality", "-q",
        choices=["low", "medium", "high"],
        default="medium",
        help="COLMAP quality preset (default: medium)"
    )
    parser.add_argument(
        "--colmap-dense", "-d",
        action="store_true",
        help="Run COLMAP dense reconstruction (slower, produces point cloud)"
    )
    parser.add_argument(
        "--colmap-mesh", "-m",
        action="store_true",
        help="Generate mesh from COLMAP dense reconstruction (requires --colmap-dense)"
    )
    parser.add_argument(
        "--colmap-no-masks", "-M",
        action="store_true",
        help="Disable automatic use of segmentation masks for COLMAP (default: use masks if available)"
    )

    # GS-IR options
    parser.add_argument(
        "--gsir-iterations", "-i",
        type=int,
        default=35000,
        help="GS-IR total training iterations (default: 35000)"
    )
    parser.add_argument(
        "--gsir-path", "-g",
        type=str,
        default=None,
        help="Path to GS-IR installation (default: auto-detect)"
    )

    args = parser.parse_args()

    if args.list_stages:
        print("Available stages:")
        for name, desc in STAGES.items():
            print(f"  {name}: {desc}")
        sys.exit(0)

    # Validate input
    input_path = args.input.resolve()
    if not input_path.exists():
        print(f"Error: Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Parse stages
    if args.stages.lower() == "all":
        stages = list(STAGES.keys())
    else:
        stages = [s.strip() for s in args.stages.split(",")]
        invalid = set(stages) - set(STAGES.keys())
        if invalid:
            print(f"Error: Invalid stages: {invalid}", file=sys.stderr)
            print(f"Valid stages: {', '.join(STAGES.keys())}")
            sys.exit(1)

    # Run pipeline
    success = run_pipeline(
        input_path=input_path,
        project_name=args.name,
        projects_dir=args.projects_dir,
        stages=stages,
        comfyui_url=args.comfyui_url,
        fps=args.fps,
        skip_existing=args.skip_existing,
        colmap_quality=args.colmap_quality,
        colmap_dense=args.colmap_dense,
        colmap_mesh=args.colmap_mesh,
        colmap_use_masks=not args.colmap_no_masks,
        gsir_iterations=args.gsir_iterations,
        gsir_path=args.gsir_path,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
