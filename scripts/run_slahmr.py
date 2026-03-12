#!/usr/bin/env python3
"""SLAHMR motion capture wrapper for the VFX pipeline.

Runs SLAHMR joint camera+body optimization on monocular video.
Produces motion.pkl (SMPLX format) and mocap_camera/ extrinsics.

SLAHMR outputs SMPL-H in motion chunks. This wrapper:
1. Finds or creates video from source frames
2. Sets up temp data dir with SLAHMR layout
3. Runs SLAHMR via conda run -p <prefix>
4. Stitches chunked output into a single NPZ
5. Exports camera (extrinsics.json, intrinsics.json)
6. Converts SMPL-H → SMPLX via betas fitting

Usage:
    python run_slahmr.py <project_dir> [--gender neutral] [--fps 24]

Output:
    mocap/<person>/motion.pkl      (SMPLX parameters)
    mocap_camera/extrinsics.json   (camera-to-world 4x4 matrices)
    mocap_camera/intrinsics.json   (fx, fy, cx, cy, width, height)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import re
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from env_config import (
    require_conda_env,
    SLAHMR_INSTALL_DIR,
    SLAHMR_CONDA_ENV,
    SLAHMR_CONDA_PREFIX,
    GVHMR_CONDA_PREFIX,
)
from log_manager import LogCapture


def _load_camera_metadata(project_dir: Path) -> Optional[dict]:
    """Load camera metadata from project source directory.

    Args:
        project_dir: Project directory.

    Returns:
        Metadata dict if found, None otherwise.
    """
    metadata_path = project_dir / "source" / "camera_metadata.json"
    if not metadata_path.exists():
        return None
    with open(metadata_path, encoding="utf-8") as f:
        return json.load(f)


def _create_slahmr_cameras_npz(
    output_path: Path,
    focal_px: float,
    n_frames: int,
    image_width: int,
    image_height: int,
) -> None:
    """Create a cameras.npz for SLAHMR with a fixed focal length.

    SLAHMR loads cameras.npz to initialize focal and camera poses.
    By providing identity w2c matrices with the correct focal, SLAHMR
    will start with the known focal and optimize body pose against it.

    Args:
        output_path: Path to write cameras.npz.
        focal_px: Focal length in pixels.
        n_frames: Number of frames.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    w2c = np.broadcast_to(np.eye(4, dtype=np.float32), (n_frames, 4, 4)).copy()
    np.savez(
        output_path,
        w2c=w2c,
        focal=np.float32(focal_px),
        width=np.int32(image_width),
        height=np.int32(image_height),
    )


def _find_conda() -> Optional[str]:
    """Find conda or mamba executable.

    Returns:
        Path to conda executable, or None if not found.
    """
    for cmd in ["conda", "mamba"]:
        if shutil.which(cmd):
            return cmd
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe and Path(conda_exe).exists():
        return str(conda_exe)
    return None


_SLAHMR_SUPPRESS = re.compile(
    r"^(WORLD_SCALE |FLOOR PLANE |Model saved at |Optimizer saved at |"
    r"start, end |Set param names:|saving params to |saving \d+ meshes to )"
)

_SLAHMR_ITER = re.compile(r"^ITER: (\d+)")

_FLOOR_PLANE_RE = re.compile(
    r"ESTIMATED FLOORS:\s*tensor\(\[\[([\d\s.,e+\-]+)\]\]"
)


def extract_floor_plane(slahmr_run_dir: Path) -> Optional[List[float]]:
    """Extract the last estimated floor plane from SLAHMR's opt_log.txt.

    SLAHMR's Logger writes 'ESTIMATED FLOORS: tensor([[a, b, c]])' to
    opt_log.txt in the Hydra run directory. The compact form stores
    normal*offset: normal = v/||v||, offset = ||v||.

    Args:
        slahmr_run_dir: SLAHMR Hydra run directory containing opt_log.txt.

    Returns:
        List of 4 floats [nx, ny, nz, d] (unit normal + signed offset),
        or None if not found.
    """
    log_path = slahmr_run_dir / "opt_log.txt"
    if not log_path.exists():
        return None

    last_match: Optional[str] = None
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _FLOOR_PLANE_RE.search(line)
            if m:
                last_match = m.group(1)

    if last_match is None:
        return None

    values = [float(v.strip()) for v in last_match.split(",")]
    if len(values) != 3:
        return None

    norm = (values[0] ** 2 + values[1] ** 2 + values[2] ** 2) ** 0.5
    if norm < 1e-8:
        return None

    nx, ny, nz = values[0] / norm, values[1] / norm, values[2] / norm
    if ny > 0:
        nx, ny, nz, norm = -nx, -ny, -nz, -norm

    return [nx, ny, nz, norm]


def _format_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted string like '2h 15m', '45m', or '< 1m'.
    """
    minutes = seconds / 60
    if minutes < 1:
        return "< 1m"
    if minutes < 60:
        return f"{int(minutes)}m"
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    if mins == 0:
        return f"{hours}h"
    return f"{hours}h {mins}m"


def _format_friendly_finish(now: "datetime", finish: "datetime") -> str:
    """Format a finish time as a friendly human-readable string.

    Examples:
        'Tonight @ 9:32pm'
        'Tomorrow morning, Thursday @ 7:15am'
        'Friday @ 2:30pm'

    Args:
        now: Current datetime.
        finish: Estimated finish datetime.

    Returns:
        Friendly finish string.
    """
    time_str = finish.strftime("%I:%M%p").lstrip("0").lower()
    day_name = finish.strftime("%A")

    days_ahead = (finish.date() - now.date()).days
    hour = finish.hour

    if hour < 6:
        time_of_day = "overnight"
    elif hour < 12:
        time_of_day = "morning"
    elif hour < 17:
        time_of_day = "afternoon"
    elif hour < 21:
        time_of_day = "evening"
    else:
        time_of_day = "tonight"

    if days_ahead == 0:
        if time_of_day == "tonight" or time_of_day == "overnight":
            return f"Tonight, {day_name} @ {time_str}"
        if time_of_day == "evening":
            return f"This evening, {day_name} @ {time_str}"
        if time_of_day == "afternoon":
            return f"This afternoon, {day_name} @ {time_str}"
        return f"Today, {day_name} @ {time_str}"

    if days_ahead == 1:
        return f"Tomorrow {time_of_day}, {day_name} @ {time_str}"

    return f"{day_name} {time_of_day} @ {time_str}"


def _print_stage_estimate(
    stage_name: str,
    total_iters: int,
    elapsed: float,
    done: int,
) -> None:
    """Print a time estimate box after the first few iterations.

    Args:
        stage_name: Optimizer stage name (e.g. 'motion_chunks').
        total_iters: Total iterations for this stage.
        elapsed: Seconds elapsed since stage start.
        done: Number of iterations completed.
    """
    sec_per_iter = elapsed / done
    early_stops = stage_name in ("motion_chunks", "motion_fit")

    from datetime import datetime, timedelta
    now = datetime.now()

    label = stage_name.replace("_", " ").title()

    if early_stops:
        short_iters = int(total_iters * 0.4)
        short_sec = sec_per_iter * short_iters
        long_sec = sec_per_iter * total_iters
        short_remaining = sec_per_iter * max(short_iters - done, 0)
        long_remaining = sec_per_iter * (total_iters - done)

        short_finish = now + timedelta(seconds=short_remaining)
        long_finish = now + timedelta(seconds=long_remaining)
        short_finish_str = _format_friendly_finish(now, short_finish)
        long_finish_str = _format_friendly_finish(now, long_finish)

        line1 = f"  {label}: up to {total_iters} iterations"
        line2 = f"  (usually converges early, ~{short_iters} iterations)"
        line3 = f"  Pace: {sec_per_iter:.1f}s / iteration"
        line4 = f"  Likely: ~{_format_duration(short_sec)}, {short_finish_str}"
        line5 = f"  Worst case: ~{_format_duration(long_sec)}, {long_finish_str}"

        lines = [line1, line2, line3, line4, line5]
        spacer_after = 3
    else:
        remaining_sec = sec_per_iter * (total_iters - done)
        total_sec = sec_per_iter * total_iters
        finish_time = now + timedelta(seconds=remaining_sec)
        finish_str = _format_friendly_finish(now, finish_time)

        line1 = f"  {label}: {total_iters} iterations"
        line2 = f"  Estimated time: ~{_format_duration(total_sec)}"
        line3 = f"  Pace: {sec_per_iter:.1f}s / iteration"
        line4 = f"  Finish: {finish_str}"

        lines = [line1, line2, line3, line4]
        spacer_after = 3

    width = max(len(l) for l in lines) + 4
    bar = "+" + "-" * (width - 2) + "+"
    spacer = "|" + " " * (width - 2) + "|"

    print()
    print(f"    {bar}")
    for i, line in enumerate(lines):
        if i == spacer_after:
            print(f"    {spacer}")
        print(f"    | {line:<{width - 4}} |")
    print(f"    {bar}")
    print()


def _run_slahmr_filtered(
    cmd: List[str],
    log_path: Path,
    timeout: int = 14400,
) -> int:
    """Run SLAHMR subprocess with filtered terminal output.

    All output is written to log_path. Terminal sees only stage
    transitions and periodic iteration progress (every 20 iterations).

    Args:
        cmd: Subprocess command list.
        log_path: Path to write full unfiltered log.
        timeout: Subprocess timeout in seconds.

    Returns:
        Process return code.
    """
    cur_total_iters: Optional[int] = None
    cur_stage_name: Optional[str] = None
    stage_start_time: Optional[float] = None
    stage_start_iter: int = 0
    estimate_printed = False

    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        try:
            for line in proc.stdout:
                log_f.write(line)

                stripped = line.rstrip()

                if _SLAHMR_SUPPRESS.match(stripped):
                    continue

                m_iter = _SLAHMR_ITER.match(stripped)
                if m_iter:
                    iter_num = int(m_iter.group(1))
                    if not cur_total_iters or not stage_start_time:
                        continue
                    done = iter_num - stage_start_iter
                    if done == 3 and not estimate_printed:
                        estimate_printed = True
                        _print_stage_estimate(
                            cur_stage_name or "optimization",
                            cur_total_iters,
                            time.monotonic() - stage_start_time,
                            done,
                        )
                    if done > 0 and iter_num % 20 == 0:
                        elapsed = time.monotonic() - stage_start_time
                        remaining = cur_total_iters - iter_num
                        eta_sec = (elapsed / done) * remaining
                        eta_min = eta_sec / 60
                        if eta_min >= 60:
                            eta = f"{eta_min / 60:.1f}h remaining"
                        else:
                            eta = f"{eta_min:.0f}m remaining"
                        print(f"    [{iter_num}/{cur_total_iters}] ~{eta}")
                    continue

                if "OPTIMIZING" in stripped and "ITERATIONS" in stripped:
                    m = re.search(r"FOR (\d+) ITERATIONS", stripped)
                    if m:
                        cur_total_iters = int(m.group(1))
                    m_name = re.search(r"OPTIMIZING (\S+)", stripped)
                    cur_stage_name = m_name.group(1) if m_name else None
                    stage_start_iter = 0
                    stage_start_time = time.monotonic()
                    estimate_printed = False
                    print(f"    {stripped}")
                    continue

                if stripped:
                    print(f"    {stripped}")

            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            print(f"\n  Error: SLAHMR timed out after {timeout}s", file=sys.stderr)

    return proc.returncode


def check_slahmr_available() -> bool:
    """Check if SLAHMR is installed and its conda env exists.

    Returns:
        True if SLAHMR repo and conda environment are present.
    """
    if not SLAHMR_INSTALL_DIR.exists():
        return False
    if not (SLAHMR_INSTALL_DIR / "slahmr" / "run_opt.py").exists():
        return False

    conda_exe = _find_conda()
    if not conda_exe:
        return False

    result = subprocess.run(
        [conda_exe, "env", "list"],
        capture_output=True, text=True, timeout=10,
    )
    return result.returncode == 0 and SLAHMR_CONDA_PREFIX.exists()


def find_or_create_video(
    project_dir: Path,
    fps: int = 24,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
) -> Optional[Path]:
    """Find source video or create one from frames.

    Args:
        project_dir: Project directory.
        fps: Integer frame rate for video encoding.
        start_frame: Start frame (1-indexed, inclusive).
        end_frame: End frame (1-indexed, inclusive).

    Returns:
        Path to video file, or None if creation fails.
    """
    source_dir = project_dir / "source"
    has_frame_range = start_frame is not None or end_frame is not None

    video_extensions = [".mp4", ".mov", ".avi", ".mkv", ".webm"]
    source_video = None
    for ext in video_extensions:
        for video_file in source_dir.glob(f"*{ext}"):
            if not video_file.name.startswith("_"):
                source_video = video_file
                break
        if source_video:
            break

    if source_video and has_frame_range:
        trimmed_path = source_dir / "_slahmr_trimmed.mp4"
        start_time = (start_frame - 1) / fps if start_frame else 0
        trim_cmd = ["ffmpeg", "-y", "-i", str(source_video)]
        if start_frame:
            trim_cmd.extend(["-ss", str(start_time)])
        if end_frame:
            duration = (end_frame - (start_frame or 1) + 1) / fps
            trim_cmd.extend(["-t", str(duration)])
        trim_cmd.extend([
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-pix_fmt", "yuv420p", str(trimmed_path),
        ])
        result = subprocess.run(trim_cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0 and trimmed_path.exists():
            return trimmed_path
        return None

    if source_video and not has_frame_range:
        return source_video

    frames_dir = source_dir / "frames"
    if not frames_dir.exists():
        return None

    frames = sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg"))
    if not frames:
        return None

    if has_frame_range:
        actual_start = (start_frame or 1) - 1
        actual_end = end_frame if end_frame else len(frames)
        frames = frames[actual_start:actual_end]
        if not frames:
            print(f"Error: No frames in range {start_frame}-{end_frame}", file=sys.stderr)
            return None

    range_suffix = ""
    if has_frame_range:
        range_suffix = f"_{start_frame or 1}_{end_frame or 'end'}"
    video_path = source_dir / f"_slahmr_input{range_suffix}.mp4"

    if video_path.exists() and not has_frame_range:
        return video_path

    print(f"  → Creating video from {len(frames)} frames...")

    first_frame = frames[0]
    stem = first_frame.stem
    if "_" in stem:
        prefix = stem.rsplit("_", 1)[0]
        num_part = stem.rsplit("_", 1)[1]
        num_digits = len(num_part)
        frame_pattern = first_frame.parent / f"{prefix}_%0{num_digits}d{first_frame.suffix}"
        pattern_start = int(num_part)
    else:
        frame_pattern = first_frame.parent / f"%04d{first_frame.suffix}"
        pattern_start = 1

    cmd = ["ffmpeg", "-y", "-framerate", str(fps)]
    if has_frame_range:
        cmd.extend(["-start_number", str(pattern_start)])
    cmd.extend(["-i", str(frame_pattern)])
    if has_frame_range:
        cmd.extend(["-frames:v", str(len(frames))])
    cmd.extend([
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-pix_fmt", "yuv420p", str(video_path),
    ])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode == 0 and video_path.exists():
        print(f"  OK Created video: {video_path.name}")
        return video_path

    return None


def stitch_slahmr_chunks(
    motion_chunks_dir: Path,
    seq_name: str,
) -> Optional[Path]:
    """Select or stitch SLAHMR motion chunks into a single NPZ.

    SLAHMR saves motion_chunks in two modes:
    1. Optimization checkpoints: multiple files with identical frame counts
       (each is a full-sequence snapshot at a different iteration). Use the
       last (most optimized) file.
    2. Temporal windows: files with different frame counts covering different
       parts of the sequence. Stitch by concatenating non-overlapping frames.

    Args:
        motion_chunks_dir: Directory containing chunk NPZ files.
        seq_name: Sequence name prefix.

    Returns:
        Path to final NPZ, or None if failed.
    """
    chunk_pattern = f"{seq_name}_*_world_results.npz"
    chunk_files = sorted(motion_chunks_dir.glob(chunk_pattern))

    if not chunk_files:
        chunk_files = sorted(motion_chunks_dir.glob("*_world_results.npz"))

    if not chunk_files:
        print(f"  Error: No motion chunks found in {motion_chunks_dir}", file=sys.stderr)
        return None

    if len(chunk_files) == 1:
        print("  → Single chunk, no stitching needed")
        return chunk_files[0]

    frame_counts = []
    for cf in chunk_files:
        data = np.load(cf, allow_pickle=True)
        trans = np.asarray(data["trans"])
        n_frames = trans.shape[1] if trans.ndim == 3 else trans.shape[0]
        frame_counts.append(n_frames)

    all_same = len(set(frame_counts)) == 1
    if all_same:
        print(f"  → {len(chunk_files)} checkpoints with {frame_counts[0]} frames each, using last")
        return chunk_files[-1]

    print(f"  → Stitching {len(chunk_files)} temporal chunks...")

    chunk_starts = []
    for cf in chunk_files:
        parts = cf.stem.split("_")
        for part in parts:
            if part.isdigit() and len(part) == 6:
                chunk_starts.append(int(part))
                break

    all_trans = []
    all_root_orient = []
    all_pose_body = []
    all_cam_R = []
    all_cam_t = []
    first_data = None

    for i, cf in enumerate(chunk_files):
        data = np.load(cf, allow_pickle=True)
        if i == 0:
            first_data = data

        trans = np.asarray(data["trans"])
        root_orient = np.asarray(data["root_orient"])
        pose_body = np.asarray(data["pose_body"])
        cam_R = np.asarray(data["cam_R"])
        cam_t = np.asarray(data["cam_t"])

        if trans.ndim == 3:
            trans = trans[0]
            root_orient = root_orient[0]
            pose_body = pose_body[0]
            cam_R = cam_R[0]
            cam_t = cam_t[0]

        chunk_len = len(trans)

        if i < len(chunk_starts) - 1:
            next_start = chunk_starts[i + 1] - chunk_starts[i]
            use_frames = min(next_start, chunk_len)
        else:
            use_frames = chunk_len

        all_trans.append(trans[:use_frames])
        all_root_orient.append(root_orient[:use_frames])
        all_pose_body.append(pose_body[:use_frames])
        all_cam_R.append(cam_R[:use_frames])
        all_cam_t.append(cam_t[:use_frames])

    stitched = {
        "trans": np.concatenate(all_trans, axis=0),
        "root_orient": np.concatenate(all_root_orient, axis=0),
        "pose_body": np.concatenate(all_pose_body, axis=0),
        "cam_R": np.concatenate(all_cam_R, axis=0),
        "cam_t": np.concatenate(all_cam_t, axis=0),
        "betas": np.asarray(first_data["betas"]),
        "intrins": np.asarray(first_data["intrins"]),
    }
    if "world_scale" in first_data:
        stitched["world_scale"] = np.asarray(first_data["world_scale"])

    stitched_path = motion_chunks_dir / f"{seq_name}_stitched.npz"
    np.savez(stitched_path, **stitched)

    total_frames = len(stitched["trans"])
    print(f"  OK Stitched {total_frames} frames from {len(chunk_files)} chunks")
    return stitched_path


def export_slahmr_camera(
    npz_path: Path,
    camera_output_dir: Path,
    image_width: int,
    image_height: int,
) -> bool:
    """Export SLAHMR camera to pipeline format.

    SLAHMR cam_R/cam_t are world-to-camera. We invert to camera-to-world
    for consistency with the pipeline's extrinsics.json convention.

    Args:
        npz_path: Path to stitched SLAHMR NPZ.
        camera_output_dir: Output directory for camera files.
        image_width: Source image width.
        image_height: Source image height.

    Returns:
        True if export succeeded.
    """
    data = np.load(npz_path, allow_pickle=True)

    cam_R = np.asarray(data["cam_R"], dtype=np.float64)
    cam_t = np.asarray(data["cam_t"], dtype=np.float64)
    intrins = np.asarray(data["intrins"], dtype=np.float64)

    if cam_R.ndim == 4:
        cam_R = cam_R[0]
    if cam_t.ndim == 3:
        cam_t = cam_t[0]

    n_frames = len(cam_R)

    R_c2w = np.transpose(cam_R, (0, 2, 1))
    t_c2w = -np.einsum("fij,fj->fi", R_c2w, cam_t)

    c2w = np.broadcast_to(np.eye(4), (n_frames, 4, 4)).copy()
    c2w[:, :3, :3] = R_c2w
    c2w[:, :3, 3] = t_c2w

    camera_output_dir.mkdir(parents=True, exist_ok=True)

    extrinsics_data = [m.tolist() for m in c2w]
    with open(camera_output_dir / "extrinsics.json", "w", encoding="utf-8") as f:
        json.dump(extrinsics_data, f, indent=2)

    fx, fy, cx, cy = float(intrins[0]), float(intrins[1]), float(intrins[2]), float(intrins[3])
    intrinsics_data = {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "width": image_width,
        "height": image_height,
        "model": "PINHOLE",
        "params": [fx, fy, cx, cy],
        "source": "slahmr",
    }
    with open(camera_output_dir / "intrinsics.json", "w", encoding="utf-8") as f:
        json.dump(intrinsics_data, f, indent=2)

    print(f"  → Exported camera to {camera_output_dir}")
    print(f"    Intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    print(f"    Extrinsics: {n_frames} frames (joint-optimized)")
    return True


def _find_motion_chunks(
    seq_name: str,
    tmp_root: Path,
    slahmr_outputs_dir: Path,
) -> Optional[Path]:
    """Find SLAHMR motion_chunks directory.

    SLAHMR uses Hydra and writes output to its own outputs/logs/ directory,
    not the data root. Search both locations.

    Args:
        seq_name: Sequence name.
        tmp_root: Temporary data root passed to SLAHMR.
        slahmr_outputs_dir: SLAHMR_INSTALL_DIR / "outputs".

    Returns:
        Path to motion_chunks directory, or None if not found.
    """
    search_paths = [
        tmp_root / "slahmr" / "results" / "motion_chunks" / seq_name,
        tmp_root / "slahmr" / "motion_chunks" / seq_name,
    ]

    if slahmr_outputs_dir.exists():
        for run_dir in sorted(slahmr_outputs_dir.rglob(f"{seq_name}*"), reverse=True):
            mc = run_dir / "motion_chunks"
            if mc.exists() and list(mc.glob("*_world_results.npz")):
                search_paths.insert(0, mc)
                break

    for path in search_paths:
        if path.exists() and list(path.glob("*_world_results.npz")):
            return path

    for search_root in [slahmr_outputs_dir, tmp_root]:
        if not search_root.exists():
            continue
        for p in search_root.rglob("*_world_results.npz"):
            if "motion_chunks" in str(p.parent):
                print(f"  → Found motion chunks at: {p.parent}")
                return p.parent

    return None


def run_slahmr_pipeline(
    project_dir: Path,
    gender: str = "neutral",
    fps: Optional[int] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    mocap_person: Optional[str] = None,
) -> bool:
    """Run SLAHMR motion capture pipeline.

    Args:
        project_dir: Project directory.
        gender: Body model gender.
        fps: Integer frames per second (default: 24).
        start_frame: Start frame (1-indexed, inclusive).
        end_frame: End frame (1-indexed, inclusive).
        mocap_person: Person folder name (default: "person").

    Returns:
        True if successful.
    """
    if not check_slahmr_available():
        print("Error: SLAHMR not available.", file=sys.stderr)
        print("Install with: python scripts/install_wizard.py", file=sys.stderr)
        return False

    conda_exe = _find_conda()
    if not conda_exe:
        print("Error: Conda not found — required for SLAHMR", file=sys.stderr)
        return False

    fps = fps or 24
    person_folder = mocap_person or "person"
    mocap_person_dir = project_dir / "mocap" / person_folder
    mocap_person_dir.mkdir(parents=True, exist_ok=True)
    mocap_camera_dir = project_dir / "mocap_camera"

    video_path = find_or_create_video(project_dir, fps, start_frame, end_frame)
    if not video_path:
        print("Error: No video file found and could not create from frames", file=sys.stderr)
        return False

    source_frames = project_dir / "source" / "frames"
    frame_files = sorted(list(source_frames.glob("*.png")) + list(source_frames.glob("*.jpg")))
    n_frames = len(frame_files)

    image_width, image_height = 1920, 1080
    if frame_files:
        try:
            from PIL import Image
            with Image.open(frame_files[0]) as img:
                image_width, image_height = img.size
        except Exception:
            pass

    if start_frame or end_frame:
        n_process = (end_frame or n_frames) - (start_frame or 1) + 1
    else:
        n_process = n_frames

    camera_metadata = _load_camera_metadata(project_dir)
    fixed_focal_px: Optional[float] = None
    if camera_metadata is not None:
        focal_mm = camera_metadata.get("focal_length_mm")
        sensor_w = camera_metadata.get("sensor_width_mm")
        if focal_mm is not None and sensor_w is not None:
            fixed_focal_px = float(focal_mm) * image_width / float(sensor_w)

    pipeline_start = time.time()

    print(f"\n{'='*60}")
    print("SLAHMR Motion Capture")
    print("=" * 60)
    print(f"Project: {project_dir}")
    print(f"Video: {video_path}")
    print(f"Frames: {n_process} @ {fps} fps")
    print(f"Gender: {gender}")
    print(f"Resolution: {image_width}x{image_height}")
    if fixed_focal_px is not None:
        print(f"Fixed focal: {fixed_focal_px:.1f}px ({focal_mm}mm on {sensor_w}mm sensor)")
    else:
        print(f"Focal: SLAHMR default (will be optimized)")
    if start_frame or end_frame:
        print(f"Frame range: {start_frame or 1}-{end_frame or n_frames}")
    print()

    seq_name = project_dir.name
    slahmr_output_dir = mocap_person_dir / "slahmr"
    slahmr_run_dir = slahmr_output_dir / "run"
    tmp_root = Path(tempfile.mkdtemp(prefix="slahmr_"))
    tmp_seq_dir = tmp_root / "videos" / seq_name

    completed_before = (
        (slahmr_output_dir / "slahmr_stitched.npz").exists()
        and (mocap_person_dir / "motion.pkl").exists()
    )
    has_checkpoints = slahmr_run_dir.exists() and any(
        slahmr_run_dir.glob("*_params.pth")
    )

    if completed_before:
        print("  → Previous completed run found — skipping")
        print(f"    Motion: {mocap_person_dir / 'motion.pkl'}")
        print(f"    Camera: {mocap_camera_dir}")
        return True
    elif has_checkpoints:
        print("  → Interrupted run detected — resuming from checkpoint")

    slahmr_run_dir.mkdir(parents=True, exist_ok=True)

    try:
        tmp_seq_dir.mkdir(parents=True)

        if fixed_focal_px is not None:
            cam_npz_path = (
                tmp_root / "slahmr" / "cameras" / seq_name / "shot-0" / "cameras.npz"
            )
            _create_slahmr_cameras_npz(
                cam_npz_path, fixed_focal_px, n_process, image_width, image_height,
            )
            print(f"  → Created cameras.npz with focal={fixed_focal_px:.1f}px")

        slahmr_cmd = (
            f"cd {SLAHMR_INSTALL_DIR / 'slahmr'} && "
            f"CUDA_HOME=$CONDA_PREFIX "
            f"CC=x86_64-conda-linux-gnu-gcc "
            f"CXX=x86_64-conda-linux-gnu-g++ "
            f"python run_opt.py "
            f"data=video "
            f"data.root={tmp_root} "
            f"data.seq={seq_name} "
            f"data.src_path={video_path} "
            f"data.end_idx={n_process} "
            f"data.frame_opts.fps={fps} "
            f"fps={fps} "
            f"run_opt=True "
            f"run_vis=False "
            f"log_root={slahmr_output_dir / 'logs'} "
            f"exp_name=opt "
            f"hydra.run.dir={slahmr_run_dir}"
        )

        if fixed_focal_px is not None:
            slahmr_cmd += " optim.motion_chunks.opt_cams=False"

        cmd = [
            conda_exe, "run", "-p", str(SLAHMR_CONDA_PREFIX), "--no-capture-output",
            "bash", "-c", slahmr_cmd,
        ]

        print("  → Running SLAHMR optimization...")
        print(f"    Run dir: {slahmr_run_dir}")
        print(f"    This may take 1-3 hours depending on sequence length")
        print()

        log_dir = project_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        slahmr_log = log_dir / "slahmr_opt.log"

        returncode = _run_slahmr_filtered(cmd, slahmr_log, timeout=14400)

        if returncode != 0:
            print(f"Error: SLAHMR failed (exit code {returncode})", file=sys.stderr)
            print(f"  Full log: {slahmr_log}", file=sys.stderr)
            return False

        slahmr_outputs_dir = SLAHMR_INSTALL_DIR / "outputs"
        motion_chunks_dir = _find_motion_chunks(
            seq_name, tmp_root, slahmr_outputs_dir,
        )
        if motion_chunks_dir is None:
            mc_in_run = slahmr_run_dir / "motion_chunks"
            if mc_in_run.exists() and list(mc_in_run.glob("*_world_results.npz")):
                motion_chunks_dir = mc_in_run
        if motion_chunks_dir is None:
            print("Error: No SLAHMR motion chunks found", file=sys.stderr)
            print(f"  Searched: {slahmr_run_dir}, {tmp_root}, {slahmr_outputs_dir}",
                  file=sys.stderr)
            return False

        stitched_path = stitch_slahmr_chunks(motion_chunks_dir, seq_name)
        if stitched_path is None:
            return False

        slahmr_output_dir.mkdir(parents=True, exist_ok=True)
        final_npz = slahmr_output_dir / "slahmr_stitched.npz"
        shutil.copy2(stitched_path, final_npz)

        for chunk_file in motion_chunks_dir.glob("*_world_results.npz"):
            shutil.copy2(chunk_file, slahmr_output_dir / chunk_file.name)

        print("\n--- Exporting Camera ---")
        export_slahmr_camera(final_npz, mocap_camera_dir, image_width, image_height)

        print("\n--- Extracting Floor Plane ---")
        floor_plane = extract_floor_plane(slahmr_run_dir)
        if floor_plane:
            floor_json = {
                "normal": floor_plane[:3],
                "offset": floor_plane[3],
                "equation": "nx*x + ny*y + nz*z = d",
                "coordinate_system": "slahmr_camera",
            }
            floor_path = slahmr_output_dir / "floor_plane.json"
            with open(floor_path, "w", encoding="utf-8") as f:
                json.dump(floor_json, f, indent=2)
            print(f"  → Floor plane: n=[{floor_plane[0]:.4f}, {floor_plane[1]:.4f}, "
                  f"{floor_plane[2]:.4f}], d={floor_plane[3]:.4f}")
        else:
            print("  → Floor plane not found in log (non-critical)")

        print("\n--- Converting SMPL-H → SMPLX ---")
        motion_pkl_path = mocap_person_dir / "motion.pkl"
        conda_exe = _find_conda()
        if not conda_exe:
            print("  Error: Conda not found — required for SMPL-H → SMPLX", file=sys.stderr)
            return False
        convert_script = Path(__file__).parent / "convert_smplh_to_smplx.py"
        convert_cmd = [
            conda_exe, "run", "-p", str(GVHMR_CONDA_PREFIX), "--no-capture-output",
            "python", str(convert_script),
            "--smplh-npz", str(final_npz),
            "--output", str(motion_pkl_path),
            "--gender", gender,
        ]
        result = subprocess.run(convert_cmd)
        if result.returncode != 0:
            print("  Warning: SMPL-H → SMPLX conversion failed", file=sys.stderr)
            return False

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    pipeline_end = time.time()
    total_minutes = (pipeline_end - pipeline_start) / 60

    print(f"\n{'='*60}")
    print("SLAHMR Motion Capture Complete")
    print("=" * 60)
    print(f"Motion: {mocap_person_dir / 'motion.pkl'}")
    print(f"Camera: {mocap_camera_dir}")
    print(f"SLAHMR raw: {mocap_person_dir / 'slahmr'}")
    print(f"Time: {total_minutes:.1f} minutes")
    print()

    return True


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run SLAHMR motion capture on a project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "project_dir",
        type=Path,
        help="Project directory containing source/frames/",
    )
    parser.add_argument(
        "--gender",
        choices=["neutral", "male", "female"],
        default="neutral",
        help="Body model gender (default: neutral)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Integer FPS (default: 24)",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="Start frame (1-indexed, inclusive)",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="End frame (1-indexed, inclusive)",
    )
    parser.add_argument(
        "--mocap-person",
        type=str,
        default=None,
        help="Person folder name (default: 'person')",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if SLAHMR is available and exit",
    )

    args = parser.parse_args()

    require_conda_env()

    if args.check:
        if check_slahmr_available():
            print("SLAHMR is available")
            print(f"  Install dir: {SLAHMR_INSTALL_DIR}")
            print(f"  Conda env: {SLAHMR_CONDA_ENV}")
            sys.exit(0)
        else:
            print("SLAHMR is not available")
            print(f"  Expected: {SLAHMR_INSTALL_DIR}")
            sys.exit(1)

    project_dir = args.project_dir.resolve()
    if not project_dir.exists():
        print(f"Error: Project directory not found: {project_dir}", file=sys.stderr)
        sys.exit(1)

    success = run_slahmr_pipeline(
        project_dir=project_dir,
        gender=args.gender,
        fps=args.fps,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        mocap_person=args.mocap_person,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    with LogCapture():
        main()
