#!/usr/bin/env python3
"""Align mocap body to matchmove camera world space.

Transforms SLAHMR body from its own world space into VGGSfM's world
space by chaining per-frame transforms:

    SMPLX mesh → undo pelvis offset → SLAHMR w2c → VGGSfM c2w

SLAHMR jointly estimates camera and body in its own coordinate system.
VGGSfM produces a scene-level camera solve. This script bridges them
so the body mesh can be viewed through the output camera with correct
2D alignment.

The output camera uses VGGSfM extrinsics (trajectory) with SLAHMR
intrinsics (focal length). Uniform camera-space scaling cannot fix a
focal length mismatch — it cancels in the perspective divide:
f*(sX)/(sZ) = f*X/Z. The body only projects correctly through
SLAHMR's jointly-estimated focal, so the output camera must use it.

Environment:
    Requires 'gvhmr' conda environment (has smplx, torch).

Usage:
    conda run -n gvhmr python align_mocap_to_mmcam.py <project_dir> [options]

Example:
    conda run -n gvhmr python align_mocap_to_mmcam.py /path/to/TNIS0012 --fps 24
"""

import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from env_config import require_conda_env
from transforms import quaternion_to_rotation_matrix
from export_mocap import (
    export_alembic,
    export_tpose,
    export_usd,
    generate_meshes,
    get_body_model_path,
    load_motion_data,
    parse_formats,
)

REQUIRED_ENV = "gvhmr"


def load_slahmr_camera(
    npz_path: Path,
    person_idx: int = 0,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Load SLAHMR camera and translation from stitched NPZ.

    Args:
        npz_path: Path to slahmr_stitched.npz.
        person_idx: Person index in multi-person output.

    Returns:
        Tuple of (cam_R (N,3,3), cam_t (N,3), focal_length, trans_frame0 (3,)).
    """
    data = np.load(npz_path, allow_pickle=True)
    cam_R = np.asarray(data["cam_R"][person_idx], dtype=np.float64)
    cam_t = np.asarray(data["cam_t"][person_idx], dtype=np.float64)
    intrins = np.asarray(data["intrins"], dtype=np.float64)
    focal = float(intrins[0])
    trans_0 = np.asarray(data["trans"][person_idx, 0], dtype=np.float64)
    return cam_R, cam_t, focal, trans_0


def load_vggsfm_camera(camera_dir: Path) -> Tuple[np.ndarray, dict]:
    """Load VGGSfM camera-to-world matrices and intrinsics.

    Args:
        camera_dir: Directory containing extrinsics.json and intrinsics.json.

    Returns:
        Tuple of (c2w (N,4,4), intrinsics dict).
    """
    with open(camera_dir / "extrinsics.json", encoding="utf-8") as f:
        c2w = np.array(json.load(f), dtype=np.float64)
    with open(camera_dir / "intrinsics.json", encoding="utf-8") as f:
        intrinsics = json.load(f)
    return c2w, intrinsics


def compute_pelvis_offset(
    motion_trans_0: np.ndarray,
    slahmr_trans_0: np.ndarray,
) -> np.ndarray:
    """Compute pelvis offset between SMPLX motion.pkl and SLAHMR SMPL-H.

    motion.pkl has pelvis-corrected translation (from convert_smplh_to_smplx).
    SLAHMR's camera was calibrated against the uncorrected SMPL-H body.
    The offset is the difference applied during SMPL-H → SMPLX conversion.

    Args:
        motion_trans_0: First-frame translation from motion.pkl.
        slahmr_trans_0: First-frame translation from slahmr_stitched.npz.

    Returns:
        (3,) pelvis offset vector.
    """
    return np.asarray(motion_trans_0, dtype=np.float64) - np.asarray(
        slahmr_trans_0, dtype=np.float64
    )


def compute_gravity_rotation(
    c2w: np.ndarray,
    camera_dir: Path,
) -> np.ndarray:
    """Load the gravity alignment rotation saved during camera export.

    The pipeline applies a gravity alignment rotation to camera extrinsics
    (so world +Y = up), but the sparse pointcloud PLY from COLMAP is in the
    raw (unrotated) world frame. This rotation brings the pointcloud into
    the same gravity-aligned frame as the body mesh.

    Args:
        c2w: (N, 4, 4) gravity-aligned camera-to-world matrices.
        camera_dir: Directory containing gravity_align.npy.

    Returns:
        (3, 3) rotation matrix mapping raw COLMAP world to gravity-aligned world.
    """
    npy_path = camera_dir / "gravity_align.npy"
    return np.load(npy_path)


def compute_world_scale(
    c2w: np.ndarray,
    cam_R: np.ndarray,
    cam_t: np.ndarray,
) -> float:
    """Compute VGGSfM-to-metric scale via Umeyama alignment of camera trajectories.

    VGGSfM's monocular SfM produces an arbitrary-scale reconstruction;
    SLAHMR's body-aware optimization produces metric-scale results.
    The Umeyama algorithm finds the similarity transform (rotation + scale +
    translation) that best maps SLAHMR camera positions to VGGSfM positions.

    Args:
        c2w: (N, 4, 4) VGGSfM camera-to-world matrices.
        cam_R: (N, 3, 3) SLAHMR world-to-camera rotation.
        cam_t: (N, 3) SLAHMR world-to-camera translation.

    Returns:
        Scale factor (VGGSfM_units / meters).
    """
    vggsfm_pos = c2w[:, :3, 3]
    slahmr_pos = np.array([-cam_R[i].T @ cam_t[i] for i in range(len(cam_t))])

    mu_src = slahmr_pos.mean(0)
    mu_dst = vggsfm_pos.mean(0)
    src_c = slahmr_pos - mu_src
    dst_c = vggsfm_pos - mu_dst

    sigma_src = np.sum(src_c ** 2) / len(src_c)
    if sigma_src < 1e-12:
        raise ValueError("SLAHMR cameras are stationary — cannot compute scale")

    h = (dst_c.T @ src_c) / len(src_c)
    u, d, vt = np.linalg.svd(h)

    s_mat = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_mat[2, 2] = -1

    return float(np.trace(np.diag(d) @ s_mat) / sigma_src)


def transform_sparse_ply(
    input_path: Path,
    output_path: Path,
    rotation: np.ndarray,
    scale: float,
    centroid: np.ndarray,
) -> None:
    """Transform sparse pointcloud from raw COLMAP world to body-mesh space.

    Applies two corrections:
      1. Gravity alignment rotation (raw COLMAP → gravity-aligned world)
      2. Scale around camera centroid (VGGSfM arbitrary scale → metric)

    The body mesh sits in a hybrid space: VGGSfM camera positions with metric
    body offsets. Scaling the rotated pointcloud around the camera centroid
    brings scene features to metric distance from the cameras, matching
    the body's metric offsets.

    Preserves vertex colors and PLY structure (binary or ASCII).

    Args:
        input_path: Input PLY file in raw COLMAP world space.
        output_path: Output PLY file in gravity-aligned metric-scale space.
        rotation: (3, 3) gravity alignment rotation.
        scale: VGGSfM_units / meters ratio.
        centroid: (3,) camera trajectory centroid in gravity-aligned space.
    """
    with open(input_path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            header_lines.append(line)
            if line.strip() == b"end_header":
                break
        header = b"".join(header_lines)
        body = f.read()

    header_text = header.decode("ascii")
    n_verts = 0
    for line in header_text.splitlines():
        if line.startswith("element vertex"):
            n_verts = int(line.split()[-1])
            break

    is_binary = "binary_little_endian" in header_text
    vertex_dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])

    if is_binary:
        vertices = np.frombuffer(body, dtype=vertex_dtype, count=n_verts)
    else:
        vertices = np.loadtxt(
            input_path,
            dtype=vertex_dtype,
            skiprows=len(header_lines),
            max_rows=n_verts,
        )

    xyz = np.column_stack([vertices["x"], vertices["y"], vertices["z"]])
    xyz_rotated = (rotation @ xyz.T).T
    xyz_scaled = centroid + (xyz_rotated - centroid) / scale

    vertices = vertices.copy()
    vertices["x"] = xyz_scaled[:, 0].astype(np.float32)
    vertices["y"] = xyz_scaled[:, 1].astype(np.float32)
    vertices["z"] = xyz_scaled[:, 2].astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if is_binary:
        with open(output_path, "wb") as f:
            f.write(header)
            f.write(vertices.tobytes())
    else:
        with open(output_path, "w", encoding="ascii") as f:
            f.write(header_text)
            for row in vertices:
                vals = " ".join(str(v) for v in row)
                f.write(vals + "\n")


def transform_meshes(
    meshes: List[Tuple],
    cam_R: np.ndarray,
    cam_t: np.ndarray,
    c2w: np.ndarray,
    pelvis_offset: np.ndarray,
) -> List[Tuple]:
    """Transform meshes from SLAHMR world to VGGSfM world space.

    Per-frame chain:
        1. Undo pelvis offset (SMPLX→SMPL-H correction)
        2. SLAHMR world-to-camera: V_cam = R @ V + t
        3. VGGSfM camera-to-world: V_world = c2w_R @ V_cam + c2w_t

    No camera-space scaling is applied — uniform scaling cancels in
    the perspective divide (f*sX/sZ = f*X/Z), so the output camera
    must use SLAHMR's focal for correct 2D projection.

    Args:
        meshes: List of (vertices, faces) in SMPLX motion.pkl space.
        cam_R: (N, 3, 3) SLAHMR world-to-camera rotation.
        cam_t: (N, 3) SLAHMR world-to-camera translation.
        c2w: (N, 4, 4) VGGSfM camera-to-world matrices.
        pelvis_offset: (3,) offset to subtract from vertices.

    Returns:
        List of (vertices, faces) in VGGSfM world space.
    """
    transformed = []
    for i, (verts, faces) in enumerate(meshes):
        v = verts.astype(np.float64) - pelvis_offset
        v = (cam_R[i] @ v.T).T + cam_t[i]
        v = (c2w[i, :3, :3] @ v.T).T + c2w[i, :3, 3]
        transformed.append((v.astype(np.float32), faces))
    return transformed


def compute_ground_height(meshes: List[Tuple], percentile: float = 2.0) -> float:
    """Compute ground plane Y height from transformed mesh vertices.

    Uses a low percentile of per-frame minimum Y values to robustly
    estimate the ground contact height (filtering single-frame outliers).

    Args:
        meshes: List of (vertices, faces) in world space.
        percentile: Percentile of per-frame min-Y values to use.

    Returns:
        Ground Y height.
    """
    frame_mins = [verts[:, 1].min() for verts, _ in meshes]
    return float(np.percentile(frame_mins, percentile))


def generate_ground_plane_ply(
    output_path: Path,
    height: float,
    center: np.ndarray,
    size: float = 4.0,
    divisions: int = 10,
) -> None:
    """Write a grid mesh PLY at the given Y height.

    Args:
        output_path: Path to write the PLY file.
        height: Y coordinate for the grid.
        center: (2,) XZ center of the grid.
        size: Half-extent of the grid in world units.
        divisions: Number of subdivisions per axis.
    """
    step = (2.0 * size) / divisions
    vertices = []
    for iz in range(divisions + 1):
        for ix in range(divisions + 1):
            x = center[0] - size + ix * step
            z = center[1] - size + iz * step
            vertices.append((x, height, z))

    faces = []
    for iz in range(divisions):
        for ix in range(divisions):
            v0 = iz * (divisions + 1) + ix
            v1 = v0 + 1
            v2 = v0 + (divisions + 1)
            v3 = v2 + 1
            faces.append((v0, v1, v3))
            faces.append((v0, v3, v2))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def export_static_geometry(
    output_dir: Path,
    meshes: List[Tuple],
    project_dir: Path,
    fps: int,
    gravity_rotation: np.ndarray,
    world_scale: float,
    camera_centroid: np.ndarray,
) -> List[Path]:
    """Export ground plane and sparse pointcloud as PLY and Alembic.

    Generates a ground plane grid mesh at foot-contact height and transforms
    the sparse pointcloud from raw COLMAP world into gravity-aligned metric
    space (matching the body mesh). Both are converted to Alembic via Blender.

    Args:
        output_dir: Output directory for exported files.
        meshes: List of (vertices, faces) in world space.
        project_dir: Project directory (for sparse PLY source).
        fps: Frame rate for Alembic export.
        gravity_rotation: (3, 3) rotation from raw COLMAP → gravity-aligned world.
        world_scale: VGGSfM_units / meters ratio.
        camera_centroid: (3,) camera centroid in gravity-aligned world.

    Returns:
        List of exported file paths.
    """
    from blender import check_blender_available, export_ply_to_alembic

    exported: List[Path] = []

    ground_height = compute_ground_height(meshes)
    all_verts = np.concatenate([v for v, _ in meshes], axis=0)
    xz_center = np.array([np.median(all_verts[:, 0]), np.median(all_verts[:, 2])])
    xz_range = max(all_verts[:, 0].ptp(), all_verts[:, 2].ptp())
    grid_half = max(xz_range * 0.75, 2.0)

    ground_ply = output_dir / "ground_plane.ply"
    generate_ground_plane_ply(ground_ply, ground_height, xz_center, size=grid_half)
    print(f"  Ground plane at Y={ground_height:.4f}")

    with open(output_dir / "ground_plane.json", "w", encoding="utf-8") as f:
        json.dump({
            "height": ground_height,
            "normal": [0.0, 1.0, 0.0],
            "coordinate_system": "vggsfm_world_gravity_aligned",
        }, f, indent=2)

    sparse_ply_src = project_dir / "geometry" / "sparse_pointcloud.ply"
    sparse_ply_dst = output_dir / "sparse_pointcloud.ply"
    if sparse_ply_src.exists():
        print(f"  Transforming sparse pointcloud (gravity align + scale 1/{world_scale:.1f})...")
        transform_sparse_ply(
            sparse_ply_src, sparse_ply_dst, gravity_rotation, world_scale, camera_centroid
        )

    blender_ok, blender_msg = check_blender_available()
    if not blender_ok:
        print(f"  Warning: {blender_msg} — skipping geometry ABC export")
        return exported

    ply_targets = [("ground plane", ground_ply, output_dir / "ground_plane.abc")]
    sparse_ply_dst = output_dir / "sparse_pointcloud.ply"
    if sparse_ply_dst.exists():
        ply_targets.append(("sparse pointcloud", sparse_ply_dst, output_dir / "sparse_pointcloud.abc"))

    for label, ply_path, abc_path in ply_targets:
        print(f"  Exporting {label} Alembic...")
        export_ply_to_alembic(ply_path, abc_path, fps)
        if abc_path.exists():
            exported.append(abc_path)

    return exported


def write_combined_intrinsics(
    output_path: Path,
    f_slahmr: float,
    vggsfm_intrinsics: dict,
) -> None:
    """Write output intrinsics with SLAHMR focal and VGGSfM geometry.

    SLAHMR's focal is required for correct 2D projection of the aligned
    body. VGGSfM provides principal point and image dimensions.

    Args:
        output_path: Path to write intrinsics.json.
        f_slahmr: SLAHMR focal length in pixels.
        vggsfm_intrinsics: VGGSfM intrinsics dict (cx, cy, width, height).
    """
    intrinsics = {
        "fx": f_slahmr,
        "fy": f_slahmr,
        "cx": vggsfm_intrinsics.get("cx", vggsfm_intrinsics.get("width", 1920) / 2),
        "cy": vggsfm_intrinsics.get("cy", vggsfm_intrinsics.get("height", 1080) / 2),
        "width": vggsfm_intrinsics.get("width", 1920),
        "height": vggsfm_intrinsics.get("height", 1080),
        "model": "SLAHMR_ESTIMATED",
        "params": [f_slahmr, f_slahmr],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(intrinsics, f, indent=2)


def export_camera(output_dir: Path, fps: int = 24) -> List[Path]:
    """Export camera Alembic and USD from extrinsics/intrinsics in output_dir.

    Args:
        output_dir: Directory containing extrinsics.json and intrinsics.json.
        fps: Frames per second.

    Returns:
        List of exported camera file paths.
    """
    from blender import (
        check_blender_available,
        export_camera_to_alembic,
        export_camera_to_usd,
    )

    available, message = check_blender_available()
    if not available:
        print(f"Warning: {message} — skipping camera export", file=sys.stderr)
        return []

    exported: List[Path] = []

    abc_path = output_dir / "camera.abc"
    print("  Exporting camera Alembic...")
    export_camera_to_alembic(
        camera_dir=output_dir,
        output_path=abc_path,
        fps=fps,
        start_frame=1,
        camera_name="mocap_mmcam",
    )
    if abc_path.exists():
        exported.append(abc_path)

    usd_path = output_dir / "camera.usd"
    print("  Exporting camera USD...")
    export_camera_to_usd(
        camera_dir=output_dir,
        output_path=usd_path,
        fps=fps,
        start_frame=1,
        camera_name="mocap_mmcam",
    )
    if usd_path.exists():
        exported.append(usd_path)

    return exported


def _write_obj_sequence(meshes: List[Tuple], obj_dir: Path) -> None:
    """Write mesh sequence as numbered OBJ files.

    Args:
        meshes: List of (vertices, faces) tuples.
        obj_dir: Directory to write OBJ files into.
    """
    for frame_idx, (vertices, faces) in enumerate(meshes):
        if frame_idx % 100 == 0:
            print(f"    Frame {frame_idx}/{len(meshes)}...")
        obj_path = obj_dir / f"frame_{frame_idx:05d}.obj"
        with open(obj_path, "w", encoding="utf-8") as f:
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def export_scene(
    meshes: List[Tuple],
    camera_dir: Path,
    output_dir: Path,
    formats: List[str],
    fps: int = 24,
    ply_files: Optional[List[Path]] = None,
) -> List[Path]:
    """Export combined mesh + camera scene files.

    Writes OBJ sequence to a temp directory, then calls Blender to produce
    single Alembic/USD files containing both the animated mesh and camera.

    When ply_files is provided, also produces a combined.abc that includes
    the static geometry alongside the animated mesh and camera.

    Args:
        meshes: List of (vertices, faces) tuples.
        camera_dir: Directory containing extrinsics.json and intrinsics.json.
        output_dir: Directory for output scene files.
        formats: List of format strings ("abc", "usd").
        fps: Frames per second.
        ply_files: Optional PLY files to include in combined.abc.

    Returns:
        List of successfully exported file paths.
    """
    from blender import (
        check_blender_available,
        export_scene_to_alembic,
        export_scene_to_usd,
    )

    available, message = check_blender_available()
    if not available:
        print(f"Warning: {message} — skipping scene export", file=sys.stderr)
        return []

    temp_dir = Path(tempfile.mkdtemp(prefix="mocap_scene_"))
    obj_dir = temp_dir / "obj_sequence"
    obj_dir.mkdir()

    try:
        print(f"  Writing {len(meshes)} OBJ frames for scene export...")
        _write_obj_sequence(meshes, obj_dir)

        exported: List[Path] = []
        for fmt in formats:
            if fmt == "abc":
                out = output_dir / "scene.abc"
                print("  Exporting scene Alembic...")
                export_scene_to_alembic(
                    mesh_dir=obj_dir,
                    camera_dir=camera_dir,
                    output_path=out,
                    fps=fps,
                    start_frame=1,
                    camera_name="mocap_mmcam",
                )
                if out.exists():
                    exported.append(out)

                if ply_files:
                    combined = output_dir / "combined.abc"
                    print("  Exporting combined scene + geometry Alembic...")
                    export_scene_to_alembic(
                        mesh_dir=obj_dir,
                        camera_dir=camera_dir,
                        output_path=combined,
                        fps=fps,
                        start_frame=1,
                        camera_name="mocap_mmcam",
                        ply_files=ply_files,
                    )
                    if combined.exists():
                        exported.append(combined)

            elif fmt == "usd":
                out = output_dir / "scene.usd"
                print("  Exporting scene USD...")
                export_scene_to_usd(
                    mesh_dir=obj_dir,
                    camera_dir=camera_dir,
                    output_path=out,
                    fps=fps,
                    start_frame=1,
                    camera_name="mocap_mmcam",
                )
                if out.exists():
                    exported.append(out)
        return exported
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_alignment(
    project_dir: Path,
    fps: int = 24,
    mocap_person: str = "person",
    formats: Optional[List[str]] = None,
) -> bool:
    """Run the full mocap-to-matchmove alignment pipeline.

    Args:
        project_dir: Project directory.
        fps: Export frame rate.
        mocap_person: Person folder under mocap/.
        formats: Export formats (default: abc, usd).

    Returns:
        True if all exports successful.
    """
    if formats is None:
        formats = ["abc", "usd"]

    mocap_dir = project_dir / "mocap" / mocap_person
    motion_path = mocap_dir / "motion.pkl"
    slahmr_npz = mocap_dir / "slahmr" / "slahmr_stitched.npz"
    camera_dir = project_dir / "camera"
    output_dir = project_dir / "mocap_and_mmcam"

    for path, label in [
        (motion_path, "motion.pkl"),
        (slahmr_npz, "slahmr_stitched.npz"),
        (camera_dir / "extrinsics.json", "VGGSfM extrinsics.json"),
        (camera_dir / "intrinsics.json", "VGGSfM intrinsics.json"),
    ]:
        if not path.exists():
            print(f"Error: {label} not found: {path}", file=sys.stderr)
            return False

    print(f"\n{'=' * 60}")
    print("Align Mocap Body to Matchmove Camera")
    print("=" * 60)

    cam_R, cam_t, f_slahmr, slahmr_trans_0 = load_slahmr_camera(slahmr_npz)
    print(f"  SLAHMR: {cam_R.shape[0]} frames, f={f_slahmr:.1f}px")

    c2w, vggsfm_intrinsics = load_vggsfm_camera(camera_dir)
    f_vggsfm = float(vggsfm_intrinsics["fx"])
    print(f"  VGGSfM: {c2w.shape[0]} frames, f={f_vggsfm:.1f}px")
    print(f"  Output camera: SLAHMR focal ({f_slahmr:.1f}px) + VGGSfM trajectory")

    motion_data = load_motion_data(motion_path)
    if motion_data is None:
        return False
    n_motion = len(motion_data["poses"])
    print(f"  Motion: {n_motion} frames")

    n_slahmr = cam_R.shape[0]
    n_vggsfm = c2w.shape[0]
    if not (n_motion == n_slahmr == n_vggsfm):
        print(
            f"Error: Frame count mismatch: motion={n_motion}, "
            f"slahmr={n_slahmr}, vggsfm={n_vggsfm}",
            file=sys.stderr,
        )
        return False

    pelvis_offset = compute_pelvis_offset(motion_data["trans"][0], slahmr_trans_0)
    print(
        f"  Pelvis offset: "
        f"[{pelvis_offset[0]:.4f}, {pelvis_offset[1]:.4f}, {pelvis_offset[2]:.4f}]"
    )

    gender = motion_data.get("gender", "neutral")
    model_path = get_body_model_path(gender)
    if model_path is None:
        return False

    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    meshes = generate_meshes(motion_data, model_path, device)
    if meshes is None:
        return False

    print(f"  Transforming {len(meshes)} frames to VGGSfM world space...")
    meshes = transform_meshes(meshes, cam_R, cam_t, c2w, pelvis_offset)

    output_dir.mkdir(parents=True, exist_ok=True)

    tpose_path = output_dir / "tpose.obj"
    betas = motion_data.get("betas")
    export_tpose(model_path, tpose_path, betas=betas, gender=gender, device=device)

    all_success = True
    exported: List[Path] = []

    for fmt in formats:
        if fmt == "abc":
            out = output_dir / "body_motion.abc"
            ok = export_alembic(meshes, out, fps)
        elif fmt == "usd":
            out = output_dir / "body_motion.usd"
            ok = export_usd(meshes, out, fps)
        else:
            print(f"Warning: Unknown format '{fmt}', skipping", file=sys.stderr)
            continue
        if ok:
            exported.append(out)
        else:
            all_success = False

    shutil.copy2(camera_dir / "extrinsics.json", output_dir / "extrinsics.json")
    write_combined_intrinsics(output_dir / "intrinsics.json", f_slahmr, vggsfm_intrinsics)

    camera_paths = export_camera(output_dir, fps)
    exported.extend(camera_paths)

    gravity_rot = compute_gravity_rotation(c2w, camera_dir)
    world_scale = compute_world_scale(c2w, cam_R, cam_t)
    camera_centroid = c2w[:, :3, 3].mean(axis=0)
    print(f"  World scale (VGGSfM/metric): {world_scale:.1f}x")
    geometry_paths = export_static_geometry(
        output_dir, meshes, project_dir, fps, gravity_rot, world_scale, camera_centroid
    )
    exported.extend(geometry_paths)

    ply_files = [
        p for p in [output_dir / "ground_plane.ply", output_dir / "sparse_pointcloud.ply"]
        if p.exists()
    ]
    scene_paths = export_scene(meshes, output_dir, output_dir, formats, fps, ply_files=ply_files)
    exported.extend(scene_paths)

    if tpose_path.exists():
        exported.append(tpose_path)

    print(f"\n{'=' * 60}")
    print("Alignment Complete")
    print("=" * 60)
    for p in exported:
        print(f"  {p}")
    print()

    return all_success


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Align mocap body to matchmove camera world space",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "project_dir",
        type=Path,
        help="Project directory",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Export frame rate (default: 24)",
    )
    parser.add_argument(
        "--mocap-person",
        default="person",
        help="Person folder under mocap/ (default: person)",
    )
    parser.add_argument(
        "--format",
        "-f",
        default="all",
        help="Export format(s): abc, usd, or comma-separated (default: all = abc,usd)",
    )

    args = parser.parse_args()
    require_conda_env(REQUIRED_ENV)

    project_dir = args.project_dir.resolve()
    if not project_dir.exists():
        print(f"Error: Project directory not found: {project_dir}", file=sys.stderr)
        sys.exit(1)

    formats = parse_formats(args.format)
    if not formats:
        print("Error: No valid formats specified", file=sys.stderr)
        sys.exit(1)

    success = run_alignment(
        project_dir=project_dir,
        fps=args.fps,
        mocap_person=args.mocap_person,
        formats=formats,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
