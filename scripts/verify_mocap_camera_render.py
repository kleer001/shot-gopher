#!/usr/bin/env python3
"""Verify GVHMR camera extraction by comparing incam vs exported camera projection.

Reads hmr4d_results.pt and runs the SMPL body model to get mesh vertices in
both camera-space (incam) and world-space (global).  Then renders two overlays:
  Left  (green):  incam vertices projected with K only (ground truth)
  Right (orange): world vertices projected through exported camera (extrinsics.json)

If the camera extraction is correct, both overlays should be identical.

Must run in the gvhmr conda env (needs pytorch3d, smplx, einops).

Usage:
    conda run -n gvhmr python scripts/verify_mocap_camera_render.py <project_dir>
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch


GVHMR_ROOT = Path(__file__).resolve().parent.parent / ".vfx_pipeline" / "GVHMR"


def _smpl_forward(smpl_params: dict, gvhmr_root: Path) -> np.ndarray:
    """Run SMPL forward pass to get mesh vertices.

    Args:
        smpl_params: Dict with global_orient, body_pose, betas, transl tensors.
        gvhmr_root: GVHMR installation root.

    Returns:
        (F, V, 3) numpy array of SMPL vertices.
    """
    old_cwd = os.getcwd()
    try:
        os.chdir(str(gvhmr_root))
        if str(gvhmr_root) not in sys.path:
            sys.path.insert(0, str(gvhmr_root))

        from hmr4d.utils.smplx_utils import make_smplx

        smplx_model = make_smplx("supermotion")
        smplx2smpl = torch.load(
            str(gvhmr_root / "hmr4d" / "utils" / "body_model" / "smplx2smpl_sparse.pt"),
            map_location='cpu',
        )

        with torch.no_grad():
            smplx_out = smplx_model(**smpl_params)
            verts = torch.stack([smplx2smpl @ v for v in smplx_out.vertices])

        return verts.numpy()
    finally:
        os.chdir(old_cwd)


def load_gvhmr_data(project_dir: Path) -> dict:
    """Load GVHMR results and compute mesh vertices.

    Returns dict with:
        verts_incam: (F, V, 3) mesh in per-frame camera space
        verts_global: (F, V, 3) mesh in world space
        K: (3, 3) intrinsics
    """
    gvhmr_dir = project_dir / "mocap" / "person" / "gvhmr"
    pt_files = list(gvhmr_dir.rglob("hmr4d*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No hmr4d_results.pt in {gvhmr_dir}")

    gvhmr_data = torch.load(pt_files[0], map_location='cpu', weights_only=False)

    print("  Running SMPL forward (incam)...")
    verts_incam = _smpl_forward(gvhmr_data["smpl_params_incam"], GVHMR_ROOT)

    print("  Running SMPL forward (global)...")
    verts_global = _smpl_forward(gvhmr_data["smpl_params_global"], GVHMR_ROOT)

    K = gvhmr_data["K_fullimg"]
    if hasattr(K, 'numpy'):
        K = K.numpy()
    K = np.array(K, dtype=np.float64)
    if K.ndim == 3:
        K = K[0]

    return {
        "verts_incam": verts_incam,
        "verts_global": verts_global,
        "K": K,
    }


def load_exported_camera(project_dir: Path) -> list:
    """Load per-frame camera-to-world from extrinsics.json."""
    ext_path = project_dir / "mocap_camera" / "extrinsics.json"
    if not ext_path.exists():
        raise FileNotFoundError(f"No extrinsics.json at {ext_path}")

    with open(ext_path, encoding='utf-8') as f:
        data = json.load(f)

    return [np.array(m, dtype=np.float64) for m in data]


def project_vertices(
    verts_3d: np.ndarray,
    K: np.ndarray,
    w2c: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Project 3D vertices to 2D pixel coordinates.

    Args:
        verts_3d: (V, 3) vertices.
        K: (3, 3) intrinsic matrix.
        w2c: (4, 4) world-to-camera, or None if already in camera space.

    Returns:
        (V, 2) pixel coordinates, -1 for behind-camera points.
    """
    if w2c is not None:
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        cam_verts = (R @ verts_3d.T).T + t
    else:
        cam_verts = verts_3d

    proj = (K @ cam_verts.T).T
    z = proj[:, 2]
    mask = z > 0.01
    px = np.full(len(z), -1.0)
    py = np.full(len(z), -1.0)
    px[mask] = proj[mask, 0] / z[mask]
    py[mask] = proj[mask, 1] / z[mask]
    return np.stack([px, py], axis=-1)


def draw_vertex_dots(
    img: np.ndarray,
    pts_2d: np.ndarray,
    color: tuple,
    width: int,
    height: int,
    radius: int = 1,
) -> None:
    """Draw vertex positions as colored dots."""
    for x, y in pts_2d:
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < width and 0 <= iy < height:
            y_lo = max(0, iy - radius)
            y_hi = min(height, iy + radius + 1)
            x_lo = max(0, ix - radius)
            x_hi = min(width, ix + radius + 1)
            img[y_lo:y_hi, x_lo:x_hi] = color


def render_verification(
    project_dir: Path,
    output_dir: Optional[Path] = None,
    max_frames: Optional[int] = None,
    skip_every: int = 1,
) -> Path:
    """Render verification frames comparing incam vs exported camera.

    For each frame, renders side-by-side:
      Left  (green dots):  incam vertices projected with K (ground truth)
      Right (orange dots): world vertices projected through exported camera

    Args:
        project_dir: Project directory.
        output_dir: Output directory (default: project/verify_camera/).
        max_frames: Max frames to render.
        skip_every: Render every Nth frame.

    Returns:
        Output directory path.
    """
    from PIL import Image

    project = Path(project_dir)
    output_dir = output_dir or project / "verify_camera"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading GVHMR results...")
    gvhmr = load_gvhmr_data(project)
    print(f"  incam:  {gvhmr['verts_incam'].shape}")
    print(f"  global: {gvhmr['verts_global'].shape}")

    print("Loading exported camera...")
    c2w_matrices = load_exported_camera(project)
    print(f"  {len(c2w_matrices)} frames")

    K = gvhmr["K"]
    width = int(K[0, 2] * 2)
    height = int(K[1, 2] * 2)

    frames_dir = project / "source" / "frames"
    source_frames = sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg"))

    n_gvhmr = len(gvhmr["verts_incam"])
    n_cam = len(c2w_matrices)
    n_render = min(n_gvhmr, n_cam)
    if max_frames:
        n_render = min(n_render, max_frames * skip_every)

    print(f"\nRendering {(n_render + skip_every - 1) // skip_every} frames "
          f"({width}x{height}) to {output_dir}")

    for i in range(0, n_render, skip_every):
        if i < len(source_frames):
            bg = np.array(Image.open(source_frames[i]).convert("RGB").resize((width, height)))
        else:
            bg = np.full((height, width, 3), 40, dtype=np.uint8)

        img_incam = bg.copy()
        img_camera = bg.copy()

        pts_incam = project_vertices(gvhmr["verts_incam"][i], K)
        draw_vertex_dots(img_incam, pts_incam, (0, 255, 0), width, height)

        c2w = c2w_matrices[min(i, n_cam - 1)]
        w2c = np.linalg.inv(c2w)
        pts_cam = project_vertices(gvhmr["verts_global"][i], K, w2c)
        draw_vertex_dots(img_camera, pts_cam, (255, 100, 0), width, height)

        combined = np.concatenate([img_incam, img_camera], axis=1)

        out_path = output_dir / f"verify_{i:04d}.png"
        Image.fromarray(combined).save(out_path)
        sys.stdout.write(f"\r  Frame {i+1}/{n_render}")
        sys.stdout.flush()

    readme_path = output_dir / "README.txt"
    readme_path.write_text(
        "Left (green):  GVHMR incam vertices projected directly (ground truth)\n"
        "Right (orange): World vertices projected through exported camera\n\n"
        "If camera extraction is correct, both sides should be identical.\n"
    )

    print(f"\n\nDone.")
    print(f"  Left  (green):  incam projected directly (ground truth)")
    print(f"  Right (orange): world + exported camera")
    incam_mp4 = project / "mocap" / "person" / "gvhmr" / "_gvhmr_input" / "1_incam.mp4"
    if incam_mp4.exists():
        print(f"\n  Reference: {incam_mp4}")
    print(f"  Output:    {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("project_dir", type=Path, help="Project directory")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to render")
    parser.add_argument("--skip", type=int, default=10, help="Render every Nth frame (default: 10)")
    parser.add_argument("--output", type=Path, default=None, help="Output directory")
    args = parser.parse_args()

    render_verification(
        project_dir=args.project_dir.resolve(),
        output_dir=args.output,
        max_frames=args.max_frames,
        skip_every=args.skip,
    )


if __name__ == "__main__":
    main()
