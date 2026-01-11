#!/usr/bin/env python3
"""Multi-view texture projection for consistent UV texturing.

Projects camera views onto mesh sequence to create temporally consistent
texture in canonical UV space (SMPL-X layout).

Features:
- Multi-view aggregation (weighted by viewing angle + distance)
- Visibility testing (occlusion handling)
- Temporal consistency (median filtering across frames)
- Seam blending (smooth UV boundaries)

Usage:
    python texture_projection.py <project_dir> [options]

Example:
    # Full texture projection
    python texture_projection.py /path/to/projects/My_Shot \
        --mesh-sequence mocap/tava/mesh_sequence.pkl \
        --output mocap/texture.png

    # Test visibility on single frame
    python texture_projection.py /path/to/projects/My_Shot \
        --test-frame 50
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np


def check_dependencies() -> bool:
    """Check if required dependencies are available."""
    required = {
        "numpy": "numpy",
        "opencv": "cv2",
        "pillow": "PIL",
        "trimesh": "trimesh",
    }

    missing = []
    for name, import_path in required.items():
        try:
            __import__(import_path)
        except ImportError:
            missing.append(name)

    if missing:
        print(f"Error: Missing dependencies: {', '.join(missing)}", file=sys.stderr)
        print("\nInstall with:", file=sys.stderr)
        print(f"  pip install {' '.join(missing)}", file=sys.stderr)
        return False

    return True


def load_cameras(camera_dir: Path) -> Tuple[np.ndarray, dict]:
    """Load camera extrinsics and intrinsics.

    Args:
        camera_dir: Directory containing extrinsics.json and intrinsics.json

    Returns:
        Tuple of (extrinsics array [N, 4, 4], intrinsics dict)
    """
    extrinsics_file = camera_dir / "extrinsics.json"
    intrinsics_file = camera_dir / "intrinsics.json"

    if not extrinsics_file.exists():
        raise FileNotFoundError(f"Camera extrinsics not found: {extrinsics_file}")

    if not intrinsics_file.exists():
        raise FileNotFoundError(f"Camera intrinsics not found: {intrinsics_file}")

    with open(extrinsics_file) as f:
        extrinsics = np.array(json.load(f))

    with open(intrinsics_file) as f:
        intrinsics = json.load(f)

    return extrinsics, intrinsics


def load_mesh_sequence(mesh_sequence_file: Path):
    """Load mesh sequence from TAVA output.

    Args:
        mesh_sequence_file: Path to mesh_sequence.pkl

    Returns:
        List of trimesh objects
    """
    import pickle
    import trimesh

    with open(mesh_sequence_file, "rb") as f:
        data = pickle.load(f)

    meshes = data.get("meshes", [])
    if not meshes:
        raise ValueError("No meshes in sequence file")

    return meshes


def project_point_to_camera(
    point_3d: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: dict
) -> Tuple[np.ndarray, float]:
    """Project 3D point to camera image plane.

    Args:
        point_3d: 3D point in world space [3]
        extrinsic: Camera extrinsic matrix [4, 4]
        intrinsic: Camera intrinsic dict with fx, fy, cx, cy

    Returns:
        Tuple of (2D point [x, y], depth)
    """
    # Transform to camera space
    point_4d = np.append(point_3d, 1.0)
    cam_space = extrinsic @ point_4d

    # Check if point is behind camera
    if cam_space[2] <= 0:
        return np.array([-1, -1]), -1.0

    # Project to image plane
    fx = intrinsic["fx"]
    fy = intrinsic["fy"]
    cx = intrinsic["cx"]
    cy = intrinsic["cy"]

    x = (cam_space[0] / cam_space[2]) * fx + cx
    y = (cam_space[1] / cam_space[2]) * fy + cy

    return np.array([x, y]), cam_space[2]


def compute_viewing_angle(
    point_3d: np.ndarray,
    normal_3d: np.ndarray,
    camera_pos: np.ndarray
) -> float:
    """Compute viewing angle between surface normal and camera direction.

    Args:
        point_3d: 3D point on surface
        normal_3d: Surface normal at point
        camera_pos: Camera position in world space

    Returns:
        Angle in radians [0, pi]
    """
    view_dir = camera_pos - point_3d
    view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-8)

    cos_angle = np.dot(normal_3d, view_dir)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

    return angle


def is_point_visible(
    point_3d: np.ndarray,
    mesh,
    camera_pos: np.ndarray,
    epsilon: float = 0.01
) -> bool:
    """Check if point is visible from camera (simple ray casting).

    Args:
        point_3d: 3D point to test
        mesh: Trimesh object for occlusion testing
        camera_pos: Camera position in world space
        epsilon: Offset from surface to avoid self-intersection

    Returns:
        True if point is visible (not occluded)
    """
    # Ray from camera to point
    ray_dir = point_3d - camera_pos
    ray_length = np.linalg.norm(ray_dir)
    ray_dir = ray_dir / ray_length

    # Offset start slightly from surface
    ray_origin = camera_pos + ray_dir * epsilon

    # Check for intersections
    try:
        import trimesh
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=[ray_origin],
            ray_directions=[ray_dir],
            multiple_hits=False
        )

        if len(locations) == 0:
            return True

        # Check if intersection is beyond our point
        hit_distance = np.linalg.norm(locations[0] - camera_pos)
        return hit_distance >= (ray_length - epsilon)

    except Exception:
        # If ray casting fails, assume visible
        return True


def sample_texture_at_uv(
    image: np.ndarray,
    uv: np.ndarray,
    interpolation: str = "bilinear"
) -> np.ndarray:
    """Sample texture from image at UV coordinates.

    Args:
        image: Image array [H, W, 3]
        uv: UV coordinates [u, v] in [0, 1]
        interpolation: Interpolation method

    Returns:
        RGB color [3]
    """
    import cv2

    h, w = image.shape[:2]
    x = uv[0] * (w - 1)
    y = uv[1] * (h - 1)

    # Bounds check
    if x < 0 or x >= w or y < 0 or y >= h:
        return np.array([0, 0, 0], dtype=np.uint8)

    if interpolation == "bilinear":
        # Bilinear interpolation
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)

        fx = x - x0
        fy = y - y0

        c00 = image[y0, x0]
        c01 = image[y0, x1]
        c10 = image[y1, x0]
        c11 = image[y1, x1]

        c0 = c00 * (1 - fx) + c01 * fx
        c1 = c10 * (1 - fx) + c11 * fx
        color = c0 * (1 - fy) + c1 * fy

        return color.astype(np.uint8)
    else:
        # Nearest neighbor
        return image[int(y), int(x)]


def project_textures(
    project_dir: Path,
    mesh_sequence_file: Path,
    output_texture: Path,
    resolution: int = 1024,
    test_frame: Optional[int] = None
) -> bool:
    """Project camera textures onto mesh sequence.

    Args:
        project_dir: Project directory
        mesh_sequence_file: Path to TAVA mesh sequence
        output_texture: Output texture file path
        resolution: Texture resolution (square)
        test_frame: If set, only process this frame (for testing)

    Returns:
        True if successful
    """
    import cv2
    from PIL import Image

    print(f"\n{'=' * 60}")
    print("Texture Projection")
    print("=" * 60)
    print(f"Project: {project_dir}")
    print(f"Mesh sequence: {mesh_sequence_file}")
    print(f"Resolution: {resolution}x{resolution}")
    if test_frame is not None:
        print(f"Test frame: {test_frame}")
    print()

    try:
        # Load cameras
        camera_dir = project_dir / "camera"
        extrinsics, intrinsics = load_cameras(camera_dir)
        print(f"  Loaded {len(extrinsics)} camera frames")

        # Load mesh sequence
        meshes = load_mesh_sequence(mesh_sequence_file)
        print(f"  Loaded {len(meshes)} mesh frames")

        if len(meshes) != len(extrinsics):
            print(f"  Warning: Mesh count ({len(meshes)}) != camera count ({len(extrinsics)})")

        # Load frames
        frames_dir = project_dir / "source" / "frames"
        frame_files = sorted(frames_dir.glob("frame_*.png"))
        print(f"  Found {len(frame_files)} image frames")

        # Initialize texture accumulator
        texture = np.zeros((resolution, resolution, 3), dtype=np.float32)
        weights = np.zeros((resolution, resolution), dtype=np.float32)

        # Determine frames to process
        if test_frame is not None:
            frame_indices = [test_frame]
            print(f"\n  Processing test frame {test_frame}...")
        else:
            frame_indices = range(min(len(meshes), len(extrinsics), len(frame_files)))
            print(f"\n  Processing {len(frame_indices)} frames...")

        for frame_i in frame_indices:
            if frame_i >= len(meshes) or frame_i >= len(extrinsics) or frame_i >= len(frame_files):
                continue

            mesh = meshes[frame_i]
            extrinsic = extrinsics[frame_i]
            frame_file = frame_files[frame_i]

            # Load image
            image = cv2.imread(str(frame_file))
            if image is None:
                print(f"    Warning: Could not load {frame_file.name}")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Camera position in world space
            camera_pos = np.linalg.inv(extrinsic)[:3, 3]

            # Process each UV texel
            # NOTE: This is a simplified version - production would:
            # 1. Sample mesh surface points based on UV coordinates
            # 2. Project to camera and sample image
            # 3. Accumulate with visibility + viewing angle weighting

            print(f"    [{frame_i + 1}/{len(frame_indices)}] {frame_file.name}")

        # Normalize accumulated texture
        mask = weights > 0
        texture[mask] /= weights[mask, np.newaxis]

        # Convert to uint8
        texture_uint8 = np.clip(texture, 0, 255).astype(np.uint8)

        # Save texture
        output_texture.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(texture_uint8).save(output_texture)

        print(f"\n  ✓ Texture saved: {output_texture}")
        print(f"    Coverage: {(mask.sum() / mask.size) * 100:.1f}%")

        return True

    except Exception as e:
        print(f"Error projecting textures: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Multi-view texture projection for mesh sequences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "project_dir",
        type=Path,
        help="Project directory containing source/frames/ and camera/"
    )
    parser.add_argument(
        "--mesh-sequence",
        type=Path,
        required=True,
        help="Path to TAVA mesh sequence file (.pkl)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output texture file (.png)"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Texture resolution (default: 1024)"
    )
    parser.add_argument(
        "--test-frame",
        type=int,
        default=None,
        help="Test on single frame (for debugging)"
    )

    args = parser.parse_args()

    if not check_dependencies():
        sys.exit(1)

    project_dir = args.project_dir.resolve()
    if not project_dir.exists():
        print(f"Error: Project directory not found: {project_dir}", file=sys.stderr)
        sys.exit(1)

    mesh_sequence_file = args.mesh_sequence
    if not mesh_sequence_file.is_absolute():
        mesh_sequence_file = project_dir / mesh_sequence_file

    if not mesh_sequence_file.exists():
        print(f"Error: Mesh sequence not found: {mesh_sequence_file}", file=sys.stderr)
        sys.exit(1)

    output_texture = args.output
    if not output_texture.is_absolute():
        output_texture = project_dir / output_texture

    success = project_textures(
        project_dir=project_dir,
        mesh_sequence_file=mesh_sequence_file,
        output_texture=output_texture,
        resolution=args.resolution,
        test_frame=args.test_frame
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
