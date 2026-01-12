#!/usr/bin/env python3
"""UV-based mesh deformation for animating clothed ECON meshes with SMPL-X motion.

This script transfers animation from SMPL-X (driven by WHAM motion data) to
higher-resolution ECON clothed meshes using UV-space correspondence.

Why UV-based (not distance-based):
- Distance-based binding creates artifacts where geometry folds (armpits, thighs)
- UV coordinates are pose-invariant - vertices far apart on body surface stay
  far apart in UV space even when mesh folds in 3D

Pipeline:
    1. REST POSE: Build correspondence (ECON vertex -> SMPL-X triangle via UV)
    2. EACH FRAME: Interpolate ECON positions from animated SMPL-X + stored offsets

Usage:
    python mesh_deform.py <project_dir> [options]

Example:
    # Full deformation pipeline
    python mesh_deform.py /path/to/project \\
        --smplx-rest mocap/smplx_rest.obj \\
        --econ-rest mocap/econ/mesh_0001.obj \\
        --smplx-sequence mocap/smplx_animated/ \\
        --output mocap/econ_animated/

    # With smoothing control map
    python mesh_deform.py /path/to/project \\
        --smoothing-map mocap/smoothing_weights.png
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import numpy as np


def check_dependencies() -> bool:
    """Check if required dependencies are available."""
    required = {
        "numpy": "numpy",
        "scipy": "scipy",
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


# =============================================================================
# UV-Space Triangle Lookup
# =============================================================================

class UVTriangleLookup:
    """Efficient UV-space triangle lookup using spatial hashing.

    Builds a grid-based acceleration structure for finding which triangle
    contains a given UV coordinate.
    """

    def __init__(self, uvs: np.ndarray, faces: np.ndarray, grid_resolution: int = 64):
        """Initialize UV triangle lookup.

        Args:
            uvs: UV coordinates [N_verts, 2] in [0, 1] range
            faces: Triangle indices [N_faces, 3]
            grid_resolution: Number of grid cells per UV axis
        """
        self.uvs = uvs
        self.faces = faces
        self.grid_res = grid_resolution
        self.grid: Dict[Tuple[int, int], List[int]] = {}

        self._build_grid()

    def _build_grid(self):
        """Build spatial hash grid for UV triangles."""
        for face_idx, face in enumerate(self.faces):
            # Get UV coordinates of triangle vertices
            tri_uvs = self.uvs[face]  # [3, 2]

            # Compute bounding box in grid space
            uv_min = tri_uvs.min(axis=0)
            uv_max = tri_uvs.max(axis=0)

            grid_min = np.floor(uv_min * self.grid_res).astype(int)
            grid_max = np.ceil(uv_max * self.grid_res).astype(int)

            # Clamp to valid range
            grid_min = np.clip(grid_min, 0, self.grid_res - 1)
            grid_max = np.clip(grid_max, 0, self.grid_res - 1)

            # Add triangle to all overlapping cells
            for gx in range(grid_min[0], grid_max[0] + 1):
                for gy in range(grid_min[1], grid_max[1] + 1):
                    key = (gx, gy)
                    if key not in self.grid:
                        self.grid[key] = []
                    self.grid[key].append(face_idx)

    def find_triangle(self, uv: np.ndarray) -> Tuple[int, np.ndarray]:
        """Find triangle containing UV coordinate and compute barycentric weights.

        Args:
            uv: UV coordinate [2]

        Returns:
            Tuple of (triangle_index, barycentric_weights [3])
            Returns (-1, None) if no triangle found
        """
        # Get grid cell
        gx = int(np.clip(uv[0] * self.grid_res, 0, self.grid_res - 1))
        gy = int(np.clip(uv[1] * self.grid_res, 0, self.grid_res - 1))

        candidates = self.grid.get((gx, gy), [])

        # Check each candidate triangle
        for face_idx in candidates:
            face = self.faces[face_idx]
            tri_uvs = self.uvs[face]

            bary = self._compute_barycentric(uv, tri_uvs)

            # Check if point is inside triangle (all weights >= 0, sum to 1)
            if bary is not None and np.all(bary >= -1e-6) and np.abs(bary.sum() - 1.0) < 1e-6:
                return face_idx, np.clip(bary, 0, 1)

        # Fallback: search neighboring cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                key = (gx + dx, gy + dy)
                for face_idx in self.grid.get(key, []):
                    face = self.faces[face_idx]
                    tri_uvs = self.uvs[face]

                    bary = self._compute_barycentric(uv, tri_uvs)

                    if bary is not None and np.all(bary >= -1e-6):
                        return face_idx, np.clip(bary, 0, 1)

        return -1, None

    def _compute_barycentric(self, p: np.ndarray, tri: np.ndarray) -> Optional[np.ndarray]:
        """Compute barycentric coordinates of point p in triangle tri.

        Args:
            p: Point [2]
            tri: Triangle vertices [3, 2]

        Returns:
            Barycentric weights [3] or None if degenerate
        """
        v0 = tri[1] - tri[0]
        v1 = tri[2] - tri[0]
        v2 = p - tri[0]

        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        denom = dot00 * dot11 - dot01 * dot01
        if abs(denom) < 1e-10:
            return None

        inv_denom = 1.0 / denom
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom
        w = 1.0 - u - v

        return np.array([w, u, v])


# =============================================================================
# Correspondence Building
# =============================================================================

class MeshCorrespondence:
    """Stores UV-based correspondence between target (ECON) and source (SMPL-X) meshes."""

    def __init__(self):
        self.triangle_indices: np.ndarray = None  # [N_target_verts] -> source triangle
        self.bary_weights: np.ndarray = None      # [N_target_verts, 3] -> barycentric
        self.rest_offsets: np.ndarray = None      # [N_target_verts, 3] -> 3D offset
        self.valid_mask: np.ndarray = None        # [N_target_verts] -> bool

        # For smooth offset transformation
        self.local_frames: np.ndarray = None      # [N_target_verts, 3, 3] -> rotation

    def save(self, path: Path):
        """Save correspondence to file."""
        np.savez(
            path,
            triangle_indices=self.triangle_indices,
            bary_weights=self.bary_weights,
            rest_offsets=self.rest_offsets,
            valid_mask=self.valid_mask,
            local_frames=self.local_frames
        )
        print(f"  Saved correspondence: {path}")

    @classmethod
    def load(cls, path: Path) -> 'MeshCorrespondence':
        """Load correspondence from file."""
        data = np.load(path)
        corr = cls()
        corr.triangle_indices = data['triangle_indices']
        corr.bary_weights = data['bary_weights']
        corr.rest_offsets = data['rest_offsets']
        corr.valid_mask = data['valid_mask']
        corr.local_frames = data.get('local_frames', None)
        return corr


def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute per-vertex normals by averaging face normals.

    Args:
        vertices: Vertex positions [N, 3]
        faces: Triangle indices [F, 3]

    Returns:
        Vertex normals [N, 3], normalized
    """
    # Compute face normals
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    face_normals = np.cross(v1 - v0, v2 - v0)
    face_normals = face_normals / (np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-8)

    # Accumulate to vertices
    vertex_normals = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(vertex_normals, faces[:, i], face_normals)

    # Normalize
    vertex_normals = vertex_normals / (np.linalg.norm(vertex_normals, axis=1, keepdims=True) + 1e-8)

    return vertex_normals


def compute_local_frame(normal: np.ndarray) -> np.ndarray:
    """Compute local coordinate frame from normal vector.

    Args:
        normal: Surface normal [3]

    Returns:
        Rotation matrix [3, 3] where Z axis = normal
    """
    n = normal / (np.linalg.norm(normal) + 1e-8)

    # Choose up vector that's not parallel to normal
    up = np.array([0, 1, 0])
    if abs(np.dot(n, up)) > 0.9:
        up = np.array([1, 0, 0])

    # Gram-Schmidt orthogonalization
    tangent = up - np.dot(up, n) * n
    tangent = tangent / (np.linalg.norm(tangent) + 1e-8)

    bitangent = np.cross(n, tangent)

    # Column matrix: [tangent, bitangent, normal]
    return np.column_stack([tangent, bitangent, n])


def build_correspondence(
    source_mesh,  # trimesh: SMPL-X rest pose
    target_mesh,  # trimesh: ECON rest pose
    verbose: bool = True
) -> MeshCorrespondence:
    """Build UV-based correspondence from target (ECON) to source (SMPL-X).

    For each target vertex:
    1. Look up its UV coordinate
    2. Find the source triangle containing that UV
    3. Compute barycentric weights within that triangle
    4. Compute 3D offset from interpolated source position

    Args:
        source_mesh: SMPL-X mesh at rest pose (has UVs)
        target_mesh: ECON mesh at rest pose (has matching UVs)
        verbose: Print progress

    Returns:
        MeshCorrespondence object with binding data
    """
    if verbose:
        print("\n  Building UV correspondence...")
        print(f"    Source (SMPL-X): {len(source_mesh.vertices)} vertices")
        print(f"    Target (ECON): {len(target_mesh.vertices)} vertices")

    # Extract UV coordinates
    # trimesh stores UVs in visual.uv
    if not hasattr(source_mesh.visual, 'uv') or source_mesh.visual.uv is None:
        raise ValueError("Source mesh has no UV coordinates")
    if not hasattr(target_mesh.visual, 'uv') or target_mesh.visual.uv is None:
        raise ValueError("Target mesh has no UV coordinates")

    source_uvs = np.array(source_mesh.visual.uv)
    target_uvs = np.array(target_mesh.visual.uv)
    source_faces = np.array(source_mesh.faces)

    if verbose:
        print(f"    Source UVs: {source_uvs.shape}")
        print(f"    Target UVs: {target_uvs.shape}")

    # Build UV lookup structure for source mesh
    uv_lookup = UVTriangleLookup(source_uvs, source_faces)

    # Allocate correspondence arrays
    n_target = len(target_mesh.vertices)
    corr = MeshCorrespondence()
    corr.triangle_indices = np.zeros(n_target, dtype=np.int32)
    corr.bary_weights = np.zeros((n_target, 3), dtype=np.float32)
    corr.rest_offsets = np.zeros((n_target, 3), dtype=np.float32)
    corr.valid_mask = np.zeros(n_target, dtype=bool)

    # Compute source vertex normals for local frames
    source_normals = compute_vertex_normals(
        np.array(source_mesh.vertices),
        source_faces
    )

    # Build correspondence for each target vertex
    valid_count = 0
    for i in range(n_target):
        target_uv = target_uvs[i]

        # Find source triangle containing this UV
        tri_idx, bary = uv_lookup.find_triangle(target_uv)

        if tri_idx >= 0:
            corr.triangle_indices[i] = tri_idx
            corr.bary_weights[i] = bary
            corr.valid_mask[i] = True
            valid_count += 1

            # Compute interpolated source position
            face = source_faces[tri_idx]
            source_pos = (
                source_mesh.vertices[face[0]] * bary[0] +
                source_mesh.vertices[face[1]] * bary[1] +
                source_mesh.vertices[face[2]] * bary[2]
            )

            # Store offset from source to target
            corr.rest_offsets[i] = target_mesh.vertices[i] - source_pos
        else:
            # No valid correspondence - will use fallback
            corr.triangle_indices[i] = -1
            corr.valid_mask[i] = False

    # Compute local frames for valid vertices (for offset transformation)
    corr.local_frames = np.zeros((n_target, 3, 3), dtype=np.float32)
    for i in range(n_target):
        if corr.valid_mask[i]:
            face = source_faces[corr.triangle_indices[i]]
            bary = corr.bary_weights[i]

            # Interpolate normal
            interp_normal = (
                source_normals[face[0]] * bary[0] +
                source_normals[face[1]] * bary[1] +
                source_normals[face[2]] * bary[2]
            )

            corr.local_frames[i] = compute_local_frame(interp_normal)

    if verbose:
        coverage = valid_count / n_target * 100
        print(f"    Valid correspondence: {valid_count}/{n_target} ({coverage:.1f}%)")

        if coverage < 90:
            print(f"    Warning: Low coverage - check UV overlap between meshes")

    return corr


# =============================================================================
# Per-Frame Deformation
# =============================================================================

def deform_frame(
    source_verts: np.ndarray,
    source_faces: np.ndarray,
    target_rest_verts: np.ndarray,
    correspondence: MeshCorrespondence,
    smoothing_weights: Optional[np.ndarray] = None,
    offset_mode: str = "smooth"
) -> np.ndarray:
    """Deform target mesh based on animated source mesh.

    Args:
        source_verts: Animated SMPL-X vertices [N_source, 3]
        source_faces: Source triangle indices [F, 3]
        target_rest_verts: Target rest pose vertices (for fallback) [N_target, 3]
        correspondence: Pre-computed UV correspondence
        smoothing_weights: Per-vertex smoothing weights [N_target] in [0, 1]
                          0 = rigid (offset moves exactly with surface)
                          1 = maximum smoothing (offset interpolated more)
        offset_mode: How to transform offsets
                    "rigid" - offset stays fixed relative to local frame
                    "smooth" - linear interpolation (simple, fast)
                    "normal" - offset projected along interpolated normal

    Returns:
        Deformed target vertices [N_target, 3]
    """
    n_target = len(target_rest_verts)
    result = np.zeros((n_target, 3), dtype=np.float32)

    # Compute animated source normals if needed
    if offset_mode in ["rigid", "normal"]:
        source_normals = compute_vertex_normals(source_verts, source_faces)

    for i in range(n_target):
        if not correspondence.valid_mask[i]:
            # Fallback: keep rest position
            result[i] = target_rest_verts[i]
            continue

        tri_idx = correspondence.triangle_indices[i]
        bary = correspondence.bary_weights[i]
        offset = correspondence.rest_offsets[i]
        face = source_faces[tri_idx]

        # Interpolate animated source position
        animated_pos = (
            source_verts[face[0]] * bary[0] +
            source_verts[face[1]] * bary[1] +
            source_verts[face[2]] * bary[2]
        )

        # Transform offset based on mode
        if offset_mode == "smooth":
            # Simple: add offset directly (linear interpolation)
            # This works well for small offsets (tight clothing)
            transformed_offset = offset

        elif offset_mode == "normal":
            # Project offset along interpolated normal
            interp_normal = (
                source_normals[face[0]] * bary[0] +
                source_normals[face[1]] * bary[1] +
                source_normals[face[2]] * bary[2]
            )
            interp_normal = interp_normal / (np.linalg.norm(interp_normal) + 1e-8)

            # Offset magnitude along rest normal
            offset_mag = np.linalg.norm(offset)
            offset_sign = 1.0 if np.dot(offset, interp_normal) >= 0 else -1.0

            transformed_offset = interp_normal * offset_mag * offset_sign

        elif offset_mode == "rigid":
            # Transform offset by animated local frame
            # Requires computing both rest and animated frames
            rest_frame = correspondence.local_frames[i]

            # Compute animated frame
            interp_normal = (
                source_normals[face[0]] * bary[0] +
                source_normals[face[1]] * bary[1] +
                source_normals[face[2]] * bary[2]
            )
            animated_frame = compute_local_frame(interp_normal)

            # Transform: rest_local -> world -> animated_local
            # offset_local = rest_frame.T @ offset
            # transformed = animated_frame @ offset_local
            offset_local = rest_frame.T @ offset
            transformed_offset = animated_frame @ offset_local
        else:
            transformed_offset = offset

        # Apply optional per-vertex smoothing
        if smoothing_weights is not None:
            w = smoothing_weights[i]
            # Blend between rigid offset and smoothed (scaled) offset
            # Higher weight = more damping of offset
            transformed_offset = transformed_offset * (1.0 - w * 0.5)

        result[i] = animated_pos + transformed_offset

    return result


# =============================================================================
# Smoothing Weight Map
# =============================================================================

def load_smoothing_map(
    map_path: Path,
    mesh,  # trimesh with UVs
) -> np.ndarray:
    """Load per-vertex smoothing weights from UV-space image map.

    The image is sampled at each vertex's UV coordinate.
    White (1.0) = maximum smoothing, Black (0.0) = rigid.

    Args:
        map_path: Path to grayscale PNG image
        mesh: Mesh with UV coordinates

    Returns:
        Per-vertex weights [N_verts] in [0, 1]
    """
    from PIL import Image

    img = Image.open(map_path).convert('L')  # Grayscale
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]

    h, w = img_array.shape
    uvs = np.array(mesh.visual.uv)

    weights = np.zeros(len(uvs), dtype=np.float32)

    for i, uv in enumerate(uvs):
        # Sample image at UV (flip V for image coordinates)
        x = int(np.clip(uv[0] * (w - 1), 0, w - 1))
        y = int(np.clip((1 - uv[1]) * (h - 1), 0, h - 1))
        weights[i] = img_array[y, x]

    return weights


def create_default_smoothing_map(output_path: Path, resolution: int = 1024):
    """Create a default smoothing weight map (all gray = 0.5).

    Users can paint this in Photoshop/GIMP to control per-region smoothing.

    Args:
        output_path: Where to save the PNG
        resolution: Image resolution
    """
    from PIL import Image

    # Create gray image (0.5 = moderate smoothing)
    img = Image.new('L', (resolution, resolution), color=128)
    img.save(output_path)
    print(f"  Created default smoothing map: {output_path}")
    print(f"    Paint white for more smoothing, black for rigid")


# =============================================================================
# Main Pipeline
# =============================================================================

def run_deformation_pipeline(
    project_dir: Path,
    smplx_rest_path: Path,
    econ_rest_path: Path,
    smplx_sequence_path: Path,
    output_dir: Path,
    smoothing_map_path: Optional[Path] = None,
    offset_mode: str = "smooth",
    correspondence_cache: Optional[Path] = None,
    verbose: bool = True
) -> bool:
    """Run full mesh deformation pipeline.

    Args:
        project_dir: Project root directory
        smplx_rest_path: Path to SMPL-X rest pose mesh (.obj)
        econ_rest_path: Path to ECON rest pose mesh (.obj)
        smplx_sequence_path: Directory with animated SMPL-X meshes (frame_*.obj)
        output_dir: Output directory for deformed meshes
        smoothing_map_path: Optional UV-space smoothing weight image
        offset_mode: Offset transformation mode ("smooth", "rigid", "normal")
        correspondence_cache: Path to save/load correspondence (.npz)
        verbose: Print progress

    Returns:
        True if successful
    """
    import trimesh

    print(f"\n{'=' * 60}")
    print("UV-Based Mesh Deformation")
    print("=" * 60)
    print(f"Project: {project_dir}")
    print(f"Offset mode: {offset_mode}")
    print()

    try:
        # Load meshes
        print("Loading meshes...")
        smplx_rest = trimesh.load(smplx_rest_path, process=False)
        econ_rest = trimesh.load(econ_rest_path, process=False)

        print(f"  SMPL-X rest: {len(smplx_rest.vertices)} verts, {len(smplx_rest.faces)} faces")
        print(f"  ECON rest: {len(econ_rest.vertices)} verts, {len(econ_rest.faces)} faces")

        # Build or load correspondence
        if correspondence_cache and correspondence_cache.exists():
            print(f"\nLoading cached correspondence: {correspondence_cache}")
            correspondence = MeshCorrespondence.load(correspondence_cache)
        else:
            correspondence = build_correspondence(smplx_rest, econ_rest, verbose=verbose)

            if correspondence_cache:
                correspondence_cache.parent.mkdir(parents=True, exist_ok=True)
                correspondence.save(correspondence_cache)

        # Load smoothing map if provided
        smoothing_weights = None
        if smoothing_map_path and smoothing_map_path.exists():
            print(f"\nLoading smoothing map: {smoothing_map_path}")
            smoothing_weights = load_smoothing_map(smoothing_map_path, econ_rest)
            print(f"  Weight range: [{smoothing_weights.min():.2f}, {smoothing_weights.max():.2f}]")

        # Find animated SMPL-X frames
        smplx_frames = sorted(smplx_sequence_path.glob("*.obj"))
        if not smplx_frames:
            # Try other patterns
            smplx_frames = sorted(smplx_sequence_path.glob("frame_*.obj"))
        if not smplx_frames:
            smplx_frames = sorted(smplx_sequence_path.glob("mesh_*.obj"))

        if not smplx_frames:
            print(f"Error: No OBJ files found in {smplx_sequence_path}", file=sys.stderr)
            return False

        print(f"\nFound {len(smplx_frames)} animation frames")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get source face connectivity (constant)
        source_faces = np.array(smplx_rest.faces)
        target_rest_verts = np.array(econ_rest.vertices)
        target_faces = np.array(econ_rest.faces)
        target_uvs = np.array(econ_rest.visual.uv) if hasattr(econ_rest.visual, 'uv') else None

        # Process each frame
        print(f"\nDeforming {len(smplx_frames)} frames...")

        for frame_idx, smplx_frame_path in enumerate(smplx_frames):
            # Load animated SMPL-X
            smplx_animated = trimesh.load(smplx_frame_path, process=False)
            source_verts = np.array(smplx_animated.vertices)

            # Deform ECON mesh
            deformed_verts = deform_frame(
                source_verts=source_verts,
                source_faces=source_faces,
                target_rest_verts=target_rest_verts,
                correspondence=correspondence,
                smoothing_weights=smoothing_weights,
                offset_mode=offset_mode
            )

            # Create output mesh
            deformed_mesh = trimesh.Trimesh(
                vertices=deformed_verts,
                faces=target_faces,
                process=False
            )

            # Preserve UVs if available
            if target_uvs is not None:
                deformed_mesh.visual = trimesh.visual.TextureVisuals(uv=target_uvs)

            # Save output
            frame_name = smplx_frame_path.stem
            output_path = output_dir / f"{frame_name}.obj"
            deformed_mesh.export(output_path)

            if verbose and (frame_idx + 1) % 10 == 0:
                print(f"    [{frame_idx + 1}/{len(smplx_frames)}] {output_path.name}")

        print(f"\n  Deformed {len(smplx_frames)} frames to: {output_dir}")
        return True

    except Exception as e:
        print(f"Error in deformation pipeline: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="UV-based mesh deformation for clothed character animation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "project_dir",
        type=Path,
        help="Project directory"
    )
    parser.add_argument(
        "--smplx-rest",
        type=Path,
        required=True,
        help="SMPL-X rest pose mesh (.obj)"
    )
    parser.add_argument(
        "--econ-rest",
        type=Path,
        required=True,
        help="ECON rest pose mesh (.obj)"
    )
    parser.add_argument(
        "--smplx-sequence",
        type=Path,
        required=True,
        help="Directory with animated SMPL-X meshes"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for deformed meshes"
    )
    parser.add_argument(
        "--smoothing-map",
        type=Path,
        default=None,
        help="UV-space smoothing weight image (grayscale PNG)"
    )
    parser.add_argument(
        "--offset-mode",
        choices=["smooth", "rigid", "normal"],
        default="smooth",
        help="How to transform offsets: smooth (linear), rigid (local frame), normal (along surface)"
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Path to cache correspondence data (.npz)"
    )
    parser.add_argument(
        "--create-smoothing-template",
        type=Path,
        default=None,
        help="Create a default smoothing map template at this path"
    )

    args = parser.parse_args()

    if not check_dependencies():
        sys.exit(1)

    # Handle smoothing template creation
    if args.create_smoothing_template:
        create_default_smoothing_map(args.create_smoothing_template)
        print("\nEdit this image to control per-region smoothing:")
        print("  - White (255): Maximum smoothing / damping")
        print("  - Black (0): Rigid offset (clothing moves exactly with body)")
        print("  - Gray (128): Moderate smoothing (default)")
        sys.exit(0)

    project_dir = args.project_dir.resolve()
    if not project_dir.exists():
        print(f"Error: Project directory not found: {project_dir}", file=sys.stderr)
        sys.exit(1)

    # Resolve paths relative to project
    def resolve_path(p: Path) -> Path:
        if p and not p.is_absolute():
            return project_dir / p
        return p

    success = run_deformation_pipeline(
        project_dir=project_dir,
        smplx_rest_path=resolve_path(args.smplx_rest),
        econ_rest_path=resolve_path(args.econ_rest),
        smplx_sequence_path=resolve_path(args.smplx_sequence),
        output_dir=resolve_path(args.output),
        smoothing_map_path=resolve_path(args.smoothing_map) if args.smoothing_map else None,
        offset_mode=args.offset_mode,
        correspondence_cache=resolve_path(args.cache) if args.cache else None
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
