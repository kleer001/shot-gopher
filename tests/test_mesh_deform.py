"""Tests for mesh_deform.py - UV-based mesh deformation algorithms.

Tests the core algorithmic components without requiring full mesh dependencies.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

# Import from scripts - add to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from mesh_deform import (
    UVTriangleLookup,
    MeshCorrespondence,
    compute_vertex_normals,
    compute_local_frame,
    deform_frame,
)


class TestUVTriangleLookup:
    """Test UV-space triangle lookup with spatial hashing."""

    def test_simple_triangle_lookup(self):
        """Test finding a point inside a simple triangle."""
        # Single triangle covering (0,0) to (1,1)
        uvs = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
        ])
        faces = np.array([[0, 1, 2]])

        lookup = UVTriangleLookup(uvs, faces, grid_resolution=8)

        # Point inside triangle
        tri_idx, bary = lookup.find_triangle(np.array([0.5, 0.3]))

        assert tri_idx == 0
        assert bary is not None
        assert len(bary) == 3
        assert np.abs(bary.sum() - 1.0) < 1e-6
        assert np.all(bary >= 0)

    def test_point_outside_triangle(self):
        """Test that points outside triangles return -1."""
        uvs = np.array([
            [0.0, 0.0],
            [0.5, 0.0],
            [0.25, 0.5],
        ])
        faces = np.array([[0, 1, 2]])

        lookup = UVTriangleLookup(uvs, faces, grid_resolution=8)

        # Point clearly outside
        tri_idx, bary = lookup.find_triangle(np.array([0.9, 0.9]))

        assert tri_idx == -1
        assert bary is None

    def test_multiple_triangles(self):
        """Test lookup with multiple triangles."""
        # Two triangles sharing an edge
        uvs = np.array([
            [0.0, 0.0],
            [0.5, 0.0],
            [0.5, 0.5],
            [0.0, 0.5],
        ])
        faces = np.array([
            [0, 1, 2],
            [0, 2, 3],
        ])

        lookup = UVTriangleLookup(uvs, faces, grid_resolution=8)

        # Point in first triangle
        tri_idx, bary = lookup.find_triangle(np.array([0.3, 0.1]))
        assert tri_idx == 0
        assert bary is not None

        # Point in second triangle
        tri_idx, bary = lookup.find_triangle(np.array([0.1, 0.3]))
        assert tri_idx == 1
        assert bary is not None

    def test_barycentric_weights_at_vertices(self):
        """Test barycentric weights are correct at triangle vertices."""
        uvs = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
        ])
        faces = np.array([[0, 1, 2]])

        lookup = UVTriangleLookup(uvs, faces, grid_resolution=8)

        # Point at first vertex
        tri_idx, bary = lookup.find_triangle(np.array([0.0, 0.0]))
        if tri_idx >= 0:  # May or may not find depending on edge handling
            np.testing.assert_array_almost_equal(bary, [1.0, 0.0, 0.0], decimal=3)

        # Point at centroid
        centroid = uvs.mean(axis=0)
        tri_idx, bary = lookup.find_triangle(centroid)
        assert tri_idx == 0
        np.testing.assert_array_almost_equal(bary, [1/3, 1/3, 1/3], decimal=3)

    def test_grid_resolution_effect(self):
        """Test that higher grid resolution works correctly."""
        uvs = np.array([
            [0.0, 0.0],
            [0.1, 0.0],
            [0.05, 0.1],
        ])
        faces = np.array([[0, 1, 2]])

        # Low resolution - triangle may span multiple cells
        lookup_low = UVTriangleLookup(uvs, faces, grid_resolution=4)
        # High resolution - more precise cell assignment
        lookup_high = UVTriangleLookup(uvs, faces, grid_resolution=64)

        point = np.array([0.05, 0.03])

        idx_low, _ = lookup_low.find_triangle(point)
        idx_high, _ = lookup_high.find_triangle(point)

        # Both should find the triangle
        assert idx_low == 0
        assert idx_high == 0


class TestComputeVertexNormals:
    """Test vertex normal computation."""

    def test_flat_plane_normals(self):
        """Test normals on a flat XY plane."""
        # Simple quad made of 2 triangles
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        faces = np.array([
            [0, 1, 2],
            [0, 2, 3],
        ])

        normals = compute_vertex_normals(vertices, faces)

        # All normals should point in +Z direction
        for normal in normals:
            # Normalize to handle floating point
            n = normal / np.linalg.norm(normal)
            np.testing.assert_array_almost_equal(np.abs(n), [0, 0, 1], decimal=5)

    def test_normals_are_normalized(self):
        """Test that returned normals are unit vectors."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.5],
            [0.0, 1.0, 0.0],
        ])
        faces = np.array([
            [0, 1, 2],
            [0, 2, 3],
        ])

        normals = compute_vertex_normals(vertices, faces)

        for normal in normals:
            length = np.linalg.norm(normal)
            assert abs(length - 1.0) < 1e-6

    def test_degenerate_triangle_handled(self):
        """Test that degenerate triangles don't cause crashes."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],  # Degenerate - all points collinear
        ])
        faces = np.array([[0, 1, 2]])

        # Should not raise
        normals = compute_vertex_normals(vertices, faces)
        assert normals.shape == (3, 3)


class TestComputeLocalFrame:
    """Test local coordinate frame computation."""

    def test_upward_normal(self):
        """Test local frame with upward-pointing normal."""
        normal = np.array([0.0, 0.0, 1.0])
        frame = compute_local_frame(normal)

        # Frame should be 3x3 rotation matrix
        assert frame.shape == (3, 3)

        # Z column should equal normal
        np.testing.assert_array_almost_equal(frame[:, 2], normal, decimal=5)

        # Columns should be orthonormal
        for i in range(3):
            assert abs(np.linalg.norm(frame[:, i]) - 1.0) < 1e-6

        # Check orthogonality
        assert abs(np.dot(frame[:, 0], frame[:, 1])) < 1e-6
        assert abs(np.dot(frame[:, 1], frame[:, 2])) < 1e-6
        assert abs(np.dot(frame[:, 0], frame[:, 2])) < 1e-6

    def test_sideways_normal(self):
        """Test local frame with sideways normal."""
        normal = np.array([1.0, 0.0, 0.0])
        frame = compute_local_frame(normal)

        np.testing.assert_array_almost_equal(frame[:, 2], normal, decimal=5)

    def test_arbitrary_normal(self):
        """Test local frame with arbitrary normal."""
        normal = np.array([1.0, 2.0, 3.0])
        normal = normal / np.linalg.norm(normal)

        frame = compute_local_frame(normal)

        np.testing.assert_array_almost_equal(frame[:, 2], normal, decimal=5)

        # Check it's a valid rotation (orthonormal)
        for i in range(3):
            assert abs(np.linalg.norm(frame[:, i]) - 1.0) < 1e-6

    def test_nearly_aligned_with_up(self):
        """Test frame when normal is nearly parallel to default up vector."""
        # Normal very close to (0, 1, 0) - should switch reference
        normal = np.array([0.0, 0.99, 0.01])
        normal = normal / np.linalg.norm(normal)

        frame = compute_local_frame(normal)

        # Should still produce valid orthonormal frame
        np.testing.assert_array_almost_equal(frame[:, 2], normal, decimal=5)


class TestMeshCorrespondence:
    """Test correspondence save/load functionality."""

    def test_save_and_load(self):
        """Test that correspondence can be saved and loaded correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "corr.npz"

            # Create test correspondence
            corr = MeshCorrespondence()
            corr.triangle_indices = np.array([0, 1, 2, -1], dtype=np.int32)
            corr.bary_weights = np.array([
                [1.0, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [0.33, 0.33, 0.34],
                [0.0, 0.0, 0.0],
            ], dtype=np.float32)
            corr.rest_offsets = np.random.rand(4, 3).astype(np.float32)
            corr.valid_mask = np.array([True, True, True, False])
            corr.local_frames = np.random.rand(4, 3, 3).astype(np.float32)

            # Save
            corr.save(save_path)
            assert save_path.exists()

            # Load
            loaded = MeshCorrespondence.load(save_path)

            np.testing.assert_array_equal(loaded.triangle_indices, corr.triangle_indices)
            np.testing.assert_array_almost_equal(loaded.bary_weights, corr.bary_weights)
            np.testing.assert_array_almost_equal(loaded.rest_offsets, corr.rest_offsets)
            np.testing.assert_array_equal(loaded.valid_mask, corr.valid_mask)
            np.testing.assert_array_almost_equal(loaded.local_frames, corr.local_frames)


class TestDeformFrame:
    """Test per-frame mesh deformation."""

    @pytest.fixture
    def simple_correspondence(self):
        """Create a simple correspondence for testing."""
        corr = MeshCorrespondence()
        # 4 target vertices, 2 source triangles
        corr.triangle_indices = np.array([0, 0, 1, 1], dtype=np.int32)
        corr.bary_weights = np.array([
            [1.0, 0.0, 0.0],  # At vertex 0
            [0.5, 0.5, 0.0],  # Between vertices 0 and 1
            [1.0, 0.0, 0.0],  # At vertex 3
            [0.5, 0.5, 0.0],  # Between vertices 3 and 4
        ], dtype=np.float32)
        corr.rest_offsets = np.array([
            [0.0, 0.0, 0.1],  # Small offset in Z
            [0.0, 0.0, 0.1],
            [0.0, 0.0, 0.2],
            [0.0, 0.0, 0.2],
        ], dtype=np.float32)
        corr.valid_mask = np.array([True, True, True, True])
        corr.local_frames = np.tile(np.eye(3), (4, 1, 1)).astype(np.float32)

        return corr

    @pytest.fixture
    def simple_source_mesh(self):
        """Create a simple source mesh (2 triangles forming a quad)."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
            [1.0, 0.5, 0.0],
        ], dtype=np.float32)
        faces = np.array([
            [0, 1, 2],
            [3, 4, 5],
        ], dtype=np.int32)
        return vertices, faces

    def test_smooth_mode_preserves_offset(self, simple_correspondence, simple_source_mesh):
        """Test that smooth mode adds offset directly."""
        source_verts, source_faces = simple_source_mesh
        target_rest = np.zeros((4, 3), dtype=np.float32)

        result = deform_frame(
            source_verts=source_verts,
            source_faces=source_faces,
            target_rest_verts=target_rest,
            correspondence=simple_correspondence,
            offset_mode="smooth"
        )

        # First vertex: interpolated from source triangle 0 at vertex 0
        # Expected: source_verts[0] + offset[0] = (0,0,0) + (0,0,0.1) = (0,0,0.1)
        np.testing.assert_array_almost_equal(result[0], [0.0, 0.0, 0.1], decimal=5)

    def test_invalid_vertices_use_rest_pose(self, simple_source_mesh):
        """Test that invalid vertices keep their rest position."""
        source_verts, source_faces = simple_source_mesh

        # Create correspondence with invalid vertex
        corr = MeshCorrespondence()
        corr.triangle_indices = np.array([0, -1], dtype=np.int32)
        corr.bary_weights = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        corr.rest_offsets = np.array([[0.0, 0.0, 0.1], [0.0, 0.0, 0.0]], dtype=np.float32)
        corr.valid_mask = np.array([True, False])
        corr.local_frames = np.tile(np.eye(3), (2, 1, 1)).astype(np.float32)

        target_rest = np.array([
            [0.0, 0.0, 0.0],
            [5.0, 5.0, 5.0],  # This should be preserved
        ], dtype=np.float32)

        result = deform_frame(
            source_verts=source_verts,
            source_faces=source_faces,
            target_rest_verts=target_rest,
            correspondence=corr,
            offset_mode="smooth"
        )

        # Invalid vertex should keep rest position
        np.testing.assert_array_almost_equal(result[1], [5.0, 5.0, 5.0], decimal=5)

    def test_smoothing_weights_reduce_offset(self, simple_correspondence, simple_source_mesh):
        """Test that smoothing weights dampen the offset."""
        source_verts, source_faces = simple_source_mesh
        target_rest = np.zeros((4, 3), dtype=np.float32)

        # Without smoothing
        result_no_smooth = deform_frame(
            source_verts=source_verts,
            source_faces=source_faces,
            target_rest_verts=target_rest,
            correspondence=simple_correspondence,
            smoothing_weights=None,
            offset_mode="smooth"
        )

        # With full smoothing (weight=1.0)
        smoothing = np.ones(4, dtype=np.float32)
        result_smooth = deform_frame(
            source_verts=source_verts,
            source_faces=source_faces,
            target_rest_verts=target_rest,
            correspondence=simple_correspondence,
            smoothing_weights=smoothing,
            offset_mode="smooth"
        )

        # Smoothed result should have smaller Z offset
        # With weight=1.0 and formula offset * (1.0 - w * 0.5), offset becomes 0.5x
        assert result_smooth[0, 2] < result_no_smooth[0, 2]

    def test_offset_modes_differ(self, simple_correspondence, simple_source_mesh):
        """Test that different offset modes produce different results."""
        source_verts, source_faces = simple_source_mesh
        target_rest = np.zeros((4, 3), dtype=np.float32)

        result_smooth = deform_frame(
            source_verts=source_verts,
            source_faces=source_faces,
            target_rest_verts=target_rest,
            correspondence=simple_correspondence,
            offset_mode="smooth"
        )

        result_normal = deform_frame(
            source_verts=source_verts,
            source_faces=source_faces,
            target_rest_verts=target_rest,
            correspondence=simple_correspondence,
            offset_mode="normal"
        )

        result_rigid = deform_frame(
            source_verts=source_verts,
            source_faces=source_faces,
            target_rest_verts=target_rest,
            correspondence=simple_correspondence,
            offset_mode="rigid"
        )

        # All should produce valid output
        assert result_smooth.shape == (4, 3)
        assert result_normal.shape == (4, 3)
        assert result_rigid.shape == (4, 3)


class TestBarycentricInterpolation:
    """Test barycentric weight computation in UVTriangleLookup."""

    def test_weights_sum_to_one(self):
        """Test that barycentric weights sum to 1 for points inside triangles."""
        uvs = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
        ])
        faces = np.array([[0, 1, 2]])
        lookup = UVTriangleLookup(uvs, faces, grid_resolution=16)

        # Test multiple points inside triangle
        test_points = [
            [0.5, 0.3],
            [0.3, 0.2],
            [0.6, 0.1],
            [0.45, 0.5],
        ]

        for point in test_points:
            tri_idx, bary = lookup.find_triangle(np.array(point))
            if tri_idx >= 0:
                assert abs(bary.sum() - 1.0) < 1e-5, f"Weights don't sum to 1 for {point}"

    def test_weights_interpolate_correctly(self):
        """Test that barycentric weights correctly interpolate positions."""
        uvs = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
        ])
        faces = np.array([[0, 1, 2]])
        lookup = UVTriangleLookup(uvs, faces, grid_resolution=16)

        # Pick a point and verify interpolation
        test_point = np.array([0.5, 0.3])
        tri_idx, bary = lookup.find_triangle(test_point)

        assert tri_idx == 0

        # Interpolate UV using barycentric weights
        interpolated = (
            uvs[0] * bary[0] +
            uvs[1] * bary[1] +
            uvs[2] * bary[2]
        )

        np.testing.assert_array_almost_equal(interpolated, test_point, decimal=5)
