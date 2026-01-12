# Motion Capture Testing Guide

This guide explains how to test each component of the motion capture pipeline in isolation for development and debugging.

## Overview

The mocap pipeline has 3 main stages that can be tested independently:

1. **Motion tracking** (WHAM) - Skeleton animation in world space
2. **Geometry reconstruction** (ECON) - Clothed body meshes with SMPL-X topology
3. **Texture projection** - Multi-view UV texturing

Each stage has specific input requirements, outputs, and validation steps.

## Prerequisites

### Check All Dependencies

```bash
# Check what's installed
python scripts/run_mocap.py --check

# Expected output:
# Core (required):
#   ✓ numpy
#   ✓ pytorch
#   ✓ smplx
#   ✓ trimesh
#   ✓ opencv
#   ✓ pillow
# Optional (for specific methods):
#   ✓ wham
#   ✓ econ
```

### Test Data Setup

For testing, you need a project with:
```
projects/Test_Shot/
├── source/frames/          # PNG sequence (frame_1001.png, frame_1002.png, ...)
├── roto/                   # Optional: person segmentation masks
└── camera/                 # Required for mocap
    ├── extrinsics.json     # Camera matrices [N, 4, 4]
    └── intrinsics.json     # Camera parameters (fx, fy, cx, cy)
```

**Create test project:**
```bash
# Extract frames from test footage
python scripts/run_pipeline.py test_footage.mp4 \
  --name Test_Shot \
  --stages ingest,colmap

# This creates:
# - source/frames/ (frame extraction)
# - camera/ (COLMAP camera data)
```

## Stage 1: Motion Tracking (WHAM)

**Purpose:** Extract skeleton animation in world coordinates

### Test Command

```bash
python scripts/run_mocap.py projects/Test_Shot --test-stage motion
```

### Expected Behavior

1. Loads frames from `source/frames/`
2. Optionally uses masks from `roto/` to focus on person
3. Runs WHAM inference
4. Saves results to `mocap/wham/motion.pkl`

### Success Criteria

```bash
# Check output file exists
ls projects/Test_Shot/mocap/wham/motion.pkl

# Validate motion data
python -c "
import pickle
with open('projects/Test_Shot/mocap/wham/motion.pkl', 'rb') as f:
    data = pickle.load(f)

# Check structure
assert 'poses' in data, 'Missing poses'
assert 'trans' in data, 'Missing translation'
assert 'betas' in data, 'Missing shape parameters'

# Check dimensions
poses = data['poses']  # Should be [N, 72] (SMPL-X pose parameters)
trans = data['trans']  # Should be [N, 3] (world-space translation)
betas = data['betas']  # Should be [10] or [N, 10] (shape)

print(f'✓ Frames: {len(poses)}')
print(f'✓ Pose params: {poses.shape}')
print(f'✓ Translation range: {trans.min(axis=0)} to {trans.max(axis=0)}')
print(f'✓ Root motion: {(trans[-1] - trans[0])} meters')
"
```

### Troubleshooting

**"WHAM not available":**
- Run `python scripts/run_mocap.py --check`
- Install WHAM: `git clone https://github.com/yohanshin/WHAM.git && cd WHAM && pip install -e .`
- Download checkpoints from WHAM project page

**"No frames found":**
- Check `projects/Test_Shot/source/frames/` contains frame_*.png files
- Run ingest stage first: `python scripts/run_pipeline.py footage.mp4 --stages ingest`

**Motion looks wrong:**
- Verify person is visible and upright in frames
- Check segmentation masks if using roto/ (should tightly bound person)
- Try adjusting WHAM parameters (not currently exposed - would need code edit)

## Stage 2: Geometry Reconstruction (ECON)

**Purpose:** Reconstruct clothed body geometry from keyframes

### Test Command

```bash
python scripts/run_mocap.py projects/Test_Shot \
  --test-stage econ \
  --keyframe-interval 25
```

### Expected Behavior

1. Selects keyframes every 25 frames (configurable)
2. Runs ECON on each keyframe
3. Saves meshes to `mocap/econ/mesh_NNNN.obj`

### Success Criteria

```bash
# Check output meshes
ls projects/Test_Shot/mocap/econ/

# Expected:
# mesh_1001.obj
# mesh_1026.obj
# mesh_1051.obj
# ... (every 25 frames)

# Validate mesh quality
python -c "
import trimesh
import glob

meshes = sorted(glob.glob('projects/Test_Shot/mocap/econ/mesh_*.obj'))
print(f'✓ Found {len(meshes)} keyframe meshes')

# Check first mesh
mesh = trimesh.load(meshes[0])
print(f'✓ Vertices: {len(mesh.vertices)}')
print(f'✓ Faces: {len(mesh.faces)}')
print(f'✓ Bounds: {mesh.bounds}')

# ECON meshes typically have ~50k-100k vertices
assert len(mesh.vertices) > 10000, 'Mesh too small'
assert mesh.is_watertight, 'Mesh has holes'
"
```

### Validation Checklist

- [ ] Mesh count matches expected (frame_count / keyframe_interval)
- [ ] Each mesh loads without errors
- [ ] Vertex count is reasonable (50k-100k typical for ECON)
- [ ] Meshes are watertight (no holes)
- [ ] Bounding box size is realistic (~1-2 meters for human)

### Troubleshooting

**"ECON not available":**
- Install: `git clone https://github.com/YuliangXiu/ECON.git && cd ECON && pip install -r requirements.txt`
- Download SMPL-X models: Register at https://smpl-x.is.tue.mpg.de/
- Download ECON checkpoints from project page

**"ECON failed for frame X":**
- Check if person is clearly visible in that frame
- Try different keyframe (skip problematic frames)
- Verify segmentation mask exists and is accurate

**Meshes look wrong:**
- Check input frames - person should be upright, well-lit
- Increase keyframe density (lower `--keyframe-interval`)
- Verify ECON checkpoint is correct version

## Stage 3: Texture Projection

**Purpose:** Project camera views to canonical UV texture

### Test Command

```bash
# Test on single frame first
python scripts/texture_projection.py projects/Test_Shot \
  --mesh-sequence mocap/econ/mesh_sequence/mesh_sequence.pkl \
  --output mocap/texture_test.png \
  --test-frame 10

# Full multi-view aggregation
python scripts/texture_projection.py projects/Test_Shot \
  --mesh-sequence mocap/econ/mesh_sequence/mesh_sequence.pkl \
  --output mocap/texture.png \
  --resolution 1024
```

### Expected Behavior

1. Loads mesh sequence from ECON
2. Loads camera data (extrinsics + intrinsics)
3. Loads source frames
4. Projects each frame to UV space
5. Aggregates with visibility/viewing angle weighting
6. Saves texture to PNG

### Success Criteria

```bash
# Check texture file
ls projects/Test_Shot/mocap/texture.png

# Validate texture
python -c "
from PIL import Image
import numpy as np

texture = np.array(Image.open('projects/Test_Shot/mocap/texture.png'))
print(f'✓ Texture resolution: {texture.shape}')

# Check coverage (non-black pixels)
mask = texture.sum(axis=2) > 0
coverage = mask.sum() / mask.size * 100
print(f'✓ Coverage: {coverage:.1f}%')

# Good coverage is >70%
if coverage > 70:
    print('✓ Good coverage')
elif coverage > 40:
    print('  Warning: Low coverage - check camera angles')
else:
    print('  Error: Very low coverage - likely projection issue')
"
```

### Validation Checklist

- [ ] Texture file created
- [ ] Resolution matches requested (default 1024x1024)
- [ ] Coverage >70% (not mostly black)
- [ ] No visible seams or discontinuities
- [ ] Colors match source footage

### Troubleshooting

**Low coverage (<40%):**
- Check camera data loaded correctly
- Verify mesh and cameras are in same coordinate system
- Try single-frame test first (`--test-frame 0`)

**Texture looks blurry:**
- Increase resolution: `--resolution 2048`
- Check source frame quality
- Verify UV coordinates are valid

**Seams visible:**
- This is expected with simple projection
- Production would need seam blending (not yet implemented)
- Can be fixed in texturing software (Substance Painter, etc.)

## Integration Testing

Test full pipeline end-to-end:

```bash
# Full mocap pipeline
python scripts/run_mocap.py projects/Test_Shot

# Expected output:
# mocap/
# ├── wham/motion.pkl
# ├── econ/mesh_*.obj
# ├── econ/mesh_sequence/mesh_sequence.pkl
# └── texture.png
```

### Validation

```bash
# Check all outputs exist
test -f projects/Test_Shot/mocap/wham/motion.pkl && echo "✓ Motion"
test -d projects/Test_Shot/mocap/econ && echo "✓ Geometry"
test -d projects/Test_Shot/mocap/econ/mesh_sequence && echo "✓ Mesh Sequence"
test -f projects/Test_Shot/mocap/texture.png && echo "✓ Texture"

# Import test (requires Maya/Blender/Houdini)
# Load econ/mesh_sequence/mesh_*.obj
# Apply texture.png with SMPL-X UV layout
# Verify mesh animates correctly
```

## Performance Benchmarks

Typical processing times on GPU (RTX 3090):

| Stage | Time (100 frames) | Notes |
|-------|------------------|-------|
| WHAM | ~2-3 minutes | GPU inference |
| ECON | ~2-3 minutes (4 keyframes) | ~30s per keyframe |
| Texture | ~5-10 minutes | Depends on resolution |

**Total:** ~10-20 minutes for 100-frame sequence

## Common Issues

### GPU Out of Memory

```bash
# Reduce batch size in WHAM/ECON (requires code edit)
# Or process shorter sequences
# Or use lower-res frames
```

### Mesh Not Aligned with Camera

```bash
# Check coordinate systems match
python -c "
import numpy as np
import json

# Load camera
with open('projects/Test_Shot/camera/extrinsics.json') as f:
    cameras = np.array(json.load(f))

print('Camera 0 position:', cameras[0][:3, 3])

# Load motion
import pickle
with open('projects/Test_Shot/mocap/wham/motion.pkl', 'rb') as f:
    motion = pickle.load(f)

print('Person position:', motion['trans'][0])

# Should be in similar coordinate system (both in meters, similar origin)
"
```

### Stage Dependencies

Remember dependency order:
1. Ingest → frames
2. COLMAP → camera data
3. Motion (WHAM) → requires frames
4. ECON → requires frames
5. Texture → requires ECON mesh sequence + camera + frames

## Debugging Tips

1. **Start small:** Test on 10-20 frame sequence first
2. **Visualize intermediate results:** View ECON meshes in Blender
3. **Check data formats:** Print shapes/types of all loaded data
4. **Test stages independently:** Don't run full pipeline until each stage works
5. **Compare to examples:** Check WHAM/ECON example outputs

## Unit Tests

The repository includes unit tests for core algorithms that can be run without GPU or external dependencies.

### Running Unit Tests

```bash
# Install test dependencies
pip install pytest numpy

# Run all unit tests
pytest tests/ -v

# Run specific test file
pytest tests/test_mesh_deform.py -v
pytest tests/test_smplx_from_motion.py -v

# Run specific test class
pytest tests/test_mesh_deform.py::TestUVTriangleLookup -v
```

### Test Coverage

**test_mesh_deform.py** (19 tests):
- `TestUVTriangleLookup`: UV-space triangle lookup and spatial hashing
- `TestComputeVertexNormals`: Vertex normal computation
- `TestComputeLocalFrame`: Local coordinate frame construction
- `TestMeshCorrespondence`: Correspondence save/load functionality
- `TestDeformFrame`: Per-frame mesh deformation
- `TestBarycentricInterpolation`: Barycentric weight correctness

**test_smplx_from_motion.py** (14 tests):
- `TestLoadMotionData`: Motion.pkl loading and validation
- `TestFindSmplxModels`: SMPL-X model directory discovery
- `TestPoseParameterExtraction`: SMPL-X pose parameter handling
- `TestMotionDataIntegrity`: Edge cases and data integrity

### Writing New Tests

When adding new functionality:

1. Create test file in `tests/` directory
2. Import the script module:
   ```python
   import sys
   sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
   from your_script import function_to_test
   ```
3. Use pytest fixtures for shared test data
4. Test edge cases (empty inputs, invalid data, etc.)

### Expected Output

```
============================= test session starts ==============================
platform linux -- Python 3.11.x, pytest-9.x.x
collected 33 items

tests/test_mesh_deform.py::TestUVTriangleLookup::test_simple_triangle_lookup PASSED
tests/test_mesh_deform.py::TestUVTriangleLookup::test_point_outside_triangle PASSED
...
tests/test_smplx_from_motion.py::TestMotionDataIntegrity::test_motion_data_shapes_match PASSED

============================== 33 passed in 0.37s ==============================
```

## Reporting Issues

When reporting bugs, include:
- Output of `python scripts/run_mocap.py --check`
- Error messages (full stack trace)
- Test data characteristics (frame count, resolution, person visibility)
- GPU info (`nvidia-smi`)
- Python version (`python --version`)

## Next Steps

After successful testing:
1. Test on real production footage
2. Tune keyframe intervals for quality vs. speed
3. Implement Alembic export (currently exports OBJ sequence)
4. Add seam blending to texture projection
5. Support multi-person workflows
