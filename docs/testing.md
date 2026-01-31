# Motion Capture Testing Guide

This guide explains how to test each component of the motion capture pipeline in isolation for development and debugging.

## Overview

The mocap pipeline uses GVHMR (preferred) or WHAM (fallback) for motion tracking:

1. **Motion tracking** (GVHMR/WHAM) - Skeleton animation in world space
2. **Mesh generation** - SMPL-X mesh sequence from motion data
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
# Motion capture methods:
#   ✓ gvhmr (preferred)
#   ✓ wham (fallback)
```

### Test Data Setup

For testing, you need a project with:
```
projects/Test_Shot/
├── source/frames/          # PNG sequence (frame_0001.png, frame_0002.png, ...)
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

## Stage 1: Motion Tracking (GVHMR/WHAM)

**Purpose:** Extract skeleton animation in world coordinates

### Test Command

```bash
# Auto-select best method (GVHMR preferred)
python scripts/run_mocap.py projects/Test_Shot

# Force specific method
python scripts/run_mocap.py projects/Test_Shot --method gvhmr
python scripts/run_mocap.py projects/Test_Shot --method wham
```

### Expected Behavior

1. Loads frames from `source/frames/`
2. Optionally uses masks from `roto/` to focus on person
3. Runs GVHMR or WHAM inference
4. Saves results to `mocap/gvhmr/output.pkl` or `mocap/wham/motion.pkl`

### Success Criteria

```bash
# Check output file exists (GVHMR)
ls projects/Test_Shot/mocap/gvhmr/output.pkl

# Or for WHAM
ls projects/Test_Shot/mocap/wham/motion.pkl

# Validate motion data
python -c "
import pickle

# Try GVHMR output first, fall back to WHAM
try:
    with open('projects/Test_Shot/mocap/gvhmr/output.pkl', 'rb') as f:
        data = pickle.load(f)
    print('Using GVHMR output')
except FileNotFoundError:
    with open('projects/Test_Shot/mocap/wham/motion.pkl', 'rb') as f:
        data = pickle.load(f)
    print('Using WHAM output')

# Check structure
print(f'Keys: {data.keys()}')

# Check for pose data
if 'body_pose' in data:
    poses = data['body_pose']
    print(f'✓ Body pose shape: {poses.shape}')
if 'global_orient' in data:
    orient = data['global_orient']
    print(f'✓ Global orient shape: {orient.shape}')
if 'transl' in data:
    trans = data['transl']
    print(f'✓ Translation range: {trans.min(axis=0)} to {trans.max(axis=0)}')
"
```

### Troubleshooting

**"GVHMR not available":**
- Run `python scripts/run_mocap.py --check`
- Install GVHMR: `git clone https://github.com/zju3dv/GVHMR.git && cd GVHMR && pip install -e .`
- Download checkpoints from GVHMR project page

**"WHAM not available":**
- Install WHAM: `git clone https://github.com/yohanshin/WHAM.git && cd WHAM && pip install -e .`
- Download checkpoints from WHAM project page

**"No frames found":**
- Check `projects/Test_Shot/source/frames/` contains frame_*.png files
- Run ingest stage first: `python scripts/run_pipeline.py footage.mp4 --stages ingest`

**Motion looks wrong:**
- Verify person is visible and upright in frames
- Check segmentation masks if using roto/ (should tightly bound person)
- Try the other method (`--method wham` if GVHMR fails, or vice versa)

## Stage 2: Mesh Generation

**Purpose:** Generate SMPL-X mesh sequence from motion data

### Test Command

```bash
python scripts/smplx_from_motion.py projects/Test_Shot \
    --motion mocap/gvhmr/output.pkl \
    --output mocap/smplx_animated/
```

### Success Criteria

```bash
# Check output meshes
ls projects/Test_Shot/mocap/smplx_animated/

# Expected (numbering matches source/frames/ sequence):
# frame_0001.obj, frame_0002.obj, ... (if source starts at 0001)
# frame_1001.obj, frame_1002.obj, ... (if source starts at 1001)

# Validate mesh quality
python -c "
import trimesh
import glob

meshes = sorted(glob.glob('projects/Test_Shot/mocap/smplx_animated/frame_*.obj'))
print(f'✓ Found {len(meshes)} frame meshes')

# Check first mesh
mesh = trimesh.load(meshes[0])
print(f'✓ Vertices: {len(mesh.vertices)}')
print(f'✓ Faces: {len(mesh.faces)}')
print(f'✓ Bounds: {mesh.bounds}')
"
```

### Validation Checklist

- [ ] Mesh count matches frame count
- [ ] Each mesh loads without errors
- [ ] Vertex count is consistent across frames
- [ ] Bounding box size is realistic (~1-2 meters for human)

## Stage 3: Texture Projection

**Purpose:** Project camera views to canonical UV texture

### Test Command

```bash
# Test on single frame first
python scripts/texture_projection.py projects/Test_Shot \
  --mesh-sequence mocap/smplx_animated/ \
  --output mocap/texture_test.png \
  --test-frame 10

# Full multi-view aggregation
python scripts/texture_projection.py projects/Test_Shot \
  --mesh-sequence mocap/smplx_animated/ \
  --output mocap/texture.png \
  --resolution 1024
```

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

## Integration Testing

Test full pipeline end-to-end:

```bash
# Full mocap pipeline
python scripts/run_mocap.py projects/Test_Shot

# Expected output:
# mocap/
# ├── gvhmr/output.pkl (or wham/motion.pkl)
# └── mesh_sequence/frame_*.obj
```

### Validation

```bash
# Check all outputs exist
test -d projects/Test_Shot/mocap/gvhmr && echo "✓ GVHMR output" || \
test -d projects/Test_Shot/mocap/wham && echo "✓ WHAM output"
test -d projects/Test_Shot/mocap/mesh_sequence && echo "✓ Mesh Sequence"

# Import test (requires Maya/Blender/Houdini)
# Load mesh_sequence/frame_*.obj
# Verify mesh animates correctly
```

## Performance Benchmarks

Typical processing times on GPU (RTX 3090):

| Stage | Time (100 frames) | Notes |
|-------|------------------|-------|
| GVHMR | ~2-3 minutes | GPU inference |
| WHAM | ~2-3 minutes | GPU inference |
| Mesh generation | ~1-2 minutes | Per-frame SMPL-X |
| Texture | ~5-10 minutes | Depends on resolution |

**Total:** ~10-15 minutes for 100-frame sequence

## Common Issues

### GPU Out of Memory

```bash
# Process shorter sequences or use lower-res frames
# GVHMR and WHAM have different memory profiles
# Try --method wham if GVHMR runs out of memory
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
try:
    with open('projects/Test_Shot/mocap/gvhmr/output.pkl', 'rb') as f:
        motion = pickle.load(f)
except FileNotFoundError:
    with open('projects/Test_Shot/mocap/wham/motion.pkl', 'rb') as f:
        motion = pickle.load(f)

if 'transl' in motion:
    print('Person position:', motion['transl'][0])

# Should be in similar coordinate system (both in meters, similar origin)
"
```

### Stage Dependencies

Remember dependency order:
1. Ingest → frames
2. COLMAP → camera data
3. Motion (GVHMR/WHAM) → requires frames
4. Mesh generation → requires motion data
5. Texture → requires mesh sequence + camera + frames

## Debugging Tips

1. **Start small:** Test on 10-20 frame sequence first
2. **Check method availability:** Run `python scripts/run_mocap.py --check`
3. **Try both methods:** If GVHMR fails, try WHAM and vice versa
4. **Check data formats:** Print shapes/types of all loaded data
5. **Test stages independently:** Don't run full pipeline until each stage works

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

**test_gvhmr.py** (new):
- `TestGVHMRConversion`: GVHMR output format conversion
- `TestBodyPoseHandling`: Empty/1D body_pose array handling

### Expected Output

```
============================= test session starts ==============================
platform linux -- Python 3.11.x, pytest-9.x.x
collected 33+ items

tests/test_mesh_deform.py::TestUVTriangleLookup::test_simple_triangle_lookup PASSED
tests/test_mesh_deform.py::TestUVTriangleLookup::test_point_outside_triangle PASSED
...
tests/test_gvhmr.py::TestGVHMRConversion::test_output_format PASSED

============================== 33+ passed in 0.37s ==============================
```

## Reporting Issues

When reporting bugs, include:
- Output of `python scripts/run_mocap.py --check`
- Error messages (full stack trace)
- Test data characteristics (frame count, resolution, person visibility)
- GPU info (`nvidia-smi`)
- Python version (`python --version`)
- Which method was used (GVHMR or WHAM)

## Next Steps

After successful testing:
1. Test on real production footage
2. Compare GVHMR vs WHAM results for your use case
3. Implement Alembic export (currently exports OBJ sequence)
4. Add seam blending to texture projection
5. Support multi-person workflows
