# GVHMR Transition Roadmap

Transition plan from WHAM to [GVHMR](https://github.com/zju3dv/GVHMR) for world-grounded SMPL/SMPL-X human motion recovery.

## Overview

GVHMR (Gravity-View Human Motion Recovery) is a SIGGRAPH Asia 2024 paper that provides improved world-grounded human motion recovery compared to WHAM. Key advantages:

- **Gravity-View Coordinates**: More accurate world-grounded motion
- **Camera Motion Support**: Works with both static and moving cameras
- **Focal Length Input**: Can leverage COLMAP intrinsics for better accuracy
- **Active Development**: Recent updates (March 2025) include SimpleVO and `f_mm` parameter
- **Multi-Person Support**: Handles multiple people in scene (outputs per-person results)

### Accuracy Improvements vs WHAM

According to the [SIGGRAPH Asia 2024 paper](https://arxiv.org/abs/2409.06662):

- **7-10mm lower error** on standard benchmarks (PA-MPJPE, MPJPE, PVE)
- **Better handling of camera rotation errors** - GVHMR is more robust to visual odometry drift
- **Lower accumulated drift on long sequences** - WHAM's autoregressive RNN accumulates errors over time, while GVHMR's gravity-view coordinates maintain consistency
- **Up to 6mm improvement** in PA-MPJPE for camera-space metrics

## Current Pipeline State

### What We Have (Working)

1. **COLMAP SfM** (`run_colmap.py:387-451`)
   - Extracts camera intrinsics: `fx`, `fy`, `cx`, `cy`
   - Exports to `camera/intrinsics.json`
   - Supports multiple camera models: PINHOLE, SIMPLE_RADIAL, OPENCV, etc.

2. **WHAM Integration** (`run_mocap.py:134-208`)
   - Runs via `python -m wham.run`
   - Outputs `mocap/wham/motion.pkl` with:
     - `poses`: [N, 72] SMPL body pose parameters
     - `trans`: [N, 3] root translation
     - `betas`: [10] shape parameters

3. **SMPL-X Mesh Generation** (`smplx_from_motion.py`)
   - Converts motion.pkl to animated OBJ sequence
   - Expects: `poses`, `trans`, `betas` keys

### GVHMR Key Differences

| Aspect | WHAM | GVHMR |
|--------|------|-------|
| Input | Image frames | Video file |
| Camera | Auto-estimated | Can accept `f_mm` focal length |
| Static camera | N/A | `--static_cam` flag |
| Output format | `poses`, `trans`, `betas` | `smpl_params_global`, `smpl_params_incam` |
| Visual odometry | Built-in SLAM | SimpleVO (default) or DPVO |
| Multi-person | Single person | Multiple people (outputs to `person_X/` subdirs) |

---

## Phase 1: Installation Infrastructure

### 1.1 Update `wizard.py` Component Definition

Replace WHAM component with GVHMR:

```python
# scripts/install_wizard/wizard.py

# Remove WHAM component (lines 106-118)
# Add GVHMR component:

self.components['gvhmr'] = {
    'name': 'GVHMR',
    'required': False,
    'installers': [
        GitRepoInstaller(
            'GVHMR',
            'https://github.com/zju3dv/GVHMR.git',
            self.install_dir / "GVHMR",
            size_gb=4.0  # Code + checkpoints
        )
    ]
}
```

### 1.2 Checkpoint Downloads

GVHMR requires these checkpoints in `inputs/checkpoints/`:

| Checkpoint | Path | Source |
|------------|------|--------|
| GVHMR model | `gvhmr/gvhmr_siga24_release.ckpt` | Google Drive |
| HMR2 | `hmr2/epoch=10-step=25000.ckpt` | Google Drive |
| ViTPose | `vitpose/vitpose-h-multi-coco.pth` | Google Drive |
| YOLO | `yolo/yolov8x.pt` | Ultralytics |
| SMPL | `body_models/smpl/` | smpl.is.tue.mpg.de |
| SMPLX | `body_models/smplx/` | smpl-x.is.tue.mpg.de |

Update `downloader.py` to add GVHMR checkpoint definitions.

### 1.3 Conda Environment

GVHMR requires Python 3.10:

```bash
conda create -y -n gvhmr python=3.10
conda activate gvhmr
pip install -r requirements.txt
pip install -e .
```

**Recommendation**: Integrate into `vfx-pipeline` conda env rather than maintaining a separate environment.

Rationale:
- Python 3.10 is compatible with our other dependencies (PyTorch, ComfyUI, etc.)
- Simpler user experience - one environment to activate
- Easier CI/testing - single environment to manage
- Avoids environment switching complexity in `run_pipeline.py`

If dependency conflicts arise, fall back to separate `gvhmr` env with subprocess activation.

---

## Phase 2: Core Integration (`run_mocap.py`)

### 2.1 New GVHMR Runner Function

```python
def run_gvhmr_motion_tracking(
    project_dir: Path,
    focal_mm: Optional[float] = None,
    static_camera: bool = False,
    output_dir: Optional[Path] = None
) -> bool:
    """Run GVHMR for world-grounded motion tracking.

    Args:
        project_dir: Project directory
        focal_mm: Focal length in mm (from COLMAP intrinsics)
        static_camera: Skip visual odometry for static cameras
        output_dir: Output directory for results

    Returns:
        True if successful
    """
    gvhmr_dir = INSTALL_DIR / "GVHMR"
    output_dir = output_dir or project_dir / "mocap" / "gvhmr"
    output_dir.mkdir(parents=True, exist_ok=True)

    # GVHMR expects video file, not frames
    # Need to either:
    # 1. Use original source video if available
    # 2. Re-encode frames to video
    video_path = find_or_create_video(project_dir)

    cmd = [
        "python", str(gvhmr_dir / "tools/demo/demo.py"),
        "--video", str(video_path),
        "--output_root", str(output_dir),
    ]

    if static_camera:
        cmd.append("--static_cam")

    if focal_mm:
        cmd.extend(["--f_mm", str(focal_mm)])

    # Run GVHMR
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0
```

### 2.2 Convert COLMAP Intrinsics to Focal Length (mm)

```python
def colmap_intrinsics_to_focal_mm(
    intrinsics_path: Path,
    sensor_width_mm: Optional[float] = None,
    metadata_path: Optional[Path] = None
) -> float:
    """Convert COLMAP intrinsics to focal length in mm.

    Args:
        intrinsics_path: Path to camera/intrinsics.json
        sensor_width_mm: Sensor width in mm (None = auto-detect or assume 36mm)
        metadata_path: Path to source video/image for EXIF extraction

    Returns:
        Focal length in millimeters
    """
    import json

    with open(intrinsics_path) as f:
        intrinsics = json.load(f)

    fx = intrinsics["fx"]  # Focal length in pixels
    width = intrinsics["width"]  # Image width in pixels

    # Try to get sensor width from metadata if not provided
    if sensor_width_mm is None:
        sensor_width_mm = extract_sensor_width_from_exif(metadata_path)
        if sensor_width_mm is None:
            # Default to 36mm (full-frame equivalent)
            # Common assumptions: 36mm (full-frame), 23.5mm (APS-C), 13.2mm (1-inch)
            sensor_width_mm = 36.0
            print(f"    Warning: Assuming {sensor_width_mm}mm sensor width (full-frame)")

    # focal_mm = fx * sensor_width_mm / image_width
    focal_mm = fx * sensor_width_mm / width

    return focal_mm


def extract_sensor_width_from_exif(metadata_path: Optional[Path]) -> Optional[float]:
    """Try to extract sensor width from EXIF data.

    Returns:
        Sensor width in mm, or None if not available
    """
    if metadata_path is None or not metadata_path.exists():
        return None

    try:
        # Could use exifread, PIL, or ffprobe for video
        # Implementation depends on input format
        pass
    except Exception:
        return None

    return None
```

**Note**: Sensor width significantly affects focal length calculation. Consider adding a `--sensor-width` CLI option for users who know their camera specs.

### 2.3 Video Handling

GVHMR requires video input. Options:

**Option A**: Use original source video (preferred)
```python
source_video = project_dir / "source" / "input.mp4"
if source_video.exists():
    video_path = source_video
```

**Option B**: Re-encode frames to video (fallback)

Use high-quality encoding to minimize quality loss:

```python
frames_dir = project_dir / "source" / "frames"
video_path = project_dir / "source" / "_gvhmr_input.mp4"

# Get fps from project metadata
fps = get_project_fps(project_dir) or 24

subprocess.run([
    "ffmpeg", "-y",
    "-framerate", str(fps),
    "-i", str(frames_dir / "frame_%04d.png"),
    "-c:v", "libx264",
    "-crf", "18",           # High quality (18-23 is visually lossless range)
    "-preset", "slow",      # Better compression at cost of encode time
    "-pix_fmt", "yuv420p",  # Compatibility
    str(video_path)
])
```

**Quality considerations**:
- CRF 18 provides near-lossless quality (CRF 0 = lossless, 23 = default, 51 = worst)
- For truly lossless, use `-c:v ffv1` but file size will be much larger
- Original source video is always preferred to avoid any generation loss

---

## Phase 3: Output Format Adapter

### 3.1 GVHMR Output Structure

GVHMR outputs two coordinate systems:

```python
{
    'smpl_params_global': {      # World-grounded coordinates
        'body_pose': [N, 63],    # 21 body joints × 3 (axis-angle)
        'global_orient': [N, 3], # Root orientation
        'transl': [N, 3],        # Root translation
        'betas': [N, 10]         # Shape parameters
    },
    'smpl_params_incam': {       # Camera-relative coordinates
        'body_pose': [N, 63],
        'global_orient': [N, 3],
        'transl': [N, 3],
        'betas': [N, 10]
    }
}
```

### 3.2 Adapter Function

Convert GVHMR output to WHAM-compatible format for `smplx_from_motion.py`:

```python
def convert_gvhmr_to_wham_format(gvhmr_output_path: Path, output_path: Path):
    """Convert GVHMR output to WHAM-compatible motion.pkl format.

    This allows smplx_from_motion.py to work with GVHMR output.
    """
    import pickle
    import numpy as np

    with open(gvhmr_output_path, 'rb') as f:
        gvhmr_data = pickle.load(f)

    # Use global (world-grounded) coordinates
    params = gvhmr_data['smpl_params_global']

    # Convert to WHAM format:
    # WHAM poses: [N, 72] = [global_orient(3) + body_pose(63) + jaw(3) + eyes(3)]
    # GVHMR: [N, 63] body_pose + [N, 3] global_orient

    n_frames = len(params['body_pose'])

    # Concatenate global_orient + body_pose + zeros for face/hands
    poses = np.concatenate([
        params['global_orient'],              # [N, 3]
        params['body_pose'],                  # [N, 63]
        np.zeros((n_frames, 6))               # [N, 6] jaw + eyes placeholder
    ], axis=1)  # Result: [N, 72]

    wham_format = {
        'poses': poses,
        'trans': params['transl'],
        'betas': params['betas'][0] if params['betas'].ndim > 1 else params['betas']
    }

    with open(output_path, 'wb') as f:
        pickle.dump(wham_format, f)

    print(f"Converted GVHMR output to WHAM format: {output_path}")
```

---

## Phase 4: Pipeline Integration

### 4.1 Update `run_pipeline.py`

Modify the mocap stage to use GVHMR:

```python
# In run_mocap() function around line 351

def run_mocap(project_dir: Path, use_colmap_intrinsics: bool = True) -> bool:
    """Run GVHMR motion capture."""

    focal_mm = None
    static_camera = False

    # Get focal length from COLMAP if available
    intrinsics_path = project_dir / "camera" / "intrinsics.json"
    if use_colmap_intrinsics and intrinsics_path.exists():
        focal_mm = colmap_intrinsics_to_focal_mm(intrinsics_path)
        print(f"Using COLMAP focal length: {focal_mm:.1f}mm")

    # Detect static camera from COLMAP extrinsics
    extrinsics_path = project_dir / "camera" / "extrinsics.json"
    if extrinsics_path.exists():
        static_camera = detect_static_camera(extrinsics_path)
        if static_camera:
            print("Detected static camera, skipping visual odometry")

    # Run GVHMR
    success = run_gvhmr_motion_tracking(
        project_dir,
        focal_mm=focal_mm,
        static_camera=static_camera
    )

    if success:
        # Convert output to WHAM-compatible format
        gvhmr_output = project_dir / "mocap" / "gvhmr" / "output.pkl"
        wham_compat = project_dir / "mocap" / "wham" / "motion.pkl"
        wham_compat.parent.mkdir(parents=True, exist_ok=True)
        convert_gvhmr_to_wham_format(gvhmr_output, wham_compat)

    return success
```

### 4.2 Detect Static Camera

```python
def detect_static_camera(
    extrinsics_path: Path,
    threshold_meters: float = 0.01
) -> bool:
    """Detect if camera is static from COLMAP extrinsics.

    Analyzes camera translation variance across all frames. A "static" camera
    is one where the total movement is below the threshold (e.g., tripod shot
    with minor vibration).

    Args:
        extrinsics_path: Path to extrinsics.json
        threshold_meters: Max camera movement variance to consider "static"
                          Default 0.01 = ~1cm movement tolerance

    Returns:
        True if camera appears static (enables --static_cam in GVHMR)
    """
    import json
    import numpy as np

    with open(extrinsics_path) as f:
        extrinsics = json.load(f)

    # Extract translations from 4x4 matrices
    # Each matrix is a list of 4 rows, translation is column 3 (index 3) of rows 0-2
    translations = np.array([np.array(m)[:3, 3] for m in extrinsics])

    # Check variance across all axes
    variance = np.var(translations, axis=0).sum()

    is_static = variance < threshold_meters
    if is_static:
        max_movement = np.max(np.abs(translations - translations.mean(axis=0)))
        print(f"    Camera variance: {variance:.6f} (max movement: {max_movement:.4f}m)")

    return is_static
```

**Threshold tuning**:
- `0.01` (default): Strict - only truly locked-off shots
- `0.05`: Moderate - allows minor handheld wobble
- `0.1`: Loose - accepts slow dolly/pan movements

When in doubt, don't use `--static_cam` - GVHMR's SimpleVO handles moving cameras well.

---

## Phase 5: Testing & Validation

### 5.1 Test Matrix

| Test Case | Input | Expected Output |
|-----------|-------|-----------------|
| Static camera, known focal | COLMAP solve | GVHMR with `--f_mm`, `--static_cam` |
| Moving camera, known focal | COLMAP solve | GVHMR with `--f_mm` |
| Unknown camera | Raw video | GVHMR auto-estimates |
| Multi-person scene | Video with 2+ people | Separate `person_0/`, `person_1/` outputs |
| GVHMR failure | Corrupted/edge case | Fallback to WHAM |

### 5.2 Validation Checklist

- [ ] GVHMR installs via wizard
- [ ] Checkpoints download correctly
- [ ] COLMAP intrinsics → `f_mm` conversion accurate
- [ ] Static camera detection works
- [ ] Output converts to WHAM format
- [ ] `smplx_from_motion.py` generates meshes from converted output
- [ ] Mesh deformation pipeline still works
- [ ] Multi-person scenes produce separate outputs per person
- [ ] WHAM fallback triggers correctly on GVHMR failure

---

## Implementation Order

1. **Installation**
   - Update `wizard.py` with GVHMR component
   - Add checkpoint download definitions
   - Test clean installation

2. **Core Integration**
   - Implement `run_gvhmr_motion_tracking()`
   - Add COLMAP intrinsics conversion
   - Handle video input (find or create)

3. **Output Adapter**
   - Implement `convert_gvhmr_to_wham_format()`
   - Test with `smplx_from_motion.py`
   - Validate mesh output

4. **Pipeline Integration**
   - Update `run_pipeline.py` mocap stage
   - Add static camera detection
   - End-to-end testing

---

## Migration Strategy

### Backward Compatibility

Keep WHAM as fallback option with automatic failover:

```python
def run_mocap(
    project_dir: Path,
    method: str = "auto",
    use_colmap_intrinsics: bool = True
) -> bool:
    """Run motion capture with automatic fallback.

    Args:
        project_dir: Project directory
        method: "gvhmr", "wham", or "auto" (try GVHMR first, fallback to WHAM)
        use_colmap_intrinsics: Whether to use COLMAP focal length for GVHMR

    Returns:
        True if successful
    """
    if method == "auto":
        # Try GVHMR first
        print("  → Attempting GVHMR motion capture...")
        try:
            if run_gvhmr_motion_tracking(project_dir, use_colmap_intrinsics):
                return True
        except Exception as e:
            print(f"    GVHMR failed: {e}")

        # Fallback to WHAM
        print("  → Falling back to WHAM...")
        return run_wham_motion_tracking(project_dir)

    elif method == "gvhmr":
        return run_gvhmr_motion_tracking(project_dir, use_colmap_intrinsics)

    elif method == "wham":
        return run_wham_motion_tracking(project_dir)

    else:
        raise ValueError(f"Unknown mocap method: {method}")
```

**CLI option**: Add `--mocap-method` argument to `run_pipeline.py`:
```python
parser.add_argument(
    "--mocap-method",
    choices=["auto", "gvhmr", "wham"],
    default="auto",
    help="Motion capture method (default: auto = GVHMR with WHAM fallback)"
)
```

### Deprecation Path

1. **v1.0**: Add GVHMR, keep WHAM as fallback
2. **v1.1**: Make GVHMR default, WHAM optional
3. **v2.0**: Remove WHAM support

---

## References

- [GVHMR GitHub](https://github.com/zju3dv/GVHMR)
- [GVHMR Project Page](https://zju3dv.github.io/gvhmr/)
- [GVHMR Paper (arXiv)](https://arxiv.org/abs/2409.06662)
- [WHAM GitHub](https://github.com/yohanshin/WHAM)
- [SMPLX GitHub](https://github.com/vchoutas/smplx)
