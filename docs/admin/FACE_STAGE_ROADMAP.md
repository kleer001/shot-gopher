# Face Stage Roadmap (SPECTRE 3D Facial Reconstruction)

Add a `face` pipeline stage that reconstructs per-frame 3D facial expressions
from video using [SPECTRE](https://github.com/filby89/spectre) and the FLAME
parametric face model. Complements the existing `mocap` (body) stage.

## Rationale

- **mocap** (GVHMR) captures full-body SMPL skeleton + world-space translation. No facial detail.
- **face** (SPECTRE) captures FLAME face mesh + expression/jaw animation. Speech-aware lip sync.
- Combined: full body rig with detailed facial animation, suitable for CG character retargeting.

## What SPECTRE Provides

SPECTRE (CVPR 2023 Workshop) reconstructs 3D facial expressions from monocular
video. Key innovation: a lip-reading perceptual loss that ensures mouth shapes
match perceived speech. Built on DECA/FLAME with a temporal convolutional
expression encoder (kernel_size=5 over consecutive frames).

### Per-Frame FLAME Output

| Parameter | Dimensions | Description |
|-----------|-----------|-------------|
| `shape` | 100 | Identity geometry PCA coefficients (shared across frames) |
| `exp` | 50 | Expression PCA coefficients (lip sync, emotions) |
| `pose` | 6 | Neck rotation (3, axis-angle) + jaw rotation (3, axis-angle) |
| `tex` | 50 | Texture/albedo PCA coefficients |
| `cam` | 3 | Weak-perspective camera [scale, tx, ty] |
| `light` | 9x3=27 | Spherical harmonics lighting |

Plus: 3D mesh vertices, 2D/3D landmarks (68 points), rendered images.

### SPECTRE Dependencies

- **Python 3.8** (cannot share `vfx-pipeline` 3.10 or `gvhmr` 3.10 envs)
- PyTorch 1.11.0 + PyTorch3D 0.6.2
- `chumpy`, `kornia 0.6.6`, `librosa`, `scikit-image`, `yacs`
- External submodules: `face_alignment`, `face_detection` (from SPECTRE repo)
- Pretrained models: FLAME 2020, DECA, SPECTRE weights, EMOCA ResNet50

### SPECTRE Invocation

```bash
python demo.py --input video.mp4 --audio --device cuda
```

Processes video in 50-frame overlapping chunks (4-frame overlap). Outputs
rendered shape videos and FLAME parameters per frame.

## Decisions

| Question | Answer |
|----------|--------|
| Stage key | `"face"` |
| Stage order | After `mocap`, before `gsir` |
| On-disk output dir | `face/` |
| Conda environment | `spectre` (Python 3.8, dedicated) |
| CLI flags | `--face-*` (e.g. `--face-no-export`, `--face-person`) |
| Alias | None needed (short enough) |
| Standardized output | `face/face_params.pkl` |
| Export formats | Alembic (.abc), USD (.usd), OBJ T-pose |

### Output Format: `face/face_params.pkl`

```python
{
    'shape': np.array(100,),           # identity (constant across frames)
    'exp': np.array(N, 50),            # expression per frame
    'pose': np.array(N, 6),            # neck(3) + jaw(3) rotation per frame
    'tex': np.array(50,),              # texture (constant across frames)
    'cam': np.array(N, 3),             # weak-perspective camera per frame
    'landmarks2d': np.array(N, 68, 2), # 2D landmarks per frame
    'landmarks3d': np.array(N, 68, 3), # 3D landmarks per frame
    'n_frames': int,
    'source': 'spectre',
}
```

## Files to Create

| File | Description |
|------|-------------|
| `scripts/run_face.py` | Standalone face reconstruction script (mirrors `run_mocap.py`) |
| `scripts/export_face.py` | FLAME mesh export to ABC/USD (mirrors `export_mocap.py`) |
| `tests/test_run_face.py` | Unit tests (mirrors `tests/test_gvhmr.py`) |

## Files to Modify

### Phase 1: Pipeline Constants & Config

| File | Changes |
|------|---------|
| `scripts/pipeline_constants.py` | Add `"face"` to `STAGES` dict, `STAGE_ORDER` (after mocap, before gsir), `STAGES_REQUIRING_FRAMES` |
| `scripts/pipeline_config.py` | Add fields to `PipelineConfig`: `face_no_export: bool`, `face_fps: Optional[float]`, `face_start_frame: Optional[int]`, `face_end_frame: Optional[int]`, `face_person: Optional[str]`. Update both `from_args()` branches. |

### Phase 2: Stage Runners

| File | Changes |
|------|---------|
| `scripts/stage_runners.py` | Add `run_face()` wrapper (subprocess to `run_face.py`). Add `run_stage_face()` handler. Register in `STAGE_HANDLERS["face"]`. Add to `__all__`. |

### Phase 3: CLI Arguments

| File | Changes |
|------|---------|
| `scripts/run_pipeline.py` | Add `--face-no-export`, `--face-fps`, `--face-start-frame`, `--face-end-frame`, `--face-person` arguments. Wire into `PipelineConfig` constructor. |

### Phase 4: Installation

| File | Changes |
|------|---------|
| `scripts/install_wizard/installers.py` | Add SPECTRE installer: clone with `--recurse-submodules` to `.vfx_pipeline/SPECTRE/`, create `spectre` conda env, run `quick_install.sh`, install external submodule packages |
| `scripts/install_wizard/config.py` | Add `"spectre"` path to config dict and activation script |
| `scripts/verify_models.py` | Add SPECTRE model verification (FLAME model, SPECTRE weights, DECA weights) |

### Phase 5: Documentation & Web

| File | Changes |
|------|---------|
| `docs/which-stages.md` | Add `face` to stage tables, quick reference, common recipes |
| `web/config/pipeline_config.json` | Add `"face"` stage entry with dependencies, output dir, timing estimate |

## Implementation Details

### `scripts/run_face.py` — Structure

Mirrors `run_mocap.py` pattern:

1. **Dependency checking** — verify numpy, torch, SPECTRE installation
2. **Video preparation** — reuse `find_or_create_video()` logic (extract to shared utility in `pipeline_utils.py` to avoid duplication with `run_mocap.py`)
3. **Run SPECTRE** — `conda run -n spectre python demo.py --input <video> --device cuda`
4. **Parse output** — load SPECTRE's codedict (FLAME parameters per frame)
5. **Save standardized format** — write `face/face_params.pkl`
6. **Optional person isolation** — if `--face-person person_00`, composite with roto matte first (same as mocap)
7. **CLI** — `python run_face.py <project_dir> [--no-export] [--fps N] [--start-frame N] [--end-frame N] [--face-person person_00]`

### `scripts/export_face.py` — Structure

Mirrors `export_mocap.py` pattern:

1. Load `face_params.pkl`
2. Load FLAME model from `.vfx_pipeline/SPECTRE/data/FLAME2020/`
3. Generate mesh vertices per frame from shape + expression + pose
4. Export `tpose.obj` (neutral FLAME face)
5. Export animated mesh to Alembic (.abc) and USD (.usd) via Blender headless
6. Export `landmarks.json` (per-frame 2D/3D landmarks for downstream tools)
7. Runs in `spectre` conda env (has FLAME/chumpy dependencies)

### `scripts/stage_runners.py` — Handler Pattern

```python
def run_face(project_dir, no_export=False, fps=None, ...) -> bool:
    """Run SPECTRE face reconstruction via subprocess."""
    script_path = Path(__file__).parent / "run_face.py"
    cmd = [sys.executable, str(script_path), str(project_dir), ...]
    run_command(cmd, "Running face reconstruction")

def run_stage_face(ctx, config) -> bool:
    """Face stage handler."""
    print("\n=== Stage: face ===")
    face_output = ctx.project_dir / "face" / "face_params.pkl"
    if ctx.skip_existing and face_output.exists():
        print("  → Skipping (face data exists)")
        return True
    # ... call run_face(), clear_gpu_memory()
```

### Shared Utility Extraction

`find_or_create_video()` in `run_mocap.py` (lines 445-651) should be extracted
to `pipeline_utils.py` so both mocap and face stages can reuse it without
duplication. This is the only refactor needed — the function is self-contained
with clear inputs/outputs.

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| SPECTRE requires Python 3.8 (old stack) | Dedicated `spectre` conda env, isolated from pipeline and gvhmr envs |
| PyTorch3D build is notoriously fragile | `quick_install.sh` handles it; document platform-specific workarounds |
| FLAME model requires MPI registration | Same registration gate as SMPL-X (already required for mocap); document in install wizard |
| GPU VRAM (~4GB for SPECTRE) | Already handled by `clear_gpu_memory()` between stages |
| Video-only input (not frames) | `find_or_create_video()` pattern already solved in mocap |
| Frontal faces only | Document limitation; SPECTRE expects roughly frontal talking heads |

## Build Order

1. **Install wizard** — get SPECTRE running standalone in `.vfx_pipeline/SPECTRE/`
2. **`run_face.py`** — prove end-to-end outside the pipeline
3. **Pipeline wiring** — `pipeline_constants.py`, `pipeline_config.py`, `stage_runners.py`
4. **CLI args** — `run_pipeline.py`
5. **`export_face.py`** — Alembic/USD export
6. **Tests** — `test_run_face.py`
7. **Docs & web** — `which-stages.md`, `pipeline_config.json`

## Verification Criteria

- [ ] `python run_face.py <project_dir>` produces `face/face_params.pkl` with correct keys and dimensions
- [ ] `python export_face.py <project_dir>` produces `face/export/face_motion.abc` and `face/export/tpose.obj`
- [ ] `python run_pipeline.py video.mp4 --stages ingest,face` runs face stage successfully
- [ ] `--face-person person_00` isolates a single face via roto matte
- [ ] `--skip-existing` skips face stage when `face_params.pkl` exists
- [ ] `pytest tests/test_run_face.py` passes
- [ ] Face stage appears in `--list-stages` output and web UI
