# Pipeline Script Refactor Roadmap

**Goal:** Split `run_pipeline.py` (1544 lines) into 6 focused modules.

## Current Structure Analysis

| Section | Lines | Description |
|---------|-------|-------------|
| Imports/Constants | 1-77 | Stage definitions, formats |
| General Utilities | 80-507 | FFmpeg, subprocess, file ops |
| Workflow Manipulation | 520-794 | ComfyUI JSON updates |
| Stage Runners | 267-520, 797-855 | External script wrappers |
| Main Orchestration | 857-1373 | `run_pipeline()` function |
| CLI | 1376-1544 | argparse `main()` |

## Proposed Module Split (6 Modules)

### 1. `pipeline_constants.py` (~40 lines)
Central location for all pipeline constants.

**Move:**
- `START_FRAME`
- `SUPPORTED_FORMATS`
- `STAGES` dict
- `STAGE_ORDER` list
- `STAGES_REQUIRING_FRAMES` set
- `WORKFLOW_TEMPLATES_DIR`

### 2. `pipeline_utils.py` (~180 lines)
General-purpose utilities with no pipeline-specific logic.

**Move:**
- `run_command()` - subprocess wrapper with streaming
- `get_frame_count()` - ffprobe frame counter
- `extract_frames()` - ffmpeg frame extraction
- `get_video_info()` - ffprobe metadata
- `generate_preview_movie()` - ffmpeg movie generation
- `get_image_dimensions()` - ffprobe dimensions
- `clear_gpu_memory()` - torch VRAM cleanup

### 3. `workflow_utils.py` (~150 lines)
ComfyUI workflow JSON manipulation.

**Move:**
- `get_comfyui_output_dir()`
- `refresh_workflow_from_template()`
- `update_segmentation_prompt()`
- `update_matanyone_input()`
- `update_cleanplate_resolution()`

### 4. `matte_utils.py` (~100 lines)
Matte/mask combination and manipulation.

**Move:**
- `combine_mattes()`
- `combine_mask_sequences()`

### 5. `stage_runners.py` (~120 lines)
Thin wrappers that invoke external scripts.

**Move:**
- `run_export_camera()`
- `run_colmap_reconstruction()`
- `export_camera_to_vfx_formats()`
- `run_mocap()`
- `run_gsir_materials()`
- `setup_project()`

### 6. `run_pipeline.py` (~450 lines, trimmed)
Orchestration only - imports from above modules.

**Keep:**
- `sanitize_stages()`
- `run_pipeline()` - main orchestration
- `main()` - CLI entry point

## Dependency Graph

```
run_pipeline.py
├── pipeline_constants.py (no deps)
├── pipeline_utils.py (no internal deps)
├── matte_utils.py (no internal deps)
├── workflow_utils.py
│   ├── pipeline_constants.py
│   └── pipeline_utils.py (for get_image_dimensions)
└── stage_runners.py
    └── pipeline_utils.py (for run_command)
```

## Implementation Steps

1. **Create `pipeline_constants.py`**
   - Extract all constants
   - Add `__all__` exports

2. **Create `pipeline_utils.py`**
   - Extract utility functions
   - Add `__all__` exports

3. **Create `matte_utils.py`**
   - Extract matte combination functions
   - Add `__all__` exports

4. **Create `workflow_utils.py`**
   - Extract workflow manipulation functions
   - Import from `pipeline_constants` and `pipeline_utils`

5. **Create `stage_runners.py`**
   - Extract external script wrappers
   - Import `run_command` from `pipeline_utils`

6. **Trim `run_pipeline.py`**
   - Update imports to use new modules
   - Remove moved functions
   - Result: ~450 lines of pure orchestration

7. **Verify & Test**
   - Syntax check all new modules
   - Test `--list-stages`, `--help`
   - Run a simple pipeline

## File Size Targets

| Module | Target Lines | Content |
|--------|-------------|---------|
| `pipeline_constants.py` | ~40 | Constants and stage definitions |
| `pipeline_utils.py` | ~180 | FFmpeg/subprocess utilities |
| `matte_utils.py` | ~100 | Matte combination functions |
| `workflow_utils.py` | ~150 | Workflow JSON manipulation |
| `stage_runners.py` | ~120 | External script wrappers |
| `run_pipeline.py` | ~450 | Orchestration + CLI |
| **Total** | ~1040 | ~500 lines saved via modularization |

---

**Priority:** Medium (code quality, maintainability)
**Risk:** Low (pure refactor, no behavior change)
**Testing:** Existing integration tests cover all paths
**Status:** IMPLEMENTED
