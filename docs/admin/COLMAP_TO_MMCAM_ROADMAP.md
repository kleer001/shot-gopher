# COLMAP → Matchmove Camera (mmcam) Roadmap

Rename the user-facing "colmap" stage to "matchmove_camera" with short alias "mmcam."
The underlying tool is still COLMAP — only the public-facing name changes.

## Rationale

- "COLMAP" is an implementation detail; users care about the *capability* (camera tracking)
- "matchmove_camera" / "mmcam" is descriptive of the VFX task being performed
- Aligns naming with other stages that describe *what* they do, not *how*

## Decisions

| Question | Answer |
|----------|--------|
| Stage key | `"matchmove_camera"` |
| CLI long flags | `--matchmove-camera-*` (e.g. `--matchmove-camera-quality`) |
| CLI short flags | `--mmcam-*` (e.g. `--mmcam-quality`), plus keep existing single-letter shortcuts (`-q`, `-d`, `-m`, `-M`) |
| On-disk output dir | `mmcam/` (not `matchmove_camera/` — too long) |
| Internal COLMAP references | Keep (conda env name `colmap`, binary calls, `run_colmap.py` internals) |
| Script filename | Rename `run_colmap.py` → `run_matchmove_camera.py` |
| Backward compat | None. Clean break. |

## Files to Modify

### Phase 1: Constants & Config (Cascading Root)

| File | Line(s) | Changes Required |
|------|---------|------------------|
| `scripts/pipeline_constants.py:31` | Rename stage key `"colmap"` → `"matchmove_camera"`, update description |
| `scripts/pipeline_constants.py:39` | Update `STAGE_ORDER`: `"colmap"` → `"matchmove_camera"` |
| `scripts/pipeline_constants.py:44` | Update `STAGES_REQUIRING_FRAMES`: `"colmap"` → `"matchmove_camera"` |
| `scripts/pipeline_config.py:34-38` | Rename 5 fields: `colmap_quality` → `mmcam_quality`, `colmap_dense` → `mmcam_dense`, `colmap_mesh` → `mmcam_mesh`, `colmap_use_masks` → `mmcam_use_masks`, `colmap_max_size` → `mmcam_max_size` |
| `scripts/pipeline_config.py:82-86,113-117` | Update field names in both `from_args()` branches |

### Phase 2: CLI Arguments

| File | Line(s) | Changes Required |
|------|---------|------------------|
| `scripts/run_pipeline.py:236-261` | Rename all 5 `--colmap-*` arguments to `--matchmove-camera-*` / `--mmcam-*` (dual long-form) |
| `scripts/run_pipeline.py:419-423` | Update config construction: `colmap_quality=` → `mmcam_quality=`, etc. |
| `scripts/run_pipeline.py:11-12` | Update module docstring examples |

**Argument mapping:**

| Old | New (long) | New (short alias) | Letter |
|-----|-----------|-------------------|--------|
| `--colmap-quality` | `--matchmove-camera-quality` | `--mmcam-quality` | `-q` |
| `--colmap-dense` | `--matchmove-camera-dense` | `--mmcam-dense` | `-d` |
| `--colmap-mesh` | `--matchmove-camera-mesh` | `--mmcam-mesh` | `-m` |
| `--colmap-no-masks` | `--matchmove-camera-no-masks` | `--mmcam-no-masks` | `-M` |
| `--colmap-max-size` | `--matchmove-camera-max-size` | `--mmcam-max-size` | (none) |

### Phase 3: Stage Runners

| File | Line(s) | Changes Required |
|------|---------|------------------|
| `scripts/stage_runners.py:99-145` | Update `run_colmap_reconstruction()` → `run_matchmove_camera()`: update script path to `run_matchmove_camera.py` |
| `scripts/stage_runners.py:42,53` | Update `__all__`: `run_colmap_reconstruction` → `run_matchmove_camera`, `run_stage_colmap` → `run_stage_matchmove_camera` |
| `scripts/stage_runners.py:824-857` | Rename `run_stage_colmap()` → `run_stage_matchmove_camera()`: update print banner, path from `"colmap"` → `"mmcam"`, field refs |
| `scripts/stage_runners.py:1000` | Update `STAGE_HANDLERS`: key `"colmap"` → `"matchmove_camera"`, value → `run_stage_matchmove_camera` |
| `scripts/stage_runners.py:929` | `run_stage_gsir()`: update `colmap_sparse` path from `"colmap"` → `"mmcam"` |
| `scripts/stage_runners.py:970,975` | `run_stage_camera()`: update variable names and print text from "COLMAP" → "matchmove camera" |
| `scripts/stage_runners.py:884` | `run_stage_mocap()`: update "Using COLMAP camera data" print message |

### Phase 4: Core Script Rename

| File | Action |
|------|--------|
| `scripts/run_colmap.py` | Rename to `scripts/run_matchmove_camera.py` |
| `scripts/run_colmap.py` (internal) | Keep COLMAP binary calls, conda env name. Only update module docstring and user-facing print messages |
| `scripts/debug_colmap_images.py` | Rename to `scripts/debug_mmcam_images.py`, update references to output dir |

### Phase 5: Shell Launchers

| File | Action |
|------|--------|
| `src/run-colmap.sh` | Rename to `src/run-matchmove-camera.sh`, update internal script path |
| `src/run-colmap.bat` | Rename to `src/run-matchmove-camera.bat`, update internal script path |

### Phase 6: Downstream Consumers

| File | Changes Required |
|------|------------------|
| `scripts/validate_gsir.py` | Update paths from `"colmap"` → `"mmcam"` (intrinsics/extrinsics validation) |
| `scripts/run_gsir.py` | Update COLMAP data path references from `"colmap"` → `"mmcam"` |
| `scripts/run_mocap.py` | Update intrinsics path references from `"colmap"` → `"mmcam"` |
| `scripts/export_camera.py` | Update default camera dir references |
| `scripts/subprocess_utils.py:278-279` | Rename `create_colmap_patterns()` → `create_mmcam_patterns()` |

### Phase 7: Web Layer

| File | Changes Required |
|------|------------------|
| `web/config/pipeline_config.json:74-85` | Rename stage key `"colmap"` → `"matchmove_camera"`, update `outputDir` to `"mmcam"` |
| `web/config/pipeline_config.json:93-94` | Update `gsir` and `mocap` dependencies from `"colmap"` → `"matchmove_camera"` |
| `web/config/pipeline_config.json:123` | Update "all" preset stage list (if it still exists after `-s all` removal) |
| `web/static/js/controllers/ProjectsController.js:16,23,42,311` | Rename `colmap` → `matchmove_camera` in `ALL_STAGES`, stage mappings, dependency graph, display names |
| `web/pipeline_runner.py:217` | Update `stage_options.get("colmap")` → `stage_options.get("matchmove_camera")` |
| `web/services/pipeline_service.py:82` | Update stage name in error message |
| `web/repositories/project_repository.py:163` | Update output dir path from `"colmap"` → `"mmcam"` |
| `web/server.py:62` | Update stage name in docstring |

### Phase 8: Tests

| File | Changes Required |
|------|------------------|
| `tests/test_run_colmap.py` | Rename to `tests/test_run_matchmove_camera.py`, update internal references |
| `tests/test_run_gsir.py:13,49,71,78,86-93,114,122-127,138,155-161` | Rename `colmap` paths → `mmcam`, update `COLMAP_AVAILABLE` variable name, update stage ordering test `"colmap"` → `"matchmove_camera"` |
| `tests/test_validate_gsir.py:17,85,93,400-402` | Rename `count_colmap_images` → `count_mmcam_images`, update `colmap/` paths → `mmcam/` |
| `web/tests/unit/test_outputs_api.py:36,50,146-152,215,229,250,257` | Rename `"colmap"` → `"matchmove_camera"` and paths → `"mmcam"` |
| `web/tests/integration/test_outputs_api.py:52-56,104,123-124` | Rename `"colmap"` → `"matchmove_camera"` and paths → `"mmcam"` |

### Phase 9: Documentation

| File | Changes Required |
|------|------------------|
| `README.md` | Replace COLMAP stage references with matchmove_camera/mmcam |
| `docs/reference/stages.md` | Update stage name, CLI options, output paths, dependency info |
| `docs/reference/cli.md` | Rename all `--colmap-*` options to `--matchmove-camera-*` / `--mmcam-*` |
| `docs/reference/scripts.md` | Update `run_colmap.py` → `run_matchmove_camera.py` |
| `docs/testing.md` | Update `--stages ingest,colmap` → `--stages ingest,matchmove_camera` |
| `docs/troubleshooting.md` | Update COLMAP troubleshooting section headings and paths |
| `docs/which-stages.md` | Update stage name references |
| `docs/admin/LICENSE_AUDIT_REPORT.md` | Update stage name (keep COLMAP as the tool name in the entry) |

## Output Directory Migration

| Old Path | New Path |
|----------|----------|
| `project/colmap/sparse/0/` | `project/mmcam/sparse/0/` |
| `project/colmap/dense/` | `project/mmcam/dense/` |
| `project/colmap/colmap.log` | `project/mmcam/mmcam.log` |

The `camera/` directory (extrinsics, intrinsics, exports) stays as-is — it's already
abstracted from the COLMAP implementation.

## What Stays as "COLMAP"

These are internal implementation details and should NOT be renamed:

- `COLMAP_CONDA_ENV = "colmap"` (conda environment name)
- Actual COLMAP binary invocations (`colmap feature_extractor`, etc.)
- COLMAP-specific function internals in `run_matchmove_camera.py`
- COLMAP file format references (`cameras.bin`, `images.bin`, `points3D.bin`)

## Breadcrumbs (temporary debugging aids)

Add during refactor. Remove before merging to main.

### 1. Output directory assertion in stage runner

In `run_stage_matchmove_camera()`, add an assertion that confirms the output dir is
correct:

```python
def run_stage_matchmove_camera(ctx, config):
    print("\n=== Stage: matchmove_camera ===")
    mmcam_sparse = ctx.project_dir / "mmcam" / "sparse" / "0"
    assert "colmap" not in str(mmcam_sparse), \
        f"BREADCRUMB: path still contains 'colmap': {mmcam_sparse}"  # remove after verifying
```

**Why:** The rename touches 23 files. If any path construction still concatenates
`"colmap"` instead of `"mmcam"`, this blows up at stage entry, not deep inside COLMAP
processing where the error would be `FileNotFoundError` on a missing directory.

### 2. Config field guard in run_pipeline.py

After renaming the config fields, temporarily add a guard in config construction:

```python
config = PipelineConfig(
    ...
    mmcam_quality=args.mmcam_quality,
    ...
)
# BREADCRUMB: remove after verifying all callers use mmcam_ fields
assert not hasattr(args, 'colmap_quality'), \
    "BREADCRUMB: argparse still defines --colmap-quality"
```

**Why:** If one of the 5 argparse arguments was missed during rename, this catches it
at startup rather than silently using a default value.

### 3. Downstream consumer path checks

In each of the three downstream consumers, add a path existence check with a
rename-aware message:

```python
# In run_gsir.py, run_mocap.py, validate_gsir.py:
mmcam_dir = project_dir / "mmcam"
if not mmcam_dir.exists():
    colmap_dir = project_dir / "colmap"
    if colmap_dir.exists():
        print("BREADCRUMB: Found 'colmap/' but expected 'mmcam/'. "
              "This project may predate the rename.", file=sys.stderr)  # remove after verifying
```

**Why:** During the transition, existing project directories on disk still have `colmap/`.
This message clearly tells the user (and developer) what happened, rather than a generic
"no camera data found" error. Particularly useful when running the refactored code against
existing test fixtures.

### 4. Shell launcher tombstones

After renaming the shell launchers, leave tombstone scripts at the old paths:

```bash
# src/run-colmap.sh
#!/bin/bash
echo "ERROR: run-colmap.sh has been renamed to run-matchmove-camera.sh" >&2
echo "See: docs/admin/COLMAP_TO_MMCAM_ROADMAP.md" >&2
exit 1
```

**Why:** Anyone with the old script in their PATH or aliases gets an actionable error.
Delete after one release cycle.

## Verification Checklist

- [ ] `python scripts/run_pipeline.py --list-stages` shows `matchmove_camera` (not `colmap`)
- [ ] `python scripts/run_pipeline.py --help` shows `--matchmove-camera-*` and `--mmcam-*` flags
- [ ] `-q`, `-d`, `-m`, `-M` single-letter flags still work
- [ ] `grep -r '"colmap"' scripts/pipeline_constants.py` returns no results
- [ ] `grep -r 'colmap_quality' scripts/` returns no results
- [ ] Output lands in `project/mmcam/sparse/0/` (not `project/colmap/`)
- [ ] GS-IR stage finds mmcam data correctly
- [ ] Mocap stage finds mmcam intrinsics correctly
- [ ] Camera export stage finds mmcam extrinsics correctly
- [ ] All tests pass: `pytest tests/`
- [ ] Web UI shows "Camera Tracking" with correct stage key

## Breadcrumb Removal Checklist

- [ ] Remove `assert "colmap" not in str(mmcam_sparse)` from `run_stage_matchmove_camera()`
- [ ] Remove `assert not hasattr(args, 'colmap_quality')` from `run_pipeline.py`
- [ ] Remove `colmap_dir.exists()` breadcrumb messages from `run_gsir.py`, `run_mocap.py`, `validate_gsir.py`
- [ ] Delete tombstone scripts `src/run-colmap.sh` and `src/run-colmap.bat`
