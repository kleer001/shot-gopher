# ProPainter Deprecation Roadmap

Remove ProPainter cleanplate tooling. The temporal median approach becomes the sole cleanplate
method, and the stage description changes to signal "static camera only."

## Rationale

- ProPainter does not hold up in real-world production testing
- Temporal median is faster, lighter, and sufficient for static-camera shots
- Removing ProPainter eliminates a non-commercial dependency (NTU S-Lab license)
- Simplifies install wizard, VRAM management, and ComfyUI node requirements

## Decisions

| Question | Answer |
|----------|--------|
| Stage key | Keep `"cleanplate"` (no rename) |
| Stage description | Update to indicate static-shot-only: `"Clean plate via temporal median (static camera)"` |
| `--cleanplate-median` flag | Remove (median is now the only path, always used) |
| ProPainter env vars | Remove (`PROPAINTER_INTERNAL_SCALE`, `PROPAINTER_REFINE_ITERS`, `PROPAINTER_NUM_FLOWS`) |
| Workflow templates | Delete the 3 ProPainter-based templates |
| ComfyUI requirement | `cleanplate` no longer needs ComfyUI |

## Files to Modify

### Phase 1: Core Pipeline

| File | Changes Required |
|------|------------------|
| `scripts/pipeline_constants.py:30` | Update `STAGES["cleanplate"]` description |
| `scripts/stage_runners.py:746-821` | Remove ProPainter branch from `run_stage_cleanplate()`, always run median |
| `scripts/stage_runners.py:20` | Remove `cleanplate_median` import if it stays, keep `run_cleanplate_median` |
| `scripts/pipeline_config.py:50` | Remove `cleanplate_use_median` field from `PipelineConfig` |
| `scripts/pipeline_config.py:152` | Remove `cleanplate_use_median` field from `StageContext` |
| `scripts/pipeline_config.py:189` | Remove `cleanplate_use_median` from `StageContext.from_config()` |
| `scripts/pipeline_config.py:95,126` | Remove `cleanplate_use_median` from `PipelineConfig.from_args()` (both branches) |
| `scripts/run_pipeline.py:70` | Remove `"cleanplate"` from `comfyui_stages` set (no longer needs ComfyUI) |
| `scripts/run_pipeline.py:312-316` | Remove `--cleanplate-median` argument |
| `scripts/run_pipeline.py:408-433` | Remove `cleanplate_use_median=args.cleanplate_median` from inline config construction (note: `PipelineConfig.from_args()` handles this separately) |

### Phase 2: Workflow Utilities

| File | Changes Required |
|------|------------------|
| `scripts/workflow_utils.py:29-54` | Remove `_get_propainter_internal_scale()` and `_get_propainter_quality_params()` |
| `scripts/workflow_utils.py:216-258` | Remove `update_cleanplate_resolution()` function |
| `scripts/workflow_utils.py:261-352` | Remove `update_propainter` parameter and ProPainterInpaint handling from `update_workflow_resolution()` |
| `scripts/workflow_utils.py:13-19` | Remove `update_cleanplate_resolution` from `__all__` |
| `scripts/comfyui_utils.py:168` | Remove `ProPainterInpaint` from widget mapping dict |

### Phase 3: Workflow Templates

| File | Action |
|------|--------|
| `workflow_templates/03_cleanplate.json` | Delete |
| `workflow_templates/03_cleanplate_batched.json` | Delete |
| `workflow_templates/03_cleanplate_chunk_template.json` | Delete |

### Phase 4: Install Wizard

| File | Changes Required |
|------|------------------|
| `scripts/install_wizard/wizard.py:208-213` | Remove `ComfyUI-ProPainter-Nodes` GitRepoInstaller entry (full block) |
| `scripts/install_wizard/installers.py:342-343` | Remove `ComfyUI_ProPainter_Nodes` from custom nodes list |

### Phase 5: Stage Runners Cleanup

All `update_propainter=False` and `update_propainter=True` call sites in `stage_runners.py`:

| Location | Change |
|----------|--------|
| `stage_runners.py:474` (interactive) | Remove `update_propainter=False` arg |
| `stage_runners.py:535` (depth) | Remove `update_propainter=False` arg |
| `stage_runners.py:587` (roto) | Remove `update_propainter=False` arg |

### Phase 6: Web UI

| File | Changes Required |
|------|------------------|
| `web/config/pipeline_config.json:63-72` | Update cleanplate stage: `requiresComfyUI: false`, update description |

### Phase 7: Documentation

| File | Changes Required |
|------|------------------|
| `docs/reference/stages.md` | Remove ProPainter method from cleanplate section, update to median-only |
| `docs/reference/cli.md` | Remove `--cleanplate-median` option, remove ProPainter env vars |
| `README.md` | Remove ProPainter from dependencies list |
| `docs/admin/LICENSE_AUDIT_REPORT.md` | Remove ProPainter license entry |

## Key Transformation: `run_stage_cleanplate()`

**Before (two paths):**
```python
if ctx.cleanplate_use_median:
    run_cleanplate_median(ctx.project_dir)
else:
    update_workflow_resolution(workflow_path, ..., update_propainter=True)
    run_comfyui_workflow(workflow_path, ...)
```

**After (one path):**
```python
run_cleanplate_median(ctx.project_dir)
```

The entire ComfyUI workflow path, resolution scaling, and GPU memory cleanup for this stage
are removed. The function becomes significantly simpler.

## Key Transformation: `update_workflow_resolution()`

**Remove:**
- `update_propainter` parameter
- `propainter_scale`, `propainter_width`, `propainter_height` calculations
- `ProPainterInpaint` node handling in the loop

**Keep:**
- `update_loaders` and `update_scales` logic (used by depth, roto, interactive stages)
- `_calculate_processing_resolution()` (still useful for other stages)
- `_get_max_processing_dimensions()` (still used for general resolution capping)

## Breadcrumbs (temporary debugging aids)

Add these during the refactor. Remove before merging to main.

### 1. Cleanplate stage entry confirmation

In `run_stage_cleanplate()`, add a print at the top to confirm the new code path runs:

```python
def run_stage_cleanplate(ctx, config):
    print("\n=== Stage: cleanplate (median-only) ===")  # BREADCRUMB: remove after verifying
```

**Why:** Confirms the old ProPainter branch is truly gone. If you ever see
`=== Stage: cleanplate ===` without `(median-only)`, stale code is running.

### 2. Guard against phantom update_propainter kwarg

After removing `update_propainter` from `update_workflow_resolution()`, temporarily add
this at the top of the function body:

```python
def update_workflow_resolution(workflow_path, width, height, *, update_loaders=True, update_scales=True, **kwargs):
    if "update_propainter" in kwargs:
        raise RuntimeError(
            "BREADCRUMB: update_propainter was passed to update_workflow_resolution() "
            f"— caller still using old API. kwargs={kwargs}"
        )
```

**Why:** Any lingering call site that still passes `update_propainter=True/False` will
explode immediately with a traceback pointing to the exact caller, instead of silently
being ignored. Remove the `**kwargs` guard once all 4 call sites are confirmed clean.

### 3. Workflow template tombstone

After deleting the three `03_cleanplate*.json` files, place a single file:

```
workflow_templates/03_cleanplate_REMOVED.txt
```

Contents: `ProPainter cleanplate removed. See docs/admin/PROPAINTER_DEPRECATION_ROADMAP.md`

**Why:** If any code still tries to `refresh_workflow_from_template("03_cleanplate.json")`,
the file-not-found will be immediate and obvious. The tombstone tells the developer *why*
and *where to look*. Delete after merging.

## Verification Checklist

- [ ] `python scripts/run_pipeline.py --help` shows no `--cleanplate-median` flag
- [ ] `python scripts/run_pipeline.py --list-stages` shows updated cleanplate description
- [ ] `grep -ri "propainter" scripts/` returns no results
- [ ] `grep -ri "propainter" workflow_templates/` returns no results (tombstone .txt is fine)
- [ ] `grep -ri "cleanplate_use_median" scripts/` returns no results
- [ ] All tests pass: `pytest tests/`
- [ ] Cleanplate stage runs temporal median without any flags

## Breadcrumb Removal Checklist

- [ ] Remove `(median-only)` from cleanplate stage banner
- [ ] Remove `**kwargs` / `update_propainter` guard from `update_workflow_resolution()`
- [ ] Delete `workflow_templates/03_cleanplate_REMOVED.txt` tombstone
