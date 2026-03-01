# Hands Stage Web UI Integration

Add the "hands" (WiLoR hand pose estimation) stage to the web UI.

## Scope

Backend is fully wired (`STAGE_HANDLERS["hands"]` exists, CLI dispatches correctly).
Only the web layer needs updating.

## Files to Modify

### 1. `web/config/pipeline_config.json`

Add after the `"mocap"` entry:

```json
"hands": {
  "name": "Hand Poses",
  "displayName": "Estimate Hand Poses (WiLoR)",
  "description": "Extract detailed hand pose using WiLoR, re-export body mesh with articulated fingers",
  "outputDir": "mocap",
  "estimatedTimePerFrame": 0.8,
  "required": false,
  "dependencies": ["mocap"],
  "defaultEnabled": false,
  "requiresComfyUI": false
}
```

### 2. `web/static/js/controllers/ProjectsController.js`

Three additions:

| Location | Current | Add |
|----------|---------|-----|
| `ALL_STAGES` array (line ~16) | `...'mocap', 'gsir'...` | `...'mocap', 'hands', 'gsir'...` |
| `STAGE_OUTPUT_DIRS` (line ~18) | after `mocap: 'mocap',` | `hands: 'mocap',` |
| `stageLabels` (line ~304) | after `mocap: 'Human Mocap',` | `hands: 'Hand Poses',` |

No stage options needed (hands has no user-configurable parameters).

## No Changes Required

- `web/pipeline_runner.py` -- dispatches via CLI, already works
- `scripts/stage_runners.py` -- `STAGE_HANDLERS["hands"]` already registered
- `scripts/pipeline_constants.py` -- `"hands"` in `STAGES`, `STAGE_ORDER`, `STAGES_REQUIRING_FRAMES`
- `scripts/run_pipeline.py` -- already dispatches all stages from `STAGE_HANDLERS`

## Verification

- [ ] `test_refactor_validation.py::TestWebConsistencyPostRefactor::test_web_config_stages_match_pipeline_constants` passes
- [ ] Web UI shows "Hand Poses" after "Human Mocap" in stage list
- [ ] "Hand Poses" toggle is disabled unless "Human Mocap" is enabled (dependency)
- [ ] Running hands stage from web UI produces `hand_poses.npz` + re-exported `body_motion.abc`
