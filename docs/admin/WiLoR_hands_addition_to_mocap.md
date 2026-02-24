# Roadmap: WiLoR Hand Pose Integration into Mocap Pipeline

**Status:** Proposed
**Date:** 2026-02-23
**Tracking:** [GitHub Issue TBD]

---

## Problem Statement

GVHMR outputs body pose parameters (21 joints, 63 DoF) with no hand articulation.
The pipeline already uses SMPL-X for mesh generation (10475 vertices, switched in
commit `5f56037`), but hand parameters are zeroed — producing flat, T-pose hands
regardless of what the performer's hands are doing on camera. GVHMR's preview video
(`1_incam.mp4`) misleadingly shows curled fingers — this comes from the renderer's
cosmetic hand pose, not from actual estimated data.

## Proposed Solution

Run **WiLoR** (CVPR 2025) as a dedicated hand estimator alongside GVHMR, then feed
the hand poses into the existing SMPL-X mesh generation.

- **GVHMR** provides: world-space body motion, camera parameters, body shape
- **WiLoR** provides: per-frame MANO hand poses (left and right)
- **SMPL-X** consumes both: body_pose from GVHMR + hand_pose from WiLoR

### Why WiLoR

| Criteria | WiLoR | HaMeR | Dyn-HaMeR |
|----------|-------|-------|-----------|
| Speed | ~130 FPS detection | ~1.3 FPS (ViTDet) | Slower (adds SLAM) |
| Accuracy | Equal or better | Baseline | Best (temporal) |
| Built-in detector | Yes (DarkNet) | No (needs ViTDet) | No (uses HaMeR) |
| Model size | ~25 MB | ~819 MB | Larger |
| Output format | MANO (45 per hand) | MANO (45 per hand) | MANO (45 per hand) |
| Temporal smoothing | No | No | Yes |
| Complexity | Low | Medium | High |

WiLoR gives the best speed/accuracy/simplicity tradeoff. If temporal jitter becomes
a problem, Dyn-HaMeR can be evaluated later as it uses HaMeR internally and the
output format is identical.

### Why This Works

SMPL-X's hand parameterization is literally MANO embedded inside SMPL-X. The smplx
library maintainers confirm: "The hand poses are the same" and "there is no need for
an offset R because MANO shares a coordinate system with SMPLX." The 45-value
`hand_pose` from WiLoR maps directly to SMPL-X `left_hand_pose` / `right_hand_pose`
with zero conversion.

Critical detail: we keep GVHMR's wrist rotation (from `body_pose` joints 20-21) and
only take WiLoR's 15 finger joint rotations per hand. This avoids any wrist alignment
issues between the two estimators.

Reference implementation: `VincentHu19/Mano2Smpl-X` on GitHub — tested specifically
with GVHMR + HaMeR, MIT license.

---

## Current State

The pipeline already uses SMPL-X throughout (commit `5f56037`):

- `export_mocap.py` calls `smplx.create(model_type='smplx')` at lines 247 and 538
- `get_body_model_path()` (line 179) resolves SMPLX model files
- `camera_alignment.py` computes J_pelvis from the SMPLX model
- Mesh output is 10475 vertices per frame

What's missing: the `model()` forward call (line 280) only passes `body_pose` and
`global_orient` — `left_hand_pose` and `right_hand_pose` default to zeros internally.

---

## Prerequisites

### Already Available

- SMPL-X model files already installed at:
  - `.vfx_pipeline/GVHMR/inputs/checkpoints/body_models/smplx/`
  - `.vfx_pipeline/models/smplx/models/smplx/`
- The `smplx` Python package is already in the GVHMR conda environment
- `export_mocap.py` already uses `model_type='smplx'` — only needs hand_pose args
- Blender ABC export pipeline is topology-agnostic (reads OBJ vertex positions)

### Needs Investigation

- WiLoR's exact conda/pip dependency footprint — can it share the `gvhmr` env or
  does it need its own?
- WiLoR's GPU memory usage — can it coexist with GVHMR in a single pipeline run,
  or does it need sequential execution with GPU memory clearing between stages?
- Whether WiLoR handles left/right hand flipping internally or if we need the
  manual axis-angle sign flip (negate y,z components for left hands)

---

## Implementation Phases

### Phase 1: WiLoR Installation

**Goal:** WiLoR runs standalone on a test video and produces MANO hand poses.

**Tasks:**

1. Install WiLoR into `.vfx_pipeline/tools/wilor/` following the sandboxed install
   pattern (never install to user home directories)
2. Create a conda environment — either extend `gvhmr` or create `wilor` depending
   on dependency conflicts
3. Add WiLoR to the install wizard (`scripts/install_wizard/installers.py`) following
   the existing `ComponentInstaller` pattern
4. Verify WiLoR runs on sample frames and produces MANO parameters:
   - `hand_pose` (45 values per hand, axis-angle)
   - `global_orient` (3 values per hand — we will discard this)
   - Left/right hand classification

**Validation:** Run WiLoR on a few frames from an existing test project, confirm
sensible-looking hand poses are produced.

---

### Phase 2: Hand Estimation Pipeline Step

**Goal:** A new script estimates hand poses for every frame of a mocap sequence and
saves them alongside the existing GVHMR output.

**Tasks:**

1. Create `scripts/run_hand_estimation.py` — standalone script that:
   - Takes a project directory and person identifier
   - Reads the source video frames (same frames GVHMR used)
   - Runs WiLoR per frame to detect hands and estimate MANO parameters
   - Associates detected hands with left/right based on WiLoR's classifier
   - Outputs `hand_poses.npz` with arrays `left_hand_pose[N, 45]` and
     `right_hand_pose[N, 45]`, where N matches the frame count from GVHMR
   - Handles missing detections by writing zeros for occluded frames

2. Output location: `mocap/<person>/hand_poses.npz` (sibling to `hmr4d_results.pt`)

3. Frame alignment: GVHMR may subsample frames (based on `mocap_fps`). The hand
   estimator must process the same frames at the same indices. Read the frame list
   from GVHMR's output metadata to ensure alignment.

**Validation:** Spot-check a few frames — verify the hand_pose values are non-zero
when hands are visible and zero when occluded. Verify array shapes match GVHMR's
frame count.

---

### Phase 3: Hand Pose Integration in Export

**Goal:** `export_mocap.py` passes WiLoR hand poses to the existing SMPL-X model,
producing meshes with articulated fingers.

No topology change — the pipeline already outputs SMPL-X meshes (10475 vertices).
The only change is filling in the hand_pose parameters that currently default to zero.

**Tasks:**

1. **Modify `convert_gvhmr_to_motion()`** (`export_mocap.py`, line 65):
   - Load `hand_poses.npz` if it exists alongside the GVHMR output
   - Extend the pose concatenation (lines 151-162) from:
     `[global_orient(3) + body_pose(63)]` = 66 dims
     to:
     `[global_orient(3) + body_pose(63) + left_hand(45) + right_hand(45)]` = 156 dims
   - If hand_poses.npz is missing, current behavior is preserved (66 dims, SMPL-X
     zeros the hand params internally)
   - Store left/right hand poses as separate keys in `motion.pkl` for clarity

2. **Modify `generate_meshes()`** (`export_mocap.py`, line 210):
   - Update the `smplx.create()` call (line 247) to add:
     - `flat_hand_mean=True` — makes zero-pose hands flat (matching current behavior)
       rather than using MANO's slightly-curled mean pose
     - `use_pca=False` — we're passing raw axis-angle rotations, not PCA components
   - Update the model forward call (line 280) to pass `left_hand_pose` and
     `right_hand_pose` as separate tensor arguments when available in motion data

3. **No Blender export changes needed.** The Blender scripts
   (`scripts/blender/export_mesh_alembic.py`, `scripts/blender/export_mesh_usd.py`)
   validate inter-frame vertex consistency (line 85-101) but don't check for a
   specific vertex count. Since the topology is already SMPL-X, nothing changes.

4. **Add config option** (`scripts/pipeline_config.py`, line 17):
   - New field: `mocap_hand_estimation: bool = False`
   - When True, Phase 2 runs before export
   - When False, current behavior is preserved exactly (SMPL-X with flat hands)
   - This keeps the feature opt-in until validated

**Validation:**
- Generate meshes from an existing GVHMR result + hand_poses.npz
- Visually inspect in Blender: fingers should be articulated, body pose unchanged
- Verify vertex count remains 10475 throughout the OBJ sequence
- Verify Alembic export completes without errors
- Compare body motion with and without hand data — body should be identical

---

### Phase 4: Pipeline Integration

**Goal:** Hand estimation runs automatically as part of the mocap stage when enabled.

**Tasks:**

1. **Wire into `run_mocap.py`** (`run_mocap_pipeline()`, line 1067):
   - After `run_gvhmr_motion_tracking()` (line 1144) and before the camera export
     section (line 1161), add a call to the hand estimation script
   - Run in the WiLoR conda environment (same pattern as GVHMR conda execution)
   - Pass through project_dir, person identifier, and frame range

2. **Wire into `stage_runners.py`**:
   - `run_stage_mocap()` (line 1035) calls `run_mocap()` (line 222) which calls
     `run_mocap.py` as a subprocess — hand estimation will be included automatically
     once wired into `run_mocap_pipeline()`
   - No changes needed at the stage runner level
   - Pass `mocap_hand_estimation` config field through to `run_mocap()` (line 222)
     and forward it to `run_mocap.py` as a CLI flag

3. **GPU memory management:**
   - GVHMR and WiLoR should run sequentially, not concurrently
   - Call `clear_gpu_memory()` between GVHMR and WiLoR if sharing a GPU
   - WiLoR is small (~25 MB model) so memory pressure should be minimal

4. **CLI passthrough:**
   - `run_mocap.py` needs `--hand-estimation` flag
   - `run_pipeline.py` needs `--hand-estimation` flag or reads from project config
   - `PipelineConfig.from_args()` (line 56) maps the CLI arg to the config field

**Validation:**
- Run full mocap stage on a test project with `mocap_hand_estimation=True`
- Verify the pipeline completes end-to-end: GVHMR → WiLoR → SMPL-X mesh → ABC
- Verify the pipeline still works with `mocap_hand_estimation=False` (no regression)

---

### Phase 5: Temporal Smoothing (Optional)

**Goal:** Reduce frame-to-frame hand jitter for cleaner animation.

WiLoR is per-frame with no temporal model. Hand poses may jitter, especially for
partially occluded or fast-moving hands.

**Tasks:**

1. Apply a simple smoothing filter (e.g., Savitzky-Golay or exponential moving
   average) to the `hand_poses.npz` arrays before merging into motion.pkl
2. Interpolate across short occlusion gaps (hands missing for <N frames) using
   the neighboring detected poses
3. For long occlusion gaps, keep zeros (flat hands are better than hallucinated
   poses)

**Validation:** Side-by-side comparison of smoothed vs raw hand animation. Smoothed
should look more natural without losing intentional hand gestures.

This phase is deferred until Phase 4 is validated. Jitter may be acceptable for
many use cases, and over-smoothing can lose detail.

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| WiLoR dependency conflicts with gvhmr env | Medium | Medium | Separate conda env if needed |
| Hand detection fails on small/distant hands | High | Low | Zero-fill (flat hands) — same as current behavior |
| Frame count mismatch between GVHMR and WiLoR | Low | High | Read frame indices from GVHMR metadata |
| WiLoR left/right flip produces wrong hand | Low | High | Validate against video overlay in Phase 2 |
| GPU OOM running WiLoR after GVHMR | Low | Medium | Sequential execution + memory clearing |

---

## Open Questions

1. **Separate env or shared env?** — WiLoR's dependency footprint needs testing
   against the gvhmr conda environment. If conflicts exist, a separate `wilor` env
   adds install complexity but is safer.

2. **Face expressions?** — SMPL-X also supports jaw_pose and facial expressions.
   No estimator is wired up for this, but the SMPL-X model is already in place.
   Leave as zeros for now.

3. **Multi-person hand association** — When multiple people are tracked, how do we
   associate WiLoR's detected hands with the correct GVHMR body? Likely need
   spatial proximity matching (hand bounding box center near GVHMR wrist position).

---

## References

- WiLoR: https://github.com/rolpotamias/WiLoR (CVPR 2025)
- HaMeR: https://github.com/geopavlakos/hamer (CVPR 2024)
- Mano2Smpl-X reference impl: https://github.com/VincentHu19/Mano2Smpl-X
- SMPL-X library: https://github.com/vchoutas/smplx
- SMPL-X issue #124 (MANO compatibility confirmed): https://github.com/vchoutas/smplx/issues/124
- SMPL-X issue #222 (no wrist offset needed): https://github.com/vchoutas/smplx/issues/222
- HaMeR issue #26 (direct SMPL-X compatibility): https://github.com/geopavlakos/hamer/issues/26
