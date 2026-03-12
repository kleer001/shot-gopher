# GVHMR / COLMAP SfM Deprecation Roadmap

Replace GVHMR with SLAHMR as the default motion capture engine, remove COLMAP as an SfM
choice (retain for dense MVS), and deprecate `camera_alignment.py`.

## Rationale

- SLAHMR jointly optimizes camera + body, producing smooth cameras (1.3mm/frame jitter)
  and physically plausible motion via the HuMoR prior
- Eliminates the need for separate camera alignment (`camera_alignment.py`)
- COLMAP SfM replaced by VGGSfM for all sparse reconstruction use cases
- SMPL-H → SMPLX conversion validated at <5cm mean joint error

## Current State (v-current)

| Component | Status |
|-----------|--------|
| SLAHMR | **Default** mocap engine (`--mocap-engine=slahmr`) |
| GVHMR | **Deprecated** — available via `--mocap-engine=gvhmr`, prints warning |
| COLMAP SfM | **Removed** from `--mmcam-engine` choices |
| COLMAP MVS | **Retained** for `dense` stage (point clouds, mesh, depth/normals) |
| `camera_alignment.py` | **Obsolete** — SLAHMR camera is joint-optimized, no alignment needed |
| VGGSfM | **Default** (and only) SfM engine |

## Timeline

### v-next: GVHMR not installed by default

- Remove `gvhmr` from default "Full stack" install selection
- Keep `gvhmr` component in wizard for manual selection
- SLAHMR is the only recommended mocap path
- `camera_alignment.py` no longer imported anywhere

### v-remove: GVHMR code removed

- Remove `gvhmr` component from install wizard
- Remove `GVHMRInstaller` from `installers.py`
- Remove `--mocap-engine` flag (SLAHMR only)
- Remove `run_mocap.py` (GVHMR wrapper)
- Remove `camera_alignment.py`
- Remove GVHMR-specific code from `export_mocap.py` (`convert_gvhmr_to_motion`)
- Remove COLMAP SfM path from `run_stage_matchmove_camera` (already VGGSfM-only)
- Clean up `run_matchmove_camera.py` to be COLMAP-MVS-only utility

## Files Changed (v-current)

| File | Action | Change |
|------|--------|--------|
| `scripts/run_slahmr.py` | **Created** | SLAHMR wrapper |
| `scripts/convert_smplh_to_smplx.py` | **Created** | SMPL-H → SMPLX betas fitting + pose transfer |
| `scripts/env_config.py` | Modified | Added `SLAHMR_INSTALL_DIR`, `SLAHMR_CONDA_ENV` |
| `scripts/pipeline_config.py` | Modified | Added `mocap_engine` field (default: `"slahmr"`) |
| `scripts/pipeline_constants.py` | Modified | Updated stage descriptions |
| `scripts/run_pipeline.py` | Modified | Removed `colmap` from `--mmcam-engine`, added `--mocap-engine` |
| `scripts/stage_runners.py` | Modified | Mocap dispatch by engine, GVHMR deprecation warning |
| `scripts/export_mocap.py` | Modified | Added `convert_slahmr_to_motion()` |
| `scripts/install_wizard/installers.py` | Modified | Added `SLAHMRInstaller` |
| `scripts/install_wizard/wizard.py` | Modified | Added `slahmr` component |
| `docs/admin/deprecation_roadmap.md` | **Created** | This document |
