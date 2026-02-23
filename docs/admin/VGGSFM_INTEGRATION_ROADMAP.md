# VGGSfM v2 Integration Roadmap

Add VGGSfM v2 as an alternative SfM engine for the `matchmove_camera` stage.
COLMAP remains the default; VGGSfM is selected via `--mmcam-engine vggsfm`.

## Rationale

COLMAP fails on slow handheld tracking shots with large dynamic foreground
(person walking toward camera, 30%+ of frame). Even with masks and `--quality slow`,
narrow baseline + forward/backward translation produces degenerate solves.
Registration drops to 40-70% of frames, often splitting into disconnected sub-models.

VGGSfM v2 (Meta/Oxford, CVPR 2024, ranked 1st in IMC Challenge) addresses this:
- **Video-native** — designed for sequential frames with sliding window for 1000+ frames
- **Learned features** — robust on narrow-baseline, texture-poor scenes where SIFT fails
- **Shared intrinsics** — enforces consistent camera model across a sequence
- **COLMAP-compatible output** — `cameras.bin`, `images.bin`, `points3D.bin` directly
- **Mask support** — accepts binary masks (SAM2-compatible, same as roto pipeline)

**License:** CC BY-NC 4.0 (non-commercial). Fine for this project.

### Known Limitations (accepted tradeoffs)

- **Dynamic scenes still benefit from masks** — VGGSfM can "generally handle" small
  dynamic objects, but for 30%+ foreground occupancy (our target scenario) it
  recommends masking. Use existing roto pipeline masks.
- **No distortion estimation** — PINHOLE only. COLMAP remains better for distorted lenses.
- **CLI-based interface** — invoked via `demo.py` / `video_demo.py` with Hydra config,
  not a clean Python import. Runner wraps this as subprocess.

## Decisions

| Question | Answer |
|----------|--------|
| CLI flag | `--mmcam-engine vggsfm` (default: `colmap`) |
| Output dir | Same `mmcam/sparse/0/` (COLMAP format) |
| Installation | `pip install` into dedicated conda env `vggsfm` (PyTorch + CUDA) |
| Sandbox location | `.vfx_pipeline/tools/vggsfm/` |
| Model weights | Auto-download from HuggingFace on first run (~200MB) |
| Mask support | Recommended for shots with people (use existing roto pipeline masks) |
| Long sequences | `video_demo.py` sliding window handles 1000+ frames natively |
| Downstream impact | Zero — all consumers read COLMAP format from `camera/` and `mmcam/` |

## Architecture

```
run_pipeline.py  --mmcam-engine vggsfm
       │
       ▼
stage_runners.py::run_stage_matchmove_camera()
       │
       ├── engine == "colmap"  → run_matchmove_camera()  [existing, subprocess]
       │
       └── engine == "vggsfm" → run_vggsfm()             [NEW, subprocess]
                                       │
                                       ▼
                              scripts/run_vggsfm.py
                                       │
                           ┌───────────┴───────────┐
                           │                       │
                      ≤400 frames             >400 frames
                      (demo.py)            (video_demo.py)
                           │               sliding window
                           └───────┬───────┘
                                   │
                                   ▼
                          mmcam/sparse/0/
                          (cameras.bin, images.bin, points3D.bin)
                                   │
                                   ▼
                      export_colmap_to_pipeline_format()  [existing]
                                   │
                                   ▼
                          camera/extrinsics.json
                          camera/intrinsics.json
                                   │
                     ┌─────────────┼─────────────┐
                     ▼             ▼             ▼
                 mocap         export_camera     gsir
               (unchanged)    (unchanged)     (unchanged)
```

## Phase 1: Installation & Environment

**Goal:** VGGSfM v2 installs and runs inference on a test scene.

| Task | File | Details |
|------|------|---------|
| Create installer | `scripts/install_wizard/installers.py` | Add `VGGSfMInstaller`: conda env `vggsfm`, pip deps, clone repo to `.vfx_pipeline/tools/vggsfm/`, verify model weights download |
| Add to install wizard | `scripts/install_wizard/wizard.py` | Register VGGSfM as optional component |
| Add platform detection | `scripts/install_wizard/platform.py` | `find_vggsfm()` — check conda env `vggsfm`, check `.vfx_pipeline/tools/vggsfm/`, verify HuggingFace weights cached |
| Environment diagnostics | `scripts/run_vggsfm.py` | `diagnose_vggsfm_environment()` — CUDA, VRAM, model weights, PyTorch version |
| Manual test | — | Run VGGSfM on 20-frame test scene, verify `sparse/` output |

**Dependencies:**
- Python 3.10+, PyTorch 2.1+, CUDA 12.1+
- `lightglue`, `pycolmap`, `poselib`
- ~200MB model weights (auto-download from HuggingFace)
- 8GB+ VRAM recommended (32GB for full resolution)

## Phase 2: Core Pipeline

**Goal:** `run_vggsfm_pipeline()` produces valid COLMAP output from `source/frames/`.

| Task | File | Details |
|------|------|---------|
| Create VGGSfM runner | `scripts/run_vggsfm.py` | `run_vggsfm_pipeline(project_dir, max_image_size, max_gap) -> bool` |
| Frame loading | `scripts/run_vggsfm.py` | Read frames from `source/frames/`, optional downscaling |
| Inference dispatch | `scripts/run_vggsfm.py` | Use `demo.py` for ≤400 frames, `video_demo.py` for longer sequences |
| Mask preparation | `scripts/run_vggsfm.py` | Convert roto/matte masks to VGGSfM format if present |
| Output writing | `scripts/run_vggsfm.py` | VGGSfM writes COLMAP binary to `mmcam/sparse/0/` directly |
| Camera export | `scripts/run_vggsfm.py` | Call existing `export_colmap_to_pipeline_format()` with gap interpolation |
| Standalone CLI | `scripts/run_vggsfm.py` | `python run_vggsfm.py <project_dir> [--max-size N]` |
| Unit tests | `tests/test_run_vggsfm.py` | Output format validation, export compatibility |

**VGGSfM invocation** (subprocess, Hydra config):
```bash
# Standard (≤400 frames)
python demo.py \
    SCENE_DIR=/path/to/scene \
    query_frame_num=3 \
    max_query_pts=2048 \
    shared_camera=True

# Long sequences (>400 frames, sliding window)
python video_demo.py \
    SCENE_DIR=/path/to/scene \
    query_frame_num=3 \
    shared_camera=True
```

Output lands in `SCENE_DIR/sparse/` as `cameras.bin`, `images.bin`, `points3D.bin`.

**Key parameters to expose:**
- `query_frame_num` — number of initial seed frames (default 3)
- `max_query_pts` — max points for triangulation (default 2048, tune for VRAM)
- `shared_camera` — enforce single camera model across sequence (default True)

## Phase 3: Pipeline Integration

**Goal:** `--mmcam-engine vggsfm` works end-to-end through `run_pipeline.py`.

| Task | File | Details |
|------|------|---------|
| Add engine config | `scripts/pipeline_config.py` | `mmcam_engine: str = "colmap"` field |
| Add CLI argument | `scripts/run_pipeline.py` | `--mmcam-engine {colmap,vggsfm}` |
| Add subprocess wrapper | `scripts/stage_runners.py` | `run_vggsfm(project_dir, max_image_size) -> bool` — subprocess call to `run_vggsfm.py` |
| Add dispatcher | `scripts/stage_runners.py` | Branch on `config.mmcam_engine` in `run_stage_matchmove_camera()` |
| Availability check | `scripts/stage_runners.py` | Fail early with install instructions if VGGSfM not found |
| Integration tests | `tests/integration/` | Full pipeline: ingest -> matchmove_camera (vggsfm) -> export_camera |

**Dispatcher logic in stage_runners.py:**
```python
def run_stage_matchmove_camera(
    ctx: "StageContext",
    config: "PipelineConfig",
) -> bool:
    print("\n=== Stage: matchmove_camera ===")
    mmcam_sparse = ctx.project_dir / "mmcam" / "sparse" / "0"

    if ctx.skip_existing and mmcam_sparse.exists():
        print("  -> Skipping (sparse model exists)")
    else:
        if config.mmcam_engine == "vggsfm":
            if not run_vggsfm(
                ctx.project_dir,
                max_image_size=config.mmcam_max_size,
            ):
                print("  -> VGGSfM reconstruction failed", file=sys.stderr)
        else:
            if not run_matchmove_camera(
                ctx.project_dir,
                quality=config.mmcam_quality,
                use_masks=config.mmcam_use_masks,
                max_image_size=config.mmcam_max_size,
            ):
                print("  -> COLMAP reconstruction failed", file=sys.stderr)

    clear_gpu_memory(ctx.comfyui_url)
    run_stage_camera(ctx, config)
    return True
```

## Phase 4: Validation & Hardening

**Goal:** VGGSfM produces reliable results across shot types.

| Task | Details |
|------|---------|
| Benchmark vs COLMAP | Run both engines on 5+ reference shots, compare registration count and reprojection error |
| Walking shot validation | Test on `hacompcc_1001-1233` and similar handheld tracking shots |
| Multi-person test | Test on `library_6334478` (16 people) and `TNIS0025` (4 people) |
| GPU memory profiling | Document VRAM usage at various resolutions and frame counts |
| Mask impact measurement | Compare with and without roto masks on shots with large foreground people |
| Downstream verify | Confirm mocap, gsir, dense, export_camera all work with VGGSfM output |
| Error messages | Clear install instructions, GPU memory guidance, fallback suggestion to COLMAP |
| Docs update | Update `docs/reference/stages.md` matchmove_camera section |

## Files Summary

### New Files
| File | Purpose |
|------|---------|
| `scripts/run_vggsfm.py` | VGGSfM pipeline runner (inference, COLMAP export, CLI) |
| `tests/test_run_vggsfm.py` | Unit tests |

### Modified Files
| File | Change |
|------|--------|
| `scripts/pipeline_config.py` | Add `mmcam_engine` field |
| `scripts/run_pipeline.py` | Add `--mmcam-engine` CLI arg |
| `scripts/stage_runners.py` | Add `run_vggsfm()` wrapper, add engine dispatcher |
| `scripts/install_wizard/installers.py` | Add `VGGSfMInstaller` |
| `scripts/install_wizard/wizard.py` | Register VGGSfM as optional component |
| `scripts/install_wizard/platform.py` | Add `find_vggsfm()` detection |
| `docs/reference/stages.md` | Document engine selection |

### Unchanged Files
| File | Why |
|------|-----|
| `scripts/run_matchmove_camera.py` | COLMAP path unchanged |
| `scripts/export_camera.py` | Reads `camera/` dir — format unchanged |
| `scripts/run_mocap.py` | Reads `camera/intrinsics.json` — unchanged |
| `scripts/run_gsir.py` | Reads `mmcam/sparse/0/` — COLMAP format unchanged |

## Risks

| Risk | Mitigation |
|------|------------|
| Model weights download blocked (corporate firewall) | Support manual download path, document HuggingFace mirror URLs |
| VRAM too high for user GPUs | Expose `max_query_pts` and `max_image_size` downscaling, document VRAM requirements per resolution |
| No distortion estimation (PINHOLE only) | Document as known limitation, recommend COLMAP for heavily distorted lenses |
| Reconstruction quality regression on orbital/high-parallax shots | COLMAP remains default — VGGSfM is opt-in |
| VGGSfM output subtly differs from COLMAP binary format | Validate with `pycolmap` read/write roundtrip in tests |
| Dynamic scenes with 30%+ foreground still need masks | Use existing roto pipeline masks, document in user-facing docs |

## Out of Scope

- Replacing COLMAP as default (keep `colmap` default until battle-tested)
- Auto-detection of which engine to use per shot
- Dense reconstruction via VGGSfM (use existing COLMAP dense path)
- VGGT integration (successor from same team, CVPR 2025 — potential future upgrade)
- GLOMAP or FastMap integration (separate roadmap if needed)

## References

- [VGGSfM GitHub](https://github.com/facebookresearch/vggsfm)
- [VGGSfM Paper (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_VGGSfM_Visual_Geometry_Grounded_Deep_Structure_From_Motion_CVPR_2024_paper.pdf)
- [VGGT (successor, future consideration)](https://github.com/facebookresearch/vggt)
