# VFX Pipeline

A FOSS automated VFX pipeline built on ComfyUI for first-pass rotoscoping, depth extraction, and clean plate generation.

## Goal

**Hands-off batch processing.** Ingest a movie file, get production-ready outputs (depth maps, segmentation masks, clean plates) with minimal manual intervention. Manual refinement happens downstream in Nuke/Fusion/Houdini—not here.

## Why This Exists

Traditional VFX prep work (roto, depth, clean plates) is tedious. Modern ML models (SAM3, Depth Anything V3, ProPainter) can automate 80% of it. This pipeline stitches them together into a single workflow that:

- Takes raw footage in
- Outputs VFX-ready passes out
- Follows real production folder conventions (frame numbering starts at 1001, etc.)

## Architecture

```
movie.mp4 → ingest.sh → project folder → ComfyUI workflows → VFX passes
```

**Core components:**
- **ComfyUI** - Node-based workflow engine (not for image generation here—just ML inference)
- **SAM3** - Segment Anything Model 3 for text-prompted rotoscoping
- **Depth Anything V3** - Monocular depth estimation with temporal consistency
- **ProPainter** - Video inpainting for clean plate generation
- **VideoHelperSuite** - Frame I/O handling

**Project structure** (per-shot):
```
projects/My_Shot_Name/
├── source/frames/    # Input frames (frame_1001.png, ...)
├── depth/            # Depth maps
├── roto/             # Segmentation masks
├── cleanplate/       # Inpainted plates
└── camera/           # Camera/geometry data
```

ComfyUI's output folder is symlinked to the active project.

## Current State

### Working
- `install_vfx_pipeline.sh` - Full installation (ComfyUI + nodes + models + conda env)
- `ingest.sh` - Movie → project folder + frame extraction
- `set_project.sh` - Switch between projects
- `01_analysis.json` - Depth Anything V3 workflow
- `02_segmentation.json` - SAM3 video segmentation (text prompt → masks)
- `03_cleanplate.json` - ProPainter inpainting (needs testing)

### Known Issues
- SAM3 text prompts like "person" don't capture carried items (bags, purses) - need multi-prompt approach
- Frame numbering in ComfyUI SaveImage doesn't support custom start numbers (1001+) - outputs need post-rename or custom node
- Large frame counts (150+) can stall SAM3 propagation - need batching strategy

### Not Yet Implemented
- Automated mask combination (when using multi-prompt segmentation)
- Depth-to-normals conversion for relighting
- Alembic camera export from DA3 data
- Batch processing wrapper (multiple shots)

## Design Decisions

1. **Frames, not video files** - All processing uses PNG sequences. More flexible, easier to debug, standard in VFX.

2. **1001 frame numbering** - Industry convention. Leaves room for handles/pre-roll.

3. **Project symlinks over path configs** - ComfyUI's SaveImage is path-limited. Symlinking `output/` to the active project is cleaner than fighting the node.

4. **Text prompts over manual selection** - For first-pass automation. Manual point-clicking is available but defeats the hands-off goal.

5. **Separate passes, not monolithic workflow** - Depth, roto, and clean plate as individual workflows. Easier to re-run one stage without redoing everything.

## For the Agent

When continuing development:

- **Test with real footage** - The workflows were built from node documentation, not extensive testing. Expect connection/type errors.
- **Check actual node signatures** - ComfyUI node inputs/outputs change between versions. Use `grep -A 20 "INPUT_TYPES" ...` to verify.
- **VRAM is the constraint** - 12GB minimum, 24GB comfortable. Batch sizes and model choices are driven by this.
- **User's system** - Ubuntu, NVIDIA GPU, conda environment at `/media/menser/fauna/META_VFX/VFX/`

The goal is a pipeline that a VFX artist can run overnight and come back to usable first-pass outputs.
