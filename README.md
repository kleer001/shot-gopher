# VFX Pipeline

A FOSS automated VFX pipeline built on ComfyUI for first-pass rotoscoping, depth extraction, and clean plate generation.

## Goal

**Hands-off batch processing.** Ingest a movie file, get production-ready outputs (depth maps, segmentation masks, clean plates) with minimal manual intervention. Manual refinement happens downstream in Nuke/Fusion/Houdini—not here.

## Why This Exists

Traditional VFX prep work (roto, depth, clean plates) is tedious. Modern ML models (SAM3, Depth Anything V3, ProPainter) can automate 80% of it. This pipeline stitches them together into a single workflow that:

- Takes raw footage in
- Outputs VFX-ready passes out
- Follows real production folder conventions (frame numbering starts at 1001, etc.)

## Quick Start

```bash
# Single command - processes everything
python scripts/run_pipeline.py /path/to/footage.mp4 --name "My_Shot"

# Or run specific stages
python scripts/run_pipeline.py footage.mp4 --stages depth,camera
```

Requires ComfyUI running: `cd /path/to/ComfyUI && python main.py --listen`

## Architecture

```
movie.mp4 → run_pipeline.py → project folder → ComfyUI API → VFX passes
                   │
                   ├── Extract frames (ffmpeg)
                   ├── Setup project structure
                   ├── Populate workflow templates with project paths
                   ├── Queue workflows via ComfyUI API
                   └── Post-process (camera export)
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
├── camera/           # Camera/geometry data (.abc, .json)
└── workflows/        # Project-specific workflows (with absolute paths)
```

## Current State

### Working
- `scripts/run_pipeline.py` - **Main entry point** - automated pipeline runner
- `scripts/setup_project.py` - Project setup with workflow templating
- `scripts/export_camera.py` - Camera data → Alembic/JSON export
- `workflow_templates/01_analysis.json` - Depth Anything V3 + camera estimation

### Needs Testing
- `02_segmentation.json` - SAM3 video segmentation (text prompt → masks)
- `03_cleanplate.json` - ProPainter inpainting

### Known Issues
- SAM3 text prompts like "person" don't capture carried items (bags, purses) - need multi-prompt approach
- Frame numbering in ComfyUI SaveImage doesn't support custom start numbers (1001+) - outputs need post-rename or custom node
- Large frame counts (150+) can stall SAM3 propagation - need batching strategy

### Not Yet Implemented
- Automated mask combination (when using multi-prompt segmentation)
- Depth-to-normals conversion for relighting
- Batch processing wrapper (multiple shots in parallel)

## Design Decisions

1. **Frames, not video files** - All processing uses PNG sequences. More flexible, easier to debug, standard in VFX.

2. **1001 frame numbering** - Industry convention. Leaves room for handles/pre-roll.

3. **Templated workflows per project** - Each project gets its own copy of workflow JSONs with absolute paths populated. Avoids symlink management and makes projects self-contained.

4. **Text prompts over manual selection** - For first-pass automation. Manual point-clicking is available but defeats the hands-off goal.

5. **Separate passes, not monolithic workflow** - Depth, roto, and clean plate as individual workflows. Easier to re-run one stage without redoing everything.

## For the Agent

When continuing development:

- **Test with real footage** - The workflows were built from node documentation, not extensive testing. Expect connection/type errors.
- **Check actual node signatures** - ComfyUI node inputs/outputs change between versions. Use `grep -A 20 "INPUT_TYPES" ...` to verify.
- **VRAM is the constraint** - 12GB minimum, 24GB comfortable. Batch sizes and model choices are driven by this.
- **User's system** - Ubuntu, NVIDIA GPU, conda environment at `/media/menser/fauna/META_VFX/VFX/`

The goal is a pipeline that a VFX artist can run overnight and come back to usable first-pass outputs.
