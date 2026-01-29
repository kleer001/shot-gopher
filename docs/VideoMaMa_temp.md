# VideoMaMa Integration (Prototype)

**Status**: Prototype / Standalone
**Added**: 2026-01-29

## Overview

VideoMaMa refines coarse segmentation masks (from SAM3) into high-quality alpha mattes using video diffusion priors. It's particularly good at handling fine details like hair and face edges that SAM3 struggles with.

Paper: [VideoMaMa: Mask-Guided Video Matting via Generative Prior](https://arxiv.org/abs/2601.14255)
Project: [cvlab-kaist/VideoMaMa](https://github.com/cvlab-kaist/VideoMaMa)

## Requirements

> **DISK SPACE: ~12GB**
> - Stable Video Diffusion base model: ~10GB
> - VideoMaMa checkpoint: ~1.5GB
> - Code + dependencies: ~500MB

> **VRAM: ~16-24GB** (depending on resolution)

## Installation

```bash
python scripts/video_mama_install.py
```

Check installation status:
```bash
python scripts/video_mama_install.py --check
```

Installation creates:
- Conda environment: `videomama`
- Tools: `.vfx_pipeline/tools/VideoMaMa/`
- Models: `.vfx_pipeline/models/VideoMaMa/`

## Usage (Standalone)

After running the `roto` stage to generate SAM3 masks:

```bash
python scripts/video_mama.py /path/to/project
```

**Input**: `<project>/roto/person/*.png` (binary masks from SAM3)
**Output**: `<project>/roto/person_mama/*.png` (refined alpha mattes)

### Options

```bash
python scripts/video_mama.py <project> --num-frames 25    # Frames per batch
python scripts/video_mama.py <project> --width 1280 --height 720  # Resolution
```

## TODO

- [ ] **Pipeline integration**: Add `-s roto_mama` stage to run_pipeline.py
- [ ] **Memory optimization**: Add resolution scaling options
- [ ] **Batch processing**: Handle long sequences efficiently
- [ ] **Quality comparison**: Benchmark against MatAnyone

## Technical Notes

VideoMaMa uses Stable Video Diffusion as a generative prior to convert binary masks into soft alpha mattes. Key advantages:

1. **Temporal consistency**: Leverages video diffusion for smooth frame-to-frame results
2. **Fine detail preservation**: Trained on 50K real-world videos (MA-V dataset)
3. **Zero-shot generalization**: Works on diverse footage without fine-tuning

The current prototype processes `roto/person` only. Future versions will support arbitrary mask directories.

## Troubleshooting

**CUDA out of memory**
- Reduce `--width` and `--height` (try 512x288)
- Reduce `--num-frames` (try 8)

**No output generated**
- Check that `roto/person/` has PNG files
- Verify `source/frames/` exists and has matching frame count

**Installation fails**
- Ensure conda/mamba is installed
- Check internet connection for HuggingFace downloads
- Verify ~12GB free disk space
