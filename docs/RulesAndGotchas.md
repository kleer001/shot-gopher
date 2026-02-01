# The Gopher's Rules

*Opinionated by design*

Shot-Gopher is built for VFX artists who need fast, consistent first-pass results. It achieves this by being **opinionated** and **destructive**. Understanding these design choices will save you frustration.

---

## The Three Laws

### 1. Movie In, Frames Out

**Input:** A single video file (`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`, `.mxf`)

**Not supported:**
- Image sequences as input
- Multiple video files in one run
- Mixing formats

The gopher expects you to hand it a movie. It extracts frames internally. If you already have frames, create a video from them first, or manually place them in `source/frames/` and skip the `ingest` stage.

### 2. Every Run Replaces Previous Output

**This is the most important rule to understand.**

When you run a stage, Shot-Gopher **deletes** the previous output for that stage and creates fresh results. There is no append, no versioning, no "keep both."

```bash
# First run - creates depth maps
python scripts/run_pipeline.py shot.mp4 -s depth

# Second run - DESTROYS previous depth maps, creates new ones
python scripts/run_pipeline.py shot.mp4 -s depth
```

**Why?** VFX iteration is messy. Artists tweak parameters, re-run, compare. Accumulating old outputs creates confusion about which version is current. The gopher keeps it simple: one stage, one output, always current.

**How to preserve previous attempts:**

```bash
# Option 1: Copy before re-running
cp -r ../vfx_projects/MyShot/depth ../vfx_projects/MyShot/depth_v1

# Option 2: Use --no-overwrite (keeps existing, skips stage if output exists)
python scripts/run_pipeline.py shot.mp4 -s depth --no-overwrite

# Option 3: Different project names for different versions
python scripts/run_pipeline.py shot.mp4 -s depth --name MyShot_v1
python scripts/run_pipeline.py shot.mp4 -s depth --name MyShot_v2
```

### 3. Trust the Project Structure

The gopher creates a specific directory layout:

```
../vfx_projects/MyShot/
├── source/frames/      # Extracted frames (ingest)
├── depth/              # Depth maps
├── roto/               # Roto masks
├── matte/              # Refined alpha mattes
├── cleanplate/         # Inpainted backgrounds
├── colmap/             # Camera reconstruction
├── camera/             # Exported camera data
└── mocap/              # Motion capture data
```

**Don't:**
- Manually add files to stage output directories (they'll be deleted on next run)
- Rename or restructure directories (stages expect specific paths)
- Mix outputs from different shots in one project

**Do:**
- Add custom data to new subdirectories (e.g., `../vfx_projects/MyShot/reference/`)
- Copy outputs elsewhere before modifying them
- Use the project as a **source** for your downstream work, not a workspace

---

## Common Gotchas

### "I gave it an image sequence and nothing happened"

Shot-Gopher expects a video file. If you have frames:

```bash
# Convert frames to video first
ffmpeg -framerate 24 -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p temp_input.mp4

# Then run the pipeline
python scripts/run_pipeline.py temp_input.mp4 -s depth,roto
```

Or skip ingest and work with existing frames:

```bash
# Manually place frames in the project
mkdir -p ../vfx_projects/MyShot/source/frames
cp /path/to/frames/*.png ../vfx_projects/MyShot/source/frames/

# Run stages that don't need ingest
python scripts/run_pipeline.py --name MyShot -s depth,roto
```

### "My previous roto is gone!"

You re-ran the roto stage. By design, this replaces the previous output.

**Prevention:** Copy outputs you want to keep before re-running:

```bash
cp -r ../vfx_projects/MyShot/roto ../vfx_projects/MyShot/roto_backup
```

### "I want to compare two different settings"

Use separate project names:

```bash
python scripts/run_pipeline.py shot.mp4 -s roto --prompt "person" --name Shot_PersonOnly
python scripts/run_pipeline.py shot.mp4 -s roto --prompt "person,car" --name Shot_PersonAndCar
```

### "The stage didn't run because output already exists"

If you used `--no-overwrite` or `-e` (skip-existing), the stage was skipped.

To force re-run:

```bash
# Remove --no-overwrite flag (default behavior is to overwrite)
python scripts/run_pipeline.py shot.mp4 -s depth
```

### "I added files to the output folder and they disappeared"

Stage output directories are cleared on each run. Store custom files elsewhere:

```bash
# Bad - will be deleted
../vfx_projects/MyShot/roto/my_custom_mask.png

# Good - safe location
../vfx_projects/MyShot/custom/my_custom_mask.png
```

### "First run of a stage is slow"

First run downloads ML models. This is normal:
- SAM3: ~3.2 GB
- Depth Anything: ~1.5 GB
- VideoMaMa: ~10 GB
- GVHMR: ~4 GB

Subsequent runs use cached models.

### "Stage X requires stage Y to run first"

Some stages have dependencies:

| Stage | Requires |
|-------|----------|
| `depth` | `ingest` (frames must exist) |
| `roto` | `ingest` |
| `mama` | `roto` (needs masks to refine) |
| `cleanplate` | `roto` (needs masks for inpainting) |
| `camera` | `colmap` (needs reconstruction data) |
| `mocap` | `colmap` (needs camera data) |
| `gsir` | `colmap` (needs reconstruction data) |

Run stages in order, or use `all`:

```bash
python scripts/run_pipeline.py shot.mp4 -s ingest,roto,mama,cleanplate
```

---

## Philosophy: Why Destructive?

VFX pipelines often accumulate cruft:
- `roto_v1/`, `roto_v2_final/`, `roto_v2_final_FINAL/`
- Orphaned files from abandoned experiments
- Confusion about which output is "current"

Shot-Gopher takes a different approach: **the output is always the current version**. If you need history, that's what version control and manual backups are for.

This isn't the right approach for everyone. If you need non-destructive workflows:
- Use `--no-overwrite` to prevent accidental deletion
- Create versioned project names (`Shot_v1`, `Shot_v2`)
- Copy outputs before re-running stages
- Integrate with your studio's asset management system

The gopher is a tool, not a prison. Work with it, not against it.

---

## Quick Reference

| Expectation | Reality |
|-------------|---------|
| "I'll give it my EXR sequence" | Give it a movie file |
| "It will keep my old outputs" | It replaces them |
| "I can organize files my way" | Use the gopher's structure |
| "Stages run in any order" | Dependencies must be satisfied |
| "First run is instant" | First run downloads models |

---

**Version:** 1.0
**Last Updated:** 2026-02-01
