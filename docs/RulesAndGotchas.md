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

The gopher expects you to hand it a movie. It extracts frames internally. If you already have an image sequence, convert it to a video first (your compositor or DaVinci can do this), then feed that to the gopher.

### 2. Every Run Replaces Previous Output

**This is the most important rule to understand.**

When you run a stage, Shot-Gopher **deletes** the previous output for that stage and creates fresh results. There is no append, no versioning, no "keep both."

Run depth once, you get depth maps. Run depth again with different settings, the old depth maps are **gone** and replaced with new ones.

**Why?** VFX iteration is messy. Artists tweak parameters, re-run, compare. Accumulating old outputs creates confusion about which version is current. The gopher keeps it simple: one stage, one output, always current.

**Want to preserve previous attempts?**

- **Copy before re-running:** Use your file browser to copy the output folder before hitting run again
- **Use different project names:** Create `Shot_v1`, `Shot_v2`, etc. as separate projects

### 3. Trust the Project Structure

The gopher creates a specific directory layout for each project:

```
MyShot/
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
- Manually add files to stage output folders (they'll be deleted on next run)
- Rename or restructure folders (stages expect specific paths)
- Mix outputs from different shots in one project

**Do:**
- Add custom data to new subfolders (e.g., `MyShot/reference/`)
- Copy outputs elsewhere before modifying them
- Use the project as a **source** for your downstream work, not a workspace

---

## Common Gotchas

### "I gave it an image sequence and nothing happened"

Shot-Gopher expects a video file. Convert your frames to a video first using your preferred tool (After Effects, Nuke, DaVinci, ffmpeg).

### "My previous roto is gone!"

You re-ran the roto stage. By design, this replaces the previous output.

**Prevention:** Copy the output folder you want to keep before re-running the stage.

### "I want to compare two different settings"

Create separate projects with different names. Each project maintains its own independent outputs.

### "I added files to the output folder and they disappeared"

Stage output folders are cleared on each run. Store custom files in a separate folder within the project.

### "First run of a stage is slow"

First run downloads ML models. This is normal and only happens once:
- Depth models: ~1.5 GB
- Roto models: ~3 GB
- Matte refinement: ~10 GB
- Motion capture: ~4 GB

Subsequent runs use cached models.

### "Stage X says it needs something from stage Y"

Some stages have dependencies:

| Stage | Requires |
|-------|----------|
| Depth | Ingest (frames must exist) |
| Roto | Ingest |
| Matte | Roto (needs masks to refine) |
| Cleanplate | Roto (needs masks for inpainting) |
| Camera | Colmap (needs reconstruction data) |
| Mocap | Colmap (needs camera data) |

Run stages in order from top to bottom.

---

## Philosophy: Why Destructive?

VFX pipelines often accumulate cruft:
- `roto_v1/`, `roto_v2_final/`, `roto_v2_final_FINAL/`
- Orphaned files from abandoned experiments
- Confusion about which output is "current"

Shot-Gopher takes a different approach: **the output is always the current version**. If you need history, that's what version control and manual backups are for.

This isn't the right approach for everyone. If you need non-destructive workflows:
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

*For command-line usage, see the [CLI Reference](reference/cli.md).*
