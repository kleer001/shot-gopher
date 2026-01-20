# Interactive Segmentation Guide

For complex shots where automatic text-prompt segmentation doesn't work well (e.g., segmenting individual legs, specific body parts, or objects that merge together), use the interactive segmentation workflow.

## When to Use This

- Multiple similar objects that merge in automatic segmentation (e.g., two legs touching)
- Body parts that need separate masks (left leg, right leg, each arm)
- Objects partially occluded or overlapping
- Fine control over what gets included/excluded in a mask

## Prerequisites

- VFX Pipeline installed (via `install_wizard.py` or Docker)
- Project set up with source frames (`setup_project.py` or ingest stage complete)
- ComfyUI running

## Docker Usage

When running in Docker, the interactive segmentation workflow requires accessing ComfyUI's web interface from your host machine.

### Docker Quick Start

1. **Start the container with ComfyUI running:**
   ```bash
   docker compose up -d
   ```

2. **Set up the project and run initial stages:**
   ```bash
   docker compose exec vfx-ingest python /app/scripts/run_pipeline.py \
     /workspace/projects/MyProject --stages ingest
   ```

3. **Prepare the interactive segmentation workflow:**
   ```bash
   docker compose exec vfx-ingest python /app/scripts/launch_interactive_segmentation.py \
     /workspace/projects/MyProject
   ```

4. **Access ComfyUI from your host browser:**
   - Open http://localhost:8188
   - Click **Menu** > **Load**
   - Navigate to `/workspace/projects/MyProject/workflows/05_interactive_segmentation.json`

5. **Click points and run as described below**

### Docker Path Mapping

| Host Path | Container Path |
|-----------|----------------|
| `$VFX_PROJECTS_DIR/MyProject` | `/workspace/projects/MyProject` |
| (Read-only models) | `/models` |

The workflow file paths in ComfyUI will show container paths (e.g., `/workspace/projects/...`). This is expected and correct when running in Docker.

## Quick Start

### 1. Prepare the Workflow

```bash
python scripts/launch_interactive_segmentation.py /path/to/your/project
```

This copies the interactive workflow template to your project and populates it with the correct paths.

### 2. Open ComfyUI

Either add `--open` to automatically open your browser:

```bash
python scripts/launch_interactive_segmentation.py /path/to/your/project --open
```

Or manually navigate to `http://localhost:8188`

### 3. Load the Workflow

In ComfyUI:
1. Click **Menu** (top-left hamburger icon)
2. Click **Load**
3. Navigate to: `your_project/workflows/05_interactive_segmentation.json`

### 4. Select Objects

In the **Interactive Selector** node:

| Action | What it does |
|--------|--------------|
| **Left-click** | Add positive point (include in mask) |
| **Right-click** | Add negative point (exclude from mask) |
| **Different object IDs** | Each ID becomes a separate mask sequence |

### 5. Run the Workflow

Click **Queue Prompt** to execute. SAM3 will:
1. Segment objects based on your points on frame 0
2. Propagate masks through the entire video sequence
3. Save results to `roto/custom/`

## Segmenting Multiple Legs

For the running shot example with multiple people's legs:

1. **Click once on each leg** you want to track
2. **Use different object IDs** for each leg (the selector assigns these automatically)
3. **Add negative points** between legs if they tend to merge
4. **Add negative points** on shorts/clothing to exclude them from the leg mask

### Tips for Clean Leg Masks

- Click on the **middle of the calf** (most distinctive part)
- If a leg includes the shoe, add a **negative point on the shoe**
- If two adjacent legs merge, add a **negative point between them**
- If the sock merges with skin, that's usually fine for roto purposes

## Output Structure

```
your_project/
  roto/
    custom/
      mask/           # Combined masks (if multiple objects)
      mask_0001.png   # Or individual frame masks
      mask_0002.png
      ...
```

Each object ID you clicked will generate a separate mask sequence.

## Troubleshooting

### "Node not found" error in ComfyUI

The ComfyUI-SAM3 extension isn't installed or ComfyUI needs a restart. If you installed via `install_wizard.py`, re-run the wizard to ensure all custom nodes are installed. Then restart ComfyUI.

### Masks are merging together

Add negative points between objects to force separation. SAM uses visual boundaries, so similar colors/textures may merge without guidance.

### Mask drifts off the object mid-video

This can happen with fast motion or heavy occlusion. Solutions:
1. Add more positive points on the object (not just one)
2. Use the workflow on a shorter clip
3. For extreme cases, split the video and run separately

### Points aren't showing up

Make sure you're clicking in the **Interactive Selector** node's image preview area, not elsewhere in the UI.

### Docker: "Node not found" error

If running in Docker and seeing missing node errors, the Docker image may need rebuilding:
```bash
docker compose build --no-cache
```

### Docker: Cannot load workflow file

When using Docker, ensure you're loading the workflow from the container path:
- Container path: `/workspace/projects/MyProject/workflows/05_interactive_segmentation.json`
- NOT the host path

The workflow paths are relative to the container's filesystem.

### Docker: Masks not saving

Verify your project directory is correctly mounted. Check:
```bash
docker compose exec vfx-ingest ls -la /workspace/projects/MyProject/roto/custom/
```

## Comparison with Automatic Segmentation

| Feature | Automatic (`02_segmentation.json`) | Interactive (`05_interactive_segmentation.json`) |
|---------|-----------------------------------|------------------------------------------------|
| Input | Text prompt ("person", "car") | Click points on image |
| Control | Low - segments all matching objects | High - you choose exactly what |
| Speed | Fast, no interaction needed | Requires manual selection |
| Best for | Single dominant object, quick passes | Multiple similar objects, precision work |
| Output | `roto/[prompt_name]/` | `roto/custom/` |

## Command Reference

### Local Installation

```bash
# Basic usage
python scripts/launch_interactive_segmentation.py <project_dir>

# Open browser automatically
python scripts/launch_interactive_segmentation.py <project_dir> --open

# Custom ComfyUI URL
python scripts/launch_interactive_segmentation.py <project_dir> --url http://localhost:8188

# Force overwrite existing workflow
python scripts/launch_interactive_segmentation.py <project_dir> --force
```

### Docker

```bash
# Prepare workflow (use container path for project)
docker compose exec vfx-ingest python /app/scripts/launch_interactive_segmentation.py \
  /workspace/projects/MyProject

# Then open http://localhost:8188 in your host browser
```
