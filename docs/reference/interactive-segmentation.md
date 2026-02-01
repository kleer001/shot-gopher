# Interactive Roto Guide

For complex shots where automatic text-prompt roto doesn't work well, use the interactive roto workflow. This gives you precise control over exactly what gets segmented.

## When to Use This

- Multiple similar objects that merge in automatic roto
- Objects partially occluded or overlapping
- Anything text prompts can't reliably identify
- Fine control over what gets included/excluded in a mask

## Prerequisites

- VFX Pipeline installed (via `install_wizard.py`)
- Project set up with source frames (`setup_project.py` or ingest stage complete)
- ComfyUI installed via `install_wizard.py`
- SAM3 extension (can install via ComfyUI Manager in browser if missing)

## Quick Start

```bash
python scripts/launch_interactive_segmentation.py /path/to/your/project
```

This starts ComfyUI, loads the workflow, and opens your browser.

---

## Step-by-Step Workflow

### Step 1: Choose Your Reference Frame

In the **"🎬 Select Frame"** node (`VHS_SelectEveryNthImage`):

- Set **`skip_first_images`** to the frame number you want to annotate
- Frame 0 = first frame, Frame 50 = 50th frame, etc.
- Choose a frame where your target object is clearly visible and unoccluded

### Step 2: Run the Interactive Selector

1. Find the **"🎯 Interactive Selector"** node (`SAM3PointCollector`)
2. Click the **Run** button on just this node (not Queue Prompt yet)
3. Wait for the frame image to appear in the node's preview area

### Step 3: Click to Select Objects

Click directly on the image preview in the Interactive Selector node:

| Action | Result |
|--------|--------|
| **Left-click** | Positive point — include this area in mask |
| **Right-click** (or Shift+Left) | Negative point — exclude this area from mask |

**Clicking Tips:**

- **Start simple**: One click in the center of your object is often enough
- **Add more points** only if the initial mask is wrong
- **Use negative points** to carve out areas that shouldn't be included
- **Click on distinct features**: Choose areas with clear visual boundaries
- **Separate objects**: Each object you want tracked separately needs its own selection

### Step 4: Propagate Through Video

Once your points are set:

1. Click **Queue Prompt** to run the full workflow
2. SAM3 propagates your selection through all frames (forward and backward)
3. Masks save to `roto/custom/`

---

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

## Comparison with Automatic Roto

| Feature | Automatic (`02_segmentation.json`) | Interactive (`05_interactive_segmentation.json`) |
|---------|-----------------------------------|------------------------------------------------|
| Input | Text prompt ("person", "car") | Click points on image |
| Control | Low - segments all matching objects | High - you choose exactly what |
| Speed | Fast, no interaction needed | Requires manual selection |
| Best for | Single dominant object, quick passes | Multiple similar objects, precision work |
| Output | `roto/[prompt_name]/` | `roto/custom/` |

## Command Reference

```bash
# Launch interactive roto
python scripts/launch_interactive_segmentation.py <project_dir>

# Custom ComfyUI URL
python scripts/launch_interactive_segmentation.py <project_dir> --url http://localhost:9999
```
