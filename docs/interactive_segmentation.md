# Interactive Segmentation Guide

For complex shots where automatic text-prompt segmentation doesn't work well (e.g., segmenting individual legs, specific body parts, or objects that merge together), use the interactive segmentation workflow.

## When to Use This

- Multiple similar objects that merge in automatic segmentation (e.g., two legs touching)
- Body parts that need separate masks (left leg, right leg, each arm)
- Objects partially occluded or overlapping
- Fine control over what gets included/excluded in a mask

## Prerequisites

1. **ComfyUI running** with the TBG-SAM3 extension installed
2. **Project already set up** with source frames (run `setup_project.py` first)

### Installing TBG-SAM3

If not already installed, add the extension to ComfyUI:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/Ltamann/ComfyUI-TBG-SAM3
pip install -r ComfyUI-TBG-SAM3/requirements.txt
```

Then restart ComfyUI.

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

The TBG-SAM3 extension isn't installed. See [Installing TBG-SAM3](#installing-tbg-sam3) above.

### Masks are merging together

Add negative points between objects to force separation. SAM uses visual boundaries, so similar colors/textures may merge without guidance.

### Mask drifts off the object mid-video

This can happen with fast motion or heavy occlusion. Solutions:
1. Add more positive points on the object (not just one)
2. Use the workflow on a shorter clip
3. For extreme cases, split the video and run separately

### Points aren't showing up

Make sure you're clicking in the **Interactive Selector** node's image preview area, not elsewhere in the UI.

## Comparison with Automatic Segmentation

| Feature | Automatic (`02_segmentation.json`) | Interactive (`05_interactive_segmentation.json`) |
|---------|-----------------------------------|------------------------------------------------|
| Input | Text prompt ("person", "car") | Click points on image |
| Control | Low - segments all matching objects | High - you choose exactly what |
| Speed | Fast, no interaction needed | Requires manual selection |
| Best for | Single dominant object, quick passes | Multiple similar objects, precision work |
| Output | `roto/[prompt_name]/` | `roto/custom/` |

## Command Reference

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
