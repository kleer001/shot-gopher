---
name: comfyui-vfx
description: ComfyUI workflow development for VFX pipelines. Use when editing workflow JSON files, debugging ComfyUI API calls, or understanding node connections for depth, segmentation, inpainting, and matting workflows.
---

# ComfyUI VFX Pipeline Skill

ComfyUI is a node-based UI for Stable Diffusion and ML inference. This skill covers workflow JSON structure, the REST API, and the specific nodes used in this VFX pipeline (SAM3, Video-Depth-Anything, ProPainter).

## Quick Reference

- **Server**: `http://127.0.0.1:8188`
- **Submit workflow**: `POST /prompt` with `{"prompt": <workflow>, "client_id": "<id>"}`
- **Check status**: `GET /history/<prompt_id>`
- **Workflow files**: `workflow_templates/*.json`

## Workflow JSON Structure

```json
{
  "id": "uuid",
  "nodes": [
    {
      "id": 1,
      "type": "NodeType",
      "pos": [x, y],
      "inputs": [{"name": "input_name", "type": "TYPE", "link": 1}],
      "outputs": [{"name": "output_name", "type": "TYPE", "links": [2], "slot_index": 0}],
      "widgets_values": ["value1", 123, true]
    }
  ],
  "links": [
    [link_id, from_node_id, from_slot, to_node_id, to_slot, "TYPE"]
  ]
}
```

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/prompt` | POST | Queue workflow execution |
| `/history/{id}` | GET | Get results for prompt_id |
| `/queue` | GET | Current queue status |
| `/interrupt` | POST | Stop current execution |
| `/view` | GET | Retrieve output images |
| `/object_info` | GET | All available node types |

See [reference/api.md](reference/api.md) for details.

## Node Packages Used

### VideoHelperSuite (VHS)
Frame sequence I/O for video workflows.
- `VHS_LoadImagesPath` - Load image sequence from directory
- `VHS_SelectEveryNthImage` - Sample/select frames

### Video-Depth-Anything
Temporally consistent depth estimation.
- `LoadVideoDepthAnythingModel` - Load depth model
- `VideoDepthAnythingProcess` - Run depth inference
- `VideoDepthAnythingOutput` - Extract depth maps

### SAM3 (Segment Anything Model 3)
Video object segmentation with text/point prompts.
- `LoadSAM3Model` - Load SAM3 checkpoint
- `SAM3VideoSegmentation` - Initialize segmentation (text or point mode)
- `SAM3Propagate` - Propagate masks through video
- `SAM3VideoOutput` - Extract final masks
- `SAM3PointCollector` - Interactive point selection UI

### ProPainter
Video inpainting for object removal.
- `ProPainterInpaint` - Remove masked regions with temporal consistency

### Built-in Nodes
- `SaveImage` / `PreviewImage` - Output handling
- `MaskToImage` / `ImageToMask` - Type conversion
- `ImageScale` - Resize images
- `Note` - Workflow documentation

See [reference/nodes.md](reference/nodes.md) for parameters.

## Common Patterns

### Loading Frame Sequences
```json
{
  "type": "VHS_LoadImagesPath",
  "widgets_values": ["source/frames", 0, 0, 1, "Disabled", 1920, 1080, null, null]
}
```
Parameters: `[path, skip_first, select_every_nth, batch_size, ...]`

### Text-Prompt Segmentation
```json
{
  "type": "SAM3VideoSegmentation",
  "widgets_values": ["text", "person", 0, 0.3]
}
```
Parameters: `[mode, prompt, frame_idx, threshold]` (frame_idx must be >= 0)

### Point-Prompt Segmentation
```json
{
  "type": "SAM3VideoSegmentation",
  "widgets_values": ["point", "", 0, 0.3]
}
```
Connect `SAM3PointCollector` outputs to positive/negative point inputs.

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| CUDA OOM | VRAM exceeded | Reduce batch size, downscale images |
| Empty masks | Bad prompt/threshold | Adjust threshold, try different prompt |
| Server 500 | Missing model | Check model paths, run install |
| Workflow fails silently | Type mismatch | Check link types match input/output |
