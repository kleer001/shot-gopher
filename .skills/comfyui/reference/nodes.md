# Node Reference for VFX Pipeline

Detailed parameters for nodes used in shot-gopher workflows.

---

## VideoHelperSuite (VHS)

### VHS_LoadImagesPath
Load image sequence from a directory.

| Input | Type | Description |
|-------|------|-------------|
| directory | STRING | Path to image folder |
| skip_first_images | INT | Skip N frames from start (default: 0) |
| select_every_nth | INT | Sample every Nth frame (default: 1) |
| batch_size | INT | Images per batch (default: 1) |
| image_load_cap | STRING | "Disabled" or max frames |
| resolution | INT, INT | Target width, height |

| Output | Type |
|--------|------|
| IMAGE | Batch of images |
| MASK | Associated masks (if present) |
| frame_count | INT |

**Example widgets_values:**
```json
["source/frames", 0, 0, 1, "Disabled", 1920, 1080, null, null]
```

### VHS_SelectEveryNthImage
Select specific frames from a batch.

| Input | Type | Description |
|-------|------|-------------|
| images | IMAGE | Input batch |
| skip_first_images | INT | Skip N frames (use as frame selector) |
| select_every_nth | INT | Sample rate |

| Output | Type |
|--------|------|
| IMAGE | Selected frames |
| count | INT |

---

## Video-Depth-Anything

### LoadVideoDepthAnythingModel
Load depth estimation model.

| Widget | Type | Options |
|--------|------|---------|
| model_name | STRING | `video_depth_anything_vits.pth` (small), `video_depth_anything_vitb.pth` (base), `video_depth_anything_vitl.pth` (large) |

| Output | Type |
|--------|------|
| VDAMODEL | Model reference |

### VideoDepthAnythingProcess
Run depth estimation on frames.

| Input | Type |
|-------|------|
| vda_model | VDAMODEL |
| images | IMAGE |

| Widget | Default | Description |
|--------|---------|-------------|
| max_resolution | 518 | Processing resolution |
| target_fps | 512 | Target framerate |
| dtype | "fp16" | Precision (fp16/fp32) |

| Output | Type |
|--------|------|
| DEPTHS | Depth tensor |

### VideoDepthAnythingOutput
Convert depth tensor to images.

| Input | Type |
|-------|------|
| depths | DEPTHS |

| Widget | Options |
|--------|---------|
| colormap | "gray", "viridis", "plasma", "inferno", "magma" |

| Output | Type |
|--------|------|
| IMAGE | Depth visualization |

---

## SAM3 (Segment Anything Model 3)

### LoadSAM3Model
Load SAM3 checkpoint (~3.2GB).

| Widget | Options |
|--------|---------|
| model_name | `sam3.pt` |

| Output | Type |
|--------|------|
| SAM3_MODEL | Model reference |

### SAM3VideoSegmentation
Initialize video segmentation state.

| Input | Type | Required |
|-------|------|----------|
| video_frames | IMAGE | Yes |
| positive_points | SAM3_POINTS_PROMPT | For point mode |
| negative_points | SAM3_POINTS_PROMPT | For point mode |

| Widget | Type | Description |
|--------|------|-------------|
| mode | STRING | `"text"` or `"point"` |
| text_prompt | STRING | Object description (text mode) |
| frame_idx | INT | Reference frame (0 = first frame) |
| threshold | FLOAT | Detection confidence (0.0-1.0, default: 0.3) |

| Output | Type |
|--------|------|
| SAM3_VIDEO_STATE | Segmentation state |

**Text mode example:**
```json
["text", "person", 0, 0.3]
```

**Point mode example:**
```json
["point", "", 0, 0.3]
```

### SAM3Propagate
Propagate masks through video frames.

| Input | Type |
|-------|------|
| sam3_model | SAM3_MODEL |
| video_state | SAM3_VIDEO_STATE |

| Widget | Type | Description |
|--------|------|-------------|
| start_frame | INT | Begin propagation (default: 0) |
| end_frame | INT | End propagation (-1 = all) |
| direction | STRING | `"forward"`, `"backward"`, `"bidirectional"` |
| reset_state | BOOL | Clear previous masks |

| Output | Type |
|--------|------|
| SAM3_VIDEO_MASKS | Mask tensor |
| SAM3_VIDEO_SCORES | Confidence scores |
| SAM3_VIDEO_STATE | Updated state |

### SAM3VideoOutput
Extract masks from propagation result.

| Input | Type |
|-------|------|
| masks | SAM3_VIDEO_MASKS |
| video_state | SAM3_VIDEO_STATE |

| Widget | Type | Description |
|--------|------|-------------|
| object_id | INT | Which object (-1 = all) |
| binary_mask | BOOL | Threshold to 0/1 |

| Output | Type |
|--------|------|
| MASK | Mask sequence |
| IMAGE | Original frames |
| IMAGE | Overlay visualization |

### SAM3PointCollector
Interactive UI for selecting points.

| Input | Type |
|-------|------|
| image | IMAGE | Reference frame |

| Output | Type |
|--------|------|
| positive_points | SAM3_POINTS_PROMPT |
| negative_points | SAM3_POINTS_PROMPT |

**Interaction:**
- Left click: Add positive point (include)
- Shift+click or Right click: Add negative point (exclude)

---

## ProPainter

### ProPainterInpaint
Video inpainting with temporal consistency.

| Input | Type |
|-------|------|
| image | IMAGE | Source frames |
| mask | MASK | Regions to inpaint |

| Widget | Default | Description |
|--------|---------|-------------|
| width | 1280 | Processing width |
| height | 720 | Processing height |
| neighbor_length | 5 | Frames to consider |
| ref_stride | 8 | Reference frame interval |
| raft_iter | 10 | Optical flow iterations |
| mask_dilation | 2 | Expand mask edges |
| subvideo_length | 80 | Chunk size for long videos |
| fp16 | 5 | Mixed precision level |
| mask_mode | "enable" | Use provided mask |

| Output | Type |
|--------|------|
| IMAGE | Inpainted frames |
| FLOW_MASK | Flow confidence |
| MASK_DILATE | Dilated mask |

**VRAM tip:** Process at 720p, upscale output.

---

## Built-in Nodes

### SaveImage
Save images to output directory.

| Input | Type |
|-------|------|
| images | IMAGE |

| Widget | Description |
|--------|-------------|
| filename_prefix | Path prefix (e.g., `"depth/depth"` → `depth/depth_00001.png`) |

### PreviewImage
Display images in ComfyUI interface.

| Input | Type |
|-------|------|
| images | IMAGE |

### MaskToImage / ImageToMask
Convert between MASK and IMAGE types.

**MaskToImage:** MASK → IMAGE (grayscale)
**ImageToMask:** IMAGE → MASK (extracts channel: "red", "green", "blue", "alpha")

### ImageScale
Resize images.

| Input | Type |
|-------|------|
| image | IMAGE |

| Widget | Description |
|--------|-------------|
| upscale_method | `"nearest"`, `"bilinear"`, `"bicubic"`, `"lanczos"` |
| width | Target width |
| height | Target height |
| crop | `"disabled"`, `"center"` |

### Note
Text annotation node (no execution).

| Widget | Description |
|--------|-------------|
| text | Markdown-style text content |
