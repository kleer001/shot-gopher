# Troubleshooting

Common issues, solutions, and performance tips.

**Quick links**: [CLI Reference](reference/cli.md) | [Stages](reference/stages.md) | [Installation](installation.md)

---

## Quick Diagnostics

Run these commands to identify problems:

```bash
python scripts/janitor.py -H          # Installation health check
curl http://127.0.0.1:8188/system_stats  # ComfyUI status
nvidia-smi                            # GPU memory usage
ls -lh MyShot/*.log                   # Project logs
```

---

## Common Errors

### ComfyUI Not Running

**Symptom:** `ConnectionError: Cannot connect to ComfyUI`

**Fix:** Start ComfyUI manually or let pipeline auto-start:

```bash
# Manual start
cd .vfx_pipeline/ComfyUI && python main.py --listen

# Or let pipeline handle it (default)
python scripts/run_pipeline.py footage.mp4 -s depth
```

**Verify:** `curl http://127.0.0.1:8188/system_stats`

---

### Workflow Not Found

**Symptom:** `Workflow not found: workflows/01_analysis.json`

**Cause:** Workflows copy from `workflow_templates/` on first run.

**Fix:**
```bash
# Option 1: Re-run with ingest to trigger setup
python scripts/run_pipeline.py footage.mp4 -s ingest,depth

# Option 2: Manual copy
cp workflow_templates/*.json MyShot/workflows/
```

---

### COLMAP Reconstruction Failed

**Symptom:** Empty `colmap/sparse/0/` or reconstruction error

**Common causes:**

| Problem | Solution |
|---------|----------|
| Insufficient features | Use `-q high`, ensure textured surfaces |
| Moving objects | Run `roto` first, pipeline uses masks automatically |
| Minimal camera motion | Use `-q slow` preset |
| Motion blur | No easy fix — need sharper frames |

**Debug:** `cat MyShot/colmap.log`

---

### Motion Capture Requires Camera Data

**Symptom:** `Skipping (camera data required)`

**Cause:** Mocap needs `camera/extrinsics.json` from colmap or depth stage.

**Fix:**
```bash
# Option 1: COLMAP (more accurate)
python scripts/run_pipeline.py footage.mp4 -s colmap,mocap

# Option 2: Depth stage (faster)
python scripts/run_pipeline.py footage.mp4 -s depth,mocap
```

**Verify:** `ls MyShot/camera/` should show `extrinsics.json`

---

### Out of Memory (GPU)

**Symptom:** `CUDA out of memory`

**VRAM requirements by stage:**

| Stage | VRAM |
|-------|------|
| ingest | CPU |
| depth | 7 GB |
| roto | 4 GB |
| mama | 12 GB |
| cleanplate | 6 GB |
| colmap | 2-4 GB |
| mocap | 12 GB |
| gsir | 8 GB |
| camera | CPU |

**Fixes:**
1. Check GPU usage: `nvidia-smi`
2. Restart ComfyUI between stages
3. Process fewer frames at once
4. Edit workflow batch sizes in `MyShot/workflows/*.json`

---

### SAM3 Model Download Failed

**Symptom:** `Error downloading model`

**Note:** SAM3 uses public `1038lab/sam3` repo — no token needed.

**Fix:**
```bash
# Manual download
pip install huggingface_hub
python -c "from huggingface_hub import hf_hub_download; \
  hf_hub_download('1038lab/sam3', 'sam3.pt', \
  local_dir='.vfx_pipeline/ComfyUI/models/sam3')"
```

---

### Stage Failed (General)

**Symptom:** Generic failure without specific error

**Debug steps:**
1. Check logs: `cat MyShot/<stage>.log`
2. Run stage alone: `python scripts/run_pipeline.py footage.mp4 -s depth`
3. Check ComfyUI console for Python errors
4. Verify inputs: `ls MyShot/source/frames/ | head`

---

## Performance

### Stage Dependencies

Which stages can run in parallel:

```
ingest ─┬─► depth ──────────────────────┬─► camera
        │                               │
        ├─► roto ─► mama ─► cleanplate
        │
        └─► colmap ─┬─► mocap
                    └─► gsir
```

**Parallel example:**
```bash
# Terminal 1
python scripts/run_pipeline.py footage.mp4 -s depth

# Terminal 2 (simultaneously)
python scripts/run_pipeline.py footage.mp4 -s roto
```

---

### Skip Existing Output

Avoid re-processing with `-e`:

```bash
python scripts/run_pipeline.py footage.mp4 -s all -e
```

Checks for existing output in each stage directory before running.

---

### Quality vs Speed

| Workflow | Command | Use Case |
|----------|---------|----------|
| Fast preview | `-s depth,colmap,camera -q low` | Quick check |
| Balanced | `-s all` | Default |
| Production | `-s all -q high -d -m` | Final delivery |

---

### Disk Space

**Size per 1000 frames:**

| Output | Size |
|--------|------|
| Frames (PNG) | 2-5 GB |
| Depth maps | 1-2 GB |
| Roto masks | 500 MB |
| Clean plates | 2-5 GB |
| COLMAP sparse | 50-200 MB |
| COLMAP dense | 5-20 GB |
| Mocap | 0.5-2 GB |
| GS-IR | 2-5 GB |

**Total:** 15-40 GB per shot

**Cleanup:**
```bash
du -sh MyShot/*              # Check usage
rm -rf MyShot/depth MyShot/roto  # Remove intermediates
```

---

## Advanced Debugging

### Verbose Logging

```bash
export VFX_DEBUG=1
python scripts/run_pipeline.py footage.mp4 -s depth
```

### Manual Workflow Execution

1. Open http://127.0.0.1:8188
2. Load workflow from `MyShot/workflows/`
3. Run manually to see node-by-node execution
4. Red nodes indicate errors

### Installation Repair

```bash
python scripts/janitor.py -H     # Health check
python scripts/janitor.py -R     # Detailed report
python scripts/janitor.py -r -y  # Attempt repairs
```

---

## Getting Help

Before reporting issues, gather:

```bash
python scripts/janitor.py -H   # Installation status
uname -a                       # OS info
python --version               # Python version
nvidia-smi                     # GPU info
```

**Report issues:** [GitHub Issues](https://github.com/kleer001/shot-gopher/issues)

---

## Related Documentation

- [CLI Reference](reference/cli.md) — Command reference
- [Stages](reference/stages.md) — Stage details and VRAM requirements
- [Installation](installation.md) — Setup guide
- [Maintenance](reference/maintenance.md) — Maintenance tools
