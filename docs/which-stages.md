# Which Stages Can I Use?

> **Before you run:** Shot-Gopher uses destructive workflows. See [The Gopher's Rules](RulesAndGotchas.md).

A dichotomous key to find the right pipeline stages for your footage.

---

## Start Here

### 1. Is your camera static (tripod/locked-off)?

**Yes** → Go to [2](#2-do-you-need-to-remove-objects-from-the-shot)
- Available: `ingest`, `depth`, `roto`, `interactive`, `mama`, `cleanplate`

**No** → Go to [3](#3-is-your-camera-motion-smooth-dolly-crane-gimbal)

---

### 2. Do you need to remove objects from the shot?

**Yes** → Go to [7](#7-do-you-need-separate-masks-for-multiple-similar-objects)
- Available: `ingest`, `depth`, `roto`, `interactive`, `mama`, `cleanplate`

**No** → Go to [7](#7-do-you-need-separate-masks-for-multiple-similar-objects)
- Available: `ingest`, `depth`, `roto`, `interactive`, `mama`

---

### 3. Is your camera motion smooth (dolly, crane, gimbal)?

**Yes** → Go to [4](#4-does-your-scene-have-good-texture-and-detail)

**No** → Go to [5](#5-is-there-fast-motion-or-motion-blur)

---

### 4. Does your scene have good texture and detail?

**Yes** → Go to [6](#6-do-you-need-motion-capture)
- Available: `ingest`, `depth`, `roto`, `interactive`, `mama`, `cleanplate`, `matchmove_camera`, `gsir`, `mocap`

**No** → Go to [7](#7-do-you-need-separate-masks-for-multiple-similar-objects)
- Available: `ingest`, `depth`, `roto`, `interactive`, `mama`, `cleanplate`

---

### 5. Is there fast motion or motion blur?

**Yes** → Go to [7](#7-do-you-need-separate-masks-for-multiple-similar-objects)
- Available: `ingest`, `depth`

**No** → Go to [6](#6-do-you-need-motion-capture)
- Available: `ingest`, `depth`, `roto`, `interactive`, `mama`, `cleanplate`, `matchmove_camera`, `mocap`

---

### 6. Do you need motion capture?

**Yes** → Go to [7](#7-do-you-need-separate-masks-for-multiple-similar-objects)
- Requires full body visible; add `mocap` after `matchmove_camera`

**No** → Go to [7](#7-do-you-need-separate-masks-for-multiple-similar-objects)

---

### 7. Do you need separate masks for multiple similar objects?

**Yes** → Go to [8](#8-can-text-prompts-distinguish-between-your-objects)

**No** → Done! Use `roto` with a text prompt:
```bash
-s roto --prompt "person"
```

---

### 8. Can text prompts distinguish between your objects?

**Yes** → Done! Use `roto` with specific prompts:
```bash
-s roto --prompt "red car,blue car"
```

**No** → Done! Use `interactive` roto:
```bash
-s interactive
```
See [Interactive Roto Guide](reference/interactive-segmentation.md)

---

## Quick Reference

| Your Situation | Available Stages |
|----------------|------------------|
| Static camera | `ingest`, `depth`, `roto`, `interactive`, `mama`, `cleanplate` |
| Smooth moving + textured | `ingest`, `depth`, `roto`, `interactive`, `mama`, `cleanplate`, `matchmove_camera`, `gsir`, `mocap` |
| Smooth moving + low texture | `ingest`, `depth`, `roto`, `interactive`, `mama`, `cleanplate` |
| Handheld + no fast motion | `ingest`, `depth`, `roto`, `interactive`, `mama`, `cleanplate`, `matchmove_camera`, `mocap` |
| Handheld + fast motion | `ingest`, `depth` |

---

## Common Recipes

**Remove a person:**
```bash
python scripts/run_pipeline.py video.mp4 -s ingest,roto,mama,cleanplate --prompt "person"
```

**Camera track for CGI:**
```bash
python scripts/run_pipeline.py video.mp4 -s ingest,matchmove_camera -q high
```

**Motion capture:**
```bash
python scripts/run_pipeline.py video.mp4 -s ingest,matchmove_camera,mocap
```

**Quick depth + roto preview:**
```bash
python scripts/run_pipeline.py video.mp4 -s ingest,depth,roto --prompt "person"
```
