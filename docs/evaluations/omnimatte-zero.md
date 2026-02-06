# Evaluation: OmnimatteZero as Clean Plate Replacement

**Date:** 2026-02-06
**Paper:** [OmnimatteZero: Fast Training-free Omnimatte with Pre-trained Video Diffusion Models](https://arxiv.org/abs/2503.18033) (SIGGRAPH Asia 2025)
**Repo:** [dvirsamuel/OmnimatteZero](https://github.com/dvirsamuel/OmnimatteZero)
**Current tooling:** ProPainter (default) + Temporal Median (static camera fallback)

---

## Summary

OmnimatteZero is a training-free, zero-shot video layer decomposition method built on
pre-trained video diffusion models (LTX-Video). It performs object removal, foreground
extraction, and layer composition. The paper reports state-of-the-art PSNR/SSIM/LPIPS on
standard omnimatte benchmarks (Movies, Kubric) and claims 0.04 sec/frame on an A100 GPU.

**Verdict: Not ready to replace ProPainter. Promising for a future iteration.**

---

## Comparison Matrix

| Criterion | ProPainter (current) | OmnimatteZero |
|---|---|---|
| **Task** | Video inpainting | Full omnimatte (removal + extraction + composition) |
| **VRAM** | ~6 GB base, ~18 GB peak | 32 GB+ recommended |
| **Speed** | Moderate (optical flow + refinement) | 0.04 sec/frame (paper, A100) |
| **Camera motion** | Handles moving cameras | Not explicitly addressed; relies on diffusion temporal priors |
| **Resolution** | Up to 1920x1080 (configurable cap) | Default 512x768, optional 720x1280 |
| **Input format** | PNG frames + PNG masks | MP4 video + MP4 masks (object + total) |
| **Mask requirements** | Single combined binary mask | Two masks: object-only AND object+effects (shadows/reflections) |
| **Training required** | No (pretrained weights) | No (zero-shot, uses LTX-Video 0.9.7) |
| **Dependencies** | ComfyUI + ProPainter custom nodes | PyTorch 2.4+, diffusers, transformers, LTX-Video model |
| **License** | NTU S-Lab 1.0 (non-commercial) | **No LICENSE file in repo** |
| **Maturity** | Established, widely used | Research code, recent model migration bug |
| **Temporal consistency** | Optical flow propagation | Diffusion temporal priors (attention guidance disabled due to model migration) |
| **Benchmarks** | Industry standard for video inpainting | SOTA on Movies/Kubric (PSNR 35.11 / 44.97) |

---

## Advantages Over ProPainter

1. **Superior quantitative results.** Outperforms ProPainter on PSNR, SSIM, and LPIPS across
   both Movies and Kubric benchmarks.

2. **Effect-aware removal.** Can remove not just the object but its shadows, reflections, and
   other associated effects — something ProPainter does not explicitly handle.

3. **Faster inference (claimed).** 0.04 sec/frame on A100 vs ProPainter's iterative refinement
   approach.

4. **Additional capabilities.** Foreground layer extraction and layer composition are bonuses
   that could feed downstream stages (compositing, relighting).

5. **Zero-shot.** No per-video optimization or fine-tuning needed.

---

## Blockers and Risks

### Hard Blockers

1. **No license file.** The repo has no LICENSE. This is worse than ProPainter's non-commercial
   license — without a license, all rights are reserved by default. Cannot integrate until
   licensing is clarified.

2. **VRAM requirement (32 GB+).** The current pipeline targets 6-18 GB GPUs. OmnimatteZero's
   32 GB+ recommendation would exclude most production hardware. The pipeline already has
   low-VRAM batched processing for ProPainter — no equivalent exists for OmnimatteZero.

3. **Low default resolution (512x768).** Production footage is typically 1920x1080 or higher.
   The maximum documented resolution is 720x1280. Unclear how quality and VRAM scale beyond
   that.

4. **Dual-mask requirement.** The pipeline currently generates a single binary roto mask.
   OmnimatteZero needs both an `object_mask` and a `total_mask` (object + effects). Generating
   the total mask requires either manual annotation or running their `self_attention_map.py`
   script (additional GPU pass). This adds pipeline complexity and another failure point.

### Significant Risks

5. **Active model migration issues.** The repo documents a bug that forced migration from
   LTX-Video 0.9.1 to 0.9.7. The temporal and spatial attention guidance modules described in
   the paper are currently **disabled** in the released code. The paper's reported numbers may
   not be reproducible with the current codebase.

6. **Input format mismatch.** Pipeline works with PNG frame sequences; OmnimatteZero expects
   MP4 videos. Requires conversion wrappers.

7. **Camera motion handling undocumented.** The paper evaluates on Movies and Kubric datasets
   which contain limited camera motion. ProPainter explicitly handles moving cameras via optical
   flow propagation — a core requirement for the pipeline.

8. **Research code maturity.** No versioned releases, no CI, limited error handling. Contrast
   with ProPainter which has an established ComfyUI integration and is battle-tested in the
   current pipeline.

9. **Large model download.** LTX-Video 0.9.7 is a full video diffusion model — likely 10+ GB
   download vs ProPainter's ~0.5 GB.

---

## Integration Effort Estimate

If blockers were resolved, integration would require:

- [ ] License clarification with authors
- [ ] PNG frame sequence <-> MP4 conversion wrappers
- [ ] Total mask generation step (self_attention_map.py or SAM2 integration)
- [ ] Resolution upscaling strategy (process at 720p, upscale? tile-based?)
- [ ] VRAM profiling and low-VRAM batching strategy
- [ ] ComfyUI custom node wrapper OR direct Python integration
- [ ] Pipeline argument for method selection (`--cleanplate-omnimatte`)
- [ ] Tests with production footage (moving camera, bokeh, multiple subjects)
- [ ] Update env_config.py with new environment variables

---

## Recommendation

**Do not replace ProPainter now.** The blockers (no license, 32 GB VRAM, low resolution, disabled
attention guidance) are individually disqualifying.

**Track for future evaluation when:**

1. A LICENSE file is added to the repo (contact authors: Dvir Samuel et al.)
2. Temporal/spatial attention guidance is re-enabled for LTX-Video 0.9.7+
3. Higher resolution support (1080p+) is demonstrated
4. VRAM requirements are reduced or a memory-efficient mode is released

**Short-term alternative:** If effect-aware removal (shadows, reflections) is the primary
motivation, consider augmenting the current ProPainter pipeline with a shadow/reflection
detection pre-pass to expand the inpainting mask — simpler and no new dependencies.

---

## Sources

- [OmnimatteZero GitHub](https://github.com/dvirsamuel/OmnimatteZero)
- [OmnimatteZero Paper (arXiv)](https://arxiv.org/abs/2503.18033)
- [SIGGRAPH Asia 2025 Proceedings](https://dl.acm.org/doi/10.1145/3757377.3763917)
- [ProPainter GitHub](https://github.com/sczhou/ProPainter)
- [ProPainter License (NTU S-Lab 1.0)](https://github.com/sczhou/ProPainter/blob/main/LICENSE)
