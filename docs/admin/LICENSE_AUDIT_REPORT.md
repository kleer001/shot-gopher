# License Audit Report: VFX Pipeline (shot-gopher)

**Report Date:** 2026-01-30
**Project:** shot-gopher / VFX Pipeline
**Report Type:** Executive Summary

---

## Executive Summary

This VFX pipeline integrates multiple open-source projects with **varying license restrictions**. While the pipeline code itself is MIT-licensed (fully permissive), several core dependencies have **non-commercial use restrictions** that significantly impact how the complete system can be deployed.

### Key Finding: Mixed Commercial Viability

| Use Case | Permitted? | Notes |
|----------|------------|-------|
| Personal/hobby use | ✅ Yes | All components permitted |
| Academic research | ✅ Yes | All components permitted |
| Internal R&D (non-commercial) | ✅ Yes | All components permitted |
| Commercial production | ⚠️ Partial | Some components require licensing |
| Commercial distribution | ⚠️ Partial | Requires commercial licenses for restricted components |

---

## License Classification Summary

### Tier 1: Fully Permissive (Commercial OK)

| Component | License | Commercial Use |
|-----------|---------|----------------|
| **shot-gopher** (this project) | MIT | ✅ Unrestricted |
| ComfyUI | GPL-3.0 | ✅ Yes (with copyleft obligations) |
| PyTorch | BSD-3-Clause | ✅ Unrestricted |
| SAM3 (Segment Anything) | Apache-2.0 | ✅ Unrestricted |
| COLMAP | BSD-3-Clause | ✅ Unrestricted (source); GPL when built with deps |
| FFmpeg | LGPL-2.1+ (default) | ✅ Yes (with compliance requirements) |
| GS-IR | MIT | ✅ Unrestricted |
| NumPy | BSD-3-Clause | ✅ Unrestricted |
| SciPy | BSD-3-Clause | ✅ Unrestricted |
| Pillow | HPND (PIL Software License) | ✅ Unrestricted |
| trimesh | MIT | ✅ Unrestricted |
| FastAPI | MIT | ✅ Unrestricted |
| uvicorn | BSD-3-Clause | ✅ Unrestricted |
| Jinja2 | BSD-3-Clause | ✅ Unrestricted |
| alembic (database) | MIT | ✅ Unrestricted |

### Tier 2: Non-Commercial Only (Requires Licensing for Commercial)

| Component | License | Commercial Use |
|-----------|---------|----------------|
| **ProPainter** | NTU S-Lab License 1.0 | ❌ Non-commercial only |
| **ECON** | MPI Non-Commercial License | ❌ Non-commercial only |
| **Depth Anything V3** (Giant/Nested models) | CC BY-NC 4.0 | ❌ Non-commercial only |
| **3D Gaussian Splatting** (INRIA) | INRIA Non-Commercial | ❌ Non-commercial only |
| **VideoMaMa** (code) | CC BY-NC 4.0 | ❌ Non-commercial only |
| **VideoMaMa** (model weights) | Stability AI Community License | ⚠️ Free under $1M revenue; Enterprise license above |

### Tier 3: Registration/Agreement Required

| Component | License | Requirements |
|-----------|---------|--------------|
| **SMPL-X** | MPI Academic License | Free registration required; commercial via Meshcapade |

---

## Detailed Analysis by Component

### This Project: shot-gopher

**License:** MIT License
**Copyright:** 2026 kleer001

**What you can do:**
- Use commercially without restriction
- Modify and create derivative works
- Distribute freely
- Sublicense

**Obligations:**
- Include the copyright notice and license in distributions

---

### ComfyUI (Core Engine)

**License:** [GPL-3.0](https://github.com/comfyanonymous/ComfyUI/blob/master/LICENSE)
**Source:** [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI)

**What you can do:**
- Use commercially
- Modify and distribute
- Use for any purpose

**Obligations (Copyleft):**
- If you distribute modified versions, you must:
  - Release source code under GPL-3.0
  - Include license and copyright notices
  - Document changes made
- Using ComfyUI as a service (SaaS) has specific considerations

**Note:** ComfyUI itself is open source, but the models you use through it have their own licenses.

---

### Segment Anything Model 3 (SAM3)

**License:** [Apache-2.0](https://github.com/1038lab/ComfyUI-SAM3/blob/main/LICENSE)
**Source:** [1038lab/ComfyUI-SAM3](https://github.com/1038lab/ComfyUI-SAM3)

**What you can do:**
- Use commercially without restriction
- Modify and create derivative works
- Distribute with or without modifications

**Obligations:**
- Include license notice
- State significant changes if modified
- Include NOTICE file if present

---

### Depth Anything V3

**License:** Mixed (Apache-2.0 or CC BY-NC 4.0 depending on model)
**Source:** [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3)

**Commercial-friendly models:**
- DA3Mono-Large: Apache-2.0 ✅
- DA3Metric-Large: Apache-2.0 ✅

**Non-commercial only models:**
- DA3-Giant: CC BY-NC 4.0 ❌
- DA3Nested series: CC BY-NC 4.0 ❌

**Recommendation:** Use Large models for commercial applications; Giant/Nested models require non-commercial license or commercial agreement with ByteDance.

---

### ProPainter (Clean Plates)

**License:** [NTU S-Lab License 1.0](https://github.com/sczhou/ProPainter/blob/main/LICENSE)
**Source:** [sczhou/ProPainter](https://github.com/sczhou/ProPainter)

**Permitted uses:**
- Academic research
- Teaching and education
- Public demonstrations
- Personal experimentation

**Prohibited uses:**
- Running business operations
- Licensing, leasing, or selling
- Use in commercial products
- Any activity for commercial gain

**Commercial licensing:** Contact Dr. Shangchen Zhou (shangchenzhou@gmail.com)

---

### ECON (Clothed Human Reconstruction)

**License:** [MPI Non-Commercial License](https://github.com/YuliangXiu/ECON/blob/master/LICENSE)
**Source:** [YuliangXiu/ECON](https://github.com/YuliangXiu/ECON)

**Permitted uses:**
- Non-commercial scientific research
- Non-commercial education
- Non-commercial artistic projects

**Prohibited uses:**
- Commercial purposes
- Pornographic content
- Military applications
- Surveillance
- Defamatory content

**Commercial licensing:** Contact ps-licensing@tue.mpg.de

---

### SMPL-X (Body Model)

**License:** [MPI Academic License](https://smpl-x.is.tue.mpg.de/modellicense.html)
**Source:** [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/)

**Access requirements:**
- Free registration required (approval within 24-48 hours)
- Must agree to license terms

**Permitted uses:**
- Non-commercial scientific research
- Non-commercial education
- Non-commercial artistic projects

**Prohibited uses:**
- Commercial use (without separate license)
- Pornographic content
- Military/surveillance applications
- Training commercial AI systems

**Commercial licensing:** Contact sales@meshcapade.com

---

### GS-IR (Material Decomposition)

**License:** [MIT](https://github.com/lzhnb/GS-IR)
**Source:** [lzhnb/GS-IR](https://github.com/lzhnb/GS-IR)

**What you can do:**
- Use commercially
- Modify and distribute
- Create derivative works

**⚠️ Important caveat:** GS-IR builds upon the original 3D Gaussian Splatting codebase from INRIA, which is **non-commercial only**. Commercial use of GS-IR may require licensing from INRIA (contact: stip-sophia.transfert@inria.fr).

---

### 3D Gaussian Splatting (INRIA Reference Implementation)

**License:** [INRIA Non-Commercial License](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md)
**Source:** [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)

**Permitted uses:**
- Non-commercial research
- Academic evaluation
- Educational purposes

**Prohibited uses:**
- Commercial exploitation without explicit consent
- Distribution for commercial purposes

**Commercial licensing:** Contact stip-sophia.transfert@inria.fr

---

### VideoMaMa (Video Matting)

**License:** Dual-licensed
- **Code:** [CC BY-NC 4.0](https://github.com/cvlab-kaist/VideoMaMa) (Creative Commons Attribution-NonCommercial 4.0)
- **Model Weights:** [Stability AI Community License](https://stability.ai/license) (inherited from Stable Video Diffusion)

**Source:** [cvlab-kaist/VideoMaMa](https://github.com/cvlab-kaist/VideoMaMa)

**Code License (CC BY-NC 4.0) - Permitted uses:**
- Academic research
- Teaching and education
- Personal non-commercial projects
- Public demonstrations (non-commercial)

**Code License (CC BY-NC 4.0) - Prohibited uses:**
- Commercial products or services
- Revenue-generating applications

**Model Weights (Stability AI Community License):**
- ✅ Free for individuals/organizations with <$1M annual revenue
- ⚠️ Requires Enterprise license for organizations with >$1M annual revenue
- Attribution required: "Powered by Stability AI"
- Cannot use outputs to train competing foundational models

**Important notes:**
- The code itself is strictly non-commercial (CC BY-NC 4.0)
- Model weights have a revenue-based threshold via Stability AI
- For commercial use, both licenses must be satisfied (code must be licensed separately)
- The model builds upon Stable Video Diffusion (stabilityai/stable-video-diffusion-img2vid-xt)

**Commercial licensing:**
- Code: Contact KAIST CV Lab via GitHub
- Model weights: Contact Stability AI for Enterprise license (revenue >$1M)

---

### COLMAP (Camera Tracking)

**License:** [BSD-3-Clause](https://colmap.github.io/license.html) (source code)
**Source:** [colmap/colmap](https://github.com/colmap/colmap)

**Complexity:** The COLMAP source code is BSD-licensed, but:
- Pre-built binaries may be GPL due to dependency linking
- Building from source allows BSD compliance if GPL dependencies are excluded

**For commercial use:** Build from source without GPL dependencies, or use under GPL terms.

---

### FFmpeg (Video Processing)

**License:** [LGPL-2.1+](https://www.ffmpeg.org/legal.html) (default); GPL if built with GPL components
**Source:** [FFmpeg/FFmpeg](https://github.com/FFmpeg/FFmpeg)

**For commercial use (LGPL compliance):**
1. Do NOT compile with `--enable-gpl` or `--enable-nonfree`
2. Use dynamic linking (not static)
3. Distribute corresponding FFmpeg source code
4. Include license notices

**GPL triggers:** Building with x264, x265, or other GPL libraries changes the entire binary to GPL.

---

### PyTorch

**License:** [BSD-3-Clause](https://github.com/pytorch/pytorch/blob/main/LICENSE)
**Source:** [pytorch/pytorch](https://github.com/pytorch/pytorch)

**What you can do:**
- Use commercially without restriction
- Modify and distribute
- No copyleft obligations

---

## Commercial Use Roadmap

### Scenario A: Full Commercial Deployment

To use this pipeline commercially with all features, you would need:

1. **Contact for commercial licenses:**
   - NTU S-Lab (ProPainter): shangchenzhou@gmail.com
   - Max Planck Institute (ECON, SMPL-X): ps-licensing@tue.mpg.de
   - Meshcapade (SMPL-X commercial): sales@meshcapade.com
   - INRIA (3D Gaussian Splatting): stip-sophia.transfert@inria.fr
   - ByteDance (Depth Anything Giant/Nested): Contact via GitHub
   - KAIST CV Lab (VideoMaMa code): Contact via GitHub
   - Stability AI (VideoMaMa model weights, if revenue >$1M): stability.ai/license

2. **Use Apache-2.0 Depth Anything models** (Large variants)

3. **Ensure FFmpeg LGPL compliance**

4. **Include GPL notices** for ComfyUI if distributing

### Scenario B: Commercial with Limited Features

Use only permissively-licensed components:

| Stage | Permissive Alternative |
|-------|----------------------|
| Depth estimation | Depth Anything V3 Large (Apache-2.0) ✅ |
| Segmentation | SAM3 (Apache-2.0) ✅ |
| Clean plates | ❌ No permissive alternative (ProPainter restricted) |
| Video matting | VideoMaMa (CC BY-NC 4.0 + Stability AI) ⚠️ |
| Camera tracking | COLMAP (BSD, with care) ✅ |
| Motion capture | GVHMR + SMPL-X (requires license) ⚠️ |
| Material decomposition | GS-IR (MIT, but base tech restricted) ⚠️ |

### Scenario C: Research/Academic Use

All components are freely available for:
- Academic research
- Educational purposes
- Non-commercial demonstrations
- Personal projects

---

## Compliance Checklist

### For Any Distribution

- [ ] Include this project's MIT LICENSE file
- [ ] Include ComfyUI's GPL-3.0 license
- [ ] Include attribution notices for all dependencies
- [ ] Document which optional components are included

### For Commercial Use

- [ ] Audit which non-commercial components are used
- [ ] Obtain commercial licenses for restricted components
- [ ] Use only Apache-2.0/BSD/MIT licensed models
- [ ] Verify FFmpeg build configuration (LGPL vs GPL)
- [ ] Register for SMPL-X if using motion capture features
- [ ] Consider INRIA licensing for Gaussian Splatting features

### For Open Source Distribution

- [ ] Comply with GPL-3.0 for ComfyUI integration
- [ ] Clearly document license restrictions in README
- [ ] Do not bundle non-redistributable components

---

## Recommendations

1. **For production VFX studios:** Contact NTU S-Lab and MPI for commercial licenses before deploying ProPainter and ECON in production pipelines.

2. **For research teams:** All components are freely available; ensure proper citation in publications.

3. **For indie developers:** Use the permissive stack (SAM, Depth Anything Large, COLMAP, GVHMR) for commercial projects; VideoMaMa requires careful consideration of the dual-license (code is CC BY-NC 4.0, model weights are free only under $1M revenue); avoid ProPainter/ECON without licensing.

4. **For SaaS deployment:** ComfyUI's GPL-3.0 and component licenses may have specific implications; consult legal counsel.

---

## Sources

- [ComfyUI License](https://github.com/comfyanonymous/ComfyUI/blob/master/LICENSE)
- [SAM3 License](https://github.com/1038lab/ComfyUI-SAM3/blob/main/LICENSE)
- [Depth Anything V3](https://github.com/ByteDance-Seed/Depth-Anything-3)
- [ProPainter License](https://github.com/sczhou/ProPainter/blob/main/LICENSE)
- [ECON License](https://github.com/YuliangXiu/ECON/blob/master/LICENSE)
- [SMPL-X License](https://smpl-x.is.tue.mpg.de/modellicense.html)
- [GVHMR](https://github.com/zju3dv/GVHMR)
- [GS-IR](https://github.com/lzhnb/GS-IR)
- [3D Gaussian Splatting License](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md)
- [COLMAP License](https://colmap.github.io/license.html)
- [FFmpeg Legal](https://www.ffmpeg.org/legal.html)
- [PyTorch License](https://github.com/pytorch/pytorch/blob/main/LICENSE)
- [VideoMaMa](https://github.com/cvlab-kaist/VideoMaMa)
- [Stability AI Community License](https://stability.ai/license)

---

*This report is provided for informational purposes only and does not constitute legal advice. Consult with legal counsel for specific licensing questions.*
