# GVHMR Camera Stabilization — Attempts & Findings

Status: **Resolved** — mmcam-direct approach (#7) achieves 0px reprojection error with sub-2mm Y jitter.

## Problem

The GVHMR-derived mocap camera (from `camera_alignment.py`) bounces violently:
- **Y-axis jitter:** 10cm/frame jumps (delta_max ~105mm)
- **Rotation jitter:** ~0.58 deg/frame from body orient noise

Root cause: `t_c2w = -R_c2w @ t_w2c` amplifies small per-frame body orient errors across the multi-meter camera-body distance vector. GVHMR's orient estimates are per-frame regression outputs, not temporally optimized.

Meanwhile, the COLMAP/VGGSfM camera solve (feature-track based) is 30x smoother: ~0.02 deg/frame rotation, millimeter-level position stability.

## Gold Standard

GVHMR's `1_incam.mp4` output video — the body mesh rendered from the camera-space perspective — is the single source of truth for camera-body consistency. Any stabilized camera must reproduce this view when projecting the GVHMR body mesh.

## Approaches Tried

### 1. COLMAP Rotation Substitution (rotation only)

**Idea:** Replace GVHMR's noisy `R_w2c` with COLMAP's smooth rotation, aligned into GVHMR's frame via quaternion-averaged `R_align`. Keep GVHMR's translation formula.

**Math:**
```
R_rel[i] = R_gvhmr_c2w[i] @ R_colmap_c2w[i]^T
R_align = quat_mean(R_rel)
R_w2c_stable = (R_align @ R_colmap_c2w)^T
t_w2c = (I - R_w2c_stable) @ J_pelvis + transl_c - R_w2c_stable @ transl_w
```

**Result:** Rotation smoothness improved dramatically. Translation still jittery because `transl_c` and `transl_w` from GVHMR are independently noisy, and the pelvis correction term still amplifies through the new (different) R_w2c.

**Conclusion:** Fixes rotation but not translation. Half the problem.

### 2. Savitzky-Golay Smoothing (fallback)

**Idea:** When no COLMAP solve exists, smooth the raw GVHMR c2w trajectory:
- Translation: savgol_filter per axis (window=11, poly=3)
- Rotation: convert to quaternions (hemisphere-consistent), savgol each component, renormalize

**Result:** Reduces jitter noticeably but introduces lag/overshoot on fast motion. Still noisy compared to feature-track cameras. Acceptable as a fallback.

**Conclusion:** "Better than nothing" fallback. Not a real fix.

### 3. Sim(3) Procrustes / Umeyama Alignment (full trajectory)

**Idea:** Find optimal similarity transform `(s, R, t)` mapping COLMAP positions to GVHMR positions via SVD. Replace the entire GVHMR trajectory with the aligned COLMAP trajectory.

**Math (Umeyama 1991):**
```
sigma = tgt_centered.T @ src_centered / N
U, D, Vt = svd(sigma)
R = U @ S @ Vt  (S ensures det=+1)
s = trace(D @ S) / var_src
t = mu_tgt - s * R @ mu_src
```

**Result:** Introduced ~15 degree spurious Z rotation. The SVD cross-covariance matrix picks up structured noise from GVHMR's body orient errors as a rotation component. The noise is anisotropic (body lateral sway errors differ from depth errors), which biases the SVD.

**Conclusion:** Fails because the target positions (GVHMR) are too noisy for reliable rotation estimation via SVD. Standard Procrustes assumes low-noise correspondences.

### 4. Scale + Translation Only (no rotation)

**Idea:** Since COLMAP cameras are injected into GVHMR as VO input, both coordinate systems share the same orientation. Drop rotation from alignment, solve only for scale `s` and offset `t`.

**Math:**
```
s = sum(tgt_centered * src_centered) / sum(src_centered^2)
t = mu_tgt - s * mu_src
```

**Result:** Camera orientation was wrong — COLMAP's world frame IS arbitrary (set by whichever image pair initialized reconstruction). Without rotation alignment, the camera can point in the wrong direction entirely.

**Conclusion:** Rotation alignment is necessary; can't skip it.

### 5. First-Frame Rotation Anchor + Scale/Offset

**Idea:** Use GVHMR's first-frame camera orientation to establish the rotation between coordinate frames:

```
R_align = R_gvhmr_c2w[0] @ R_colmap_c2w[0]^T
```

Then apply `R_align` to all COLMAP frames (preserving smooth relative motion), followed by scale+offset on rotated positions.

**Result:** First frame did not match the gold standard `1_incam.mp4` render. The GVHMR-derived first-frame orientation is itself noisy/unreliable as an anchor, and the resulting camera view diverges from what GVHMR internally uses.

**Conclusion:** The GVHMR-derived c2w at any single frame is not reliable enough to serve as a rotation anchor.

## Key Insight

All approaches share a fundamental problem: the GVHMR-derived camera (`R_w2c` from orient params, `t_w2c` from transl params with pelvis correction) is **algebraically correct but numerically noisy**. It matches the body mesh by construction, but any attempt to substitute external rotation or position data breaks the algebraic consistency.

The gold standard `1_incam.mp4` is rendered by GVHMR internally using `smpl_params_incam` directly — NOT the derived w2c camera. This means the "camera" that produces the correct view may not be the same as the w2c we compute.

## Answers to Previous Open Questions

1. **What camera does GVHMR actually use for `1_incam.mp4`?** Confirmed by code inspection: an **identity camera** (R=I, T=0). Vertices from `smpl_params_incam` are already in camera space; the renderer projects them through K with no extrinsic transform. There is no hidden camera matrix.

2. **Can we extract GVHMR's internal camera directly?** No — there is no internal camera. The `1_incam.mp4` render pipeline does SMPLX forward kinematics with `smpl_params_incam`, producing camera-space vertices directly.

3. **Would a hybrid approach work?** Yes — this is what approach #6 implements (see below).

4. **Is the problem actually the rotation or the intrinsics?** The problem is noise in both rotation and translation. Intrinsics (`K_fullimg`) are correct and consistent between rendering and our extraction.

### 6. Robust Alignment + Complementary Filter (current)

**Idea:** Treat this as a sensor fusion problem. GVHMR provides correct-but-noisy camera poses; COLMAP provides smooth-but-arbitrary-frame poses. Fuse them using a complementary filter to get both accuracy and stability.

**Three stages:**

**Stage 1 — Robust rotation alignment (Wahba problem):**
Instead of anchoring to a single noisy frame, compute R_rel[t] = R_gvhmr @ R_colmap^T for ALL frames, then find the optimal rotation via chordal L2 mean:
```
M = mean(R_rel)
U, _, Vt = svd(M)
R_align = U @ diag(1, 1, det(U @ Vt)) @ Vt
```
Noise averages out across N frames. Far more robust than any single-frame anchor.

**Stage 2 — Scale + translation offset:**
Same as before — least-squares fit for scale `s` and offset on rotated COLMAP positions.

**Stage 3 — Complementary filter on translation:**
```
residual[t] = pos_gvhmr[t] - pos_colmap_aligned[t]
pos_final[t] = pos_colmap_aligned[t] + savgol(residual, window=31, poly=2)[t]
```
High-frequency motion from COLMAP (feature-tracked, 30x smoother than GVHMR). Low-frequency body tracking from GVHMR (corrects alignment drift). The smoothed residual cancels systematic alignment bias while rejecting per-frame GVHMR noise.

**Result:** Evaluated on real shots. Rotation stabilized but translation still showed 20-110px reprojection error and 20-93mm Y jitter. The fundamental problem is that fusing two noisy signals doesn't recover the exact camera-body relationship.

**Properties:**
- Rotation: COLMAP smoothness preserved (constant R_align applied globally)
- Translation: COLMAP smoothness at high frequencies + GVHMR tracking at low frequencies
- No per-shot tuning — fixed parameters
- Mathematically grounded: Wahba problem (attitude estimation) + complementary filter (standard sensor fusion)

**Conclusion:** Better than approaches 1-5 but still not good enough. Superseded by approach #7.

### 7. mmcam-direct (current, resolved)

**Idea:** Skip camera derivation entirely. Instead of deriving a noisy camera from GVHMR body params and trying to stabilize it, use the mmcam (VGGSfM) camera directly and transform the body mesh to match.

**How it works:**
1. Camera = mmcam extrinsics (c2w matrices from VGGSfM feature tracking)
2. Body = SMPLX mesh from `smpl_params_incam` (camera-space body params)
3. Per-frame transform: `v_world[t] = R_c2w[t] @ v_cam[t] + t_c2w[t]`

Camera-body consistency is guaranteed by construction: the incam body is what GVHMR predicts the camera sees, and the mmcam camera defines where that camera is in world space.

**Result (eval_camera.py on 3 test projects):**
- **Reprojection error: 0.00 px** (exact by construction)
- **Y-axis jitter: 1.6-2.1 mm/frame** (inherits mmcam smoothness)
- **Rotation jitter: ~0.02 deg/frame** (30x smoother than GVHMR-derived)

Compare with approach 6 (Wahba+CF): 20-110px reprojection, 20-93mm Y jitter.

**Caveat:** Requires a good VGGSfM solve. If the solve has large jumps (e.g., TNIS0025 with 10m position jumps), the body will follow those jumps. Fallback to smoothed GVHMR camera (approach #2) when no mmcam is available.

**Implementation:**
- `camera_alignment.py`: `compute_aligned_camera()` returns mmcam c2w directly
- `export_mocap.py`: `convert_gvhmr_to_motion(use_incam=True)` + `transform_meshes_to_world()`
- `run_mocap.py`: `run_export_pipeline()` passes `--camera-extrinsics` when mmcam metadata says `mmcam_direct`

## Files

- `scripts/camera_alignment.py` — camera alignment math, mmcam-direct passthrough
- `scripts/export_mocap.py` — mesh export with optional incam->world transform
- `scripts/run_mocap.py` — pipeline plumbing, camera metadata routing
- `scripts/transforms.py` — quaternion/rotation utilities
- `scripts/eval_camera.py` — camera evaluation tool
- `tests/test_camera_alignment.py` — unit tests (35 tests)

## References

- Original closed-form solution: `docs/admin/GVHMR_camera_extrinsics.md`
- GVHMR issue: https://github.com/zju3dv/GVHMR/issues/30
- Umeyama (1991): "Least-Squares Estimation of Transformation Parameters Between Two Point Patterns"
- Memory notes: `.claude/projects/.../memory/gvhmr_camera.md`
