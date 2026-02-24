# GVHMR Camera Extrinsics — Closed-Form Solution

## Problem

GVHMR outputs `smpl_params_global` and `smpl_params_incam` but no explicit camera extrinsics. The GVHMR authors say "extra optimization is required" (issue #30). Existing workarounds use expensive vertex-based SVD alignment.

## Solution

SMPL rotates around the pelvis joint, not the origin. The correct world-to-camera transform is:

```
R_w2c = R_c @ R_w^T
t_w2c = (I - R_w2c) @ J_pelvis + transl_c - R_w2c @ transl_w
```

Where `J_pelvis` is computed from the SMPL-X model's `J_regressor[0] @ v_shaped`.

The naive formula `t = transl_c - R @ transl_w` omits the pelvis correction term and produces ~0.74m error for typical frontal views.

## Key Gotcha: SMPL vs SMPL-X

The pelvis joint position differs by ~13.6cm between SMPL (6890 verts) and SMPL-X (10475 verts). The J_pelvis used for the camera **must match** the body model used for mesh rendering. Our pipeline uses SMPL-X throughout.

## Accuracy

Verified on 233 frames against vertex-based ground truth: **0.7mm max error** (vs 738mm without the correction).

## Reference

- Implementation: `scripts/camera_alignment.py`
- Posted to: https://github.com/zju3dv/GVHMR/issues/30
