# UnderPressure Integration Roadmap

## Background

UnderPressure (InterDigital, SCA 2022) provides two things:
1. **Contact detection** — a neural net that predicts per-frame foot contact labels (heel/toe, left/right) from joint positions. Fast forward pass.
2. **Footskate IK cleanup** — an optimization loop that pins feet to the ground during contact. Operates on UnderPressure's own 23-joint Xsens skeleton, not SMPL/SMPLX.

## Key Challenge

UnderPressure does **not** use SMPL or SMPLX. It uses a 23-joint Xsens MVN skeleton with quaternion rotations. It provides an AMASS bridge that:
- Takes 26 AMASS-format joint positions (a subset of SMPLX joints)
- Retargets them to its internal skeleton via 150-iteration optimization
- Runs contact detection on the retargeted skeleton

The footskate IK cleanup is tightly coupled to this internal skeleton. Applying it to our SMPLX `motion.pkl` would require round-tripping: SMPLX → UnderPressure skeleton → cleanup → somehow back to SMPLX. This is lossy and architecturally messy.

## Recommended Approach: Contact Labels Only + Custom SMPLX IK

Use UnderPressure **only for contact detection** (the neural net forward pass). Then apply foot planting in SMPLX space directly.

**Why:**
- Contact detection is the hard/novel part — the neural net is what's valuable
- IK foot planting on SMPLX is straightforward (constrain ankle/toe joint positions, solve with inverse kinematics)
- Avoids lossy skeleton retargeting round-trip
- Keeps everything in SMPLX space, consistent with the rest of the pipeline

## Pipeline Placement

```
mocap → hands → foot_contact → gsir → camera
```

Foot contact runs after hands so the final re-export includes both hand poses and foot cleanup. The stage depends on `mocap` (needs `motion.pkl`).

## Implementation Phases

### Phase 1: Install & Contact Detection

**New files:**
- `scripts/run_foot_contact.py` — main script (runs in `gvhmr` conda env)

**Modified files:**
- `env_config.py` — add `UNDERPRESSURE_INSTALL_DIR`
- `pipeline_constants.py` — add `foot_contact` to `STAGE_ORDER`, `STAGES`, `STAGES_REQUIRING_FRAMES`
- `stage_runners.py` — add `run_stage_foot_contact()`, register in `STAGE_HANDLERS`
- `run_pipeline.py` — add dependency in `sanitize_stages()` (`foot_contact` → `mocap`)
- `install_wizard/installers.py` — add `UnderPressureInstaller`

**Work:**
1. Clone UnderPressure to `.vfx_pipeline/tools/underpressure/`
2. Install into `gvhmr` conda env (only dep is PyTorch, already present)
3. Load `motion.pkl`, run SMPLX forward kinematics to get 26 AMASS joint positions
4. Call `contacts_detection_from_amass()` with the pretrained model
5. Save `foot_contacts.npz` to `mocap/<person>/` with per-frame labels:
   - `left_heel`, `left_toe`, `right_heel`, `right_toe` (boolean arrays)
   - `contact_confidence` (raw vGRF values for debugging)

**Output:** `mocap/<person>/foot_contacts.npz` — contact labels only, no motion modification yet.

### Phase 2: SMPLX Foot Planting IK

**Work:**
1. Load contact labels from Phase 1
2. For each contact segment (contiguous frames where a foot is in contact):
   - Compute the target ground position (median XY of the foot joint during contact, Z=0 or ground plane height)
   - Solve a simple IK: adjust `trans` (root translation) and ankle/knee joint angles in `poses` to pin the foot
3. Smooth the transitions at contact boundaries (Savitzky-Golay or similar, matching the hands stage pattern)
4. Write modified `motion.pkl` back in place
5. Re-export via `export_mocap.py` (same pattern as `run_stage_hands`)

**IK strategy options** (decide during implementation):
- **Translation-only:** Only adjust root `trans` to plant feet. Simplest, handles most sliding. Doesn't fix toe rotation.
- **Translation + ankle rotation:** Adjust `trans` and the 2 ankle joints (indices 7, 8 in SMPLX body_pose). Handles both sliding and penetration.
- **Full lower-body IK:** Use a proper IK chain (hip → knee → ankle → foot). Most correct but most complex. Libraries like `ik_solver` or custom Jacobian IK.

**Recommendation:** Start with translation + ankle rotation. It covers 90% of footskate artifacts with minimal complexity.

### Phase 3: Skip Logic & Polish

- Add completion check: skip if `foot_contacts.npz` exists and `motion.pkl` is unmodified
- Add `--no-foot-cleanup` flag to bypass
- Add `foot_contact` to the `hands` dependency chain (so `--stages hands` also triggers foot cleanup if needed, or vice versa)
- Console output: print contact stats (e.g., "Left foot: 142/217 frames grounded, Right foot: 128/217 frames grounded")

## Dependencies & Environment

- **Conda env:** `gvhmr` (already has PyTorch, smplx, numpy, scipy)
- **New dependency:** UnderPressure repo (git clone, ~50MB with pretrained model included in repo)
- **No new conda env needed** — UnderPressure's only real dep is PyTorch

## Risks & Open Questions

1. **AMASS joint mapping accuracy:** The 26 AMASS joints need to be extracted correctly from SMPLX. SMPLX joints 0-21 map to AMASS joints, but the 4 finger joints need careful mapping. Verify with a test case.
2. **Retargeting quality:** UnderPressure's AMASS bridge does 150 iterations of optimization to fit its skeleton. This adds ~2-5 seconds per sequence. Verify contact labels are accurate on SLAHMR output (SLAHMR already has some ground awareness, so contacts should be reasonably clean input).
3. **Ground plane assumption:** UnderPressure assumes Z=0 is ground. SLAHMR estimates a ground plane — need to align these.
4. **IK stability:** Modifying ankle joints without a proper IK chain can introduce knee popping. May need to add knee constraints in Phase 2.

## References

- [UnderPressure GitHub](https://github.com/InterDigitalInc/UnderPressure)
- [UnderPressure Paper (arXiv)](https://arxiv.org/abs/2208.04598)
