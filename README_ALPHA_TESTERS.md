# Alpha Testing Guide - VFX Pipeline

**Status**: Alpha Release
**Target Users**: Technical users comfortable with Python and troubleshooting
**Expected Time**: 30-60 minutes for full setup

**⚠️ IMPORTANT**: SAM3 and SMPL-X models require requesting access/registration before use. See "What Needs Real-World Validation" section below.

---

## What Works Solidly ✅

### Installation Wizard
The installation wizard (`scripts/install_wizard.py`) is **production-quality**:

- ✅ Automatic conda environment creation
- ✅ Disk space checking and estimation
- ✅ Resume interrupted installations
- ✅ Checkpoint downloading with progress bars
- ✅ Post-install validation tests
- ✅ Configuration file generation
- ✅ Clear error messages and graceful degradation

**This should "just work"** on Ubuntu/Debian with conda installed.

### Core Pipeline
- ✅ Video frame extraction
- ✅ Depth analysis (Depth Anything V3)
- ✅ Segmentation (SAM3)
- ✅ Clean plate generation
- ✅ COLMAP integration for SfM

---

## What Needs Real-World Validation ⚠️

### 1. Checkpoint Downloads
The wizard downloads checkpoints from:

- **WHAM**: Google Drive (uses `gdown` library)
- **ECON**: icon.is.tue.mpg.de (requires separate registration from SMPL-X)

**If downloads fail:**
1. Visit the project GitHub pages (URLs shown in error messages)
2. Download checkpoints manually
3. Place in:
   - WHAM: `<repo>/.vfx_pipeline/WHAM/checkpoints/`
   - ECON: `<repo>/.vfx_pipeline/ECON/data/`

### 2. Motion Capture Integration
The `run_mocap.py` script makes assumptions about WHAM and ECON CLI interfaces.

**Expected Issues**:
- Actual entry points might have different names
- Arguments might differ from assumptions
- Models might need specific preprocessing

**If mocap fails:**
1. Check `--test-stage motion` or `--test-stage econ` to isolate which component fails
2. Run WHAM/ECON manually to understand their CLI
3. Update `scripts/run_mocap.py` with correct commands
4. Report findings so we can fix it

### 3. SAM3 Model Access (HuggingFace)
**Now automated with HuggingFace token:**

SAM3 (Segment Anything Model 3) requires accepting the license on HuggingFace:

1. Create a HuggingFace account at https://huggingface.co/join
2. Visit https://huggingface.co/facebook/sam3
3. Click "Agree and access repository" to accept the license
4. Go to https://huggingface.co/settings/tokens
5. Create a new token with "Read" access
6. Create `HF_TOKEN.dat` in repository root with your token:
   ```
   hf_yourTokenHere1234567890abcdefghijklmnop
   ```
7. Run wizard - model downloads automatically to `.vfx_pipeline/ComfyUI/models/sam/`

**Template file**: Copy `HF_TOKEN.dat.template` and fill in your token.

**Without SAM3 access**: Segmentation workflows (roto, cleanplate stages) will fail.

**Testing priority**: Please report if SAM3 access approval and automated download work in your region.

### 4. SMPL-X and ECON Models (MPG)
**Now automated with credentials file:**

**IMPORTANT: You need to register at TWO separate websites!**

#### What Each Component Provides

**SMPL-X** (Skinned Multi-Person Linear model, eXpressive):
- Parametric body model that defines skeleton, mesh topology, and UV layout
- The "rigged character" - a deformable human mesh with consistent vertex ordering
- Provides the underlying body structure that can be posed and animated
- Includes face and hand articulation

**ECON** (Explicit Clothed humans Obtained from Normals):
- Clothed human reconstruction from monocular video
- Takes video frames and creates a detailed 3D mesh including clothing geometry
- Uses SMPL-X as the body prior to ensure consistent topology under the clothes
- Outputs meshes that match SMPL-X vertex ordering for animation compatibility

#### Registration Sites

1. **SMPL-X**: https://smpl-x.is.tue.mpg.de/
2. **ECON**: https://icon.is.tue.mpg.de/

These are **separate registrations** - you need accounts on both sites.

#### Setup Steps

1. Register at both websites
2. Wait for approval emails (usually 24-48 hours for each site)
3. Create `SMPL.login.dat` in repository root:
   ```
   your.email@example.com
   your_password_here
   ```
4. Run wizard - models download automatically to `.vfx_pipeline/smplx_models/` and `.vfx_pipeline/ECON/data/`

**Template file**: Copy `SMPL.login.dat.template` and fill in credentials.

**Testing priority**: Please report if SMPL-X and ECON downloads succeed with authentication.

---

## System Requirements

### Minimum
- Ubuntu 20.04+ or Debian-based Linux
- Python 3.8+ (3.10 recommended)
- Conda or Miniconda
- 40GB free disk space
- Git
- ffmpeg

### Recommended
- NVIDIA GPU with 12GB+ VRAM (for motion capture)
- 16GB+ system RAM
- 100GB+ free disk space (for projects)
- Fast internet (large checkpoint downloads)

---

## Quick Start

### Option A: One-Liner Install (Fastest) ⚡

**Just run this:**
```bash
curl -fsSL https://raw.githubusercontent.com/kleer001/comfyui_ingest/main/bootstrap.sh | bash
```

This will:
1. Clone the repository to `./comfyui_ingest` (current directory)
2. Launch the installation wizard automatically
3. Handle everything for you

**Or if you prefer wget:**
```bash
wget -qO- https://raw.githubusercontent.com/kleer001/comfyui_ingest/main/bootstrap.sh | bash
```

---

### Option B: Manual Install (If You Prefer Control)

#### 1. Install Prerequisites
```bash
# Install conda (if not already installed)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Install system dependencies
sudo apt update
sudo apt install git ffmpeg colmap
```

#### 2. Clone Repository
```bash
git clone https://github.com/kleer001/comfyui_ingest.git
cd comfyui_ingest
```

#### 3. Run Installation Wizard
```bash
# Interactive installation (recommended)
python scripts/install_wizard.py

# Or check status first
python scripts/install_wizard.py --check-only
```

**Follow the prompts:**
- Option 1: Core pipeline only (~7GB)
- Option 2: Core + ComfyUI (~10GB) ← Choose this for workflow testing
- Option 3: Full stack (~42GB) ← Choose this for complete testing
- Option 4: Custom selection
- Option 5: Nothing (check only)

The wizard will:
1. Create conda environment (`vfx-pipeline`)
2. Install Python packages
3. Clone ComfyUI and custom nodes (if selected)
4. Download checkpoints (if URLs work)
5. Validate installation
6. Generate activation script

#### 4. Activate Environment
```bash
conda activate vfx-pipeline
# Or use generated script
source .vfx_pipeline/activate.sh
```

#### 5. Verify Installation Health

Use the janitor tool to check everything is working:

```bash
# Quick health check
python scripts/janitor.py -H

# Detailed status report
python scripts/janitor.py -R
```

This will verify:
- Conda environment exists
- Git repositories are clean
- Checkpoints downloaded correctly
- All components pass validation tests

#### 6. Test Pipeline
```bash
# Create test project
python scripts/run_pipeline.py init test_project

# Run frame extraction
python scripts/run_pipeline.py ingest test_project /path/to/video.mp4

# Check what stages are available
python scripts/run_pipeline.py run test_project --help
```

---

## Maintenance During Testing

### Keep Installation Healthy

```bash
# Daily health check (5 seconds)
python scripts/janitor.py -H

# Update components
python scripts/janitor.py -u -y

# Clean temporary files
python scripts/janitor.py -c

# Full maintenance
python scripts/janitor.py -a
```

### Troubleshooting with Janitor

**If something isn't working:**

```bash
# 1. Check health
python scripts/janitor.py -H

# 2. Try repairs
python scripts/janitor.py -r -y

# 3. Verify fix worked
python scripts/janitor.py -H
```

**If checkpoints are missing:**

```bash
# Re-download all checkpoints
python scripts/janitor.py -r
```

**Check disk usage:**

```bash
python scripts/janitor.py -R | grep -A 20 "Disk Usage"
```

---

## Testing Checklist

Please test and report results for:

### Installation Wizard
- [ ] Conda environment creation works
- [ ] Disk space check accurate
- [ ] Checkpoint downloads succeed (or fail gracefully)
- [ ] Resume after Ctrl+C works
- [ ] Validation tests pass
- [ ] Generated `activate.sh` works

### Core Pipeline
- [ ] Frame extraction from video
- [ ] Depth analysis completes
- [ ] Segmentation works
- [ ] COLMAP reconstruction works

### ComfyUI (if testing)
- [ ] ComfyUI cloned successfully
- [ ] Custom nodes installed (VideoHelperSuite, DepthAnythingV3, SAM3)
- [ ] Server starts: `python main.py --listen`
- [ ] Can load workflow from `workflows/` directory

### Motion Capture (if testing)
- [ ] WHAM checkpoint downloads
- [ ] ECON checkpoint downloads
- [ ] `run_mocap.py --check` passes
- [ ] Motion tracking works on test video
- [ ] Geometry reconstruction works
- [ ] Texture projection works

---

## Known Issues & Workarounds

### Issue: "Checkpoint download failed"
**Cause**: URL incorrect or authentication required
**Workaround**: Manual download from project GitHub, place in correct directory

### Issue: "SMPL-X models not found"
**Cause**: Requires manual registration
**Expected**: Follow instructions at https://smpl-x.is.tue.mpg.de/

### Issue: "COLMAP not found"
**Cause**: Not installed via system package manager
**Fix**: `sudo apt install colmap` or `conda install -c conda-forge colmap`

### Issue: "Motion capture stage fails"
**Cause**: CLI interface mismatch
**Workaround**: Test individual components with `--test-stage`, report findings

### Issue: "Out of VRAM"
**Cause**: GPU too small for mocap
**Expected**: Need 12GB+ VRAM for WHAM/ECON

---

## How to Report Issues

Please include in bug reports:

1. **Installation log**: Run with `--check-only` and paste output
2. **System info**:
   ```bash
   uname -a
   python --version
   conda --version
   nvidia-smi  # if GPU testing
   ```
3. **Validation results**:
   ```bash
   python scripts/install_wizard.py --validate
   ```
4. **Error messages**: Full error output, not just summaries
5. **Steps to reproduce**: Exact commands run

**Where to report**:
- GitHub Issues: [link to your repo]
- Email: [your email]
- Discord/Slack: [if applicable]

---

## Success Criteria for Alpha

We consider alpha successful if:

✅ **Installation wizard completes** without manual intervention (excluding SMPL-X)
✅ **Core pipeline runs** end-to-end on test video
⚠️ **Motion capture attempts** to run (even if fails - we'll fix integration)
✅ **Documentation is clear** enough to debug issues

---

## Expected Rough Edges

This is **alpha software**. Expect:

- Checkpoint URLs might need updating
- Motion capture CLI integration might need tweaks
- Error messages might be too technical
- Some edge cases not handled
- Performance not optimized

**What we guarantee**:
- No data loss (atomic writes)
- Safe to Ctrl+C and resume
- Clear error messages
- No security vulnerabilities

**What we don't guarantee (yet)**:
- Everything works first try
- Performance is optimal
- All edge cases handled
- Works on non-Linux systems

---

## Next Steps After Alpha

Based on your feedback, we'll:

1. Fix checkpoint URLs with verified working links
2. Update mocap integration with correct CLI interfaces
3. Improve error messages based on common issues
4. Add more validation tests
5. Document workarounds for common problems
6. Beta release with real-world tested components

---

## Questions During Testing?

**Before asking:**
1. Check `TESTING.md` for component-specific testing
2. Run `--validate` to see what's broken
3. Check logs in `<repo>/.vfx_pipeline/install_state.json`

**Still stuck?**
Ask! Include the info from "How to Report Issues" above.

---

## Thank You! 🙏

Your alpha testing helps make this pipeline production-ready. Every bug report, suggestion, and success story helps us improve.

**Remember**: Alpha means "expect some friction". If something breaks, that's valuable data!

---

## Quick Command Reference

```bash
# Installation
python scripts/install_wizard.py                    # Interactive install
python scripts/install_wizard.py --check-only       # Check status
python scripts/install_wizard.py --validate         # Run validation
python scripts/install_wizard.py --resume           # Resume interrupted

# Pipeline
python scripts/run_pipeline.py init <name>          # Create project
python scripts/run_pipeline.py ingest <proj> <vid>  # Extract frames
python scripts/run_pipeline.py run <proj> <stage>   # Run stage

# Motion Capture
python scripts/run_mocap.py --check                 # Check dependencies
python scripts/run_mocap.py <project>               # Run full pipeline
python scripts/run_mocap.py <proj> --test-stage motion  # Test WHAM only

# Activation
conda activate vfx-pipeline                         # Activate environment
source .vfx_pipeline/activate.sh                  # Use generated script
```

---

**Version**: Alpha 1.0
**Last Updated**: 2026-01-11
**Expected Alpha Duration**: 2-4 weeks
