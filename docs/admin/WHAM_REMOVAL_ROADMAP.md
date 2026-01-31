# WHAM Removal Roadmap

Complete removal of WHAM from the codebase in favor of GVHMR-only motion capture.

## Rationale

- GVHMR provides 7-10mm better accuracy on standard benchmarks
- GVHMR is actively developed (SIGGRAPH Asia 2024)
- Maintaining two motion capture backends adds complexity
- Simplifies installation and reduces dependencies (~3GB saved)

## Files to Modify

### Phase 1: Core Code (High Priority)

| File | Line(s) | Changes Required |
|------|---------|------------------|
| `scripts/run_mocap.py` | 140-226, 727-793 | Remove `run_wham_motion_tracking()`, `--method` arg, fallback logic |
| `scripts/stage_runners.py` | 187, 191, 726, 733-734 | Update mocap stage runner, remove WHAM references |
| `scripts/pipeline_constants.py` | 32 | Update mocap description |
| `scripts/verify_models.py` | 27-31 | Remove WHAM model verification entry |
| `scripts/env_config.py` | 42 | Update comment (remove WHAM mention) |

### Phase 2: Install Wizard

| File | Line(s) | Changes Required |
|------|---------|------------------|
| `scripts/install_wizard/config.py` | 45, 86, 91, 113-187 | Remove WHAM paths, env vars from shell scripts |
| `scripts/install_wizard/cli.py` | - | Remove WHAM from component choices |
| `scripts/install_wizard/downloader.py` | - | Remove WHAM download function |
| `scripts/install_wizard/validator.py` | - | Remove WHAM validation |
| `scripts/install_wizard/wizard.py` | - | Remove WHAM wizard steps |
| `scripts/install_wizard/__init__.py` | - | Update exports if needed |
| `scripts/install_wizard.py` | - | Remove WHAM from main script |
| `scripts/janitor.py` | - | Update cleanup paths |

### Phase 3: Tests

| File | Changes Required |
|------|------------------|
| `tests/test_smplx_from_motion.py` | Update class docstring (line 27) |
| `tests/test_gvhmr.py` | Remove any WHAM fallback tests |
| `tests/test_phase_1_complete.py` | Remove WHAM completion checks |

### Phase 4: Documentation

| File | Changes Required |
|------|------------------|
| `README.md` | Lines 34, 55, 197, 220 - Remove WHAM mentions |
| `docs/installation.md` | Remove WHAM installation, update component list |
| `docs/first-project.md` | Update mocap requirements (line 56) |
| `docs/testing.md` | Remove WHAM test instructions throughout |
| `docs/manual-install.md` | Remove "WHAM (Fallback)" section |
| `docs/reference/cli.md` | Remove `--method wham` option, update output dirs |
| `docs/reference/stages.md` | Update mocap stage description |
| `docs/reference/scripts.md` | Update run_mocap.py documentation |
| `docs/reference/maintenance.md` | Remove WHAM maintenance info |
| `docs/platforms/windows.md` | Remove WHAM Windows notes |
| `docs/admin/LICENSE_AUDIT_REPORT.md` | Remove WHAM license section |
| `docs/admin/TODO.md` | Update TODO items |
| `docs/admin/GVHMR_TRANSITION_ROADMAP.md` | Delete (transition complete) |

### Phase 5: Output Path Migration

**Critical:** Change motion output path for simplicity.

| Current | New |
|---------|-----|
| `mocap/wham/motion.pkl` | `mocap/motion.pkl` |
| `mocap/gvhmr/` | `mocap/gvhmr/` (keep for raw GVHMR output) |

Files to update:
- `scripts/run_mocap.py` - Output path
- `scripts/stage_runners.py` - Expected output path
- `scripts/smplx_from_motion.py` - Default motion path in docs
- `docs/reference/scripts.md` - Example paths
- `docs/testing.md` - Expected output paths

## Execution Order

1. **Phase 1: Core code** - Remove WHAM functions from run_mocap.py first
2. **Phase 2: Install wizard** - Update installation system
3. **Phase 3: Tests** - Update test expectations
4. **Phase 4: Documentation** - Update all user-facing docs
5. **Phase 5: Output paths** - Migrate to cleaner paths
6. **Cleanup** - Delete obsolete files, final verification

## Key Transformations

### run_mocap.py

**Remove entirely:**
- `run_wham_motion_tracking()` function (~70 lines)
- `check_dependency("wham", ...)` call
- `wham_available` variable and checks
- `--method` argument from argparse
- Fallback logic in `run_mocap_pipeline()`
- WHAM deprecation warning

**Simplify:**
- `run_mocap_pipeline()` - Remove method parameter, always use GVHMR
- Keep `convert_gvhmr_to_wham_format()` but rename to `save_motion_output()`

### install_wizard/config.py

**Remove from shell script templates:**
- `WHAM_DIR` environment variable
- WHAM path in `PYTHONPATH`
- `"wham"` key from paths dict

## Verification Checklist

- [ ] `python scripts/run_mocap.py --help` shows no `--method` option
- [ ] `python scripts/install_wizard.py --list` shows no WHAM component
- [ ] `grep -ri "wham" scripts/` returns only this roadmap reference
- [ ] `grep -ri "wham" docs/` returns no results
- [ ] All tests pass: `pytest tests/`
- [ ] Pipeline runs: `python scripts/run_pipeline.py test.mp4 -s mocap --dry-run`
- [ ] Motion output at `mocap/motion.pkl` (not `mocap/wham/motion.pkl`)

