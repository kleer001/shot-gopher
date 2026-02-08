# Refactor Execution Order

Master sequencing guide for the three concurrent refactors. Read this first.

## Execution Order (mandatory)

```
Phase 1: ProPainter Deprecation     ← MUST be first
Phase 2: COLMAP → mmcam rename      ← MUST be after Phase 1
Phase 3: Remove -s all              ← simultaneous with Phase 2 or after
```

### Why this order?

1. **ProPainter first** because `run_stage_cleanplate()` currently has two branches
   (ProPainter vs. median). Simplifying this to one branch *before* touching
   anything else means fewer moving parts during the COLMAP rename.

2. **COLMAP rename second** because:
   - The `web/config/pipeline_config.json` "all" preset lists both `"cleanplate"`
     and `"colmap"`. Changing the cleanplate description first (Phase 1) then
     renaming colmap (Phase 2) avoids editing the same JSON block twice.
   - `stage_runners.py` has cross-stage references (`run_stage_gsir` reads
     `colmap/sparse/0`). Doing the COLMAP rename as a focused pass catches all
     these without ProPainter noise in the diff.

3. **`-s all` removal last (or with Phase 2)** because:
   - The "all" preset in `pipeline_config.json` references `"colmap"`. If you
     remove the preset before renaming, you lose a reference that needed updating.
   - If you remove `default="all"` before fixing all the stage names, it's harder
     to test the pipeline end-to-end during development.
   - Safest: update the "all" preset to use `"matchmove_camera"` in Phase 2,
     then delete the whole "all" mechanism in Phase 3.

## Commit Strategy

Each phase should be **one atomic commit** (or a small series within a branch).
This lets us bisect if something breaks.

```
commit 1: ProPainter deprecation (all 7 phases from that roadmap)
commit 2: COLMAP → mmcam rename (all 9 phases from that roadmap)
commit 3: Remove -s all (all 5 phases from that roadmap)
commit 4: Remove breadcrumbs (all three roadmaps' breadcrumb removal checklists)
```

## Test Baseline (pre-refactor)

Captured on `claude/deprecate-propainter-refactor-4XPBZ` at commit `1f3d797`:

```
Existing suite:  433 passed, 3 failed (pre-existing), 7 skipped
Refactor suite:  1 passed, 38 failed (expected — these are the refactor targets)
```

Pre-existing failures (not ours, do not fix during this refactor):
- `test_platform_manager.py::TestToolDownloads::test_colmap_has_linux_url`
- `test_platform_manager.py::TestInstallToolValidation::test_force_bypasses_existing_check`
- `test_run_colmap.py::TestCheckColmapAvailable::test_colmap_found`

### After each phase, the target is:

| Phase | Existing suite | Refactor suite |
|-------|---------------|----------------|
| 1 (ProPainter) | 433 pass, 3 fail, 7 skip | ~13 pass, 26 fail |
| 2 (COLMAP rename) | 433 pass, 3 fail, 7 skip | ~35 pass, 4 fail |
| 3 (Remove -s all) | 433 pass, 3 fail, 7 skip | 39 pass, 0 fail |
| 4 (Breadcrumb removal) | 433 pass, 3 fail, 7 skip | delete file |

## Cross-Cutting Files (touch in multiple phases)

These files appear in more than one roadmap. Edit carefully.

| File | Phase 1 | Phase 2 | Phase 3 |
|------|---------|---------|---------|
| `scripts/pipeline_constants.py` | description | stage key, STAGE_ORDER | — |
| `scripts/pipeline_config.py` | remove `cleanplate_use_median` | rename `colmap_*` → `mmcam_*` | change `stages` default |
| `scripts/run_pipeline.py` | remove `--cleanplate-median`, remove comfyui_stages | rename `--colmap-*` → `--mmcam-*` | remove `default="all"` |
| `scripts/stage_runners.py` | simplify cleanplate stage | rename colmap stage + paths | — |
| `web/config/pipeline_config.json` | cleanplate: `requiresComfyUI: false` | rename colmap → matchmove_camera | remove "all" preset |
| `docs/reference/cli.md` | remove `--cleanplate-median` | rename `--colmap-*` | remove `--stages all` |
| `docs/reference/stages.md` | update cleanplate section | update colmap section | update stage selection docs |
| `README.md` | remove ProPainter dep | rename colmap refs | — |

## Rollback Plan

Each phase is a single commit. If a phase breaks:

```bash
git revert <commit-hash>   # undo that phase only
```

Phases are designed to be independently revertible because:
- Phase 1 doesn't touch COLMAP code
- Phase 2 doesn't touch cleanplate code or the "all" mechanism (just updates the preset)
- Phase 3 only touches the "all" parsing — no stage logic
