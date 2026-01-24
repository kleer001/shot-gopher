# Documentation Cleanup Roadmap for GitHub Publication

**Status:** Planning
**Goal:** Prepare documentation for public GitHub release following best practices

---

## Current State Analysis

### Inventory Summary
- **34 total documentation files** (~15K lines)
- **5 root-level MD files** (should be 2-3 max)
- **27 files in docs/** (many are internal roadmaps)
- **2 scattered files** (scripts/, web/tests/)

### Key Issues

| Issue | Impact | Priority |
|-------|--------|----------|
| Root directory clutter | Poor first impression | High |
| Internal roadmaps exposed | Confuses users | High |
| Overlapping content | Maintenance burden | Medium |
| Alpha tester doc (temporary) | Outdated quickly | Medium |

---

## GitHub Best Practices Checklist

### Root Directory (Keep Minimal)
Essential files only:
- [ ] `README.md` - Project overview, badges, quick start
- [ ] `LICENSE` - Required for open source
- [ ] `CONTRIBUTING.md` - How to contribute (optional but recommended)
- [ ] `CHANGELOG.md` - Version history (optional)
- [ ] `CLAUDE.md` - LLM guidance (keep for AI-assisted development)

### docs/ Directory Structure

**Flat structure for essentials** - Keep critical docs at top level for immediate visibility:

```
docs/
├── README.md              # Documentation hub / index
│
│   # ESSENTIAL - Before you can start
├── installation.md        # Merged from QUICKSTART + install_wizard
├── docker.md              # Docker setup (from README-DOCKER.md)
├── windows-it-docker.md   # IT admin setup (Docker path)
├── windows-it-native.md   # IT admin setup (native path)
│
│   # ESSENTIAL - Once you start
├── first-project.md       # Your first project walkthrough
├── troubleshooting.md     # When things go wrong
│
│   # REFERENCE - Subdirectories OK for deep dives
├── reference/
│   ├── stages.md          # Pipeline stage details
│   ├── cli.md             # Command-line reference
│   ├── api.md             # REST API documentation
│   ├── scripts.md         # Component scripts
│   └── maintenance.md     # Janitor / system maintenance
└── platforms/
    └── windows.md         # General Windows compatibility + troubleshooting
```

**Rationale:** Users opening `docs/` immediately see what they need. No folder-diving for basics.

---

## Action Items

### Phase 1: Remove Internal/Temporary Docs (High Priority)

These files are internal planning documents that should NOT be in a public release:

| File | Action | Reason |
|------|--------|--------|
| `docs/ATLAS.md` | **Delete or archive** | Internal strategic planning |
| `docs/ROADMAP-1-DOCKER.md` | **Delete or archive** | Completed internal roadmap |
| `docs/ROADMAP-2-API.md` | **Delete or archive** | Internal roadmap |
| `docs/ROADMAP-3-WEB-UI.md` | **Delete or archive** | Internal roadmap |
| `docs/ROADMAP-5-MODAL.md` | **Delete or archive** | Internal roadmap |
| `docs/GVHMR_TRANSITION_ROADMAP.md` | **Delete or archive** | Internal transition plan |
| `scripts/ROADMAP-PIPELINE-REFACTOR.md` | **Delete or archive** | Internal refactoring plan |
| `README_ALPHA_TESTERS.md` | **Delete** | Temporary alpha testing doc |
| `docs/UI-TEST-PLAN.md` | **Delete or move to wiki** | Internal test plan |

**Archive option:** Create a `docs/internal/` directory (add to `.gitignore`) or move to a project wiki.

### Phase 2: Declutter Root Directory (High Priority)

| File | Action | New Location |
|------|--------|--------------|
| `QUICKSTART.md` | **Merge into README.md** | Root (as section) |
| `TESTING.md` | **Move** | `docs/contributing/testing.md` |
| `README_ALPHA_TESTERS.md` | **Delete** | N/A (temporary doc) |

**Result:** Root contains only `README.md`, `LICENSE`, `CONTRIBUTING.md`, `CLAUDE.md`

### Phase 3: Consolidate Redundant Docs (Medium Priority)

| Files to Merge | New File | Notes |
|----------------|----------|-------|
| `QUICKSTART.md` + `install_wizard.md` | `docs/installation.md` | Single source of truth (top-level) |
| `windows-compatibility.md` + `windows-troubleshooting.md` | `docs/platforms/windows.md` | General Windows content (subdirectory OK) |
| `troubleshooting.md` (general) | `docs/troubleshooting.md` | Keep at top level - essential |

**Keep at Top Level - IT Admin Docs:**
- `windows_for_it_dept_docker.md` → `docs/windows-it-docker.md`
- `windows_for_it_dept_native.md` → `docs/windows-it-native.md`

These serve corporate users without admin access. Essential "hand to IT" one-pagers:
- One-time admin setup (WSL2, Docker, CUDA, registry settings)
- Users operate independently afterward without elevated privileges

### Phase 4: Rename for Clarity (Low Priority)

| Current Name | New Location | Reason |
|--------------|--------------|--------|
| `your_first_project.md` | `docs/first-project.md` | Top-level essential |
| `README-DOCKER.md` | `docs/docker.md` | Top-level essential |
| `API-USAGE.md` | `docs/reference/api.md` | Reference material |
| `stages.md` | `docs/reference/stages.md` | Reference material |
| `run_pipeline.md` | `docs/reference/cli.md` | Reference material |
| `component_scripts.md` | `docs/reference/scripts.md` | Reference material |
| `janitor.md` | `docs/reference/maintenance.md` | Reference material |
| `searching_for_COLMAP.md` | Consider deletion | Development notes |

### Phase 5: Create Missing Standard Files (Low Priority)

| File | Purpose | Priority |
|------|---------|----------|
| `CONTRIBUTING.md` | Contribution guidelines | Recommended |
| `CHANGELOG.md` | Version history | Optional |
| `SECURITY.md` | Security policy | Optional (for mature projects) |
| `CODE_OF_CONDUCT.md` | Community guidelines | Optional |

---

## Content Consolidation Plan

### README.md Restructure

Current README.md is comprehensive but could be streamlined. Recommended structure:

```markdown
# Shot Gopher

[Badges: build status, license, Python version]

One-paragraph description of what it does.

## Quick Start
[3-5 lines to get running - link to full installation docs]

## Features
[Bullet list with links to detailed docs]

## Requirements
[Brief list, link to detailed requirements]

## Documentation
[Links to docs/ structure]

## Contributing
[Brief note, link to CONTRIBUTING.md]

## License
[License statement]
```

### What to Keep in README vs Link Out

| Content | README.md | Linked Doc |
|---------|-----------|------------|
| Project overview | Full | - |
| Quick install (1-liner) | Full | - |
| Full installation | Link only | `docs/installation.md` |
| Feature list | Bullets | Detailed in `docs/reference/stages.md` |
| Requirements | Brief | Full in `docs/installation.md` |
| Shot compatibility matrix | Link only | Move to user-guide |

---

## Files to Preserve As-Is

These files are well-structured and user-facing:

- `docs/troubleshooting.md` - Comprehensive, useful
- `docs/stages.md` - Good reference
- `docs/interactive_segmentation.md` - Feature documentation
- `docs/LICENSE_AUDIT_REPORT.md` - Important for legal compliance
- `docs/ACCESSIBILITY.md` - Shows project values
- `docs/OS_SUPPORT_ANALYSIS.md` - Useful platform reference
- `web/tests/README.md` - Appropriate location for test docs

---

## Implementation Order

1. **Delete/archive internal roadmaps** (ATLAS, ROADMAP-*, UI-TEST-PLAN)
2. **Delete alpha tester doc** (README_ALPHA_TESTERS.md)
3. **Move TESTING.md** to docs/ (top-level or reference/)
4. **Merge QUICKSTART.md** into README.md, then delete
5. **Consolidate Windows user docs** into `platforms/windows.md`
6. **Create reference/ and platforms/ subdirectories** (essentials stay flat)
7. **Rename files** for consistency
8. **Create CONTRIBUTING.md** if desired
9. **Final review** of README.md content

---

## Post-Cleanup Validation

- [ ] All links in docs work (no broken references)
- [ ] README.md renders correctly on GitHub
- [ ] No internal/planning docs exposed
- [ ] Clear path from README to detailed docs
- [ ] Installation instructions tested and accurate
- [ ] No duplicate content between files

---

## Estimated Final State

```
shot-gopher/
├── README.md              # Streamlined overview + quick start
├── LICENSE
├── CONTRIBUTING.md        # New
├── CLAUDE.md              # Keep for AI development
└── docs/
    ├── README.md          # Documentation hub / index
    │
    │   # Top-level essentials (no folder diving)
    ├── installation.md    # Before you start
    ├── docker.md          # Before you start (Docker path)
    ├── first-project.md   # Once you start
    ├── troubleshooting.md # When things go wrong
    ├── windows-it-docker.md   # IT admin handoff
    ├── windows-it-native.md   # IT admin handoff
    │
    │   # Subdirectories for deep reference
    ├── reference/
    │   ├── stages.md
    │   ├── cli.md
    │   ├── api.md
    │   ├── scripts.md
    │   ├── maintenance.md
    │   └── interactive-segmentation.md
    └── platforms/
        └── windows.md     # General Windows info
```

**Result:**
- Root: 4 files (down from 5)
- docs/: ~17 files (down from 27)
- Total: ~21 files (down from 34)
- **38% reduction** in documentation files while preserving all user-facing content

---

**Created:** 2026-01-24
**For:** GitHub publication preparation
