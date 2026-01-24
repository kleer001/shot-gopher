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
Recommended organization:
```
docs/
├── getting-started/
│   ├── installation.md      # Merged from QUICKSTART + install_wizard
│   ├── first-project.md     # Renamed from your_first_project.md
│   └── docker.md            # Renamed from README-DOCKER.md
├── user-guide/
│   ├── pipeline-stages.md   # Renamed from stages.md
│   ├── cli-reference.md     # Renamed from run_pipeline.md
│   └── maintenance.md       # Renamed from janitor.md
├── reference/
│   ├── api.md               # Renamed from API-USAGE.md
│   ├── scripts.md           # Renamed from component_scripts.md
│   └── troubleshooting.md   # Keep as-is
├── platforms/
│   ├── windows.md           # General Windows users
│   ├── windows-it-docker.md # IT admin setup (Docker/WSL2 path)
│   └── windows-it-native.md # IT admin setup (native path)
└── contributing/
    ├── testing.md           # Moved from root TESTING.md
    └── development.md       # New: development setup
```

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
| `QUICKSTART.md` + `install_wizard.md` | `docs/getting-started/installation.md` | Single source of truth for installation |
| `windows-compatibility.md` + `windows-troubleshooting.md` | `docs/platforms/windows.md` | General Windows user content |
| `troubleshooting.md` (general) | Keep separate | Good standalone reference |

**Keep Separate - IT Admin Docs:**
- `windows_for_it_dept_docker.md` → `docs/platforms/windows-it-docker.md`
- `windows_for_it_dept_native.md` → `docs/platforms/windows-it-native.md`

These serve corporate users without admin access. They're "hand to IT" one-pagers that enable:
- One-time admin setup (WSL2, Docker, CUDA, registry settings)
- Users operate independently afterward without elevated privileges

### Phase 4: Rename for Clarity (Low Priority)

| Current Name | New Name | Reason |
|--------------|----------|--------|
| `your_first_project.md` | `first-project.md` | Kebab-case consistency |
| `README-DOCKER.md` | `docker.md` | Simpler, in getting-started/ |
| `API-USAGE.md` | `api.md` | Simpler naming |
| `stages.md` | `pipeline-stages.md` | More descriptive |
| `run_pipeline.md` | `cli-reference.md` | Describes actual content |
| `component_scripts.md` | `scripts.md` | Simpler |
| `janitor.md` | `maintenance.md` | More intuitive |
| `searching_for_COLMAP.md` | Consider deletion | Appears to be development notes |

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
| Full installation | Link only | `docs/getting-started/installation.md` |
| Feature list | Bullets | Detailed in stages.md |
| Requirements | Brief | Full in installation.md |
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
3. **Move TESTING.md** to docs/contributing/
4. **Merge QUICKSTART.md** into README.md, then delete
5. **Consolidate Windows user docs** (keep IT admin docs separate)
6. **Create directory structure** (getting-started/, user-guide/, etc.)
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
    ├── README.md          # Documentation hub
    ├── getting-started/
    │   ├── installation.md
    │   ├── first-project.md
    │   └── docker.md
    ├── user-guide/
    │   ├── pipeline-stages.md
    │   ├── cli-reference.md
    │   └── maintenance.md
    ├── reference/
    │   ├── api.md
    │   ├── scripts.md
    │   ├── troubleshooting.md
    │   └── interactive-segmentation.md
    ├── platforms/
    │   ├── windows.md
    │   ├── windows-it-docker.md
    │   └── windows-it-native.md
    └── contributing/
        └── testing.md
```

**Result:**
- Root: 4 files (down from 5)
- docs/: ~17 files (down from 27)
- Total: ~21 files (down from 34)
- **38% reduction** in documentation files while preserving all user-facing content

---

**Created:** 2026-01-24
**For:** GitHub publication preparation
