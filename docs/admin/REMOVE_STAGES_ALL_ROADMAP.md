# Remove `-s all` Roadmap

Remove the `--stages all` shorthand. Stages are now buffet-style: users must explicitly
list the stages they want. No `-s` flag (or `-s` with empty value) does nothing and
prints a helpful message.

## Rationale

- Forces users to think about what they actually need
- Prevents accidental long-running pipelines
- Aligns with production reality: nobody runs every stage on every shot
- Simplifies the mental model: stages are a la carte, not a preset

## Decisions

| Question | Answer |
|----------|--------|
| No `-s` flag behavior | Do nothing, print available stages and explain why |
| `-s ""` (empty) behavior | Same as no `-s` flag |
| `STAGE_ORDER` constant | Keep (still defines execution ordering for any selected set) |
| `PipelineConfig.stages` default | Change to empty list |
| Web UI presets | Remove "all" preset, keep "quick" and "full" |

## Files to Modify

### Phase 1: CLI Parsing

| File | Line(s) | Changes Required |
|------|---------|------------------|
| `scripts/run_pipeline.py:205-210` | Remove `default="all"` (line 208) from `--stages` arg, default to `None` |
| `scripts/run_pipeline.py:375-384` | Remove entire `if args.stages.lower() == "all"` / `else` block; replace with early exit when no stages, then flat validation |

### Phase 2: Config Default

| File | Line(s) | Changes Required |
|------|---------|------------------|
| `scripts/pipeline_config.py:26` | Change `stages` default from `STAGE_ORDER.copy()` to empty list `[]` |
| `scripts/pipeline_config.py:75,106` | Update `from_args()` — no longer fall back to `STAGE_ORDER.copy()` |

### Phase 3: Web Config

| File | Changes Required |
|------|------------------|
| `web/config/pipeline_config.json:120-124` | Remove `"all"` preset |

### Phase 4: Web UI

| File | Changes Required |
|------|------------------|
| `web/static/js/controllers/ProjectsController.js:16` | Remove or repurpose `ALL_STAGES` constant (used for UI rendering, may still be needed for stage list display — audit usage) |

### Phase 5: Documentation & Scripts

| File | Changes Required |
|------|------------------|
| `scripts/run_pipeline.py:11` | Update module docstring: remove `--stages all` example |
| `docs/reference/cli.md:58,193,280` | Remove `or 'all'` from help text, remove `--stages all` examples |
| `docs/reference/stages.md` | Update stage selection documentation |
| `docs/testing.md` | Update any `--stages all` test examples |
| `docs/which-stages.md` | Update stage selection guidance |
| `src/run-pipeline.sh:11` | Remove `--stages all` from example in docstring |
| `src/run-pipeline.bat:11` | Remove `--stages all` from example in docstring |
| `scripts/bug_reporter.py:123` | Remove `--stages all` from bug report template example |

## Key Transformation: `main()` in `run_pipeline.py`

**Before:**
```python
parser.add_argument("--stages", "-s", type=str, default="all",
    help=f"Comma-separated stages to run: {','.join(STAGES.keys())} or 'all'")
...
if args.stages.lower() == "all":
    stages = STAGE_ORDER.copy()
else:
    stages = [s.strip() for s in args.stages.split(",")]
```

**After:**
```python
parser.add_argument("--stages", "-s", type=str, default=None,
    help=f"Comma-separated stages to run: {','.join(STAGES.keys())}")
...
if not args.stages:
    print("No stages specified. Use --stages/-s to select stages to run.")
    print()
    print("Available stages:")
    for name, desc in STAGES.items():
        print(f"  {name}: {desc}")
    print()
    print("Example: --stages depth,roto,cleanplate")
    sys.exit(0)

stages = [s.strip() for s in args.stages.split(",")]
invalid = set(stages) - set(STAGES.keys())
if invalid:
    print(f"Error: Invalid stages: {invalid}", file=sys.stderr)
    print(f"Valid stages: {', '.join(STAGE_ORDER)}")
    sys.exit(1)
stages = sanitize_stages(stages)
```

## Edge Cases

- **`--list-stages` still works** — no change needed, it already exits early
- **Web UI** — The web UI has its own preset system. Users select stages via checkboxes,
  so removing the "all" preset just removes one convenience button. The individual stage
  checkboxes remain.
- **`sanitize_stages()`** — No change needed. It still reorders and injects `ingest`.

## Breadcrumbs (temporary debugging aids)

Add during refactor. Remove before merging to main.

### 1. Explicit rejection of "all"

In the new stage validation block, temporarily add a specific error message for "all":

```python
if not args.stages:
    ...  # print usage, exit 0
    sys.exit(0)

stages = [s.strip() for s in args.stages.split(",")]

if "all" in stages:
    print("Error: '--stages all' is no longer supported.", file=sys.stderr)  # BREADCRUMB
    print("Select stages individually: --stages depth,roto,cleanplate", file=sys.stderr)
    sys.exit(1)
```

**Why:** Users (and scripts/aliases) that still pass `-s all` get a clear migration
message instead of `Error: Invalid stages: {'all'}`. Keep this for one release cycle,
then let it fall through to the generic invalid-stage error.

## Verification Checklist

- [ ] `python scripts/run_pipeline.py` (no args) prints stage list and exits cleanly
- [ ] `python scripts/run_pipeline.py input.mp4` (no `-s`) prints stage list and exits cleanly
- [ ] `python scripts/run_pipeline.py input.mp4 -s depth,roto` works as before
- [ ] `python scripts/run_pipeline.py input.mp4 -s all` prints specific migration message
- [ ] `python scripts/run_pipeline.py --list-stages` still works
- [ ] `grep -r '"all"' scripts/run_pipeline.py` returns only the BREADCRUMB migration message
- [ ] All tests pass: `pytest tests/`

## Breadcrumb Removal Checklist

- [ ] Remove explicit `"all"` migration message (let generic invalid-stage error handle it)
