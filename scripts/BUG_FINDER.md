# PR Bug Pattern Validation Checklist

A systematic checklist for catching bugs that historically slip through Claude's code reviews. Based on analysis of 200+ commits across 30+ PRs in this repository.

**Usage**: Run this validation loop until ALL checks pass with zero findings. When any check fails, fix the issue and restart from Check 1 (fixes can introduce new issues).

---

## Pre-Validation: Understand the Changes

Before running any checks:

1. Read EVERY modified file completely
2. List all new imports, subprocess calls, and external tool invocations
3. Identify any code that runs in: subprocesses, conda environments, Blender headless, or other isolated contexts
4. Note any mathematical operations, file I/O, or iteration patterns

---

## Tier 1: Most Common Bugs

### Check 1: Environment Isolation

**Pattern**: Claude assumes imports work identically in subprocesses, conda environments, and headless contexts.

For EVERY subprocess call, conda run, or Blender script execution:

- [ ] Is `PYTHONPATH` explicitly set if importing local modules?
- [ ] Is `sys.path` modified if scripts import from non-standard locations?
- [ ] Are conda environment boundaries respected? (tools in env A aren't available in env B)
- [ ] For `shell=True`: are environment variables exported correctly?

**Test**: Trace each import statement in subprocess scripts—would it resolve in that environment?

**Example failure** (3 fix attempts required):
```python
# Bad: assumes GS-IR imports work in subprocess
subprocess.run(["python", "gs_ir/train.py"])

# Good: explicit PYTHONPATH
env = os.environ.copy()
env["PYTHONPATH"] = str(gsir_path)
subprocess.run(["python", "gs_ir/train.py"], env=env)
```

---

### Check 2: API/Library Verification

**Pattern**: Claude invents or assumes API options based on naming conventions without verifying against actual library versions.

For EVERY external library call or CLI tool flag:

- [ ] Verify the function/method EXISTS in the installed version (not just named plausibly)
- [ ] Verify CLI flags exist in the tool version used (check `--help` or docs)
- [ ] Verify package names are correct (e.g., PyPI `alembic` ≠ VFX Alembic)
- [ ] Verify file extensions match what tools actually produce (`.pt` vs `.pkl`, etc.)

**Test**: For each new import, can you run `python -c "from X import Y; print(Y.__doc__)"`?

**Example failure**:
```python
# Bad: SiftMatching.max_num_matches doesn't exist in newer COLMAP
args["SiftMatching.max_num_matches"] = 32768

# Bad: PyPI 'alembic' is a database migration tool, not VFX Alembic
import alembic  # Wrong package entirely!
```

---

### Check 3: Headless/Subprocess Context

**Pattern**: Code works in interactive shells but fails in headless mode.

For ANY code running in Blender, subprocess, or background context:

- [ ] Replace `bpy.context.X` with explicit `scene.X` or passed references
- [ ] Don't assume window managers, UI elements, or interactive state exist
- [ ] Frame handlers and animation systems may need explicit evaluation
- [ ] stdout/stderr may be captured—print statements won't show

**Test**: Would this code work if run via `blender --background --python script.py`?

**Example failure**:
```python
# Bad: bpy.context.collection doesn't exist in headless mode
obj.users_collection.append(bpy.context.collection)

# Good: use scene.collection explicitly
obj.users_collection.append(scene.collection)
```

---

### Check 4: Dependency Completeness

**Pattern**: Claude doesn't trace the full import chain before implementing.

Before adding ANY new import:

- [ ] Trace the full import chain: what does THIS package import?
- [ ] Check `requirements.txt`/`setup.py` of the package for hidden dependencies
- [ ] For ML packages: verify CUDA/PyTorch compatibility
- [ ] For packages with C extensions: verify they install on all platforms

**Test**: Fresh venv install with only your requirements—does it import cleanly?

**Example failure** (5 separate dependency fixes for GVHMR):
```
Fix 1: add colorlog
Fix 2: add pytorch_lightning
Fix 3: add opencv-python
Fix 4: add chumpy (installed separately to prevent pip rollback)
Fix 5: cross-platform PyTorch CUDA installation
```

---

### Check 5: Cross-Platform Behavior

**Pattern**: Claude assumes Linux/Mac behavior is universal.

For ALL file operations, paths, subprocesses, and encoding:

- [ ] Use `pathlib.Path`, never string concatenation for paths
- [ ] Specify `encoding='utf-8'` for ALL `file open()` calls
- [ ] Use `subprocess` with list args, not shell strings with path interpolation
- [ ] Check if tools exist on Windows (no snap, different conda, etc.)
- [ ] Temp directories: use `tempfile` module, don't hardcode `/tmp`

**Test**: Replace all `/` with `\\` mentally—does the code still work?

**Example failure**:
```python
# Bad: hardcoded Unix path separator
path = base_dir + "/output/" + filename

# Good: pathlib handles platform differences
path = base_dir / "output" / filename
```

---

### Check 6: Test Synchronization

**Pattern**: API changes aren't propagated to test mocks and assertions.

After ANY API change:

- [ ] Update ALL test mocks to match new signatures/return values
- [ ] Update ALL test assertions for new fields or changed formats
- [ ] Run the actual tests—don't assume they pass

**Test**: `grep` for the function name in `tests/` and verify each usage matches.

**Example failure**:
```python
# API added new field: vram_available_gb
# Tests still expected old format:
assert result == {"name": "Unknown", "vram_gb": 0}  # Missing new field!

# Fixed:
assert result == {"name": "Unknown", "vram_gb": 0, "vram_available_gb": 0}
```

---

## Tier 2: Second Most Common Bugs

### Check 7: Output File Verification

**Pattern**: Claude assumes output files will have specific names/extensions based on documentation.

For EVERY tool/subprocess that produces output files:

- [ ] Verify actual output filename (run the tool and check, don't assume)
- [ ] Verify actual file extension produced (not what docs say)
- [ ] Check if output is written synchronously or asynchronously
- [ ] If checking file existence, verify timing—does it exist YET?

**Test**: After implementing, run the tool once manually and `ls -la` the output directory.

**Example failure**:
```python
# Bad: assumed GVHMR outputs .pkl files
output_file = output_dir / "motion.pkl"

# Reality: GVHMR outputs .pt files
output_files = list(output_dir.rglob("hmr4d*.pt"))
```

---

### Check 8: No Silent Fallbacks

**Pattern**: Claude adds "defensive" fallbacks that hide the actual failure cause.

For EVERY `try/except`, `if/else` fallback, or `or default` pattern:

- [ ] Is this fallback REQUESTED or just "defensive"?
- [ ] Does the fallback hide information needed for debugging?
- [ ] Could the fallback find the WRONG resource (different tool, wrong version)?
- [ ] If something should exist and doesn't, FAIL—don't silently substitute

**Test**: Remove each fallback mentally. Would failure be clearer? If yes, remove it.

**Example failure**:
```python
# Bad: fallback defeats snap detection, finds wrong COLMAP
def get_colmap_executable():
    found = PlatformManager.find_tool("colmap")
    if found:
        return str(found)
    else:
        return "colmap"  # This finds snap COLMAP via PATH!

# Good: fail explicitly, let caller handle
def get_colmap_executable() -> Optional[str]:
    found = PlatformManager.find_tool("colmap")
    return str(found) if found else None
```

---

### Check 9: Mathematical Formula Verification

**Pattern**: Claude produces math that looks correct but has wrong indices, signs, or composition order.

For ANY mathematical operations (transforms, matrices, coordinates):

- [ ] Derive the formula step-by-step—don't copy from memory
- [ ] For rotation matrices: verify composition order matches extraction order
- [ ] For coordinate systems: verify axis conventions (Y-up vs Z-up, handedness)
- [ ] Test with known values (90° rotations, identity transforms)
- [ ] Check edge cases (gimbal lock at ±90°, zero denominators)

**Test**: Create a roundtrip test: `input → transform → inverse → should equal input`.

**Example failure**:
```python
# Bad: extraction formulas didn't match composition order for zxy
# Took careful derivation to get correct indices
x = np.arctan2(-rotation[1, 2], cx)
z = np.arctan2(rotation[1, 0], rotation[1, 1])
y = np.arctan2(rotation[0, 2], rotation[2, 2])
```

---

### Check 10: Iteration Variable Tracking

**Pattern**: When Claude samples/filters a list mid-function, it loses track of which variable to use at the end.

When filtering, sampling, or subsetting collections mid-function:

- [ ] Name variables explicitly: `all_frames`, `sampled_frames`, `output_frames`
- [ ] At each loop, ask: "which collection should this iterate?"
- [ ] Before returning/writing, verify you're using the correct variable
- [ ] Check if subset variable shadowed the original intent

**Test**: Add a comment at each loop: `# Iterating over: [X] (N items) because [reason]`

**Example failure**:
```python
# Bad: lost track of which frames to write
source_frames = get_all_frames()
if sample_count > 0:
    source_frames = sample(source_frames)  # Shadowed!
# ... processing ...
for src_path in source_frames:  # Bug: only writes sampled frames!
    write_output(src_path)

# Good: explicit variable names
all_source_frames = get_all_frames()
frames_to_process = all_source_frames
if sample_count > 0:
    frames_to_process = sample(all_source_frames)
# ... processing with frames_to_process ...
for src_path in all_source_frames:  # Write ALL frames
    write_output(src_path)
```

---

### Check 11: Full Error Output Preservation

**Pattern**: Claude truncates error output to be "helpful," hiding the actual traceback.

For ALL subprocess calls and error handling:

- [ ] Never truncate stderr or stdout (no `[:500]`, no `[-2000:]`)
- [ ] Always use `capture_output=True` or explicit `stderr=PIPE, stdout=PIPE`
- [ ] Print BOTH stdout AND stderr on failure
- [ ] Don't summarize—let the user see the full traceback

**Test**: Intentionally break something and verify the full error is visible.

**Example failure**:
```python
# Bad: truncates the actual error
if result.stderr:
    print(result.stderr[:500], file=sys.stderr)  # Traceback cut off!

# Bad: hides stdout which often contains the real error
result = subprocess.run(cmd, capture_output=False)  # Can't see output!

# Good: show everything
if result.returncode != 0:
    if result.stdout:
        print(result.stdout, file=sys.stderr)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
```

---

### Check 12: Temporary File Lifecycle

**Pattern**: Claude creates temp files but forgets to clean them up.

For ANY temporary files or directories created:

- [ ] Track creation location explicitly
- [ ] Ensure cleanup in `finally` block or context manager
- [ ] Don't leave temp files on success OR failure
- [ ] If cleanup is conditional, document why

**Test**: Run the function, then check for orphaned temp files in `/tmp` or working directory.

**Example failure**:
```python
# Bad: temp masks left behind after reconstruction
temp_mask_dir = colmap_dir / "temp_masks"
shutil.copytree(mask_dir, temp_mask_dir)
run_reconstruction()
# Forgot cleanup!

# Good: cleanup after use
temp_mask_dir = colmap_dir / "temp_masks"
try:
    shutil.copytree(mask_dir, temp_mask_dir)
    run_reconstruction()
finally:
    if temp_mask_dir.exists():
        shutil.rmtree(temp_mask_dir)
```

---

## Loop Exit Criteria

All of the following must be true:

- [ ] All 12 checks pass with zero findings
- [ ] Tests pass: `pytest tests/`
- [ ] Type check passes: `mypy --strict` (if applicable)
- [ ] Linter passes: `pylint scripts/`
- [ ] No new imports without verified existence
- [ ] No subprocess calls without explicit environment setup
- [ ] No subprocess calls without full output capture
- [ ] No mathematical formulas without roundtrip tests
- [ ] No fallbacks without explicit user request
- [ ] Manual test of at least one happy path executed

---

## If Any Check Fails

1. Fix the issue
2. Document what was wrong in the commit message
3. **Restart from Check 1** (fixes can introduce new issues)
4. Continue until the loop completes with zero findings

---

## Quick Reference: Red Flags

Patterns that should trigger immediate scrutiny:

| Pattern | Likely Bug |
|---------|-----------|
| `subprocess.run([...])` without `env=` | Check 1: Environment isolation |
| New `import X` statement | Check 2: Verify package exists; Check 4: Dependency chain |
| `bpy.context.` | Check 3: Headless compatibility |
| Any file path as string | Check 5: Cross-platform |
| `try: ... except: ...` | Check 8: Silent fallback |
| Matrix/rotation code | Check 9: Math verification |
| `list = filtered_list` mid-function | Check 10: Variable shadowing |
| `output[:500]` or `[-N:]` | Check 11: Error truncation |
| `Path(...) / "temp"` | Check 12: Cleanup lifecycle |

---

**Version**: 1.0
**Based on**: Analysis of shot-gopher repository commits through 2026-02-05
**Maintained by**: Project Team
