# Janitor Tool Documentation

Maintenance, diagnostics, and update management for VFX pipeline installations.

## Overview

The janitor tool keeps your VFX pipeline installation healthy by:

- **Health checks** - Verify conda, git repos, checkpoints, and validation tests
- **Update management** - Check for and apply git repository updates
- **Cleanup** - Remove temporary files and cached data
- **Repair** - Re-download missing checkpoints and fix broken components
- **Reporting** - Generate detailed status reports with disk usage

Think of it as `apt update && apt upgrade` for your VFX pipeline.

## Quick Start

### Run All Maintenance

```bash
python scripts/janitor.py -a
```

Runs: health check, update check, cleanup, and generates report.

### Quick Health Check

```bash
python scripts/janitor.py -H
```

### Update All Components

```bash
python scripts/janitor.py -u -y
```

The `-y` flag auto-confirms updates.

## Command Line Options

### Short Options

```bash
-H             # Health check
-u             # Update check and apply
-c             # Cleanup temporary files
-r             # Repair broken components
-R             # Generate status report
-a             # All operations (health + update + clean + report)

# Options
-i DIR         # Override installation directory
-d             # Dry run (preview cleanup without deleting)
-y             # Auto-confirm all prompts
```

### Long Options

```bash
--health           # Run health check
--update           # Check for and apply updates
--clean            # Clean up temporary files
--repair           # Repair broken components
--report           # Generate detailed status report
--all              # Run all operations

# Options
--install-dir DIR  # Override installation directory
--dry-run          # Preview cleanup without deleting (for --clean)
--yes              # Auto-confirm all prompts
```

## Operations

### Health Check (`-H`)

Comprehensive system verification:

```bash
python scripts/janitor.py -H
```

**Checks**:

1. **Installation Directory**
   - Verifies `.vfx_pipeline/` exists
   - Shows path

2. **Conda Environment**
   - Detects conda/mamba
   - Verifies `vfx-pipeline` environment exists

3. **Git Repositories**
   - GVHMR, ComfyUI
   - Checks working directory is clean
   - Shows current commit hash

4. **Checkpoints**
   - Validates checkpoint files exist
   - Verifies file sizes

5. **Validation Tests**
   - Python imports (torch, numpy, cv2, etc.)
   - CUDA availability
   - Component integration tests

**Example Output**:

```
============================================================
HEALTH CHECK
============================================================

[Installation Directory]
✓ Found: /home/user/shot-gopher/.vfx_pipeline

[Conda Environment]
✓ Conda detected: conda
✓ Environment 'vfx-pipeline' exists

[Git Repositories]
✓ GVHMR: Clean (b4d2e1a)
⚠ ComfyUI: Uncommitted changes

[Checkpoints]
✓ GVHMR: Found

[Validation Tests]
✓ PyTorch with CUDA
✓ Core dependencies
✓ COLMAP installed
✓ Motion capture ready

============================================================
⚠ HEALTH CHECK: Some issues detected
============================================================
```

**Exit codes**:
- `0` - All checks passed
- `1` - Some issues detected

### Update Management (`-u`)

Check for and apply updates to git repositories:

```bash
python scripts/janitor.py -u
```

**Process**:

1. **Fetch latest** from all repositories
2. **Compare** local and remote commits
3. **Show** what needs updating
4. **Prompt** for confirmation (use `-y` to skip)
5. **Pull** updates for each repository

**Example Output**:

```
============================================================
UPDATE CHECK
============================================================

[GVHMR] Checking for updates...
✓ → Up to date

[ComfyUI] Checking for updates...
⚠ → 12 commit(s) behind

1 component(s) have updates available:
  - ComfyUI: 12 commit(s) behind

Apply updates? [y/N]: y

============================================================
APPLYING UPDATES
============================================================

[ComfyUI] Updating...
✓ → Updated successfully
```

**Safety**:
- Only updates repositories with clean working directories
- Skips repositories with uncommitted changes
- Shows what will be updated before proceeding

**Auto-confirm**:

```bash
python scripts/janitor.py -u -y
```

Useful for automated maintenance scripts.

### Cleanup (`-c`)

Remove temporary files and cached data:

```bash
python scripts/janitor.py -c
```

**Removes**:

- `**/*.tmp` - Temporary files
- `**/*.pyc` - Python bytecode
- `**/__pycache__/` - Python cache directories
- `**/temp_*` - Temporary files with temp prefix
- `**/.DS_Store` - macOS metadata files

**Example Output**:

```
============================================================
CLEANUP
============================================================

[Scanning for temporary files...]

Found 47 temporary file(s) (234.56 MB):
  - GVHMR/__pycache__/utils.cpython-310.pyc
  - WHAM/__pycache__/tracker.cpython-310.pyc
  - ComfyUI/custom_nodes/__pycache__/
  ... and 44 more

Delete these files? [y/N]: y

✓ Deleted 47 file(s), freed 234.56 MB
```

**Dry run** (preview without deleting):

```bash
python scripts/janitor.py -c -d
```

Shows what would be deleted without actually deleting:

```
============================================================
CLEANUP (DRY RUN)
============================================================

Found 47 temporary file(s) (234.56 MB):
  - GVHMR/__pycache__/utils.cpython-310.pyc
  ...

⚠ Dry run - no files deleted
```

### Repair (`-r`)

Fix broken components:

```bash
python scripts/janitor.py -r
```

**Repairs**:

1. **Missing conda environment** - Recreates if deleted
2. **Missing checkpoints** - Re-downloads from source
3. **Corrupted installations** - Reinstalls broken components

**Example Output**:

```
============================================================
REPAIR
============================================================

[Checking for issues...]

Found 1 issue(s):
  - GVHMR checkpoint missing

Attempt repairs? [Y/n]: y

[Applying repairs...]

Repairing: GVHMR checkpoint missing
  → Checkpoint downloaded
```

**Auto-confirm**:

```bash
python scripts/janitor.py -r -y
```

**Note**: Repair operations may take time (checkpoint downloads are large).

### Status Report (`-R`)

Generate detailed installation status:

```bash
python scripts/janitor.py -R
```

**Report includes**:

1. **Disk Usage** - Size breakdown by component
2. **Component Status** - Installation state
3. **Repository Status** - Git commits and cleanliness
4. **Conda Environment** - Environment info

**Example Output**:

```
============================================================
VFX PIPELINE STATUS REPORT
============================================================
Generated: 2026-01-12 14:30:00
Install directory: /home/user/shot-gopher/.vfx_pipeline
============================================================

[Disk Usage]
  ComfyUI              8.45 GB
  GVHMR                4.00 GB
  State files         12.34 KB
  Config files         3.45 KB
  ----------------------------------
  TOTAL               12.45 GB

[Components]
  pytorch              completed
  colmap               completed
  gvhmr                completed
  comfyui              completed

[Git Repositories]
  GVHMR                 b4d2e1a ✓
  ComfyUI               f4d8a1b ✗

[Conda Environment]
  Conda: conda
  Environment: vfx-pipeline (exists)

============================================================
```

**Use cases**:
- Check disk usage before adding more components
- Verify installation status
- Audit git repository versions
- Generate reports for bug reports

### All Operations (`-a`)

Run everything at once:

```bash
python scripts/janitor.py -a
```

**Executes** (in order):
1. Health check
2. Update check and apply
3. Cleanup temporary files
4. Status report

**Example**:

```bash
python scripts/janitor.py -a -y
```

Good for weekly maintenance or CI/CD pipelines.

## Usage Examples

### Example 1: Daily Quick Check

```bash
python scripts/janitor.py -H
```

Takes ~5 seconds, verifies everything is working.

### Example 2: Weekly Maintenance

```bash
python scripts/janitor.py -u -y -c
```

Updates components, cleans temp files.

### Example 3: Pre-Project Check

Before starting a new VFX project:

```bash
python scripts/janitor.py -H -R
```

Health check + status report to verify everything is ready.

### Example 4: Post-Installation Validation

After running install wizard:

```bash
python scripts/janitor.py -H
```

Verify installation succeeded.

### Example 5: Troubleshooting

When something isn't working:

```bash
# Check what's broken
python scripts/janitor.py -H

# Try to fix it
python scripts/janitor.py -r -y

# Verify fix worked
python scripts/janitor.py -H
```

### Example 6: Disk Space Recovery

```bash
# Preview cleanup
python scripts/janitor.py -c -d

# Actual cleanup
python scripts/janitor.py -c
```

### Example 7: CI/CD Integration

```bash
#!/bin/bash
# Automated maintenance script

# Update everything
python scripts/janitor.py -u -y || exit 1

# Verify health
python scripts/janitor.py -H || exit 1

# Generate report
python scripts/janitor.py -R > maintenance_report.txt
```

## Advanced Usage

### Custom Installation Directory

If you installed to a non-default location:

```bash
python scripts/janitor.py -H -i /mnt/storage/.vfx_pipeline
```

### Scripted Maintenance

Weekly maintenance cron job:

```bash
# Add to crontab: crontab -e
0 2 * * 0 cd /home/user/shot-gopher && python scripts/janitor.py -a -y
```

Runs every Sunday at 2 AM.

### Selective Updates

Update only specific repositories:

Currently janitor updates all or none. For selective updates, use git directly:

```bash
cd .vfx_pipeline/GVHMR
git pull

cd ../ComfyUI
git pull
```

Then verify with janitor:

```bash
python scripts/janitor.py -H
```

### Report to File

Save status report:

```bash
python scripts/janitor.py -R > status_$(date +%Y%m%d).txt
```

Creates: `status_20260112.txt`

## Disk Usage Analysis

Janitor tracks disk usage by component:

| Component | Typical Size | Notes |
|-----------|--------------|-------|
| GVHMR | 4-5 GB | Includes checkpoints (~4.0 GB) |
| ComfyUI | 5-10 GB | Depends on custom nodes |
| State/Config | < 1 MB | Negligible |

**Total installation**: 10-15 GB

**View breakdown**:

```bash
python scripts/janitor.py -R
```

## Troubleshooting

### "Installation directory not found"

Janitor looks for `.vfx_pipeline/` in repository root.

If installed elsewhere:

```bash
python scripts/janitor.py -H -i /path/to/.vfx_pipeline
```

Or re-run install wizard:

```bash
python scripts/install_wizard.py
```

### "Conda not detected"

Ensure conda is in PATH:

```bash
which conda
```

If not found, activate conda (adjust path to your conda installation):

```bash
source /path/to/miniconda3/etc/profile.d/conda.sh
```

### "Git repository has uncommitted changes"

Updates are skipped for dirty repositories.

**View changes**:

```bash
cd .vfx_pipeline/ComfyUI
git status
```

**Commit or discard**:

```bash
# Commit changes
git add .
git commit -m "Custom modifications"

# Or discard changes
git reset --hard HEAD
```

Then re-run janitor:

```bash
python scripts/janitor.py -u
```

### "Checkpoint download failed"

Network issues or outdated URLs.

**Retry**:

```bash
python scripts/janitor.py -r
```

**Manual download**:

If repair fails, manually download and place in:
- GVHMR: `.vfx_pipeline/GVHMR/checkpoints/`

### "Permission denied" during cleanup

Files may be in use or require elevated permissions.

**Stop running processes**:

```bash
# Stop ComfyUI if running
pkill -f "python.*main.py"

# Re-run cleanup
python scripts/janitor.py -c
```

## Integration with Other Tools

### With Installation Wizard

**After installation**:

```bash
python scripts/install_wizard.py
python scripts/janitor.py -H  # Verify installation
```

**Before reinstall**:

```bash
python scripts/janitor.py -c  # Clean up first
python scripts/install_wizard.py --component gvhmr
```

### With Pipeline

**Before processing**:

```bash
python scripts/janitor.py -H  # Ensure everything works
python scripts/run_pipeline.py footage.mp4 -s all
```

**After long processing**:

```bash
python scripts/janitor.py -c  # Clean temp files
```

### With Version Control

Keep janitor reports in git:

```bash
python scripts/janitor.py -R > docs/installation_status.txt
git add docs/installation_status.txt
git commit -m "Update installation status"
```

## Best Practices

### Regular Maintenance

Run janitor weekly:

```bash
python scripts/janitor.py -a -y
```

Or add to cron.

### Before Important Work

Health check before starting a project:

```bash
python scripts/janitor.py -H
```

Ensures no surprises mid-project.

### After System Updates

After OS or driver updates:

```bash
python scripts/janitor.py -H
```

CUDA versions may change, requiring PyTorch reinstall.

### Disk Space Monitoring

Check disk usage monthly:

```bash
python scripts/janitor.py -R | grep -A 20 "Disk Usage"
```

Plan for expansion if running low.

### Keep Reports

Save monthly reports for trend analysis:

```bash
python scripts/janitor.py -R > reports/status_$(date +%Y%m).txt
```

Track disk usage growth over time.

## Automation Examples

### Daily Health Check

```bash
#!/bin/bash
# daily_check.sh

LOGDIR="/var/log/vfx_pipeline"
mkdir -p "$LOGDIR"

python scripts/janitor.py -H > "$LOGDIR/health_$(date +%Y%m%d).log" 2>&1

if [ $? -ne 0 ]; then
    echo "Health check failed!" | mail -s "VFX Pipeline Alert" admin@example.com
fi
```

Add to cron:

```
0 8 * * * /home/user/shot-gopher/daily_check.sh
```

### Weekly Maintenance

```bash
#!/bin/bash
# weekly_maintenance.sh

cd /home/user/shot-gopher

# Update all components
python scripts/janitor.py -u -y

# Clean temp files
python scripts/janitor.py -c -y

# Generate report
python scripts/janitor.py -R > reports/weekly_$(date +%Y%m%d).txt

# Health check
python scripts/janitor.py -H || {
    echo "Health check failed after maintenance!" | mail -s "VFX Pipeline Alert" admin@example.com
}
```

Add to cron:

```
0 2 * * 0 /home/user/shot-gopher/weekly_maintenance.sh
```

### Pre-Commit Hook

Verify installation before commits:

```bash
#!/bin/bash
# .git/hooks/pre-commit

python scripts/janitor.py -H -q || {
    echo "Health check failed! Fix issues before committing."
    exit 1
}
```

Make executable:

```bash
chmod +x .git/hooks/pre-commit
```

## Related Tools

- **[Installation](../installation.md)** - Initial setup and component installation
- **[CLI Reference](cli.md)** - Use the maintained installation for VFX work

## See Also

- Main documentation: [README.md](../README.md)
- Testing guide: [testing.md](../testing.md)
