# Windows Installation Debug Log

## Test Environment
- VM: win11 (actually Windows 10 Home, 10.0.19045)
- SSH: `ssh Username@192.168.122.201`
- Host: DESKTOP-JROGEJ4
- PowerShell: 5.1.19041.6456
- Starting state: Clean install (no git, no conda, no python)

---

## Errors Found & Fixed

### Error 1: CRITICAL — `$ErrorActionPreference = "Stop"` + git stderr
- **Symptom:** `git clone` aborts the entire bootstrap script. Error message is literally "Cloning into 'shot-gopher'..." — a progress message, not an actual error.
- **Root Cause:** Git writes progress output to stderr. PowerShell treats any stderr output as a non-terminating error. With `$ErrorActionPreference = "Stop"` (line 7 of old bootstrap), these become terminating errors that trigger the catch block.
- **Reproduction:** `$ErrorActionPreference = 'Stop'; git clone <url> <path>` → throws exception
- **Fix:** Removed global `$ErrorActionPreference = "Stop"`. Added `Invoke-NativeCommand` wrapper that temporarily sets `$ErrorActionPreference = "Continue"` and routes stderr through `ForEach-Object` to prevent PowerShell from converting it to error records. Uses `$LASTEXITCODE` for actual error detection.
- **Result:** VERIFIED — git clone runs cleanly, progress output displays normally

### Error 2: CRITICAL — conda init creates profile.ps1 under Restricted execution policy
- **Symptom:** After bootstrap installs Miniconda and runs `conda init powershell`, every subsequent PowerShell session shows: `File profile.ps1 cannot be loaded because running scripts is disabled on this system`
- **Root Cause:** Default Windows execution policy is "Restricted" which blocks all .ps1 files. `conda init powershell` creates `$HOME\Documents\WindowsPowerShell\profile.ps1` which then fails to load on every session.
- **Fix:** `Initialize-CondaShell` now checks execution policy and sets it to `RemoteSigned` for `CurrentUser` scope (doesn't require admin) BEFORE running `conda init`. Graceful fallback with manual instructions if the set fails.
- **Result:** VERIFIED — profile.ps1 loads without errors after fix

### Error 3: CRITICAL — `conda run` stderr triggers PowerShell error
- **Symptom:** When install wizard exits non-zero (e.g., disk space check fails), `conda run` writes `ERROR conda.cli.main_run:execute(142)` to stderr. With `$ErrorActionPreference = "Stop"`, this becomes a confusing `NativeCommandError` on top of the actual error message.
- **Root Cause:** Same class of bug as Error 1 — stderr output from native commands becomes PowerShell errors.
- **Fix:** `conda run` now called through `Invoke-NativeCommand` wrapper. Exit code checked via `$LASTEXITCODE`.
- **Result:** VERIFIED — wizard errors display cleanly

### Error 4: MODERATE — `Read-Host "Press Enter to close"` hangs in non-interactive contexts
- **Symptom:** When run via `irm | iex` (recommended install method) or via SSH, the final `Read-Host` hangs forever waiting for input that will never come.
- **Root Cause:** `irm | iex` consumes stdin with the pipe. `Read-Host` blocks on empty/piped stdin.
- **Fix:** Final prompt only shown when `[Environment]::UserInteractive` and `[Console]::IsInputRedirected` indicate a real interactive session.
- **Result:** VERIFIED — script exits cleanly in non-interactive contexts

### Error 5: MODERATE — Invalid directory detected as existing installation
- **Symptom:** If the install directory exists but isn't a valid git repo (e.g., partial cleanup, interrupted clone, empty leftover from killed process), the bootstrap asks "Update existing installation?" and then fails because there's no repo to pull or wizard scripts to run.
- **Root Cause:** Bootstrap only checked `Test-Path $INSTALL_DIR`, not whether it's a valid git repo.
- **Fix:** Now checks for `.git` subdirectory. If directory exists without `.git`, removes it (with retry loop for Windows filesystem delays) and clones fresh.
- **Result:** VERIFIED — handles leftover directories correctly

### Error 6: MODERATE — `Read-Host` prompts unreliable via `irm | iex`
- **Symptom:** When run via `irm ... | iex`, `Read-Host` receives empty string (stdin consumed by pipe). Defaults to "yes" only because empty doesn't match the "no" regex — accidental, not intentional.
- **Root Cause:** `Read-Host` reads from stdin which is the pipe.
- **Fix:** Added `Read-UserInput` function that detects redirected input and falls back to `[Console]::ReadLine()` for direct console access. Returns default value when no console is available.
- **Result:** VERIFIED — prompts work correctly via irm|iex, empty defaults handled explicitly

### Error 7: LOW — Winget source error on fresh Windows
- **Symptom:** `0x8a15000f : Data required by the source is missing` / `Failed when opening source(s)`
- **Root Cause:** Fresh Windows installs may not have winget sources initialized.
- **Fix:** Added `winget source update` call before install attempt. Script already fell back to direct download, so this just reduces noise.
- **Result:** Non-blocking (fallback works)

---

## Summary of Changes

### Files Modified
1. **`scripts/bootstrap_conda.ps1`** — Major rewrite of error handling and interactive input

### Key Changes
- Removed `$ErrorActionPreference = "Stop"` (the root cause of most failures)
- Added `Invoke-NativeCommand` wrapper for git/conda calls (handles stderr cleanly)
- Added `Read-UserInput` function (works via irm|iex, SSH, and interactive console)
- `Initialize-CondaShell` now sets execution policy before `conda init`
- Git clone section validates `.git` directory, not just directory existence
- Final "Press Enter" prompt only shown in truly interactive sessions
- Added winget source update before git install attempt

### Test Results (Final)
- Bootstrap: Git detection ✓, Conda detection ✓, Git clone ✓, Wizard launch ✓
- No PowerShell `NativeCommandError` noise
- No profile.ps1 execution policy errors
- Script exits cleanly in non-interactive contexts
