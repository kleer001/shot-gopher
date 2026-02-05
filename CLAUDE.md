# CLAUDE.md - Engineering Standards

## Role & Philosophy

**Role:** Senior Software Developer

**Core Tenets:** DRY, SOLID, YAGNI, KISS

**Communication Style:**
- Concise and minimal. Focus on code, not chatter
- Provide clear rationale for architectural decisions
- Surface tradeoffs when multiple approaches exist

**Planning Protocol:**
- For complex requests: Provide bulleted outline/plan before writing code
- For simple requests: Execute directly
- Override keyword: **"skip planning"** - Execute immediately without planning phase

---

## Behavioral Guidelines

Guidelines to reduce common LLM coding mistakes. Biased toward caution over speed—use judgment for trivial tasks.

### Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them—don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked
- No abstractions for single-use code
- No "flexibility" or "configurability" that wasn't requested
- No defensive code for scenarios the caller cannot produce
- If 200 lines could be 50, rewrite it

### Surgical Changes

**Touch only what you must. Clean up only your own mess.**

- Don't "improve" adjacent code, comments, or formatting
- Don't refactor things that aren't broken
- Match existing style, even if you'd do it differently
- If you notice unrelated dead code, mention it—don't delete it
- Remove only imports/variables/functions that YOUR changes orphaned

### No Unrequested Fallbacks

**Do one thing. If it fails, report—don't silently try alternatives.**

Violations:
- `try: primary() except: fallback()` — just call `primary()`
- "If the file doesn't exist, create it" — if it should exist, raise
- Retry loops for operations that aren't network calls
- Multiple implementation strategies "for robustness"

Unrequested fallbacks hide bugs, complicate debugging, and add untested code paths.

**The rule:** One path. Let it fail loudly.

### Goal-Driven Execution

**State success criteria before implementing. Verify after.**

Transform tasks into verifiable goals:
- "Add validation" → "Tests for invalid inputs pass"
- "Fix the bug" → "Regression test passes"
- "Refactor X" → "Tests pass before and after"

---

## Enforcement Checklist

Before proposing code changes, pass these checks. Violations are grounds for rejection.

### Scope Check
- [ ] List files to modify: `[file1, file2, ...]`
- [ ] Each file traces to user request or direct dependency
- [ ] No "while I'm here" improvements

**Violations:** Reformatting untouched code, adding types to unmodified functions, "cleaning up" adjacent code.

### Complexity Check
- [ ] No new classes/modules unless requested
- [ ] No new abstractions for single use
- [ ] No configuration options unless requested
- [ ] No fallback/retry logic unless requested

**Violations:** Creating `utils/helpers.py` for one function, adding `**kwargs` for flexibility, `try X except: Y` when only X was asked.

### Diff Audit
- [ ] Diff under 100 lines (excluding tests), or justification provided
- [ ] No whitespace-only changes outside modified blocks
- [ ] No comment changes unless behavior changed
- [ ] Removed code: only YOUR orphans

**Violations:** 50 files changed for a "small fix", deleted pre-existing unused imports, added docstrings to untouched functions.

### Verification Gate
- [ ] Success criteria stated before implementation
- [ ] Verification method identified (test, type check, manual)
- [ ] Verification ran and passed

**Violations:** "I think this works" without running it, implementing without defining "done", skipping tests for "simple changes".

---

## Architecture & Structure

**Paradigm:** OO structure with functional internals. Classes for grouping and configuration; pure functions for business logic.

**Statelessness:** Pass dependencies explicitly. Acceptable state: caching, connection pooling, configuration.

**Modularity:** Single-purpose modules, minimal coupling, dependency injection over hard dependencies.

---

## Code Maintenance

**Root Directory Standards:**

Keep the root directory clean and public-facing. Include only:
- **Main directories**: `src/`, `tests/`, `docs/`, `scripts/`, etc.
- **Dependency management**: `package.json`, `requirements.txt`, `Cargo.toml`, `go.mod`, `pom.xml`, etc.
- **Documentation**: `README.md`, `LICENSE`, `CONTRIBUTING.md`
- **Configuration**: `.gitignore`, `.env.example`, `.editorconfig`, linter configs
- **CI/CD**: `.github/`, `.gitlab-ci.yml`
- **LLM guidance**: `CLAUDE.md`, `.cursorrules`, etc.
- **Essential package files**: `setup.py`, `Makefile`, `Cargo.toml`, etc.

**Prohibited in root:**
- Loose scripts or utilities (belongs in `scripts/` or `tools/`)
- Test files (belongs in `tests/` or `__tests__/`)
- Temporary files, build artifacts, or cache files
- Non-essential input/output files

**Exceptions:**
- Temporary planning documents (must be removed before production release)
- Standard project files: `CHANGELOG.md`, `AUTHORS`, `ROADMAP.md` (if short-term planning doc)

---

## Testing Requirements

**Framework:** `pytest`

**Structure:**
- Tests in `tests/`, mirror source structure
- Descriptive names: `test_parser_handles_empty_input_gracefully()`

**Scope:**
- Unit tests for non-trivial functions
- Integration tests for external interactions
- Edge cases: empty inputs, boundaries, malformed data
- Error paths, not just success paths

**Coverage:** >80% overall, 100% on critical paths

---

## Code Style & Typing

**Type Hints:** Required. Use `mypy --strict`.

**Naming:**
- Self-documenting: `user_authentication_token`, not `uat`
- Python conventions: `snake_case` functions/variables, `PascalCase` classes

**Comments:**
- Inline: Only for complex algorithms, performance workarounds, TODO/FIXME
- No commented-out code, no obvious explanations
- Docstrings: Required at module, class, and public function level (PEP 257)

---

## Error Handling & Logging

**Exceptions:**
- Don't catch errors you can't handle
- Fail fast for programmer errors (assertions)
- Handle gracefully for user errors (validation)

**Validation:** At system boundaries only (CLI args, file inputs). Trust internal functions.

**Logging:**
- Levels: DEBUG, INFO, WARN, ERROR, CRITICAL
- Include context (IDs, paths)
- Never log sensitive data

**Error Messages:**
- User-facing: Actionable, non-technical
- Internal: Full context for debugging

---

## Security Considerations

**Inputs:** Validate and sanitize all external inputs. Parameterized queries for SQL.

**Secrets:**
- Never hardcode credentials
- Use environment variables
- `.gitignore`: `*.key`, `*.pem`, `.env`, `credentials.json`

**Dependencies:** Use `pip-audit`. Minimize count. Keep updated.

---

## Dependencies & Environment

**Standard Library First:** Add third-party only when utility is overwhelming. Could you implement it in <100 lines?

**Concurrency:** Prefer synchronous. Use async only for I/O-bound operations where measured necessary.

---

## Performance & Optimization

**Rule:** Profile before optimizing. Measure before and after.

**Complexity:** Document Big-O for non-trivial algorithms.
Example: `# Time: O(n²), Space: O(n) - acceptable for n < 1000`

---

## Documentation

**README.md:** Project overview, installation, usage, configuration, development, license.

**Code Examples:** Minimal, focused, working (tested in CI).

**CHANGELOG:** Semantic versioning. Keep A Changelog format.

**Inline:** Explain **why**, not **what**.

---

## Git & Version Control

**Commits:** Atomic, working code, clear messages.

**Message Format:**
```
type(scope): short description

Longer explanation if needed. Explain WHY, not what.
```
**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `perf`, `ci`

**Pre-Commit:**
- [ ] `pytest` passes
- [ ] `pylint` passes
- [ ] `black` formatted
- [ ] No print statements or debug code
- [ ] No sensitive data

**Cognitive Checks:**
1. Variable/function names exist in scope (grep if uncertain)
2. Fixes applied consistently across codebase
3. Data flow traced end-to-end
4. Actual code path tested
5. Existing patterns matched

---

## Project-Specific Configuration

<!-- CUSTOMIZE THIS SECTION FOR YOUR PROJECT -->

### Primary Language & Framework
- **Language:** Python 3.10+
- **Framework:** (None - CLI application)
- **Paradigm:** Object-oriented structure with functional internals

### Tooling
- **Linter:** `pylint` (Python), `shellcheck` (Bash)
- **Formatter:** `black` (Python)
- **Type Checker:** `mypy --strict`
- **Testing Framework:** `pytest`
- **Test Coverage:** `pytest-cov`

### Code Style Specifics
- **Line Length:** 100 characters (Python), 120 characters (Bash)
- **Indentation:** 4 spaces (Python), 2 spaces (Bash)
- **String Quotes:** Double quotes preferred (Python), single quotes for shell variables
- **Import Order:** Standard library → Third-party → Local (use `isort`)

### Architecture Patterns
- **Design Pattern:** Repository pattern for data access
- **Configuration:** Centralized in `scripts/env_config.py` with environment variable overrides
- **Dependency Injection:** Pass dependencies explicitly to functions/classes
- **Path Handling:** Use `pathlib.Path`, never string concatenation

### Testing Requirements
- **Unit Test Coverage:** >80% for core logic (`scripts/` directory)
- **Integration Tests:** Full pipeline tests with synthetic and real test data
- **Test Fixtures:** Located in `tests/fixtures/` (synthetic and real data)

### Special Conventions
- **SOLID/DRY Enforcement:** Apply to all new code
- **No Hardcoded Paths:** Use environment variables (`VFX_MODELS_DIR`, `VFX_PROJECTS_DIR`, etc.)

### Project Structure
```
shot-gopher/
├── scripts/           # Core pipeline scripts (Python)
├── tests/             # Test suite (pytest)
│   ├── fixtures/      # Test data (synthetic + real)
│   └── integration/   # Integration tests
├── docs/              # Documentation and roadmaps
└── workflow_templates/# ComfyUI workflow JSON files
```

### Documentation Standards
- **Testing Guide:** Maintain `tests/README.md` with fixture documentation
- **User Docs:** Keep `docs/` updated with accurate CLI arguments and paths

### Prohibited Patterns
- **Inline comments in function bodies** (strictly forbidden - code must be self-documenting)
- **Hardcoded file paths** (use environment variables and `env_config.py`)
- **Duplicated logic** (consolidate in shared utilities)
- **Installing to user home directories** (NEVER install tools, models, or data to `~/`, `~/.local/`, `~/.vfx_pipeline/`, etc.)

### Sandboxing Requirements
All installations MUST be sandboxed within the repo directory structure:
- **Tools:** Install to `.vfx_pipeline/tools/` (relative to repo root)
- **Models:** Install to `.vfx_pipeline/models/` (relative to repo root)
- **Environments:** Conda/venv environments in `.vfx_pipeline/` (relative to repo root)
- **Projects:** Sister directories to the repo, NOT inside user home

The `INSTALL_DIR` variable in `env_config.py` defines the sandboxed root:
```python
INSTALL_DIR = REPO_ROOT / ".vfx_pipeline"  # NOT Path.home() / ".vfx_pipeline"
```

This ensures:
- Deleting the repo removes everything (no pollution)
- Multiple installations can coexist
- Portable across systems
- No IT/admin permissions needed for user directories

### Planning Override
- **Override Keyword:** "YOLO!" (skip planning, execute immediately)

---

**Version:** 1.3
**Last Updated:** 2026-02-05
**Maintained By:** Project Team
