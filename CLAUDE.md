<!--
CLAUDE.md - Engineering Standards Template v1.0
This file provides coding standards and guidelines for AI assistants working on this project.

CUSTOMIZATION INSTRUCTIONS:
1. Review all sections and adjust for your project's language/framework
2. Fill in the "Project-Specific Configuration" section at the bottom
3. Adjust policies (inline comments, planning override) based on team preferences
4. Remove this comment block after customization
-->

# CLAUDE.md - Engineering Standards

## Role & Philosophy

**Role:** Senior Software Developer

**Core Tenets:**
- **DRY** (Don't Repeat Yourself) - Eliminate duplication across codebase
- **SOLID** - Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion
- **YAGNI** (You Aren't Gonna Need It) - Build only what's needed now
- **KISS** (Keep It Simple, Stupid) - Prefer simple solutions over complex ones

**Communication Style:**
- Concise and minimal. Focus on code, not chatter
- Provide clear rationale for architectural decisions
- Surface tradeoffs when multiple approaches exist

**Planning Protocol:**
- For complex requests: Provide bulleted outline/plan before writing code
- For simple requests: Execute directly
- Override keyword: **"skip planning"** - Execute immediately without planning phase

---

## Architecture & Structure

**Paradigm Guidance:**
- Follow the dominant paradigm of the project's primary language
  - **OO languages** (Python, Java, C#): Use classes for logical grouping and configuration encapsulation
  - **Functional languages** (Haskell, Elixir, Clojure): Prefer pure functions with explicit dependencies
  - **Multi-paradigm** (JavaScript, Scala, Rust): Match existing codebase conventions
- Methods/functions should be **stateless where practical**
  - Pass dependencies explicitly (dependency injection)
  - Acceptable stateful patterns: caching, connection pooling, configuration management
  - Prefer pure functions for business logic and data transformations

**Modularity:**
- Organize code into focused modules with clear responsibilities
- Each module should have a single, well-defined purpose
- Minimize coupling between modules; prefer dependency injection over hard dependencies

**Version Compatibility:**
- Use syntax compatible with project dependencies
- Check version constraints in: `package.json`, `requirements.txt`, `go.mod`, `Cargo.toml`, `pom.xml`, etc.
- When in doubt, match the style of existing code

---

## Code Maintenance

**Root Directory Standards:**

Keep the root directory clean and public-facing. Include only:
- **Main directories**: `src/`, `tests/`, `docs/`, `scripts/`, etc.
- **Dependency management**: `package.json`, `requirements.txt`, `Cargo.toml`, `go.mod`, `pom.xml`, etc.
- **Documentation**: `README.md`, `LICENSE`, `CONTRIBUTING.md`
- **Configuration**: `.gitignore`, `.env.example`, `.editorconfig`, linter configs
- **CI/CD**: `.github/`, `.gitlab-ci.yml`, `Dockerfile`, `docker-compose.yml`
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

**Testing Framework:**
- Use project-appropriate framework:
  - Python: `pytest` (preferred) or `unittest`
  - JavaScript/TypeScript: `jest`, `vitest`, or `mocha`
  - Rust: `cargo test`
  - Go: `go test`
  - Java: `JUnit` or `TestNG`

**Test Structure:**
- Separate test files in designated test directories (`tests/`, `__tests__/`, `test/`)
- Mirror source structure: `src/utils/parser.py` → `tests/test_parser.py`
- Use descriptive test names: `test_parser_handles_empty_input_gracefully()`

**Test Scope:**
- **Unit tests**: All non-trivial functions/methods
- **Integration tests**: API endpoints, database interactions, external service calls
- **Edge cases**: Empty inputs, null values, boundary conditions, malformed data
- **Error paths**: Test failure modes, not just success paths

**Fixtures & Setup:**
- Use framework-appropriate fixtures (pytest fixtures, Jest beforeEach, etc.)
- Keep test data minimal and focused
- Prefer synthetic/mock data over real data in unit tests
- Use test databases or in-memory storage for integration tests

**Coverage Goals:**
- Aim for >80% code coverage
- 100% coverage on critical paths (authentication, payment, data integrity)
- Don't test framework code or third-party libraries

---

## Code Style & Typing

**Type Safety:**
- **Mandatory** for statically typed languages (TypeScript, Rust, Go, Java)
- **Strongly recommended** for dynamically typed languages with type hints (Python, PHP)
- Define explicit interfaces/types for all function inputs
- Provide explicit return type hints/annotations
- Use strict type checking modes where available (`mypy --strict`, `tsc --strict`)

**Naming Conventions:**
- **Self-documenting names**: Variable and function names must be verbose and descriptive
  - Good: `user_authentication_token`, `calculate_monthly_revenue()`
  - Bad: `uat`, `calc()`, `do_thing()`
- **Follow language conventions**:
  - Python: `snake_case` for functions/variables, `PascalCase` for classes
  - JavaScript: `camelCase` for functions/variables, `PascalCase` for classes
  - Rust: `snake_case` for functions/variables, `PascalCase` for types
  - Go: `camelCase` (unexported), `PascalCase` (exported)

**Comments Policy:**
- **Inline comments**: Minimize within function/method bodies. Code should be self-explanatory
  - **Permitted inline comments**:
    - Complex algorithms requiring explanation
    - Non-obvious performance optimizations
    - Workarounds for known bugs in dependencies
    - TODO/FIXME/HACK markers with context
  - **Prohibited inline comments**:
    - Explaining what obvious code does
    - Commented-out code (use version control instead)
    - Redundant descriptions (`i++  // increment i`)

- **Docstrings/JSDoc**: Required at module, class, and public function level
  - Include: Purpose, parameters, return values, exceptions/errors, usage examples (for complex APIs)
  - Format: Follow language standards (PEP 257 for Python, JSDoc for JavaScript, rustdoc for Rust)

---

## Error Handling & Logging

**Error Strategy:**
- Use language-appropriate error handling:
  - **Exceptions**: Python, Java, C# (for exceptional conditions only)
  - **Result types**: Rust, Haskell (for expected failures)
  - **Error returns**: Go (for expected failures)
  - **Either/Option types**: Scala, functional languages
- Don't catch errors you can't handle
- Fail fast for programmer errors (assertions, panics)
- Handle gracefully for user errors (validation, retries)

**Input Validation:**
- **Validate at system boundaries**: API endpoints, CLI arguments, file inputs, user forms
- **Trust internal boundaries**: Don't re-validate data passed between internal functions
- **Sanitize external inputs**: Prevent injection attacks (SQL, XSS, command injection)

**Logging:**
- Use appropriate log levels:
  - **DEBUG**: Detailed diagnostic information
  - **INFO**: General informational messages (startup, shutdown, major state changes)
  - **WARN**: Unexpected but recoverable conditions
  - **ERROR**: Error conditions that don't stop execution
  - **CRITICAL/FATAL**: Severe errors requiring immediate attention
- Include context in logs: user ID, request ID, transaction ID
- Never log sensitive data: passwords, tokens, credit cards, PII

**Error Messages:**
- **User-facing errors**: Actionable and non-technical
  - Good: "Email address is invalid. Please check the format."
  - Bad: "ValueError: email regex match failed at line 47"
- **Internal errors**: Include full context for debugging
  - Good: "Failed to connect to database: timeout after 30s (host=db.prod.example.com, port=5432)"
  - Bad: "Database error"

---

## Security Considerations

**Input Validation:**
- Validate and sanitize all external inputs
- Use parameterized queries for SQL (prevent SQL injection)
- Escape user content in templates (prevent XSS)
- Validate file uploads: type, size, content

**Secrets Management:**
- **Never hardcode credentials** in source code
- Use environment variables for configuration
- Use secret management systems for production (AWS Secrets Manager, HashiCorp Vault, etc.)
- Add patterns to `.gitignore`: `*.key`, `*.pem`, `.env`, `credentials.json`

**Dependencies:**
- Check for known vulnerabilities before adding dependencies
- Use tools: `npm audit`, `pip-audit`, `cargo audit`, Snyk, Dependabot
- Keep dependencies updated (security patches)
- Minimize dependency count (smaller attack surface)

**Least Privilege:**
- Run services with minimum required permissions
- Use read-only file systems where possible
- Limit network access to required endpoints only

**Authentication & Authorization:**
- Never trust client-side validation alone
- Validate permissions on every request (server-side)
- Use established libraries (don't roll your own crypto)

---

## Dependencies & Environment

**Standard Library First:**
- Prioritize built-in language features and standard library
- Add third-party dependencies only when utility is overwhelming
- Evaluate tradeoffs: bundle size, maintenance burden, security surface

**Dependency Evaluation Criteria:**
- Is it actively maintained?
- Does it have good documentation?
- What's the security track record?
- How large is the bundle/binary size impact?
- Could we implement this ourselves in <100 lines?

**Concurrency:**
- **Prefer synchronous code** to reduce complexity
- Use async/await **only when necessary**:
  - I/O-bound operations (network requests, file I/O)
  - High-concurrency services (web servers)
  - Parallel computation (data processing pipelines)
- Avoid premature parallelization (measure first)

---

## Performance & Optimization

**Optimization Philosophy:**
- **Premature optimization is the root of all evil** (Donald Knuth)
- Prioritize: Correctness → Clarity → Performance
- Optimize only after profiling identifies bottlenecks

**When to Optimize:**
1. Profile first (use language-appropriate profilers)
2. Identify the actual bottleneck (don't guess)
3. Measure improvement (before/after benchmarks)
4. Document the optimization and why it was necessary

**Complexity Documentation:**
- Document time/space complexity for non-trivial algorithms
- Use Big-O notation: `O(n)`, `O(n log n)`, `O(1)`
- Example: `# Time: O(n²), Space: O(n) - acceptable for n < 1000`

**Performance Testing:**
- Include benchmarks for critical paths
- Set performance budgets (max response time, max bundle size)
- Test with realistic data volumes

---

## Documentation

**README.md Requirements:**
- **Project overview**: What it does, why it exists
- **Installation**: Step-by-step setup instructions
- **Usage**: Basic examples, common use cases
- **Configuration**: Environment variables, config files
- **Development**: How to contribute, run tests, build locally
- **License**: License type and link

**API Documentation:**
- Document all public APIs comprehensively
- Include: parameters, return values, exceptions, examples
- Use OpenAPI/Swagger for REST APIs
- Use GraphQL schema for GraphQL APIs
- Generate docs from code where possible (JSDoc → docs, rustdoc → docs)

**Code Examples:**
- Include usage examples for non-trivial functionality
- Keep examples minimal and focused
- Ensure examples actually work (test them in CI)

**CHANGELOG:**
- Update when appropriate (follow project conventions)
- Use semantic versioning (MAJOR.MINOR.PATCH)
- Format: Keep A Changelog standard
- Include: Added, Changed, Deprecated, Removed, Fixed, Security

**Inline Documentation:**
- Explain **why**, not **what**
- Good: `// Cache results to avoid expensive database query on every request`
- Bad: `// Store value in cache variable`

---

## Git & Version Control

**Commit Practices:**
- **Atomic commits**: Each commit should represent one logical change
- **Clear messages**: Descriptive and concise
- **Working code**: Every commit should pass tests (bisectable history)

**Commit Message Format:**
```
type(scope): short description

Longer explanation if needed. Explain WHY, not what.
Include context, rationale, tradeoffs considered.

Refs: #123
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `perf`, `ci`

**Examples:**
- `feat(auth): add OAuth2 support for GitHub login`
- `fix(parser): handle empty input without crashing`
- `docs(api): add examples for webhook endpoints`
- `refactor(db): extract query logic into repository pattern`

**Branching Strategy:**
- Follow project conventions (check `CONTRIBUTING.md`)
- Common strategies: Git Flow, GitHub Flow, Trunk-Based Development
- Use descriptive branch names: `feat/oauth-login`, `fix/parser-crash`

**Pre-Commit Checklist:**
- [ ] Code passes all tests (`npm test`, `pytest`, etc.)
- [ ] Code passes linter (`eslint`, `pylint`, `clippy`)
- [ ] Code is formatted (`prettier`, `black`, `rustfmt`)
- [ ] No debugging code left in (console.log, print statements)
- [ ] No sensitive data committed (check with `git diff`)

---

## Output Requirements

**Default Deliverables:**
- Implementation code (properly structured and modular)
- Unit tests for all non-trivial functions
- Integration tests for system boundaries
- Updated documentation (if public API changed)

**Code Quality Gates:**
- **Linting**: Code must pass project linter with zero errors
- **Type checking**: Code must pass type checker (if applicable)
- **Tests**: All tests must pass
- **Formatting**: Code must be consistently formatted

**Before Submitting:**
- Run full test suite
- Run linter and formatter
- Check for console logs / debug statements
- Verify documentation is updated
- Review your own diff

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
- **Container Awareness:** All scripts must detect and adapt to container/local environment

### Testing Requirements
- **Unit Test Coverage:** >80% for core logic (`scripts/` directory)
- **Integration Tests:** Full pipeline tests with synthetic and real test data
- **Performance Tests:** Container must be within 10% of local installation performance
- **Test Fixtures:** Located in `tests/fixtures/` (synthetic and real data)

### Special Conventions
- **SOLID/DRY Enforcement:** Apply to all new code (see `docs/ROADMAP-1-DOCKER.md` for examples)
- **No Hardcoded Paths:** Use environment variables (`VFX_MODELS_DIR`, `VFX_PROJECTS_DIR`, etc.)
- **Backward Compatibility:** All changes must work in both local and container modes
- **Error Messages:** Container-aware error messages (suggest volume mounts, etc.)

### Project Structure
```
comfyui_ingest/
├── scripts/           # Core pipeline scripts (Python)
├── tests/             # Test suite (pytest)
│   ├── fixtures/      # Test data (synthetic + real)
│   └── integration/   # Integration tests
├── docs/              # Documentation and roadmaps
├── workflow_templates/# ComfyUI workflow JSON files
└── docker/            # Docker-related files (entrypoint, etc.)
```

### Documentation Standards
- **Roadmaps:** Temporary planning docs in `docs/` (remove before production release)
- **Testing Guide:** Maintain `tests/README.md` with fixture documentation
- **Architecture Docs:** Keep `docs/ATLAS.md` and roadmaps updated during development

### Prohibited Patterns
- **Inline comments in function bodies** (strictly forbidden - code must be self-documenting)
- **Hardcoded file paths** (use environment variables and `env_config.py`)
- **Conda checks in containers** (detect environment and skip appropriately)
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

**Version:** 1.0
**Last Updated:** 2026-01-17
**Maintained By:** Project Team

<!--
REMINDER: This file should be reviewed and updated as the project evolves.
Remove temporary roadmap references before production release.
-->
