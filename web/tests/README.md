# VFX Pipeline - Test Suite

This directory contains the test suite for the VFX Pipeline web application.

## Test Structure

```
web/tests/
├── conftest.py           # Pytest fixtures and configuration
├── integration/          # Integration tests (API, UI flows)
│   ├── test_api.py
│   └── test_ui_flows.py
└── unit/                 # Unit tests
    ├── test_repositories.py
    ├── test_project_service.py
    └── test_javascript_utils.html
```

## Running Tests

### Run All Tests

```bash
pytest
```

### Run Specific Test Files

```bash
# Integration tests only
pytest web/tests/integration/

# Unit tests only
pytest web/tests/unit/

# Specific file
pytest web/tests/integration/test_api.py
```

### Run Tests by Marker

```bash
# Unit tests
pytest -m unit

# Integration tests
pytest -m integration

# Slow tests
pytest -m slow

# Tests that require ComfyUI
pytest -m requires_comfyui
```

### Verbose Output

```bash
pytest -v
pytest -vv  # Extra verbose
```

### Show Coverage

```bash
pytest --cov=web --cov-report=html
open htmlcov/index.html  # View coverage report
```

## Test Environment

### Automatic Mocking

The test suite automatically mocks ComfyUI when it's not available:

- **ComfyUI Directory**: A mock directory structure is created in `/tmp`
- **ComfyUI HTTP Endpoints**: HTTP requests to ComfyUI are mocked to return success
- **Environment Variables**: Test-specific directories are used

This allows tests to run in CI environments without requiring a full ComfyUI installation.

### Environment Variables

Tests use these environment variables (automatically set by `conftest.py`):

- `VFX_PROJECTS_DIR`: Directory for test projects
- `VFX_MODELS_DIR`: Directory for test models
- `INSTALL_DIR`: Installation directory (contains `.vfx_pipeline/ComfyUI`)

### Testing with Real ComfyUI

If you want to test against a real ComfyUI instance:

1. Start ComfyUI on port 8188:
   ```bash
   cd /path/to/ComfyUI
   python main.py
   ```

2. Run tests:
   ```bash
   pytest
   ```

The test suite will detect the running ComfyUI instance and skip mocking.

## Test Fixtures

### Available Fixtures

#### `mock_comfyui_directory` (session scope, autouse)
Creates a temporary ComfyUI directory structure for testing.

#### `mock_comfyui_http` (function scope, autouse)
Mocks ComfyUI HTTP endpoints if ComfyUI is not running.

#### `mock_env_vars` (function scope)
Sets up temporary directories for projects and models.

#### `temp_projects_dir` (function scope)
Creates a temporary directory for project testing.

#### `client` (function scope)
FastAPI test client with mocked environment.

### Using Fixtures

```python
def test_with_env_vars(mock_env_vars):
    """Test that uses mock environment variables."""
    projects_dir = mock_env_vars["projects_dir"]
    # Your test code here

def test_with_client(client):
    """Test that uses FastAPI test client."""
    response = client.get("/api/system/status")
    assert response.status_code == 200
```

## Writing Tests

### Test Naming

- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`

### Test Structure

```python
import pytest
from fastapi.testclient import TestClient

class TestFeatureName:
    """Tests for specific feature."""

    def test_success_case(self, client):
        """Test the happy path."""
        response = client.get("/api/endpoint")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_error_case(self, client):
        """Test error handling."""
        response = client.get("/api/invalid")
        assert response.status_code == 404
```

### Test Markers

Add markers to categorize tests:

```python
import pytest

@pytest.mark.unit
def test_unit_function():
    """Unit test."""
    pass

@pytest.mark.integration
def test_integration_flow(client):
    """Integration test."""
    pass

@pytest.mark.slow
def test_long_running_operation():
    """Slow test."""
    pass

@pytest.mark.requires_comfyui
def test_with_real_comfyui():
    """Test that needs real ComfyUI."""
    pass
```

### Async Tests

Use `pytest-asyncio` for async tests:

```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    """Test async function."""
    result = await some_async_function()
    assert result == expected
```

## Continuous Integration

Tests run automatically on GitHub Actions for:

- All pushes to any branch
- All pull requests

### CI Environment

- **OS**: Ubuntu Latest
- **Python**: 3.x (latest)
- **ComfyUI**: Mocked (not installed)
- **Test Command**: `pytest -v --tb=short`

### Test Failures

If tests fail in CI:

1. Check the GitHub Actions logs
2. Reproduce locally: `pytest -v --tb=short`
3. Fix the issue
4. Commit and push

### Skipping Tests in CI

To skip tests in CI only:

```python
import os
import pytest

@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Requires local resources"
)
def test_local_only():
    """Test that only runs locally."""
    pass
```

## Debugging Tests

### Show Print Statements

```bash
pytest -s
```

### Drop into Debugger on Failure

```bash
pytest --pdb
```

### Run Last Failed Tests Only

```bash
pytest --lf
```

### Show Local Variables on Failure

```bash
pytest -l
```

### Full Traceback

```bash
pytest --tb=long
```

## Test Coverage

### Generate Coverage Report

```bash
pytest --cov=web --cov-report=html --cov-report=term
```

### View HTML Report

```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Coverage Goals

- **Overall**: 80%+
- **Core Logic**: 90%+
- **Critical Paths**: 100%

## Troubleshooting

### ImportError: No module named 'X'

Install missing dependencies:
```bash
pip install -r requirements.txt
```

### Tests Pass Locally but Fail in CI

- Check environment variables
- Verify mock fixtures are applied
- Check for hardcoded paths
- Ensure no local-only resources

### ComfyUI Tests Failing

If ComfyUI-dependent tests fail:

1. Check if ComfyUI is running: `curl http://127.0.0.1:8188/system_stats`
2. Verify mock is working: Check `conftest.py`
3. Mark test as `@pytest.mark.requires_comfyui` if it needs real ComfyUI

### Slow Tests

If tests are too slow:

1. Mark slow tests: `@pytest.mark.slow`
2. Skip slow tests: `pytest -m "not slow"`
3. Use smaller test data
4. Mock expensive operations

## Best Practices

1. **One test, one assertion focus**: Test one thing per test function
2. **Use descriptive names**: `test_user_cannot_delete_other_users_projects`
3. **Arrange-Act-Assert**: Structure tests clearly
4. **Use fixtures**: Don't repeat setup code
5. **Mock external dependencies**: Don't rely on external services
6. **Test edge cases**: Empty inputs, null values, boundaries
7. **Test error paths**: Not just success cases
8. **Keep tests fast**: Use mocks, small data
9. **Clean up**: Use fixtures with teardown
10. **Document complex tests**: Add docstrings explaining why

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [FastAPI testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [Coverage.py](https://coverage.readthedocs.io/)

---

**Last Updated**: 2026-01-20
**Maintained By**: VFX Pipeline Team
