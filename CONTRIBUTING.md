# Contributing to Shot Gopher

Thank you for your interest in contributing to Shot Gopher! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/shot-gopher.git
   cd shot-gopher
   ```
3. **Set up the development environment**:
   ```bash
   python scripts/install_wizard.py
   ```

## Development Guidelines

### Code Style

- **Python**: Follow PEP 8, use `black` for formatting
- **Line length**: 100 characters (Python), 120 characters (Bash)
- **Type hints**: Required for all function signatures
- **Docstrings**: Required for modules, classes, and public functions

### Testing

Run tests before submitting:
```bash
pytest tests/
```

See [docs/testing.md](docs/testing.md) for detailed testing guidelines.

### Commit Messages

Follow conventional commits format:
```
type(scope): short description

Longer explanation if needed.

Refs: #123
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

**Examples**:
- `feat(pipeline): add support for EXR output`
- `fix(colmap): handle spaces in file paths`
- `docs(readme): update installation instructions`

## Pull Request Process

1. **Create a feature branch**:
   ```bash
   git checkout -b feat/my-feature
   ```

2. **Make your changes** with clear, atomic commits

3. **Test your changes**:
   ```bash
   pytest tests/
   python scripts/install_wizard.py -v  # Validation
   ```

4. **Push to your fork**:
   ```bash
   git push origin feat/my-feature
   ```

5. **Open a Pull Request** with:
   - Clear description of the changes
   - Reference to any related issues
   - Test plan or verification steps

## Reporting Issues

When reporting bugs, include:

1. **System information**: OS, GPU, VRAM, Python version
2. **Steps to reproduce**: Minimal commands to trigger the issue
3. **Expected vs actual behavior**
4. **Error messages**: Full traceback if available
5. **Logs**: Relevant output from the pipeline

Use the issue templates when available.

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow

## Questions?

- **Issues**: https://github.com/kleer001/shot-gopher/issues
- **Discussions**: https://github.com/kleer001/shot-gopher/discussions

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.
