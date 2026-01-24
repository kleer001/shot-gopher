# VFX Pipeline Documentation

Complete documentation for the shot-gopher VFX pipeline toolset.

## Essential Docs

**Before you start:**
- **[Installation](installation.md)** - Setup and component installation
- **[Docker Guide](docker.md)** - Docker-based deployment
- **[Windows IT Setup (Docker)](windows-it-docker.md)** - For IT admins (WSL2/Docker)
- **[Windows IT Setup (Native)](windows-it-native.md)** - For IT admins (native)

**Once you start:**
- **[Your First Project](first-project.md)** - Complete walkthrough
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions
- **[Testing](testing.md)** - Running and writing tests

## Reference Documentation

Detailed technical reference in [reference/](reference/):

| Document | Description |
|----------|-------------|
| [Pipeline Stages](reference/stages.md) | Stage-by-stage pipeline details |
| [CLI Reference](reference/cli.md) | Command-line options |
| [API](reference/api.md) | REST API documentation |
| [Scripts](reference/scripts.md) | Component script reference |
| [Maintenance](reference/maintenance.md) | System health and updates |
| [Interactive Segmentation](reference/interactive-segmentation.md) | SAM3 segmentation guide |

## Platform Support

| Platform | Status | Guide |
|----------|--------|-------|
| Linux | Fully supported | [Docker](docker.md) or [Installation](installation.md) |
| Windows | WSL2 recommended | [Windows Guide](platforms/windows.md) |
| macOS | Local conda only | [Installation](installation.md) |

## Quick Start

```bash
# Linux / WSL2
curl -fsSL https://raw.githubusercontent.com/kleer001/shot-gopher/main/scripts/bootstrap_docker.sh | bash

# Then run
bash scripts/run_docker.sh --name MyProject --stages all video.mp4
```

See [First Project Guide](first-project.md) for complete walkthrough.

## Architecture

```
shot-gopher/
├── scripts/                 # All executable tools
│   ├── install_wizard/      # Installation wizard package
│   ├── run_pipeline.py      # Pipeline orchestrator
│   ├── janitor.py           # Maintenance tool
│   └── env_config.py        # Centralized config
├── workflow_templates/      # ComfyUI workflows
├── .vfx_pipeline/           # Installation directory
└── docs/                    # This documentation
    ├── reference/           # Technical reference
    └── platforms/           # Platform-specific guides

../vfx_projects/             # Projects (sibling to repo)
```

## Getting Help

- **Issues**: https://github.com/kleer001/shot-gopher/issues
- **Discussions**: https://github.com/kleer001/shot-gopher/discussions
