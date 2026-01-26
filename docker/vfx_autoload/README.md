# ComfyUI Workflow Auto-Loader

A simple ComfyUI frontend extension that automatically loads a workflow when the page opens.

## Features

- **URL parameter support**: Load any workflow via `?workflow=my_workflow.json`
- **Default location fallback**: Automatically loads `workflow.json` from extension directory
- **No browser caching**: Works reliably with Docker, tunnels, and changing URLs
- **Minimal footprint**: Single file, ~70 lines, no dependencies

## Installation

Copy this folder to your ComfyUI extensions directory:

```bash
cp -r vfx_autoload /path/to/ComfyUI/web/extensions/
```

Or with Docker, mount it as a volume:

```yaml
volumes:
  - ./vfx_autoload:/path/to/ComfyUI/web/extensions/vfx_autoload
```

## Usage

### Method 1: URL Parameter

Open ComfyUI with a workflow filename in the URL:

```
http://localhost:8188/?workflow=my_workflow.json
```

The workflow file must be in the extension's directory (served as a static file).

### Method 2: Default Location

Place your workflow in the extension directory with the default name:

```bash
cp my_workflow.json /path/to/ComfyUI/web/extensions/vfx_autoload/workflow.json
```

The URL parameter takes precedence if both are present.

## Why This Extension?

ComfyUI doesn't have native support for loading a workflow via command line or URL parameter ([feature request #9858](https://github.com/comfyanonymous/ComfyUI/issues/9858)).

Existing solutions like ComfyUI-Custom-Scripts use browser localStorage, which:
- Doesn't work with Docker (different browser sessions)
- Breaks when URLs change (tunnels, port forwarding)
- Requires manual setup in the browser

This extension solves these issues by:
- Using URL parameters (stateless, shareable)
- Fetching from extension's static directory (served by ComfyUI)
- Requiring zero browser-side configuration

## How It Works

The extension fetches the workflow from its own static directory:

```
/extensions/vfx_autoload/workflow.json
```

ComfyUI serves the `web/extensions/` directory as static files, so any JSON file placed there is accessible via HTTP.

## Integration Example

Launch ComfyUI and auto-load a specific workflow:

```python
import shutil
import webbrowser
import subprocess

# Copy workflow to extension directory
shutil.copy("my_workflow.json", "/path/to/ComfyUI/web/extensions/vfx_autoload/workflow.json")

# Start ComfyUI
subprocess.Popen(["python", "main.py", "--listen", "0.0.0.0"])

# Open browser - workflow loads automatically
webbrowser.open("http://localhost:8188/")
```

## Browser Console

When loading, you should see these messages in the browser console (F12):

```
[AutoLoad] Extension initializing, waiting 1500ms...
[AutoLoad] Attempting to load workflow from default location: workflow.json
[AutoLoad] Fetching from: /extensions/vfx_autoload/workflow.json
[AutoLoad] Workflow fetched, loading into graph...
[AutoLoad] Workflow loaded successfully from default location
```

## License

MIT - Use freely in your own projects.

## Credits

Created for [shot-gopher](https://github.com/kleer001/shot-gopher) VFX pipeline.
