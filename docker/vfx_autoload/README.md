# ComfyUI Workflow Auto-Loader

A simple ComfyUI frontend extension that automatically loads a workflow when the page opens.

## Features

- **URL parameter support**: Load any workflow via `?workflow=my_workflow.json`
- **Default location fallback**: Automatically loads `auto_load_workflow.json` from output directory
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
  - ./vfx_autoload:/path/to/ComfyUI/web/extensions/vfx_autoload:ro
```

## Usage

### Method 1: URL Parameter

Open ComfyUI with a workflow filename in the URL:

```
http://localhost:8188/?workflow=my_workflow.json
```

The workflow file must be in ComfyUI's **output directory** (it's fetched via ComfyUI's `/view` endpoint).

### Method 2: Default Location

Place your workflow in ComfyUI's output directory with the default name:

```bash
cp my_workflow.json /path/to/ComfyUI/output/auto_load_workflow.json
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
- Fetching from ComfyUI's output directory via `/view` endpoint
- Requiring zero browser-side configuration

## How It Works

The extension uses ComfyUI's built-in `/view` endpoint which serves files from the output directory:

```
/view?filename=auto_load_workflow.json&type=output&format=json
```

This is the same endpoint used to view generated images, so it's guaranteed to work.

## Integration Example

Launch ComfyUI and open a specific workflow:

```python
import shutil
import webbrowser
import subprocess

# Copy workflow to output directory
shutil.copy("my_workflow.json", "/path/to/ComfyUI/output/auto_load_workflow.json")

# Start ComfyUI
subprocess.Popen(["python", "main.py", "--listen", "0.0.0.0"])

# Open browser - workflow loads automatically
webbrowser.open("http://localhost:8188/")
```

Or with URL parameter:

```python
webbrowser.open("http://localhost:8188/?workflow=my_workflow.json")
```

## License

MIT - Use freely in your own projects.

## Credits

Created for [shot-gopher](https://github.com/kleer001/shot-gopher) VFX pipeline.
