# API Usage Guide

## Base URL

```
http://localhost:5000/api
```

## Interactive Documentation

- **Swagger UI**: http://localhost:5000/api/docs
- **ReDoc**: http://localhost:5000/api/redoc

## Authentication

Currently no authentication required (local deployment).

---

## Projects

### List All Projects

```bash
curl http://localhost:5000/api/projects
```

**Response:**
```json
{
  "projects": [
    {
      "name": "Shot_01",
      "status": "created",
      "video_path": "/workspace/Shot_01/source/input.mp4",
      "stages": ["ingest", "colmap"],
      "created_at": "2026-01-20T10:00:00",
      "updated_at": "2026-01-20T10:00:00"
    }
  ]
}
```

### Get Project

```bash
curl http://localhost:5000/api/projects/Shot_01
```

### Upload Video

```bash
curl -X POST http://localhost:5000/api/upload \
  -F "video=@/path/to/video.mp4" \
  -F "name=Shot_01"
```

**Response:**
```json
{
  "project_id": "Shot_01",
  "name": "Shot_01",
  "project_dir": "/workspace/Shot_01",
  "video_info": {
    "duration": 10.5,
    "fps": 24.0,
    "resolution": [1920, 1080],
    "frame_count": 252
  }
}
```

### Get Project Outputs

```bash
curl http://localhost:5000/api/projects/Shot_01/outputs
```

---

## Pipeline Execution

### Start Pipeline Job

```bash
curl -X POST http://localhost:5000/api/projects/Shot_01/start \
  -H "Content-Type: application/json" \
  -d '{
    "stages": ["ingest", "depth", "colmap"],
    "roto_prompt": "person",
    "skip_existing": false
  }'
```

**Response:**
```json
{
  "status": "started",
  "project_id": "Shot_01",
  "job": {
    "project_name": "Shot_01",
    "stages": ["ingest", "depth", "colmap"],
    "status": "running",
    "current_stage": "ingest",
    "progress": 0.0,
    "message": "Starting pipeline...",
    "started_at": "2026-01-20T10:05:00",
    "completed_at": null,
    "error": null
  }
}
```

### Stop Pipeline Job

```bash
curl -X POST http://localhost:5000/api/projects/Shot_01/stop
```

---

## System

### System Status

```bash
curl http://localhost:5000/api/system/status
```

**Response:**
```json
{
  "comfyui": true,
  "disk_space_gb": 125.3,
  "projects_dir": "/workspace/projects",
  "install_dir": "/workspace/shot-gopher"
}
```

### Get Configuration

```bash
curl http://localhost:5000/api/config
```

**Response:**
```json
{
  "stages": {
    "ingest": {
      "name": "Ingest",
      "description": "Extract frames from video",
      "outputDir": "source/frames"
    },
    "depth": {
      "name": "Depth Estimation",
      "description": "Generate depth maps",
      "outputDir": "depth"
    }
  },
  "presets": {
    "quick": {
      "name": "Quick Test",
      "stages": ["ingest", "depth"]
    }
  },
  "supportedVideoFormats": [".mp4", ".mov", ".avi", ".mkv"]
}
```

---

## WebSocket

### Real-Time Progress Updates

```javascript
const ws = new WebSocket('ws://localhost:5000/ws/Shot_01');

ws.onmessage = (event) => {
  const update = JSON.parse(event.data);

  switch (update.type) {
    case 'progress':
      console.log(`Stage: ${update.stage}, Progress: ${update.progress}`);
      console.log(`Frame ${update.frame} of ${update.total_frames}`);
      break;

    case 'stage_complete':
      console.log(`Stage ${update.stage} completed`);
      break;

    case 'pipeline_complete':
      console.log(`Pipeline finished. Success: ${update.success}`);
      break;

    case 'error':
      console.error(`Error: ${update.error}`);
      break;
  }
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('WebSocket connection closed');
};
```

**Progress Update Message:**
```json
{
  "type": "progress",
  "project_id": "Shot_01",
  "stage": "depth",
  "stage_index": 1,
  "total_stages": 3,
  "frame": 42,
  "total_frames": 252,
  "progress": 0.167,
  "message": "Processing frame 42/252"
}
```

---

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Invalid stages: ['invalid_stage']. Valid: {'ingest', 'depth', 'roto', ...}"
}
```

### 404 Not Found
```json
{
  "detail": "Project not found"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Failed to save video: disk full"
}
```

---

## Common Workflows

### Complete Project Setup and Execution

1. **Upload video and create project:**
   ```bash
   curl -X POST http://localhost:5000/api/upload \
     -F "video=@shot.mp4" \
     -F "name=Shot_01"
   ```

2. **Start pipeline:**
   ```bash
   curl -X POST http://localhost:5000/api/projects/Shot_01/start \
     -H "Content-Type: application/json" \
     -d '{"stages": ["ingest", "depth", "colmap"]}'
   ```

3. **Monitor progress via WebSocket** (see WebSocket section)

4. **Check outputs:**
   ```bash
   curl http://localhost:5000/api/projects/Shot_01/outputs
   ```

---

## Python Client Example

```python
import requests
import json

BASE_URL = "http://localhost:5000/api"

# Upload video
with open("shot.mp4", "rb") as f:
    files = {"video": f}
    data = {"name": "Shot_01"}
    response = requests.post(f"{BASE_URL}/upload", files=files, data=data)
    project = response.json()
    print(f"Created project: {project['project_id']}")

# Start pipeline
job_config = {
    "stages": ["ingest", "depth", "colmap"],
    "roto_prompt": "person",
    "skip_existing": False
}
response = requests.post(
    f"{BASE_URL}/projects/Shot_01/start",
    json=job_config
)
job = response.json()
print(f"Started job: {job['status']}")

# Get outputs
response = requests.get(f"{BASE_URL}/projects/Shot_01/outputs")
outputs = response.json()
print(f"Outputs: {json.dumps(outputs, indent=2)}")
```

---

## TypeScript/JavaScript Client Example

```typescript
// Upload video
const uploadVideo = async (file: File, name: string) => {
  const formData = new FormData();
  formData.append('video', file);
  formData.append('name', name);

  const response = await fetch('/api/upload', {
    method: 'POST',
    body: formData
  });

  return await response.json();
};

// Start pipeline
const startPipeline = async (projectId: string, stages: string[]) => {
  const response = await fetch(`/api/projects/${projectId}/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      stages,
      roto_prompt: 'person',
      skip_existing: false
    })
  });

  return await response.json();
};

// WebSocket connection
const connectWebSocket = (projectId: string) => {
  const ws = new WebSocket(`ws://localhost:5000/ws/${projectId}`);

  ws.onmessage = (event) => {
    const update = JSON.parse(event.data);
    console.log('Progress update:', update);
  };

  return ws;
};
```

---

## Notes

- All timestamps are in ISO 8601 format
- Progress values are between 0.0 and 1.0
- File sizes are in bytes
- Durations are in seconds
- Resolution is [width, height]
