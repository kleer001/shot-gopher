"""System status service for checking ComfyUI, GPU, and disk status."""

import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Optional


class SystemService:
    """Service for checking system status and resources."""

    def check_comfyui_status(self) -> bool:
        """Check if ComfyUI is running."""
        try:
            req = urllib.request.Request("http://127.0.0.1:8188/system_stats", method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    def get_gpu_info(self) -> dict:
        """Get GPU information using nvidia-smi."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(", ")
                if len(parts) >= 2:
                    return {
                        "name": parts[0].strip(),
                        "vram_gb": round(int(parts[1].strip()) / 1024, 1)
                    }
        except Exception:
            pass
        return {"name": "Unknown", "vram_gb": 0}

    def get_disk_usage(self, path: Path) -> Optional[dict]:
        """Get disk usage statistics for a path."""
        try:
            target_dir = path if path.exists() else path.parent
            if target_dir.exists():
                stat = shutil.disk_usage(target_dir)
                if stat.total == 0:
                    return None
                return {
                    "free_gb": round(stat.free / (1024**3), 1),
                    "total_gb": round(stat.total / (1024**3), 1),
                    "used_percent": round((stat.used / stat.total) * 100),
                }
        except Exception:
            pass
        return None


_system_service = None


def get_system_service() -> SystemService:
    """Get singleton system service instance."""
    global _system_service
    if _system_service is None:
        _system_service = SystemService()
    return _system_service
