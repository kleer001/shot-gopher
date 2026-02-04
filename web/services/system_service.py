"""System status service for checking ComfyUI, GPU, and disk status."""

import platform
import re
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
        """Get GPU information with cross-platform support.

        Attempts detection in order: NVIDIA, AMD (Linux), macOS.
        Returns total and available VRAM when possible.
        """
        result = self._get_nvidia_gpu_info()
        if result["name"] != "Unknown":
            return result

        result = self._get_amd_gpu_info()
        if result["name"] != "Unknown":
            return result

        result = self._get_macos_gpu_info()
        if result["name"] != "Unknown":
            return result

        return {"name": "Unknown", "vram_gb": 0, "vram_available_gb": 0}

    def _get_nvidia_gpu_info(self) -> dict:
        """Get NVIDIA GPU info using nvidia-smi (Linux/Windows)."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.free",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(", ")
                if len(parts) >= 3:
                    total_mb = int(parts[1].strip())
                    free_mb = int(parts[2].strip())
                    return {
                        "name": parts[0].strip(),
                        "vram_gb": round(total_mb / 1024, 1),
                        "vram_available_gb": round(free_mb / 1024, 1)
                    }
        except Exception:
            pass
        return {"name": "Unknown", "vram_gb": 0, "vram_available_gb": 0}

    def _get_amd_gpu_info(self) -> dict:
        """Get AMD GPU info using rocm-smi (Linux) or sysfs fallback."""
        if platform.system() != "Linux":
            return {"name": "Unknown", "vram_gb": 0, "vram_available_gb": 0}

        result = self._get_amd_rocm_info()
        if result["name"] != "Unknown":
            return result

        return self._get_amd_sysfs_info()

    def _get_amd_rocm_info(self) -> dict:
        """Get AMD GPU info via rocm-smi."""
        try:
            name_result = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True,
                text=True,
                timeout=5
            )
            gpu_name = "AMD GPU"
            if name_result.returncode == 0:
                for line in name_result.stdout.splitlines():
                    if "Card series:" in line or "GPU" in line:
                        gpu_name = line.split(":")[-1].strip() or "AMD GPU"
                        break

            mem_result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if mem_result.returncode == 0:
                total_bytes = 0
                used_bytes = 0
                for line in mem_result.stdout.splitlines():
                    if "Total Memory" in line:
                        match = re.search(r"(\d+)", line)
                        if match:
                            total_bytes = int(match.group(1))
                    elif "Used Memory" in line:
                        match = re.search(r"(\d+)", line)
                        if match:
                            used_bytes = int(match.group(1))

                if total_bytes > 0:
                    total_gb = total_bytes / (1024 ** 3)
                    free_gb = (total_bytes - used_bytes) / (1024 ** 3)
                    return {
                        "name": gpu_name,
                        "vram_gb": round(total_gb, 1),
                        "vram_available_gb": round(free_gb, 1)
                    }
        except Exception:
            pass
        return {"name": "Unknown", "vram_gb": 0, "vram_available_gb": 0}

    def _get_amd_sysfs_info(self) -> dict:
        """Get AMD GPU info from Linux sysfs (fallback without rocm-smi)."""
        try:
            drm_path = Path("/sys/class/drm")
            for card_dir in drm_path.glob("card[0-9]*"):
                vram_total_path = card_dir / "device" / "mem_info_vram_total"
                vram_used_path = card_dir / "device" / "mem_info_vram_used"

                if vram_total_path.exists():
                    total_bytes = int(vram_total_path.read_text().strip())
                    used_bytes = 0
                    if vram_used_path.exists():
                        used_bytes = int(vram_used_path.read_text().strip())

                    gpu_name = "AMD GPU"
                    name_path = card_dir / "device" / "product_name"
                    if name_path.exists():
                        gpu_name = name_path.read_text().strip() or "AMD GPU"

                    total_gb = total_bytes / (1024 ** 3)
                    free_gb = (total_bytes - used_bytes) / (1024 ** 3)
                    return {
                        "name": gpu_name,
                        "vram_gb": round(total_gb, 1),
                        "vram_available_gb": round(free_gb, 1)
                    }
        except Exception:
            pass
        return {"name": "Unknown", "vram_gb": 0, "vram_available_gb": 0}

    def _get_macos_gpu_info(self) -> dict:
        """Get GPU info on macOS using system_profiler."""
        if platform.system() != "Darwin":
            return {"name": "Unknown", "vram_gb": 0, "vram_available_gb": 0}

        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                gpu_name = None
                vram_gb = 0

                for line in result.stdout.splitlines():
                    line = line.strip()
                    if "Chipset Model:" in line:
                        gpu_name = line.split(":")[-1].strip()
                    elif "VRAM" in line and ":" in line:
                        vram_str = line.split(":")[-1].strip()
                        match = re.search(r"(\d+)\s*(GB|MB)", vram_str, re.IGNORECASE)
                        if match:
                            value = int(match.group(1))
                            unit = match.group(2).upper()
                            vram_gb = value if unit == "GB" else value / 1024

                if gpu_name:
                    if "Apple" in gpu_name or "M1" in gpu_name or "M2" in gpu_name or "M3" in gpu_name or "M4" in gpu_name:
                        vram_gb = self._get_apple_silicon_memory()

                    return {
                        "name": gpu_name,
                        "vram_gb": round(vram_gb, 1),
                        "vram_available_gb": round(vram_gb * 0.75, 1)
                    }
        except Exception:
            pass
        return {"name": "Unknown", "vram_gb": 0, "vram_available_gb": 0}

    def _get_apple_silicon_memory(self) -> float:
        """Get unified memory on Apple Silicon (shared between CPU/GPU)."""
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                total_bytes = int(result.stdout.strip())
                total_gb = total_bytes / (1024 ** 3)
                return total_gb * 0.75
        except Exception:
            pass
        return 8.0

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
