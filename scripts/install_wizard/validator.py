"""Installation validation for the installation wizard.

This module provides smoke tests to verify that all components
are properly installed and functional.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from env_config import INSTALL_DIR

from .platform import PlatformManager
from .utils import (
    print_error,
    print_header,
    print_info,
    print_success,
    print_warning,
    run_command,
)

if TYPE_CHECKING:
    from .conda import CondaEnvironmentManager


class InstallationValidator:
    """Validates installation with smoke tests."""

    def __init__(self, conda_manager: 'CondaEnvironmentManager', install_dir: Optional[Path] = None):
        self.conda_manager = conda_manager
        self.install_dir = install_dir
        self.base_dir = install_dir or INSTALL_DIR
        self._checkpoint_issues: Dict[str, list] = {}

    def validate_python_imports(self) -> Dict[str, bool]:
        """Test importing key Python packages.

        Returns:
            Dict mapping package name to success status
        """
        packages_to_test = [
            ('numpy', 'numpy'),
            ('cv2', 'opencv-python'),
            ('PIL', 'Pillow'),
            ('torch', 'PyTorch'),
            ('smplx', 'SMPL-X'),
            ('trimesh', 'trimesh'),
        ]

        results = {}
        for import_name, display_name in packages_to_test:
            try:
                __import__(import_name)
                results[display_name] = True
            except ImportError:
                results[display_name] = False

        return results

    def validate_pytorch_cuda(self) -> Tuple[bool, str]:
        """Check if PyTorch can access CUDA.

        Returns:
            (success, message)
        """
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
                return True, f"CUDA available: {device_count} device(s), {device_name}"
            else:
                return False, "CUDA not available (CPU-only mode)"
        except ImportError:
            return False, "PyTorch not installed"
        except Exception as e:
            return False, f"Error checking CUDA: {e}"

    def validate_colmap(self) -> Tuple[bool, str]:
        """Check if COLMAP is accessible.

        Checks dedicated 'colmap' conda env first, then fallback locations.

        Returns:
            (success, message)
        """
        colmap_path = self._find_colmap_executable()
        if not colmap_path:
            return False, "COLMAP not found in PATH or standard locations"

        colmap_exe = str(colmap_path)
        is_bat = colmap_exe.lower().endswith('.bat')

        success, output = run_command(
            [colmap_exe, "--version"],
            check=False, capture=True, shell=is_bat
        )
        if success and output:
            version = output.strip().splitlines()[0] if output else "unknown"
            return True, f"COLMAP {version}"

        return True, f"COLMAP available at {colmap_path}"

    def _find_colmap_executable(self) -> Optional[Path]:
        """Find COLMAP executable, checking dedicated conda env first."""
        import os
        import sys

        # Check dedicated colmap conda env first
        conda_exe = os.environ.get("CONDA_EXE")
        if conda_exe:
            conda_base = Path(conda_exe).parent.parent
        else:
            conda_base = None
            for base_name in ["anaconda3", "miniconda3", "miniforge3"]:
                candidate = Path.home() / base_name
                if candidate.exists():
                    conda_base = candidate
                    break

        if conda_base:
            env_path = conda_base / "envs" / "colmap"
            if sys.platform == "win32":
                candidates = [
                    env_path / "Scripts" / "colmap.exe",
                    env_path / "Library" / "bin" / "colmap.exe",
                ]
            else:
                candidates = [env_path / "bin" / "colmap"]

            for candidate in candidates:
                if candidate.exists():
                    return candidate

        # Fall back to PlatformManager
        return PlatformManager.find_tool("colmap")

    def validate_checkpoint_files(self, base_dir: Optional[Path] = None) -> Dict[str, bool]:
        """Check if checkpoint files exist and meet minimum size requirements.

        Validates existence and minimum file size to detect corrupted or
        incomplete downloads (e.g., HTML error pages from Google Drive).

        Args:
            base_dir: Base directory for checkpoints

        Returns:
            Dict mapping component to checkpoint status
        """
        base_dir = base_dir or self.install_dir
        if not base_dir:
            return {}

        results = {}

        gvhmr_checkpoints = {
            "gvhmr/gvhmr_siga24_release.ckpt": 1200,
            "hmr2/epoch=10-step=25000.ckpt": 900,
            "vitpose/vitpose-h-multi-coco.pth": 1100,
            "dpvo/dpvo.pth": 200,
        }

        gvhmr_ckpt_dir = base_dir / "GVHMR" / "inputs" / "checkpoints"
        gvhmr_valid = True
        gvhmr_issues = []

        for ckpt_path, min_size_mb in gvhmr_checkpoints.items():
            full_path = gvhmr_ckpt_dir / ckpt_path
            min_size_bytes = min_size_mb * 1024 * 1024
            if not full_path.exists():
                gvhmr_valid = False
                gvhmr_issues.append(f"missing: {ckpt_path}")
            else:
                actual_size = full_path.stat().st_size
                if actual_size < min_size_bytes:
                    gvhmr_valid = False
                    actual_mb = actual_size / (1024 * 1024)
                    gvhmr_issues.append(f"corrupted: {ckpt_path} ({actual_mb:.1f} MB, expected >={min_size_mb} MB)")

        results['gvhmr'] = gvhmr_valid
        if gvhmr_issues:
            self._checkpoint_issues['gvhmr'] = gvhmr_issues

        econ_data_dir = base_dir / "ECON" / "data"
        if econ_data_dir.exists():
            has_data = any(econ_data_dir.glob("**/*.pkl")) or \
                       any(econ_data_dir.glob("**/*.pth")) or \
                       any(econ_data_dir.glob("**/smpl_related"))
            results['econ'] = has_data
        else:
            results['econ'] = False

        return results

    def validate_smplx_models(self) -> Tuple[bool, str]:
        """Check if SMPL-X models are installed.

        The SMPL-X v1.1 zip extracts to: models/smplx/SMPLX_*.npz

        Returns:
            (success, message)
        """
        smplx_dir = self.base_dir / "smplx_models"
        if not smplx_dir.exists():
            return False, f"SMPL-X directory not found ({INSTALL_DIR}/smplx_models/)"

        # Look for model files - check multiple possible locations and extensions
        # The SMPL-X v1.1 zip extracts to: models/smplx/SMPLX_*.npz
        search_patterns = [
            "models/smplx/SMPLX_*.npz",  # Standard v1.1 zip structure
            "smplx/SMPLX_*.npz",          # Alternative structure
            "SMPLX_*.npz",                # Flat structure
            "models/smplx/SMPLX_*.pkl",   # Legacy .pkl format
            "smplx/SMPLX_*.pkl",
            "SMPLX_*.pkl",
        ]

        model_files = []
        for pattern in search_patterns:
            model_files.extend(smplx_dir.glob(pattern))
            if model_files:
                break

        if model_files:
            return True, f"Found {len(model_files)} SMPL-X model file(s)"
        return False, "No SMPL-X model files found"

    def run_all_tests(self) -> Dict[str, Dict]:
        """Run all validation tests.

        Returns:
            Dict with test categories and results
        """
        results = {
            'python_packages': self.validate_python_imports(),
            'pytorch_cuda': self.validate_pytorch_cuda(),
            'colmap': self.validate_colmap(),
            'checkpoints': self.validate_checkpoint_files(),
            'smplx_models': self.validate_smplx_models(),
        }
        return results

    def print_validation_report(self, results: Dict[str, Dict]):
        """Print formatted validation report.

        Args:
            results: Results from run_all_tests()
        """
        print_header("Installation Validation")

        # Python packages
        print("\n[Python Packages]")
        for pkg, status in results['python_packages'].items():
            if status:
                print_success(f"{pkg}")
            else:
                print_error(f"{pkg} - not found")

        # PyTorch CUDA
        print("\n[GPU Support]")
        cuda_success, cuda_msg = results['pytorch_cuda']
        if cuda_success:
            print_success(cuda_msg)
        else:
            print_warning(cuda_msg)

        # COLMAP
        print("\n[COLMAP]")
        colmap_success, colmap_msg = results['colmap']
        if colmap_success:
            print_success(colmap_msg)
        else:
            print_warning(colmap_msg)

        # Checkpoints
        print("\n[Motion Capture Checkpoints]")
        for comp, status in results['checkpoints'].items():
            if status:
                print_success(f"{comp.upper()} checkpoints valid")
            else:
                issues = self._checkpoint_issues.get(comp, [])
                if issues:
                    print_error(f"{comp.upper()} checkpoints invalid:")
                    for issue in issues:
                        print(f"    - {issue}")
                    print_info("  Re-download from: https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD")
                else:
                    print_info(f"{comp.upper()} checkpoint not found (install with wizard)")

        # SMPL-X models
        print("\n[SMPL-X Models]")
        smplx_success, smplx_msg = results['smplx_models']
        if smplx_success:
            print_success(smplx_msg)
        else:
            print_warning(smplx_msg)
            print_info("Register at https://smpl-x.is.tue.mpg.de/ to download")

    def validate_and_report(self):
        """Run validation tests and print report."""
        results = self.run_all_tests()
        self.print_validation_report(results)
