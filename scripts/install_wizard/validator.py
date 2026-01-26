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

        Returns:
            (success, message)
        """
        colmap_path = PlatformManager.find_tool("colmap")
        if not colmap_path:
            return False, "COLMAP not found in PATH or standard locations"

        colmap_exe = str(colmap_path)
        is_bat = colmap_exe.lower().endswith('.bat')

        success, output = run_command(
            [colmap_exe, "--version"],
            check=False, capture=True, shell=is_bat
        )
        if success and output:
            version = output.strip().split('\n')[0] if output else "unknown"
            return True, f"COLMAP {version}"

        return True, f"COLMAP available at {colmap_path}"

    def validate_checkpoint_files(self, base_dir: Optional[Path] = None) -> Dict[str, bool]:
        """Check if checkpoint files exist.

        Args:
            base_dir: Base directory for checkpoints

        Returns:
            Dict mapping component to checkpoint status
        """
        base_dir = base_dir or self.install_dir
        if not base_dir:
            return {}

        results = {}

        # GVHMR: Check for the main checkpoint file
        gvhmr_ckpt = base_dir / "GVHMR" / "inputs" / "checkpoints" / "gvhmr" / "gvhmr_siga24_release.ckpt"
        results['gvhmr'] = gvhmr_ckpt.exists()

        # WHAM: Check for the main checkpoint file
        wham_ckpt = base_dir / "WHAM" / "checkpoints" / "wham_vit_w_3dpw.pth.tar"
        results['wham'] = wham_ckpt.exists()

        # ECON: Check for extracted data from econ_data.zip
        econ_data_dir = base_dir / "ECON" / "data"
        if econ_data_dir.exists():
            # Check for any model/data files in the extracted directory
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
        print("\n📦 Python Packages:")
        for pkg, status in results['python_packages'].items():
            if status:
                print_success(f"{pkg}")
            else:
                print_error(f"{pkg} - not found")

        # PyTorch CUDA
        print("\n🎮 GPU Support:")
        cuda_success, cuda_msg = results['pytorch_cuda']
        if cuda_success:
            print_success(cuda_msg)
        else:
            print_warning(cuda_msg)

        # COLMAP
        print("\n📐 COLMAP:")
        colmap_success, colmap_msg = results['colmap']
        if colmap_success:
            print_success(colmap_msg)
        else:
            print_warning(colmap_msg)

        # Checkpoints
        print("\n🎯 Motion Capture Checkpoints:")
        for comp, status in results['checkpoints'].items():
            if status:
                print_success(f"{comp.upper()} checkpoint found")
            else:
                print_info(f"{comp.upper()} checkpoint not found (install with wizard)")

        # SMPL-X models
        print("\n🧍 SMPL-X Models:")
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
