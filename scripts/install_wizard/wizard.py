"""Main installation wizard orchestrator.

This module contains the InstallationWizard class that coordinates
all installation steps and provides the interactive installation flow.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

from env_config import INSTALL_DIR

from .conda import CondaEnvironmentManager
from .config import ConfigurationGenerator
from .downloader import CheckpointDownloader
from .installers import CondaPackageInstaller, GitRepoInstaller, PythonPackageInstaller, SystemPackageInstaller
from .state import InstallationStateManager
from .utils import (
    Colors,
    ask_yes_no,
    check_command_available,
    check_gpu_available,
    format_size_gb,
    get_disk_space,
    print_error,
    print_header,
    print_info,
    print_success,
    print_warning,
    tty_input,
)
from .validator import InstallationValidator


class InstallationWizard:
    """Main installation wizard."""

    def __init__(self):
        self.components = {}
        self.repo_root = INSTALL_DIR.parent
        self.install_dir = INSTALL_DIR

        self.conda_manager = CondaEnvironmentManager()
        self.state_manager = InstallationStateManager(self.install_dir / "install_state.json")
        self.checkpoint_downloader = CheckpointDownloader(self.install_dir)
        self.validator = InstallationValidator(self.conda_manager, self.install_dir)
        self.config_generator = ConfigurationGenerator(self.conda_manager, self.install_dir)
        self.setup_components()

    def setup_components(self):
        """Define all installable components."""

        # Core dependencies
        self.components['core'] = {
            'name': 'Core Pipeline',
            'required': True,
            'installers': [
                PythonPackageInstaller('NumPy', 'numpy', size_gb=0.1),
                PythonPackageInstaller('OpenCV', 'opencv-python', 'cv2', size_gb=0.3),
                PythonPackageInstaller('Pillow', 'pillow', 'PIL', size_gb=0.05),
            ]
        }

        # Web GUI dependencies (FastAPI backend)
        self.components['web_gui'] = {
            'name': 'Web GUI',
            'required': True,
            'installers': [
                PythonPackageInstaller('FastAPI', 'fastapi', size_gb=0.02),
                PythonPackageInstaller('Uvicorn', 'uvicorn', size_gb=0.01),
                PythonPackageInstaller('Python-Multipart', 'python-multipart', size_gb=0.01),
                PythonPackageInstaller('WebSockets', 'websockets', size_gb=0.01),
                PythonPackageInstaller('Jinja2', 'jinja2', size_gb=0.01),
            ]
        }

        # PyTorch (special handling for CUDA)
        self.components['pytorch'] = {
            'name': 'PyTorch',
            'required': True,
            'installers': [
                PythonPackageInstaller('PyTorch', 'torch', size_gb=6.0),  # With CUDA
            ]
        }

        # COLMAP (installed via apt on Linux)
        self.components['colmap'] = {
            'name': 'COLMAP',
            'required': False,
            'installers': [
                SystemPackageInstaller('COLMAP', 'colmap', size_gb=0.5),
            ],
            'size_gb': 0.5,
        }

        # Motion capture dependencies
        self.components['mocap_core'] = {
            'name': 'Motion Capture Core',
            'required': False,
            'installers': [
                PythonPackageInstaller('SMPL-X', 'smplx', size_gb=0.1),
                PythonPackageInstaller('Trimesh', 'trimesh', size_gb=0.05),
            ]
        }

        # WHAM (code ~0.1GB + checkpoints ~2.5GB)
        self.components['wham'] = {
            'name': 'WHAM',
            'required': False,
            'installers': [
                GitRepoInstaller(
                    'WHAM',
                    'https://github.com/yohanshin/WHAM.git',
                    self.install_dir / "WHAM",
                    size_gb=3.0  # Code + checkpoints
                )
            ]
        }

        # ComfyUI and custom nodes
        comfyui_dir = self.install_dir / "ComfyUI"
        self.components['comfyui'] = {
            'name': 'ComfyUI',
            'required': False,
            'size_gb': 1.0,  # Video Depth Anything model (downloaded automatically)
            'installers': [
                GitRepoInstaller(
                    'ComfyUI',
                    'https://github.com/comfyanonymous/ComfyUI.git',
                    comfyui_dir,
                    size_gb=2.0
                ),
                GitRepoInstaller(
                    'ComfyUI-VideoHelperSuite',
                    'https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git',
                    comfyui_dir / "custom_nodes" / "ComfyUI-VideoHelperSuite",
                    size_gb=0.1
                ),
                GitRepoInstaller(
                    'ComfyUI-Video-Depth-Anything',
                    'https://github.com/yuvraj108c/ComfyUI-Video-Depth-Anything.git',
                    comfyui_dir / "custom_nodes" / "ComfyUI-Video-Depth-Anything",
                    size_gb=0.1
                ),
                GitRepoInstaller(
                    'ComfyUI-SAM3',
                    'https://github.com/PozzettiAndrea/ComfyUI-SAM3.git',
                    comfyui_dir / "custom_nodes" / "ComfyUI-SAM3",
                    size_gb=3.5,  # ~3.2GB model + code
                ),
                GitRepoInstaller(
                    'ComfyUI-ProPainter-Nodes',
                    'https://github.com/daniabib/ComfyUI_ProPainter_Nodes.git',
                    comfyui_dir / "custom_nodes" / "ComfyUI_ProPainter_Nodes",
                    size_gb=1.5,  # Models auto-downloaded
                )
            ]
        }

    def setup_conda_environment(self) -> bool:
        """Set up conda environment."""
        print_header("Conda Environment Setup")

        # Check conda
        is_ready, message = self.conda_manager.check_setup()
        if not is_ready:
            print_error(message)
            print_info("Install from: https://docs.conda.io/en/latest/miniconda.html")
            return False

        print_info(message)

        # Create environment if needed
        if not self.conda_manager.environment_exists(self.conda_manager.env_name):
            print_info(f"Creating dedicated environment '{self.conda_manager.env_name}'...")
            if not self.conda_manager.create_environment():
                return False
            self.state_manager.set_environment(self.conda_manager.env_name)
        else:
            print_success(f"Environment '{self.conda_manager.env_name}' already exists")

        # Show activation command
        current_env = self.conda_manager.get_current_env()
        if current_env != self.conda_manager.env_name:
            print("\n" + "="*60)
            print_info("After installation, activate the environment:")
            print(f"  {Colors.OKBLUE}{self.conda_manager.get_activation_command()}{Colors.ENDC}")
            print("="*60 + "\n")

        return True

    def check_system_requirements(self):
        """Check system requirements."""
        print_header("System Requirements Check")

        # Python version
        py_version = sys.version_info
        if py_version >= (3, 8):
            print_success(f"Python {py_version.major}.{py_version.minor}.{py_version.micro}")
        else:
            print_error(f"Python {py_version.major}.{py_version.minor} (3.8+ required)")
            return False

        # Conda
        if self.conda_manager.detect_conda():
            print_success(f"Conda available ({self.conda_manager.conda_exe})")
        else:
            print_error("Conda not found (required for environment management)")
            print_info("Install from: https://docs.conda.io/en/latest/miniconda.html")
            return False

        # Git
        if check_command_available("git"):
            print_success("Git available")
        else:
            print_error("Git not found (required for cloning repositories)")
            return False

        # Disk space
        available_gb, total_gb = get_disk_space()
        if available_gb > 0:
            used_pct = ((total_gb - available_gb) / total_gb) * 100
            if available_gb >= 50:
                print_success(f"Disk space: {format_size_gb(available_gb)} available ({used_pct:.0f}% used)")
            elif available_gb >= 20:
                print_warning(f"Disk space: {format_size_gb(available_gb)} available ({used_pct:.0f}% used)")
                print_info("Full installation requires ~40 GB")
            else:
                print_error(f"Disk space: {format_size_gb(available_gb)} available ({used_pct:.0f}% used)")
                print_info("Insufficient space - full installation requires ~40 GB")
                return False
        else:
            print_warning("Could not check disk space")

        # GPU
        has_gpu, gpu_info = check_gpu_available()
        if has_gpu:
            print_success(f"GPU: {gpu_info}")
        else:
            print_warning("No GPU detected - CPU-only mode (slower)")
            print_info("Motion capture requires NVIDIA GPU with 12GB+ VRAM")

        # ffmpeg
        if check_command_available("ffmpeg"):
            print_success("ffmpeg available")
        else:
            print_warning("ffmpeg not found (required for video ingestion)")
            print_info("Install: sudo apt install ffmpeg")

        # COLMAP
        if check_command_available("colmap"):
            print_success("COLMAP available")
        else:
            print_warning("COLMAP not found (optional, for 3D reconstruction)")
            print_info("Install: sudo apt install colmap")

        return True

    def check_all_components(self) -> Dict[str, bool]:
        """Check status of all components."""
        status = {}

        for comp_id, comp_info in self.components.items():
            all_installed = True
            for installer in comp_info['installers']:
                # Set conda manager for environment-aware checking
                if hasattr(installer, 'set_conda_manager'):
                    installer.set_conda_manager(self.conda_manager)
                if not installer.check():
                    all_installed = False
            status[comp_id] = all_installed

        return status

    def print_status(self, status: Dict[str, bool]):
        """Print installation status."""
        print_header("Installation Status")

        for comp_id, comp_info in self.components.items():
            is_installed = status.get(comp_id, False)
            required = " (required)" if comp_info['required'] else ""

            if is_installed:
                print_success(f"{comp_info['name']}{required}")
            else:
                if comp_info['required']:
                    print_error(f"{comp_info['name']}{required}")
                else:
                    print_warning(f"{comp_info['name']} - not installed")

    def calculate_space_needed(self, component_ids: List[str]) -> float:
        """Calculate total disk space needed for components.

        Args:
            component_ids: List of component IDs to install

        Returns:
            Total disk space needed in GB
        """
        total_gb = 0.0
        for comp_id in component_ids:
            comp_info = self.components.get(comp_id, {})
            for installer in comp_info.get('installers', []):
                total_gb += installer.size_gb
            # Add component-level size (for things like COLMAP)
            total_gb += comp_info.get('size_gb', 0.0)
        return total_gb

    def show_space_estimate(self, component_ids: List[str]):
        """Show disk space estimate for installation.

        Args:
            component_ids: List of component IDs to install
        """
        print_header("Disk Space Estimate")

        # Calculate per-component
        breakdown = []
        total_gb = 0.0

        for comp_id in component_ids:
            comp_info = self.components.get(comp_id, {})
            comp_size = sum(inst.size_gb for inst in comp_info.get('installers', []))
            comp_size += comp_info.get('size_gb', 0.0)

            if comp_size > 0:
                breakdown.append((comp_info['name'], comp_size))
                total_gb += comp_size

        # Sort by size (largest first)
        breakdown.sort(key=lambda x: x[1], reverse=True)

        # Print breakdown
        for name, size_gb in breakdown:
            print(f"  {name:30s} {format_size_gb(size_gb):>10s}")

        print("  " + "-" * 42)
        print(f"  {'Total':30s} {format_size_gb(total_gb):>10s}")

        # Additional space for working data
        working_space = 10.0  # ~10 GB per project
        print(f"\n  {'Working space (per project)':30s} ~{format_size_gb(working_space):>9s}")
        print(f"  {'Recommended total':30s} ~{format_size_gb(total_gb + working_space):>9s}")

        # Check available space
        available_gb, _ = get_disk_space()
        if available_gb > 0:
            print(f"\n  Available disk space: {format_size_gb(available_gb)}")

            if available_gb >= total_gb + working_space:
                print_success("Sufficient disk space available")
            elif available_gb >= total_gb:
                print_warning("Sufficient for installation, but limited working space")
            else:
                print_error(f"Insufficient disk space (need {format_size_gb(total_gb)})")
                return False

        print()
        return True

    def install_component(self, comp_id: str) -> bool:
        """Install a component."""
        comp_info = self.components[comp_id]

        print(f"\n{Colors.BOLD}Installing {comp_info['name']}...{Colors.ENDC}")

        # Check if already completed AND actually present on disk
        status = self.state_manager.get_component_status(comp_id)
        if status == "completed":
            # Verify files actually exist before trusting state
            all_present = True
            for installer in comp_info['installers']:
                if hasattr(installer, 'set_conda_manager'):
                    installer.set_conda_manager(self.conda_manager)
                if not installer.check():
                    all_present = False
                    break
            if all_present:
                print_success(f"{comp_info['name']} already installed (from previous run)")
                return True
            else:
                print_warning(f"{comp_info['name']} marked complete but files missing, reinstalling...")

        # Mark as started
        self.state_manager.mark_component_started(comp_id)

        success = True
        try:
            for installer in comp_info['installers']:
                # Set conda manager for environment-aware installation
                if hasattr(installer, 'set_conda_manager'):
                    installer.set_conda_manager(self.conda_manager)

                if not installer.check():
                    if not installer.install():
                        success = False
                        break
                else:
                    print_success(f"{installer.name} already installed")

            if success:
                self.state_manager.mark_component_completed(comp_id)
            else:
                self.state_manager.mark_component_failed(comp_id, "Installation failed")

        except Exception as e:
            self.state_manager.mark_component_failed(comp_id, str(e))
            print_error(f"Error installing {comp_info['name']}: {e}")
            success = False

        return success

    def setup_credentials(self, repo_root: Path) -> None:
        """Prompt user to set up credentials for authenticated downloads.

        Sets up:
        - HF_TOKEN.dat for HuggingFace (SAM3, etc.)
        - SMPL.login.dat for SMPL-X body models (motion capture)
        """
        print_header("Credentials Setup")
        print("Some components require authentication to download:")
        print("  - SAM3 segmentation model (HuggingFace)")
        print("  - SMPL-X body models (smpl-x.is.tue.mpg.de)")
        print("")

        # Check existing credentials
        hf_token_file = repo_root / "HF_TOKEN.dat"
        smpl_creds_file = repo_root / "SMPL.login.dat"

        hf_exists = hf_token_file.exists()
        smpl_exists = smpl_creds_file.exists()

        if hf_exists and smpl_exists:
            print_success("All credential files already exist")
            if ask_yes_no("Update credentials?", default=False):
                hf_exists = False
                smpl_exists = False
            else:
                return

        # HuggingFace token setup
        if not hf_exists:
            print(f"\n{Colors.BOLD}HuggingFace Token Setup{Colors.ENDC}")
            print("Required for: SAM3 segmentation model")
            print("")
            print("Steps:")
            print("  1. Go to https://huggingface.co/facebook/sam3")
            print("  2. Click 'Access repository' and accept the license")
            print("  3. Go to https://huggingface.co/settings/tokens")
            print("  4. Create a new token with 'Read' access")
            print("")

            if ask_yes_no("Set up HuggingFace token now?", default=True):
                token = tty_input("Enter your HuggingFace token (hf_...): ").strip()
                if token:
                    if token.startswith("hf_") or len(token) > 20:
                        with open(hf_token_file, 'w') as f:
                            f.write(token + '\n')
                        hf_token_file.chmod(0o600)
                        print_success(f"Token saved to {hf_token_file}")
                    else:
                        print_warning("Token looks invalid (should start with 'hf_')")
                        if ask_yes_no("Save anyway?", default=False):
                            with open(hf_token_file, 'w') as f:
                                f.write(token + '\n')
                            hf_token_file.chmod(0o600)
                            print_success(f"Token saved to {hf_token_file}")
                else:
                    print_info("Skipped - you can add HF_TOKEN.dat later")
            else:
                print_info("Skipped - you can add HF_TOKEN.dat later")

        # SMPL-X credentials setup
        if not smpl_exists:
            print(f"\n{Colors.BOLD}SMPL-X Credentials Setup{Colors.ENDC}")
            print("Required for: Body model (skeleton, mesh topology, UV layout)")
            print("")
            print("SMPL-X provides the parametric body model - the 'rigged character'")
            print("that can be posed and animated with consistent vertex ordering.")
            print("")
            print("Registration: https://smpl-x.is.tue.mpg.de/")
            print("")
            print("Steps:")
            print("  1. Register at the website above")
            print("  2. Wait for approval email (usually within 24-48 hours)")
            print("  3. Enter your credentials below")
            print("")

            if ask_yes_no("Set up SMPL-X credentials now?", default=True):
                email = tty_input("Enter your SMPL-X registered email: ").strip()
                if email and '@' in email:
                    password = tty_input("Enter your SMPL-X password: ").strip()
                    if password:
                        with open(smpl_creds_file, 'w') as f:
                            f.write(email + '\n')
                            f.write(password + '\n')
                        # Set restrictive permissions
                        smpl_creds_file.chmod(0o600)
                        print_success(f"Credentials saved to {smpl_creds_file}")
                    else:
                        print_info("Skipped - you can add SMPL.login.dat later")
                else:
                    print_info("Skipped - you can add SMPL.login.dat later")
            else:
                print_info("Skipped - you can add SMPL.login.dat later")

        print("")

    def interactive_install(self, component: Optional[str] = None, resume: bool = False):
        """Interactive installation flow."""
        print_header("VFX Pipeline Installation Wizard")

        # Check for resumable installation
        if not resume and self.state_manager.can_resume():
            incomplete = self.state_manager.get_incomplete_components()
            print_warning("Found incomplete installation from previous run:")
            for comp_id in incomplete:
                status = self.state_manager.get_component_status(comp_id)
                print(f"  - {self.components.get(comp_id, {}).get('name', comp_id)}: {status}")

            if ask_yes_no("\nResume previous installation?", default=True):
                resume = True
            else:
                if ask_yes_no("Start fresh (clear previous state)?", default=False):
                    self.state_manager.clear_state()

        # System requirements
        if not self.check_system_requirements():
            print_error("\nSystem requirements not met. Please install missing components.")
            return False

        # Conda environment setup
        if not self.setup_conda_environment():
            return False

        # Check current status
        status = self.check_all_components()
        self.print_status(status)

        # Set up credentials for authenticated downloads
        self.setup_credentials(self.repo_root)

        # Determine what to install
        if component:
            # Specific component requested
            if component not in self.components:
                print_error(f"Unknown component: {component}")
                print_info(f"Available: {', '.join(self.components.keys())}")
                return False

            to_install = [component]
        else:
            # Interactive selection
            print("\n" + "="*60)
            print("What would you like to install?")
            print("="*60)
            print("1. Core pipeline only (COLMAP, segmentation)")
            print("2. Core + ComfyUI (workflows ready to use)")
            print("3. Full stack (Core + ComfyUI + Motion capture)")
            print("4. Custom selection")
            print("5. Nothing (check only)")

            while True:
                choice = tty_input("\nChoice [1-5]: ").strip()
                if choice == '1':
                    to_install = ['core', 'web_gui', 'pytorch', 'colmap']
                    break
                elif choice == '2':
                    to_install = ['core', 'web_gui', 'pytorch', 'colmap', 'comfyui']
                    break
                elif choice == '3':
                    to_install = ['core', 'web_gui', 'pytorch', 'colmap', 'comfyui', 'mocap_core', 'wham']
                    break
                elif choice == '4':
                    to_install = []
                    for comp_id, comp_info in self.components.items():
                        if ask_yes_no(f"Install {comp_info['name']}?", default=False):
                            to_install.append(comp_id)
                    break
                elif choice == '5':
                    return True
                else:
                    print("Invalid choice")

        # Show disk space estimate
        if to_install:
            if not self.show_space_estimate(to_install):
                print_error("\nInsufficient disk space for installation")
                return False

            # Confirm installation
            if not ask_yes_no("\nProceed with installation?", default=True):
                print_info("Installation cancelled")
                return True

        # Install components
        print_header("Installing Components")

        for comp_id in to_install:
            if not status.get(comp_id, False):
                if not self.install_component(comp_id):
                    print_error(f"Failed to install {self.components[comp_id]['name']}")
                    if self.components[comp_id]['required']:
                        return False

        # Download Video Depth Anything model (required for ComfyUI depth workflows)
        if 'comfyui' in to_install:
            print("\nDownloading Video Depth Anything model (for temporally consistent depth)...")
            self.checkpoint_downloader.download_all_checkpoints(['video_depth_anything'], self.state_manager)

            # Download SAM3 model for segmentation
            print("\nDownloading SAM3 model (for segmentation/roto workflows)...")
            self.checkpoint_downloader.download_all_checkpoints(['sam3'], self.state_manager)

        # Download checkpoints for motion capture components
        mocap_components = [cid for cid in to_install if cid in ['wham']]
        if mocap_components:
            if ask_yes_no("\nDownload checkpoints for motion capture components?", default=True):
                self.checkpoint_downloader.download_all_checkpoints(mocap_components, self.state_manager)

        # Final status
        final_status = self.check_all_components()
        self.print_status(final_status)

        # Generate configuration files
        if to_install:
            self.config_generator.generate_all()

        # Run validation tests
        if to_install and ask_yes_no("\nRun installation validation tests?", default=True):
            self.validator.validate_and_report()

        # Post-installation instructions
        self.print_post_install_instructions(final_status)

        return True

    def print_post_install_instructions(self, status: Dict[str, bool]):
        """Print post-installation instructions."""
        print_header("Next Steps")

        # SMPL-X models (manual download required - registration needed)
        if status.get('mocap_core', False):
            print("\n📦 SMPL-X Body Models (Manual Download Required):")
            print("  1. Register at https://smpl-x.is.tue.mpg.de/")
            print("  2. Download SMPL-X models")
            print(f"  3. Place in {INSTALL_DIR}/smplx_models/")
            print(f"     mkdir -p {INSTALL_DIR}/smplx_models && cp SMPLX_*.pkl {INSTALL_DIR}/smplx_models/")

        # Checkpoints status
        if status.get('wham', False):
            print("\n📦 Motion Capture Checkpoints:")
            if self.state_manager.is_checkpoint_downloaded('wham'):
                print("  ✓ WHAM checkpoints downloaded")
            else:
                print("  ⚠ WHAM checkpoints not downloaded - run wizard again or visit:")
                print("    https://github.com/yohanshin/WHAM")

        # ComfyUI
        if status.get('comfyui', False):
            comfyui_path = self.install_dir / "ComfyUI"
            print("\n🎨 ComfyUI:")
            print(f"  ✓ Installed at {comfyui_path}")
            print("  ✓ Custom nodes installed")
            print("\n  Start server:")
            print(f"    cd {comfyui_path}")
            print("    python main.py --listen")
        else:
            print("\n🎨 ComfyUI (Optional):")
            print("  Not installed. Run wizard again to add ComfyUI support.")

        # Web GUI
        if status.get('web_gui', False):
            print("\n🌐 Web GUI:")
            print("  ✓ FastAPI and dependencies installed")
            print("\n  Start web interface:")
            print("    ./start_web.py")
            print("  Or with custom port:")
            print("    ./start_web.py --port 8080")

        # Testing
        print("\n✅ Test Installation:")
        print("  python scripts/run_pipeline.py --help")
        print("  python scripts/run_mocap.py --check")

        # Documentation
        print("\n📖 Documentation:")
        print("  README.md - Pipeline overview and usage")
        print("  TESTING.md - Testing and validation guide")
        print("  IMPLEMENTATION_NOTES.md - Developer notes")
