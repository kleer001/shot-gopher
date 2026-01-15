"""Checkpoint downloading for the installation wizard.

This module handles downloading model checkpoints from various sources
including Google Drive, HuggingFace, and authenticated download servers.
"""

import hashlib
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from env_config import INSTALL_DIR

from .utils import Colors, print_error, print_header, print_info, print_success, print_warning

if TYPE_CHECKING:
    from .state import InstallationStateManager


class CheckpointDownloader:
    """Handles automatic checkpoint downloading."""

    # Checkpoint metadata
    CHECKPOINTS: Dict = {
        'wham': {
            'name': 'WHAM Checkpoints',
            'use_gdown': True,  # Use gdown for Google Drive downloads
            'files': [
                {
                    'url': 'https://drive.google.com/uc?id=1i7kt9RlCCCNEW2aYaDWVr-G778JkLNcB',
                    'filename': 'wham_vit_w_3dpw.pth.tar',
                    'size_mb': 1200,
                    'sha256': None  # TODO: Add checksums
                }
            ],
            'dest_dir_rel': 'WHAM/checkpoints',
            'instructions': f'''WHAM checkpoints are hosted on Google Drive.
If automatic download fails, manually download from:
  https://drive.google.com/file/d/1i7kt9RlCCCNEW2aYaDWVr-G778JkLNcB/view
Or run the fetch_demo_data.sh script from the WHAM repository:
  cd {INSTALL_DIR}/WHAM && bash fetch_demo_data.sh'''
        },
        'smplx': {
            'name': 'SMPL-X Models',
            'requires_auth': True,
            'auth_type': 'smplx',  # Special handling for SMPL-X cross-domain auth
            'auth_file': 'SMPL.login.dat',
            'login_url': 'https://smpl-x.is.tue.mpg.de/login.php',
            'download_page': 'https://smpl-x.is.tue.mpg.de/download.php',
            'files': [
                {
                    'url': 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip',
                    'filename': 'models_smplx_v1_1.zip',
                    'size_mb': 830,
                    'sha256': None,
                    'extract': True
                }
            ],
            'dest_dir_rel': 'smplx_models',
            'use_home_dir': False,
            'instructions': '''SMPL-X models require registration:
1. Register at https://smpl-x.is.tue.mpg.de/register.php
2. Wait for approval email (usually within 24-48 hours)
3. Create SMPL.login.dat in repository root with:
   Line 1: your email
   Line 2: your password
4. Re-run the wizard to download models'''
        },
        'sam3': {
            'name': 'SAM3 Model',
            'requires_auth': False,  # Public repo at 1038lab/sam3
            'use_huggingface': True,
            'hf_repo_id': '1038lab/sam3',
            'files': [
                {
                    'filename': 'sam3.pt',
                    'size_mb': 3200,  # ~3.2GB model
                }
            ],
            'dest_dir_rel': 'ComfyUI/models/sam3',
            'instructions': '''SAM3 model will be downloaded from HuggingFace (1038lab/sam3).
The model is publicly accessible and will be placed in ComfyUI/models/sam3/.
If automatic download fails, manually download from:
  https://huggingface.co/1038lab/sam3/blob/main/sam3.pt'''
        },
        'video_depth_anything': {
            'name': 'Video Depth Anything Model',
            'requires_auth': False,  # Public model
            'use_huggingface': True,  # Use huggingface_hub snapshot_download
            'hf_repo_id': 'depth-anything/Video-Depth-Anything-Small',
            'files': [
                {
                    'filename': 'video_depth_anything_vits.pth',
                    'size_mb': 120,  # Small model is ~116MB
                }
            ],
            'dest_dir_rel': 'ComfyUI/models/videodepthanything',
            'instructions': '''Video Depth Anything Small model will be downloaded from HuggingFace.
This model uses ~6.8GB VRAM (vs 23.6GB for Large), suitable for most GPUs.'''
        },
        'matanyone': {
            'name': 'MatAnyone Model',
            'requires_auth': False,  # Public model
            'use_huggingface': True,
            'hf_repo_id': 'not-lain/matanyone-files',  # ComfyUI-compatible repo with matanyone.pth
            'files': [
                {
                    'filename': 'matanyone.pth',
                    'size_mb': 141,  # ~141MB model
                }
            ],
            'dest_dir_rel': 'ComfyUI/custom_nodes/ComfyUI-MatAnyone/checkpoint',
            'instructions': '''MatAnyone model for stable video matting.
Download from HuggingFace: https://huggingface.co/not-lain/matanyone-files/resolve/main/matanyone.pth
Place in ComfyUI/custom_nodes/ComfyUI-MatAnyone/checkpoint/matanyone.pth'''
        }
    }

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or INSTALL_DIR

    def download_from_huggingface(
        self,
        repo_id: str,
        dest_dir: Path,
        target_filename: str,
        token: Optional[str] = None,
    ) -> bool:
        """Download model from HuggingFace using snapshot_download.

        Downloads to HuggingFace cache, then copies the model file
        to the destination directory with the specified filename.

        Args:
            repo_id: HuggingFace repository ID (e.g., 'depth-anything/Metric-Video-Depth-Anything-Large')
            dest_dir: Destination directory for the model file
            target_filename: Filename to save as (e.g., 'metric_video_depth_anything_vitl.pth')
            token: Optional HuggingFace token for gated models

        Returns:
            True if successful
        """
        import shutil

        dest_path = dest_dir / target_filename

        # Check if already exists
        if dest_path.exists():
            print_success(f"{target_filename} already exists")
            return True

        print(f"  Downloading {repo_id} from HuggingFace...")

        # Determine file pattern based on target extension
        ext = Path(target_filename).suffix  # .pth, .safetensors, etc.
        pattern = f"*{ext}" if ext else "*.pth"

        try:
            from huggingface_hub import snapshot_download

            # Download to HuggingFace cache (shows progress bar)
            cache_dir = snapshot_download(
                repo_id=repo_id,
                allow_patterns=[pattern],
                token=token,  # Pass token for gated models
            )

            # Find the downloaded model file
            cache_path = Path(cache_dir)
            model_files = list(cache_path.glob(pattern))

            if model_files:
                # Ensure destination directory exists
                dest_dir.mkdir(parents=True, exist_ok=True)

                # Copy the file (repos typically have one model file)
                src_file = model_files[0]
                shutil.copy2(src_file, dest_path)
                print_success(f"Downloaded: {target_filename}")
                return True
            else:
                print_error(f"No {ext} file found in {repo_id}")
                return False

        except ImportError:
            print_error("huggingface_hub not installed")
            print_info("Install with: pip install huggingface_hub")
            return False
        except Exception as e:
            print_error(f"Download failed: {e}")
            return False

    def download_file(self, url: str, dest: Path, expected_size_mb: Optional[int] = None) -> bool:
        """Download file with progress tracking.

        Args:
            url: URL to download from
            dest: Destination file path
            expected_size_mb: Expected file size in MB (for validation)

        Returns:
            True if successful
        """
        try:
            import requests
        except ImportError:
            print_warning("requests library not found, installing...")
            subprocess.run([sys.executable, "-m", "pip", "install", "requests"], check=True)
            import requests

        try:
            print(f"  Downloading from {url}...")
            print(f"  -> {dest}")

            # Ensure directory exists
            dest.parent.mkdir(parents=True, exist_ok=True)

            # Stream download with progress
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(dest, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Simple progress indicator
                        if total_size > 0:
                            pct = (downloaded / total_size) * 100
                            mb_downloaded = downloaded / (1024 * 1024)
                            mb_total = total_size / (1024 * 1024)
                            progress_msg = f"\r  Progress: {pct:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)"
                            print(progress_msg, end='', flush=True)

            print()  # New line after progress
            print_success(f"Downloaded {dest.name}")
            return True

        except requests.exceptions.RequestException as e:
            print_error(f"Download failed: {e}")
            # Clean up partial download
            if dest.exists():
                dest.unlink()
            return False

    def verify_checksum(self, file_path: Path, expected_sha256: str) -> bool:
        """Verify file checksum.

        Args:
            file_path: Path to file
            expected_sha256: Expected SHA256 hash

        Returns:
            True if checksum matches
        """
        if not expected_sha256:
            return True  # Skip verification if no checksum provided

        print(f"  Verifying checksum for {file_path.name}...")
        sha256_hash = hashlib.sha256()

        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    sha256_hash.update(chunk)

            actual = sha256_hash.hexdigest()
            if actual == expected_sha256:
                print_success("Checksum verified")
                return True
            else:
                print_error(f"Checksum mismatch: expected {expected_sha256}, got {actual}")
                return False

        except IOError as e:
            print_error(f"Could not read file for checksum: {e}")
            return False

    def _install_gdown(self) -> Optional['module']:
        """Install gdown handling PEP 668 externally-managed environments.

        Tries multiple installation methods in order:
        1. pipx (preferred for CLI tools)
        2. pip with --user flag
        3. pip with --break-system-packages (last resort)

        Returns:
            The gdown module if successful, None otherwise
        """
        install_methods = [
            # Method 1: Try pipx (best for externally-managed environments)
            {
                'name': 'pipx',
                'cmd': ['pipx', 'install', 'gdown'],
                'check_cmd': ['pipx', 'list'],
            },
            # Method 2: Try pip with --user flag
            {
                'name': 'pip --user',
                'cmd': [sys.executable, '-m', 'pip', 'install', '--user', 'gdown'],
                'check_cmd': None,
            },
            # Method 3: Try pip with --break-system-packages (last resort)
            {
                'name': 'pip --break-system-packages',
                'cmd': [sys.executable, '-m', 'pip', 'install', '--break-system-packages', 'gdown'],
                'check_cmd': None,
            },
        ]

        for method in install_methods:
            # Check if the tool is available (for pipx)
            if method['check_cmd']:
                check_result = subprocess.run(
                    method['check_cmd'],
                    capture_output=True,
                    text=True
                )
                if check_result.returncode != 0:
                    continue

            print_info(f"Trying to install gdown via {method['name']}...")
            result = subprocess.run(
                method['cmd'],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                print_success(f"Installed gdown via {method['name']}")
                try:
                    import gdown
                    return gdown
                except ImportError:
                    # pipx installs to a different location, need to use subprocess
                    # Check if gdown CLI is available
                    gdown_check = subprocess.run(
                        ['gdown', '--version'],
                        capture_output=True,
                        text=True
                    )
                    if gdown_check.returncode == 0:
                        # gdown CLI is available, use wrapper class
                        return self._create_gdown_cli_wrapper()
                    continue
            else:
                # Check for PEP 668 error and try next method
                if 'externally-managed-environment' in result.stderr:
                    continue
                # Other errors, log and try next method
                print_warning(f"Failed with {method['name']}: {result.stderr[:100]}")

        # All methods failed - provide manual instructions
        print_error("Could not install gdown automatically.")
        print_info("To install gdown manually, try one of these options:")
        print("  1. Using pipx (recommended): pipx install gdown")
        print("  2. Using pip with user flag: pip install --user gdown")
        print("  3. In a virtual environment: python -m venv venv && source venv/bin/activate && pip install gdown")
        return None

    def _create_gdown_cli_wrapper(self):
        """Create a wrapper object that mimics gdown module using CLI."""
        class GdownCLIWrapper:
            @staticmethod
            def download(url, output, quiet=False):
                cmd = ['gdown', url, '-O', output]
                if quiet:
                    cmd.append('--quiet')
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    return output
                return None
        return GdownCLIWrapper()

    def _download_gdrive_wget(self, file_id: str, dest: Path) -> bool:
        """Download from Google Drive using wget (fallback method).

        Uses the confirmation cookie trick for large files.

        Args:
            file_id: Google Drive file ID
            dest: Destination file path

        Returns:
            True if successful
        """
        # Check if wget is available
        wget_check = subprocess.run(['which', 'wget'], capture_output=True)
        if wget_check.returncode != 0:
            return False

        print_info("Trying wget fallback for Google Drive...")

        # Google Drive download URL with confirmation bypass
        # For large files, we need to handle the virus scan warning
        base_url = "https://drive.google.com/uc?export=download"
        confirm_url = f"{base_url}&id={file_id}&confirm=t"

        try:
            dest.parent.mkdir(parents=True, exist_ok=True)

            # Use wget with appropriate flags for Google Drive
            cmd = [
                'wget',
                '--no-check-certificate',
                '-q', '--show-progress',
                '-O', str(dest),
                confirm_url
            ]

            result = subprocess.run(cmd, capture_output=False)

            if result.returncode == 0 and dest.exists() and dest.stat().st_size > 1000:
                print_success("Downloaded via wget")
                return True

            # If file is too small, might be HTML error page
            if dest.exists():
                dest.unlink()
            return False

        except Exception as e:
            print_warning(f"wget fallback failed: {e}")
            if dest.exists():
                dest.unlink()
            return False

    def _download_gdrive_curl(self, file_id: str, dest: Path) -> bool:
        """Download from Google Drive using curl (fallback method).

        Args:
            file_id: Google Drive file ID
            dest: Destination file path

        Returns:
            True if successful
        """
        # Check if curl is available
        curl_check = subprocess.run(['which', 'curl'], capture_output=True)
        if curl_check.returncode != 0:
            return False

        print_info("Trying curl fallback for Google Drive...")

        # Google Drive download URL with confirmation bypass
        confirm_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"

        try:
            dest.parent.mkdir(parents=True, exist_ok=True)

            # Use curl with appropriate flags
            cmd = [
                'curl',
                '-L',  # Follow redirects
                '-o', str(dest),
                '--progress-bar',
                confirm_url
            ]

            result = subprocess.run(cmd)

            if result.returncode == 0 and dest.exists() and dest.stat().st_size > 1000:
                print_success("Downloaded via curl")
                return True

            if dest.exists():
                dest.unlink()
            return False

        except Exception as e:
            print_warning(f"curl fallback failed: {e}")
            if dest.exists():
                dest.unlink()
            return False

    def _download_with_wget_auth(self, url: str, dest: Path, username: str, password: str) -> bool:
        """Download file using wget with HTTP basic auth (fallback).

        Args:
            url: URL to download
            dest: Destination file path
            username: HTTP basic auth username
            password: HTTP basic auth password

        Returns:
            True if successful
        """
        wget_check = subprocess.run(['which', 'wget'], capture_output=True)
        if wget_check.returncode != 0:
            return False

        print_info("Trying wget fallback with authentication...")

        try:
            dest.parent.mkdir(parents=True, exist_ok=True)

            cmd = [
                'wget',
                '--no-check-certificate',
                '-q', '--show-progress',
                f'--user={username}',
                f'--password={password}',
                '-O', str(dest),
                url
            ]

            result = subprocess.run(cmd)

            if result.returncode == 0 and dest.exists() and dest.stat().st_size > 1000:
                print_success("Downloaded via wget")
                return True

            if dest.exists():
                dest.unlink()
            return False

        except Exception as e:
            print_warning(f"wget auth fallback failed: {e}")
            if dest.exists():
                dest.unlink()
            return False

    def _download_with_curl_auth(self, url: str, dest: Path, username: str, password: str) -> bool:
        """Download file using curl with HTTP basic auth (fallback).

        Args:
            url: URL to download
            dest: Destination file path
            username: HTTP basic auth username
            password: HTTP basic auth password

        Returns:
            True if successful
        """
        curl_check = subprocess.run(['which', 'curl'], capture_output=True)
        if curl_check.returncode != 0:
            return False

        print_info("Trying curl fallback with authentication...")

        try:
            dest.parent.mkdir(parents=True, exist_ok=True)

            cmd = [
                'curl',
                '-L',
                '-u', f'{username}:{password}',
                '-o', str(dest),
                '--progress-bar',
                url
            ]

            result = subprocess.run(cmd)

            if result.returncode == 0 and dest.exists() and dest.stat().st_size > 1000:
                print_success("Downloaded via curl")
                return True

            if dest.exists():
                dest.unlink()
            return False

        except Exception as e:
            print_warning(f"curl auth fallback failed: {e}")
            if dest.exists():
                dest.unlink()
            return False

    def _extract_gdrive_file_id(self, url: str) -> Optional[str]:
        """Extract Google Drive file ID from various URL formats.

        Args:
            url: Google Drive URL

        Returns:
            File ID or None
        """
        import re

        # Format: https://drive.google.com/uc?id=FILE_ID
        match = re.search(r'[?&]id=([a-zA-Z0-9_-]+)', url)
        if match:
            return match.group(1)

        # Format: https://drive.google.com/file/d/FILE_ID/view
        match = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
        if match:
            return match.group(1)

        return None

    def download_file_gdown(self, url: str, dest: Path, expected_size_mb: Optional[int] = None) -> bool:
        """Download file from Google Drive using gdown with wget/curl fallbacks.

        Args:
            url: Google Drive URL (format: https://drive.google.com/uc?id=FILE_ID)
            dest: Destination file path
            expected_size_mb: Expected file size in MB (for info only)

        Returns:
            True if successful
        """
        print(f"  Downloading from Google Drive...")
        print(f"  -> {dest}")
        if expected_size_mb:
            print(f"  Expected size: ~{expected_size_mb} MB")

        # Extract file ID for fallback methods
        file_id = self._extract_gdrive_file_id(url)

        # Method 1: Try gdown (preferred)
        gdown = None
        try:
            import gdown
        except ImportError:
            print_warning("gdown library not found, installing...")
            gdown = self._install_gdown()

        if gdown is not None:
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                output = gdown.download(url, str(dest), quiet=False)

                if output is not None and dest.exists() and dest.stat().st_size > 1000:
                    print_success(f"Downloaded {dest.name}")
                    return True

                print_warning("gdown download failed, trying fallbacks...")
                if dest.exists():
                    dest.unlink()

            except Exception as e:
                print_warning(f"gdown failed: {e}")
                if dest.exists():
                    dest.unlink()

        # Method 2: Try wget fallback
        if file_id:
            if self._download_gdrive_wget(file_id, dest):
                return True

            # Method 3: Try curl fallback
            if self._download_gdrive_curl(file_id, dest):
                return True

        print_error("All download methods failed")
        return False

    def read_credentials(self, repo_root: Path, auth_file: str) -> Optional[Tuple[str, str]]:
        """Read credentials from a login file.

        Args:
            repo_root: Repository root directory
            auth_file: Name of the credentials file (e.g., 'SMPL.login.dat')

        Returns:
            Tuple of (username, password) or None if file not found
        """
        cred_file = repo_root / auth_file

        if not cred_file.exists():
            return None

        try:
            with open(cred_file, 'r') as f:
                lines = f.read().strip().split('\n')
                if len(lines) >= 2:
                    username = lines[0].strip()
                    password = lines[1].strip()
                    return (username, password)
                else:
                    print_error(f"{auth_file} must contain username on line 1 and password on line 2")
                    return None
        except IOError as e:
            print_error(f"Could not read {auth_file}: {e}")
            return None

    def read_hf_token(self, repo_root: Path) -> Optional[str]:
        """Read HuggingFace token from HF_TOKEN.dat file.

        Args:
            repo_root: Repository root directory

        Returns:
            HuggingFace token or None if file not found
        """
        token_file = repo_root / "HF_TOKEN.dat"

        if not token_file.exists():
            return None

        try:
            with open(token_file, 'r') as f:
                token = f.read().strip()
                if token:
                    return token
                else:
                    print_error("HF_TOKEN.dat is empty")
                    return None
        except IOError as e:
            print_error(f"Could not read HF_TOKEN.dat: {e}")
            return None

    def download_file_with_auth(
        self,
        url: str,
        dest: Path,
        auth: Optional[Tuple[str, str]] = None,
        token: Optional[str] = None,
        expected_size_mb: Optional[int] = None
    ) -> bool:
        """Download file with optional HTTP authentication.

        Args:
            url: URL to download from
            dest: Destination file path
            auth: Optional (username, password) tuple for basic auth
            token: Optional bearer token for token-based auth
            expected_size_mb: Expected file size in MB

        Returns:
            True if successful
        """
        try:
            import requests
        except ImportError:
            print_warning("requests library not found, installing...")
            # Try pip with --user to handle PEP 668 externally-managed environments
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--user", "requests"],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                # Fall back to --break-system-packages if --user fails
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--break-system-packages", "requests"],
                    capture_output=True, text=True
                )
                if result.returncode != 0:
                    print_error(f"Failed to install requests: {result.stderr}")
                    return False
            import requests

        try:
            print(f"  Downloading from {url}...")
            print(f"  -> {dest}")

            # Ensure directory exists
            dest.parent.mkdir(parents=True, exist_ok=True)

            # Create session with auth if provided
            session = requests.Session()
            if auth:
                session.auth = auth
            elif token:
                session.headers.update({'Authorization': f'Bearer {token}'})

            # Stream download with progress
            response = session.get(url, stream=True, timeout=30)
            response.raise_for_status()

            # Check content-type to catch HTML error pages early
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' in content_type:
                print_error("Server returned HTML instead of the expected file")
                print_info("This usually means:")
                print("  - Your credentials may be invalid or expired")
                print("  - You may not have been approved for access yet")
                print("  - The download URL may have changed")
                return False

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(dest, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Simple progress indicator
                        if total_size > 0:
                            pct = (downloaded / total_size) * 100
                            mb_downloaded = downloaded / (1024 * 1024)
                            mb_total = total_size / (1024 * 1024)
                            progress_msg = f"\r  Progress: {pct:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)"
                            print(progress_msg, end='', flush=True)

            print()  # New line after progress
            print_success(f"Downloaded {dest.name}")
            return True

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                if token:
                    print_error("Authentication failed - check your HF_TOKEN.dat")
                else:
                    print_error("Authentication failed - check your credentials")
            elif e.response.status_code == 403:
                if token:
                    print_error("Access denied - have you accepted the model license on HuggingFace?")
                else:
                    print_error("Access denied - have you been approved for access?")
            else:
                print_error(f"Download failed: {e}")
            # Clean up partial download
            if dest.exists():
                dest.unlink()
            # Don't try fallbacks for auth errors
            return False
        except requests.exceptions.RequestException as e:
            print_warning(f"requests download failed: {e}")
            # Clean up partial download
            if dest.exists():
                dest.unlink()

            # Try wget/curl fallbacks for basic auth
            if auth:
                username, password = auth
                print_info("Trying alternative download methods...")

                if self._download_with_wget_auth(url, dest, username, password):
                    return True

                if self._download_with_curl_auth(url, dest, username, password):
                    return True

            print_error("All download methods failed")
            return False

    def download_file_with_form_auth(
        self,
        login_url: str,
        download_url: str,
        dest: Path,
        username: str,
        password: str,
        expected_size_mb: Optional[int] = None
    ) -> bool:
        """Download file after form-based login (for sites like SMPL-X).

        Args:
            login_url: URL of the login form
            download_url: URL to download after authentication
            dest: Destination file path
            username: Login username/email
            password: Login password
            expected_size_mb: Expected file size in MB

        Returns:
            True if successful
        """
        try:
            import requests
        except ImportError:
            print_warning("requests library not found, installing...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--user", "requests"],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--break-system-packages", "requests"],
                    capture_output=True, text=True
                )
                if result.returncode != 0:
                    print_error(f"Failed to install requests: {result.stderr}")
                    return False
            import requests

        try:
            print(f"  Logging in to {login_url}...")

            # Create session to maintain cookies
            session = requests.Session()

            # POST login form
            login_data = {
                'username': username,
                'password': password
            }
            login_response = session.post(login_url, data=login_data, timeout=30)

            # Check if login succeeded (usually redirects or returns 200)
            if login_response.status_code not in [200, 302]:
                print_error(f"Login failed with status {login_response.status_code}")
                return False

            # Check for login error in response
            if 'invalid' in login_response.text.lower() or 'error' in login_response.text.lower():
                if 'password' in login_response.text.lower() or 'credentials' in login_response.text.lower():
                    print_error("Login failed - invalid credentials")
                    return False

            print_success("Login successful")
            print(f"  Downloading from {download_url}...")
            print(f"  -> {dest}")

            # Ensure directory exists
            dest.parent.mkdir(parents=True, exist_ok=True)

            # Download file with authenticated session
            response = session.get(download_url, stream=True, timeout=30)
            response.raise_for_status()

            # Check content-type
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' in content_type:
                print_error("Server returned HTML instead of the expected file")
                print_info("This usually means:")
                print("  - Login succeeded but you don't have download access")
                print("  - You may not have been approved for access yet")
                print("  - The download URL may have changed")
                return False

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(dest, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total_size > 0:
                            pct = (downloaded / total_size) * 100
                            mb_downloaded = downloaded / (1024 * 1024)
                            mb_total = total_size / (1024 * 1024)
                            progress_msg = f"\r  Progress: {pct:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)"
                            print(progress_msg, end='', flush=True)

            print()  # New line after progress
            print_success(f"Downloaded {dest.name}")
            return True

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                print_error("Authentication failed - check your credentials")
            elif e.response.status_code == 403:
                print_error("Access denied - have you been approved for access?")
            else:
                print_error(f"Download failed: {e}")
            if dest.exists():
                dest.unlink()
            return False
        except requests.exceptions.RequestException as e:
            print_error(f"Download failed: {e}")
            if dest.exists():
                dest.unlink()
            return False

    def download_smplx_model(
        self,
        login_url: str,
        download_page: str,
        download_url: str,
        dest: Path,
        username: str,
        password: str,
    ) -> bool:
        """Download SMPL-X model with special cross-domain authentication.

        The SMPL-X site requires:
        1. Login at smpl-x.is.tue.mpg.de
        2. Access download page to establish session
        3. Download from download.is.tue.mpg.de (may need token from download page)

        This method first tries Playwright (headless browser) for reliable
        cross-domain authentication, then falls back to requests-based approach.

        Args:
            login_url: URL of the login form
            download_page: URL of the download page (to get session/tokens)
            download_url: Final download URL
            dest: Destination file path
            username: Login email
            password: Login password

        Returns:
            True if successful
        """
        # Try Playwright first (more reliable for cross-domain auth)
        print_info("Attempting download with headless browser (Playwright)...")
        if self._download_smplx_with_playwright(
            login_url, download_page, download_url, dest, username, password
        ):
            return True

        # Fall back to requests-based approach
        print_warning("Playwright method failed, trying requests-based approach...")
        return self._download_smplx_with_requests(
            login_url, download_page, download_url, dest, username, password
        )

    def _download_smplx_with_playwright(
        self,
        login_url: str,
        download_page: str,
        download_url: str,
        dest: Path,
        username: str,
        password: str,
    ) -> bool:
        """Download SMPL-X model using Playwright headless browser.

        Playwright handles cross-domain cookies automatically and works
        even if the site uses JavaScript for authentication.
        """
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            print_warning("Playwright not installed. Installing...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--user", "playwright"],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                print_info(f"pip --user failed, trying --break-system-packages...")
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--break-system-packages", "playwright"],
                    capture_output=True, text=True
                )
            if result.returncode != 0:
                print_warning(f"Could not install playwright package: {result.stderr[:200] if result.stderr else 'unknown error'}")
                return False

            # Install browser binaries
            print_info("Installing Playwright browser (this may take a minute)...")
            browser_result = subprocess.run(
                [sys.executable, "-m", "playwright", "install", "chromium"],
                capture_output=True, text=True
            )
            if browser_result.returncode != 0:
                print_warning(f"Could not install Playwright browser: {browser_result.stderr[:300] if browser_result.stderr else 'unknown error'}")
                return False
            try:
                from playwright.sync_api import sync_playwright
            except ImportError as e:
                print_warning(f"Playwright installation failed: {e}")
                return False

        try:
            with sync_playwright() as p:
                # Launch headless browser
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                )
                page = context.new_page()

                # Step 1: Go to login page
                print(f"  Navigating to login page...")
                page.goto(login_url, wait_until='networkidle')

                # Step 2: Fill in login form and submit
                print(f"  Logging in to SMPL-X website...")

                # Wait for form fields to be visible
                page.wait_for_selector('input[name="username"], input[type="email"], input[name="email"]', timeout=10000)

                # Try different form field names
                username_selectors = ['input[name="username"]', 'input[type="email"]', 'input[name="email"]']
                password_selectors = ['input[name="password"]', 'input[type="password"]']

                for selector in username_selectors:
                    try:
                        page.fill(selector, username)
                        break
                    except Exception:
                        continue

                for selector in password_selectors:
                    try:
                        page.fill(selector, password)
                        break
                    except Exception:
                        continue

                # Submit the form
                submit_selectors = ['button[type="submit"]', 'input[type="submit"]', 'button:has-text("Login")', 'button:has-text("Sign in")']
                for selector in submit_selectors:
                    try:
                        page.click(selector)
                        break
                    except Exception:
                        continue

                # Wait for navigation after login
                page.wait_for_load_state('networkidle')

                # Check if login was successful (should not be on login page anymore)
                if 'login' in page.url.lower() and 'invalid' in page.content().lower():
                    print_error("Login failed - invalid credentials")
                    browser.close()
                    return False

                print_success("Login successful")

                # Step 3: Navigate to download page
                print(f"  Accessing download page...")
                page.goto(download_page, wait_until='networkidle')

                # Step 4: Find the download link
                page_content = page.content()

                import re
                import html as html_module
                from urllib.parse import urljoin

                download_link = None
                patterns = [
                    r'href=["\']([^"\']*models_smplx_v1_1\.zip[^"\']*)["\']',
                    r'href=["\']([^"\']*download\.php\?[^"\']*smplx[^"\']*v1_1[^"\']*)["\']',
                    r'href=["\']([^"\']*download[^"\']*smplx[^"\']*)["\']',
                ]

                for pattern in patterns:
                    match = re.search(pattern, page_content, re.IGNORECASE)
                    if match:
                        link = html_module.unescape(match.group(1))
                        if not link.startswith('http'):
                            link = urljoin(download_page, link)
                        download_link = link
                        break

                if download_link:
                    print(f"  Found download link: {download_link}")
                    final_url = download_link
                else:
                    print(f"  Using configured download URL: {download_url}")
                    final_url = download_url

                # Step 5: Get cookies from browser and download with requests
                # This ensures we have all cookies including httpOnly ones
                cookies = context.cookies()
                print(f"  Got {len(cookies)} cookies from browser session")

                # Debug: show cookies
                for c in cookies:
                    print(f"    Cookie: {c['name']} (domain: {c.get('domain', 'N/A')})")

                browser.close()

                # Use requests with the browser cookies for the actual download
                import requests
                session = requests.Session()
                session.headers.update({
                    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Referer': download_page,
                })

                # Add all cookies to session
                for cookie in cookies:
                    session.cookies.set(
                        cookie['name'],
                        cookie['value'],
                        domain=cookie.get('domain', ''),
                        path=cookie.get('path', '/'),
                    )

                # Also set cookies specifically for download domain
                download_domain = 'download.is.tue.mpg.de'
                for cookie in cookies:
                    session.cookies.set(
                        cookie['name'],
                        cookie['value'],
                        domain=download_domain,
                        path=cookie.get('path', '/'),
                    )

                # Download the file
                print(f"  Downloading SMPL-X models...")
                print(f"  URL: {final_url}")
                print(f"  -> {dest}")

                dest.parent.mkdir(parents=True, exist_ok=True)

                response = session.get(final_url, stream=True, timeout=60)

                # Check content-type
                content_type = response.headers.get('content-type', '').lower()
                if 'text/html' in content_type:
                    if len(response.content) < 10000:
                        print_error("Server returned HTML instead of the file")
                        print_info("Response content:")
                        print("-" * 60)
                        try:
                            html_content = response.content.decode('utf-8', errors='ignore')
                            print(html_content[:2000])
                            if len(html_content) > 2000:
                                print(f"... (truncated, total {len(html_content)} chars)")
                        except Exception as e:
                            print(f"Could not decode response: {e}")
                        print("-" * 60)
                        return False

                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0

                with open(dest, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                            if total_size > 0:
                                pct = (downloaded / total_size) * 100
                                mb_downloaded = downloaded / (1024 * 1024)
                                mb_total = total_size / (1024 * 1024)
                                print(f"\r  Progress: {pct:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='', flush=True)

                print()
                print_success(f"Downloaded {dest.name}")
                return True

        except Exception as e:
            print_warning(f"Playwright download failed: {e}")
            if dest.exists():
                dest.unlink()
            return False

    def _download_smplx_with_requests(
        self,
        login_url: str,
        download_page: str,
        download_url: str,
        dest: Path,
        username: str,
        password: str,
    ) -> bool:
        """Download SMPL-X model with requests-based cross-domain authentication.

        This is the fallback method if Playwright is not available.
        """
        try:
            import requests
            from urllib.parse import urljoin
            import re
            import html
        except ImportError:
            print_warning("requests library not found, installing...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--user", "requests"],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--break-system-packages", "requests"],
                    capture_output=True, text=True
                )
            import requests
            from urllib.parse import urljoin
            import re
            import html

        try:
            print(f"  Logging in to SMPL-X website...")

            # Create session to maintain cookies
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
            })

            # First, get the login page to extract any CSRF token
            print(f"  Fetching login page for CSRF token...")
            login_page_response = session.get(login_url, timeout=30)

            # Extract CSRF token if present
            csrf_token = None
            csrf_patterns = [
                r'name=["\']csrf_token["\'][^>]*value=["\']([^"\']+)["\']',
                r'name=["\']_token["\'][^>]*value=["\']([^"\']+)["\']',
                r'name=["\']token["\'][^>]*value=["\']([^"\']+)["\']',
                r'value=["\']([^"\']+)["\'][^>]*name=["\']csrf_token["\']',
                r'value=["\']([^"\']+)["\'][^>]*name=["\']_token["\']',
            ]
            for pattern in csrf_patterns:
                match = re.search(pattern, login_page_response.text, re.IGNORECASE)
                if match:
                    csrf_token = match.group(1)
                    print(f"  Found CSRF token")
                    break

            # POST login form
            login_data = {
                'username': username,
                'password': password
            }
            if csrf_token:
                login_data['csrf_token'] = csrf_token
                login_data['_token'] = csrf_token
                login_data['token'] = csrf_token

            # Set referer for login
            session.headers['Referer'] = login_url

            login_response = session.post(login_url, data=login_data, timeout=30, allow_redirects=True)

            # Check for login errors in response
            if 'invalid' in login_response.text.lower() and 'password' in login_response.text.lower():
                print_error("Login failed - invalid credentials")
                return False

            # Check if we got redirected to login page (login failed)
            if 'login.php' in login_response.url and login_response.url != login_url:
                print_error("Login failed - redirected back to login page")
                return False

            print_success("Login successful")

            # Debug: Show cookies after login
            print(f"  Cookies after login:")
            for cookie in session.cookies:
                print(f"    {cookie.name}: domain={cookie.domain}, path={cookie.path}")

            # Visit download page to establish session and find download link
            print(f"  Accessing download page...")
            session.headers['Referer'] = login_response.url
            dl_page_response = session.get(download_page, timeout=30)

            if dl_page_response.status_code != 200:
                print_error(f"Could not access download page (status {dl_page_response.status_code})")
                return False

            # Look for the actual download link in the page
            page_content = dl_page_response.text

            # Try to find a direct link to the model file
            download_link = None

            patterns = [
                r'href=["\']([^"\']*models_smplx_v1_1\.zip[^"\']*)["\']',
                r'href=["\']([^"\']*download\.php\?[^"\']*smplx[^"\']*v1_1[^"\']*)["\']',
                r'href=["\']([^"\']*download[^"\']*smplx[^"\']*)["\']',
            ]

            for pattern in patterns:
                match = re.search(pattern, page_content, re.IGNORECASE)
                if match:
                    link = html.unescape(match.group(1))  # Decode &amp; -> &
                    if not link.startswith('http'):
                        link = urljoin(download_page, link)
                    download_link = link
                    break

            if download_link:
                print(f"  Found download link on page: {download_link}")
                final_url = download_link
            else:
                # Use the configured URL but try with the session
                print(f"  Using configured download URL: {download_url}")
                final_url = download_url

            print(f"  Downloading SMPL-X models...")
            print(f"  URL: {final_url}")
            print(f"  -> {dest}")

            # Copy ALL cookie attributes to download domain
            # Both are subdomains of is.tue.mpg.de but cookies don't auto-transfer
            download_domain = 'download.is.tue.mpg.de'
            for cookie in list(session.cookies):
                if 'smpl-x.is.tue.mpg.de' in cookie.domain or 'is.tue.mpg.de' in cookie.domain:
                    # Create a proper cookie with all attributes
                    session.cookies.set(
                        cookie.name,
                        cookie.value,
                        domain=download_domain,
                        path=cookie.path if cookie.path else '/',
                    )

            # Set referer for download request
            session.headers['Referer'] = download_page

            # Ensure directory exists
            dest.parent.mkdir(parents=True, exist_ok=True)

            # Try downloading with the session (cookies will be sent)
            response = session.get(final_url, stream=True, timeout=60)

            # Check content-type
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' in content_type:
                # Check if it's a redirect page or error
                if len(response.content) < 10000:
                    # Small HTML response - likely an error page
                    print_error("Server returned HTML instead of the file")
                    print_info("Response content:")
                    print("-" * 60)
                    try:
                        html_content = response.content.decode('utf-8', errors='ignore')
                        print(html_content[:2000])  # First 2000 chars
                        if len(html_content) > 2000:
                            print(f"... (truncated, total {len(html_content)} chars)")
                    except Exception as e:
                        print(f"Could not decode response: {e}")
                    print("-" * 60)
                    return False

            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(dest, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total_size > 0:
                            pct = (downloaded / total_size) * 100
                            mb_downloaded = downloaded / (1024 * 1024)
                            mb_total = total_size / (1024 * 1024)
                            print(f"\r  Progress: {pct:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='', flush=True)

            print()
            print_success(f"Downloaded {dest.name}")
            return True

        except requests.exceptions.RequestException as e:
            print_error(f"Download failed: {e}")
            if dest.exists():
                dest.unlink()
            return False

    def _validate_zip_file(self, file_path: Path) -> Tuple[bool, str]:
        """Validate that a file is actually a zip archive.

        Args:
            file_path: Path to the file to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        import zipfile

        # Check file exists and has content
        if not file_path.exists():
            return False, "File does not exist"

        file_size = file_path.stat().st_size
        if file_size == 0:
            return False, "File is empty"

        # Check magic bytes (PK signature for zip files)
        try:
            with open(file_path, 'rb') as f:
                magic = f.read(4)

            # Zip files start with PK\x03\x04 (regular) or PK\x05\x06 (empty archive)
            if not (magic[:2] == b'PK'):
                # Check if it's an HTML response (common error)
                with open(file_path, 'rb') as f:
                    start = f.read(500)

                if b'<!DOCTYPE' in start or b'<html' in start.lower() or b'<HTML' in start:
                    # Try to extract error message from HTML
                    try:
                        text = start.decode('utf-8', errors='ignore')
                        return False, f"Server returned HTML instead of zip file. This usually means:\n" \
                                      f"  - Authentication failed or session expired\n" \
                                      f"  - You haven't been approved for access yet\n" \
                                      f"  - The download link has changed"
                    except Exception:
                        pass
                    return False, "Server returned HTML instead of zip file"

                return False, f"File is not a zip archive (magic bytes: {magic.hex()})"

            # Verify it's a valid zip structure
            if not zipfile.is_zipfile(file_path):
                return False, "File has zip signature but is not a valid zip archive (possibly truncated)"

            return True, ""

        except IOError as e:
            return False, f"Could not read file: {e}"

    def extract_zip(self, zip_path: Path, dest_dir: Path) -> bool:
        """Extract zip file.

        Args:
            zip_path: Path to zip file
            dest_dir: Destination directory

        Returns:
            True if successful
        """
        import zipfile

        # Validate the zip file first
        is_valid, error_msg = self._validate_zip_file(zip_path)
        if not is_valid:
            print_error(f"Invalid zip file: {error_msg}")
            # Show file size for debugging
            if zip_path.exists():
                size_kb = zip_path.stat().st_size / 1024
                print_info(f"Downloaded file size: {size_kb:.1f} KB")
                if size_kb < 100:
                    # Small file - likely an error page, show first few lines
                    try:
                        with open(zip_path, 'r', errors='ignore') as f:
                            preview = f.read(500)
                        print_info(f"File preview:\n{preview[:300]}...")
                    except Exception:
                        pass
            return False

        try:
            print(f"  Extracting {zip_path.name}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dest_dir)
            print_success(f"Extracted to {dest_dir}")
            return True
        except zipfile.BadZipFile as e:
            print_error(f"Extraction failed - corrupted zip file: {e}")
            return False
        except Exception as e:
            print_error(f"Extraction failed: {e}")
            return False

    def download_checkpoint(
        self,
        comp_id: str,
        state_manager: Optional['InstallationStateManager'] = None,
        repo_root: Optional[Path] = None
    ) -> bool:
        """Download checkpoints for a component.

        Args:
            comp_id: Component ID (e.g., 'wham', 'econ', 'smplx')
            state_manager: Optional state manager for tracking
            repo_root: Repository root for finding credentials

        Returns:
            True if successful or already downloaded
        """
        if comp_id not in self.CHECKPOINTS:
            print_warning(f"No checkpoints defined for {comp_id}")
            return True  # Not an error, just skip

        checkpoint_info = self.CHECKPOINTS[comp_id]

        # Determine destination directory
        if checkpoint_info.get('use_home_dir'):
            dest_dir = Path.cwd() / checkpoint_info['dest_dir_rel']
        else:
            dest_dir = self.base_dir / checkpoint_info['dest_dir_rel']

        # Verify actual file exists before trusting state (handles filename changes)
        files_exist = True
        for file_info in checkpoint_info['files']:
            dest_path = dest_dir / file_info['filename']
            if not dest_path.exists():
                files_exist = False
                break

        # Check if already downloaded AND files exist
        if state_manager and state_manager.is_checkpoint_downloaded(comp_id) and files_exist:
            print_success(f"{checkpoint_info['name']} already downloaded")
            return True

        print(f"\n{Colors.BOLD}Downloading {checkpoint_info['name']}...{Colors.ENDC}")

        # Handle skip_download flag (for components without available checkpoints)
        if checkpoint_info.get('skip_download'):
            print_warning(f"Automatic download not available for {checkpoint_info['name']}")
            print_info(checkpoint_info['instructions'])
            return True  # Not a failure, just manual setup required

        # Handle authentication if required
        auth = None
        token = None
        form_auth = None  # For form-based login (username, password, login_url)
        use_gdown = checkpoint_info.get('use_gdown', False)
        if checkpoint_info.get('requires_auth'):
            if not repo_root:
                repo_root = Path.cwd()  # Fallback to current directory

            auth_type = checkpoint_info.get('auth_type', 'basic')
            auth_file = checkpoint_info.get('auth_file', '')

            if auth_type == 'basic':
                auth = self.read_credentials(repo_root, auth_file)
                if not auth:
                    print_error(f"{checkpoint_info['name']} requires authentication")
                    print_info(checkpoint_info['instructions'])
                    return False
                print_success(f"Loaded credentials from {auth_file}")
            elif auth_type == 'form':
                # Form-based login
                creds = self.read_credentials(repo_root, auth_file)
                if not creds:
                    print_error(f"{checkpoint_info['name']} requires authentication")
                    print_info(checkpoint_info['instructions'])
                    return False
                login_url = checkpoint_info.get('login_url')
                if not login_url:
                    print_error(f"No login_url configured for {checkpoint_info['name']}")
                    return False
                form_auth = (creds[0], creds[1], login_url)
                print_success(f"Loaded credentials from {auth_file}")
            elif auth_type == 'smplx':
                # Special SMPL-X cross-domain auth
                creds = self.read_credentials(repo_root, auth_file)
                if not creds:
                    print_error(f"{checkpoint_info['name']} requires authentication")
                    print_info(checkpoint_info['instructions'])
                    return False
                print_success(f"Loaded credentials from {auth_file}")
            elif auth_type == 'bearer':
                token = self.read_hf_token(repo_root)
                if not token:
                    print_error(f"{checkpoint_info['name']} requires HuggingFace token")
                    print_info(checkpoint_info['instructions'])
                    return False
                print_success(f"Loaded token from {auth_file}")

        # Ensure destination directory exists
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Handle HuggingFace downloads (uses snapshot_download)
        use_huggingface = checkpoint_info.get('use_huggingface', False)
        if use_huggingface:
            hf_repo_id = checkpoint_info.get('hf_repo_id')
            if hf_repo_id:
                if not self.download_from_huggingface(
                    hf_repo_id,
                    dest_dir,
                    checkpoint_info['files'][0]['filename'],
                    token=token,  # Pass token for gated models like SAM3
                ):
                    print_info(checkpoint_info['instructions'])
                    return False
                if state_manager:
                    state_manager.mark_checkpoint_downloaded(comp_id, dest_dir)
                return True

        success = True
        for file_info in checkpoint_info['files']:
            dest_path = dest_dir / file_info['filename']

            # Skip if already exists (unless it needs extraction)
            if dest_path.exists() and not file_info.get('extract'):
                print_success(f"{file_info['filename']} already exists")
                continue

            # Download with appropriate method
            if use_gdown:
                # Use gdown for Google Drive downloads
                if not self.download_file_gdown(
                    file_info['url'],
                    dest_path,
                    expected_size_mb=file_info.get('size_mb')
                ):
                    success = False
                    print_info(checkpoint_info['instructions'])
                    break
            elif checkpoint_info.get('auth_type') == 'smplx':
                # Special SMPL-X cross-domain authentication
                if not self.download_smplx_model(
                    checkpoint_info['login_url'],
                    checkpoint_info['download_page'],
                    file_info['url'],
                    dest_path,
                    creds[0],
                    creds[1],
                ):
                    success = False
                    print_info(checkpoint_info['instructions'])
                    break
            elif form_auth:
                # Use form-based login
                username, password, login_url = form_auth
                if not self.download_file_with_form_auth(
                    login_url,
                    file_info['url'],
                    dest_path,
                    username,
                    password,
                    expected_size_mb=file_info.get('size_mb')
                ):
                    success = False
                    break
            elif auth or token:
                # Use HTTP basic auth or bearer token
                if not self.download_file_with_auth(
                    file_info['url'],
                    dest_path,
                    auth=auth,
                    token=token,
                    expected_size_mb=file_info.get('size_mb')
                ):
                    success = False
                    break
            else:
                # Standard download
                if not self.download_file(file_info['url'], dest_path, file_info.get('size_mb')):
                    success = False
                    break

            # Verify checksum if provided
            if file_info.get('sha256'):
                if not self.verify_checksum(dest_path, file_info['sha256']):
                    success = False
                    dest_path.unlink()  # Remove corrupted file
                    break

            # Extract if needed
            if file_info.get('extract'):
                if not self.extract_zip(dest_path, dest_dir):
                    success = False
                    break
                # Optionally remove zip after extraction
                dest_path.unlink()
                print_info(f"Removed {dest_path.name} after extraction")

        if success and state_manager:
            state_manager.mark_checkpoint_downloaded(comp_id, dest_dir)

        if not success:
            print_error(f"Failed to download {checkpoint_info['name']}")
            print_info(checkpoint_info['instructions'])

        return success

    def download_all_checkpoints(
        self,
        component_ids: List[str],
        state_manager: Optional['InstallationStateManager'] = None,
        repo_root: Optional[Path] = None
    ) -> bool:
        """Download checkpoints for multiple components.

        Args:
            component_ids: List of component IDs
            state_manager: Optional state manager
            repo_root: Repository root for credentials

        Returns:
            True if all downloads successful
        """
        print_header("Downloading Checkpoints")

        # Auto-detect repo root if not provided
        if not repo_root:
            repo_root = Path(__file__).parent.parent.parent.resolve()

        success = True
        for comp_id in component_ids:
            if comp_id in self.CHECKPOINTS:
                if not self.download_checkpoint(comp_id, state_manager, repo_root):
                    success = False

        return success
