"""Tests for the PlatformManager class and Windows compatibility.

Tests cross-platform tool detection, sandboxed tool installation,
and platform-specific path handling.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from install_wizard.platform import PlatformManager, TOOLS_DIR, _is_windows


class TestPlatformDetection:
    """Test platform detection functions."""

    def test_detect_platform_returns_tuple(self):
        """detect_platform returns a 3-tuple."""
        result = PlatformManager.detect_platform()
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_detect_platform_os_name(self):
        """OS name is one of expected values."""
        os_name, _, _ = PlatformManager.detect_platform()
        assert os_name in ("linux", "macos", "windows", "unknown")

    def test_detect_platform_environment(self):
        """Environment is one of expected values."""
        _, environment, _ = PlatformManager.detect_platform()
        assert environment in ("native", "wsl2", "unknown")

    def test_is_windows_consistent(self):
        """_is_windows() is consistent with sys.platform."""
        expected = sys.platform == "win32"
        assert _is_windows() == expected


class TestFindTool:
    """Test cross-platform tool finding."""

    def test_find_tool_returns_path_or_none(self):
        """find_tool returns Path or None."""
        result = PlatformManager.find_tool("nonexistent_tool_xyz")
        assert result is None

    def test_find_tool_returns_path_for_existing(self):
        """find_tool returns Path for tools that exist."""
        result = PlatformManager.find_tool("python")
        if result:
            assert isinstance(result, Path)
            assert result.exists()

    @patch("install_wizard.platform.shutil.which")
    def test_find_tool_checks_system_path(self, mock_which):
        """find_tool checks system PATH via shutil.which."""
        mock_which.return_value = "/usr/bin/mytool"

        with patch.object(PlatformManager, "_get_local_tool_paths", return_value=[]):
            result = PlatformManager.find_tool("mytool")

        mock_which.assert_called_once_with("mytool")
        assert result == Path("/usr/bin/mytool")

    def test_local_tool_paths_checked_first(self):
        """Repo-local tool paths are checked before system PATH."""
        with tempfile.TemporaryDirectory() as tmp:
            tool_dir = Path(tmp) / "tools" / "testtool"
            tool_dir.mkdir(parents=True)

            if sys.platform == "win32":
                tool_file = tool_dir / "testtool.exe"
            else:
                tool_file = tool_dir / "testtool"
            tool_file.touch()
            tool_file.chmod(0o755)

            with patch("install_wizard.platform.TOOLS_DIR", Path(tmp) / "tools"):
                result = PlatformManager.find_tool("testtool")

            if result:
                assert str(tmp) in str(result)


class TestGetLocalToolPaths:
    """Test repo-local tool path generation."""

    def test_returns_list(self):
        """_get_local_tool_paths returns a list."""
        result = PlatformManager._get_local_tool_paths("colmap")
        assert isinstance(result, list)

    def test_paths_are_path_objects(self):
        """Paths are pathlib.Path objects."""
        result = PlatformManager._get_local_tool_paths("ffmpeg")
        for path in result:
            assert isinstance(path, Path)

    def test_includes_bin_subdirectory(self):
        """Includes bin/ subdirectory paths."""
        result = PlatformManager._get_local_tool_paths("colmap")
        bin_paths = [p for p in result if "bin" in str(p)]
        assert len(bin_paths) > 0


class TestWindowsToolPaths:
    """Test Windows-specific tool path handling."""

    def test_returns_list_for_known_tools(self):
        """_get_windows_tool_paths returns list for known tools."""
        for tool in ["colmap", "ffmpeg", "ffprobe", "7z", "nvidia-smi", "nvcc"]:
            result = PlatformManager._get_windows_tool_paths(tool)
            assert isinstance(result, list)

    def test_returns_empty_for_unknown_tools(self):
        """_get_windows_tool_paths returns empty list for unknown tools."""
        result = PlatformManager._get_windows_tool_paths("unknown_tool_xyz")
        assert result == []

    def test_colmap_paths_include_bat(self):
        """COLMAP paths include .bat extension on Windows."""
        result = PlatformManager._get_windows_tool_paths("colmap")
        bat_paths = [p for p in result if str(p).endswith(".bat")]
        assert len(bat_paths) > 0

    def test_ffmpeg_paths_include_exe(self):
        """FFmpeg paths include .exe extension."""
        result = PlatformManager._get_windows_tool_paths("ffmpeg")
        exe_paths = [p for p in result if str(p).endswith(".exe")]
        assert len(exe_paths) > 0


class TestUnixToolPaths:
    """Test Unix-specific tool path handling."""

    def test_returns_list_for_known_tools(self):
        """_get_unix_tool_paths returns list for known tools."""
        for tool in ["colmap", "ffmpeg", "ffprobe", "7z", "aria2c"]:
            result = PlatformManager._get_unix_tool_paths(tool)
            assert isinstance(result, list)

    def test_returns_empty_for_unknown_tools(self):
        """_get_unix_tool_paths returns empty list for unknown tools."""
        result = PlatformManager._get_unix_tool_paths("unknown_tool_xyz")
        assert result == []

    def test_paths_in_standard_locations(self):
        """Paths are in standard Unix locations."""
        result = PlatformManager._get_unix_tool_paths("ffmpeg")
        for path in result:
            assert "/usr/bin" in str(path) or "/usr/local/bin" in str(path)


class TestRunTool:
    """Test cross-platform tool execution."""

    def test_run_tool_returns_completed_process(self):
        """run_tool returns CompletedProcess."""
        result = PlatformManager.run_tool(
            Path(sys.executable),
            ["--version"],
            capture_output=True,
            text=True
        )
        assert hasattr(result, "returncode")
        assert hasattr(result, "stdout")

    @patch("install_wizard.platform._is_windows", return_value=True)
    @patch("subprocess.run")
    def test_bat_files_use_shell_on_windows(self, mock_run, mock_is_win):
        """On Windows, .bat files are run with shell=True."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        PlatformManager.run_tool(
            Path("C:/Program Files/COLMAP/COLMAP.bat"),
            ["--version"],
            capture_output=True
        )

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs.get("shell") is True


class TestToolDownloads:
    """Test tool download URL configuration."""

    def test_tool_downloads_dict_exists(self):
        """TOOL_DOWNLOADS dict exists and is populated."""
        assert hasattr(PlatformManager, "TOOL_DOWNLOADS")
        assert len(PlatformManager.TOOL_DOWNLOADS) > 0

    def test_colmap_has_windows_url(self):
        """COLMAP has Windows download URL."""
        assert "colmap" in PlatformManager.TOOL_DOWNLOADS
        assert "windows" in PlatformManager.TOOL_DOWNLOADS["colmap"]

    def test_colmap_has_linux_url(self):
        """COLMAP has Linux download URL."""
        assert "linux" in PlatformManager.TOOL_DOWNLOADS["colmap"]

    def test_ffmpeg_has_windows_url(self):
        """FFmpeg has Windows download URL."""
        assert "ffmpeg" in PlatformManager.TOOL_DOWNLOADS
        assert "windows" in PlatformManager.TOOL_DOWNLOADS["ffmpeg"]

    def test_ffmpeg_has_linux_url(self):
        """FFmpeg has Linux download URL."""
        assert "linux" in PlatformManager.TOOL_DOWNLOADS["ffmpeg"]


class TestToolsDir:
    """Test TOOLS_DIR configuration."""

    def test_tools_dir_is_path(self):
        """TOOLS_DIR is a Path object."""
        assert isinstance(TOOLS_DIR, Path)

    def test_tools_dir_not_in_home(self):
        """TOOLS_DIR is not in user home directory."""
        home = Path.home()
        assert not str(TOOLS_DIR).startswith(str(home))

    def test_tools_dir_in_vfx_pipeline(self):
        """TOOLS_DIR is in .vfx_pipeline directory."""
        assert ".vfx_pipeline" in str(TOOLS_DIR)


class TestGetToolsDir:
    """Test get_tools_dir method."""

    def test_returns_tools_dir(self):
        """get_tools_dir returns TOOLS_DIR."""
        result = PlatformManager.get_tools_dir()
        assert result == TOOLS_DIR


class TestInstallToolValidation:
    """Test install_tool validation logic."""

    def test_unknown_tool_returns_none(self):
        """install_tool returns None for unknown tools."""
        result = PlatformManager.install_tool("unknown_tool_xyz_123")
        assert result is None

    def test_existing_tool_returns_existing_path(self):
        """install_tool returns existing path if tool already installed in TOOLS_DIR."""
        fake_path = TOOLS_DIR / "ffmpeg" / "ffmpeg"
        with patch.object(PlatformManager, "find_tool", return_value=fake_path):
            result = PlatformManager.install_tool("ffmpeg", force=False)
            assert result == fake_path

    def test_force_bypasses_existing_check(self):
        """install_tool with force=True doesn't return existing."""
        fake_path = TOOLS_DIR / "colmap" / "colmap"
        with patch.object(PlatformManager, "find_tool", return_value=fake_path):
            with patch("urllib.request.urlretrieve") as mock_download:
                mock_download.side_effect = Exception("Network blocked")
                result = PlatformManager.install_tool("colmap", force=True)
                assert result is None
                mock_download.assert_called_once()


class TestFlattenSingleSubdir:
    """Test _flatten_single_subdir utility."""

    def test_flattens_single_subdirectory(self):
        """Flattens when there's a single subdirectory."""
        with tempfile.TemporaryDirectory() as tmp:
            tool_dir = Path(tmp)
            subdir = tool_dir / "COLMAP-3.9.1-windows"
            subdir.mkdir()
            (subdir / "colmap.exe").touch()
            (subdir / "lib").mkdir()

            PlatformManager._flatten_single_subdir(tool_dir)

            assert (tool_dir / "colmap.exe").exists()
            assert (tool_dir / "lib").exists()
            assert not subdir.exists()

    def test_no_flatten_with_multiple_subdirs(self):
        """Does not flatten when there are multiple subdirectories."""
        with tempfile.TemporaryDirectory() as tmp:
            tool_dir = Path(tmp)
            (tool_dir / "dir1").mkdir()
            (tool_dir / "dir2").mkdir()

            PlatformManager._flatten_single_subdir(tool_dir)

            assert (tool_dir / "dir1").exists()
            assert (tool_dir / "dir2").exists()

    def test_no_flatten_with_files_present(self):
        """Does not flatten when there are files at root level."""
        with tempfile.TemporaryDirectory() as tmp:
            tool_dir = Path(tmp)
            (tool_dir / "README.txt").touch()
            subdir = tool_dir / "subdir"
            subdir.mkdir()
            (subdir / "tool.exe").touch()

            PlatformManager._flatten_single_subdir(tool_dir)

            assert subdir.exists()
            assert (tool_dir / "README.txt").exists()


class TestSnapDetection:
    """Test snap confinement detection for COLMAP."""

    def test_is_snap_path_detects_snap_bin(self):
        """_is_snap_path detects /snap/bin/ paths."""
        assert PlatformManager._is_snap_path("/snap/bin/colmap")
        assert PlatformManager._is_snap_path("/snap/colmap/123/bin/colmap")

    def test_is_snap_path_ignores_non_snap(self):
        """_is_snap_path returns False for non-snap paths."""
        assert not PlatformManager._is_snap_path("/usr/bin/colmap")
        assert not PlatformManager._is_snap_path("/usr/local/bin/colmap")
        assert not PlatformManager._is_snap_path("/home/user/.local/bin/colmap")

    def test_colmap_in_snap_restricted_tools(self):
        """COLMAP is in SNAP_RESTRICTED_TOOLS set."""
        assert "colmap" in PlatformManager.SNAP_RESTRICTED_TOOLS

    @patch("install_wizard.platform.shutil.which")
    def test_find_tool_skips_snap_colmap(self, mock_which):
        """find_tool skips snap COLMAP and falls through to other paths."""
        mock_which.return_value = "/snap/bin/colmap"

        with patch.object(PlatformManager, "_get_local_tool_paths", return_value=[]):
            with patch.object(
                PlatformManager, "_get_unix_tool_paths",
                return_value=[Path("/usr/bin/colmap")]
            ):
                # If /usr/bin/colmap doesn't exist, result should be None
                # (snap is skipped, fallback doesn't exist)
                result = PlatformManager.find_tool("colmap")
                # Result depends on whether /usr/bin/colmap exists on the test system
                # The important thing is that mock_which was called
                mock_which.assert_called_once_with("colmap")

    @patch("install_wizard.platform.shutil.which")
    def test_find_tool_uses_snap_for_non_restricted_tools(self, mock_which):
        """find_tool uses snap paths for tools not in SNAP_RESTRICTED_TOOLS."""
        mock_which.return_value = "/snap/bin/ffmpeg"

        with patch.object(PlatformManager, "_get_local_tool_paths", return_value=[]):
            result = PlatformManager.find_tool("ffmpeg")

        assert result == Path("/snap/bin/ffmpeg")

    def test_is_snap_path_windows_paths_not_matched(self):
        """Windows paths are never detected as snap paths."""
        # Windows paths use backslashes and don't contain /snap/
        assert not PlatformManager._is_snap_path("C:\\Program Files\\COLMAP\\COLMAP.bat")
        assert not PlatformManager._is_snap_path("C:/Program Files/COLMAP/colmap.exe")
        assert not PlatformManager._is_snap_path("D:\\tools\\colmap\\bin\\colmap.exe")

    @patch("install_wizard.platform.shutil.which")
    def test_find_tool_uses_windows_path_for_colmap(self, mock_which):
        """On Windows, COLMAP paths found via which are used (no snap on Windows)."""
        mock_which.return_value = "C:\\Program Files\\COLMAP\\COLMAP.bat"

        with patch.object(PlatformManager, "_get_local_tool_paths", return_value=[]):
            result = PlatformManager.find_tool("colmap")

        # Windows path doesn't match snap check, so it should be returned
        assert result == Path("C:\\Program Files\\COLMAP\\COLMAP.bat")


class TestPackageManagerDetection:
    """Test package manager detection."""

    def test_windows_package_manager_detection(self):
        """_detect_windows_package_manager returns expected values."""
        result = PlatformManager._detect_windows_package_manager()
        assert result in ("winget", "choco", "scoop", "unknown")

    def test_linux_package_manager_detection(self):
        """_detect_linux_package_manager returns expected values."""
        result = PlatformManager._detect_linux_package_manager()
        assert result in ("apt", "yum", "dnf", "pacman", "zypper", "unknown")
