"""Tests for Blender-based Alembic export functionality.

Tests the Blender integration module and mesh sequence export.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from install_wizard.platform import PlatformManager


class TestBlenderPlatformKey:
    """Test platform key detection for Blender downloads."""

    def test_get_platform_key_windows(self):
        """Windows platform key is 'windows'."""
        with patch("install_wizard.platform._is_windows", return_value=True):
            key = PlatformManager._get_platform_key("blender")
            assert key == "windows"

    def test_get_platform_key_linux(self):
        """Linux platform key is 'linux'."""
        with patch("install_wizard.platform._is_windows", return_value=False):
            with patch("install_wizard.platform.platform.system", return_value="Linux"):
                key = PlatformManager._get_platform_key("blender")
                assert key == "linux"

    def test_get_platform_key_macos_arm(self):
        """macOS ARM platform key is 'macos_arm'."""
        with patch("install_wizard.platform._is_windows", return_value=False):
            with patch("install_wizard.platform.platform.system", return_value="Darwin"):
                with patch("install_wizard.platform.platform.machine", return_value="arm64"):
                    key = PlatformManager._get_platform_key("blender")
                    assert key == "macos_arm"

    def test_get_platform_key_macos_intel(self):
        """macOS Intel platform key is 'macos_intel'."""
        with patch("install_wizard.platform._is_windows", return_value=False):
            with patch("install_wizard.platform.platform.system", return_value="Darwin"):
                with patch("install_wizard.platform.platform.machine", return_value="x86_64"):
                    key = PlatformManager._get_platform_key("blender")
                    assert key == "macos_intel"


class TestBlenderToolPaths:
    """Test Blender tool path detection."""

    def test_local_tool_paths_include_blender_windows(self):
        """Windows local paths include blender.exe."""
        with patch("install_wizard.platform._is_windows", return_value=True):
            paths = PlatformManager._get_local_tool_paths("blender")
            path_strs = [str(p) for p in paths]
            assert any("blender.exe" in p for p in path_strs)

    def test_local_tool_paths_include_blender_linux(self):
        """Linux local paths include blender executable."""
        with patch("install_wizard.platform._is_windows", return_value=False):
            with patch("install_wizard.platform.platform.system", return_value="Linux"):
                paths = PlatformManager._get_local_tool_paths("blender")
                path_strs = [str(p) for p in paths]
                assert any("blender" in p and ".exe" not in p for p in path_strs)

    def test_local_tool_paths_include_blender_macos(self):
        """macOS local paths include Blender.app."""
        with patch("install_wizard.platform._is_windows", return_value=False):
            with patch("install_wizard.platform.platform.system", return_value="Darwin"):
                paths = PlatformManager._get_local_tool_paths("blender")
                path_strs = [str(p) for p in paths]
                assert any("Blender.app" in p for p in path_strs)

    def test_windows_system_paths_include_blender(self):
        """Windows system paths include Blender Foundation paths."""
        paths = PlatformManager._get_windows_tool_paths("blender")
        path_strs = [str(p) for p in paths]
        assert any("Blender Foundation" in p for p in path_strs)

    def test_unix_system_paths_include_blender(self):
        """Unix system paths include common Blender locations."""
        paths = PlatformManager._get_unix_tool_paths("blender")
        path_strs = [str(p) for p in paths]
        assert any("/usr/bin/blender" in p or "/usr/local/bin/blender" in p for p in path_strs)


class TestBlenderDownloadConfig:
    """Test Blender download URL configuration."""

    def test_blender_urls_configured(self):
        """Blender download URLs are configured for all platforms."""
        assert "blender" in PlatformManager.TOOL_DOWNLOADS
        blender_config = PlatformManager.TOOL_DOWNLOADS["blender"]

        assert "windows" in blender_config
        assert "linux" in blender_config
        assert "macos_arm" in blender_config
        assert "macos_intel" in blender_config

    def test_blender_urls_are_valid(self):
        """Blender download URLs are valid HTTPS URLs."""
        blender_config = PlatformManager.TOOL_DOWNLOADS["blender"]

        for platform_key, url in blender_config.items():
            assert url.startswith("https://"), f"{platform_key} URL should be HTTPS"
            assert "blender" in url.lower(), f"{platform_key} URL should contain 'blender'"


class TestExportAlembicModule:
    """Test the export_alembic.py module."""

    def test_export_mesh_alembic_missing_blender(self):
        """export_mesh_alembic returns False when Blender not available."""
        with patch.dict("sys.modules", {"blender": None}):
            import importlib
            import scripts.export_alembic as export_alembic
            importlib.reload(export_alembic)

            with tempfile.TemporaryDirectory() as tmp:
                input_dir = Path(tmp) / "meshes"
                input_dir.mkdir()
                output_path = Path(tmp) / "output.abc"

                result = export_alembic.export_mesh_alembic(input_dir, output_path)
                assert result is False

    def test_export_mesh_alembic_missing_input_dir(self):
        """export_mesh_alembic handles missing input directory."""
        mock_blender = MagicMock()
        mock_blender.export_mesh_sequence_to_alembic.side_effect = FileNotFoundError(
            "Input directory not found"
        )

        with patch.dict("sys.modules", {"blender": mock_blender}):
            import importlib
            import scripts.export_alembic as export_alembic
            importlib.reload(export_alembic)
            export_alembic.HAS_BLENDER = True
            export_alembic.export_mesh_sequence_to_alembic = mock_blender.export_mesh_sequence_to_alembic

            with tempfile.TemporaryDirectory() as tmp:
                input_dir = Path(tmp) / "nonexistent"
                output_path = Path(tmp) / "output.abc"

                result = export_alembic.export_mesh_alembic(input_dir, output_path)
                assert result is False


class TestBlenderModule:
    """Test the blender/__init__.py module."""

    def test_find_blender_uses_platform_manager(self):
        """find_blender uses PlatformManager.find_tool."""
        from blender import find_blender

        with patch.object(PlatformManager, "find_tool", return_value=None) as mock:
            result = find_blender()
            mock.assert_called_once_with("blender")
            assert result is None

    def test_install_blender_uses_platform_manager(self):
        """install_blender uses PlatformManager.install_tool."""
        from blender import install_blender

        with patch.object(PlatformManager, "install_tool", return_value=None) as mock:
            result = install_blender()
            mock.assert_called_once_with("blender")
            assert result is None

    def test_check_blender_available_not_found(self):
        """check_blender_available returns False when Blender not found."""
        from blender import check_blender_available

        with patch("blender.find_blender", return_value=None):
            available, message = check_blender_available()
            assert available is False
            assert "not found" in message.lower()

    def test_check_blender_available_found(self):
        """check_blender_available returns True when Blender found and works."""
        from blender import check_blender_available

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Blender 4.2.5\n"

        with patch("blender.find_blender", return_value=Path("/usr/bin/blender")):
            with patch("blender.subprocess.run", return_value=mock_result):
                available, message = check_blender_available()
                assert available is True
                assert "4.2.5" in message


class TestBlenderExportScript:
    """Test the Blender export script argument parsing."""

    def test_export_script_exists(self):
        """The Blender export script exists."""
        script_path = Path(__file__).parent.parent / "scripts" / "blender" / "export_mesh_alembic.py"
        assert script_path.exists(), f"Export script not found: {script_path}"

    def test_export_script_has_main(self):
        """The Blender export script has a main function."""
        script_path = Path(__file__).parent.parent / "scripts" / "blender" / "export_mesh_alembic.py"
        content = script_path.read_text()
        assert "def main():" in content
        assert 'if __name__ == "__main__":' in content


class TestDMGExtraction:
    """Test macOS DMG extraction helper."""

    def test_extract_dmg_not_on_macos(self):
        """_extract_dmg works on macOS (mocked)."""
        with patch("install_wizard.platform.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            with patch("install_wizard.platform.Path.exists", return_value=True):
                with patch("install_wizard.platform.Path.iterdir", return_value=[]):
                    with tempfile.TemporaryDirectory() as tmp:
                        dmg_path = Path(tmp) / "test.dmg"
                        dmg_path.touch()
                        dest_dir = Path(tmp) / "dest"
                        dest_dir.mkdir()

                        result = PlatformManager._extract_dmg(
                            dmg_path, dest_dir, "Blender.app"
                        )

    def test_extract_dmg_handles_mount_failure(self):
        """_extract_dmg handles mount failure gracefully."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Mount failed"

        with patch("install_wizard.platform.subprocess.run", return_value=mock_result):
            with tempfile.TemporaryDirectory() as tmp:
                dmg_path = Path(tmp) / "test.dmg"
                dmg_path.touch()
                dest_dir = Path(tmp) / "dest"
                dest_dir.mkdir()

                result = PlatformManager._extract_dmg(
                    dmg_path, dest_dir, "Blender.app"
                )
                assert result is False
