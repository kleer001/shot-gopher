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


class TestScriptsPathContextManager:
    """Test the _scripts_path context manager."""

    def test_scripts_path_adds_and_removes(self):
        """Context manager adds path and removes it after."""
        from blender import _scripts_path
        import sys

        scripts_dir = str(Path(__file__).parent.parent / "scripts")
        original_path = sys.path.copy()

        with _scripts_path():
            assert scripts_dir in sys.path

        assert sys.path == original_path

    def test_scripts_path_already_present(self):
        """Context manager doesn't duplicate if path already present."""
        from blender import _scripts_path
        import sys

        scripts_dir = str(Path(__file__).parent.parent / "scripts")

        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
            added = True
        else:
            added = False

        try:
            initial_count = sys.path.count(scripts_dir)
            with _scripts_path():
                assert sys.path.count(scripts_dir) == initial_count
            assert sys.path.count(scripts_dir) == initial_count
        finally:
            if added:
                sys.path.remove(scripts_dir)


class TestExportMeshSequenceValidation:
    """Test input validation in export_mesh_sequence_to_alembic."""

    def test_raises_on_nonexistent_input_dir(self):
        """Raises FileNotFoundError for nonexistent input directory."""
        from blender import export_mesh_sequence_to_alembic

        with tempfile.TemporaryDirectory() as tmp:
            nonexistent = Path(tmp) / "nonexistent"
            output = Path(tmp) / "output.abc"

            with pytest.raises(FileNotFoundError, match="Input directory not found"):
                export_mesh_sequence_to_alembic(nonexistent, output)

    def test_raises_on_file_not_directory(self):
        """Raises ValueError when input is a file, not directory."""
        from blender import export_mesh_sequence_to_alembic

        with tempfile.TemporaryDirectory() as tmp:
            input_file = Path(tmp) / "file.txt"
            input_file.touch()
            output = Path(tmp) / "output.abc"

            with pytest.raises(ValueError, match="not a directory"):
                export_mesh_sequence_to_alembic(input_file, output)

    def test_raises_on_invalid_fps(self):
        """Raises ValueError for non-positive FPS."""
        from blender import export_mesh_sequence_to_alembic

        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp) / "meshes"
            input_dir.mkdir()
            output = Path(tmp) / "output.abc"

            with pytest.raises(ValueError, match="FPS must be positive"):
                export_mesh_sequence_to_alembic(input_dir, output, fps=0)

            with pytest.raises(ValueError, match="FPS must be positive"):
                export_mesh_sequence_to_alembic(input_dir, output, fps=-24)

    def test_raises_on_negative_start_frame(self):
        """Raises ValueError for negative start frame."""
        from blender import export_mesh_sequence_to_alembic

        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp) / "meshes"
            input_dir.mkdir()
            output = Path(tmp) / "output.abc"

            with pytest.raises(ValueError, match="Start frame must be non-negative"):
                export_mesh_sequence_to_alembic(input_dir, output, start_frame=-1)


class TestExportSubprocessBehavior:
    """Test subprocess behavior in export_mesh_sequence_to_alembic."""

    def test_raises_on_blender_not_found(self):
        """Raises FileNotFoundError when Blender not found and install fails."""
        from blender import export_mesh_sequence_to_alembic

        with patch("blender.find_blender", return_value=None):
            with patch("blender.install_blender", return_value=None):
                with tempfile.TemporaryDirectory() as tmp:
                    input_dir = Path(tmp) / "meshes"
                    input_dir.mkdir()
                    output = Path(tmp) / "output.abc"

                    with pytest.raises(FileNotFoundError, match="Blender not found"):
                        export_mesh_sequence_to_alembic(input_dir, output)

    def test_raises_on_export_timeout(self):
        """Raises RuntimeError on Blender subprocess timeout."""
        from blender import export_mesh_sequence_to_alembic
        import subprocess

        mock_process = MagicMock()
        mock_process.communicate.side_effect = subprocess.TimeoutExpired("blender", 3600)
        mock_process.kill = MagicMock()
        mock_process.wait = MagicMock()

        with patch("blender.find_blender", return_value=Path("/usr/bin/blender")):
            with patch("blender.subprocess.Popen", return_value=mock_process):
                with tempfile.TemporaryDirectory() as tmp:
                    input_dir = Path(tmp) / "meshes"
                    input_dir.mkdir()
                    output = Path(tmp) / "output.abc"

                    with pytest.raises(RuntimeError, match="timed out"):
                        export_mesh_sequence_to_alembic(input_dir, output)
                    mock_process.kill.assert_called_once()

    def test_raises_on_blender_error(self):
        """Raises RuntimeError when Blender returns non-zero exit code."""
        from blender import export_mesh_sequence_to_alembic

        mock_process = MagicMock()
        mock_process.communicate.return_value = ("", "Error: No OBJ files found")
        mock_process.returncode = 1

        with patch("blender.find_blender", return_value=Path("/usr/bin/blender")):
            with patch("blender.subprocess.Popen", return_value=mock_process):
                with tempfile.TemporaryDirectory() as tmp:
                    input_dir = Path(tmp) / "meshes"
                    input_dir.mkdir()
                    output = Path(tmp) / "output.abc"

                    with pytest.raises(RuntimeError, match="Blender export failed"):
                        export_mesh_sequence_to_alembic(input_dir, output)

    def test_raises_when_output_not_created(self):
        """Raises RuntimeError when Blender succeeds but output not created."""
        from blender import export_mesh_sequence_to_alembic

        mock_process = MagicMock()
        mock_process.communicate.return_value = ("Export complete", "")
        mock_process.returncode = 0

        with patch("blender.find_blender", return_value=Path("/usr/bin/blender")):
            with patch("blender.subprocess.Popen", return_value=mock_process):
                with tempfile.TemporaryDirectory() as tmp:
                    input_dir = Path(tmp) / "meshes"
                    input_dir.mkdir()
                    output = Path(tmp) / "output.abc"

                    with pytest.raises(RuntimeError, match="output file not created"):
                        export_mesh_sequence_to_alembic(input_dir, output)

    def test_auto_installs_blender_when_not_found(self):
        """Attempts to install Blender when not found."""
        from blender import export_mesh_sequence_to_alembic

        install_mock = MagicMock(return_value=Path("/installed/blender"))
        mock_process = MagicMock()
        mock_process.communicate.return_value = ("Success", "")
        mock_process.returncode = 0

        with patch("blender.find_blender", return_value=None):
            with patch("blender.install_blender", install_mock):
                with patch("blender.subprocess.Popen", return_value=mock_process):
                    with tempfile.TemporaryDirectory() as tmp:
                        input_dir = Path(tmp) / "meshes"
                        input_dir.mkdir()
                        output = Path(tmp) / "output.abc"
                        output.touch()

                        export_mesh_sequence_to_alembic(input_dir, output)
                        install_mock.assert_called_once()


class TestCheckBlenderAvailableEdgeCases:
    """Test edge cases in check_blender_available."""

    def test_handles_version_check_timeout(self):
        """Returns False with message on version check timeout."""
        from blender import check_blender_available
        import subprocess

        with patch("blender.find_blender", return_value=Path("/usr/bin/blender")):
            with patch("blender.subprocess.run", side_effect=subprocess.TimeoutExpired("blender", 30)):
                available, message = check_blender_available()
                assert available is False
                assert "timed out" in message.lower()

    def test_handles_version_check_failure(self):
        """Returns False with message on non-zero return code."""
        from blender import check_blender_available

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Error: something went wrong"

        with patch("blender.find_blender", return_value=Path("/usr/bin/blender")):
            with patch("blender.subprocess.run", return_value=mock_result):
                available, message = check_blender_available()
                assert available is False
                assert "failed" in message.lower()

    def test_handles_unexpected_exception(self):
        """Returns False with message on unexpected exception."""
        from blender import check_blender_available

        with patch("blender.find_blender", return_value=Path("/usr/bin/blender")):
            with patch("blender.subprocess.run", side_effect=OSError("Permission denied")):
                available, message = check_blender_available()
                assert available is False
                assert "Permission denied" in message


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


# NOTE: The following tests require Blender to be installed and would be
# integration tests. They test the Blender-side script (export_mesh_alembic.py):
#
# - find_obj_files: mixed case handling, sorting, empty directory
# - import_obj_sequence_as_shape_keys: vertex mismatch error, keyframe timing
# - export_alembic: Alembic output parameters
#
# These would need to be run with: blender -b --python test_script.py
# or marked with @pytest.mark.skipif(not BLENDER_AVAILABLE, ...)
