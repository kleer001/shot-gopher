"""Tests for install_wizard/utils.py - Installation utility functions."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from install_wizard.utils import (
    Colors,
    print_header,
    print_success,
    print_warning,
    print_error,
    print_info,
    ask_yes_no,
    run_command,
    check_python_package,
    check_command_available,
    get_disk_space,
    format_size_gb,
)


class TestColors:
    def test_colors_are_strings(self):
        assert isinstance(Colors.HEADER, str)
        assert isinstance(Colors.OKGREEN, str)
        assert isinstance(Colors.WARNING, str)
        assert isinstance(Colors.FAIL, str)
        assert isinstance(Colors.ENDC, str)

    def test_colors_are_ansi_escapes(self):
        assert Colors.HEADER.startswith('\033[')
        assert Colors.ENDC.startswith('\033[')


class TestPrintFunctions:
    def test_print_header(self, capsys):
        print_header("Test Header")
        captured = capsys.readouterr()
        assert "Test Header" in captured.out
        assert "=" in captured.out

    def test_print_success(self, capsys):
        print_success("Success message")
        captured = capsys.readouterr()
        assert "Success message" in captured.out
        assert "OK" in captured.out

    def test_print_warning(self, capsys):
        print_warning("Warning message")
        captured = capsys.readouterr()
        assert "Warning message" in captured.out
        assert "!" in captured.out

    def test_print_error(self, capsys):
        print_error("Error message")
        captured = capsys.readouterr()
        assert "Error message" in captured.out
        assert "X" in captured.out

    def test_print_info(self, capsys):
        print_info("Info message")
        captured = capsys.readouterr()
        assert "Info message" in captured.out
        assert ">" in captured.out


class TestAskYesNo:
    @patch('install_wizard.utils.tty_input', return_value='y')
    def test_yes_response(self, mock_input):
        result = ask_yes_no("Test question?")
        assert result is True

    @patch('install_wizard.utils.tty_input', return_value='yes')
    def test_yes_full_word(self, mock_input):
        result = ask_yes_no("Test question?")
        assert result is True

    @patch('install_wizard.utils.tty_input', return_value='n')
    def test_no_response(self, mock_input):
        result = ask_yes_no("Test question?")
        assert result is False

    @patch('install_wizard.utils.tty_input', return_value='no')
    def test_no_full_word(self, mock_input):
        result = ask_yes_no("Test question?")
        assert result is False

    @patch('install_wizard.utils.tty_input', return_value='')
    def test_empty_uses_default_yes(self, mock_input):
        result = ask_yes_no("Test?", default=True)
        assert result is True

    @patch('install_wizard.utils.tty_input', return_value='')
    def test_empty_uses_default_no(self, mock_input):
        result = ask_yes_no("Test?", default=False)
        assert result is False


class TestRunCommand:
    def test_run_successful_command(self):
        success, output = run_command(["echo", "hello"], capture=True)
        assert success is True
        assert "hello" in output

    def test_run_failing_command(self):
        success, output = run_command(["false"], check=False, capture=True)
        assert success is False

    def test_run_nonexistent_command(self):
        success, output = run_command(["nonexistent_cmd_12345"], check=False, capture=True)
        assert success is False

    def test_run_with_timeout(self):
        success, output = run_command(
            ["sleep", "0.1"],
            capture=True,
            timeout=10
        )
        assert success is True

    def test_run_streaming(self, capsys):
        success, output = run_command(
            ["echo", "streaming"],
            stream=True
        )
        assert success is True
        captured = capsys.readouterr()
        assert "streaming" in captured.out


class TestCheckPythonPackage:
    def test_existing_package(self):
        result = check_python_package("sys")
        assert result is True

    def test_existing_package_with_import_name(self):
        result = check_python_package("pathlib", "pathlib")
        assert result is True

    def test_nonexistent_package(self):
        result = check_python_package("nonexistent_package_12345")
        assert result is False


class TestCheckCommandAvailable:
    def test_existing_command(self):
        result = check_command_available("python")
        assert result is True

    def test_nonexistent_command(self):
        result = check_command_available("nonexistent_command_12345")
        assert result is False

    def test_common_commands(self):
        assert check_command_available("ls") or check_command_available("dir")


class TestGetDiskSpace:
    def test_returns_tuple(self):
        result = get_disk_space()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_positive_values(self):
        available, total = get_disk_space()
        assert available >= 0
        assert total >= 0
        assert total >= available

    def test_with_custom_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            available, total = get_disk_space(Path(tmpdir))
            assert available > 0
            assert total > 0

    def test_nonexistent_path_returns_zeros(self):
        available, total = get_disk_space(Path("/nonexistent/path/12345"))
        assert available == 0.0
        assert total == 0.0


class TestFormatSizeGb:
    def test_format_mb(self):
        result = format_size_gb(0.5)
        assert "MB" in result
        assert "512" in result

    def test_format_small_gb(self):
        result = format_size_gb(5.5)
        assert "GB" in result
        assert "5.5" in result

    def test_format_large_gb(self):
        result = format_size_gb(100)
        assert "GB" in result
        assert "100" in result

    def test_format_zero(self):
        result = format_size_gb(0)
        assert "0" in result
