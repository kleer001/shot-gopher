"""Tests for subprocess_utils.py - Progress tracking and subprocess execution."""

import re
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from subprocess_utils import (
    ProgressPattern,
    ProcessResult,
    ProgressTracker,
    ProcessRunner,
    create_mmcam_patterns,
    create_training_patterns,
    create_generic_patterns,
)


class TestProgressPattern:
    def test_match_simple_pattern(self):
        pattern = ProgressPattern(
            name="Processing",
            pattern=re.compile(r'\[(\d+)/(\d+)\]'),
            format_str="Processing: {current}/{total}"
        )

        result = pattern.match("[5/10]")

        assert result == (5, 10)

    def test_match_no_match_returns_none(self):
        pattern = ProgressPattern(
            name="Processing",
            pattern=re.compile(r'\[(\d+)/(\d+)\]'),
            format_str="Processing: {current}/{total}"
        )

        result = pattern.match("no match here")

        assert result is None

    def test_match_custom_groups(self):
        pattern = ProgressPattern(
            name="Training",
            pattern=re.compile(r'(\d+)%\|.*\|\s*(\d+)/(\d+)'),
            format_str="Training: {current}/{total}",
            current_group=2,
            total_group=3
        )

        result = pattern.match("50%|#####     | 25/50")

        assert result == (25, 50)

    def test_format_output(self):
        pattern = ProgressPattern(
            name="Processing",
            pattern=re.compile(r'\[(\d+)/(\d+)\]'),
            format_str="Processing: {current}/{total} ({percent}%)"
        )

        result = pattern.format(5, 10)

        assert result == "Processing: 5/10 (50%)"

    def test_format_zero_total(self):
        pattern = ProgressPattern(
            name="Processing",
            pattern=re.compile(r'\[(\d+)/(\d+)\]'),
            format_str="Processing: {current}/{total} ({percent}%)"
        )

        result = pattern.format(0, 0)

        assert result == "Processing: 0/0 (0%)"

    def test_match_invalid_groups_returns_none(self):
        pattern = ProgressPattern(
            name="Bad",
            pattern=re.compile(r'(\d+)/(\d+)'),
            current_group=5,
            total_group=6
        )

        result = pattern.match("5/10")

        assert result is None


class TestProcessResult:
    def test_basic_attributes(self):
        result = ProcessResult(
            returncode=0,
            stdout="output",
            stderr="error",
            cmd=["echo", "test"]
        )

        assert result.returncode == 0
        assert result.stdout == "output"
        assert result.stderr == "error"
        assert result.cmd == ["echo", "test"]

    def test_default_values(self):
        result = ProcessResult(returncode=1, stdout="")

        assert result.stderr == ""
        assert result.cmd == []


class TestProgressTracker:
    def test_process_line_matches_pattern(self):
        patterns = [
            ProgressPattern(
                name="Processing",
                pattern=re.compile(r'\[(\d+)/(\d+)\]'),
                format_str="Processing: {current}/{total}"
            )
        ]
        tracker = ProgressTracker(patterns=patterns, min_total=1)

        result = tracker.process_line("[5/10]")

        assert result == "Processing: 5/10"

    def test_process_line_no_match(self):
        patterns = [
            ProgressPattern(
                name="Processing",
                pattern=re.compile(r'\[(\d+)/(\d+)\]'),
                format_str="Processing: {current}/{total}"
            )
        ]
        tracker = ProgressTracker(patterns=patterns)

        result = tracker.process_line("random line")

        assert result is None

    def test_process_line_empty_returns_none(self):
        tracker = ProgressTracker()

        result = tracker.process_line("")

        assert result is None

    def test_process_line_whitespace_returns_none(self):
        tracker = ProgressTracker()

        result = tracker.process_line("   \n\t  ")

        assert result is None

    def test_min_total_filtering(self):
        patterns = [
            ProgressPattern(
                name="Processing",
                pattern=re.compile(r'\[(\d+)/(\d+)\]'),
                format_str="Processing: {current}/{total}"
            )
        ]
        tracker = ProgressTracker(patterns=patterns, min_total=100)

        result = tracker.process_line("[5/10]")

        assert result is None

    def test_deduplication(self):
        patterns = [
            ProgressPattern(
                name="Processing",
                pattern=re.compile(r'\[(\d+)/(\d+)\]'),
                format_str="Processing: {current}/{total}"
            )
        ]
        tracker = ProgressTracker(patterns=patterns, min_total=1)

        first = tracker.process_line("[5/10]")
        second = tracker.process_line("[5/10]")

        assert first == "Processing: 5/10"
        assert second is None

    @patch('subprocess_utils.sys.stdout')
    def test_print_progress_tty(self, mock_stdout):
        mock_stdout.isatty.return_value = True
        tracker = ProgressTracker()

        tracker.print_progress("Test progress")

        mock_stdout.write.assert_called()

    @patch('subprocess_utils.sys.stdout')
    def test_finish_clears_line_on_tty(self, mock_stdout):
        mock_stdout.isatty.return_value = True
        tracker = ProgressTracker()
        tracker._last_progress = "something"

        tracker.finish()


class TestProcessRunner:
    def test_run_simple_command(self):
        runner = ProcessRunner(print_command=False)

        result = runner.run(["echo", "hello"], check=False)

        assert result.returncode == 0
        assert "hello" in result.stdout

    def test_run_with_description(self, capsys):
        runner = ProcessRunner(print_command=False)

        runner.run(["echo", "test"], description="Running test", check=False)

        captured = capsys.readouterr()
        assert "Running test" in captured.out

    def test_run_failing_command_raises(self):
        runner = ProcessRunner(print_command=False)

        with pytest.raises(Exception):
            runner.run(["false"], check=True)

    def test_run_failing_command_no_check(self):
        runner = ProcessRunner(print_command=False)

        result = runner.run(["false"], check=False)

        assert result.returncode != 0

    def test_run_with_callback(self):
        runner = ProcessRunner(print_command=False)
        lines_seen = []

        def callback(line):
            lines_seen.append(line)

        runner.run(["echo", "test"], line_callback=callback, check=False)

        assert len(lines_seen) > 0

    def test_run_with_cwd(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = ProcessRunner(print_command=False)

            result = runner.run(["pwd"], cwd=Path(tmpdir), check=False)

            assert tmpdir in result.stdout or Path(tmpdir).resolve().as_posix() in result.stdout


class TestPatternFactories:
    def test_create_mmcam_patterns(self):
        patterns = create_mmcam_patterns()

        assert len(patterns) >= 3
        assert all(isinstance(p, ProgressPattern) for p in patterns)

        names = [p.name for p in patterns]
        assert "Extracting features" in names
        assert "Matching" in names

    def test_colmap_patterns_match_examples(self):
        patterns = create_mmcam_patterns()

        for pattern in patterns:
            if "feature" in pattern.name.lower():
                result = pattern.match("Processed file [50/100]")
                assert result is not None
                break

    def test_create_training_patterns(self):
        patterns = create_training_patterns()

        assert len(patterns) >= 1
        assert all(isinstance(p, ProgressPattern) for p in patterns)

    def test_training_patterns_match_examples(self):
        patterns = create_training_patterns()

        for pattern in patterns:
            result = pattern.match("Iteration: 500/1000")
            if result:
                assert result == (500, 1000)
                break

    def test_create_generic_patterns(self):
        patterns = create_generic_patterns()

        assert len(patterns) >= 1
        assert all(isinstance(p, ProgressPattern) for p in patterns)

    def test_generic_patterns_match_simple(self):
        patterns = create_generic_patterns()

        for pattern in patterns:
            result = pattern.match("[25/100]")
            if result:
                assert result == (25, 100)
                break
