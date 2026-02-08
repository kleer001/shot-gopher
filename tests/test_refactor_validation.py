"""Refactor validation tests.

Temporary test suite to verify the three refactors described in:
  - docs/admin/PROPAINTER_DEPRECATION_ROADMAP.md
  - docs/admin/REMOVE_STAGES_ALL_ROADMAP.md
  - docs/admin/COLMAP_TO_MMCAM_ROADMAP.md

Run BEFORE refactoring to establish baseline (tests marked 'pre' should pass,
tests marked 'post' should fail). Run AFTER to confirm completion (all pass).

Delete this file once the refactors are verified and merged.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

REPO_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grep_files(directory: Path, pattern: str, extensions: tuple[str, ...] = (".py",)) -> list[tuple[Path, int, str]]:
    """Return (path, line_number, line_text) for every match."""
    hits: list[tuple[Path, int, str]] = []
    for ext in extensions:
        for path in directory.rglob(f"*{ext}"):
            for i, line in enumerate(path.read_text().splitlines(), 1):
                if pattern in line:
                    hits.append((path, i, line))
    return hits


# ===================================================================
# 1. ProPainter Deprecation
# ===================================================================


class TestProPainterRemoval:
    """Verify ProPainter is fully removed from the codebase."""

    def test_no_propainter_in_scripts(self) -> None:
        """No Python file under scripts/ should reference ProPainter."""
        hits = _grep_files(REPO_ROOT / "scripts", "ProPainter")
        hits += _grep_files(REPO_ROOT / "scripts", "propainter")
        filtered = [h for h in hits if "ROADMAP" not in str(h[0]) and "BREADCRUMB" not in h[2]]
        assert filtered == [], (
            f"ProPainter references remain:\n"
            + "\n".join(f"  {p}:{n}: {t.strip()}" for p, n, t in filtered)
        )

    def test_no_propainter_in_workflow_templates(self) -> None:
        """No workflow template should contain ProPainter nodes."""
        template_dir = REPO_ROOT / "workflow_templates"
        hits: list[tuple[Path, str]] = []
        for path in template_dir.glob("*.json"):
            text = path.read_text()
            if "ProPainter" in text or "propainter" in text:
                hits.append((path, "contains ProPainter reference"))
        assert hits == [], f"ProPainter in templates: {hits}"

    def test_cleanplate_templates_deleted(self) -> None:
        """The three ProPainter-based cleanplate workflows must not exist."""
        template_dir = REPO_ROOT / "workflow_templates"
        forbidden = [
            "03_cleanplate.json",
            "03_cleanplate_batched.json",
            "03_cleanplate_chunk_template.json",
        ]
        remaining = [f for f in forbidden if (template_dir / f).exists()]
        assert remaining == [], f"ProPainter workflow templates still exist: {remaining}"

    def test_no_cleanplate_use_median_field(self) -> None:
        """The cleanplate_use_median field must be removed from config."""
        hits = _grep_files(REPO_ROOT / "scripts", "cleanplate_use_median")
        assert hits == [], (
            f"cleanplate_use_median still referenced:\n"
            + "\n".join(f"  {p}:{n}: {t.strip()}" for p, n, t in hits)
        )

    def test_no_cleanplate_median_cli_flag(self) -> None:
        """--cleanplate-median should no longer appear in argparse."""
        hits = _grep_files(REPO_ROOT / "scripts", "--cleanplate-median")
        hits += _grep_files(REPO_ROOT / "scripts", "cleanplate_median")
        # Allow the module cleanplate_median.py itself and its import
        filtered = [
            h for h in hits
            if h[0].name != "cleanplate_median.py"
            and "from cleanplate_median" not in h[2]
            and "import cleanplate_median" not in h[2]
            and "run_cleanplate_median" not in h[2]
        ]
        assert filtered == [], (
            f"--cleanplate-median flag still referenced:\n"
            + "\n".join(f"  {p}:{n}: {t.strip()}" for p, n, t in filtered)
        )

    def test_cleanplate_stage_description_updated(self) -> None:
        """Cleanplate stage description should mention static camera."""
        from pipeline_constants import STAGES

        desc = STAGES["cleanplate"].lower()
        assert "static" in desc or "median" in desc, (
            f"Cleanplate description not updated for static-camera-only: '{STAGES['cleanplate']}'"
        )

    def test_cleanplate_not_in_comfyui_stages(self) -> None:
        """Cleanplate should not require ComfyUI after ProPainter removal."""
        run_pipeline_path = REPO_ROOT / "scripts" / "run_pipeline.py"
        text = run_pipeline_path.read_text()
        for line in text.splitlines():
            if "comfyui_stages" in line and "cleanplate" in line:
                pytest.fail(
                    f"'cleanplate' still listed in comfyui_stages: {line.strip()}"
                )

    def test_update_workflow_resolution_no_propainter_param(self) -> None:
        """update_workflow_resolution should not accept update_propainter."""
        from workflow_utils import update_workflow_resolution
        import inspect

        sig = inspect.signature(update_workflow_resolution)
        assert "update_propainter" not in sig.parameters, (
            "update_workflow_resolution still has update_propainter parameter"
        )

    def test_no_propainter_in_install_wizard(self) -> None:
        """Install wizard should not reference ProPainter."""
        wizard_dir = REPO_ROOT / "scripts" / "install_wizard"
        hits = _grep_files(wizard_dir, "ProPainter")
        hits += _grep_files(wizard_dir, "propainter")
        assert hits == [], (
            f"ProPainter in install wizard:\n"
            + "\n".join(f"  {p}:{n}: {t.strip()}" for p, n, t in hits)
        )

    def test_web_config_cleanplate_no_comfyui(self) -> None:
        """Web config should mark cleanplate as not requiring ComfyUI."""
        config_path = REPO_ROOT / "web" / "config" / "pipeline_config.json"
        config = json.loads(config_path.read_text())
        cleanplate = config["stages"]["cleanplate"]
        assert cleanplate["requiresComfyUI"] is False, (
            "web pipeline_config.json still has cleanplate requiresComfyUI=true"
        )

    def test_batched_cleanplate_removed(self) -> None:
        """run_cleanplate_batched.py is dead code after ProPainter removal."""
        assert not (REPO_ROOT / "scripts" / "run_cleanplate_batched.py").exists(), (
            "run_cleanplate_batched.py still exists (dead code: batching was for ProPainter VRAM)"
        )
        assert not (REPO_ROOT / "tests" / "test_run_cleanplate_batched.py").exists(), (
            "test_run_cleanplate_batched.py still exists (tests dead code)"
        )

    def test_web_pipeline_runner_no_cleanplate_median_flag(self) -> None:
        """web/pipeline_runner.py should not append --cleanplate-median (flag removed)."""
        runner_path = REPO_ROOT / "web" / "pipeline_runner.py"
        text = runner_path.read_text()
        assert "--cleanplate-median" not in text, (
            "web/pipeline_runner.py still appends --cleanplate-median flag"
        )

    def test_web_js_no_propainter_option(self) -> None:
        """Frontend cleanplate options should not offer 'propainter' as a method."""
        js_path = REPO_ROOT / "web" / "static" / "js" / "controllers" / "ProjectsController.js"
        text = js_path.read_text()
        assert "propainter" not in text, (
            "ProjectsController.js still references 'propainter' as cleanplate method"
        )


# ===================================================================
# 2. Remove -s all
# ===================================================================


class TestRemoveStagesAll:
    """Verify the 'all' stage shorthand is removed."""

    def test_no_stages_default_all_in_argparse(self) -> None:
        """--stages argument should not default to 'all'."""
        run_pipeline_path = REPO_ROOT / "scripts" / "run_pipeline.py"
        text = run_pipeline_path.read_text()
        # The default="all" may be on a different line than "--stages",
        # so scan the add_argument block context around --stages
        lines = text.splitlines()
        in_stages_block = False
        for line in lines:
            if '"--stages"' in line or "'--stages'" in line:
                in_stages_block = True
            if in_stages_block:
                if 'default="all"' in line or "default='all'" in line:
                    pytest.fail(f"--stages still defaults to 'all': {line.strip()}")
                if line.strip() == ")":
                    in_stages_block = False

    def test_no_stages_all_branch(self) -> None:
        """The 'if stages == all' branch should be gone from run_pipeline.py."""
        run_pipeline_path = REPO_ROOT / "scripts" / "run_pipeline.py"
        text = run_pipeline_path.read_text()
        for line in text.splitlines():
            stripped = line.strip()
            if '== "all"' in stripped or "== 'all'" in stripped:
                pytest.fail(f"'all' stage branch still present: {stripped}")

    def test_pipeline_config_default_not_all_stages(self) -> None:
        """PipelineConfig.stages default should be empty, not STAGE_ORDER."""
        from pipeline_config import PipelineConfig

        config = PipelineConfig()
        assert config.stages == [], (
            f"PipelineConfig.stages defaults to {config.stages}, expected []"
        )

    def test_web_config_no_all_preset(self) -> None:
        """Web pipeline config should not have an 'all' preset."""
        config_path = REPO_ROOT / "web" / "config" / "pipeline_config.json"
        config = json.loads(config_path.read_text())
        presets = config.get("presets", {})
        assert "all" not in presets, "Web config still has 'all' preset"

    def test_no_stages_all_in_docstring(self) -> None:
        """Module docstring of run_pipeline.py should not show --stages all."""
        run_pipeline_path = REPO_ROOT / "scripts" / "run_pipeline.py"
        text = run_pipeline_path.read_text()
        # Only check lines that look like example usage, not grep-like references
        for line in text.splitlines():
            if "--stages all" in line and "BREADCRUMB" not in line:
                pytest.fail(f"'--stages all' in run_pipeline.py: {line.strip()}")


# ===================================================================
# 3. COLMAP → matchmove_camera / mmcam
# ===================================================================


class TestColmapToMmcam:
    """Verify COLMAP is renamed to matchmove_camera / mmcam in all public-facing code."""

    def test_stage_key_is_matchmove_camera(self) -> None:
        """STAGES dict should use 'matchmove_camera', not 'colmap'."""
        from pipeline_constants import STAGES

        assert "matchmove_camera" in STAGES, "STAGES missing 'matchmove_camera' key"
        assert "colmap" not in STAGES, "STAGES still has 'colmap' key"

    def test_stage_order_uses_matchmove_camera(self) -> None:
        """STAGE_ORDER should list 'matchmove_camera', not 'colmap'."""
        from pipeline_constants import STAGE_ORDER

        assert "matchmove_camera" in STAGE_ORDER
        assert "colmap" not in STAGE_ORDER

    def test_stages_requiring_frames_updated(self) -> None:
        """STAGES_REQUIRING_FRAMES should use 'matchmove_camera'."""
        from pipeline_constants import STAGES_REQUIRING_FRAMES

        assert "matchmove_camera" in STAGES_REQUIRING_FRAMES
        assert "colmap" not in STAGES_REQUIRING_FRAMES

    def test_config_fields_renamed(self) -> None:
        """PipelineConfig should have mmcam_ fields, not colmap_ fields."""
        from pipeline_config import PipelineConfig

        config = PipelineConfig()
        assert hasattr(config, "mmcam_quality"), "Missing mmcam_quality field"
        assert hasattr(config, "mmcam_dense"), "Missing mmcam_dense field"
        assert hasattr(config, "mmcam_mesh"), "Missing mmcam_mesh field"
        assert hasattr(config, "mmcam_use_masks"), "Missing mmcam_use_masks field"
        assert hasattr(config, "mmcam_max_size"), "Missing mmcam_max_size field"
        assert not hasattr(config, "colmap_quality"), "colmap_quality still exists"

    def test_stage_handler_registered(self) -> None:
        """STAGE_HANDLERS should have 'matchmove_camera', not 'colmap'."""
        from stage_runners import STAGE_HANDLERS

        assert "matchmove_camera" in STAGE_HANDLERS, "matchmove_camera handler missing"
        assert "colmap" not in STAGE_HANDLERS, "colmap handler still registered"

    def test_run_matchmove_camera_script_exists(self) -> None:
        """run_matchmove_camera.py should exist, run_colmap.py should not."""
        scripts_dir = REPO_ROOT / "scripts"
        assert (scripts_dir / "run_matchmove_camera.py").exists(), "run_matchmove_camera.py not found"
        assert not (scripts_dir / "run_colmap.py").exists(), "run_colmap.py still exists"

    def test_cli_args_renamed(self) -> None:
        """CLI should have --mmcam-* flags, not --colmap-* flags."""
        run_pipeline_path = REPO_ROOT / "scripts" / "run_pipeline.py"
        text = run_pipeline_path.read_text()
        for line in text.splitlines():
            if '"--colmap-' in line:
                pytest.fail(f"Old --colmap- flag still present: {line.strip()}")
        assert "--mmcam-quality" in text or "--matchmove-camera-quality" in text, (
            "Neither --mmcam-quality nor --matchmove-camera-quality found in run_pipeline.py"
        )

    def test_no_colmap_config_fields_in_run_pipeline(self) -> None:
        """run_pipeline.py should not reference colmap_ config fields."""
        run_pipeline_path = REPO_ROOT / "scripts" / "run_pipeline.py"
        lines = run_pipeline_path.read_text().splitlines()
        old_fields = ["colmap_quality", "colmap_dense", "colmap_mesh", "colmap_use_masks", "colmap_max_size"]
        for field in old_fields:
            for line in lines:
                if field in line and "BREADCRUMB" not in line:
                    pytest.fail(f"Old config field '{field}' still in run_pipeline.py")

    def test_web_config_stage_key_renamed(self) -> None:
        """Web pipeline config should use 'matchmove_camera', not 'colmap'."""
        config_path = REPO_ROOT / "web" / "config" / "pipeline_config.json"
        config = json.loads(config_path.read_text())
        stages = config.get("stages", {})
        assert "matchmove_camera" in stages, "Web config missing 'matchmove_camera' stage"
        assert "colmap" not in stages, "Web config still has 'colmap' stage"

    def test_web_config_output_dir_is_mmcam(self) -> None:
        """Web config outputDir for matchmove_camera should be 'mmcam'."""
        config_path = REPO_ROOT / "web" / "config" / "pipeline_config.json"
        config = json.loads(config_path.read_text())
        stage = config["stages"]["matchmove_camera"]
        assert stage["outputDir"] == "mmcam", (
            f"matchmove_camera outputDir is '{stage['outputDir']}', expected 'mmcam'"
        )

    def test_web_config_dependencies_updated(self) -> None:
        """Stages that depended on 'colmap' should now depend on 'matchmove_camera'."""
        config_path = REPO_ROOT / "web" / "config" / "pipeline_config.json"
        config = json.loads(config_path.read_text())
        stages = config.get("stages", {})
        for name, stage in stages.items():
            deps = stage.get("dependencies", [])
            assert "colmap" not in deps, (
                f"Stage '{name}' still depends on 'colmap', should be 'matchmove_camera'"
            )

    def test_gsir_stage_uses_mmcam_path(self) -> None:
        """GS-IR stage runner should look for mmcam/ directory, not colmap/."""
        stage_runners_path = REPO_ROOT / "scripts" / "stage_runners.py"
        text = stage_runners_path.read_text()
        # Find the gsir stage function and check it references mmcam
        in_gsir = False
        for line in text.splitlines():
            if "def run_stage_gsir" in line:
                in_gsir = True
            elif in_gsir and line.startswith("def "):
                break
            elif in_gsir and '"colmap"' in line:
                pytest.fail(f"run_stage_gsir still references 'colmap' path: {line.strip()}")

    def test_no_colmap_stage_key_in_constants(self) -> None:
        """pipeline_constants.py should not contain 'colmap' as a stage key."""
        constants_path = REPO_ROOT / "scripts" / "pipeline_constants.py"
        text = constants_path.read_text()
        # Check specifically for "colmap" as a dict key or list element
        if '"colmap"' in text:
            pytest.fail("'colmap' string literal still in pipeline_constants.py")

    def test_subprocess_utils_pattern_renamed(self) -> None:
        """create_colmap_patterns should be renamed to create_mmcam_patterns."""
        sub_path = REPO_ROOT / "scripts" / "subprocess_utils.py"
        text = sub_path.read_text()
        assert "create_colmap_patterns" not in text, "create_colmap_patterns still in subprocess_utils.py"
        assert "create_mmcam_patterns" in text, "create_mmcam_patterns not found in subprocess_utils.py"

    def test_debug_script_renamed(self) -> None:
        """debug_colmap_images.py should be renamed to debug_mmcam_images.py."""
        scripts_dir = REPO_ROOT / "scripts"
        assert not (scripts_dir / "debug_colmap_images.py").exists(), "debug_colmap_images.py still exists"
        assert (scripts_dir / "debug_mmcam_images.py").exists(), "debug_mmcam_images.py not found"

    def test_shell_launchers_renamed(self) -> None:
        """Shell launcher scripts should be renamed from colmap to matchmove-camera."""
        src_dir = REPO_ROOT / "src"
        assert (src_dir / "run-matchmove-camera.sh").exists(), "run-matchmove-camera.sh not found"
        assert (src_dir / "run-matchmove-camera.bat").exists(), "run-matchmove-camera.bat not found"
        if (src_dir / "run-colmap.sh").exists():
            content = (src_dir / "run-colmap.sh").read_text()
            assert "renamed" in content.lower() or "ERROR" in content, \
                "run-colmap.sh exists but is not a tombstone"
        if (src_dir / "run-colmap.bat").exists():
            content = (src_dir / "run-colmap.bat").read_text()
            assert "renamed" in content.lower() or "ERROR" in content, \
                "run-colmap.bat exists but is not a tombstone"


# ===================================================================
# 4. Cross-cutting: Web frontend/backend consistency
# ===================================================================


class TestWebConsistencyPostRefactor:
    """Verify frontend and backend stay in sync after all refactors."""

    def test_web_config_stages_match_pipeline_constants(self) -> None:
        """Web config stage keys should match pipeline_constants.STAGES keys."""
        from pipeline_constants import STAGES

        config_path = REPO_ROOT / "web" / "config" / "pipeline_config.json"
        config = json.loads(config_path.read_text())
        web_stages = set(config.get("stages", {}).keys())

        # Web config may omit 'camera' since it's auto-triggered
        pipeline_stages = set(STAGES.keys()) - {"camera"}
        web_stages_comparable = web_stages - {"camera"}

        assert pipeline_stages == web_stages_comparable, (
            f"Stage key mismatch.\n"
            f"  Pipeline only: {pipeline_stages - web_stages_comparable}\n"
            f"  Web only: {web_stages_comparable - pipeline_stages}"
        )

    def test_frontend_backend_output_dirs_consistent(self) -> None:
        """Output directory mapping should be consistent after renames."""
        config_path = REPO_ROOT / "web" / "config" / "pipeline_config.json"
        config = json.loads(config_path.read_text())

        for stage_key, stage_config in config.get("stages", {}).items():
            output_dir = stage_config.get("outputDir", "")
            assert "colmap" not in output_dir, (
                f"Stage '{stage_key}' outputDir still references colmap: '{output_dir}'"
            )

    def test_web_project_repository_uses_mmcam(self) -> None:
        """Project repository should map to 'mmcam' directory, not 'colmap'."""
        repo_path = REPO_ROOT / "web" / "repositories" / "project_repository.py"
        text = repo_path.read_text()
        if "'colmap'" in text or '"colmap"' in text:
            pytest.fail("project_repository.py still references 'colmap' directory")


# ===================================================================
# 5. Cross-cutting: No orphaned references
# ===================================================================


class TestNoOrphanedReferences:
    """Sweep for stale references that slipped through individual checks."""

    @pytest.mark.parametrize("term", ["ProPainter", "propainter", "PROPAINTER"])
    def test_no_propainter_in_python(self, term: str) -> None:
        """No ProPainter references in any Python file (except roadmap and this test)."""
        hits = _grep_files(REPO_ROOT / "scripts", term)
        hits += _grep_files(REPO_ROOT / "web", term)
        hits += _grep_files(REPO_ROOT / "tests", term)
        filtered = [
            h for h in hits
            if "ROADMAP" not in str(h[0])
            and "BREADCRUMB" not in h[2]
            and "test_refactor_validation" not in str(h[0])
        ]
        assert filtered == [], (
            f"'{term}' still found:\n"
            + "\n".join(f"  {p}:{n}: {t.strip()}" for p, n, t in filtered)
        )

    def test_no_colmap_stage_key_anywhere(self) -> None:
        """'colmap' as a stage key should not appear in any Python or JSON file."""
        hits: list[tuple[Path, int, str]] = []
        hits += _grep_files(REPO_ROOT / "scripts", '"colmap"')
        hits += _grep_files(REPO_ROOT / "web", '"colmap"', (".py", ".json", ".js"))
        # Filter: allow internal references (COLMAP binary, conda env, comments about the tool)
        filtered = []
        for path, num, line in hits:
            stripped = line.strip()
            # Allow: COLMAP_CONDA_ENV, colmap binary calls, roadmap docs
            if any(ok in stripped for ok in [
                "COLMAP_CONDA_ENV",
                "colmap feature",
                "colmap exhaustive",
                "colmap mapper",
                "colmap image",
                "colmap model",
                "colmap patch",
                "colmap stereo",
                "colmap dense",
                "colmap_raw",
                "ROADMAP",
                '"apt"',
                '"yum"',
                '"dnf"',
                '"brew"',
                "find_tool",
                "install_tool",
                'dependency == "colmap"',
                'envs" / "colmap"',
                '"bin" / "colmap"',
                "BREADCRUMB",
                "test_refactor_validation",
                "ENV_NAME",
                "CHANNEL",
                "SNAP_RESTRICTED",
                "colmap_urls",
            ]):
                continue
            # Allow install_wizard/ internal COLMAP tool references
            if "install_wizard" in str(path):
                continue
            # Allow internal COLMAP binary/data-source refs in scripts
            if path.name in ("run_matchmove_camera.py", "export_camera.py",
                             "debug_mmcam_images.py", "diagnose.py"):
                continue
            filtered.append((path, num, line))
        assert filtered == [], (
            f"'colmap' stage key still found:\n"
            + "\n".join(f"  {p}:{n}: {t.strip()}" for p, n, t in filtered)
        )

    def test_no_stages_all_in_cli_help_text(self) -> None:
        """No help text should mention 'all' as a valid stage value."""
        run_pipeline_path = REPO_ROOT / "scripts" / "run_pipeline.py"
        text = run_pipeline_path.read_text()
        for line in text.splitlines():
            if "or 'all'" in line or 'or "all"' in line:
                pytest.fail(f"Help text still mentions 'all': {line.strip()}")
