#!/usr/bin/env python3
"""GS-IR output validation module.

Validates that GS-IR material decomposition produced useful outputs:
  - File existence (materials, normals, depth, environment)
  - Frame count consistency with input
  - Image validity (loadable, correct dimensions)
  - Content quality (not degenerate - all black/white)
  - Metadata consistency

Usage:
    python validate_gsir.py <project_dir> [options]

Example:
    python validate_gsir.py /path/to/projects/My_Shot --strict
"""

import argparse
import json
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from env_config import check_conda_env_or_warn


@dataclass
class ValidationResult:
    """Result of a validation check."""

    valid: bool
    message: str
    details: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.valid


@dataclass
class GsirValidationReport:
    """Complete validation report for GS-IR outputs."""

    project_dir: Path
    checks: dict[str, ValidationResult] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return all(check.valid for check in self.checks.values())

    @property
    def warnings(self) -> list[str]:
        return [
            f"{name}: {check.message}"
            for name, check in self.checks.items()
            if not check.valid
        ]

    def add_check(self, name: str, result: ValidationResult) -> None:
        self.checks[name] = result

    def summary(self) -> str:
        lines = [f"GS-IR Validation Report: {self.project_dir.name}"]
        lines.append("=" * 60)

        passed = sum(1 for c in self.checks.values() if c.valid)
        total = len(self.checks)
        lines.append(f"Result: {passed}/{total} checks passed")
        lines.append("")

        for name, check in self.checks.items():
            status = "PASS" if check.valid else "FAIL"
            lines.append(f"  [{status}] {name}: {check.message}")
            for detail in check.details:
                lines.append(f"         {detail}")

        return "\n".join(lines)


def count_colmap_images(sparse_model: Path) -> int:
    """Count registered images in a COLMAP sparse model.

    Args:
        sparse_model: Path to sparse model directory (containing images.bin)

    Returns:
        Number of registered images, or 0 if unable to read
    """
    images_bin = sparse_model / "images.bin"
    if images_bin.exists():
        try:
            with open(images_bin, "rb") as f:
                return struct.unpack("<Q", f.read(8))[0]
        except (IOError, struct.error):
            pass
    return 0


def check_output_directories(output_dir: Path) -> ValidationResult:
    """Check that expected output directories exist.

    Args:
        output_dir: Path to GS-IR output directory (camera/)

    Returns:
        ValidationResult with directory existence status
    """
    expected_dirs = ["materials", "normals", "depth_gsir"]
    missing = []
    found = []

    for dirname in expected_dirs:
        dirpath = output_dir / dirname
        if dirpath.exists() and dirpath.is_dir():
            found.append(dirname)
        else:
            missing.append(dirname)

    if missing:
        return ValidationResult(
            valid=False,
            message=f"Missing directories: {', '.join(missing)}",
            details=[f"Found: {', '.join(found)}" if found else "No output directories found"],
        )

    return ValidationResult(
        valid=True,
        message="All output directories present",
        details=[f"Found: {', '.join(found)}"],
    )


def check_file_counts(output_dir: Path, expected_count: int) -> ValidationResult:
    """Check that output directories contain expected number of files.

    Args:
        output_dir: Path to GS-IR output directory
        expected_count: Expected number of frames

    Returns:
        ValidationResult with file count status
    """
    directories = {
        "materials": output_dir / "materials",
        "normals": output_dir / "normals",
        "depth_gsir": output_dir / "depth_gsir",
    }

    details = []
    mismatches = []

    for name, dirpath in directories.items():
        if not dirpath.exists():
            mismatches.append(f"{name}: directory missing")
            continue

        png_count = len(list(dirpath.glob("*.png")))
        details.append(f"{name}: {png_count} files")

        if png_count == 0:
            mismatches.append(f"{name}: no PNG files")
        elif png_count != expected_count:
            mismatches.append(f"{name}: {png_count} files (expected {expected_count})")

    if mismatches:
        return ValidationResult(
            valid=False,
            message=f"File count mismatches found",
            details=mismatches + details,
        )

    return ValidationResult(
        valid=True,
        message=f"All directories contain {expected_count} files",
        details=details,
    )


def check_image_validity(output_dir: Path, sample_size: int = 5) -> ValidationResult:
    """Check that output images are valid and loadable.

    Args:
        output_dir: Path to GS-IR output directory
        sample_size: Number of images to sample from each directory

    Returns:
        ValidationResult with image validity status
    """
    try:
        from PIL import Image
    except ImportError:
        return ValidationResult(
            valid=True,
            message="Skipped (PIL not available)",
            details=["Install Pillow to enable image validation"],
        )

    directories = {
        "materials": output_dir / "materials",
        "normals": output_dir / "normals",
        "depth_gsir": output_dir / "depth_gsir",
    }

    invalid_files = []
    checked_count = 0
    dimensions = {}

    for name, dirpath in directories.items():
        if not dirpath.exists():
            continue

        png_files = sorted(dirpath.glob("*.png"))[:sample_size]

        for png_file in png_files:
            checked_count += 1
            try:
                with Image.open(png_file) as img:
                    img.verify()
                    size = img.size
                    if name not in dimensions:
                        dimensions[name] = size
                    elif dimensions[name] != size:
                        invalid_files.append(
                            f"{png_file.name}: inconsistent size {size} vs {dimensions[name]}"
                        )
            except Exception as e:
                invalid_files.append(f"{name}/{png_file.name}: {e}")

    if invalid_files:
        return ValidationResult(
            valid=False,
            message=f"Invalid images found ({len(invalid_files)} errors)",
            details=invalid_files[:10],
        )

    details = [f"Checked {checked_count} images"]
    for name, size in dimensions.items():
        details.append(f"{name}: {size[0]}x{size[1]}")

    return ValidationResult(
        valid=True,
        message="All sampled images valid",
        details=details,
    )


def check_content_quality(
    output_dir: Path,
    sample_size: int = 3,
    black_threshold: float = 0.01,
    white_threshold: float = 0.99,
    uniformity_threshold: float = 0.95,
) -> ValidationResult:
    """Check that output images contain meaningful content (not degenerate).

    Detects:
      - All black images
      - All white images
      - Nearly uniform images (single color)

    Args:
        output_dir: Path to GS-IR output directory
        sample_size: Number of images to sample
        black_threshold: Fraction below which image is considered black
        white_threshold: Fraction above which image is considered white
        uniformity_threshold: Fraction of pixels that must differ for non-uniform

    Returns:
        ValidationResult with content quality status
    """
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        return ValidationResult(
            valid=True,
            message="Skipped (PIL/numpy not available)",
            details=["Install Pillow and numpy to enable content validation"],
        )

    directories = {
        "materials": output_dir / "materials",
        "normals": output_dir / "normals",
    }

    degenerate_files = []
    checked_count = 0

    for name, dirpath in directories.items():
        if not dirpath.exists():
            continue

        png_files = sorted(dirpath.glob("*.png"))[:sample_size]

        for png_file in png_files:
            checked_count += 1
            try:
                with Image.open(png_file) as img:
                    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0

                    mean_val = arr.mean()
                    std_val = arr.std()

                    if mean_val < black_threshold:
                        degenerate_files.append(
                            f"{name}/{png_file.name}: appears all black (mean={mean_val:.4f})"
                        )
                    elif mean_val > white_threshold:
                        degenerate_files.append(
                            f"{name}/{png_file.name}: appears all white (mean={mean_val:.4f})"
                        )
                    elif std_val < 0.01:
                        degenerate_files.append(
                            f"{name}/{png_file.name}: appears uniform (std={std_val:.4f})"
                        )

            except Exception as e:
                degenerate_files.append(f"{name}/{png_file.name}: error reading - {e}")

    if degenerate_files:
        return ValidationResult(
            valid=False,
            message=f"Degenerate images detected ({len(degenerate_files)} issues)",
            details=degenerate_files,
        )

    return ValidationResult(
        valid=True,
        message=f"Content quality acceptable ({checked_count} images checked)",
        details=[],
    )


def check_environment_map(output_dir: Path) -> ValidationResult:
    """Check that environment map exists and is valid.

    Args:
        output_dir: Path to GS-IR output directory

    Returns:
        ValidationResult with environment map status
    """
    env_map = output_dir / "environment.png"

    if not env_map.exists():
        return ValidationResult(
            valid=False,
            message="Environment map not found",
            details=[f"Expected: {env_map}"],
        )

    try:
        from PIL import Image

        with Image.open(env_map) as img:
            img.verify()
            size = img.size
            return ValidationResult(
                valid=True,
                message=f"Environment map valid ({size[0]}x{size[1]})",
                details=[],
            )
    except ImportError:
        if env_map.stat().st_size > 0:
            return ValidationResult(
                valid=True,
                message="Environment map exists (validity not checked)",
                details=["Install Pillow to enable image validation"],
            )
        return ValidationResult(
            valid=False,
            message="Environment map is empty",
            details=[],
        )
    except Exception as e:
        return ValidationResult(
            valid=False,
            message=f"Environment map invalid: {e}",
            details=[],
        )


def check_metadata(output_dir: Path) -> ValidationResult:
    """Check that metadata JSON is valid and consistent with outputs.

    Args:
        output_dir: Path to GS-IR output directory

    Returns:
        ValidationResult with metadata status
    """
    metadata_path = output_dir / "gsir_metadata.json"

    if not metadata_path.exists():
        return ValidationResult(
            valid=False,
            message="Metadata file not found",
            details=[f"Expected: {metadata_path}"],
        )

    try:
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
    except json.JSONDecodeError as e:
        return ValidationResult(
            valid=False,
            message=f"Invalid JSON: {e}",
            details=[],
        )

    required_keys = ["source", "checkpoint", "iteration", "outputs"]
    missing_keys = [k for k in required_keys if k not in metadata]

    if missing_keys:
        return ValidationResult(
            valid=False,
            message=f"Missing required keys: {', '.join(missing_keys)}",
            details=[f"Found keys: {list(metadata.keys())}"],
        )

    if metadata.get("source") != "gs-ir":
        return ValidationResult(
            valid=False,
            message=f"Unexpected source: {metadata.get('source')}",
            details=["Expected: gs-ir"],
        )

    outputs = metadata.get("outputs", {})
    inconsistencies = []

    output_checks = {
        "materials": output_dir / "materials",
        "normals": output_dir / "normals",
        "depth": output_dir / "depth_gsir",
    }

    for key, expected_path in output_checks.items():
        if key in outputs and outputs[key]:
            if not expected_path.exists():
                inconsistencies.append(f"{key}: referenced but directory missing")

    env_ref = outputs.get("environment")
    if env_ref:
        env_path = output_dir / "environment.png"
        if not env_path.exists():
            inconsistencies.append("environment: referenced but file missing")

    if inconsistencies:
        return ValidationResult(
            valid=False,
            message="Metadata inconsistent with outputs",
            details=inconsistencies,
        )

    return ValidationResult(
        valid=True,
        message="Metadata valid and consistent",
        details=[f"Iteration: {metadata.get('iteration')}"],
    )


def validate_gsir_output(
    project_dir: Path,
    strict: bool = False,
) -> GsirValidationReport:
    """Run full validation on GS-IR outputs.

    Args:
        project_dir: Project directory
        strict: If True, treat warnings as errors

    Returns:
        GsirValidationReport with all check results
    """
    report = GsirValidationReport(project_dir=project_dir)
    output_dir = project_dir / "camera"
    colmap_undistorted = project_dir / "colmap" / "undistorted" / "sparse" / "0"

    expected_frame_count = count_colmap_images(colmap_undistorted)

    report.add_check("directories", check_output_directories(output_dir))
    report.add_check("metadata", check_metadata(output_dir))

    if expected_frame_count > 0:
        report.add_check("file_counts", check_file_counts(output_dir, expected_frame_count))
    else:
        report.add_check(
            "file_counts",
            ValidationResult(
                valid=True,
                message="Skipped (could not determine expected count)",
                details=["COLMAP undistorted model not found"],
            ),
        )

    report.add_check("image_validity", check_image_validity(output_dir))
    report.add_check("content_quality", check_content_quality(output_dir))
    report.add_check("environment_map", check_environment_map(output_dir))

    return report


def main() -> int:
    check_conda_env_or_warn()

    parser = argparse.ArgumentParser(
        description="Validate GS-IR material decomposition outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "project_dir",
        type=Path,
        help="Project directory with GS-IR outputs",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat all validation failures as errors",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only output on failure",
    )

    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    if not project_dir.exists():
        print(f"Error: Project directory not found: {project_dir}", file=sys.stderr)
        return 1

    report = validate_gsir_output(project_dir, strict=args.strict)

    if args.json:
        result = {
            "valid": report.valid,
            "project": str(report.project_dir),
            "checks": {
                name: {
                    "valid": check.valid,
                    "message": check.message,
                    "details": check.details,
                }
                for name, check in report.checks.items()
            },
        }
        print(json.dumps(result, indent=2))
    elif not args.quiet or not report.valid:
        print(report.summary())

    return 0 if report.valid else 1


if __name__ == "__main__":
    sys.exit(main())
