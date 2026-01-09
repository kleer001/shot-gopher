#!/usr/bin/env python3
"""Set up a project directory with populated workflow templates.

Copies workflow templates to the project directory and replaces
placeholder paths with actual project paths.

Usage:
    python setup_project.py <project_dir> [--workflows-dir <path>]

Example:
    python setup_project.py /path/to/projects/My_Shot_Name
"""

import argparse
import json
import sys
from pathlib import Path

# Default workflows location (relative to this script)
DEFAULT_WORKFLOWS_DIR = Path(__file__).parent.parent / "workflow_templates"

# Path placeholders used in workflow templates
PLACEHOLDERS = {
    "{{PROJECT_DIR}}": None,      # Replaced with absolute project path
    "{{SOURCE_FRAMES}}": "source/frames",
    "{{DEPTH_DIR}}": "depth",
    "{{ROTO_DIR}}": "roto",
    "{{CLEANPLATE_DIR}}": "cleanplate",
    "{{CAMERA_DIR}}": "camera",
}

# Directories to create in each project
PROJECT_DIRS = [
    "source/frames",
    "depth",
    "roto",
    "cleanplate",
    "camera",
    "workflows",
]


def create_project_structure(project_dir: Path) -> None:
    """Create standard VFX project directory structure."""
    for subdir in PROJECT_DIRS:
        (project_dir / subdir).mkdir(parents=True, exist_ok=True)
    print(f"Created project structure in {project_dir}")


def populate_workflow(workflow_data: dict, project_dir: Path) -> dict:
    """Replace placeholder paths in workflow with actual project paths.

    Handles both {{PLACEHOLDER}} style and relative paths.
    """
    project_str = str(project_dir)

    def replace_in_value(value):
        if isinstance(value, str):
            # Replace explicit placeholders
            for placeholder, relative_path in PLACEHOLDERS.items():
                if placeholder in value:
                    if relative_path is None:
                        value = value.replace(placeholder, project_str)
                    else:
                        value = value.replace(placeholder, str(project_dir / relative_path))

            # Also handle common relative paths that should be absolute
            # Only match paths (contain '/') - don't match filename patterns like 'depth_%04d'
            relative_patterns = [
                ("source/frames/", str(project_dir / "source/frames") + "/"),
                ("source/frames", str(project_dir / "source/frames")),
                ("depth/", str(project_dir / "depth") + "/"),
                ("roto/", str(project_dir / "roto") + "/"),
                ("cleanplate/", str(project_dir / "cleanplate") + "/"),
                ("camera/", str(project_dir / "camera") + "/"),
            ]
            for pattern, replacement in relative_patterns:
                # Only replace if it's a path (starts with pattern that contains /)
                if value == pattern or value.startswith(pattern):
                    value = replacement + value[len(pattern):]

            return value
        elif isinstance(value, list):
            return [replace_in_value(item) for item in value]
        elif isinstance(value, dict):
            return {k: replace_in_value(v) for k, v in value.items()}
        else:
            return value

    return replace_in_value(workflow_data)


def copy_and_populate_workflows(
    project_dir: Path,
    workflows_dir: Path,
    workflows: list[str] | None = None
) -> list[Path]:
    """Copy workflow templates to project and populate paths.

    Args:
        project_dir: Target project directory
        workflows_dir: Source directory containing workflow templates
        workflows: Specific workflow files to copy (None = all .json files)

    Returns:
        List of created workflow paths
    """
    if not workflows_dir.exists():
        raise FileNotFoundError(f"Workflows directory not found: {workflows_dir}")

    project_workflows_dir = project_dir / "workflows"
    project_workflows_dir.mkdir(parents=True, exist_ok=True)

    # Find workflow files
    if workflows:
        workflow_files = [workflows_dir / w for w in workflows]
    else:
        workflow_files = list(workflows_dir.glob("*.json"))

    created = []
    for workflow_path in workflow_files:
        if not workflow_path.exists():
            print(f"Warning: Workflow not found: {workflow_path}", file=sys.stderr)
            continue

        # Load and populate
        with open(workflow_path) as f:
            workflow_data = json.load(f)

        populated = populate_workflow(workflow_data, project_dir)

        # Write to project
        output_path = project_workflows_dir / workflow_path.name
        with open(output_path, 'w') as f:
            json.dump(populated, f, indent=2)

        created.append(output_path)
        print(f"Created: {output_path}")

    return created


def main():
    parser = argparse.ArgumentParser(
        description="Set up a VFX project with populated workflow templates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "project_dir",
        type=Path,
        help="Project directory to set up"
    )
    parser.add_argument(
        "--workflows-dir", "-w",
        type=Path,
        default=DEFAULT_WORKFLOWS_DIR,
        help=f"Source workflows directory (default: {DEFAULT_WORKFLOWS_DIR})"
    )
    parser.add_argument(
        "--workflow",
        action="append",
        dest="workflows",
        help="Specific workflow file(s) to copy (can be repeated)"
    )
    parser.add_argument(
        "--no-create-dirs",
        action="store_true",
        help="Don't create project directory structure"
    )

    args = parser.parse_args()

    project_dir = args.project_dir.resolve()

    # Create project structure
    if not args.no_create_dirs:
        create_project_structure(project_dir)

    # Copy and populate workflows
    try:
        created = copy_and_populate_workflows(
            project_dir=project_dir,
            workflows_dir=args.workflows_dir.resolve(),
            workflows=args.workflows
        )
        print(f"\nSet up {len(created)} workflow(s) for project: {project_dir}")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
