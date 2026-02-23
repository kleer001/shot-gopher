#!/usr/bin/env python3
"""Verify GVHMR camera extraction by rendering mocap mesh through exported camera.

Loads the exported camera.abc and body_motion.abc into Houdini, renders
OpenGL frames from the camera, and optionally composites with source
frames for comparison against GVHMR's 1_incam.mp4.

Run inside Houdini's Python shell or via hython:
    execfile("scripts/verify_mocap_camera.py")

Or call build_scene() / render_frames() directly from Houdini's Python shell.
"""

import hou
import os
from pathlib import Path
from typing import Optional


def build_scene(
    project_dir: str,
    camera_abc: Optional[str] = None,
    body_abc: Optional[str] = None,
    source_frames: Optional[str] = None,
) -> str:
    """Build Houdini scene with camera and body mesh for verification.

    Args:
        project_dir: Project directory path.
        camera_abc: Path to camera Alembic (default: project/mocap_camera/camera.abc).
        body_abc: Path to body Alembic (default: project/mocap/person/export/body_motion.abc).
        source_frames: Path to source frames directory (default: project/source/frames/).

    Returns:
        Camera object path in Houdini for rendering.
    """
    project = Path(project_dir)

    camera_abc = camera_abc or str(project / "mocap_camera" / "camera.abc")
    body_abc = body_abc or str(project / "mocap" / "person" / "export" / "body_motion.abc")
    source_frames = source_frames or str(project / "source" / "frames")

    for path, label in [(camera_abc, "camera.abc"), (body_abc, "body_motion.abc")]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    obj = hou.node("/obj")

    existing = [n.name() for n in obj.children()]
    for name in ["verify_camera", "verify_body", "verify_bg"]:
        if name in existing:
            obj.node(name).destroy()

    cam_archive = obj.createNode("alembicarchive", "verify_camera")
    cam_archive.parm("fileName").set(camera_abc)
    cam_archive.parm("buildHierarchy").pressButton()

    body_archive = obj.createNode("alembicarchive", "verify_body")
    body_archive.parm("fileName").set(body_abc)
    body_archive.parm("buildHierarchy").pressButton()

    cam_archive.moveToGoodPosition()
    body_archive.moveToGoodPosition()

    cam_path = _find_camera(cam_archive)
    if not cam_path:
        raise RuntimeError(
            f"No camera found in {camera_abc}. "
            f"Children: {[c.path() for c in cam_archive.allSubChildren()]}"
        )

    cam_node = hou.node(cam_path)
    frame_range = _get_frame_range(cam_node)
    if frame_range:
        hou.playbar.setFrameRange(frame_range[0], frame_range[1])
        hou.playbar.setPlaybackRange(frame_range[0], frame_range[1])
        hou.setFrame(frame_range[0])

    _setup_bg_image(cam_node, source_frames)

    print(f"Scene built:")
    print(f"  Camera: {cam_path}")
    print(f"  Body:   {body_archive.path()}")
    if frame_range:
        print(f"  Frames: {frame_range[0]}-{frame_range[1]}")
    print(f"\nTo render: render_frames('{project_dir}', '{cam_path}')")

    return cam_path


def render_frames(
    project_dir: str,
    camera_path: str,
    output_dir: Optional[str] = None,
    frame_range: Optional[tuple] = None,
    resolution: Optional[tuple] = None,
) -> str:
    """Render OpenGL frames from the verification camera.

    Args:
        project_dir: Project directory path.
        camera_path: Houdini camera node path.
        output_dir: Output directory (default: project/verify_camera/).
        frame_range: (start, end) frames, or None for full range.
        resolution: (width, height) or None for camera defaults.

    Returns:
        Path to output directory containing rendered frames.
    """
    project = Path(project_dir)
    output_dir = output_dir or str(project / "verify_camera")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cam_node = hou.node(camera_path)
    if cam_node is None:
        raise RuntimeError(f"Camera not found: {camera_path}")

    if frame_range is None:
        frame_range = _get_frame_range(cam_node)
    if frame_range is None:
        frame_range = (1, 100)

    output_pattern = str(Path(output_dir) / "verify.$F4.png")

    flipbook_opts = hou.FlipbookSettings()
    flipbook_opts.frameRange(frame_range)
    flipbook_opts.output(output_pattern)
    if resolution:
        flipbook_opts.resolution(resolution)

    scene_viewer = _get_scene_viewer()
    if scene_viewer is None:
        print("No scene viewer found. Using opengl ROP instead.")
        _render_with_rop(camera_path, output_pattern, frame_range, resolution)
    else:
        cur_viewport = scene_viewer.curViewport()
        cur_viewport.setCamera(cam_node)
        scene_viewer.flipbook(settings=flipbook_opts)

    print(f"Rendered frames to: {output_dir}")
    print(f"  Compare with: {project / 'mocap' / 'person' / 'gvhmr' / '_gvhmr_input' / '1_incam.mp4'}")
    return output_dir


def _find_camera(archive_node: hou.Node) -> Optional[str]:
    """Find the camera node inside an Alembic archive."""
    for child in archive_node.allSubChildren():
        if child.type().name() == "cam":
            return child.path()
        if hasattr(child, "type") and "alembicxform" in child.type().name():
            for sub in child.children():
                if sub.type().name() == "cam":
                    return sub.path()
    return None


def _get_frame_range(cam_node: hou.Node) -> Optional[tuple]:
    """Get frame range from camera keyframes."""
    try:
        parm = cam_node.parm("tx")
        if parm is None:
            parent = cam_node.parent()
            if parent:
                parm = parent.parm("tx")
        if parm is None:
            return None
        keyframes = parm.keyframes()
        if not keyframes:
            return None
        return (int(keyframes[0].frame()), int(keyframes[-1].frame()))
    except Exception:
        return None


def _setup_bg_image(cam_node: hou.Node, frames_dir: str) -> None:
    """Set source frames as camera background image if available."""
    frames_path = Path(frames_dir)
    if not frames_path.exists():
        return

    pngs = sorted(frames_path.glob("*.png"))
    jpgs = sorted(frames_path.glob("*.jpg"))
    frames = pngs or jpgs
    if not frames:
        return

    first = frames[0]
    stem = first.stem
    ext = first.suffix

    if "_" in stem:
        prefix = stem.rsplit("_", 1)[0]
        num_part = stem.rsplit("_", 1)[1]
        num_digits = len(num_part)
        pattern = str(first.parent / f"{prefix}_{'$F' if num_digits <= 1 else '$F' + str(num_digits)}{ext}")
    else:
        pattern = str(first)

    try:
        cam_node.parm("vm_background").set(pattern)
    except Exception:
        pass


def _get_scene_viewer() -> Optional[hou.SceneViewer]:
    """Get current scene viewer pane."""
    try:
        desktop = hou.ui.curDesktop()
        for pane in desktop.paneTabs():
            if isinstance(pane, hou.SceneViewer):
                return pane
    except Exception:
        pass
    return None


def _render_with_rop(
    camera_path: str,
    output_pattern: str,
    frame_range: tuple,
    resolution: Optional[tuple],
) -> None:
    """Render using an OpenGL ROP when no scene viewer is available."""
    out = hou.node("/out")
    existing = out.node("verify_gl")
    if existing:
        existing.destroy()

    gl_rop = out.createNode("opengl", "verify_gl")
    gl_rop.parm("camera").set(camera_path)
    gl_rop.parm("picture").set(output_pattern)
    gl_rop.parm("trange").set(1)
    gl_rop.parm("f1").set(frame_range[0])
    gl_rop.parm("f2").set(frame_range[1])
    if resolution:
        gl_rop.parm("tres").set(1)
        gl_rop.parm("res1").set(resolution[0])
        gl_rop.parm("res2").set(resolution[1])
    gl_rop.parm("execute").pressButton()


if __name__ == "__main__" or hou.isUIAvailable():
    import sys

    project = os.environ.get("VFX_PROJECT_DIR")
    if project:
        cam = build_scene(project)
        print(f"\nReady. Call render_frames('{project}', '{cam}') to render.")
    else:
        print("Usage:")
        print("  1. Set project: cam = build_scene('/path/to/project')")
        print("  2. Render:      render_frames('/path/to/project', cam)")
