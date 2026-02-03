#!/usr/bin/env python3
"""Blender script to export GS-IR materials to USD format.

This script runs inside Blender's Python environment and creates
a USD file with PBR materials from GS-IR output.

Usage (from command line):
    blender -b --python export_gsir_usd.py -- \
        --input /path/to/camera/ \
        --output /path/to/materials.usd

The script will:
1. Load material maps (albedo, roughness, metallic, normals)
2. Create a Principled BSDF material
3. Optionally create geometry (dome for environment, card for materials)
4. Export to USD format
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import bpy


def clear_scene():
    """Remove all objects from the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    for collection in list(bpy.data.collections):
        bpy.data.collections.remove(collection)

    for material in list(bpy.data.materials):
        bpy.data.materials.remove(material)

    for image in list(bpy.data.images):
        bpy.data.images.remove(image)


def load_gsir_metadata(camera_dir: Path) -> Optional[dict]:
    """Load GS-IR metadata from JSON file.

    Args:
        camera_dir: Path to camera/ directory

    Returns:
        Metadata dict or None if not found
    """
    metadata_path = camera_dir / "gsir_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, encoding='utf-8') as f:
            return json.load(f)
    return None


def find_material_textures(camera_dir: Path) -> dict:
    """Find available material texture files.

    Args:
        camera_dir: Path to camera/ directory

    Returns:
        Dict mapping texture type to file path
    """
    textures = {}

    materials_dir = camera_dir / "materials"
    normals_dir = camera_dir / "normals"
    env_map = camera_dir / "environment.png"

    if materials_dir.exists():
        albedo_candidates = list(materials_dir.glob("*albedo*.png"))
        if not albedo_candidates:
            albedo_candidates = list(materials_dir.glob("*basecolor*.png"))
        if not albedo_candidates:
            albedo_candidates = list(materials_dir.glob("*color*.png"))
        if albedo_candidates:
            textures["albedo"] = albedo_candidates[0]

        roughness_candidates = list(materials_dir.glob("*roughness*.png"))
        if roughness_candidates:
            textures["roughness"] = roughness_candidates[0]

        metallic_candidates = list(materials_dir.glob("*metallic*.png"))
        if not metallic_candidates:
            metallic_candidates = list(materials_dir.glob("*metal*.png"))
        if metallic_candidates:
            textures["metallic"] = metallic_candidates[0]

    if normals_dir.exists():
        normal_candidates = list(normals_dir.glob("*normal*.png"))
        if not normal_candidates:
            normal_candidates = list(normals_dir.glob("*.png"))
        if normal_candidates:
            textures["normal"] = normal_candidates[0]

    if env_map.exists():
        textures["environment"] = env_map

    return textures


def create_pbr_material(
    name: str,
    textures: dict,
) -> bpy.types.Material:
    """Create a PBR material with texture maps.

    Args:
        name: Material name
        textures: Dict mapping texture type to file path

    Returns:
        The created material
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    nodes.clear()

    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    output_node.location = (400, 0)

    bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf_node.location = (0, 0)

    links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])

    y_offset = 300
    x_offset = -600

    if "albedo" in textures:
        tex_node = nodes.new(type='ShaderNodeTexImage')
        tex_node.location = (x_offset, y_offset)
        tex_node.image = bpy.data.images.load(str(textures["albedo"]))
        tex_node.image.colorspace_settings.name = 'sRGB'
        links.new(tex_node.outputs['Color'], bsdf_node.inputs['Base Color'])
        y_offset -= 300

    if "roughness" in textures:
        tex_node = nodes.new(type='ShaderNodeTexImage')
        tex_node.location = (x_offset, y_offset)
        tex_node.image = bpy.data.images.load(str(textures["roughness"]))
        tex_node.image.colorspace_settings.name = 'Non-Color'
        links.new(tex_node.outputs['Color'], bsdf_node.inputs['Roughness'])
        y_offset -= 300

    if "metallic" in textures:
        tex_node = nodes.new(type='ShaderNodeTexImage')
        tex_node.location = (x_offset, y_offset)
        tex_node.image = bpy.data.images.load(str(textures["metallic"]))
        tex_node.image.colorspace_settings.name = 'Non-Color'
        links.new(tex_node.outputs['Color'], bsdf_node.inputs['Metallic'])
        y_offset -= 300

    if "normal" in textures:
        tex_node = nodes.new(type='ShaderNodeTexImage')
        tex_node.location = (x_offset - 200, y_offset)
        tex_node.image = bpy.data.images.load(str(textures["normal"]))
        tex_node.image.colorspace_settings.name = 'Non-Color'

        normal_map_node = nodes.new(type='ShaderNodeNormalMap')
        normal_map_node.location = (x_offset + 200, y_offset)

        links.new(tex_node.outputs['Color'], normal_map_node.inputs['Color'])
        links.new(normal_map_node.outputs['Normal'], bsdf_node.inputs['Normal'])

    return mat


def create_environment_material(env_path: Path) -> bpy.types.Material:
    """Create an environment/emission material.

    Args:
        env_path: Path to environment map

    Returns:
        The created material
    """
    mat = bpy.data.materials.new(name="gsir_environment")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    nodes.clear()

    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    output_node.location = (400, 0)

    emission_node = nodes.new(type='ShaderNodeEmission')
    emission_node.location = (0, 0)

    tex_node = nodes.new(type='ShaderNodeTexImage')
    tex_node.location = (-400, 0)
    tex_node.image = bpy.data.images.load(str(env_path))
    tex_node.image.colorspace_settings.name = 'sRGB'

    links.new(tex_node.outputs['Color'], emission_node.inputs['Color'])
    links.new(emission_node.outputs['Emission'], output_node.inputs['Surface'])

    return mat


def create_material_card(
    material: bpy.types.Material,
    name: str = "material_card",
) -> bpy.types.Object:
    """Create a plane with the material applied.

    Args:
        material: Material to apply
        name: Object name

    Returns:
        The created plane object
    """
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0, 0, 0))
    plane = bpy.context.active_object
    plane.name = name

    if plane.data.materials:
        plane.data.materials[0] = material
    else:
        plane.data.materials.append(material)

    return plane


def create_environment_dome(
    material: bpy.types.Material,
    radius: float = 10.0,
) -> bpy.types.Object:
    """Create a dome/sphere with environment material.

    Args:
        material: Environment material
        radius: Dome radius

    Returns:
        The created dome object
    """
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius,
        segments=64,
        ring_count=32,
        location=(0, 0, 0)
    )
    dome = bpy.context.active_object
    dome.name = "environment_dome"

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.flip_normals()
    bpy.ops.object.mode_set(mode='OBJECT')

    if dome.data.materials:
        dome.data.materials[0] = material
    else:
        dome.data.materials.append(material)

    return dome


def export_usd(
    output_path: Path,
    export_textures: bool = True,
):
    """Export scene to USD file.

    Args:
        output_path: Output .usd/.usda/.usdc file path
        export_textures: Whether to copy textures alongside USD
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.usd_export(
        filepath=str(output_path),
        selected_objects_only=False,
        visible_objects_only=True,
        export_animation=False,
        export_hair=False,
        export_uvmaps=True,
        export_normals=True,
        export_materials=True,
        use_instancing=True,
        evaluation_mode='RENDER',
        generate_preview_surface=True,
        export_textures=export_textures,
        overwrite_textures=True,
        relative_paths=True,
    )


def main():
    """Main entry point for the export script."""
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description="Export GS-IR materials to USD"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Camera directory containing GS-IR outputs"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output USD file path (.usd, .usda, or .usdc)"
    )
    parser.add_argument(
        "--create-geometry",
        action="store_true",
        help="Create geometry (card for materials, dome for environment)"
    )
    parser.add_argument(
        "--material-name",
        type=str,
        default="gsir_material",
        help="Name for the PBR material (default: gsir_material)"
    )
    parser.add_argument(
        "--no-textures",
        action="store_true",
        help="Don't copy textures (embed paths only)"
    )
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"Error: Input directory not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if not args.input.is_dir():
        print(f"Error: Input must be a directory: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading GS-IR materials from {args.input}")

    metadata = load_gsir_metadata(args.input)
    if metadata:
        print(f"  Source: {metadata.get('source', 'unknown')}")
        print(f"  Iteration: {metadata.get('iteration', 'unknown')}")

    textures = find_material_textures(args.input)

    if not textures:
        print("Error: No material textures found", file=sys.stderr)
        sys.exit(1)

    print(f"Found textures:")
    for tex_type, tex_path in textures.items():
        print(f"  {tex_type}: {tex_path.name}")

    clear_scene()

    pbr_textures = {k: v for k, v in textures.items() if k != "environment"}
    if pbr_textures:
        print(f"Creating PBR material: {args.material_name}")
        pbr_mat = create_pbr_material(args.material_name, pbr_textures)

        if args.create_geometry:
            print("Creating material card...")
            create_material_card(pbr_mat, f"{args.material_name}_card")

    if "environment" in textures:
        print("Creating environment material...")
        env_mat = create_environment_material(textures["environment"])

        if args.create_geometry:
            print("Creating environment dome...")
            create_environment_dome(env_mat)

    print(f"Exporting USD...")
    print(f"  Output: {args.output}")
    print(f"  Export textures: {not args.no_textures}")

    try:
        export_usd(
            args.output,
            export_textures=not args.no_textures,
        )
    except Exception as e:
        print(f"Error exporting USD: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Successfully exported: {args.output}")


if __name__ == "__main__":
    main()
