#!/usr/bin/env python3
"""Separate combined segmentation masks into individual instance masks.

This script takes a directory of combined masks (where multiple people are
merged into one mask) and separates them into individual instance directories.

Uses connected component analysis with temporal tracking to maintain
consistent instance IDs across frames.

Usage:
    python separate_instances.py <input_dir> [options]

Example:
    # Separate combined person masks into person_0/, person_1/, etc.
    python separate_instances.py project/roto/person/ --output-dir project/roto/

    # Specify minimum area threshold (filter noise)
    python separate_instances.py project/roto/person/ --min-area 1000
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class InstanceTracker:
    """Track instances across frames using centroid and area matching."""

    def __init__(self, max_distance: float = 100.0, area_tolerance: float = 0.5):
        """Initialize tracker.

        Args:
            max_distance: Maximum centroid distance to consider same instance
            area_tolerance: Area can vary by this factor (0.5 = 50%)
        """
        self.max_distance = max_distance
        self.area_tolerance = area_tolerance
        self.instances: Dict[int, dict] = {}  # id -> {centroid, area, last_frame}
        self.next_id = 0

    def update(self, components: List[dict], frame_idx: int) -> List[Tuple[int, dict]]:
        """Match components to existing instances or create new ones.

        Args:
            components: List of {mask, centroid, area, bbox} dicts
            frame_idx: Current frame index

        Returns:
            List of (instance_id, component) tuples
        """
        if not components:
            return []

        # If no existing instances, create new ones
        if not self.instances:
            results = []
            for comp in components:
                instance_id = self.next_id
                self.next_id += 1
                self.instances[instance_id] = {
                    'centroid': comp['centroid'],
                    'area': comp['area'],
                    'last_frame': frame_idx
                }
                results.append((instance_id, comp))
            return results

        # Match components to existing instances using Hungarian-like greedy matching
        matched_instances = set()
        matched_components = set()
        results = []

        # Sort components by area (larger first) for more stable matching
        sorted_comps = sorted(enumerate(components), key=lambda x: -x[1]['area'])

        for comp_idx, comp in sorted_comps:
            best_id = None
            best_dist = float('inf')

            for inst_id, inst in self.instances.items():
                if inst_id in matched_instances:
                    continue

                # Check if instance was seen recently (within 10 frames)
                if frame_idx - inst['last_frame'] > 10:
                    continue

                # Calculate centroid distance
                dist = np.sqrt(
                    (comp['centroid'][0] - inst['centroid'][0]) ** 2 +
                    (comp['centroid'][1] - inst['centroid'][1]) ** 2
                )

                # Check area similarity
                area_ratio = comp['area'] / inst['area'] if inst['area'] > 0 else 0
                if area_ratio < (1 - self.area_tolerance) or area_ratio > (1 + self.area_tolerance):
                    # Area too different, increase effective distance
                    dist *= 2

                if dist < best_dist and dist < self.max_distance:
                    best_dist = dist
                    best_id = inst_id

            if best_id is not None:
                # Match found
                matched_instances.add(best_id)
                matched_components.add(comp_idx)
                self.instances[best_id] = {
                    'centroid': comp['centroid'],
                    'area': comp['area'],
                    'last_frame': frame_idx
                }
                results.append((best_id, comp))

        # Create new instances for unmatched components
        for comp_idx, comp in enumerate(components):
            if comp_idx not in matched_components:
                instance_id = self.next_id
                self.next_id += 1
                self.instances[instance_id] = {
                    'centroid': comp['centroid'],
                    'area': comp['area'],
                    'last_frame': frame_idx
                }
                results.append((instance_id, comp))

        return results

    def get_instance_count(self) -> int:
        """Get total number of tracked instances."""
        return self.next_id


def find_connected_components(
    mask: np.ndarray,
    min_area: int = 500
) -> List[dict]:
    """Find connected components in a binary mask.

    Args:
        mask: Binary mask (0 or 255)
        min_area: Minimum component area in pixels

    Returns:
        List of component dictionaries with mask, centroid, area, bbox
    """
    # Ensure binary
    binary = (mask > 127).astype(np.uint8)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    components = []
    for i in range(1, num_labels):  # Skip background (0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        # Create mask for this component
        comp_mask = (labels == i).astype(np.uint8) * 255

        # Get bounding box
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        components.append({
            'mask': comp_mask,
            'centroid': (centroids[i][0], centroids[i][1]),
            'area': area,
            'bbox': (x, y, w, h)
        })

    # Sort by x-coordinate (left to right) for consistent ordering
    components.sort(key=lambda c: c['centroid'][0])

    return components


def separate_instances(
    input_dir: Path,
    output_dir: Path,
    min_area: int = 500,
    prefix: str = "person",
    max_distance: float = 100.0
) -> Dict[int, Path]:
    """Separate combined masks into individual instance directories.

    Args:
        input_dir: Directory containing combined mask images
        output_dir: Parent directory for output (will create person_0/, etc.)
        min_area: Minimum component area to consider
        prefix: Prefix for output directories (e.g., "person")
        max_distance: Max centroid distance for tracking

    Returns:
        Dictionary mapping instance_id to output directory path
    """
    # Find all mask images
    mask_files = sorted(
        list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))
    )

    if not mask_files:
        print(f"No mask images found in {input_dir}", file=sys.stderr)
        return {}

    print(f"Processing {len(mask_files)} frames...")

    # Initialize tracker
    tracker = InstanceTracker(max_distance=max_distance)

    # First pass: analyze all frames to determine instances
    frame_data = []
    for frame_idx, mask_file in enumerate(mask_files):
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Could not read {mask_file}", file=sys.stderr)
            continue

        components = find_connected_components(mask, min_area)
        matches = tracker.update(components, frame_idx)
        frame_data.append({
            'file': mask_file,
            'matches': matches
        })

        # Progress
        if (frame_idx + 1) % 50 == 0:
            print(f"  Analyzed {frame_idx + 1}/{len(mask_files)} frames...")

    # Get all instance IDs that were detected
    all_instance_ids = set()
    for data in frame_data:
        for inst_id, _ in data['matches']:
            all_instance_ids.add(inst_id)

    if not all_instance_ids:
        print("No instances found in masks", file=sys.stderr)
        return {}

    # Sort instance IDs for consistent ordering
    sorted_ids = sorted(all_instance_ids)
    id_to_index = {inst_id: idx for idx, inst_id in enumerate(sorted_ids)}

    print(f"Found {len(sorted_ids)} distinct instances")

    # Create output directories (zero-padded two-digit index)
    output_dirs = {}
    for inst_id in sorted_ids:
        idx = id_to_index[inst_id]
        out_dir = output_dir / f"{prefix}_{idx:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_dirs[inst_id] = out_dir

    # Second pass: write separated masks
    print("Writing separated masks...")
    for frame_idx, data in enumerate(frame_data):
        mask_file = data['file']

        # Read original mask to get dimensions
        original = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if original is None:
            continue

        h, w = original.shape

        # Create empty masks for each instance
        instance_masks = {inst_id: np.zeros((h, w), dtype=np.uint8)
                         for inst_id in sorted_ids}

        # Fill in detected components
        for inst_id, comp in data['matches']:
            instance_masks[inst_id] = comp['mask']

        # Save each instance mask with consistent naming: {prefix}_{idx:02d}_{frame:05d}_.png
        for inst_id, inst_mask in instance_masks.items():
            idx = id_to_index[inst_id]
            out_dir = output_dirs[inst_id]
            out_file = out_dir / f"{prefix}_{idx:02d}_{frame_idx:05d}_.png"
            cv2.imwrite(str(out_file), inst_mask)

        # Progress
        if (frame_idx + 1) % 50 == 0:
            print(f"  Written {frame_idx + 1}/{len(frame_data)} frames...")

    # Summary
    print(f"\nSeparated into {len(output_dirs)} instance directories:")
    for inst_id in sorted_ids:
        idx = id_to_index[inst_id]
        print(f"  {prefix}_{idx:02d}/")

    return {id_to_index[inst_id]: output_dirs[inst_id] for inst_id in sorted_ids}


def main():
    parser = argparse.ArgumentParser(
        description="Separate combined masks into individual instances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing combined mask images"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Output parent directory (default: parent of input_dir)"
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=500,
        help="Minimum component area in pixels (default: 500)"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="person",
        help="Prefix for output directories (default: person)"
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=100.0,
        help="Max centroid distance for tracking (default: 100.0)"
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"Error: Input directory not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir or args.input_dir.parent

    result = separate_instances(
        args.input_dir,
        output_dir,
        min_area=args.min_area,
        prefix=args.prefix,
        max_distance=args.max_distance
    )

    if not result:
        sys.exit(1)

    print(f"\nDone! Created {len(result)} instance directories in {output_dir}")


if __name__ == "__main__":
    main()
