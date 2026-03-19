"""Quality filtering for auto-generated cell masks.

Removes cells that are likely artifacts: no membrane signal, unreasonable size,
or missing nuclei. Also computes per-tile statistics for distribution monitoring.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage


def filter_cells(
    nucleus_labels: np.ndarray,
    cell_labels: np.ndarray,
    dab_channel: np.ndarray,
    max_cell_nucleus_ratio: float = 5.0,
    min_membrane_coverage: float = 0.3,
    min_nucleus_area_px: int = 20,
    max_nucleus_area_px: int = 5000,
    min_cell_area_px: int = 50,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Filter cell and nucleus masks to remove low-quality instances.

    Args:
        nucleus_labels: Integer-labeled nucleus mask (H, W).
        cell_labels: Integer-labeled cell mask (H, W), same IDs as nuclei.
        dab_channel: DAB concentration map (H, W), float32.
        max_cell_nucleus_ratio: Maximum allowed cell_area / nucleus_area.
        min_membrane_coverage: Minimum fraction of cell perimeter with DAB signal.
        min_nucleus_area_px: Minimum nucleus area in pixels.
        max_nucleus_area_px: Maximum nucleus area in pixels.
        min_cell_area_px: Minimum cell area in pixels.

    Returns:
        Tuple of (filtered_nucleus_labels, filtered_cell_labels, stats_dict).
        Labels are renumbered contiguously starting from 1.
    """
    if not np.isfinite(dab_channel).all():
        raise ValueError("dab_channel contains NaN or Inf values — check upstream deconvolution.")

    unique_ids = np.unique(nucleus_labels)
    unique_ids = unique_ids[unique_ids > 0]

    # Compute all areas in one pass via bincount
    max_id = max(int(nucleus_labels.max()), int(cell_labels.max())) + 1
    nuc_areas = np.bincount(nucleus_labels.ravel(), minlength=max_id)
    cell_areas = np.bincount(cell_labels.ravel(), minlength=max_id)

    # Get bounding boxes for efficient per-cell operations
    cell_slices = ndimage.find_objects(cell_labels)

    keep_ids = []
    removed_reasons: dict[str, int] = {
        "small_nucleus": 0,
        "large_nucleus": 0,
        "small_cell": 0,
        "high_ratio": 0,
        "low_membrane": 0,
    }

    for cell_id in unique_ids:
        nuc_area = int(nuc_areas[cell_id])
        cell_area = int(cell_areas[cell_id])

        # Ghost ID: nucleus label exists but has zero area
        if nuc_area == 0:
            continue

        # Nucleus size filter
        if nuc_area < min_nucleus_area_px:
            removed_reasons["small_nucleus"] += 1
            continue
        if nuc_area > max_nucleus_area_px:
            removed_reasons["large_nucleus"] += 1
            continue

        # Cell size filter
        if cell_area < min_cell_area_px:
            removed_reasons["small_cell"] += 1
            continue

        # Cell/nucleus ratio filter
        ratio = cell_area / max(nuc_area, 1)
        if ratio > max_cell_nucleus_ratio:
            removed_reasons["high_ratio"] += 1
            continue

        # Membrane coverage: operate on bounding-box crop only
        bbox = cell_slices[cell_id - 1] if cell_id - 1 < len(cell_slices) and cell_slices[cell_id - 1] is not None else None
        if bbox is None:
            removed_reasons["low_membrane"] += 1
            continue

        cell_crop = (cell_labels[bbox] == cell_id).astype(np.uint8)
        contours, _ = cv2.findContours(cell_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            removed_reasons["low_membrane"] += 1
            continue

        contour = max(contours, key=cv2.contourArea)
        perimeter_mask = np.zeros_like(cell_crop)
        cv2.drawContours(perimeter_mask, [contour], 0, 1, thickness=2)

        perimeter_pixels = perimeter_mask > 0
        if perimeter_pixels.sum() == 0:
            removed_reasons["low_membrane"] += 1
            continue

        dab_crop = dab_channel[bbox]
        dab_at_perimeter = dab_crop[perimeter_pixels]
        membrane_fraction = (dab_at_perimeter > 0.05).sum() / dab_at_perimeter.size

        if membrane_fraction < min_membrane_coverage:
            removed_reasons["low_membrane"] += 1
            continue

        keep_ids.append(cell_id)

    # Rebuild filtered masks with contiguous IDs using a lookup table
    filtered_nuc = np.zeros_like(nucleus_labels)
    filtered_cell = np.zeros_like(cell_labels)
    if keep_ids:
        remap = np.zeros(max_id, dtype=np.int32)
        for new_id, old_id in enumerate(keep_ids, start=1):
            remap[old_id] = new_id
        filtered_nuc = remap[nucleus_labels]
        filtered_cell = remap[cell_labels]

    stats = {
        "total_instances": len(unique_ids),
        "kept_instances": len(keep_ids),
        "removed_reasons": removed_reasons,
    }
    return filtered_nuc, filtered_cell, stats


def compute_tile_stats(
    nucleus_labels: np.ndarray,
    cell_labels: np.ndarray,
) -> dict:
    """Compute per-tile statistics for distribution monitoring.

    Args:
        nucleus_labels: Filtered nucleus mask (H, W).
        cell_labels: Filtered cell mask (H, W).

    Returns:
        Dictionary of statistics for this tile.
    """
    # Single-pass area computation via bincount
    max_id = max(int(nucleus_labels.max()), int(cell_labels.max())) + 1
    nuc_counts = np.bincount(nucleus_labels.ravel(), minlength=max_id)
    cell_counts = np.bincount(cell_labels.ravel(), minlength=max_id)

    # Exclude background (id=0)
    nuc_areas = nuc_counts[1:]
    nuc_areas = nuc_areas[nuc_areas > 0].astype(np.float64)
    cell_areas = cell_counts[1:]
    cell_areas = cell_areas[cell_areas > 0].astype(np.float64)

    # Compute ratios for IDs present in both
    ratios = []
    for cid in range(1, max_id):
        if cell_counts[cid] > 0 and nuc_counts[cid] > 0:
            ratios.append(float(cell_counts[cid] / nuc_counts[cid]))

    return {
        "num_nuclei": int(len(nuc_areas)),
        "num_cells": int(len(cell_areas)),
        "mean_nucleus_area": float(np.mean(nuc_areas)) if len(nuc_areas) > 0 else 0.0,
        "mean_cell_area": float(np.mean(cell_areas)) if len(cell_areas) > 0 else 0.0,
        "mean_cell_nucleus_ratio": float(np.mean(ratios)) if ratios else 0.0,
        "std_cell_nucleus_ratio": float(np.std(ratios)) if ratios else 0.0,
        "tile_coverage": float((cell_labels > 0).sum() / cell_labels.size),
    }
