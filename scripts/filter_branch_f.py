"""Branch F: CONCH-style zero-shot tile filtering + DAB classification.

Filters tiles using text-similarity scoring that mimics CONCH zero-shot:
- Compute a "membrane staining score" per tile based on DAB spatial pattern
- Tiles with strong, ring-shaped DAB patterns score high (epithelial + positive)
- Tiles with diffuse/absent DAB score low (stroma or negative)
- Then filter cells by DAB class >= 1+

This is a pure-vision approach (no foundation model needed) that captures
what CONCH's zero-shot text prompts describe:
  "strong CLDN18.2 membranous staining, intense dark brown DAB chromogen"

Produces: data/segmentation_dataset_branch_f.pth
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch
from tqdm import tqdm

from instanseg_brightfield.config import load_config
from instanseg_brightfield.stain import StainDeconvolver

logger = logging.getLogger(__name__)


def _membrane_staining_score(dab_channel: np.ndarray, nucleus_masks: np.ndarray) -> float:
    """Score how much the DAB signal looks like membrane staining.

    Membrane staining has a characteristic ring pattern: DAB is concentrated
    at cell boundaries, not uniformly across the cell body. This function
    measures the "ringiness" of the DAB signal around detected nuclei.

    Returns 0-1 where higher = more membrane-like DAB pattern.
    """
    if nucleus_masks.max() == 0:
        return 0.0

    ring_scores = []
    cell_ids = np.unique(nucleus_masks)
    cell_ids = cell_ids[cell_ids > 0]

    for nid in cell_ids[:50]:  # sample max 50 cells per tile for speed
        nuc_mask = (nucleus_masks == nid).astype(np.uint8)

        # Create ring: dilate nucleus by 5px, subtract nucleus
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        dilated = cv2.dilate(nuc_mask, kernel, iterations=1)
        ring = dilated.astype(bool) & ~nuc_mask.astype(bool)

        if ring.sum() < 10 or nuc_mask.sum() < 10:
            continue

        # Compare DAB in ring vs DAB in nucleus
        dab_ring = dab_channel[ring].mean()
        dab_nucleus = dab_channel[nuc_mask.astype(bool)].mean()

        # Membrane staining: ring has MORE DAB than nucleus interior
        if dab_ring + dab_nucleus > 0.01:  # avoid division by near-zero
            ring_ratio = dab_ring / (dab_ring + dab_nucleus)  # 0.5 = equal, >0.5 = ring brighter
            ring_scores.append(float(ring_ratio))

    if not ring_scores:
        return 0.0

    # Mean ring ratio across cells
    # Good membrane staining: ratio > 0.55 (ring has more DAB than nucleus)
    mean_ratio = np.mean(ring_scores)

    # Also factor in overall DAB presence
    dab_presence = min(float(dab_channel.mean()) / 0.15, 1.0)

    # Combined: must have DAB present AND in ring pattern
    score = mean_ratio * dab_presence
    return float(score)


def filter_dataset_branch_f(cfg: dict, output_path: Path | None = None) -> Path:
    """Apply Branch F filtering: membrane staining pattern + DAB class."""
    data_dir = Path(cfg["paths"]["data_dir"])
    sc = cfg["stain_deconvolution"]
    deconv = StainDeconvolver(np.array([sc["hematoxylin"], sc["dab"], sc["residual"]]))

    input_path = data_dir / "segmentation_dataset.pth"
    if output_path is None:
        output_path = data_dir / "segmentation_dataset_branch_f.pth"

    dataset = torch.load(input_path, weights_only=False)

    # Thresholds
    MEMBRANE_SCORE_THRESHOLD = 0.25  # tiles must show ring-like DAB pattern
    DAB_NEGATIVE_THRESHOLD = 0.10    # cells below this are negative

    filtered = {"Train": [], "Validation": [], "Test": []}
    stats = {"total": 0, "kept": 0, "dropped_no_membrane_pattern": 0,
             "dropped_negative_dab": 0, "dropped_no_cells": 0}

    for split in ["Train", "Validation", "Test"]:
        items = dataset[split]
        logger.info("Filtering %s: %d items", split, len(items))

        for item in tqdm(items, desc=f"Branch F {split}"):
            stats["total"] += 1

            img = tifffile.imread(data_dir / item["image"])
            nuc = tifffile.imread(data_dir / item["nucleus_masks"]).astype(np.int32)
            cell = tifffile.imread(data_dir / item["cell_masks"]).astype(np.int32)

            if nuc.max() == 0:
                stats["dropped_no_cells"] += 1
                continue

            dab = deconv.extract_dab(img)

            # Filter 1: Membrane staining pattern score
            mem_score = _membrane_staining_score(dab, nuc)
            if mem_score < MEMBRANE_SCORE_THRESHOLD:
                stats["dropped_no_membrane_pattern"] += 1
                continue

            # Filter 2: Remove DAB-negative cells
            cell_ids = np.unique(cell)
            cell_ids = cell_ids[cell_ids > 0]

            keep_ids = set()
            for cid in cell_ids:
                cell_mask = cell == cid
                dab_mean = float(dab[cell_mask].mean())
                if dab_mean >= DAB_NEGATIVE_THRESHOLD:
                    keep_ids.add(int(cid))

            if not keep_ids:
                stats["dropped_negative_dab"] += 1
                continue

            # Rebuild masks
            nuc_filtered = np.where(np.isin(nuc, list(keep_ids)), nuc, 0)
            cell_filtered = np.where(np.isin(cell, list(keep_ids)), cell, 0)

            nuc_path = data_dir / item["nucleus_masks"]
            cell_path = data_dir / item["cell_masks"]
            tifffile.imwrite(str(nuc_path), nuc_filtered.astype(np.uint16), compression="zlib")
            tifffile.imwrite(str(cell_path), cell_filtered.astype(np.uint16), compression="zlib")

            filtered[split].append(item)
            stats["kept"] += 1

    # Subsample to ~10k
    rng = random.Random(42)
    for split in filtered:
        items = filtered[split]
        slides = {}
        for item in items:
            slide = Path(item["image"]).parts[1]
            slides.setdefault(slide, []).append(item)
        sampled = []
        for sn, si in sorted(slides.items()):
            sampled.extend(rng.sample(si, min(50, len(si))))
        filtered[split] = sampled

    total_filtered = sum(len(v) for v in filtered.values())
    logger.info("Branch F filtering complete:")
    logger.info("  Total: %d -> %d (%.1f%% kept)", stats["total"], stats["kept"],
                stats["kept"] / max(stats["total"], 1) * 100)
    logger.info("  Dropped no membrane pattern: %d", stats["dropped_no_membrane_pattern"])
    logger.info("  Dropped negative DAB: %d", stats["dropped_negative_dab"])
    logger.info("  Dropped no cells: %d", stats["dropped_no_cells"])
    logger.info("  After subsampling: %d items", total_filtered)

    torch.save(filtered, output_path)
    logger.info("Saved to %s", output_path)
    return output_path


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = load_config()
    filter_dataset_branch_f(cfg)


if __name__ == "__main__":
    main()
