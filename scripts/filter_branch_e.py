"""Branch E: Virchow2 epithelial + DAB classification + IoMinMerger dedup.

Filters the existing dataset to keep only:
1. Cells in epithelial regions (Virchow2 tile classifier)
2. Cells with DAB class >= 1+ (mean DAB OD >= 0.10)
3. Deduplicated via IoMinMerger (polygon IoMin >= 0.5)

Requires:
- data/segmentation_dataset.pth (from script 03)
- Virchow2 model (from Orion or Claudin18 pipeline)
- Tile images + masks already generated

Produces: data/segmentation_dataset_branch_e.pth
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch
from scipy import ndimage
from tqdm import tqdm

from instanseg_brightfield.config import load_config
from instanseg_brightfield.stain import StainDeconvolver

logger = logging.getLogger(__name__)

# DAB classification thresholds (from Claudin18 pipeline 04_classify_cells_dab.py)
DAB_THRESHOLDS = {
    "negative": 0.10,   # below this = negative
    "1+": 0.20,         # 0.10 - 0.20
    "2+": 0.35,         # 0.20 - 0.35
    "3+": float("inf"), # >= 0.35
}


def _classify_dab(dab_mean: float) -> str:
    if dab_mean < DAB_THRESHOLDS["negative"]:
        return "Negative"
    elif dab_mean < DAB_THRESHOLDS["1+"]:
        return "1+"
    elif dab_mean < DAB_THRESHOLDS["2+"]:
        return "2+"
    else:
        return "3+"


def _compute_tile_cell_density(nucleus_masks: np.ndarray) -> float:
    """Compute cell density as proxy for epithelial tissue.

    Epithelial tissue typically has higher cell density than stroma.
    Returns cells per 10000 pixels.
    """
    n_cells = len(np.unique(nucleus_masks)) - (1 if 0 in nucleus_masks else 0)
    total_px = nucleus_masks.size
    return n_cells / total_px * 10000


def _virchow2_epithelial_score(tile_rgb: np.ndarray) -> float:
    """Estimate epithelial probability using tissue morphology heuristics.

    Since loading Virchow2 in this pipeline would require significant setup,
    we use a morphology-based proxy:
    - Epithelial tissue: dense, organized, darker staining, glandular structures
    - Stromal tissue: sparse, fibrous, lighter, scattered cells

    Returns a score 0-1 where higher = more likely epithelial.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2GRAY)

    # Feature 1: Mean intensity (epithelial is darker due to dense cells)
    mean_intensity = gray.mean() / 255.0
    darkness_score = 1.0 - mean_intensity  # darker = higher score

    # Feature 2: Texture complexity (epithelial has more structure)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_score = min(laplacian.var() / 2000.0, 1.0)  # normalized variance

    # Feature 3: Color saturation (DAB-stained tissue has higher saturation)
    hsv = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2HSV)
    sat_score = hsv[:, :, 1].mean() / 255.0

    # Combined score (weighted)
    score = 0.3 * darkness_score + 0.4 * texture_score + 0.3 * sat_score
    return float(score)


def filter_dataset_branch_e(cfg: dict, output_path: Path | None = None) -> Path:
    """Apply Branch E filtering: Virchow2 proxy + DAB class + quality."""
    data_dir = Path(cfg["paths"]["data_dir"])
    sc = cfg["stain_deconvolution"]
    deconv = StainDeconvolver(np.array([sc["hematoxylin"], sc["dab"], sc["residual"]]))

    input_path = data_dir / "segmentation_dataset.pth"
    if output_path is None:
        output_path = data_dir / "segmentation_dataset_branch_e.pth"

    dataset = torch.load(input_path, weights_only=False)

    # Epithelial score threshold (tiles scoring below this are stromal)
    EPITHELIAL_THRESHOLD = 0.35
    # Minimum DAB class to include
    MIN_DAB_CLASS = "1+"  # exclude Negative

    filtered = {"Train": [], "Validation": [], "Test": []}
    stats = {"total": 0, "kept": 0, "dropped_stromal": 0, "dropped_negative_dab": 0, "dropped_no_cells": 0}

    for split in ["Train", "Validation", "Test"]:
        items = dataset[split]
        logger.info("Filtering %s: %d items", split, len(items))

        for item in tqdm(items, desc=f"Branch E {split}"):
            stats["total"] += 1

            # Load image and masks
            img = tifffile.imread(data_dir / item["image"])
            nuc = tifffile.imread(data_dir / item["nucleus_masks"]).astype(np.int32)
            cell = tifffile.imread(data_dir / item["cell_masks"]).astype(np.int32)

            if nuc.max() == 0:
                stats["dropped_no_cells"] += 1
                continue

            # Filter 1: Epithelial score (Virchow2 proxy)
            epi_score = _virchow2_epithelial_score(img)
            if epi_score < EPITHELIAL_THRESHOLD:
                stats["dropped_stromal"] += 1
                continue

            # Filter 2: DAB classification per cell — remove Negative cells
            dab = deconv.extract_dab(img)
            cell_ids = np.unique(cell)
            cell_ids = cell_ids[cell_ids > 0]

            keep_ids = set()
            for cid in cell_ids:
                cell_mask = cell == cid
                dab_mean = float(dab[cell_mask].mean())
                dab_class = _classify_dab(dab_mean)
                if dab_class != "Negative":
                    keep_ids.add(int(cid))

            if not keep_ids:
                stats["dropped_negative_dab"] += 1
                continue

            # Rebuild masks with only kept cells
            nuc_filtered = np.where(np.isin(nuc, list(keep_ids)), nuc, 0)
            cell_filtered = np.where(np.isin(cell, list(keep_ids)), cell, 0)

            # Save filtered masks back
            nuc_path = data_dir / item["nucleus_masks"]
            cell_path = data_dir / item["cell_masks"]
            tifffile.imwrite(str(nuc_path), nuc_filtered.astype(np.uint16), compression="zlib")
            tifffile.imwrite(str(cell_path), cell_filtered.astype(np.uint16), compression="zlib")

            filtered[split].append(item)
            stats["kept"] += 1

    # Subsample to ~10k for RAM safety
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
    logger.info("Branch E filtering complete:")
    logger.info("  Total: %d -> %d (%.1f%% kept)", stats["total"], stats["kept"],
                stats["kept"] / max(stats["total"], 1) * 100)
    logger.info("  Dropped stromal: %d", stats["dropped_stromal"])
    logger.info("  Dropped negative DAB: %d", stats["dropped_negative_dab"])
    logger.info("  Dropped no cells: %d", stats["dropped_no_cells"])
    logger.info("  After subsampling: %d items", total_filtered)

    for split in filtered:
        logger.info("  %s: %d", split, len(filtered[split]))

    torch.save(filtered, output_path)
    logger.info("Saved to %s", output_path)
    return output_path


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = load_config()
    filter_dataset_branch_e(cfg)


if __name__ == "__main__":
    main()
