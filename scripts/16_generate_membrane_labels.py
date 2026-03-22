#!/usr/bin/env python3
"""Generate 4-class semantic labels from bar-filter response for membrane U-Net training.

For each tile, produces a uint8 label map:
  0 = background
  1 = nucleus
  2 = cytoplasm (cell interior, not membrane, not nucleus)
  3 = membrane (bar-filter positive within cell mask)

Labels are saved as PNG files in data/membrane_labels/{slide}/{tile_stem}_label.png
and an index CSV with per-tile membrane pixel fractions for weighted sampling.

Uses GPU for bar-filter computation. Processes ~78k tiles in ~15-25 minutes.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/16_generate_membrane_labels.py
"""
from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from tqdm import tqdm

from instanseg_brightfield.config import load_config
from instanseg_brightfield.stain import StainDeconvolver

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Bar-filter parameters (same as comprehensive_membrane_validation.py)
BAR_N = 8
BAR_KSIZE = 25
BAR_SIGMA_LONG = 5.0
BAR_SIGMA_SHORT = 1.0
BAR_BATCH_SIZE = 32  # tiles per GPU batch for bar-filter


def build_bar_kernels() -> torch.Tensor:
    """Build oriented Gaussian bar-filter kernels."""
    half = BAR_KSIZE // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)
    kernels = []
    for i in range(BAR_N):
        theta = i * np.pi / BAR_N
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x_rot = x * cos_t + y * sin_t
        y_rot = -x * sin_t + y * cos_t
        g = np.exp(-0.5 * (x_rot**2 / BAR_SIGMA_LONG**2 + y_rot**2 / BAR_SIGMA_SHORT**2))
        g -= g.mean()
        g /= np.abs(g).sum() + 1e-10
        kernels.append(g)
    return torch.from_numpy(np.stack(kernels)[:, np.newaxis, :, :]).float()


def process_batch_gpu(
    tiles_rgb: list[np.ndarray],
    masks_nuc: list[np.ndarray],
    masks_cell: list[np.ndarray],
    deconv: StainDeconvolver,
    bar_kernels: torch.Tensor,
    device: str,
) -> list[np.ndarray]:
    """Process a batch of tiles on GPU, return 4-class label maps."""
    B = len(tiles_rgb)
    H, W = tiles_rgb[0].shape[:2]

    # Stack tiles and extract DAB on GPU
    rgb_batch = np.stack(tiles_rgb).astype(np.float32) / 255.0  # (B, H, W, 3)
    rgb_t = torch.from_numpy(rgb_batch).to(device)

    # Stain deconvolution on GPU (vectorized)
    pixels = rgb_t.reshape(-1, 3)
    od = -torch.log(pixels.clamp(min=1 / 255.0))
    inv_matrix = torch.from_numpy(deconv._inv).float().to(device)
    conc = (od @ inv_matrix).clamp(min=0)
    dab_flat = conc[:, 1]  # DAB channel
    dab_batch = dab_flat.reshape(B, 1, H, W)

    # Bar-filter on GPU (all tiles at once)
    pad = bar_kernels.shape[-1] // 2
    responses = F.conv2d(dab_batch, bar_kernels, padding=pad)  # (B, N, H, W)
    bar_max = responses.max(dim=1).values  # (B, H, W)
    bar_positive = (bar_max > 0).cpu().numpy()  # (B, H, W) bool

    # Generate labels (CPU — needs instance masks)
    labels = []
    for i in range(B):
        nuc = masks_nuc[i]
        cell = masks_cell[i]
        bar_pos = bar_positive[i]

        label = np.zeros((H, W), dtype=np.uint8)

        cell_mask = cell > 0
        nuc_mask = nuc > 0
        membrane_mask = bar_pos & cell_mask & ~nuc_mask

        label[nuc_mask] = 1
        label[cell_mask & ~nuc_mask & ~membrane_mask] = 2
        label[membrane_mask] = 3

        labels.append(label)

    return labels


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0, help="Shard index (0 or 1)")
    parser.add_argument("--n-shards", type=int, default=1, help="Total shards")
    args = parser.parse_args()

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = load_config("config/default.yaml")
    data_dir = Path(cfg["paths"]["data_dir"])
    sc = cfg["stain_deconvolution"]
    deconv = StainDeconvolver(np.array([sc["hematoxylin"], sc["dab"], sc["residual"]]))

    # Output directory
    label_dir = data_dir / "membrane_labels"
    label_dir.mkdir(parents=True, exist_ok=True)

    # Load bar-filter kernels to GPU
    bar_kernels = build_bar_kernels().to(device)
    logger.info("Bar-filter kernels on %s: %s (shard %d/%d)", device, bar_kernels.shape,
                args.shard + 1, args.n_shards)

    # Find all tiles with corresponding masks — shard across GPUs
    all_tile_dirs = sorted([d for d in (data_dir / "tiles").iterdir() if d.is_dir()])
    tile_dirs = [d for i, d in enumerate(all_tile_dirs) if i % args.n_shards == args.shard]
    logger.info("Processing %d/%d slides on this shard", len(tile_dirs), len(all_tile_dirs))
    mask_base = data_dir / "masks"

    index_rows = []  # For weighted sampling CSV
    total_tiles = 0
    total_membrane_px = 0
    total_px = 0

    for slide_dir in tqdm(tile_dirs, desc="Slides"):
        slide_name = slide_dir.name
        out_dir = label_dir / slide_name
        out_dir.mkdir(parents=True, exist_ok=True)

        tile_paths = sorted(slide_dir.glob("*.png"))
        if not tile_paths:
            continue

        # Collect batches
        batch_rgb = []
        batch_nuc = []
        batch_cell = []
        batch_stems = []

        for tile_path in tile_paths:
            stem = tile_path.stem
            nuc_path = mask_base / slide_name / f"{stem}_nuclei.tiff"
            cell_path = mask_base / slide_name / f"{stem}_cells.tiff"

            # Skip if label already exists
            label_path = out_dir / f"{stem}_label.png"
            if label_path.exists():
                total_tiles += 1
                continue

            if not nuc_path.exists() or not cell_path.exists():
                continue

            rgb = cv2.imread(str(tile_path))
            if rgb is None:
                continue
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            nuc = tifffile.imread(str(nuc_path))
            cell = tifffile.imread(str(cell_path))

            batch_rgb.append(rgb)
            batch_nuc.append(nuc)
            batch_cell.append(cell)
            batch_stems.append(stem)

            # Process batch when full
            if len(batch_rgb) >= BAR_BATCH_SIZE:
                labels = process_batch_gpu(batch_rgb, batch_nuc, batch_cell,
                                           deconv, bar_kernels, device)
                for j, lbl in enumerate(labels):
                    lbl_path = out_dir / f"{batch_stems[j]}_label.png"
                    cv2.imwrite(str(lbl_path), lbl)

                    n_membrane = (lbl == 3).sum()
                    n_total = lbl.size
                    membrane_frac = n_membrane / n_total if n_total > 0 else 0
                    index_rows.append({
                        "slide": slide_name,
                        "tile": batch_stems[j],
                        "membrane_fraction": f"{membrane_frac:.6f}",
                        "n_membrane_px": int(n_membrane),
                        "n_total_px": int(n_total),
                    })
                    total_tiles += 1
                    total_membrane_px += n_membrane
                    total_px += n_total

                batch_rgb.clear()
                batch_nuc.clear()
                batch_cell.clear()
                batch_stems.clear()

        # Process remaining
        if batch_rgb:
            labels = process_batch_gpu(batch_rgb, batch_nuc, batch_cell,
                                       deconv, bar_kernels, device)
            for j, lbl in enumerate(labels):
                lbl_path = out_dir / f"{batch_stems[j]}_label.png"
                cv2.imwrite(str(lbl_path), lbl)

                n_membrane = (lbl == 3).sum()
                n_total = lbl.size
                membrane_frac = n_membrane / n_total if n_total > 0 else 0
                index_rows.append({
                    "slide": slide_name,
                    "tile": batch_stems[j],
                    "membrane_fraction": f"{membrane_frac:.6f}",
                    "n_membrane_px": int(n_membrane),
                    "n_total_px": int(n_total),
                })
                total_tiles += 1
                total_membrane_px += n_membrane
                total_px += n_total

    # Save index CSV
    index_path = label_dir / "label_index.csv"
    with open(index_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["slide", "tile", "membrane_fraction",
                                                "n_membrane_px", "n_total_px"])
        writer.writeheader()
        writer.writerows(index_rows)

    elapsed = time.time() - t0
    avg_membrane = total_membrane_px / total_px if total_px > 0 else 0
    logger.info("=" * 60)
    logger.info("LABEL GENERATION COMPLETE")
    logger.info("Tiles processed: %d", total_tiles)
    logger.info("Average membrane fraction: %.4f (%.1f%%)", avg_membrane, avg_membrane * 100)
    logger.info("Labels saved to: %s", label_dir)
    logger.info("Index saved to: %s", index_path)
    logger.info("Time: %.1f seconds (%.1f min)", elapsed, elapsed / 60)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
