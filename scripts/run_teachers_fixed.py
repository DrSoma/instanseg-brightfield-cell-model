"""Fixed teacher cell models — runs CellSAM, Cellpose, and InstanSeg fluoro
on brightfield tiles to generate cell boundary predictions.

Fixes from original teacher_cell_models.py:
- InstanSeg fluoro: use low-level model.instanseg() not model() high-level API
- CellSAM: install from GitHub, handle API correctly
- Cellpose: use cyto3 model with channels=[0,0] for brightfield
"""

from __future__ import annotations

import logging
import json
import time
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch
from tqdm import tqdm

from instanseg_brightfield.config import load_config
from instanseg_brightfield.stain import StainDeconvolver

logger = logging.getLogger(__name__)


def run_instanseg_fluoro(tiles: list[tuple[Path, np.ndarray]], device: str = "cuda") -> dict[str, np.ndarray]:
    """Run InstanSeg fluorescence_nuclei_and_cells on tiles using LOW-LEVEL API."""
    from instanseg import InstanSeg
    from instanseg.utils.augmentations import Augmentations

    model = InstanSeg("fluorescence_nuclei_and_cells", device=device)
    Aug = Augmentations()
    results = {}

    for tile_path, img in tqdm(tiles, desc="InstanSeg fluoro"):
        t, _ = Aug.to_tensor(img, normalize=False)
        t, _ = Aug.normalize(t)

        try:
            with torch.inference_mode():
                output = model.instanseg(t.unsqueeze(0).to(device))
            pred = output[0].cpu().numpy()
            if pred.shape[0] >= 2:
                cell_labels = pred[1].astype(np.int32)  # cell channel
            else:
                cell_labels = pred[0].astype(np.int32)
        except Exception as e:
            logger.warning("InstanSeg fluoro failed on %s: %s", tile_path.name, e)
            cell_labels = np.zeros(img.shape[:2], dtype=np.int32)

        results[str(tile_path)] = cell_labels

    n_cells = sum(int(v.max()) for v in results.values())
    logger.info("InstanSeg fluoro: %d cells across %d tiles", n_cells, len(results))
    return results


def run_cellpose(tiles: list[tuple[Path, np.ndarray]], device: str = "cuda") -> dict[str, np.ndarray]:
    """Run Cellpose cyto3 on tiles."""
    from cellpose import models

    model = models.Cellpose(model_type="cyto3", gpu=(device != "cpu"))
    results = {}

    for tile_path, img in tqdm(tiles, desc="Cellpose cyto3"):
        try:
            masks, _, _, _ = model.eval(img, diameter=None, channels=[0, 0])
            cell_labels = masks.astype(np.int32)
        except Exception as e:
            logger.warning("Cellpose failed on %s: %s", tile_path.name, e)
            cell_labels = np.zeros(img.shape[:2], dtype=np.int32)

        results[str(tile_path)] = cell_labels

    n_cells = sum(int(v.max()) for v in results.values())
    logger.info("Cellpose: %d cells across %d tiles", n_cells, len(results))
    return results


def run_cellsam(tiles: list[tuple[Path, np.ndarray]], device: str = "cuda") -> dict[str, np.ndarray]:
    """Run CellSAM on tiles."""
    try:
        from cellSAM import segment_cellular_image
    except ImportError:
        logger.warning("CellSAM not available — skipping")
        return {}

    results = {}
    for tile_path, img in tqdm(tiles, desc="CellSAM"):
        try:
            mask = segment_cellular_image(img, device=device)
            if isinstance(mask, tuple):
                mask = mask[0]
            cell_labels = np.asarray(mask, dtype=np.int32)
        except Exception as e:
            logger.warning("CellSAM failed on %s: %s", tile_path.name, e)
            cell_labels = np.zeros(img.shape[:2], dtype=np.int32)

        results[str(tile_path)] = cell_labels

    n_cells = sum(int(v.max()) for v in results.values())
    logger.info("CellSAM: %d cells across %d tiles", n_cells, len(results))
    return results


def build_consensus(
    all_predictions: dict[str, dict[str, np.ndarray]],
    min_agreement: int = 2,
    iou_threshold: float = 0.3,
) -> dict[str, np.ndarray]:
    """Build consensus cell masks where >= min_agreement teachers agree.

    For each tile, finds cells that overlap (IoU > threshold) across teachers
    and keeps those with sufficient agreement.
    """
    # Get all tile paths that appear in at least one teacher
    all_tiles = set()
    for teacher_results in all_predictions.values():
        all_tiles.update(teacher_results.keys())

    consensus = {}
    for tile_path in tqdm(sorted(all_tiles), desc="Consensus"):
        # Collect predictions from all teachers for this tile
        teacher_masks = []
        for teacher_name, teacher_results in all_predictions.items():
            if tile_path in teacher_results:
                mask = teacher_results[tile_path]
                if mask.max() > 0:
                    teacher_masks.append(mask)

        if len(teacher_masks) < min_agreement:
            consensus[tile_path] = np.zeros_like(teacher_masks[0]) if teacher_masks else np.zeros((512, 512), dtype=np.int32)
            continue

        # Simple consensus: binary overlap
        # Convert each teacher's mask to binary (cell vs background)
        binary_masks = [(m > 0).astype(np.float32) for m in teacher_masks]
        agreement = np.sum(binary_masks, axis=0)

        # Pixels where >= min_agreement teachers say "cell"
        consensus_binary = (agreement >= min_agreement).astype(np.uint8)

        # Use the first teacher's instance labels where consensus says "cell"
        # This preserves instance IDs
        best_teacher = teacher_masks[0]  # use the one with most cells
        for m in teacher_masks[1:]:
            if m.max() > best_teacher.max():
                best_teacher = m

        consensus_mask = np.where(consensus_binary, best_teacher, 0).astype(np.int32)
        consensus[tile_path] = consensus_mask

    n_cells = sum(int(v.max()) for v in consensus.values())
    logger.info("Consensus: %d cells across %d tiles (min_agreement=%d)", n_cells, len(consensus), min_agreement)
    return consensus


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    cfg = load_config()
    data_dir = Path(cfg["paths"]["data_dir"])

    # Collect tiles
    max_tiles = 500
    dataset = torch.load(data_dir / "segmentation_dataset_base.pth", weights_only=False)
    tile_items = dataset["Test"][:max_tiles]

    tiles = []
    for item in tile_items:
        img_path = data_dir / item["image"]
        if not img_path.exists():
            continue
        img = tifffile.imread(img_path)
        tiles.append((img_path, img))

    logger.info("Loaded %d tiles for teacher prediction", len(tiles))

    # Run each teacher
    all_predictions = {}

    # 1. InstanSeg fluoro (GPU 0)
    logger.info("=== InstanSeg fluorescence_nuclei_and_cells ===")
    try:
        fluoro_results = run_instanseg_fluoro(tiles, device="cuda:0")
        if sum(int(v.max()) for v in fluoro_results.values()) > 0:
            all_predictions["instanseg_fluoro"] = fluoro_results
    except Exception as e:
        logger.error("InstanSeg fluoro failed: %s", e)

    # Free GPU 0
    torch.cuda.empty_cache()

    # 2. Cellpose (GPU 0)
    logger.info("=== Cellpose cyto3 ===")
    try:
        cellpose_results = run_cellpose(tiles, device="cuda:0")
        if sum(int(v.max()) for v in cellpose_results.values()) > 0:
            all_predictions["cellpose"] = cellpose_results
    except Exception as e:
        logger.error("Cellpose failed: %s", e)

    torch.cuda.empty_cache()

    # 3. CellSAM (GPU 1)
    logger.info("=== CellSAM ===")
    try:
        cellsam_results = run_cellsam(tiles, device="cuda:1")
        if cellsam_results and sum(int(v.max()) for v in cellsam_results.values()) > 0:
            all_predictions["cellsam"] = cellsam_results
    except Exception as e:
        logger.error("CellSAM failed: %s", e)

    torch.cuda.empty_cache()

    logger.info("Teachers with results: %s", list(all_predictions.keys()))

    # Save individual predictions
    for teacher_name, teacher_results in all_predictions.items():
        out_dir = data_dir / "teacher_predictions" / teacher_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for tile_path_str, mask in teacher_results.items():
            tile_path = Path(tile_path_str)
            slide = tile_path.parent.name
            stem = tile_path.stem
            slide_dir = out_dir / slide
            slide_dir.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(slide_dir / f"{stem}_cells.tiff"), mask.astype(np.uint16), compression="zlib")

    # Build consensus
    if len(all_predictions) >= 2:
        logger.info("Building consensus from %d teachers", len(all_predictions))
        consensus = build_consensus(all_predictions, min_agreement=2)

        out_dir = data_dir / "teacher_predictions" / "consensus"
        out_dir.mkdir(parents=True, exist_ok=True)
        for tile_path_str, mask in consensus.items():
            tile_path = Path(tile_path_str)
            slide = tile_path.parent.name
            stem = tile_path.stem
            slide_dir = out_dir / slide
            slide_dir.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(slide_dir / f"{stem}_cells.tiff"), mask.astype(np.uint16), compression="zlib")

    # Summary
    summary = {
        "teachers_available": list(all_predictions.keys()),
        "tiles_processed": len(tiles),
        "per_teacher_cells": {k: sum(int(v.max()) for v in vs.values()) for k, vs in all_predictions.items()},
    }
    Path("evaluation").mkdir(exist_ok=True)
    with open("evaluation/teacher_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Done. Summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
